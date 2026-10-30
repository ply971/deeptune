"""Snippet feature extraction: decode once, resumable, exact snippet layout, float16 safety."""
import json
import math
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip('av')

import extract_features as ef
from datasets.feature_dataset import open_feature_source, row_video_id
from datasets.snippet_features import (FeatureConfigError, FeatureStore, FeatureValueError, load_snippet_features,
                                       parse_token_grid, safe_video_id, snippet_count, snippet_starts, to_float16,
                                       video_id_for_path)
from datasets.video_io import VideoDecodeError, VideoReader, farneback_flow
from datasets.video_synth import decode_frame_index, index_coded_clip, translating_clip
from handlers.video_args import SNIPPET_FEATURE_ARG_NAMES
from src.video.feature_backbones import SnippetBackbone, pool_token_grid, resolve_token_grid

SMALL = dict(model_version='dummy_video3d', motion_backbone='dummy_video3d', feature_weights='none', snippet_frames=8,
             snippet_stride=8, batch_size=2, feature_input_size=32)


@pytest.fixture(scope='module')
def videos(tmp_path_factory):
    folder = tmp_path_factory.mktemp('clips')
    paths = [translating_clip(folder / f'v{i}.mp4', n, 2, 0, size=32, seed=i) for i, n in enumerate((8, 20, 33))]
    return paths


def _manifest(paths):
    return pd.DataFrame({'videos': [str(p) for p in paths], 'labels': ['a', 'b', 'a'][:len(paths)],
                         'video_id': [p.stem for p in paths]})


def _arrays(root, stream='rgb'):
    return {p.stem: np.array(np.load(p)) for p in sorted((Path(root) / stream).glob('*.npy'))}


class _NoDecode:
    """Every way of getting pixels or motion out of a VideoReader fails the test."""
    def __enter__(self):
        boom = AssertionError('decoded a video that was already extracted')
        self.patches = [mock.patch.object(VideoReader, name, side_effect=boom)
                        for name in ('get_frames', 'iter_frames', 'get_motion_vectors', 'motion_curve')]
        for patch in self.patches:
            patch.start()

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()


# -- snippet arithmetic ------------------------------------------------------------
@pytest.mark.parametrize('frames,S,R,expected', [
    (16, 16, 16, 1), (17, 16, 16, 2), (32, 16, 16, 2), (33, 16, 16, 3), (5, 16, 16, 1), (1, 16, 16, 1),
    (20, 16, 2, 3), (21, 16, 2, 4), (40, 16, 32, 2), (20, 16, 32, 1), (33, 16, 32, 2), (0, 16, 16, 0),
])
def test_snippet_count_examples(frames, S, R, expected):
    assert snippet_count(frames, S, R) == expected


def test_snippet_count_is_the_documented_formula_when_stride_fits():
    for S in (1, 2, 4, 8, 16):
        for R in range(1, S + 1):
            for frames in range(S, 90):
                assert snippet_count(frames, S, R) == math.ceil((frames - S) / R) + 1


def test_snippet_generator_matches_the_formula_and_returns_exact_frames():
    """Brute force over all small cases: the streaming code really yields what the formula promises."""
    for S in (1, 2, 3, 5, 8):
        for R in (1, 2, 3, 5, 8, 11):
            for F in range(1, 30):
                frames = [np.full((2, 2, 3), i, np.uint8) for i in range(F)]
                snippets = list(ef.iter_rgb_snippets(iter(frames), S, R))
                assert len(snippets) == snippet_count(F, S, R), (F, S, R)
                for k, (snippet, valid) in enumerate(snippets):
                    start = k * R
                    assert start < F
                    assert len(snippet) == S
                    expected = [min(start + j, F - 1) for j in range(S)]
                    assert snippet[:, 0, 0, 0].tolist() == expected, (F, S, R, k)
                    assert valid == min(S, F - start)


def test_snippet_generator_holds_at_most_one_snippet_of_frames():
    alive = []

    def frames():
        for i in range(100):
            frame = np.full((2, 2, 3), i % 255, np.uint8)
            alive.append(i)
            yield frame

    for k, (snippet, _) in enumerate(ef.iter_rgb_snippets(frames(), 4, 4)):
        assert alive[-1] == k * 4 + 3  # never read ahead of the snippet being yielded


def test_snippets_read_exact_frames_from_a_real_clip(tmp_path):
    path = index_coded_clip(tmp_path / 'coded.mp4', 20, 64, 64)
    with VideoReader(path) as reader:
        snippets = list(ef.iter_rgb_snippets(reader.iter_frames(), 8, 8))
    assert [[decode_frame_index(f) for f in s] for s, _ in snippets] == [
        list(range(8)), list(range(8, 16)), [16, 17, 18, 19, 19, 19, 19, 19]]
    assert [v for _, v in snippets] == [8, 8, 4]


def test_real_extraction_snippet_count_matches_the_formula(videos, tmp_path):
    ef.extract_features(_manifest(videos), tmp_path / 'f', **SMALL)
    for path, frames in zip(videos, (8, 20, 33)):
        loaded = load_snippet_features(tmp_path / 'f', path.stem)
        assert len(loaded.features) == snippet_count(frames, 8, 8)
        side = loaded.meta['rgb']
        assert side['num_snippets'] == len(loaded.features) and side['frame_count'] == frames
        assert side['last_snippet_valid_frames'] == frames - (snippet_count(frames, 8, 8) - 1) * 8


def test_overlapping_stride(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1] + videos[1:2]), tmp_path / 'f', **{**SMALL, 'snippet_stride': 3})
    assert len(_arrays(tmp_path / 'f')['v1']) == snippet_count(20, 8, 3) == math.ceil(12 / 3) + 1


# -- decode once ----------------------------------------------------------------------
def test_second_run_decodes_nothing_builds_no_model_and_changes_nothing(videos, tmp_path):
    first = ef.extract_features(_manifest(videos), tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    assert first.extracted == {'rgb': 3, 'motion': 3} and not first.failed
    before = _arrays(tmp_path / 'f'), _arrays(tmp_path / 'f', 'motion')
    stamps = {p: p.stat().st_mtime_ns for p in (tmp_path / 'f').rglob('*.npy')}
    with _NoDecode(), mock.patch('src.video.feature_backbones.SnippetBackbone', side_effect=AssertionError('built a model')):
        second = ef.extract_features(_manifest(videos), tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    assert second.extracted == {'rgb': 0, 'motion': 0} and second.skipped == {'rgb': 3, 'motion': 3}
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps
    after = _arrays(tmp_path / 'f'), _arrays(tmp_path / 'f', 'motion')
    for one, other in zip(before, after):
        assert all(np.array_equal(one[k], other[k]) for k in one) and one.keys() == other.keys()


def test_extracting_twice_from_scratch_gives_identical_arrays(videos, tmp_path):
    ef.extract_features(_manifest(videos), tmp_path / 'a', streams=('rgb', 'motion'), **SMALL)
    ef.extract_features(_manifest(videos), tmp_path / 'b', streams=('rgb', 'motion'), **SMALL)
    for stream in ('rgb', 'motion'):
        a, b = _arrays(tmp_path / 'a', stream), _arrays(tmp_path / 'b', stream)
        assert a.keys() == b.keys() and len(a) == 3
        assert all(np.array_equal(a[k], b[k]) for k in a)


def test_changed_source_is_extracted_again(tmp_path):
    path = translating_clip(tmp_path / 'v.mp4', 16, 2, 0, size=32, seed=1)
    df = pd.DataFrame({'videos': [str(path)], 'video_id': ['v']})
    ef.extract_features(df, tmp_path / 'f', **SMALL)
    old = _arrays(tmp_path / 'f')['v']
    translating_clip(path, 24, -2, 0, size=32, seed=2)
    report = ef.extract_features(df, tmp_path / 'f', **SMALL)
    assert report.extracted['rgb'] == 1
    assert _arrays(tmp_path / 'f')['v'].shape[0] == 3 != old.shape[0]


def test_other_settings_are_refused_instead_of_mixed(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **SMALL)
    with pytest.raises(FeatureConfigError, match='snippet_frames'):
        ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **{**SMALL, 'snippet_frames': 4})
    with pytest.raises(FeatureConfigError, match='token_grid'):
        ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', cache_tokens='2x2x2', **SMALL)


# -- resume ----------------------------------------------------------------------------
class _Killed(Exception):
    """Not one of the per-video errors, like a real interruption."""


def test_resumed_extraction_equals_an_uninterrupted_one(videos, tmp_path):
    ef.extract_features(_manifest(videos), tmp_path / 'clean', streams=('rgb', 'motion'), **SMALL)
    real = FeatureStore.write
    writes = []

    def dying(self, stream, video_id, features, sidecar):
        if len(writes) == 3:  # v0 rgb, v0 motion, v1 rgb done; dies on v1 motion
            raise _Killed
        writes.append((stream, video_id))
        return real(self, stream, video_id, features, sidecar)

    with mock.patch.object(FeatureStore, 'write', dying), pytest.raises(_Killed):
        ef.extract_features(_manifest(videos), tmp_path / 'resumed', streams=('rgb', 'motion'), **SMALL)
    assert writes == [('rgb', 'v0'), ('motion', 'v0'), ('rgb', 'v1')]
    report = ef.extract_features(_manifest(videos), tmp_path / 'resumed', streams=('rgb', 'motion'), **SMALL)
    assert report.skipped == {'rgb': 2, 'motion': 1} and report.extracted == {'rgb': 1, 'motion': 2}
    for stream in ('rgb', 'motion'):
        clean, resumed = _arrays(tmp_path / 'clean', stream), _arrays(tmp_path / 'resumed', stream)
        assert clean.keys() == resumed.keys()
        assert all(np.array_equal(clean[k], resumed[k]) for k in clean)


def test_resume_only_decodes_what_is_left(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **SMALL)
    opened = []
    real = VideoReader.iter_frames

    def counting(self, *a, **k):
        opened.append(Path(self.source).stem)
        return real(self, *a, **k)

    with mock.patch.object(VideoReader, 'iter_frames', counting):
        ef.extract_features(_manifest(videos), tmp_path / 'f', **SMALL)
    assert opened == ['v1', 'v2']


@pytest.mark.parametrize('damage', ['array_missing', 'array_truncated', 'sidecar_missing', 'sidecar_garbage', 'shape_mismatch'])
def test_damaged_or_half_written_entries_are_extracted_again(videos, tmp_path, damage):
    df = _manifest(videos[1:2])
    ef.extract_features(df, tmp_path / 'f', **SMALL)
    good = _arrays(tmp_path / 'f')['v1']
    store = FeatureStore(tmp_path / 'f')
    array, side = store.array_path('rgb', 'v1'), store.sidecar_path('rgb', 'v1')
    if damage == 'array_missing':
        array.unlink()
    elif damage == 'array_truncated':
        array.write_bytes(array.read_bytes()[:40])
    elif damage == 'sidecar_missing':
        side.unlink()
    elif damage == 'sidecar_garbage':
        side.write_text('{ not json')
    else:
        meta = json.loads(side.read_text())
        meta['shape'] = [1, 1]
        side.write_text(json.dumps(meta))
    report = ef.extract_features(df, tmp_path / 'f', **SMALL)
    assert report.extracted['rgb'] == 1
    assert np.array_equal(_arrays(tmp_path / 'f')['v1'], good)


def test_interrupted_array_write_is_not_taken_as_done(videos, tmp_path):
    df = _manifest(videos[:1])
    with mock.patch('datasets.frame_cache.os.replace', side_effect=_Killed), pytest.raises(_Killed):
        ef.extract_features(df, tmp_path / 'f', **SMALL)
    report = ef.extract_features(df, tmp_path / 'f', **SMALL)
    assert report.extracted['rgb'] == 1 and report.skipped['rgb'] == 0


def test_a_bad_video_is_reported_and_the_rest_continue(videos, tmp_path):
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'not a video at all')
    df = pd.DataFrame({'videos': [str(videos[0]), str(bad), str(videos[1])]})
    report = ef.extract_features(df, tmp_path / 'f', **SMALL)
    assert report.extracted['rgb'] == 2 and len(report.failed) == 1
    failed = json.loads((tmp_path / 'f' / 'failed.json').read_text())['failed']
    assert 'bad' in failed[0]['video_id'] and failed[0]['error']
    bad.unlink()
    index_coded_clip(bad, 8, 32, 32)  # fixed: a retry succeeds and clears the failure list
    assert ef.extract_features(df, tmp_path / 'f', **SMALL).extracted['rgb'] == 1
    assert not (tmp_path / 'f' / 'failed.json').exists()


# -- float16 -----------------------------------------------------------------------------
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_values_are_refused(bad):
    with pytest.raises(FeatureValueError, match='NaN or infinite'):
        to_float16(np.array([1.0, bad, 2.0]))


@pytest.mark.parametrize('big', [65520.0, 1e5, -1e6])
def test_values_that_overflow_float16_are_refused(big):
    with pytest.raises(FeatureValueError, match='overflow'):
        to_float16(np.array([1.0, big]))


def test_largest_float16_value_is_accepted():
    out, stats = to_float16(np.array([65504.0, -65504.0]))
    assert np.isfinite(out).all() and out.dtype == np.float16 and stats['max_abs'] == 65504.0


def test_all_values_vanishing_is_refused_but_a_few_are_counted():
    with pytest.raises(FeatureValueError, match='rounds to 0'):
        to_float16(np.array([1e-9, -1e-9]))
    _, stats = to_float16(np.array([1.0, 1e-9, 0.0]))
    assert stats['flushed_to_zero'] == 1


def test_all_zero_features_are_fine():
    out, stats = to_float16(np.zeros(4))
    assert not out.any() and stats['flushed_to_zero'] == 0


def test_float16_precision_is_within_the_documented_bound_and_recorded():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 5, 10000).astype(np.float32)
    out, stats = to_float16(values)
    relative = np.abs(out.astype(np.float32) - values) / np.abs(values)
    assert relative.max() <= 2 ** -10  # half a unit in the last place of an 11-bit significand
    assert stats['max_abs_error'] == pytest.approx(float(np.abs(out.astype(np.float32) - values).max()))
    assert stats['max_error_relative_to_max'] <= 2 ** -10


def test_tiny_features_are_flagged(caplog):
    _, stats = to_float16(np.array([1e-5, -2e-5]))
    assert stats['subnormal_range_only'] is True
    assert 'precision' in caplog.text


def test_extraction_refuses_to_store_nan_features(videos, tmp_path):
    def nan_grid(self, x):
        return torch.full((x.shape[0], 1, 1, 1, 4), float('nan'))

    with mock.patch.object(SnippetBackbone, 'token_grid', nan_grid):
        report = ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **SMALL)
    assert report.extracted['rgb'] == 0 and 'NaN' in report.failed[0]['error']
    assert not list((tmp_path / 'f').rglob('*.npy')) and not list((tmp_path / 'f' / 'rgb').glob('*.json'))


def test_extraction_refuses_to_store_overflowing_features(videos, tmp_path):
    def huge_grid(self, x):
        return torch.full((x.shape[0], 1, 1, 1, 4), 1e6)

    with mock.patch.object(SnippetBackbone, 'token_grid', huge_grid):
        report = ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **SMALL)
    assert 'overflow' in report.failed[0]['error'] and not list((tmp_path / 'f').rglob('*.npy'))


def test_stored_features_are_float16_and_close_to_the_real_ones(videos, tmp_path):
    ef.extract_features(_manifest(videos[1:2]), tmp_path / 'f', **SMALL)
    stored = _arrays(tmp_path / 'f')['v1']
    assert stored.dtype == np.float16 and np.isfinite(stored).all()
    side = json.loads((tmp_path / 'f' / 'rgb' / 'v1.json').read_text())
    assert side['float16']['max_error_relative_to_max'] <= 2 ** -10
    assert side['float16']['max_abs'] > 0


# -- layout -------------------------------------------------------------------------------
def test_sidecar_holds_what_later_phases_need(videos, tmp_path):
    ef.extract_features(_manifest(videos[1:2]), tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    for stream in ('rgb', 'motion'):
        side = json.loads((tmp_path / 'f' / stream / 'v1.json').read_text())
        assert side['fps'] == pytest.approx(25)
        assert side['snippet_frames'] == 8 and side['stride'] == 8
        assert side['backbone'] == 'dummy_video3d' and side['weights'] is None
        assert side['stream'] == stream
        assert 'random' in side['weights_licence']
        assert side['shape'] == [3, side['dim']] and side['dtype'] == 'float16'
    motion = json.loads((tmp_path / 'f' / 'motion' / 'v1.json').read_text())
    assert motion['motion_source'] in ('codec', 'farneback') and 0 <= motion['intra_fraction'] <= 1
    assert 'clipped' in motion['motion_normalisation']


def test_concatenated_layout_records_the_split(videos, tmp_path):
    ef.extract_features(_manifest(videos), tmp_path / 'f', streams=('motion', 'rgb'), **SMALL)  # order in the flag is irrelevant
    loaded = load_snippet_features(tmp_path / 'f', 'v2')
    rgb = np.load(tmp_path / 'f' / 'rgb' / 'v2.npy')
    motion = np.load(tmp_path / 'f' / 'motion' / 'v2.npy')
    assert loaded.features.shape == (len(rgb), rgb.shape[1] + motion.shape[1])
    assert loaded.split == {'rgb': (0, rgb.shape[1]), 'motion': (rgb.shape[1], rgb.shape[1] + motion.shape[1])}
    assert np.array_equal(loaded.stream('rgb'), rgb) and np.array_equal(loaded.stream('motion'), motion)
    info = json.loads((tmp_path / 'f' / 'features.json').read_text())
    assert info['split'] == {'rgb': [0, rgb.shape[1]], 'motion': [rgb.shape[1], rgb.shape[1] + motion.shape[1]]}
    assert info['streams']['rgb']['dim'] == rgb.shape[1]


def test_streams_can_be_added_later_and_stay_aligned(videos, tmp_path):
    ef.extract_features(_manifest(videos), tmp_path / 'f', **SMALL)
    with mock.patch.object(VideoReader, 'iter_frames', side_effect=AssertionError('decoded rgb again')):
        ef.extract_features(_manifest(videos), tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    assert load_snippet_features(tmp_path / 'f', 'v1').features.shape[0] == 3


def test_loading_one_stream_and_missing_entries(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **SMALL)
    assert list(load_snippet_features(tmp_path / 'f', 'v0', streams=['rgb']).split) == ['rgb']
    with pytest.raises(FileNotFoundError):
        load_snippet_features(tmp_path / 'f', 'nope')
    with pytest.raises(FileNotFoundError):
        load_snippet_features(tmp_path / 'f', 'v0', streams=['motion'])
    with pytest.raises(ValueError, match='Unknown stream'):
        load_snippet_features(tmp_path / 'f', 'v0', streams=['flow'])


def test_trimmed_rows_extract_only_their_range(tmp_path):
    path = index_coded_clip(tmp_path / 'c.mp4', 50, 32, 32)
    df = pd.DataFrame({'videos': [str(path)], 'video_id': ['c'], 'start_sec': [0.4], 'end_sec': [1.0]})  # frames 10-24
    ef.extract_features(df, tmp_path / 'f', **SMALL)
    side = json.loads((tmp_path / 'f' / 'rgb' / 'c.json').read_text())
    assert (side['start_frame'], side['end_frame'], side['frame_count']) == (10, 25, 15)
    assert side['num_snippets'] == 2


def test_bytes_manifest_gets_stable_ids_and_extracts(videos, tmp_path):
    df = pd.DataFrame({'videos': [videos[0].read_bytes()], 'labels': [0]})
    ef.extract_features(df, tmp_path / 'f', **SMALL)
    dataset, columns = open_feature_source(df)
    assert (tmp_path / 'f' / 'rgb' / f'{row_video_id(dataset, columns, 0)}.npy').is_file()
    with _NoDecode():
        assert ef.extract_features(df, tmp_path / 'f', **SMALL).skipped['rgb'] == 1


def test_path_ids_do_not_depend_on_the_other_rows(tmp_path):
    a = tmp_path / 'walk' / 'x.mp4'
    b = tmp_path / 'run' / 'x.mp4'
    assert video_id_for_path(a) != video_id_for_path(b)
    assert video_id_for_path(a).startswith('x-')
    assert video_id_for_path(a) != video_id_for_path(a, 0.0, 1.0)


def test_duplicate_video_ids_are_refused(videos, tmp_path):
    df = pd.DataFrame({'videos': [str(videos[0]), str(videos[1])], 'video_id': ['same', 'same']})
    with pytest.raises(ValueError, match='unique'):
        ef.extract_features(df, tmp_path / 'f', **SMALL)


def test_unsafe_ids_cannot_escape_the_folder():
    assert '/' not in safe_video_id('../../etc/passwd') and '\\' not in safe_video_id('..\\..\\x')
    with pytest.raises(ValueError):
        safe_video_id('///')


def test_manifest_files_with_class_name_labels_work(videos, tmp_path):
    manifest = tmp_path / 'clips.csv'
    pd.DataFrame({'videos': [str(videos[0])], 'labels': ['walk']}).to_csv(manifest, index=False)
    assert ef.extract_features(manifest, tmp_path / 'f', **SMALL).extracted['rgb'] == 1


# -- motion -------------------------------------------------------------------------------
def test_motion_stream_works_on_all_intra_clips_through_farneback(tmp_path):
    path = translating_clip(tmp_path / 'intra.avi', 16, 2, 0, size=32, codec='mjpeg', pix_fmt='yuvj420p')
    report = ef.extract_features(pd.DataFrame({'videos': [str(path)], 'video_id': ['i']}), tmp_path / 'f',
                                 streams=('motion',), **SMALL)
    assert report.extracted['motion'] == 1
    assert json.loads((tmp_path / 'f' / 'motion' / 'i.json').read_text())['motion_source'] == 'farneback'


def test_codec_only_motion_fails_clearly_without_vectors(tmp_path):
    path = translating_clip(tmp_path / 'intra.avi', 16, 2, 0, size=32, codec='mjpeg', pix_fmt='yuvj420p')
    report = ef.extract_features(pd.DataFrame({'videos': [str(path)], 'video_id': ['i']}), tmp_path / 'f', streams=('motion',),
                                 motion_method='codec', **SMALL)
    assert report.extracted['motion'] == 0 and 'motion vectors' in report.failed[0]['error']


def test_motion_tensor_scales_to_input_pixels_and_clips():
    field_ = np.zeros((1, 2, 4, 4), np.float32)
    field_[:, 0] = 8.0   # 8 px/frame on a 64 px wide video
    field_[:, 1] = -400.0
    out = ef._motion_to_tensor(field_, (64, 32), 32)
    assert out.shape == (1, 2, 32, 32)
    assert out[0, 0].numpy() == pytest.approx(4.0 / ef.MOTION_CLIP_PIXELS)  # 8 * 32 / 64
    assert out[0, 1].numpy() == pytest.approx(-1.0)  # clipped


@pytest.mark.parametrize('rotation,expected', [(0, (1.0, 0.0)), (90, (0.0, -1.0)), (180, (-1.0, 0.0)), (270, (0.0, 1.0))])
def test_rotating_a_motion_field_turns_the_vectors_like_the_frames(rotation, expected):
    field_ = np.zeros((1, 2, 2, 3), np.float32)
    field_[:, 0] = 1.0  # moving right in the stored orientation
    out = ef.rotate_motion_field(field_, rotation)
    assert out[0, 0].flat[0] == pytest.approx(expected[0]) and out[0, 1].flat[0] == pytest.approx(expected[1])
    assert out.shape == ((1, 2, 3, 2) if rotation in (90, 270) else (1, 2, 2, 3))


def test_motion_differs_between_moving_and_still_clips(tmp_path):
    moving = translating_clip(tmp_path / 'm.mp4', 16, 3, 0, size=32, seed=3)
    still = translating_clip(tmp_path / 's.mp4', 16, 0, 0, size=32, seed=3)
    ef.extract_features(pd.DataFrame({'videos': [str(moving), str(still)], 'video_id': ['m', 's']}), tmp_path / 'f',
                        streams=('motion',), motion_method='farneback', **SMALL)
    arrays = _arrays(tmp_path / 'f', 'motion')
    assert not np.allclose(arrays['m'], arrays['s'])


# -- token grids --------------------------------------------------------------------------
@pytest.mark.parametrize('text,expected', [('none', None), ('', None), ('8', (8, 1, 1)), ('8x4x4', (8, 4, 4)), ('2X2X1', (2, 2, 1))])
def test_token_grid_parsing(text, expected):
    assert parse_token_grid(text) == expected


@pytest.mark.parametrize('text', ['8x4', '0x4x4', 'abc', '-1', '8x4x4x4'])
def test_bad_token_grids_are_refused(text):
    with pytest.raises(ValueError):
        parse_token_grid(text)


def test_token_grids_are_stored_and_capped_at_what_the_backbone_produces(videos, tmp_path):
    ef.extract_features(_manifest(videos), tmp_path / 'f', cache_tokens='8x2x2', **SMALL)  # r3d on 8 frames at 32 px: 1x2x2 grid
    stored = _arrays(tmp_path / 'f')['v2']
    assert stored.shape[:4] == (snippet_count(33, 8, 8), 1, 2, 2) and stored.dtype == np.float16
    side = json.loads((tmp_path / 'f' / 'rgb' / 'v2.json').read_text())
    assert side['token_grid'] == [1, 2, 2] and side['shape'] == list(stored.shape)
    assert json.loads((tmp_path / 'f' / 'features.json').read_text())['streams']['rgb']['token_grid'] == [1, 2, 2]


def test_token_grid_mean_matches_the_vector_mode(videos, tmp_path):
    ef.extract_features(_manifest(videos[1:2]), tmp_path / 'vector', **SMALL)
    ef.extract_features(_manifest(videos[1:2]), tmp_path / 'grid', cache_tokens='1x2x2', **SMALL)
    vector, grid = _arrays(tmp_path / 'vector')['v1'], _arrays(tmp_path / 'grid')['v1']
    assert np.allclose(grid.astype(np.float32).mean(axis=(1, 2, 3)), vector.astype(np.float32), atol=2e-3)


def test_grid_pooling_and_capping_are_correct():
    assert resolve_token_grid(None, (2, 7, 7)) is None
    assert resolve_token_grid((8, 4, 4), (2, 7, 7)) == (2, 4, 4)
    grid = torch.arange(16.).reshape(1, 1, 4, 4, 1)
    assert pool_token_grid(grid, (1, 2, 2)).flatten().tolist() == pytest.approx([2.5, 4.5, 10.5, 12.5])
    assert pool_token_grid(grid, None).item() == pytest.approx(7.5)
    assert pool_token_grid(grid, (1, 4, 4)) is grid


def test_concatenating_grids_of_different_shapes_is_a_clear_error(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **SMALL)
    root = tmp_path / 'f' / 'motion'
    root.mkdir()
    np.save(root / 'v0.npy', np.zeros((2, 3), np.float16))
    info = json.loads((tmp_path / 'f' / 'features.json').read_text())
    info['streams']['motion'] = {'dim': 3, 'config_hash': 'x', 'config': {}}
    (tmp_path / 'f' / 'features.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='cannot be concatenated'):
        load_snippet_features(tmp_path / 'f', 'v0')


# -- backbones ----------------------------------------------------------------------------
@pytest.mark.parametrize('version,frames,size,grid', [
    ('resnet18', 4, 32, (4, 1, 1)),
    ('densenet121', 2, 32, (2, 1, 1)),
    ('efficientnet_b0', 2, 32, (2, 1, 1)),
    ('vgg11', 2, 32, (2, 1, 1)),
    ('convnext_tiny', 2, 32, (2, 1, 1)),
    ('swin_t', 2, 64, (2, 2, 2)),
    ('vit_b_16', 2, 224, (2, 14, 14)),
    ('r3d_18', 8, 32, (1, 2, 2)),
    ('mc3_18', 8, 32, (8, 2, 2)),
    ('r2plus1d_18', 8, 32, (1, 2, 2)),
    ('swin3d_t', 4, 64, (2, 2, 2)),
    ('mvit_v2_s', 16, 224, (8, 7, 7)),
])
@pytest.mark.parametrize('channels', [3, 2])
def test_every_backbone_family_gives_a_grid_and_is_frozen(version, frames, size, grid, channels):
    model = SnippetBackbone(version, pretrained=False, in_channels=channels)
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    model.train()
    assert not model.training
    with torch.inference_mode():
        out = model.token_grid(torch.rand(1, frames, channels, size, size))
        pooled = model(torch.rand(1, frames, channels, size, size))
    assert out.ndim == 5 and tuple(out.shape[1:4]) == grid, out.shape
    assert pooled.shape == (1, out.shape[-1])
    del model, out


@pytest.mark.parametrize('version', ['resnet18', 'dummy_video3d'])
def test_motion_encoding_preserves_direction_and_uses_unchanged_rgb_weights(version):
    """Replaces the check that the first convolution held the RGB mean (V-039 item 1): with one filter repeated,
    frozen features depended only on dx + dy."""
    rgb = SnippetBackbone(version, pretrained=False, in_channels=3, seed=4)
    motion = SnippetBackbone(version, pretrained=False, in_channels=2, seed=4)
    assert all(torch.equal(a, b) for a, b in zip(rgb.state_dict().values(), motion.state_dict().values()))
    fields = torch.zeros(5, 4, 2, 32, 32)
    fields[0, :, 0] = 0.5  # right
    fields[1, :, 1] = 0.5  # down
    fields[2, :, 0], fields[2, :, 1] = 0.5, -0.5  # dx + dy = 0, which used to look like the still field
    fields[3, :, 0] = -0.5  # left
    with torch.inference_mode():
        features = motion(fields)
    still = features[4]
    assert not torch.allclose(features[0], features[1])
    assert not torch.allclose(features[2], still)
    assert not torch.allclose(features[0], features[3])


def test_random_weights_are_reproducible_and_do_not_disturb_the_global_rng():
    torch.manual_seed(5)
    expected = torch.rand(1)
    torch.manual_seed(5)
    a = SnippetBackbone('resnet18', pretrained=False)
    assert torch.equal(torch.rand(1), expected)
    b = SnippetBackbone('resnet18', pretrained=False)
    assert torch.equal(a.model.conv1.weight, b.model.conv1.weight)


def test_unknown_backbones_are_refused():
    with pytest.raises(ValueError, match='Unknown feature backbone'):
        SnippetBackbone('nope')
    with pytest.raises(ValueError, match='Unknown feature backbone'):
        SnippetBackbone('resnet7', pretrained=False)


def test_frame_backbone_rgb_stream(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', **{**SMALL, 'model_version': 'resnet18'})
    assert _arrays(tmp_path / 'f')['v0'].shape == (1, 512)


def test_default_preprocessing_paths_are_the_training_ones():
    frames = np.zeros((8, 40, 60, 3), np.uint8)
    assert ef._RgbPreprocess('r3d_18', True, None)(frames).shape == (8, 3, 112, 112)  # Kinetics preset
    assert ef._RgbPreprocess('resnet18', False, None)(frames).shape == (8, 3, 224, 224)  # ImageNet transform
    assert ef._RgbPreprocess('resnet18', False, 32)(frames).shape == (8, 3, 32, 32)


# -- CLI -----------------------------------------------------------------------------------
def test_cli_end_to_end_records_its_arguments(videos, tmp_path):
    manifest = tmp_path / 'clips.csv'
    _manifest(videos[:2]).to_csv(manifest, index=False)
    argv = ['--data', str(manifest), '--out', str(tmp_path / 'f'), '--streams', 'rgb', 'motion', '--model_version', 'dummy_video3d',
            '--motion_backbone', 'dummy_video3d', '--feature_weights', 'none', '--snippet_frames', '8', '--batch_size', '2',
            '--feature_input_size', '32', '--cache_tokens', '1x1x1']
    assert ef.main(argv) == 0
    saved = json.loads((tmp_path / 'f' / 'cli_arguments.json').read_text())
    assert set(saved) == set(SNIPPET_FEATURE_ARG_NAMES)
    assert saved['snippet_frames'] == 8 and saved['snippet_stride'] is None and saved['streams'] == ['rgb', 'motion']
    assert np.load(tmp_path / 'f' / 'rgb' / 'v1.npy').shape[1:4] == (1, 1, 1)


def test_cli_defaults_match_the_task():
    args = ef.build_parser().parse_args(['--data', 'x', '--out', 'y'])
    assert args.snippet_frames == 16 and args.snippet_stride is None and args.streams == ['rgb'] and args.cache_tokens == 'none'


def test_cli_refuses_a_bad_grid(tmp_path):
    with pytest.raises(SystemExit):
        ef.main(['--data', 'x', '--out', str(tmp_path / 'f'), '--cache_tokens', '8x4'])


def test_cli_returns_nonzero_when_a_video_fails(tmp_path):
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'nope')
    manifest = tmp_path / 'clips.csv'
    pd.DataFrame({'videos': [str(bad)]}).to_csv(manifest, index=False)
    assert ef.main(['--data', str(manifest), '--out', str(tmp_path / 'f'), '--model_version', 'dummy_video3d', '--feature_weights', 'none']) == 1


def test_empty_video_is_reported(tmp_path):
    empty = tmp_path / 'empty.mp4'
    empty.write_bytes(b'')
    report = ef.extract_features(pd.DataFrame({'videos': [str(empty)]}), tmp_path / 'f', **SMALL)
    assert report.failed and 'empty' in report.failed[0]['error'].lower()


def test_duplicate_explicit_id_with_different_trims_is_refused(videos, tmp_path):
    df = pd.DataFrame({'videos': [str(videos[0]), str(videos[2]), str(videos[2])], 'video_id': ['first', 'same', 'same'],
                       'start_sec': [0.0, 0.0, 0.4], 'end_sec': [0.2, 0.4, 0.8]})
    with pytest.raises(ValueError, match='trims.*unique'):
        ef.extract_features(df, tmp_path / 'f', **SMALL)
    assert not (tmp_path / 'f' / 'rgb').exists()  # refused before the clean first row cost any decoding


def test_rows_repeating_one_trim_under_one_id_share_their_features(videos, tmp_path):
    df = pd.DataFrame({'videos': [str(videos[2])] * 2, 'video_id': ['same'] * 2, 'start_sec': [0.0] * 2, 'end_sec': [0.4] * 2})
    assert ef.extract_features(df, tmp_path / 'f', **SMALL).extracted['rgb'] == 1


def test_streams_from_different_trims_are_not_concatenated(videos, tmp_path):
    df = pd.DataFrame({'videos': [str(videos[2])], 'video_id': ['same'], 'start_sec': [0.0], 'end_sec': [0.4]})
    ef.extract_features(df, tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    ef.extract_features(df.assign(start_sec=0.4, end_sec=0.8), tmp_path / 'f', streams=('rgb',), **SMALL)  # same N
    with pytest.raises(ValueError, match='different trims'):
        load_snippet_features(tmp_path / 'f', 'same')
    assert load_snippet_features(tmp_path / 'f', 'same', streams=['rgb']).meta['rgb']['source_range']['start_sec'] == 0.4


def test_resume_reextracts_changed_trim_and_aligns_new_motion(videos, tmp_path):
    df = pd.DataFrame({'videos': [str(videos[2])], 'video_id': ['same'],
                       'start_sec': [0.0], 'end_sec': [0.4]})
    ef.extract_features(df, tmp_path / 'f', **SMALL)
    df['start_sec'], df['end_sec'] = 0.4, 1.2
    report = ef.extract_features(df, tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    assert report.extracted == {'rgb': 1, 'motion': 1}
    loaded = load_snippet_features(tmp_path / 'f', 'same')
    for side in loaded.meta.values():
        assert side['source_range'] == {'start_sec': 0.4, 'end_sec': 1.2}
        assert (side['start_frame'], side['end_frame']) == (10, 30)
    assert loaded.meta['motion']['motion_weights_trained_on_motion'] is False
    with _NoDecode():
        assert ef.extract_features(df, tmp_path / 'f', streams=('rgb', 'motion'), **SMALL).skipped == {'rgb': 1, 'motion': 1}


@pytest.mark.parametrize('version', ['r3d_18', 'mvit_v2_s', 'swin3d_t'])
def test_size_override_preserves_each_native_weights_normalisation(version):
    from src.video.video3d import _KINETICS400_WEIGHTS
    preset = _KINETICS400_WEIGHTS[version].transforms()
    output = ef._RgbPreprocess(version, True, 16)(np.full((2, 20, 30, 3), 128, np.uint8))
    expected = (128 / 255 - torch.tensor(preset.mean)) / torch.tensor(preset.std)
    assert torch.allclose(output[0, :, 0, 0], expected)


def test_sidecar_records_actual_frame_preprocessing_size(videos, tmp_path, monkeypatch):
    class TinyBackbone(torch.nn.Module):
        native = False
        default_input_size = 380
        weights_name = None
        weights_licence = 'none: random'

        def token_grid(self, x):
            return x.mean(dim=(-1, -2)).unsqueeze(2).unsqueeze(2)

    monkeypatch.setattr('src.video.feature_backbones.SnippetBackbone', lambda *a, **kw: TinyBackbone())
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f',
                        **{**SMALL, 'model_version': 'efficientnet_b4', 'feature_input_size': None})
    assert FeatureStore(tmp_path / 'f').read_sidecar('rgb', 'v0')['input_size'] == 224


def test_weight_licence_comes_from_selected_weights_without_loading_them(monkeypatch):
    from torchvision.models import ViT_H_14_Weights
    monkeypatch.setattr('src.video.feature_backbones.tv_models.get_model', lambda *a, **kw: torch.nn.Identity())
    model = SnippetBackbone('vit_h_14', pretrained=True)
    assert str(ViT_H_14_Weights.DEFAULT) in model.weights_licence
    assert ViT_H_14_Weights.DEFAULT.meta['license'] in model.weights_licence
    assert 'CC BY-NC 4.0' in model.weights_licence


@pytest.mark.parametrize('version,data', [('resnet18', 'ImageNet-1K'), ('r3d_18', 'Kinetics-400'), ('mvit_v2_s', 'Kinetics-400')])
def test_weights_without_a_licence_in_their_metadata_do_not_claim_one(monkeypatch, version, data):
    import torchvision.models as tv
    from src.video.feature_backbones import weight_licence
    monkeypatch.setattr('src.video.feature_backbones.tv_models.get_model', lambda *a, **kw: torch.nn.Identity())
    weights = tv.get_model_weights(version).DEFAULT
    assert weights.meta.get('license') is None
    licence = SnippetBackbone(version, pretrained=True).weights_licence
    assert licence == weight_licence(weights)
    assert str(weights) in licence and 'no weight licence' in licence and data in licence and weights.meta['recipe'] in licence
    assert 'random' in SnippetBackbone(version, pretrained=False).weights_licence


def test_refused_cli_run_keeps_original_arguments(videos, tmp_path):
    manifest = tmp_path / 'clips.csv'
    _manifest(videos[:1]).to_csv(manifest, index=False)
    argv = ['--data', str(manifest), '--out', str(tmp_path / 'f'), '--model_version', 'dummy_video3d',
            '--feature_weights', 'none', '--feature_input_size', '32', '--snippet_frames', '8']
    assert ef.main(argv) == 0
    before = (tmp_path / 'f' / 'cli_arguments.json').read_bytes()
    with pytest.raises(FeatureConfigError):
        ef.main(argv[:-1] + ['4'])
    assert (tmp_path / 'f' / 'cli_arguments.json').read_bytes() == before


@pytest.mark.parametrize('method', ['codec', 'farneback'])
def test_rotated_motion_is_backend_independent(tmp_path, method):
    from datasets.video_synth import write_clip
    path = translating_clip(tmp_path / 'original.mp4', 20, 2, 0, size=32, seed=5)
    with VideoReader(path, backend='pyav') as reader:
        frames = list(reader.iter_frames())
    rotated = write_clip(tmp_path / 'rotated.mp4', frames, rotation=90)
    outputs = []
    for backend in ('pyav', 'opencv'):
        extractor = ef._StreamExtractor('motion', {'snippet_frames': 8, 'stride': 8, 'motion_method': method}, None, 2, 'cpu')
        extractor._ensure_backbone = lambda: None
        extractor.input_size = 32
        extractor._run_batches = lambda tensors: np.stack([tensor.numpy() for tensor in tensors])
        with VideoReader(rotated, backend=backend) as reader:
            output, _ = extractor.extract_motion(reader, 0, 20)
            outputs.append(output)
    # Codec vectors are the same numbers. Farneback runs on stored frames for pyav and on display frames for opencv, and
    # flow is not exactly rotation-equivariant: measured 0.009 px at most, so allow 0.02 px of the +-20 px range.
    assert np.allclose(*outputs, atol=1e-5 if method == 'codec' else 0.02 / ef.MOTION_CLIP_PIXELS)
    # ... and the direction agrees with what the display-orientation frames show, measured without our rotation.
    with VideoReader(rotated, backend='pyav') as reader:
        shown = reader.get_frames(np.arange(9, 12))
    flow = np.mean([farneback_flow(a, b)[8:24, 8:24].mean(axis=(0, 1)) for a, b in zip(shown, shown[1:])], axis=0)
    vectors = outputs[0][:, :, :, 8:24, 8:24].mean(axis=(0, 1, 3, 4))
    assert np.dot(flow, vectors) > 0 and np.argmax(np.abs(flow)) == np.argmax(np.abs(vectors))


# -- second fix round (V-042) ---------------------------------------------------------
def test_ids_differing_only_by_case_are_refused(videos, tmp_path):
    """Clip and clip are one file on NTFS, so two videos would silently share one set of features."""
    df = pd.DataFrame({'videos': [str(videos[0]), str(videos[1])], 'video_id': ['Clip', 'clip'], 'labels': [0, 1]})
    with pytest.raises(ValueError, match='differ only by case'):
        ef.extract_features(df, tmp_path / 'f', **SMALL)
    assert not (tmp_path / 'f' / 'rgb').exists()
    with pytest.raises(ValueError, match='differ only by case'):  # the same clip too: on Linux these are two files
        ef.extract_features(df.assign(videos=str(videos[0])), tmp_path / 'g', **SMALL)


def test_a_later_run_cannot_overwrite_an_id_differing_only_by_case(videos, tmp_path):
    (tmp_path / 'probe').write_text('')
    if not (tmp_path / 'PROBE').exists():
        pytest.skip('this file system tells Clip and clip apart, so they are two entries')
    ef.extract_features(pd.DataFrame({'videos': [str(videos[0])], 'video_id': ['Clip']}), tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='would overwrite'):
        ef.extract_features(pd.DataFrame({'videos': [str(videos[1])], 'video_id': ['clip']}), tmp_path / 'f', **SMALL)


class _GeometryBackbone(torch.nn.Module):
    """Passes its input through as a token grid: rgb brightness, or motion magnitude."""
    native, default_input_size, weights_name, weights_licence = True, 64, None, 'none'

    def __init__(self, version, pretrained=False, in_channels=3):
        super().__init__()

    def token_grid(self, x):
        values = torch.linalg.vector_norm(x, dim=2) if x.shape[2] == 2 else x.mean(dim=2)
        return values.unsqueeze(-1)


def _marker_clip(path, width, height, left, top, frames=16):
    """A textured square jittering on black, so it is the brightest and the only moving thing."""
    from datasets.video_synth import write_clip
    texture = np.random.default_rng(0).integers(150, 255, (16, 16, 3), dtype=np.uint8)
    clip = []
    for i in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        x, y = left + (i % 4) * 2, top + (i % 3) * 2
        frame[y:y + 16, x:x + 16] = texture
        clip.append(frame)
    return write_clip(path, clip)


@pytest.mark.parametrize('tokens', ['1x4x4', 'none'])
@pytest.mark.parametrize('model,motion', [('r3d_18', 'r3d_18'), ('mvit_v2_s', 'mvit_v2_s'), ('swin3d_t', 'swin3d_t'), ('mvit_v2_s', 'r3d_18')])
def test_a_marker_lands_in_the_same_cell_in_both_streams(tmp_path, monkeypatch, model, motion, tokens):
    """16:9 on native presets, which crop the sides. x = 30% was cell 0 of rgb but cell 1 of a squashed motion field."""
    monkeypatch.setattr('src.video.feature_backbones.SnippetBackbone', _GeometryBackbone)
    path = _marker_clip(tmp_path / 'wide.mp4', 256, 144, 70, 60)
    args = dict(model_version=model, motion_backbone=motion, feature_weights='none', snippet_frames=16, batch_size=1,
                cache_tokens='1x4x4')
    manifest = pd.DataFrame({'videos': [str(path)], 'video_id': ['wide']})
    report = ef.extract_features(manifest, tmp_path / 'f', streams=('rgb', 'motion'), motion_method='codec', **args)
    assert not report.failed, report.failed
    loaded = load_snippet_features(tmp_path / 'f', 'wide')
    rgb, moving = loaded.stream('rgb')[0, 0, :, :, 0], loaded.stream('motion')[0, 0, :, :, 0]
    assert np.unravel_index(rgb.argmax(), rgb.shape) == np.unravel_index(moving.argmax(), moving.shape)
    assert loaded.meta['rgb']['field_of_view'] == loaded.meta['motion']['field_of_view']
    if tokens == 'none':  # a pooled vector sees the same part of the frame, because the cut happens before the backbone
        pooled = dict(args, cache_tokens='none')
        ef.extract_features(manifest, tmp_path / 'p', streams=('rgb', 'motion'), motion_method='codec', **pooled)
        sides = load_snippet_features(tmp_path / 'p', 'wide').meta
        assert sides['rgb']['field_of_view'] == sides['motion']['field_of_view'] == loaded.meta['rgb']['field_of_view']


@pytest.mark.parametrize('version', ['r3d_18', 'mvit_v2_s', 'swin3d_t'])
@pytest.mark.parametrize('width,height', [(256, 144), (144, 256), (120, 120), (171, 128)])
def test_crop_window_is_what_the_rgb_preset_really_keeps(version, width, height):
    """Marks each column and row of a frame, runs the real preset and reads the kept range back."""
    preprocess = ef._RgbPreprocess(version, True, None)
    ramp_x = np.tile(np.linspace(0, 255, width)[None, :, None], (height, 1, 3)).astype(np.uint8)
    ramp_y = np.tile(np.linspace(0, 255, height)[:, None, None], (1, width, 3)).astype(np.uint8)
    x0, x1, y0, y1 = ef.crop_window(version, None, width, height)
    mean, std = preprocess.mean[0, 0, 0, 0].item(), preprocess.std[0, 0, 0, 0].item()
    seen_x = preprocess(np.stack([ramp_x] * 2))[0, 0].numpy() * std + mean
    seen_y = preprocess(np.stack([ramp_y] * 2))[0, 0].numpy() * std + mean
    assert seen_x[0].min() == pytest.approx(x0, abs=0.03) and seen_x[0].max() == pytest.approx(x1, abs=0.03)
    assert seen_y[:, 0].min() == pytest.approx(y0, abs=0.03) and seen_y[:, 0].max() == pytest.approx(y1, abs=0.03)


def test_a_size_override_and_frame_backbones_keep_the_whole_frame():
    assert ef.crop_window('r3d_18', 64, 256, 144) == ef.FULL_FRAME
    assert ef.crop_window('resnet18', None, 256, 144) == ef.FULL_FRAME


def test_cropped_motion_is_in_input_pixels_of_the_crop():
    field_ = np.zeros((1, 2, 18, 32), np.float32)
    field_[:, 0] = 4.0  # 4 display pixels per frame on a 256 x 144 frame
    field_[:, 1] = 2.0
    x0, x1, y0, y1 = window = ef.crop_window('mvit_v2_s', None, 256, 144)
    out = ef._motion_to_tensor(field_, (256, 144), 224, window, gain=1.0)
    assert out[0, 0].numpy() == pytest.approx(4.0 * 224 / (256 * (x1 - x0)))
    assert out[0, 1].numpy() == pytest.approx(2.0 * 224 / (144 * (y1 - y0)))
    # A gradient across the frame shows that the sides really are cut off.
    ramp = np.zeros((1, 2, 18, 32), np.float32)
    ramp[:, 0] = np.linspace(0, 1, 32)
    full, cut = [ef._motion_to_tensor(ramp, (256, 144), 32, w, gain=1.0) for w in (ef.FULL_FRAME, (0.25, 0.75, 0.0, 1.0))]
    assert cut[0, 0, 0, 0] > 2 * full[0, 0, 0, 0]


def test_streams_seeing_different_parts_of_the_frame_are_not_concatenated(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    path = tmp_path / 'f' / 'motion' / 'v0.json'
    sidecar = json.loads(path.read_text())
    sidecar['field_of_view'] = [0.25, 0.75, 0.0, 1.0]
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match='different parts of the frame'):
        load_snippet_features(tmp_path / 'f', 'v0')


def test_motion_gain_is_recorded_and_changes_the_input_not_silently(videos, tmp_path):
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', streams=('motion',), **SMALL)
    info = FeatureStore(tmp_path / 'f').info()['streams']['motion']['config']
    sidecar = FeatureStore(tmp_path / 'f').read_sidecar('motion', 'v0')
    assert info['motion_gain'] == sidecar['motion_gain'] == ef.MOTION_GAIN == pytest.approx(1 / ef.MOTION_CLIP_PIXELS)
    assert info['motion_clip_pixels'] == sidecar['motion_clip_pixels'] == ef.MOTION_CLIP_PIXELS
    assert f'{ef.MOTION_GAIN:g}' in sidecar['motion_normalisation']
    with pytest.raises(FeatureConfigError, match='motion_gain'):  # one folder, one gain
        ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', streams=('motion',), motion_gain=1.0, **SMALL)
    ef.extract_features(_manifest(videos[:1]), tmp_path / 'g', streams=('motion',), motion_gain=1.0, **SMALL)
    assert FeatureStore(tmp_path / 'g').read_sidecar('motion', 'v0')['motion_gain'] == 1.0
    assert not np.array_equal(_arrays(tmp_path / 'f', 'motion')['v0'], _arrays(tmp_path / 'g', 'motion')['v0'])
    assert ef.build_parser().parse_args(['--data', 'd', '--out', 'o']).motion_gain == ef.MOTION_GAIN
    assert ef.build_parser().parse_args(['--data', 'd', '--out', 'o', '--motion_gain', '0.2']).motion_gain == 0.2
    with pytest.raises(SystemExit):
        ef.build_parser().parse_args(['--data', 'd', '--out', 'o', '--motion_gain', '0'])
    assert 'motion_gain' in SNIPPET_FEATURE_ARG_NAMES


@pytest.mark.parametrize('settings,stream,advice', [
    (dict(model_version='vit_h_14', feature_weights='pretrained'), 'rgb', '--feature_input_size 518'),
    (dict(model_version='vit_h_14', feature_weights='pretrained', feature_input_size=224), 'rgb', '--feature_input_size 518'),
    (dict(model_version='vit_b_16', feature_weights='none', feature_input_size=32), 'rgb', '--feature_input_size 224'),
    (dict(model_version='mvit_v2_s', feature_weights='none', snippet_frames=8), 'rgb', '--snippet_frames 16'),
    (dict(model_version='mvit_v2_s', feature_weights='none', snippet_frames=32), 'rgb', '--snippet_frames 16'),
    (dict(model_version='mvit_v2_s', feature_weights='none', snippet_frames=16, feature_input_size=112), 'rgb', '--feature_input_size 224'),
    (dict(motion_backbone='mvit_v2_s', feature_weights='none', snippet_frames=8), 'motion', '--snippet_frames 16'),
])
def test_impossible_model_input_is_refused_up_front(videos, tmp_path, settings, stream, advice):
    with mock.patch('src.video.feature_backbones.SnippetBackbone', side_effect=AssertionError('built a model')), _NoDecode():
        with pytest.raises(FeatureConfigError, match=advice):
            ef.extract_features(_manifest(videos[:1]), tmp_path / 'f', streams=(stream,), **settings)
    assert not (tmp_path / 'f').exists()


@pytest.mark.parametrize('stream,version,pretrained,size,frames', [
    ('rgb', 'vit_h_14', True, 518, 16), ('motion', 'vit_h_14', True, None, 16), ('rgb', 'vit_b_16', False, None, 2),
    ('rgb', 'mvit_v2_s', False, None, 16), ('rgb', 'r3d_18', False, 64, 8), ('rgb', 'swin3d_t', False, 64, 8), ('rgb', 'resnet18', False, 32, 3),
])
def test_workable_model_input_is_accepted(stream, version, pretrained, size, frames):
    ef.check_feature_input(stream, version, pretrained, size, frames)


@pytest.mark.parametrize('version,size,frames', [('vit_b_16', 112, 2), ('mvit_v2_s', 224, 8), ('mvit_v2_s', 112, 16)])
def test_what_the_check_refuses_really_cannot_run(version, size, frames):
    """Keeps the check honest: each refused input fails in the real model."""
    model = SnippetBackbone(version, pretrained=False)
    with pytest.raises((AssertionError, RuntimeError)), torch.inference_mode():
        model.token_grid(torch.zeros(1, frames, 3, size, size))

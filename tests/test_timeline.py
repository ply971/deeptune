"""Phase 4 Part A: sliding-window timeline (analysis/timeline.py, infer.py --timeline).

Hand-written probability sequences for windows, smoothing and segmentation; the
synthetic untrimmed oracle for boundary error and mAP (with a pixel-reading stub
scorer); the real clip pipeline for streaming and the CLI end to end."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.timeline import (Window, make_windows, median_smooth, owned_intervals, run_timeline,
                               run_video_timelines, segments_from_timeline, static_windows, write_outputs)

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_windows_cover_the_video_and_the_last_one_ends_at_the_end():
    ws = make_windows(10.0, 2.0, 1.0)
    assert [w.start for w in ws] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert ws[-1].end == 10.0 and all(w.end - w.start == pytest.approx(2.0) for w in ws)
    ws = make_windows(10.5, 2.0, 1.0)
    assert ws[-2].start == 8.0 and ws[-1].start == pytest.approx(8.5) and ws[-1].end == pytest.approx(10.5)
    assert [w.index for w in ws] == list(range(len(ws)))


def test_a_video_shorter_than_one_window_gets_one_window():
    assert make_windows(1.5, 2.0, 1.0) == [Window(0, 0.0, 1.5)]
    assert make_windows(2.0, 2.0, 1.0) == [Window(0, 0.0, 2.0)]


@pytest.mark.parametrize('duration,window,stride', [(0, 2, 1), (-1, 2, 1), (float('nan'), 2, 1), (5, 0, 1), (5, 2, 0)])
def test_bad_window_arguments_are_rejected(duration, window, stride):
    with pytest.raises(ValueError):
        make_windows(duration, window, stride)


def test_owned_intervals_partition_the_video():
    ws = make_windows(10.5, 2.0, 1.0)
    iv = owned_intervals(ws, 10.5)
    assert iv[0][0] == 0.0 and iv[-1][1] == 10.5
    assert all(a[1] == b[0] for a, b in zip(iv, iv[1:]))
    assert all(a < b for a, b in iv)
    # window k owns the stretch around its centre
    assert iv[3] == (pytest.approx(3.5), pytest.approx(4.5))


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

A, B = [0.9, 0.1], [0.2, 0.8]


def test_median_smoothing_removes_one_window_flicker_and_keeps_rows_normalised():
    probs = np.array([A, A, B, A, A])
    out = median_smooth(probs, 3)
    assert (out.argmax(axis=1) == 0).all()
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[0], A)  # edges: median of the windows available


def test_median_smoothing_keeps_a_real_change_of_class():
    probs = np.array([A, A, A, B, B, B])
    assert median_smooth(probs, 3).argmax(axis=1).tolist() == [0, 0, 0, 1, 1, 1]


def test_smoothing_width_one_changes_nothing_and_even_widths_are_rejected():
    probs = np.array([A, B, A])
    assert np.array_equal(median_smooth(probs, 1), probs)
    with pytest.raises(ValueError):
        median_smooth(probs, 2)
    with pytest.raises(ValueError):
        median_smooth(probs[0], 3)


def test_single_window_smoothing_is_identity():
    assert np.array_equal(median_smooth(np.array([B]), 5), np.array([B]))


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def test_single_window_gives_one_segment():
    segs = segments_from_timeline([(0.0, 1.5)], ['jump'], [0.7])
    assert len(segs) == 1 and (segs[0].start, segs[0].end, segs[0].label) == (0.0, 1.5, 'jump')
    assert segs[0].confidence == pytest.approx(0.7) and segs[0].windows == 1


def test_all_windows_of_one_class_merge_into_one_segment_with_mean_confidence():
    iv = [(0, 1), (1, 2), (2, 3), (3, 4)]
    segs = segments_from_timeline(iv, ['run'] * 4, [0.6, 0.8, 1.0, 0.6])
    assert len(segs) == 1
    assert (segs[0].start, segs[0].end, segs[0].windows) == (0, 4, 4)
    assert segs[0].confidence == pytest.approx(0.75)


def test_min_segment_absorbs_a_short_flicker_into_its_neighbours():
    iv = [(k, k + 1) for k in range(7)]
    labels = ['run', 'run', 'run', 'jump', 'run', 'run', 'run']
    assert len(segments_from_timeline(iv, labels, [0.9] * 7)) == 3
    segs = segments_from_timeline(iv, labels, [0.9] * 7, min_segment_sec=2)
    assert [(s.start, s.end, s.label, s.windows) for s in segs] == [(0, 7, 'run', 7)]


def test_min_segment_goes_to_the_longer_neighbour():
    iv = [(k, k + 1) for k in range(6)]
    labels = ['a', 'a', 'a', 'x', 'b', 'b']
    segs = segments_from_timeline(iv, labels, [1.0] * 6, min_segment_sec=1.5)
    assert [(s.start, s.end, s.label) for s in segs] == [(0, 4, 'a'), (4, 6, 'b')]


def test_background_is_dropped_but_still_separates_two_events():
    iv = [(k, k + 1) for k in range(5)]
    segs = segments_from_timeline(iv, ['jump', 'jump', 'none', 'jump', 'jump'], [0.8] * 5, background=['none'])
    assert [(s.start, s.end, s.label) for s in segs] == [(0, 2, 'jump'), (3, 5, 'jump')]


def test_segments_never_overlap_and_lengths_must_agree():
    iv = owned_intervals(make_windows(12.0, 2.0, 0.5), 12.0)
    rng = np.random.default_rng(0)
    labels = rng.choice(['a', 'b', 'c'], size=len(iv)).tolist()
    segs = segments_from_timeline(iv, labels, rng.random(len(iv)).tolist(), min_segment_sec=1.0)
    assert all(s.end <= t.start + 1e-9 for s, t in zip(segs, segs[1:]))
    assert segs[0].start == 0 and segs[-1].end == 12.0
    assert all(s.duration >= 1.0 - 1e-9 for s in segs)
    with pytest.raises(ValueError):
        segments_from_timeline(iv, labels[:-1], [0.5] * len(iv))


# ---------------------------------------------------------------------------
# run_timeline
# ---------------------------------------------------------------------------

def _truth_scorer(events, classes):
    """Probabilities from the share of each window inside a labelled event (a perfect, blurred model)."""
    def score(windows):
        rows = []
        for w in windows:
            p = np.zeros(len(classes))
            for (a, b, label) in events:
                p[classes.index(label)] += max(0.0, min(b, w.end) - max(a, w.start)) / (w.end - w.start)
            p[classes.index('none')] += max(0.0, 1 - p.sum())
            rows.append(p / p.sum())
        return np.array(rows)
    return score


def test_run_timeline_recovers_events_within_one_stride():
    classes = ['none', 'jump', 'run']
    events = [(4.0, 9.0, 'jump'), (14.0, 22.0, 'run')]
    res = run_timeline(30.0, _truth_scorer(events, classes), classes, window_sec=2.0, stride_sec=0.5,
                       smooth_windows=3, background=['none'])
    assert [s.label for s in res.segments] == ['jump', 'run']
    for s, (a, b, _) in zip(res.segments, events):
        assert abs(s.start - a) <= 0.5 and abs(s.end - b) <= 0.5
    assert res.scored == len(res.windows)


def test_score_windows_is_called_in_chunks_and_its_shape_is_checked():
    calls = []

    def score(ws):
        calls.append(len(ws))
        return np.tile([0.3, 0.7], (len(ws), 1))
    res = run_timeline(20.0, score, ['a', 'b'], 2.0, 1.0, batch_windows=4)
    assert max(calls) <= 4 and sum(calls) == len(res.windows) == 19
    with pytest.raises(ValueError, match='expected'):
        run_timeline(20.0, lambda ws: np.zeros((len(ws), 3)), ['a', 'b'], 2.0, 1.0)


def test_low_confidence_windows_become_uncertain_background():
    probs = {0: [0.5, 0.5], 1: [0.95, 0.05]}
    res = run_timeline(6.0, lambda ws: np.array([probs[int(w.start >= 3)] for w in ws]), ['a', 'b'], 1.0, 1.0,
                       smooth_windows=1, min_confidence=0.6)
    assert res.labels == ['uncertain'] * 3 + ['a'] * 3
    assert [(s.start, s.end, s.label) for s in res.segments] == [(3.0, 6.0, 'a')]


def test_static_windows_are_skipped_not_scored():
    scored = []

    def score(ws):
        scored.extend(w.index for w in ws)
        return np.tile([0.1, 0.9], (len(ws), 1))
    res = run_timeline(10.0, score, ['a', 'b'], 1.0, 1.0, static=lambda ws: np.array([w.start < 4 for w in ws]))
    assert scored == [4, 5, 6, 7, 8, 9] and res.scored == 6
    assert res.labels[:4] == ['static'] * 4
    assert [(s.start, s.end, s.label) for s in res.segments] == [(4.0, 10.0, 'b')]


def test_static_window_flags_from_a_motion_curve():
    fps = 10
    motion = np.r_[np.zeros(30), np.full(30, 5.0)]  # still for 3 s, then moving
    ws = make_windows(6.0, 1.0, 1.0)
    assert static_windows(ws, motion, fps, 1.0).tolist() == [True, True, True, False, False, False]
    assert not static_windows(ws, motion, fps, 0.0).any()


def test_outputs_are_written(tmp_path):
    classes = ['none', 'jump']
    res = run_timeline(8.0, _truth_scorer([(2.0, 5.0, 'jump')], classes), classes, 2.0, 1.0, background=['none'])
    paths = write_outputs(res, tmp_path, classes, video='v.mp4', settings={'window_sec': 2.0})
    table = pd.read_csv(paths['timeline'])
    assert len(table) == len(res.windows)
    assert {'start_sec', 'end_sec', 'label', 'confidence', 'probability_jump', 'raw_probability_none'} <= set(table.columns)
    assert np.allclose(table[['probability_none', 'probability_jump']].sum(axis=1), 1.0, atol=1e-3)
    seg = json.loads(paths['segments'].read_text())
    assert seg['classes'] == classes and seg['settings'] == {'window_sec': 2.0}
    assert [s['label'] for s in seg['segments']] == ['jump']
    assert set(seg['segments'][0]) == {'start_sec', 'end_sec', 'label', 'confidence', 'windows'}


# ---------------------------------------------------------------------------
# Synthetic untrimmed oracle (Phase 2): boundary error and mAP
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def untrimmed(tmp_path_factory):
    from benchmarks.synthetic_motion import make_untrimmed_dataset
    return make_untrimmed_dataset(tmp_path_factory.mktemp('untrimmed'), n_videos=4, length_range=(120, 160),
                                  n_classes=1, seed=5, size=48, max_instances=3, background_motion='mixed',
                                  codec='mp4v')


def _redness_scorer(action_label):
    """Stub model: reads only the window's frames and says "action" for the share showing the red blob."""
    from benchmarks.render import BLOB_REDNESS_THRESHOLD, redness
    from datasets.video_io import VideoReader
    decoded = []

    def score_rows(frame):
        out = []
        for row in frame.itertuples():
            with VideoReader(row.videos) as reader:
                a, b = reader.time_to_index(row.start_sec), reader.time_to_end_index(row.end_sec)
                frames = reader.get_frames(list(range(a, b)))
            decoded.append(len(frames))
            share = float(((redness(frames) > BLOB_REDNESS_THRESHOLD).sum(axis=(1, 2)) > 2).mean())
            out.append([1 - share, share])
        return np.array(out)
    score_rows.decoded = decoded
    return score_rows


def test_timeline_on_synthetic_untrimmed_videos_finds_every_action_within_one_stride(untrimmed, tmp_path):
    meta = untrimmed.load_metadata()
    label = untrimmed.classes[0]
    videos = [str(untrimmed.root / info['file']) for info in meta['videos'].values()]
    scorer = _redness_scorer(label)
    stride = 0.2
    settings = dict(window_sec=0.6, stride_sec=stride, smooth_windows=1, min_segment_sec=0.0, background=['background'])
    record = run_video_timelines(videos, tmp_path, ['background', label], scorer, settings,
                                 ground_truth=untrimmed.extra['gt_segments'], log=lambda *_: None)
    m = record['metrics']
    assert m['n_videos'] == 4
    assert m['detection_map']['map_per_threshold']['0.5'] >= 0.9
    assert m['boundary_error']['n_matched'] == m['detection_map']['n_ground_truth']
    assert m['boundary_error']['start_seconds'] <= stride and m['boundary_error']['end_seconds'] <= stride
    # streaming: every call decodes at most one window of frames, however long the video
    assert max(scorer.decoded) <= int(round(0.6 * meta['fps'])) + 1
    pred = pd.read_csv(tmp_path / 'segments.csv')
    assert list(pred.columns) == ['video-id', 't-start', 't-end', 'label', 'score']
    for vid in meta['videos']:
        assert (tmp_path / 'timeline' / vid / 'timeline.csv').exists()
        assert (tmp_path / 'timeline' / vid / 'segments.json').exists()
    run = json.loads((tmp_path / 'timeline_run.json').read_text())
    assert run['total_windows'] == run['scored_windows'] > 0


def test_skip_static_skips_still_background_and_keeps_the_actions(tmp_path):
    from benchmarks.synthetic_motion import make_untrimmed_dataset
    untrimmed = make_untrimmed_dataset(tmp_path / 'still', n_videos=2, length_range=(120, 140), n_classes=1, seed=2,
                                       size=48, max_instances=2, background_motion='static', codec='mp4v')
    meta = untrimmed.load_metadata()
    label = untrimmed.classes[0]
    videos = [str(untrimmed.root / info['file']) for info in meta['videos'].values()]
    gt = pd.read_csv(untrimmed.extra['gt_segments'])
    settings = dict(window_sec=0.6, stride_sec=0.2, smooth_windows=1, background=['background'],
                    skip_static=0.5, motion_source='framediff')
    record = run_video_timelines(videos, tmp_path / 'out', ['background', label], _redness_scorer(label), settings,
                                 ground_truth=gt, log=lambda *_: None)
    assert record['scored_windows'] < record['total_windows']
    assert record['metrics']['detection_map']['map_per_threshold']['0.5'] >= 0.9


# ---------------------------------------------------------------------------
# The real clip pipeline: one window = one decode of that window's frames
# ---------------------------------------------------------------------------

def test_clip_dataset_decodes_only_each_windows_frames(tmp_path, monkeypatch):
    pytest.importorskip('av')
    from datasets.video_datasets import VideoDataset
    from datasets.video_io import VideoReader
    from datasets.video_synth import translating_clip
    clip = translating_clip(tmp_path / 'long.mp4', 120, 1, 0, size=32, seed=0)
    with VideoReader(str(clip)) as reader:
        fps = reader.metadata().fps
    windows = make_windows(120 / fps, 1.0, 0.5)
    df = pd.DataFrame({'videos': [str(clip)] * len(windows), 'start_sec': [w.start for w in windows],
                       'end_sec': [w.end for w in windows]})
    calls = []
    original = VideoReader.get_frames

    def spy(self, indices, clamp_to_end=False):
        calls.append(np.asarray(indices))
        return original(self, indices, clamp_to_end=clamp_to_end)
    monkeypatch.setattr(VideoReader, 'get_frames', spy)
    dataset = VideoDataset(df, None, 4)
    for i, w in enumerate(windows):
        dataset[i]
        idx = calls[-1]
        assert len(idx) == 4
        assert idx.min() >= int(round(w.start * fps)) - 1 and idx.max() <= int(round(w.end * fps))
    assert len(calls) == len(windows)


# ---------------------------------------------------------------------------
# infer.py --timeline
# ---------------------------------------------------------------------------

def test_timeline_flags_are_checked_before_anything_loads(tmp_path):
    import infer
    base = ['--modality', 'video', '--checkpoint', str(tmp_path / 'missing.pt'), '--data', str(tmp_path),
            '--out', str(tmp_path / 'o')]
    for extra in (['--window_sec', '3'], ['--timeline', '--stride_sec', '3', '--window_sec', '2'],
                  ['--timeline', '--smooth_windows', '2'], ['--timeline', '--mode', 'reg'],
                  ['--timeline', '--min_confidence', '1.5'], ['--timeline', '--window_sec', '0'],
                  ['--timeline', '--ground_truth', str(tmp_path / 'nope.csv')]):
        with pytest.raises(SystemExit) as exc:
            infer.main(base + extra)
        assert exc.value.code == 2, extra
    assert not (tmp_path / 'o').exists()


def test_infer_timeline_end_to_end_with_a_trained_run(tmp_path, monkeypatch):
    pytest.importorskip('av')
    import torchvision

    import desktop.config as config
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import inference.video as infer_mod
    import trainers.video.train as train_mod
    from datasets.video_synth import translating_clip
    small = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    for module in (train_mod, eval_mod, embed_mod, infer_mod):
        monkeypatch.setattr(module, 'transformations', small)
    versions = dict(config.MODEL_VERSIONS_BY_MODALITY)
    versions['video'] = [*versions['video'], 'dummy_video3d']
    monkeypatch.setattr(config, 'MODEL_VERSIONS_BY_MODALITY', versions)

    (tmp_path / 'clips').mkdir()
    (tmp_path / 'new').mkdir()
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(6):
            path = translating_clip(tmp_path / 'clips' / f'{label}_{i}.mp4', 10, dx, 0, size=32, seed=i)
            rows.append({'videos': str(path), 'labels': label})
    manifest = tmp_path / 'clips.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    import deeptune
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'dummy_video3d', '--num_epochs', '1',
                   '--batch_size', '4', '--num_frames', '2', '--added_layers', '1', '--embed_size', '16',
                   '--fixed-seed', '--df', str(manifest), '--raw-data'])
    run = next(out.glob('deeptune-*'))

    long_clip = translating_clip(tmp_path / 'new' / 'long.mp4', 60, 2, 0, size=32, seed=9)
    from desktop.config import build_infer_argv, infer_config_from_run, read_run_results
    argv = build_infer_argv(infer_config_from_run(read_run_results(run), long_clip, tmp_path / 'tl'))
    argv += ['--timeline', '--window_sec', '1', '--stride_sec', '0.5', '--smooth_windows', '3']
    import infer
    assert infer.main(argv) == 0
    folder = tmp_path / 'tl' / 'timeline' / 'long'
    table = pd.read_csv(folder / 'timeline.csv')
    seg = json.loads((folder / 'segments.json').read_text())
    assert set(seg['classes']) == {'left', 'right'}
    assert len(table) > 1 and table['label'].isin(['left', 'right']).all()
    assert np.allclose(table[[f'raw_probability_{c}' for c in seg['classes']]].sum(axis=1), 1.0, atol=1e-3)
    assert seg['segments'] and seg['segments'][0]['start_sec'] == 0 and seg['segments'][-1]['end_sec'] == pytest.approx(seg['duration_sec'])
    run_record = json.loads((tmp_path / 'tl' / 'timeline_run.json').read_text())
    assert run_record['settings']['window_sec'] == 1.0 and run_record['videos'][0]['windows'] == len(table)
    assert not (tmp_path / 'tl' / 'predictions.csv').exists()

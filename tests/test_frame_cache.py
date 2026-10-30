"""Decoded-frame cache: no second decode, correct keys, random clips never reused, crash safety."""
import json
import logging
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip('av')

from datasets.frame_cache import (FrameCache, atomic_save_npy, clip_sampling_is_cacheable, frame_cache_key, motion_curve_key,
                                  resize_short_side)
from datasets.video_datasets import ParquetVideoDataset, VideoDataOptions, VideoDataset, video_options_from_args
from datasets.video_io import VideoReader
from datasets.video_sampling import STRATEGIES, SamplingSpec, sample_clip_indices
from datasets.video_synth import index_coded_clip, translating_clip

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def clip(tmp_path):
    return index_coded_clip(tmp_path / 'clip.mp4', 24, 64, 48)


def _dataset(path, cache, train=False, size=0, num_frames=4, num_clips=1, **options):
    return VideoDataset(pd.DataFrame({'videos': [str(path)], 'labels': [1]}), num_frames=num_frames, train=train,
                        num_clips=num_clips, options=VideoDataOptions(frame_cache=str(cache), frame_cache_size=size, **options))


def _entries(cache):
    return sorted(Path(cache).rglob('*.npy'))


# -- hits ----------------------------------------------------------------------
def test_second_read_does_not_decode_and_matches_an_uncached_read(clip, tmp_path):
    plain = VideoDataset(pd.DataFrame({'videos': [str(clip)], 'labels': [1]}), num_frames=4)[0][0]
    first = _dataset(clip, tmp_path / 'cache')[0][0]
    assert torch.equal(plain, first)
    assert len(_entries(tmp_path / 'cache')) == 1
    with mock.patch.object(VideoReader, 'get_frames', side_effect=AssertionError('decoded on a cache hit')):
        second = _dataset(clip, tmp_path / 'cache')[0][0]
    assert torch.equal(first, second)


def test_entry_is_a_memory_mappable_uint8_array(clip, tmp_path):
    _dataset(clip, tmp_path / 'cache')[0]
    mapped = np.load(_entries(tmp_path / 'cache')[0], mmap_mode='r')
    assert isinstance(mapped, np.memmap)
    assert mapped.dtype == np.uint8 and mapped.shape == (4, 48, 64, 3)
    del mapped


def test_bytes_datasets_use_the_cache_too(clip, tmp_path):
    df = pd.DataFrame({'videos': [clip.read_bytes()], 'labels': [0]})
    options = VideoDataOptions(frame_cache=str(tmp_path / 'cache'), frame_cache_size=0)
    first = ParquetVideoDataset(df, num_frames=3, options=options)[0][0]
    with mock.patch.object(VideoReader, 'get_frames', side_effect=AssertionError('decoded on a cache hit')):
        second = ParquetVideoDataset(df, num_frames=3, options=options)[0][0]
    assert torch.equal(first, second)


def test_multi_clip_samples_are_cached_with_their_shape(clip, tmp_path):
    first = _dataset(clip, tmp_path / 'cache', num_clips=2)[0][0]
    assert first.shape[:2] == (2, 4)
    with mock.patch.object(VideoReader, 'get_frames', side_effect=AssertionError('decoded on a cache hit')):
        assert torch.equal(first, _dataset(clip, tmp_path / 'cache', num_clips=2)[0][0])


def test_motion_sampling_hit_decodes_nothing_at_all(tmp_path):
    path = translating_clip(tmp_path / 'moving.mp4', 24, 2, 0, size=48, static_frames=12)
    first = _dataset(path, tmp_path / 'cache', temporal_sampling='motion')[0][0]
    assert len(_entries(tmp_path / 'cache')) == 2  # the clip and its motion curve
    with mock.patch.object(VideoReader, 'get_frames', side_effect=AssertionError('decoded frames')), \
            mock.patch.object(VideoReader, 'motion_curve', side_effect=AssertionError('recomputed the motion curve')):
        assert torch.equal(first, _dataset(path, tmp_path / 'cache', temporal_sampling='motion')[0][0])


def test_cache_works_through_a_pickled_dataset(clip, tmp_path):
    dataset = pickle.loads(pickle.dumps(_dataset(clip, tmp_path / 'cache')))
    dataset[0]
    assert len(_entries(tmp_path / 'cache')) == 1


# -- keys ----------------------------------------------------------------------
BASE = dict(fingerprint='f' * 40, spec=SamplingSpec(num_frames=8), indices=np.arange(8), size=256, backend='pyav')


@pytest.mark.parametrize('change', [
    dict(fingerprint='e' * 40),
    dict(spec=SamplingSpec(num_frames=8, strategy='segment')),
    dict(spec=SamplingSpec(num_frames=8, strategy='dense', clip_stride=2)),
    dict(spec=SamplingSpec(num_frames=8, clip_duration_sec=1.0)),
    dict(spec=SamplingSpec(num_frames=9)),
    dict(indices=np.arange(1, 9)),
    dict(size=128),
    dict(backend='opencv'),
])
def test_key_changes_with_every_input(change):
    assert frame_cache_key(**{**BASE, **change}) != frame_cache_key(**BASE)


def test_key_is_stable_across_processes():
    code = ('import numpy as np; from datasets.frame_cache import frame_cache_key; from datasets.video_sampling import SamplingSpec;'
            "print(frame_cache_key('f'*40, SamplingSpec(num_frames=8), np.arange(8), 256, 'pyav'))")
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    assert out == frame_cache_key(**BASE)


def test_changed_source_file_is_a_miss(tmp_path):
    path = index_coded_clip(tmp_path / 'clip.mp4', 24, 64, 48)
    first = _dataset(path, tmp_path / 'cache')[0][0]
    old = path.stat().st_mtime_ns
    index_coded_clip(path, 40, 64, 48)
    os.utime(path, ns=(old + 10 ** 9, old + 10 ** 9))
    second = _dataset(path, tmp_path / 'cache')[0][0]
    assert len(_entries(tmp_path / 'cache')) == 2
    assert not torch.equal(first, second)


def test_changed_sampling_is_a_miss(clip, tmp_path):
    _dataset(clip, tmp_path / 'cache', num_frames=4)[0]
    _dataset(clip, tmp_path / 'cache', num_frames=5)[0]
    _dataset(clip, tmp_path / 'cache', num_frames=4, temporal_sampling='dense')[0]
    assert len(_entries(tmp_path / 'cache')) == 3


def test_motion_curve_key_depends_on_source_and_method():
    assert motion_curve_key('a', 'auto', 'pyav') != motion_curve_key('b', 'auto', 'pyav')
    assert motion_curve_key('a', 'auto', 'pyav') != motion_curve_key('a', 'framediff', 'pyav')


# -- random clips ----------------------------------------------------------------
TRAIN_SPECS = [dict(strategy=s, clip_duration_sec=d, temporal_jitter=j, speed_perturb=p) for s in STRATEGIES
               for d in (None, 0.4) for j in (False, True) for p in (0.0, 0.2)]


@pytest.mark.parametrize('spec', TRAIN_SPECS, ids=str)
def test_cacheable_rule_agrees_with_what_the_sampler_really_does(spec):
    """Independent check of the table: a mode is cacheable exactly when repeated draws never differ."""
    full = SamplingSpec(num_frames=6, **spec)
    motion = np.sin(np.linspace(0, 9, 60)) ** 2
    draws = {sample_clip_indices(60, full, train=True, rng=np.random.default_rng(seed), fps=25, motion=motion).tobytes()
             for seed in range(30)}
    assert clip_sampling_is_cacheable(full, train=True) == (len(draws) == 1)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_evaluation_sampling_is_always_cacheable_and_deterministic(strategy):
    full = SamplingSpec(num_frames=6, strategy=strategy, clip_duration_sec=0.4)
    assert clip_sampling_is_cacheable(full, train=False)
    motion = np.sin(np.linspace(0, 9, 60)) ** 2
    draws = {sample_clip_indices(60, full, train=False, rng=np.random.default_rng(seed), fps=25, motion=motion).tobytes()
             for seed in range(10)}
    assert len(draws) == 1


def test_random_training_clips_bypass_the_cache_and_stay_random(clip, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger='deeptune.video'):
        dataset = _dataset(clip, tmp_path / 'cache', train=True, temporal_sampling='segment', num_frames=6)
    assert 'not cached' in caplog.text
    torch.manual_seed(0)
    draws = {dataset[0][0].numpy().tobytes() for _ in range(8)}
    assert len(draws) > 1
    assert _entries(tmp_path / 'cache') == []


def test_augmented_training_clips_are_not_cached_but_validation_is(clip, tmp_path):
    train = _dataset(clip, tmp_path / 'cache', train=True, video_augment=True)
    torch.manual_seed(0)
    assert len({train[0][0].numpy().tobytes() for _ in range(6)}) > 1
    assert _entries(tmp_path / 'cache') == []
    _dataset(clip, tmp_path / 'cache', train=False, video_augment=True)[0]
    assert len(_entries(tmp_path / 'cache')) == 1


def test_deterministic_training_clips_are_cached(clip, tmp_path):
    _dataset(clip, tmp_path / 'cache', train=True)[0]
    assert len(_entries(tmp_path / 'cache')) == 1


def test_bypassed_clips_get_the_same_resize_as_cached_ones(clip, tmp_path):
    random_train = _dataset(clip, tmp_path / 'cache', train=True, temporal_sampling='segment', size=24)[0][0]
    val = _dataset(clip, tmp_path / 'cache', train=False, temporal_sampling='segment', size=24)[0][0]
    assert random_train.shape == val.shape
    assert min(val.shape[-2:]) == 24


# -- resize ----------------------------------------------------------------------
def test_resize_short_side_keeps_aspect_and_never_enlarges():
    frames = np.zeros((2, 100, 200, 3), np.uint8)
    assert resize_short_side(frames, 50).shape == (2, 50, 100, 3)
    assert resize_short_side(np.zeros((2, 200, 100, 3), np.uint8), 50).shape == (2, 100, 50, 3)
    assert resize_short_side(frames, 400) is frames
    assert resize_short_side(frames, 0) is frames


def test_negative_size_is_refused():
    with pytest.raises(ValueError):
        VideoDataOptions(frame_cache='x', frame_cache_size=-1)


# -- crash safety ------------------------------------------------------------------
def test_interrupted_replace_leaves_no_entry(tmp_path):
    cache = FrameCache(tmp_path / 'cache', size=0)
    frames = np.ones((2, 4, 4, 3), np.uint8)
    with mock.patch('datasets.frame_cache.os.replace', side_effect=OSError('disk full')):
        cache.put_frames('ab' + '0' * 38, frames)  # logged, does not raise
    assert cache.get_frames('ab' + '0' * 38, 2) is None
    assert not list((tmp_path / 'cache').rglob('*.part'))


def test_process_killed_while_writing_leaves_no_readable_entry(tmp_path):
    """A real hard kill in the middle of a write (os._exit: no cleanup runs)."""
    code = f"""
import os, numpy as np
from datasets import frame_cache as fc
real_save = np.save
def half_then_die(stream, array, allow_pickle=False):
    stream.write(b'\\x93NUMPY half an entry')
    stream.flush()
    os._exit(9)
np.save = half_then_die
fc.FrameCache({str(tmp_path / 'cache')!r}, size=0).put_frames('ab' + '0' * 38, np.ones((2, 4, 4, 3), np.uint8))
"""
    assert subprocess.run([sys.executable, '-c', code], cwd=REPO).returncode == 9
    assert len(list((tmp_path / 'cache').rglob('*.part'))) == 1  # the debris exists
    cache = FrameCache(tmp_path / 'cache', size=0)
    assert cache.get_frames('ab' + '0' * 38, 2) is None
    cache.put_frames('ab' + '0' * 38, np.ones((2, 4, 4, 3), np.uint8))
    assert cache.get_frames('ab' + '0' * 38, 2) is not None


def test_stale_part_files_are_swept_and_fresh_ones_kept(tmp_path):
    (tmp_path / 'cache' / 'ab').mkdir(parents=True)
    old, fresh = tmp_path / 'cache' / 'ab' / 'old.npy.1.x.part', tmp_path / 'cache' / 'ab' / 'new.npy.2.y.part'
    old.write_bytes(b'x'), fresh.write_bytes(b'y')
    os.utime(old, (1, 1))
    FrameCache(tmp_path / 'cache')
    assert not old.exists() and fresh.exists()


@pytest.mark.parametrize('damage', ['truncate', 'garbage', 'pad', 'wrong_shape', 'wrong_dtype', 'empty'])
def test_invalid_entry_is_a_miss_and_is_rewritten(clip, tmp_path, damage):
    good = _dataset(clip, tmp_path / 'cache')[0][0]
    path = _entries(tmp_path / 'cache')[0]
    raw = path.read_bytes()
    if damage == 'truncate':
        path.write_bytes(raw[:len(raw) // 2])
    elif damage == 'garbage':
        path.write_bytes(b'not a numpy file at all')
    elif damage == 'pad':
        path.write_bytes(raw + b'\0' * 16)
    elif damage == 'empty':
        path.write_bytes(b'')
    elif damage == 'wrong_shape':
        np.save(path, np.zeros((3, 48, 64, 3), np.uint8))
    else:
        np.save(path, np.zeros((4, 48, 64, 3), np.float32))
    again = _dataset(clip, tmp_path / 'cache')[0][0]
    assert torch.equal(good, again)
    mapped = np.load(path, mmap_mode='r')
    assert mapped.shape == (4, 48, 64, 3) and mapped.dtype == np.uint8
    del mapped


def test_concurrent_writers_and_readers_never_see_a_partial_entry(tmp_path):
    frames = np.full((4, 32, 32, 3), 7, np.uint8)
    key = 'cd' + '0' * 38
    wrong, stop = [], threading.Event()

    def writer():
        cache = FrameCache(tmp_path / 'cache', size=0)
        for _ in range(25):
            cache.put_frames(key, frames)

    def reader():
        cache = FrameCache(tmp_path / 'cache', size=0)
        while not stop.is_set():
            got = cache.get_frames(key, 4)
            if got is not None and not np.array_equal(got, frames):
                wrong.append(got)

    readers = [threading.Thread(target=reader) for _ in range(2)]
    writers = [threading.Thread(target=writer) for _ in range(4)]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()
    assert not wrong
    assert np.array_equal(FrameCache(tmp_path / 'cache', size=0).get_frames(key, 4), frames)
    assert not list((tmp_path / 'cache').rglob('*.part'))


def test_atomic_save_keeps_the_old_entry_when_the_new_write_fails(tmp_path):
    path = tmp_path / 'x.npy'
    atomic_save_npy(path, np.arange(3))
    with mock.patch('datasets.frame_cache.np.save', side_effect=OSError('disk full')), pytest.raises(OSError):
        atomic_save_npy(path, np.arange(9))
    assert np.load(path).tolist() == [0, 1, 2]


def test_unusable_cache_folder_does_not_stop_a_run(clip, tmp_path, caplog):
    blocker = tmp_path / 'file_not_folder'
    blocker.write_text('x')
    with caplog.at_level(logging.WARNING, logger='deeptune.video'):
        sample = _dataset(clip, blocker / 'cache')[0][0]
    assert sample.shape[0] == 4
    assert 'Frame cache' in caplog.text


# -- flags ---------------------------------------------------------------------------
def test_flags_default_off_and_are_recorded(tmp_path):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    base = ['--modality', 'video', '--df', 'x.parquet', '--out', str(tmp_path), '--model_version', 'resnet18', '--batch_size', '2']
    default = DeepTuneVisionOptions(RunType.ONECALL, args=base)
    assert default.frame_cache is None and default.frame_cache_size == 256
    assert video_options_from_args(default) == VideoDataOptions()
    used = DeepTuneVisionOptions(RunType.ONECALL, args=base + ['--frame_cache', str(tmp_path / 'c'), '--frame_cache_size', '128'])
    used.save_args(tmp_path)
    saved = json.loads((tmp_path / 'cli_arguments.json').read_text(encoding='utf-8'))
    assert saved['frame_cache'] == str(tmp_path / 'c') and saved['frame_cache_size'] == 128
    options = video_options_from_args(saved)
    assert options.frame_cache == str(tmp_path / 'c') and options.frame_cache_size == 128


def test_inference_rebuilds_the_recorded_cache_settings(tmp_path):
    infer = pytest.importorskip('infer')
    (tmp_path / 'cli_arguments.json').write_text(json.dumps({'frame_cache': 'C:/somewhere', 'frame_cache_size': 128}), encoding='utf-8')
    argv = ['--modality', 'video', '--checkpoint', str(tmp_path / 'weights.pth'), '--data', '.', '--out', str(tmp_path)]
    args = infer.build_parser().parse_args(argv)
    infer.apply_recorded_video_settings(args, argv)
    assert args.frame_cache == 'C:/somewhere' and args.frame_cache_size == 128


def test_desktop_test_page_passes_the_cache_settings_on():
    from desktop.config import INFER_VIDEO_OPTION_DEFAULTS, InferenceConfig, build_infer_argv
    assert INFER_VIDEO_OPTION_DEFAULTS['frame_cache'] is None and INFER_VIDEO_OPTION_DEFAULTS['frame_cache_size'] == 256
    cfg = InferenceConfig(modality='video', checkpoint_path=Path('w.pth'), data=Path('d'), out=Path('o'),
                          video_options={'frame_cache': 'C:/c', 'frame_cache_size': 128})
    argv = build_infer_argv(cfg)
    assert argv[argv.index('--frame_cache') + 1] == 'C:/c' and argv[argv.index('--frame_cache_size') + 1] == '128'


# -- whole pipeline ---------------------------------------------------------------------
def test_deeptune_and_the_test_page_run_through_the_cache(tmp_path, monkeypatch):
    """deeptune.py -> train / eval / embed -> infer.py, with the settings rebuilt from the saved run."""
    import torchvision
    import deeptune
    import desktop.config as config
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import infer
    import inference.video as infer_mod
    import trainers.video.train as train_mod
    small = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    for module in (train_mod, eval_mod, embed_mod, infer_mod):
        monkeypatch.setattr(module, 'transformations', small)
    versions = dict(config.MODEL_VERSIONS_BY_MODALITY)
    versions['video'] = [*versions['video'], 'dummy_video3d']
    monkeypatch.setattr(config, 'MODEL_VERSIONS_BY_MODALITY', versions)

    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(6):
            rows.append({'videos': str(translating_clip(tmp_path / f'{label}_{i}.mp4', 10, dx, 0, size=32, seed=i)), 'labels': label})
    manifest = tmp_path / 'clips.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cache = tmp_path / 'cache'
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'dummy_video3d', '--num_epochs', '1', '--batch_size', '4',
                   '--num_frames', '2', '--added_layers', '1', '--embed_size', '16', '--fixed-seed', '--df', str(manifest), '--raw-data',
                   '--frame_cache', str(cache), '--frame_cache_size', '24'])
    run = next(out.glob('deeptune-*'))
    assert len(_entries(cache)) >= 12  # train, val and test clips (deterministic uniform sampling)
    assert all(min(np.load(p).shape[1:3]) == 24 for p in _entries(cache))
    saved = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    assert saved['frame_cache'] == str(cache) and saved['frame_cache_size'] == 24

    test_rows = pd.read_parquet(next(run.glob('data_splits_*/test_split.parquet')))
    data = tmp_path / 'new.csv'
    test_rows[['videos', 'labels']].to_csv(data, index=False)
    argv = config.build_infer_argv(config.infer_config_from_run(config.read_run_results(run), data, tmp_path / 'infer'))
    assert argv[argv.index('--frame_cache') + 1] == str(cache) and argv[argv.index('--frame_cache_size') + 1] == '24'
    with mock.patch.object(VideoReader, 'get_frames', side_effect=AssertionError('test clips were cached by the evaluation')):
        assert infer.main(argv) == 0
    assert len(pd.read_csv(tmp_path / 'infer' / 'predictions.csv')) == len(test_rows)


def test_cache_modules_do_not_import_torch():
    code = ("import sys, datasets.frame_cache, datasets.snippet_features, handlers.video_args;"
            "sys.exit('torch' in sys.modules)")
    assert subprocess.run([sys.executable, '-c', code], cwd=REPO).returncode == 0

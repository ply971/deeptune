"""Video datasets (paths and bytes), lazy loading, worker safety, multi-clip TTA,
group-aware splits, robustness and recorded CLI options."""
import builtins
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip('av')

from datasets import video_datasets as vd
from datasets.video_datasets import (ParquetVideoDataset, VideoDataOptions, VideoDataset, collate_video_samples,
                                     detect_video_storage, open_video_dataset, video_loader_kwargs,
                                     video_options_from_args)
from datasets.video_synth import decode_frame_index, index_coded_clip, translating_clip

SMALL = None  # transform=None keeps raw float frames: easy to decode back to indices


def _index(clip):
    """Frame indices of a float [T, C, H, W] clip in [0, 1]."""
    return [decode_frame_index((frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for frame in clip]


@pytest.fixture()
def clip_files(tmp_path):
    paths = [index_coded_clip(tmp_path / f'clip{i}.mp4', 20 + 5 * i) for i in range(4)]
    return tmp_path, paths


def test_path_and_bytes_formats_give_identical_samples(clip_files, tmp_path):
    _, paths = clip_files
    by_path = pd.DataFrame({'videos': [str(p) for p in paths], 'labels': [0, 1, 0, 1]})
    by_bytes = pd.DataFrame({'videos': [p.read_bytes() for p in paths], 'labels': [0, 1, 0, 1]})
    a, b = open_video_dataset(by_path, num_frames=4), open_video_dataset(by_bytes, num_frames=4)
    assert isinstance(a, VideoDataset) and isinstance(b, ParquetVideoDataset)
    for i in range(4):
        (ca, la), (cb, lb) = a[i], b[i]
        assert torch.equal(ca, cb) and la == lb
        assert _index(ca) == np.linspace(0, 20 + 5 * i - 1, 4).astype(int).tolist()


def test_class_name_labels_are_rejected_with_a_hint(clip_files):
    _, paths = clip_files
    with pytest.raises(ValueError, match='numeric'):
        VideoDataset(pd.DataFrame({'videos': [str(paths[0])], 'labels': ['walk']}))


def test_storage_is_detected_from_parquet_schema_and_csv(clip_files, tmp_path):
    _, paths = clip_files
    pq_paths = tmp_path / 'paths.parquet'
    pd.DataFrame({'videos': [str(p) for p in paths], 'labels': [0, 1, 0, 1]}).to_parquet(pq_paths)
    pq_bytes = tmp_path / 'bytes.parquet'
    pd.DataFrame({'videos': [p.read_bytes() for p in paths], 'labels': [0, 1, 0, 1]}).to_parquet(pq_bytes)
    assert detect_video_storage(pq_paths) == 'paths' and detect_video_storage(pq_bytes) == 'bytes'
    assert detect_video_storage(tmp_path / 'list.csv') == 'paths'
    assert isinstance(ParquetVideoDataset.from_parquet(pq_paths, num_frames=2), VideoDataset)
    assert isinstance(ParquetVideoDataset.from_parquet(pq_bytes, num_frames=2), ParquetVideoDataset)


def test_relative_manifest_paths_resolve_against_the_manifest(clip_files):
    root, paths = clip_files
    manifest = root / 'list.csv'
    pd.DataFrame({'videos': [p.name for p in paths], 'labels': [0, 1, 0, 1], 'group': list('wxyz')}).to_csv(manifest, index=False)
    dataset = open_video_dataset(manifest, num_frames=2)
    clip, label, extras = dataset[2]
    assert clip.shape == (2, 3, 32, 32) and extras == {'group': 'y'}


def test_thousand_clip_manifest_constructs_without_reading_any_clip(tmp_path, monkeypatch):
    clip = index_coded_clip(tmp_path / 'one.mp4', 8)
    df = pd.DataFrame({'videos': [str(clip)] * 1000, 'labels': np.arange(1000) % 3})
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).endswith('.mp4'):
            raise AssertionError(f'clip read at construction: {file}')
        return real_open(file, *args, **kwargs)

    import av
    import cv2
    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(Path, 'read_bytes', lambda self: (_ for _ in ()).throw(AssertionError('read_bytes')))
    monkeypatch.setattr(av, 'open', lambda *a, **k: (_ for _ in ()).throw(AssertionError('av.open')))
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **k: (_ for _ in ()).throw(AssertionError('VideoCapture')))
    dataset = VideoDataset(df, num_frames=4)
    assert len(dataset) == 1000
    assert len(pickle.dumps(dataset)) < 200_000  # paths only, no clip bytes


def test_bytes_dataset_moves_clips_out_of_worker_pickles(tmp_path):
    # Distinct clips: pickle would de-duplicate repeated byte objects.
    paths = [index_coded_clip(tmp_path / f'w{i}.mp4', 30 + i, 48, 48) for i in range(40)]
    df = pd.DataFrame({'videos': [p.read_bytes() for p in paths], 'labels': [0, 1] * 20})
    dataset = ParquetVideoDataset(df, num_frames=2)
    clip_bytes = sum(len(v) for v in df['videos'])
    assert len(pickle.dumps(dataset)) > clip_bytes  # in memory before
    first = dataset[3][0]
    dataset.prepare_for_workers()
    payload = pickle.dumps(dataset)
    assert len(payload) < clip_bytes / 10
    clone = pickle.loads(payload)
    assert torch.equal(clone[3][0], first)


def test_large_bytes_parquet_is_memory_mapped(clip_files, tmp_path, monkeypatch):
    _, paths = clip_files
    source = tmp_path / 'big.parquet'
    pd.DataFrame({'videos': [p.read_bytes() for p in paths], 'labels': [0, 1, 0, 1], 'note': list('abcd')}).to_parquet(source)
    monkeypatch.setattr(vd, 'LAZY_PARQUET_BYTES', 0)
    lazy = ParquetVideoDataset.from_parquet(source, num_frames=3)
    eager = ParquetVideoDataset(pd.read_parquet(source), num_frames=3)
    assert isinstance(lazy._store, vd.ArrowBytesStore)
    assert list(source.parent.glob('.deeptune_video_cache/*.arrow'))
    for i in range(4):
        assert torch.equal(lazy[i][0], eager[i][0]) and lazy[i][2] == {'note': 'abcd'[i]}
    assert len(pickle.dumps(lazy)) < 50_000
    again = ParquetVideoDataset.from_parquet(source, num_frames=3)  # reuses the cache
    assert again._store.path == lazy._store.path


def test_two_workers_load_a_path_dataset(clip_files):
    _, paths = clip_files
    dataset = VideoDataset(pd.DataFrame({'videos': [str(p) for p in paths], 'labels': [0, 1, 0, 1]}), num_frames=2)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_video_samples)
    batches = list(loader)
    assert len(batches) == 2 and batches[0][0].shape == (2, 2, 3, 32, 32)


def test_multi_clip_dataset_returns_n_clips(clip_files):
    _, paths = clip_files
    dataset = VideoDataset(pd.DataFrame({'videos': [str(paths[3])]}), num_frames=4, num_clips=3)
    clips, label = dataset[0]
    assert clips.shape == (3, 4, 3, 32, 32) and label is None
    starts = [_index(c)[0] for c in clips]
    assert len(set(starts)) == 3


def test_segment_and_motion_sampling_through_the_dataset(tmp_path):
    path = translating_clip(tmp_path / 'half.mp4', 40, 3, 0, static_frames=20, options={'refs': '1', 'bf': '0'})
    df = pd.DataFrame({'videos': [str(path)]})
    motion = VideoDataset(df, num_frames=8, options=VideoDataOptions(temporal_sampling='motion'))
    indices = motion.sample_indices(0, vd.VideoReader(str(path)))[0]
    assert (indices >= 20).sum() >= 7, indices
    seg = VideoDataset(df, num_frames=8, train=True, options=VideoDataOptions(temporal_sampling='segment'))
    torch.manual_seed(0)
    a = seg.sample_indices(0, vd.VideoReader(str(path)))
    torch.manual_seed(0)
    b = seg.sample_indices(0, vd.VideoReader(str(path)))
    np.testing.assert_array_equal(a, b)


def test_start_and_end_sec_restrict_the_clip(clip_files):
    _, paths = clip_files
    df = pd.DataFrame({'videos': [str(paths[3])], 'start_sec': [0.4], 'end_sec': [0.8]})  # frames 10..19 at 25 fps
    clip, _, extras = VideoDataset(df, num_frames=5)[0]
    assert _index(clip) == [10, 12, 14, 16, 19] and extras['start_sec'] == 0.4
    bad = pd.DataFrame({'videos': [str(paths[3])], 'start_sec': [0.8], 'end_sec': [0.4]})
    with pytest.raises(ValueError, match='end_sec'):
        VideoDataset(bad, num_frames=2)[0]


def test_very_short_clip_is_padded_and_logged(tmp_path, caplog):
    path = index_coded_clip(tmp_path / 'short.mp4', 3)
    dense = VideoDataset(pd.DataFrame({'videos': [str(path)]}), num_frames=6,
                         options=VideoDataOptions(temporal_sampling='dense'))
    with caplog.at_level('WARNING', logger='deeptune.video'):
        clip, _ = dense[0]
    assert _index(clip) == [0, 1, 2, 2, 2, 2]
    assert 'frames are repeated' in caplog.text


def test_missing_and_unreadable_files_report_the_row(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        VideoDataset(pd.DataFrame({'videos': [str(tmp_path / 'nope.mp4')]}))
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'garbage' * 100)
    with pytest.raises(ValueError, match='Video row 0'):
        VideoDataset(pd.DataFrame({'videos': [str(bad)]}))[0]


def test_mixed_resolutions_resize_with_a_transform_and_fail_clearly_without(tmp_path):
    import torchvision
    paths = [index_coded_clip(tmp_path / 'a.mp4', 6, 32, 32), index_coded_clip(tmp_path / 'b.mp4', 6, 48, 32)]
    df = pd.DataFrame({'videos': [str(p) for p in paths]})
    small = torchvision.transforms.Compose([torchvision.transforms.Resize((16, 16)), torchvision.transforms.ToTensor()])
    clips, _ = next(iter(DataLoader(VideoDataset(df, small, 2), batch_size=2, collate_fn=collate_video_samples)))
    assert clips.shape == (2, 2, 3, 16, 16)
    with pytest.raises(ValueError, match='different shapes'):
        next(iter(DataLoader(VideoDataset(df, None, 2), batch_size=2, collate_fn=collate_video_samples)))


def test_augmented_training_clip_keeps_its_shape(clip_files):
    _, paths = clip_files
    opts = VideoDataOptions(video_augment=True, video_speed_perturb=0.2)
    dataset = VideoDataset(pd.DataFrame({'videos': [str(paths[0])], 'labels': [1]}), num_frames=4, train=True, options=opts)
    clip, label = dataset[0]
    assert clip.shape == (4, 3, 32, 32) and dataset.augment is not None and not dataset.augment.config.flip
    assert VideoDataset(pd.DataFrame({'videos': [str(paths[0])]}), num_frames=4, options=opts).augment is None  # eval


def test_flip_requires_augment():
    with pytest.raises(ValueError, match='video_flip'):
        VideoDataOptions(video_flip=True)


def test_options_from_args_ignore_missing_and_non_plain_values():
    from unittest.mock import MagicMock
    assert video_options_from_args(MagicMock()) == VideoDataOptions()
    assert video_options_from_args({'test_clips': 3, 'num_workers': 0}).test_clips == 3
    assert video_options_from_args({}, temporal_sampling='dense').temporal_sampling == 'dense'


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def test_worker_defaults():
    from options import resolve_num_workers
    assert resolve_num_workers('auto', 10_000, device='cpu', platform='win32', cpu=16) == 0
    assert resolve_num_workers('auto', 100, device='cuda', platform='win32', cpu=16) == 0
    assert resolve_num_workers('auto', 5_000, device='cuda', platform='win32', cpu=16) == 4
    assert resolve_num_workers('auto', 100, device='cuda', platform='linux', cpu=16) == 8
    assert resolve_num_workers(3, 10, device='cpu') == 3
    with pytest.raises(ValueError):
        resolve_num_workers(-1)


def test_loader_kwargs_pin_memory_only_on_cuda_and_prepare_workers(clip_files):
    _, paths = clip_files
    df = pd.DataFrame({'videos': [p.read_bytes() for p in paths]})
    dataset = ParquetVideoDataset(df, num_frames=2)
    kwargs = video_loader_kwargs(dataset, 2, False, num_workers=1)
    assert kwargs['num_workers'] == 1 and kwargs['persistent_workers'] is True
    assert kwargs['pin_memory'] is (torch.cuda.is_available())
    assert isinstance(dataset._store, vd.ArrowBytesStore)
    assert video_loader_kwargs(dataset, 2, False)['num_workers'] == 0  # auto on CPU


# ---------------------------------------------------------------------------
# Multi-clip averaging
# ---------------------------------------------------------------------------
class _Toy(torch.nn.Module):
    def forward(self, x, extract_embed=False):
        mean = x.mean(dim=(1, 2, 3, 4))
        if extract_embed:
            return torch.stack([mean, -mean], 1)
        return torch.stack([mean * 4, torch.zeros_like(mean)], 1)


def test_multi_clip_model_averages_probabilities():
    from src.video.multi_clip import MultiClipModel, clip_disagreement, multi_clip_scores
    x = torch.stack([torch.zeros(2, 3, 2, 2), torch.ones(2, 3, 2, 2)]).unsqueeze(0)  # [1, 2 clips, T, C, H, W]
    single = [torch.softmax(_Toy()(x[:, i]), 1) for i in range(2)]
    expected = (single[0] + single[1]) / 2
    out = MultiClipModel(_Toy(), 'cls')(x)
    assert torch.allclose(torch.softmax(out, 1), expected, atol=1e-6)
    _, per_clip = multi_clip_scores(_Toy(), x, 'cls')
    assert clip_disagreement(per_clip, 'cls').item() > 0
    same = torch.stack([torch.ones(2, 3, 2, 2)] * 3).unsqueeze(0)
    assert clip_disagreement(multi_clip_scores(_Toy(), same, 'cls')[1], 'cls').item() == pytest.approx(0)
    emb = MultiClipModel(_Toy(), 'cls')(x, extract_embed=True)
    assert torch.allclose(emb, torch.tensor([[0.5, -0.5]]))
    assert torch.equal(MultiClipModel(_Toy(), 'cls')(x[:, 0]), _Toy()(x[:, 0]))  # 5-D passes through


def test_multi_clip_model_loads_plain_checkpoints():
    from src.video.multi_clip import MultiClipModel
    inner = torch.nn.Linear(2, 2)
    wrapped = MultiClipModel(torch.nn.Linear(2, 2))
    wrapped.load_state_dict(inner.state_dict())
    assert set(wrapped.state_dict()) == set(inner.state_dict())


# ---------------------------------------------------------------------------
# Group-aware splits
# ---------------------------------------------------------------------------
def test_group_column_never_spans_two_splits(tmp_path):
    from handlers.split_dataset import split_dataset
    rng = np.random.default_rng(0)
    for trial in range(12):
        n_groups = int(rng.integers(6, 30))
        sizes = rng.integers(1, 6, n_groups)
        groups = np.repeat([f'g{g}' for g in range(n_groups)], sizes)
        labels = np.repeat(rng.integers(0, 2, n_groups), sizes)
        if min(np.bincount(labels, minlength=2)) < 3:
            continue
        source = tmp_path / f'd{trial}.parquet'
        pd.DataFrame({'videos': [f'{i}.mp4' for i in range(len(groups))], 'labels': labels, 'group': groups}).to_parquet(source)
        out = tmp_path / f'out{trial}'
        try:
            parts = split_dataset(.7, .1, .2, source, out, True, False, 'labels', 'video')
        except ValueError as exc:
            assert 'empty' in str(exc) or 'groups' in str(exc)
            continue
        frames = [pd.read_parquet(p) for p in parts]
        seen = [set(f['group']) for f in frames]
        assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
        assert sum(map(len, frames)) == len(groups) and all(len(f) for f in frames)
        summary = json.loads(next(out.glob('data_splits_*/split_summary.json')).read_text())
        assert summary['strategy'] == 'group' and summary['grouper'] == 'group'


def test_group_split_with_too_few_groups_errors_clearly(tmp_path):
    from handlers.split_dataset import check_group_split
    part = pd.DataFrame({'group': ['a'], 'labels': [0]})
    with pytest.raises(ValueError, match='validation split empty'):
        check_group_split({'train': part, 'validation': part.iloc[:0], 'test': part}, 'group')
    with pytest.raises(AssertionError, match='appears in both'):
        check_group_split({'train': part, 'validation': part, 'test': part}, 'group')


def test_group_column_is_only_automatic_for_video():
    from handlers.split_dataset import resolve_grouper
    assert resolve_grouper(['videos', 'labels', 'group'], None, 'video') == 'group'
    assert resolve_grouper(['videos', 'labels', 'group'], 'patient', 'video') == 'patient'
    assert resolve_grouper(['images', 'labels', 'group'], None, 'images') is None


# ---------------------------------------------------------------------------
# CLI options are recorded and reach inference
# ---------------------------------------------------------------------------
def test_new_flags_are_saved_and_carried_to_the_test_page(tmp_path):
    from cli import DeepTuneVisionOptions
    from desktop.config import InferenceConfig, RunResults, build_infer_argv, infer_config_from_run
    from utils import RunType
    opts = DeepTuneVisionOptions(RunType.ONECALL, ['--out', str(tmp_path), '--modality', 'video', '--df', 'x.parquet',
                                                   '--temporal_sampling', 'segment', '--test_clips', '3',
                                                   '--clip_duration_sec', '2.5', '--num_workers', '0',
                                                   '--video_augment', '--video_speed_perturb', '0.1'])
    opts.save_args(tmp_path)
    saved = json.loads((tmp_path / 'cli_arguments.json').read_text())
    for key, value in {'temporal_sampling': 'segment', 'test_clips': 3, 'clip_duration_sec': 2.5, 'num_workers': 0,
                       'video_augment': True, 'video_flip': False, 'video_speed_perturb': 0.1,
                       'video_storage': 'paths', 'video_backend': 'auto'}.items():
        assert saved[key] == value, key
    results = RunResults(run_dir=tmp_path, checkpoint_path=tmp_path / 'model_weights.pth', cli_arguments=saved)
    cfg = infer_config_from_run(results, tmp_path / 'data', tmp_path / 'out')
    argv = build_infer_argv(cfg)
    for flag, value in [('--temporal_sampling', 'segment'), ('--test_clips', '3'), ('--clip_duration_sec', '2.5'),
                        ('--num_workers', '0')]:
        assert argv[argv.index(flag) + 1] == value
    assert '--video_augment' not in argv  # training-only
    import infer
    parsed = infer.build_parser().parse_args(argv)
    assert parsed.test_clips == 3 and parsed.temporal_sampling == 'segment' and parsed.num_workers == 0


def test_old_runs_without_new_options_keep_the_original_infer_argv(tmp_path):
    from desktop.config import RunResults, build_infer_argv, infer_config_from_run
    old = {'modality': 'video', 'model_version': 'r3d_18', 'num_classes': 2, 'num_frames': 8, 'pooling': 'mean'}
    results = RunResults(run_dir=tmp_path, checkpoint_path=tmp_path / 'w.pth', cli_arguments=old)
    argv = build_infer_argv(infer_config_from_run(results, tmp_path / 'data', tmp_path / 'out'))
    assert not any(a in argv for a in ('--temporal_sampling', '--test_clips', '--num_workers', '--video_backend'))

"""Heads on cached features (P-102): model contract, learning, the feature dataset, end-to-end training."""
import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip('av')

import extract_features as ef
import train_feature_head as tfh
from datasets.feature_dataset import CachedFeatureDataset, snippet_indices
from datasets.video_io import VideoReader
from datasets.video_synth import translating_clip
from src.video.feature_heads import FEATURE_HEADS, AttentiveProbe, LinearProbe, build_feature_head
from utils import build_video_model

SMALL = dict(model_version='dummy_video3d', motion_backbone='dummy_video3d', feature_weights='none', snippet_frames=4,
             batch_size=2, feature_input_size=32)


@pytest.fixture(scope='module')
def videos(tmp_path_factory):
    folder = tmp_path_factory.mktemp('clips')
    paths = [translating_clip(folder / f'v{i}.mp4', 16 + 4 * i, 2, 0, size=32, seed=i) for i in range(4)]
    return pd.DataFrame({'videos': [str(p) for p in paths], 'labels': [0, 1, 0, 1], 'video_id': [p.stem for p in paths]})


# -- the heads -----------------------------------------------------------------------
@pytest.mark.parametrize('shape', [(3, 16), (3, 2, 16), (3, 4, 2, 2, 2, 16), (3, 1, 1, 1, 1, 16)])
@pytest.mark.parametrize('name', sorted(FEATURE_HEADS))
def test_heads_follow_the_model_contract_on_any_grid(name, shape):
    head = build_feature_head(name, 16, 5)
    x = torch.rand(*shape)
    assert head(x).shape == (3, 5)
    assert head(x, extract_embed=True).ndim == 2 and len(head(x, extract_embed=True)) == 3


@pytest.mark.parametrize('name', sorted(FEATURE_HEADS))
def test_regression_heads_return_output_dim(name):
    assert build_feature_head(name, 16, 5, 'reg')(torch.rand(2, 3, 16)).shape == (2, 1)
    assert build_feature_head(name, 16, 5, 'reg', output_dim=3)(torch.rand(2, 3, 16)).shape == (2, 3)


def test_unknown_head_and_bad_options_are_refused():
    with pytest.raises(ValueError, match='Unknown feature head'):
        build_feature_head('magic', 8, 2)
    with pytest.raises(ValueError, match='1 to 4'):
        AttentiveProbe(8, 2, depth=5)
    with pytest.raises(ValueError, match='divisible'):
        AttentiveProbe(8, 2, width=30, num_heads=4)
    with pytest.raises(ValueError):
        LinearProbe(8, 2, task_type='magic')


@pytest.mark.parametrize('depth', [1, 2, 4])
def test_all_attentive_parameters_get_gradients(depth):
    head = AttentiveProbe(8, 3, width=16, depth=depth, num_heads=2)
    head(torch.rand(4, 2, 2, 2, 2, 8)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def _fit(head, x, y, steps=120, lr=3e-3):
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(head(x), y).backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        return float((head(x).argmax(dim=1) == y).float().mean())


def test_attentive_probe_finds_a_signal_that_mean_pooling_destroys():
    """The label is which of two spatial cells holds a marker; the mean over tokens is identical for both
    classes. This is the reason to cache token grids at all."""
    torch.manual_seed(0)
    n, D = 48, 8
    x = torch.randn(n, 2, 2, D) * 0.1
    y = torch.arange(n) % 2
    marker = torch.zeros(D)
    marker[0] = 3.0
    x[y == 0, 0, 0] += marker  # class 0: top-left
    x[y == 1, 1, 1] += marker  # class 1: bottom-right
    attentive = _fit(AttentiveProbe(D, 2, width=16, num_heads=2), x, y)
    torch.manual_seed(0)
    linear = _fit(LinearProbe(D, 2), x, y)
    assert attentive >= 0.95, attentive
    assert linear <= 0.8, linear


def test_position_feature_makes_tokens_distinguishable():
    head = AttentiveProbe(8, 2, width=16, num_heads=2)
    features = head._position_features((2, 2), torch.zeros(1))
    assert features.shape[0] == 4
    assert len({tuple(row.tolist()) for row in features}) == 4


# -- the dataset ---------------------------------------------------------------------
def test_snippet_indices_cover_evenly_and_keep_the_shape():
    assert snippet_indices(5, 3, False).tolist() == [0, 2, 4]
    assert snippet_indices(2, 4, False).tolist() == [0, 0, 1, 1]
    assert snippet_indices(1, 4, False).tolist() == [0, 0, 0, 0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        got = snippet_indices(20, 4, True, rng)
        assert ((got >= [0, 5, 10, 15]) & (got < [5, 10, 15, 20])).all()
    assert snippet_indices(2, 5, True, rng).max() <= 1


def test_vector_features_load_without_touching_video_files(videos, tmp_path):
    ef.extract_features(videos, tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    with mock.patch.object(VideoReader, '__init__', side_effect=AssertionError('opened a video')):
        data = CachedFeatureDataset(videos, tmp_path / 'f', num_snippets=3)
        x, y = data[1]
    assert x.dtype == torch.float32 and y.dtype == torch.long and y.item() == 1
    assert x.shape == (3, data.feature_dim())
    assert data.feature_dim() == 2 * 512  # two r3d_18 streams


def test_eval_samples_are_deterministic_and_training_ones_vary(videos, tmp_path):
    ef.extract_features(videos, tmp_path / 'f', **SMALL)
    fixed = CachedFeatureDataset(videos, tmp_path / 'f', num_snippets=2)
    assert torch.equal(fixed[3][0], fixed[3][0])
    moving = CachedFeatureDataset(videos, tmp_path / 'f', num_snippets=2, train=True)
    assert len({moving[3][0].numpy().tobytes() for _ in range(20)}) > 1


def test_stacks_into_batches_for_videos_of_different_lengths(videos, tmp_path):
    ef.extract_features(videos, tmp_path / 'f', cache_tokens='1x2x2', **SMALL)
    data = CachedFeatureDataset(videos, tmp_path / 'f', num_snippets=3)
    x, y = next(iter(DataLoader(data, batch_size=4)))
    assert x.shape == (4, 3, 1, 2, 2, 512) and y.tolist() == [0, 1, 0, 1]


def test_missing_features_and_labels_are_clear_errors(videos, tmp_path):
    ef.extract_features(videos.iloc[:2], tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='feature files are missing'):
        CachedFeatureDataset(videos, tmp_path / 'f')
    with pytest.raises(FileNotFoundError, match='extract_features'):
        CachedFeatureDataset(videos, tmp_path / 'empty')
    named = videos.iloc[:2].assign(labels=['walk', 'run'])
    with pytest.raises(ValueError, match='numeric'):
        CachedFeatureDataset(named, tmp_path / 'f')
    with pytest.raises(ValueError, match='nope'):
        CachedFeatureDataset(videos.iloc[:2], tmp_path / 'f', target='nope')


def test_splits_find_the_features_of_the_full_manifest_by_path(videos, tmp_path):
    """Without a video_id column the ids come from each row alone, so a split matches the full manifest."""
    plain = videos.drop(columns=['video_id'])
    ef.extract_features(plain, tmp_path / 'f', **SMALL)
    assert len(CachedFeatureDataset(plain.iloc[[3, 1]], tmp_path / 'f', num_snippets=1)) == 2


# -- end to end ------------------------------------------------------------------------
@pytest.mark.parametrize('head,tokens', [('attentive_probe', '1x2x2'), ('linear_probe', 'none')])
def test_training_end_to_end_through_the_shared_trainer(videos, tmp_path, head, tokens):
    manifest = tmp_path / 'clips.csv'
    videos.to_csv(manifest, index=False)
    ef.extract_features(manifest, tmp_path / 'f', cache_tokens=tokens, **SMALL)
    with mock.patch.object(VideoReader, '__init__', side_effect=AssertionError('training decoded a video')):
        assert tfh.main(['--features', str(tmp_path / 'f'), '--train_df', str(manifest), '--val_df', str(manifest),
                         '--test_df', str(manifest), '--out', str(tmp_path / 'probe'), '--head', head, '--num_epochs', '1',
                         '--batch_size', '2', '--num_snippets', '2', '--probe_width', '16', '--probe_heads', '2', '--fixed-seed']) == 0
    saved = json.loads((tmp_path / 'probe' / 'cli_arguments.json').read_text())
    assert saved['head'] == head and saved['input_dim'] == 512
    metrics = json.loads((tmp_path / 'probe' / 'metrics.json').read_text())
    assert metrics['n'] == 4 and 0 <= metrics['accuracy'] <= 1
    assert (tmp_path / 'probe' / 'training_log.csv').is_file()
    state = torch.load(tmp_path / 'probe' / 'model_weights.pth', map_location='cpu', weights_only=True)
    rebuilt = build_feature_head(head, saved['input_dim'], saved['num_classes'], saved['mode'],
                                 **(dict(width=16, num_heads=2) if head == 'attentive_probe' else {}))
    rebuilt.load_state_dict(state)


def test_build_video_model_is_untouched_by_the_feature_heads():
    """Heads are registered in their own module; existing model names keep building as before."""
    assert 'attentive_probe' in FEATURE_HEADS
    model = build_video_model('dummy_video3d', 2, 1, 8, True, 'cls')
    assert model(torch.rand(1, 8, 3, 32, 32)).shape == (1, 2)

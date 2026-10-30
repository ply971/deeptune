"""Heads on cached features (P-102): model contract, learning, the feature dataset, end-to-end training."""
import json
from pathlib import Path
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
from datasets.snippet_features import FeatureStore, same_path
from datasets.video_datasets import open_video_dataset
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


def _order_task(n, seed, grid, D=8):
    """Label 0: a marker in the first token. Label 1: the same tokens in reverse order. Each pair has the same tokens
    and the same mean, so a model blind to position scores exactly chance on any balanced set of pairs."""
    generator = torch.Generator().manual_seed(seed)
    first = torch.randn(n // 2, *grid, D, generator=generator) * 0.1
    first[(slice(None), *[0] * len(grid), 0)] += 3.0
    reversed_ = first.flip(dims=tuple(range(1, len(grid) + 1)))
    return torch.stack((first, reversed_), dim=1).flatten(0, 1), torch.arange(n) % 2


@pytest.mark.parametrize('grid', [(2, 2), (4,)], ids=['spatial', 'temporal'])
def test_attentive_probe_needs_position_to_generalise_where_mean_pooling_fails(grid):
    """Scored on unseen noise: on training data a head can memorise noise, and position encoding off reached 1.0 there."""
    x, y = _order_task(48, 11, grid)
    unseen, targets = _order_task(400, 29, grid)

    def heldout(model):
        _fit(model, x, y)
        with torch.no_grad():
            return float((model(unseen).argmax(dim=1) == targets).float().mean())

    torch.manual_seed(0)
    with_position = heldout(AttentiveProbe(8, 2, width=16, num_heads=2))
    torch.manual_seed(0)
    without_position = heldout(AttentiveProbe(8, 2, width=16, num_heads=2, use_position=False))
    torch.manual_seed(0)
    linear = heldout(LinearProbe(8, 2))
    assert with_position >= 0.95, with_position
    assert without_position <= 0.55, without_position
    assert linear <= 0.55, linear


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
    train_manifest, val_manifest, test_manifest = [tmp_path / f'{name}.csv' for name in ('train', 'val', 'test')]
    videos.iloc[:2].to_csv(train_manifest, index=False)
    videos.iloc[2:3].to_csv(val_manifest, index=False)
    videos.iloc[3:].to_csv(test_manifest, index=False)
    with mock.patch.object(VideoReader, '__init__', side_effect=AssertionError('training decoded a video')):
        assert tfh.main(['--features', str(tmp_path / 'f'), '--train_df', str(train_manifest), '--val_df', str(val_manifest),
                         '--test_df', str(test_manifest), '--out', str(tmp_path / 'probe'), '--head', head, '--num_epochs', '1',
                         '--batch_size', '2', '--num_snippets', '2', '--probe_width', '16', '--probe_heads', '2', '--fixed-seed']) == 0
    saved = json.loads((tmp_path / 'probe' / 'cli_arguments.json').read_text())
    assert saved['head'] == head and saved['input_dim'] == 512
    metrics = json.loads((tmp_path / 'probe' / 'metrics.json').read_text())
    assert metrics['n'] == 1 and 0 <= metrics['accuracy'] <= 1
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


@pytest.mark.parametrize('format', ['csv', 'parquet', 'dataframe'])
def test_feature_only_ids_need_no_video_column(videos, tmp_path, format):
    ef.extract_features(videos, tmp_path / 'f', **SMALL)
    table = videos[['video_id', 'labels']]
    source = table
    if format != 'dataframe':
        source = tmp_path / f'rows.{format}'
        if format == 'csv':
            table.to_csv(source, index=False)
        else:
            table.to_parquet(source, index=False)
    with mock.patch.object(VideoReader, '__init__', side_effect=AssertionError('opened video')):
        dataset = CachedFeatureDataset(source, tmp_path / 'f')
        assert dataset[1][1].item() == 1
        assert dataset.video_ids == videos.video_id.tolist()


def test_cached_features_still_load_after_source_video_is_removed(videos, tmp_path):
    import shutil
    path = tmp_path / 'temporary.mp4'
    shutil.copyfile(videos.videos.iloc[0], path)
    table = pd.DataFrame({'videos': [str(path)], 'labels': [0]})
    ef.extract_features(table, tmp_path / 'f', **SMALL)
    path.unlink()
    dataset = CachedFeatureDataset(table, tmp_path / 'f')
    assert dataset[0][0].shape == (8, 512)


def test_feature_only_rows_without_ids_are_refused(tmp_path, videos):
    ef.extract_features(videos, tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='video_id for every row'):
        CachedFeatureDataset(videos[['labels']], tmp_path / 'f')


def test_bytes_parquet_is_still_memory_mapped_rather_than_read_whole(videos, tmp_path):
    frame = pd.DataFrame({'videos': [Path(p).read_bytes() for p in videos.videos], 'labels': videos.labels})
    frame.to_parquet(tmp_path / 'bytes.parquet')
    ef.extract_features(tmp_path / 'bytes.parquet', tmp_path / 'f', **SMALL)
    # open_video_dataset is the path that spills large bytes files to a memory-mapped store.
    with mock.patch('datasets.feature_dataset.open_video_dataset', wraps=open_video_dataset) as opened:
        data = CachedFeatureDataset(tmp_path / 'bytes.parquet', tmp_path / 'f', num_snippets=1)
    opened.assert_called_once()
    assert len(data) == 4 and data[3][1].item() == 1


def test_rows_asking_for_another_trim_than_was_extracted_are_refused(videos, tmp_path):
    rows = videos.iloc[[3]].assign(start_sec=0.0, end_sec=0.4)
    ef.extract_features(rows, tmp_path / 'f', **SMALL)
    assert len(CachedFeatureDataset(rows, tmp_path / 'f', num_snippets=1)) == 1
    with pytest.raises(ValueError, match='end_sec'):
        CachedFeatureDataset(rows.assign(end_sec=0.6), tmp_path / 'f', num_snippets=1)


@pytest.mark.parametrize('leak,message', [('id', 'share video_ids'), ('trim', 'share sources'), ('group', 'share groups')])
def test_training_refuses_leaking_splits_before_it_writes_anything(videos, tmp_path, leak, message):
    rows = videos.assign(group=['a', 'b', 'c', 'd'])
    train, test = rows.iloc[:2], rows.iloc[2:]
    if leak == 'id':
        test = pd.concat([test, train.iloc[[0]]])
    elif leak == 'group':
        test = test.assign(group=['a', 'd'])
    else:  # another stretch of a training video, under its own id
        train = train.assign(start_sec=0.0, end_sec=0.2)
        test = pd.concat([test, train.iloc[[0]].assign(video_id='other', start_sec=0.2, end_sec=0.4, group='e')])
    full = pd.concat([train, test]).drop_duplicates('video_id')
    ef.extract_features(full, tmp_path / 'f', **SMALL)
    paths = {name: tmp_path / f'{name}.csv' for name in ('train', 'val', 'test')}
    train.iloc[:1].to_csv(paths['train'], index=False)
    train.iloc[1:].to_csv(paths['val'], index=False)
    test.to_csv(paths['test'], index=False)
    with pytest.raises(ValueError, match=f'Split leakage.*{message}'):
        tfh.main(['--features', str(tmp_path / 'f'), '--train_df', str(paths['train']), '--val_df', str(paths['val']),
                  '--test_df', str(paths['test']), '--out', str(tmp_path / 'probe'), '--head', 'linear_probe', '--num_epochs', '1'])
    assert not (tmp_path / 'probe').exists()


def test_two_trims_of_one_bytes_clip_across_splits_are_refused(videos, tmp_path):
    """Bytes have no path to compare, so the sidecar fingerprint identifies the source."""
    clip = Path(videos.videos.iloc[3]).read_bytes()
    other = Path(videos.videos.iloc[2]).read_bytes()
    rows = pd.DataFrame({'videos': [clip, clip, other], 'labels': [0, 1, 0], 'video_id': ['t1', 't2', 'o'],
                         'start_sec': [0.0, 0.3, 0.0], 'end_sec': [0.3, 0.6, 0.3]})
    ef.extract_features(rows, tmp_path / 'f', **SMALL)
    paths = {}
    for name, picked in (('train', [0]), ('val', [2]), ('test', [1])):
        paths[name] = tmp_path / f'{name}.parquet'
        rows.iloc[picked].to_parquet(paths[name])
    with pytest.raises(ValueError, match='Split leakage.*share sources'):
        tfh.main(['--features', str(tmp_path / 'f'), '--train_df', str(paths['train']), '--val_df', str(paths['val']),
                  '--test_df', str(paths['test']), '--out', str(tmp_path / 'probe'), '--head', 'linear_probe', '--num_epochs', '1'])
    audit = tfh.audit_splits({'train': CachedFeatureDataset(paths['train'], tmp_path / 'f', num_snippets=1),
                              'val': CachedFeatureDataset(paths['val'], tmp_path / 'f', num_snippets=1)})
    assert audit['video_and_source_overlap'] is False  # a different clip is not flagged


def test_split_audit_treats_ids_differing_only_by_case_as_one():
    from types import SimpleNamespace
    left, right = (SimpleNamespace(video_ids=[vid], sources=[vid + '.mp4'], groups=None) for vid in ('Clip', 'clip'))
    with pytest.raises(ValueError, match='share video_ids'):
        tfh.audit_splits({'train': left, 'test': right})


def _rewrite_motion_version(root, version):
    path = Path(root) / 'features.json'
    info = json.loads(path.read_text())
    config = info['streams']['motion']['config']
    config.pop('preprocessing_version', None)
    if version is not None:
        config['preprocessing_version'] = version
    path.write_text(json.dumps(info))


@pytest.mark.parametrize('version', [None, 2])
def test_older_motion_features_are_refused_at_load(videos, tmp_path, version):
    ef.extract_features(videos, tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    assert len(CachedFeatureDataset(videos, tmp_path / 'f', num_snippets=1)) == 4
    _rewrite_motion_version(tmp_path / 'f', version)
    with pytest.raises(ValueError, match='motion features from preprocessing version .*extract_features.py again'):
        CachedFeatureDataset(videos, tmp_path / 'f', num_snippets=1)
    with pytest.raises(ValueError, match='extract_features.py again'):
        CachedFeatureDataset(videos, tmp_path / 'f', streams=['motion'], num_snippets=1)
    assert len(CachedFeatureDataset(videos, tmp_path / 'f', streams=['rgb'], num_snippets=1)) == 4  # rgb is unaffected


def test_chance_row_uses_the_training_labels(tmp_path):
    from torch.utils.data import TensorDataset
    data = TensorDataset(torch.tensor([[3., 0.], [0., 3.], [0., 3.], [0., 3.]]), torch.tensor([0, 1, 1, 1]))
    data.video_ids = list('abcd')
    result = tfh.evaluate_head(torch.nn.Identity(), DataLoader(data, batch_size=2), 'cls', train_labels=[0, 0, 0, 1])
    assert result['accuracy'] == 1.0
    assert result['chance']['predicts'] == 'training majority class 0'  # not the majority of the test rows
    assert result['chance']['accuracy'] == pytest.approx(0.25)
    assert result['chance']['balanced_accuracy'] == pytest.approx(0.5)
    assert result['chance']['macro_f1'] == pytest.approx(0.2)
    regression = TensorDataset(torch.tensor([[1.], [3.]]), torch.tensor([1., 3.]))
    regression.video_ids = ['a', 'b']
    chance = tfh.evaluate_head(torch.nn.Identity(), DataLoader(regression, batch_size=2), 'reg', train_labels=[0., 4.])['chance']
    assert chance['predicts'] == 'training mean 2' and chance['mse'] == pytest.approx(1.0) and chance['mae'] == pytest.approx(1.0)


def test_the_probe_run_records_what_its_features_were_made_with(videos, tmp_path, capsys):
    ef.extract_features(videos, tmp_path / 'f', streams=('rgb', 'motion'), **SMALL)
    paths = {name: tmp_path / f'{name}.csv' for name in ('train', 'val', 'test')}
    videos.iloc[:2].to_csv(paths['train'], index=False)
    videos.iloc[2:3].to_csv(paths['val'], index=False)
    videos.iloc[3:].to_csv(paths['test'], index=False)
    assert tfh.main(['--features', str(tmp_path / 'f'), '--train_df', str(paths['train']), '--val_df', str(paths['val']),
                     '--test_df', str(paths['test']), '--out', str(tmp_path / 'probe'), '--head', 'linear_probe', '--num_epochs', '1',
                     '--batch_size', '2', '--num_snippets', '2', '--fixed-seed']) == 0
    saved = json.loads((tmp_path / 'probe' / 'cli_arguments.json').read_text())
    streams = json.loads((tmp_path / 'f' / 'features.json').read_text())['streams']
    assert saved['seed'] == 42
    assert saved['feature_config_hash'] == {name: streams[name]['config_hash'] for name in ('rgb', 'motion')}
    assert saved['motion_weights_trained_on_motion'] is False
    assert 'not trained on motion' in capsys.readouterr().out
    assert 'chance' in json.loads((tmp_path / 'probe' / 'metrics.json').read_text())


def test_a_missing_group_column_is_reported_not_passed_off_as_checked(capsys):
    from types import SimpleNamespace
    left = SimpleNamespace(video_ids=['a'], sources=['a.mp4'], groups=None)
    right = SimpleNamespace(video_ids=['b'], sources=['b.mp4'], groups=['person-b'])
    assert tfh.audit_splits({'train': left, 'test': right})['group_check'].startswith('incomplete')
    assert "'group' column" in capsys.readouterr().out


@pytest.mark.parametrize('field', ['video_ids', 'sources', 'groups'])
def test_split_audit_rejects_overlap(field):
    from types import SimpleNamespace
    left = SimpleNamespace(video_ids=['a'], sources=['a.mp4'], groups=['person-a'])
    right = SimpleNamespace(video_ids=['b'], sources=['b.mp4'], groups=['person-b'])
    setattr(right, field, getattr(left, field))
    with pytest.raises(ValueError, match=f'Split leakage.*{field}'):
        tfh.audit_splits({'train': left, 'test': right})


def test_evaluate_head_writes_aligned_predictions_and_metrics(tmp_path):
    from torch.utils.data import TensorDataset
    data = TensorDataset(torch.tensor([[3., 0.], [0., 3.], [2., 1.]]), torch.tensor([0, 1, 1]))
    data.video_ids = ['first', 'second', 'third']
    result = tfh.evaluate_head(torch.nn.Identity(), DataLoader(data, batch_size=2), 'cls', tmp_path / 'p.csv')
    rows = pd.read_csv(tmp_path / 'p.csv')
    assert rows.video_id.tolist() == data.video_ids
    assert rows.target.tolist() == [0, 1, 1]
    assert rows.prediction.tolist() == [0, 1, 0]
    assert np.allclose(rows[['probability_0', 'probability_1']].sum(axis=1), 1)
    assert result['accuracy'] == pytest.approx(2 / 3)
    assert result['balanced_accuracy'] == pytest.approx(.75)
    assert result['macro_f1'] == pytest.approx(2 / 3)
    assert result['confusion_matrix'] == [[1, 0], [1, 1]]


def test_evaluate_regression_has_per_video_errors(tmp_path):
    from torch.utils.data import TensorDataset
    data = TensorDataset(torch.tensor([[1.], [4.], [5.]]), torch.tensor([1., 2., 3.]))
    data.video_ids = ['a', 'b', 'c']
    result = tfh.evaluate_head(torch.nn.Identity(), DataLoader(data, batch_size=2), 'reg', tmp_path / 'p.csv')
    assert result['mae'] == pytest.approx(4 / 3)
    assert result['mse'] == pytest.approx(8 / 3)
    assert result['r2'] == pytest.approx(-3.)
    assert pd.read_csv(tmp_path / 'p.csv').prediction.tolist() == [1., 4., 5.]


def test_shared_trainer_learns_cached_signal_on_disjoint_test_rows(tmp_path):
    from datasets.snippet_features import FeatureStore
    store = FeatureStore(tmp_path / 'features')
    store.register_stream('rgb', {'fixture': 'separable'}, 2, None)
    rng = np.random.default_rng(13)
    paths = {}
    for split, count in [('train', 24), ('val', 8), ('test', 16)]:
        rows = []
        for i in range(count):
            vid, label = f'{split}-{i}', i % 2
            values = rng.normal(0, .05, (2, 2))
            values[:, 0] += 2 * label - 1
            store.write('rgb', vid, values.astype(np.float16), {'video_id': vid})
            rows.append({'video_id': vid, 'labels': label, 'group': vid})
        paths[split] = tmp_path / f'{split}.csv'
        pd.DataFrame(rows).to_csv(paths[split], index=False)
    # A non-learning implementation stays at exactly 50% on balanced held-out labels.
    model = LinearProbe(2, 2)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    with mock.patch.object(tfh, 'audit_splits', wraps=tfh.audit_splits) as audit:
        with mock.patch('src.video.feature_heads.build_feature_head', return_value=model):
            tfh.main(['--features', str(store.root), '--train_df', str(paths['train']),
                      '--val_df', str(paths['val']), '--test_df', str(paths['test']),
                      '--out', str(tmp_path / 'learned'), '--head', 'linear_probe', '--num_epochs', '8',
                      '--batch_size', '24', '--num_snippets', '2', '--learning_rate', '.05', '--fixed-seed'])
        audit.assert_called_once()
    metrics = json.loads((tmp_path / 'learned' / 'metrics.json').read_text())
    assert metrics['accuracy'] >= .95
    assert model.head.weight.abs().sum() > 0
    predictions = pd.read_csv(tmp_path / 'learned' / 'predictions.csv')
    assert predictions.video_id.tolist() == [f'test-{i}' for i in range(16)]
    assert predictions.target.tolist() == [i % 2 for i in range(16)]
    assert (predictions.prediction == predictions.target).mean() == metrics['accuracy']
    assert json.loads((tmp_path / 'learned' / 'split_audit.json').read_text())['group_check'] == 'checked'


# -- third fix round (V-043) ---------------------------------------------------------
def _row(path, video_id, label):
    return pd.DataFrame({'videos': [str(path)], 'video_id': [video_id], 'labels': [label]})


def test_row_naming_another_video_than_its_features_is_refused(videos, tmp_path):
    first, second = videos['videos'].iloc[:2]
    ef.extract_features(_row(first, 'y', 0), tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='features were extracted from'):
        CachedFeatureDataset(_row(second, 'y', 1), tmp_path / 'f', num_snippets=1)
    assert len(CachedFeatureDataset(_row(first, 'y', 1), tmp_path / 'f', num_snippets=1)) == 1  # the normal case
    assert len(CachedFeatureDataset(pd.DataFrame({'video_id': ['y'], 'labels': [1]}), tmp_path / 'f', num_snippets=1)) == 1


def test_row_may_spell_the_path_of_its_features_differently(videos, tmp_path):
    first = Path(videos['videos'].iloc[0])
    ef.extract_features(_row(first, 'y', 0), tmp_path / 'f', **SMALL)
    spelled = first.parent / 'elsewhere' / '..' / first.name
    assert len(CachedFeatureDataset(_row(spelled, 'y', 0), tmp_path / 'f', num_snippets=1)) == 1


def test_same_path_ignores_case_only_where_the_file_system_does(tmp_path):
    (tmp_path / 'probe').write_text('')
    ignores_case = (tmp_path / 'PROBE').exists()
    (tmp_path / 'clip.mp4').write_text('')
    assert same_path(tmp_path / 'clip.mp4', tmp_path / 'sub' / '..' / 'clip.mp4')
    assert same_path(tmp_path / 'clip.mp4', tmp_path / 'CLIP.mp4') is ignores_case
    assert not same_path(tmp_path / 'clip.mp4', tmp_path / 'other.mp4')
    assert same_path(tmp_path / 'gone.mp4', tmp_path / 'sub' / '..' / 'gone.mp4')  # the videos may have been removed
    assert not same_path(tmp_path / 'gone.mp4', tmp_path / 'gone2.mp4')


def test_dataset_refuses_streams_of_different_videos_when_built(videos, tmp_path):
    first, second = videos['videos'].iloc[:2]
    ef.extract_features(_row(first, 'x', 0), tmp_path / 'f', streams=('rgb',), **SMALL)
    ef.extract_features(_row(second, 'x', 0), tmp_path / 'f', streams=('motion',), **SMALL)
    id_only = pd.DataFrame({'video_id': ['x'], 'labels': [0]})
    with pytest.raises(ValueError, match='different videos'):
        CachedFeatureDataset(id_only, tmp_path / 'f', num_snippets=1)
    assert len(CachedFeatureDataset(id_only, tmp_path / 'f', streams=['motion'], num_snippets=1)) == 1


def _bytes_row(path, video_id, label):
    return pd.DataFrame({'videos': [Path(path).read_bytes()], 'video_id': [video_id], 'labels': [label]})


@pytest.fixture(params=['dataframe', 'parquet'])
def as_source(request, tmp_path):
    """A bytes table as given, or as a parquet file, which reaches the memory-mapped clip store."""
    def convert(frame, name):
        if request.param == 'dataframe':
            return frame
        frame.to_parquet(tmp_path / f'{name}.parquet')
        return tmp_path / f'{name}.parquet'
    return convert


def test_bytes_row_with_other_bytes_than_its_features_is_refused(videos, tmp_path, as_source):
    first, second = videos['videos'].iloc[:2]
    ef.extract_features(as_source(_bytes_row(first, 'z', 0), 'extracted'), tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='differ from the bytes its features were extracted from'):
        CachedFeatureDataset(as_source(_bytes_row(second, 'z', 1), 'other'), tmp_path / 'f', num_snippets=1)


def test_matching_bytes_rows_still_load(videos, tmp_path, as_source):
    first, second = videos['videos'].iloc[:2]
    rows = pd.concat([_bytes_row(first, 'a', 0), _bytes_row(second, 'b', 1)], ignore_index=True)
    ef.extract_features(as_source(rows, 'extracted'), tmp_path / 'f', **SMALL)
    data = CachedFeatureDataset(as_source(rows, 'again'), tmp_path / 'f', num_snippets=1)
    assert len(data) == 2 and data[1][1].item() == 1
    fingerprint = FeatureStore(tmp_path / 'f').read_sidecar('rgb', 'b')['source_fingerprint']
    assert data.sources == [FeatureStore(tmp_path / 'f').read_sidecar('rgb', 'a')['source_fingerprint'], fingerprint]


def test_path_row_against_features_from_bytes_is_refused(videos, tmp_path):
    first = videos['videos'].iloc[0]
    ef.extract_features(_bytes_row(first, 'z', 0), tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='extracted from clip bytes.*extract_features.py'):
        CachedFeatureDataset(_row(first, 'z', 0), tmp_path / 'f', num_snippets=1)  # the same clip: it cannot be told


def test_bytes_row_against_features_from_a_path_is_refused(videos, tmp_path):
    first = videos['videos'].iloc[0]
    ef.extract_features(_row(first, 'z', 0), tmp_path / 'f', **SMALL)
    with pytest.raises(ValueError, match='holds clip bytes.*extracted from the file.*extract_features.py'):
        CachedFeatureDataset(_bytes_row(first, 'z', 0), tmp_path / 'f', num_snippets=1)


@pytest.mark.parametrize('extracted_from', [_row, _bytes_row])
def test_id_only_row_stays_allowed_whatever_the_features_came_from(videos, tmp_path, extracted_from):
    ef.extract_features(extracted_from(videos['videos'].iloc[0], 'z', 0), tmp_path / 'f', **SMALL)
    id_only = pd.DataFrame({'video_id': ['z'], 'labels': [1]})
    assert len(CachedFeatureDataset(id_only, tmp_path / 'f', num_snippets=1)) == 1


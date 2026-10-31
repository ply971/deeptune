"""The df-analyze export: file contents, PCA on the training rows only, ids out of the features, split audit."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from handlers import dfanalyze_export as dx
from handlers import get_intersection as gi


def _part(n, seed, prefix, groups=True, std=False, motion=False, dim=6):
    rng = np.random.default_rng(seed)
    return dx.SplitBlocks(target=rng.integers(0, 2, n), video_ids=[f'{prefix}{i}' for i in range(n)],
                          embedding=rng.normal(size=(n, dim)).astype(np.float32),
                          embedding_std=rng.random((n, dim)).astype(np.float32) if std else None,
                          motion=pd.DataFrame({name: rng.random(n) for name in dx.MOTION_STATISTICS}) if motion else None,
                          groups=[f'{prefix}g{i // 2}' for i in range(n)] if groups else None)


def _parts(**kw):
    return {'train': _part(20, 0, 'tr', **kw), 'val': _part(6, 1, 'va', **kw), 'test': _part(10, 2, 'te', **kw)}


def test_without_groups_the_csv_holds_features_and_target_only(tmp_path):
    manifest = dx.write_dfanalyze_export(_parts(groups=False), tmp_path, 'cls')
    table = pd.read_csv(tmp_path / 'dfanalyze.csv')
    assert table.columns.tolist() == [f'emb_{i}' for i in range(6)] + ['target']
    assert len(table) == 10  # the test rows, which the model did not see
    assert manifest['ids_in_csv'] == [] and manifest['rows_in_dfanalyze_csv'] == ['test']
    for split, n in (('train', 20), ('val', 6), ('test', 10)):
        assert len(pd.read_csv(tmp_path / f'dfanalyze_{split}.csv')) == n
    assert '--grouper' not in manifest['command'] and '--target target' in manifest['command'] and '--mode=classify' in manifest['command']


def test_groups_are_included_and_passed_to_grouper_by_default(tmp_path):
    parts = _parts()
    manifest = dx.write_dfanalyze_export(parts, tmp_path, 'cls')
    table = pd.read_csv(tmp_path / 'dfanalyze.csv')
    assert table.columns.tolist() == [f'emb_{i}' for i in range(6)] + ['target', 'group']
    assert table['group'].tolist() == parts['test'].groups
    assert '--grouper group' in manifest['command'] and '--drops' not in manifest['command']
    assert manifest['ids_in_csv'] == ['group']


def test_the_export_says_these_rows_are_exploratory(tmp_path):
    manifest = dx.write_dfanalyze_export(_parts(), tmp_path, 'cls')
    assert 'exploratory, not a second sealed test' in manifest['status_of_these_rows']
    other = dx.write_dfanalyze_export({'embed': _part(5, 0, 'e')}, tmp_path / 'e', 'cls', rows=['embed'])
    assert 'exploratory' not in other['status_of_these_rows']


def test_few_rows_for_the_features_warn(tmp_path):
    few = dx.write_dfanalyze_export(_parts(), tmp_path / 'a', 'cls')  # 10 rows, 6 features
    assert any('rows per feature' in w for w in few['warnings'])
    enough = dx.write_dfanalyze_export({'test': _part(60, 0, 't')}, tmp_path / 'b', 'cls')
    assert not any('rows per feature' in w for w in enough['warnings'])


def test_pca_rows_inside_the_csv_are_called_out(tmp_path):
    manifest = dx.write_dfanalyze_export(_parts(), tmp_path, 'cls', dims=2, rows=['train', 'val', 'test'], model_saw_fit_split=False)
    assert any('PCA was fitted on' in w for w in manifest['warnings'])
    assert not any('PCA was fitted on' in w for w in dx.write_dfanalyze_export(_parts(), tmp_path / 't', 'cls', dims=2)['warnings'])


def test_ids_file_lines_up_with_the_split_files(tmp_path):
    parts = _parts()
    dx.write_dfanalyze_export(parts, tmp_path, 'cls')
    ids = pd.read_csv(tmp_path / 'dfanalyze_ids.csv')
    assert ids.columns.tolist() == ['row_id', 'split', 'video_id', 'group']
    test_ids = ids[ids['split'] == 'test']
    assert test_ids['video_id'].tolist() == list(parts['test'].video_ids) and test_ids['group'].tolist() == parts['test'].groups
    table = pd.read_csv(tmp_path / 'dfanalyze_test.csv')
    np.testing.assert_allclose(table['emb_0'], parts['test'].embedding[:, 0], rtol=1e-6)
    assert table['target'].tolist() == parts['test'].target.tolist()


def test_ids_only_when_asked_and_the_command_keeps_them_out_of_the_features(tmp_path):
    manifest = dx.write_dfanalyze_export(_parts(), tmp_path, 'cls', extras=['ids'])
    table = pd.read_csv(tmp_path / 'dfanalyze.csv')
    assert {'split', 'video_id', 'group'} <= set(table.columns)
    assert '--grouper group --drops split,video_id' in manifest['command']


def test_regression_command(tmp_path):
    parts = {'test': dx.SplitBlocks(np.linspace(0, 1, 5), list('abcde'), np.eye(5, dtype=np.float32))}
    manifest = dx.write_dfanalyze_export(parts, tmp_path, 'reg', extras=['ids'])
    assert '--mode=regress' in manifest['command'] and '--drops split,video_id' in manifest['command'] and '--grouper' not in manifest['command']
    assert 'group' not in pd.read_csv(tmp_path / 'dfanalyze.csv').columns


def test_pca_matches_sklearn_fitted_on_train_alone(tmp_path):
    parts = _parts()
    manifest = dx.write_dfanalyze_export(parts, tmp_path, 'cls', dims=3)
    reference = PCA(n_components=3, svd_solver='full').fit(np.asarray(parts['train'].embedding, dtype=np.float64))
    for name in ('train', 'val', 'test'):
        table = pd.read_csv(tmp_path / f'dfanalyze_{name}.csv')
        assert table.columns.tolist() == ['pca_0', 'pca_1', 'pca_2', 'target', 'group']
        np.testing.assert_allclose(table[['pca_0', 'pca_1', 'pca_2']].to_numpy(), reference.transform(parts[name].embedding), rtol=1e-4, atol=1e-5)
    assert manifest['pca']['fit_split'] == 'train' and manifest['pca']['n_fit_rows'] == 20 and manifest['pca']['used'] == 3
    saved = np.load(tmp_path / 'dfanalyze_pca.npz')
    np.testing.assert_allclose(saved['mean'], parts['train'].embedding.mean(axis=0), rtol=1e-6)


def test_changing_test_or_val_rows_cannot_change_the_fit(tmp_path):
    parts = _parts()
    dx.write_dfanalyze_export(parts, tmp_path / 'a', 'cls', dims=3)
    changed = _parts()
    changed['test'].embedding = changed['test'].embedding * 100 + 7
    changed['val'].embedding = changed['val'].embedding * -50
    manifest = dx.write_dfanalyze_export(changed, tmp_path / 'b', 'cls', dims=3)
    a, b = np.load(tmp_path / 'a' / 'dfanalyze_pca.npz'), np.load(tmp_path / 'b' / 'dfanalyze_pca.npz')
    np.testing.assert_array_equal(a['components'], b['components'])
    np.testing.assert_array_equal(a['mean'], b['mean'])
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'a' / 'dfanalyze_train.csv'), pd.read_csv(tmp_path / 'b' / 'dfanalyze_train.csv'))
    assert manifest['pca']['fit_video_ids_sha256']


def test_pca_refuses_without_training_rows_and_caps_at_what_they_allow(tmp_path):
    with pytest.raises(ValueError, match='training rows'):
        dx.write_dfanalyze_export({'test': _part(5, 0, 'te')}, tmp_path, 'cls', dims=2)
    manifest = dx.write_dfanalyze_export({'train': _part(4, 0, 'tr'), 'test': _part(5, 1, 'te')}, tmp_path, 'cls', dims=50)
    assert manifest['pca']['used'] == 4 and manifest['pca']['requested'] == 50
    assert any('kept 4 components' in w for w in manifest['warnings'])


def test_tta_and_motion_columns(tmp_path):
    dx.write_dfanalyze_export(_parts(std=True, motion=True), tmp_path, 'cls', extras=['tta', 'motion'])
    columns = pd.read_csv(tmp_path / 'dfanalyze.csv').columns.tolist()
    assert 'emb_5' in columns and 'emb_std_5' in columns and 'motion_p90' in columns
    assert columns[-2:] == ['target', 'group']


def test_pca_with_tta_reduces_each_block_from_training_rows(tmp_path):
    parts = _parts(std=True, motion=True)
    dx.write_dfanalyze_export(parts, tmp_path, 'cls', dims=2, extras=['tta', 'motion'])
    table = pd.read_csv(tmp_path / 'dfanalyze_test.csv')
    assert table.columns.tolist() == ['pca_0', 'pca_1', 'pca_std_0', 'pca_std_1', *dx.MOTION_STATISTICS, 'target', 'group']
    reference = PCA(n_components=2, svd_solver='full').fit(np.asarray(parts['train'].embedding_std, dtype=np.float64))
    np.testing.assert_allclose(table[['pca_std_0', 'pca_std_1']].to_numpy(), reference.transform(parts['test'].embedding_std), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('field', ['video_ids', 'groups'])
def test_a_video_or_group_in_two_splits_is_refused(tmp_path, field):
    parts = _parts()
    values = list(getattr(parts['test'], field))
    values[0] = getattr(parts['train'], field)[0]
    setattr(parts['test'], field, values)
    with pytest.raises(ValueError, match='Split leakage'):
        dx.write_dfanalyze_export(parts, tmp_path, 'cls')


def test_all_rows_warns_that_the_model_trained_on_some_of_them(tmp_path):
    manifest = dx.write_dfanalyze_export(_parts(), tmp_path, 'cls', rows=['train', 'val', 'test'])
    assert len(pd.read_csv(tmp_path / 'dfanalyze.csv')) == 36
    assert any('read too high' in w for w in manifest['warnings'])
    quiet = dx.write_dfanalyze_export(_parts(), tmp_path / 'p', 'cls', rows=['train', 'val', 'test'], model_saw_fit_split=False)
    assert not any('read too high' in w for w in quiet['warnings'])


def test_non_finite_embeddings_are_refused(tmp_path):
    parts = _parts()
    parts['test'].embedding[0, 0] = np.nan
    with pytest.raises(ValueError, match='not finite'):
        dx.write_dfanalyze_export(parts, tmp_path, 'cls')


def test_manifest_is_valid_json_and_states_the_command_was_not_run(tmp_path):
    dx.write_dfanalyze_export(_parts(), tmp_path, 'cls', provenance={'embedding_source': 'x'})
    manifest = json.loads((tmp_path / dx.MANIFEST_FILENAME).read_text())
    assert manifest['command_was_run_here'] is False and manifest['embedding_source'] == 'x'


def test_holdout_rows_can_be_found_again_with_get_intersection(tmp_path):
    """df-analyze's 40% holdout, as a CSV of features and target, back to its video ids through dfanalyze.parquet."""
    parts = _parts()
    dx.write_dfanalyze_export(parts, tmp_path, 'cls', dims=3)
    holdout = pd.read_csv(tmp_path / 'dfanalyze.csv').iloc[[1, 4, 7, 8]]
    found = gi.intersect(pd.read_parquet(tmp_path / 'dfanalyze.parquet'), holdout)
    assert found['video_id'].tolist() == ['te1', 'te4', 'te7', 'te8'] and found['split'].eq('test').all()


def test_curve_statistics_by_hand():
    got = dx.curve_statistics([99., 1., 2., 3.])  # the first value is left out
    assert got['motion_mean'] == 2 and got['motion_max'] == 3 and got['motion_std'] == pytest.approx(np.std([1, 2, 3]))
    assert got['motion_p90'] == pytest.approx(2.8)
    assert dx.curve_statistics([5.]) == dict.fromkeys(dx.MOTION_STATISTICS, 0.0)

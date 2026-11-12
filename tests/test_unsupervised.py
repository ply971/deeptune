"""Phase 4 Part C: unsupervised analysis of clip embeddings (analysis/unsupervised.py)."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis.unsupervised import (AnalysisSettings, AutoencoderSettings, agglomerative, agreement, analyse,
                                   assign_roles, auroc, elbow_k, feature_columns, fit_autoencoder, fpr_threshold,
                                   kmeans, kmeans_sweep, lda_analysis, load_embeddings, main, pca_analysis,
                                   reconstruction_error, standardise)


def blobs(k=4, per=40, dim=16, spread=0.6, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 5, size=(k, dim))
    X = np.vstack([c + rng.normal(0, spread, size=(per, dim)) for c in centres])
    y = np.repeat([f'c{i}' for i in range(k)], per)
    return X, y


def test_kmeans_and_agglomerative_recover_known_groups():
    X, y = blobs()
    Z = standardise(X)
    assert agreement(y, kmeans(Z, 4))['ari'] >= 0.9
    for link in ('ward', 'average', 'complete'):
        clusters, Z_link = agglomerative(Z, 4, link)
        assert agreement(y, clusters)['ari'] >= 0.9, link
        assert Z_link.shape == (len(X) - 1, 4)  # SciPy linkage = dendrogram data
    with pytest.raises(ValueError):
        agglomerative(Z, 4, 'centroid')


@pytest.mark.parametrize('k', [2, 3, 5])
def test_silhouette_picks_the_true_number_of_groups(k):
    X, _ = blobs(k=k, seed=k)
    sweep = kmeans_sweep(standardise(X), range(2, 8))
    assert int(sweep.loc[sweep['silhouette'].idxmax(), 'k']) == k
    assert sweep['inertia'].is_monotonic_decreasing


def test_elbow_on_a_hand_made_curve():
    assert elbow_k([1, 2, 3, 4, 5, 6], [100, 40, 12, 10, 9, 8]) == 3
    assert elbow_k([2, 3], [5, 4]) == 2


def test_kmeans_sweep_rejects_impossible_k():
    X, _ = blobs(per=3)
    with pytest.raises(ValueError):
        kmeans_sweep(X, [len(X)])


def test_pca_explained_variance_on_data_with_known_structure():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(500, 2))
    X = np.c_[t @ np.array([[3.0, 0, 0, 0], [0, 1.0, 0, 0]]) + rng.normal(0, 0.01, size=(500, 4))]
    res = pca_analysis(X)
    r = res['explained_variance_ratio']
    assert r[0] == pytest.approx(0.9, abs=0.03) and r[1] == pytest.approx(0.1, abs=0.03)
    assert r[:2].sum() > 0.999 and res['components_for_90pct'] in (1, 2)
    assert np.allclose(res['cumulative'], np.cumsum(r))


def test_lda_separates_labelled_classes():
    X, y = blobs(k=3, spread=2.0, seed=2)
    res = lda_analysis(standardise(X), y)
    assert res['projection'].shape == (len(X), 2)
    assert res['training_accuracy'] >= 0.95
    with pytest.raises(ValueError):
        lda_analysis(X, ['a'] * len(X))


def test_fpr_threshold_and_auroc_on_hand_made_scores():
    normal = np.arange(1, 101, dtype=float)  # 1..100
    assert fpr_threshold(normal, 0.05) == 96.0
    assert (normal > fpr_threshold(normal, 0.05)).mean() == pytest.approx(0.04)
    assert auroc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([1, 2]), np.array([1, 1])) is None
    with pytest.raises(ValueError):
        fpr_threshold(normal, 0)


def test_roles_keep_outliers_out_of_fitting():
    exclude = np.zeros(50, bool)
    exclude[:5] = True
    roles = assign_roles(50, 0.6, 0.2, seed=3, exclude=exclude)
    assert (roles[:5] == 'test').all()
    assert (roles == 'fit').sum() == 27 and (roles == 'val').sum() == 9


def test_autoencoder_flags_injected_out_of_distribution_clips():
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(3, 20))
    normal = rng.normal(size=(300, 3)) @ basis + rng.normal(0, 0.05, size=(300, 20))  # lives on a 3-D subspace
    ood = rng.normal(0, 1.0, size=(30, 20)) * normal.std(axis=0)                     # same scale, off the subspace
    X = np.vstack([normal, ood])
    y = np.array(['normal'] * 300 + ['ood'] * 30)
    settings = AnalysisSettings(ks=(2, 3), outlier_labels=('ood',), target_fpr=0.05, plots=False,
                                autoencoder=AutoencoderSettings(epochs=150, bottleneck=3, hidden=32, seed=0))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        summary = analyse(X, y, d, settings)
    a = summary['anomaly']
    assert a['auc'] >= 0.95
    assert a['detection_rate'] >= 0.8
    assert a['test_false_positive_rate'] <= 0.15


def test_autoencoder_training_is_seeded_and_reduces_loss():
    X, _ = blobs(k=2, dim=8)
    Z = standardise(X)
    m1 = fit_autoencoder(Z, AutoencoderSettings(epochs=30, seed=7))
    m2 = fit_autoencoder(Z, AutoencoderSettings(epochs=30, seed=7))
    assert np.allclose(reconstruction_error(m1, Z), reconstruction_error(m2, Z))
    assert m1.losses[-1] < m1.losses[0]


def test_outlier_rows_cannot_be_used_for_fitting(tmp_path):
    X, y = blobs(k=2)
    roles = np.array(['fit'] * len(X), dtype=object)
    roles[-10:] = 'val'
    with pytest.raises(ValueError, match='outlier'):
        analyse(X, y, tmp_path, AnalysisSettings(ks=(2,), outlier_labels=('c1',), plots=False), roles=roles)


def test_feature_columns_follow_the_embed_output_layout():
    frame = pd.DataFrame({'0': [1.0], '1': [2.0], 'label': ['a'], 'videos': ['x.mp4']})
    assert feature_columns(frame) == ['0', '1']
    frame2 = pd.DataFrame({'emb_a': [1.0], 'emb_b': [2.0], 'label': ['a']})
    assert feature_columns(frame2, 'emb_') == ['emb_a', 'emb_b']
    with pytest.raises(ValueError):
        feature_columns(frame2)


def test_cli_end_to_end_on_an_embedding_parquet(tmp_path):
    X, y = blobs(k=3, per=30, dim=12, seed=5)
    rng = np.random.default_rng(9)
    X = np.vstack([X, rng.normal(0, 8, size=(10, 12))])
    y = np.r_[y, ['odd'] * 10]
    frame = pd.DataFrame(X)
    frame.columns = [str(c) for c in frame.columns]
    frame['label'] = y
    frame['videos'] = [f'v{i}.mp4' for i in range(len(frame))]
    path = tmp_path / 'emb.parquet'
    frame.to_parquet(path, index=False)
    out = tmp_path / 'out'
    assert main(['--embeddings', str(path), '--out', str(out), '--k_max', '6', '--outlier_label', 'odd',
                 '--ae_epochs', '60', '--ae_bottleneck', '4']) == 0
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['rows'] == 100 and summary['dimensions'] == 12
    assert summary['kmeans']['agreement']['ari'] >= 0.9  # the scattered outliers may form small clusters of their own
    assert summary['lda']['training_accuracy'] >= 0.95
    assert summary['anomaly']['auc'] >= 0.9
    for name in ('pca.csv', 'kmeans_sweep.csv', 'clusters.csv', 'anomaly.csv', 'linkage.npy', 'pca.png', 'lda.png',
                 'kmeans_sweep.png', 'dendrogram.png', 'anomaly.png'):
        assert (out / name).exists(), name
    clusters = pd.read_csv(out / 'clusters.csv')
    assert len(clusters) == 100 and {'videos', 'label', 'kmeans_cluster', 'anomaly_score', 'anomalous', 'role'} <= set(clusters.columns)
    assert (clusters.loc[clusters['label'] == 'odd', 'role'] == 'test').all()


def test_cli_rejects_bad_arguments_before_any_work(tmp_path):
    frame = pd.DataFrame({'0': np.arange(10.0), '1': np.arange(10.0), 'label': ['a'] * 10})
    path = tmp_path / 'e.csv'
    frame.to_csv(path, index=False)
    for extra in (['--k_min', '1'], ['--k_min', '5', '--k_max', '3'], ['--target_fpr', '1.5'],
                  ['--role_column', 'nope'], ['--ae_epochs', '0']):
        with pytest.raises(SystemExit) as exc:
            main(['--embeddings', str(path), '--out', str(tmp_path / 'o'), *extra])
        assert exc.value.code == 2, extra
    with pytest.raises(SystemExit):
        main(['--embeddings', str(tmp_path / 'missing.parquet'), '--out', str(tmp_path / 'o')])


def test_load_embeddings_rejects_non_finite_values(tmp_path):
    frame = pd.DataFrame({'0': [1.0, np.nan], '1': [0.0, 1.0]})
    path = tmp_path / 'e.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='NaN'):
        load_embeddings(path)


def test_roles_by_group_never_put_one_group_in_two_roles():
    groups = np.repeat([f'g{i}' for i in range(20)], 5)             # 20 source videos x 5 clips
    exclude = np.zeros(100, bool)
    exclude[7] = True                                               # one known outlier inside group g1
    roles = assign_roles(100, 0.6, 0.2, seed=1, exclude=exclude, groups=groups)
    per_group = pd.DataFrame({'g': groups, 'r': roles}).groupby('g')['r'].nunique()
    assert (per_group == 1).all()
    assert (roles[groups == 'g1'] == 'test').all()                 # the outlier's whole group stays out of fitting
    # fractions apply to the 19 groups without an outlier
    assert (roles == 'fit').sum() == 5 * round(0.6 * 19) and (roles == 'val').sum() == 5 * round(0.2 * 19)
    with pytest.raises(ValueError):
        assign_roles(100, groups=groups[:50])


def test_cli_keeps_groups_together_and_reports_the_false_alarm_interval(tmp_path):
    X, y = blobs(k=3, per=40, dim=8, seed=2)
    frame = pd.DataFrame(X)
    frame.columns = [str(c) for c in frame.columns]
    frame['label'] = y
    frame['group'] = [f'v{i // 4}' for i in range(len(frame))]
    path = tmp_path / 'emb.parquet'
    frame.to_parquet(path, index=False)
    out = tmp_path / 'out'
    assert main(['--embeddings', str(path), '--out', str(out), '--k_max', '4', '--ae_epochs', '20', '--no_plots']) == 0
    anomaly = json.loads((out / 'summary.json').read_text())['anomaly']
    assert anomaly['role_unit'] == 'group' and anomaly['group_column'] == 'group'
    k, m = anomaly['test_false_positives']
    lo, hi = anomaly['test_false_positive_rate_ci95']
    assert anomaly['test_false_positive_rate'] == pytest.approx(k / m) and 0 <= lo <= k / m <= hi <= 1
    clusters = pd.read_csv(out / 'clusters.csv')
    assert (clusters.groupby('group')['role'].nunique() == 1).all()
    # '' turns grouping off; an unknown column is a usage error
    assert main(['--embeddings', str(path), '--out', str(tmp_path / 'o2'), '--k_max', '3', '--ae_epochs', '5',
                 '--no_plots', '--group_column', '']) == 0
    assert json.loads((tmp_path / 'o2' / 'summary.json').read_text())['anomaly']['role_unit'] == 'row'
    with pytest.raises(SystemExit) as exc:
        main(['--embeddings', str(path), '--out', str(tmp_path / 'o3'), '--group_column', 'actor'])
    assert exc.value.code == 2

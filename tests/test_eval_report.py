"""Phase 5 metrics: scikit-learn parity, group-level intervals, paired differences, calibration, report files."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk

from evaluators import calibration as cal
from evaluators import stats
from evaluators.video import report as rp

Y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1])
P = np.array([[.7, .2, .1], [.2, .5, .3], [.5, .4, .1], [.1, .8, .1], [.3, .3, .4],
              [.1, .1, .8], [.2, .2, .6], [.6, .2, .2], [.1, .3, .6], [.2, .7, .1]])


def test_classification_metrics_match_sklearn():
    pred = P.argmax(axis=1)
    got = rp.classification_metrics(Y, P)
    assert got['accuracy'] == pytest.approx(sk.accuracy_score(Y, pred))
    assert got['balanced_accuracy'] == pytest.approx(sk.balanced_accuracy_score(Y, pred))
    assert got['macro_f1'] == pytest.approx(sk.f1_score(Y, pred, average='macro'))
    assert got['weighted_f1'] == pytest.approx(sk.f1_score(Y, pred, average='weighted'))
    assert got['macro_auc'] == pytest.approx(sk.roc_auc_score(Y, P, multi_class='ovr', average='macro'))
    assert rp.confusion(Y, pred, 3).tolist() == sk.confusion_matrix(Y, pred).tolist()


def test_f1_matches_sklearn_when_a_class_is_never_seen_or_predicted():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 3])  # class 2 appears nowhere; class 3 is predicted but absent from the truth
    got = rp._confusion_scores(rp.confusion(y, pred, 4))
    assert got['macro_f1'] == pytest.approx(sk.f1_score(y, pred, average='macro', zero_division=0))
    assert got['weighted_f1'] == pytest.approx(sk.f1_score(y, pred, average='weighted', zero_division=0))


def test_binary_auc_matches_sklearn():
    y = np.array([0, 0, 1, 1, 1, 0])
    score = np.array([.1, .4, .35, .8, .4, .2])
    p = np.stack([1 - score, score], axis=1)
    assert rp.classification_metrics(y, p)['macro_auc'] == pytest.approx(sk.roc_auc_score(y, score))


def test_per_class_table_matches_sklearn_and_hand_values():
    pred = P.argmax(axis=1)
    table = rp.per_class_table(Y, pred, 3)
    for k, row in enumerate(table):
        truth, said = Y == k, pred == k
        assert row['sensitivity'] == pytest.approx(sk.recall_score(truth, said))
        assert row['specificity'] == pytest.approx(sk.recall_score(~truth, ~said))
        assert row['ppv'] == pytest.approx(sk.precision_score(truth, said))
        assert row['npv'] == pytest.approx(sk.precision_score(~truth, ~said))
    # class 1 by hand: truth at 3, 4, 9; predicted at 1, 3, 9 -> tp 2, fn 1, fp 1, tn 6
    assert (table[1]['tp'], table[1]['fn'], table[1]['fp'], table[1]['tn']) == (2, 1, 1, 6)
    assert table[1]['npv'] == pytest.approx(6 / 7)


def test_zero_denominators_are_none_not_zero():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    table = rp.per_class_table(y, pred, 3)
    assert table[1]['ppv'] is None and table[1]['sensitivity'] == 0  # never predicted: PPV 0/0; missed all: 0/2
    assert table[0]['npv'] is None  # never predicted "not 0"
    assert table[2]['sensitivity'] is None and table[2]['ppv'] is None and table[2]['specificity'] == 1  # absent class
    p = np.array([[.9, .1], [.8, .2], [.7, .3]])
    assert rp.classification_metrics(np.array([0, 0, 0]), p)['macro_auc'] is None  # one class only


def test_regression_metrics_match_sklearn():
    y = np.array([1., 2., 3., 5.])
    yhat = np.array([1.5, 1.5, 3.5, 4.])
    got = rp.regression_metrics(y, yhat)
    assert got['mae'] == pytest.approx(sk.mean_absolute_error(y, yhat))
    assert got['rmse'] == pytest.approx(np.sqrt(sk.mean_squared_error(y, yhat)))
    assert got['r2'] == pytest.approx(sk.r2_score(y, yhat))


def test_group_bootstrap_keeps_groups_whole():
    groups = np.array(['a', 'a', 'a', 'b', 'c', 'c'])
    seen = []

    def metric(idx):
        seen.append(idx.copy())
        return float(len(idx))
    stats.bootstrap_ci(metric, 6, groups=groups, n_boot=50, seed=1)
    for idx in seen[:-1]:
        counts = pd.Series(idx).value_counts()
        assert counts.get(0, 0) == counts.get(1, 0) == counts.get(2, 0)  # group a enters whole
        assert counts.get(4, 0) == counts.get(5, 0)


def test_unit_choice_group_then_source_then_row():
    assert stats.unit_labels(3, groups=['x', 'x', 'y'])[1] == 'group'
    assert stats.unit_labels(3, sources=['a.mp4', 'a.mp4', 'b.mp4'])[1] == 'source_video'
    assert stats.unit_labels(3, sources=['a.mp4', 'b.mp4', 'c.mp4'])[1] == 'row'
    with pytest.raises(ValueError, match='missing'):
        stats.unit_labels(2, groups=['x', None])


def test_row_interval_warns_and_states_what_it_covers():
    ci = stats.bootstrap_ci(lambda idx: float(np.mean(idx)), 30, n_boot=100)
    assert ci['unit'] == 'row' and "add a 'group' column" in ci['warning'] and 'test-sample' in ci['covers']


def _clustered(rng, n_groups=30, per_group=10):
    """Correct/wrong per clip, with a strong per-group effect."""
    effect = rng.random(n_groups) < 0.7
    outcomes = np.repeat(effect, per_group).astype(float)
    flips = rng.random(outcomes.shape) < 0.05
    outcomes[flips] = 1 - outcomes[flips]
    return outcomes, np.repeat(np.arange(n_groups), per_group)


def test_group_interval_reaches_nominal_coverage_where_clip_interval_falls_short():
    """P-117's test. True rate is computable: 0.7*0.95 + 0.3*0.05 = 0.68."""
    truth = 0.7 * 0.95 + 0.3 * 0.05
    rng = np.random.default_rng(0)
    covered = {'group': 0, 'row': 0}
    trials = 150
    for t in range(trials):
        outcomes, groups = _clustered(rng)
        by_group = stats.bootstrap_ci(lambda i: outcomes[i].mean(), len(outcomes), groups=groups, n_boot=200, seed=t)
        by_row = stats.bootstrap_ci(lambda i: outcomes[i].mean(), len(outcomes), n_boot=200, seed=t)
        covered['group'] += by_group['low'] <= truth <= by_group['high']
        covered['row'] += by_row['low'] <= truth <= by_row['high']
    assert covered['group'] / trials >= 0.85
    assert covered['row'] / trials <= 0.70


def test_paired_difference_is_narrower_than_separate_intervals_and_excludes_zero():
    """B is A with a constant 8% of rows broken: pairing sees the difference despite the shared difficulty of the rows."""
    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 2, n)
    easy = rng.random(n) < 0.7
    pa = np.where(easy, y, 1 - y)
    pb = pa.copy()
    broken = np.flatnonzero(pa == y)[:32]
    pb[broken] = 1 - pb[broken]
    a, b = np.eye(2)[pa], np.eye(2)[pb]
    out = rp.compare_models('cls', y, a, b, n_boot=500, seed=0)['metrics']['accuracy']
    separate_a = stats.bootstrap_ci(lambda i: (pa[i] == y[i]).mean(), n, n_boot=500)
    assert out['point'] == pytest.approx(32 / n)
    assert out['high'] - out['low'] < separate_a['high'] - separate_a['low']
    assert out['zero_excluded'] is True


def test_paired_comparison_refuses_different_test_sets(tmp_path):
    for name, labels in (('a', [0, 1, 0]), ('b', [1, 1, 0])):
        folder = tmp_path / name
        folder.mkdir()
        pd.DataFrame({'true_label': labels, 'predicted_class': [0, 1, 0], 'prob_0': [.6, .4, .7], 'prob_1': [.4, .6, .3]}).to_csv(folder / 'predictions.csv')
    with pytest.raises(ValueError, match='one test set'):
        rp.compare_eval_dirs(tmp_path / 'a', tmp_path / 'b')


def test_fold_summary_matches_hand_values():
    out = stats.fold_summary([0.7, 0.8, 0.9])
    assert out['mean'] == pytest.approx(0.8) and out['std'] == pytest.approx(0.1)
    half = 4.302652729911275 * 0.1 / np.sqrt(3)  # t(0.975, 2)
    assert out['fold_t_low'] == pytest.approx(0.8 - half) and 'too narrow' in out['covers']


# -- calibration -----------------------------------------------------------------------------------------------------
def test_ece_and_brier_against_hand_computed_values():
    p = np.array([[.9, .1], [.8, .2], [.6, .4], [.3, .7]])
    y = np.array([0, 1, 0, 1])
    # 10 bins. conf .9 right; .8 wrong; .6 right; .7 right. Bins hold one each:
    # (|1-.9| + |0-.8| + |1-.6| + |1-.7|) / 4 = 0.4
    assert cal.expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.4)
    # sum_k (p_k - y_k)^2: .02, 1.28, .32, .18 -> mean .45
    assert cal.brier_score(p, y) == pytest.approx(0.45)
    assert cal.brier_score(p, y) == pytest.approx(2 * sk.brier_score_loss(y, p[:, 1]))


def test_brier_matches_sklearn_multiclass():
    assert cal.brier_score(P, Y) == pytest.approx(sk.brier_score_loss(Y, P, scale_by_half=False))


def test_nll_matches_sklearn():
    assert cal.negative_log_likelihood(P, Y) == pytest.approx(sk.log_loss(Y, P))


def test_a_confidence_on_a_bin_edge_goes_in_the_lower_bin():
    rows = cal.reliability_bins(np.array([[.5, .5], [.6, .4], [1., 0.]]), np.array([0, 0, 0]), n_bins=10)
    assert [r['n'] for r in rows] == [0, 0, 0, 0, 1, 1, 0, 0, 0, 1]


def _overconfident(n, seed, classes=4):
    """x ~ N(1.5 e_y, I), so the Bayes posterior is softmax(1.5 x). The model says softmax(4 x): true T = 4 / 1.5."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    logits = rng.normal(size=(n, classes)) + 1.5 * np.eye(classes)[y]
    p = np.exp(4 * logits)
    return p / p.sum(axis=1, keepdims=True), y


def test_temperature_lowers_ece_on_an_overconfident_model_and_keeps_accuracy():
    fit_p, fit_y = _overconfident(3000, 0)
    test_p, test_y = _overconfident(3000, 1)
    t = cal.fit_temperature(fit_p, fit_y)
    assert t == pytest.approx(4 / 1.5, abs=0.2)
    scaled = cal.apply_temperature(test_p, t)
    assert (scaled.argmax(axis=1) == test_p.argmax(axis=1)).all()
    assert cal.expected_calibration_error(scaled, test_y) < 0.5 * cal.expected_calibration_error(test_p, test_y)


def test_temperature_is_fitted_on_the_fit_rows_only():
    """Changing the test rows must not change the temperature; refitting on them would."""
    fit_p, fit_y = _overconfident(800, 0)
    test_p, test_y = _overconfident(800, 1)
    first = cal.calibration_report(test_p, test_y, fit_p, fit_y, 'validation')
    second = cal.calibration_report(test_p, np.roll(test_y, 7), fit_p, fit_y, 'validation')
    assert first['temperature'] == second['temperature'] and first['fit_split'] == 'validation' and first['n_fit'] == 800
    assert first['temperature'] != pytest.approx(cal.fit_temperature(test_p, np.roll(test_y, 7)), abs=0.1)


@pytest.mark.parametrize('fit_split', ['test', None])
def test_fitting_on_test_is_refused(fit_split):
    with pytest.raises(ValueError, match='another split'):
        cal.calibration_report(P, Y, P, Y, fit_split)


def test_no_fit_set_reports_before_only():
    out = cal.calibration_report(P, Y)
    assert out['after'] is None and out['temperature'] is None and 'not fitted' in out['note']


# -- files -----------------------------------------------------------------------------------------------------------
def test_report_files_for_classification(tmp_path):
    groups = ['g%d' % (i // 2) for i in range(len(Y))]
    report = rp.build_report('cls', Y, P, groups=groups, fit_scores=P, fit_labels=Y, fit_split='validation',
                             class_names=['a', 'b', 'c'], train_labels=[0, 0, 1], n_boot=50, selection_used_fit_split=True)
    path = rp.write_report(report, tmp_path, Y, P)
    saved = json.loads(path.read_text())
    assert saved['resampling_unit'] == 'group' and saved['n_units'] == 5
    assert saved['calibration']['fit_split_used_for_model_selection'] is True
    assert saved['chance']['accuracy'] == pytest.approx(0.3) and 'majority class 0' in saved['chance']['predicts']
    assert any('only 5 group units' in w for w in saved['warnings'])
    for name in ('per_class_metrics.csv', 'confusion_matrix.csv', 'roc_curves.csv', 'reliability.csv', 'metric_intervals.csv',
                 'confusion_matrix.png', 'roc_curves.png', 'reliability_diagram.png'):
        assert (tmp_path / name).stat().st_size > 0, name
    table = pd.read_csv(tmp_path / 'per_class_metrics.csv')
    assert list(table['name']) == ['a', 'b', 'c']


def test_many_classes_plot_and_keep_every_probability(tmp_path):
    k = 60
    rng = np.random.default_rng(0)
    y = np.arange(120) % k
    p = rng.dirichlet(np.ones(k), size=120)
    pd.DataFrame({'sample_index': range(120), 'true_label': y, 'predicted_class': p.argmax(1)}).to_csv(tmp_path / 'predictions.csv')
    np.save(tmp_path / rp.SCORES_FILENAME, p)
    out = rp.report_eval_dir(tmp_path, n_boot=20)
    assert json.loads(out.read_text())['metrics']['macro_auc'] is not None
    assert (tmp_path / 'roc_curves.png').is_file()


def test_report_files_for_regression(tmp_path):
    y, yhat = np.arange(10.), np.arange(10.) + .5
    report = rp.build_report('reg', y, yhat, n_boot=50)
    rp.write_report(report, tmp_path, y, yhat)
    assert (tmp_path / 'residuals.png').is_file()
    assert pd.read_csv(tmp_path / 'residuals.csv')['residual'].eq(.5).all()
    assert report['metrics']['mae'] == .5 and 'not applicable' in report['calibration']['note']


def test_class_names_read_both_mapping_directions(tmp_path):
    path = tmp_path / 'map.json'
    path.write_text('{"left": 0, "right": 1}')
    assert rp._class_names(path) == ['left', 'right']
    path.write_text('{"0": "left", "1": "right"}')
    assert rp._class_names(path) == ['left', 'right']
    assert rp._class_names(None) is None


def test_report_cli_refuses_a_temperature_without_a_named_split(tmp_path):
    with pytest.raises(SystemExit):
        rp.main(['--eval_dir', str(tmp_path), '--fit_dir', str(tmp_path)])

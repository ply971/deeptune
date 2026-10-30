"""Unit tests for evaluators/error_model.py (per-prediction error estimate)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluators.error_model import (
    ErrorModel, calibration_table, error_targets, find_error_model, isotonic_fit_values,
    load_error_model_near, prediction_features, save_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _overconfident(n, rng, return_oracle=False):
    """3-class probabilities whose true error rate is a function of max_prob
    (the feature the error model sees) and is ~1.6x what 1 - max_prob
    claims: an over-confident model.

    Column 0 holds the confidence c and is always the arg-max: the other
    two columns split the remaining 1 - c with both shares <= c, i.e.
    a in [1 - c/(1-c), c/(1-c)] intersected with [0, 1]."""
    c = rng.uniform(1 / 3, 1.0, n)
    ratio = c / np.maximum(1 - c, 1e-12)
    a = rng.uniform(np.clip(1 - ratio, 0, 1), np.clip(ratio, 0, 1))
    probs = np.column_stack([c, (1 - c) * a, (1 - c) * (1 - a)])
    p_wrong = np.clip(1.6 * (1 - probs.max(axis=1)), 0, 0.95)
    wrong = (rng.uniform(size=n) < p_wrong).astype(float)
    return (probs, wrong, p_wrong) if return_oracle else (probs, wrong)


def test_synthetic_fixture_error_depends_only_on_max_prob():
    probs, _, p_wrong = _overconfident(20000, np.random.default_rng(9), return_oracle=True)
    assert np.array_equal(probs.argmax(axis=1), np.zeros(len(probs), dtype=int))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(p_wrong, np.clip(1.6 * (1 - probs[:, 0]), 0, 0.95))


def test_features_for_classification():
    probs = np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0]])
    f = prediction_features(probs, 'cls')
    assert list(f.columns) == ['max_prob', 'margin', 'entropy']
    np.testing.assert_allclose(f['max_prob'], [0.7, 1 / 3, 1.0])
    np.testing.assert_allclose(f['margin'], [0.5, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(f['entropy'][1:], [1.0, 0.0], atol=1e-9)
    assert 0 < f['entropy'][0] < 1
    from_logits = prediction_features(np.log(probs[:2]), 'cls', from_logits=True)
    np.testing.assert_allclose(from_logits.to_numpy(), f.iloc[:2].to_numpy(), atol=1e-9)


def test_features_accept_extra_columns_and_reject_bad_shapes():
    probs = np.array([[0.6, 0.4], [0.9, 0.1]])
    f = prediction_features(probs, 'cls', extra={'tta_disagreement': [0.1, 0.0]})
    assert f['tta_disagreement'].tolist() == [0.1, 0.0]
    g = prediction_features(np.array([1.0, 2.0]), 'reg', extra=np.array([[3.0], [4.0]]))
    assert list(g.columns) == ['predicted_value', 'extra_0']
    with pytest.raises(ValueError, match='rows'):
        prediction_features(probs, 'cls', extra={'x': [1.0]})
    with pytest.raises(ValueError, match='clash'):
        prediction_features(probs, 'cls', extra={'margin': [1.0, 2.0]})


def test_error_targets():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert error_targets(probs, np.array([0, 0, 1]), 'cls').tolist() == [0.0, 1.0, 1.0]
    assert error_targets(np.array([[1.0], [3.0]]), np.array([2.0, 3.0]), 'reg').tolist() == [1.0, 0.0]


def test_stated_error_matches_observed_per_decile_on_held_out_data():
    rng = np.random.default_rng(0)
    fit_probs, fit_wrong = _overconfident(4000, rng)
    test_probs, test_wrong, test_oracle = _overconfident(4000, rng, return_oracle=True)
    model = ErrorModel('cls').fit(prediction_features(fit_probs, 'cls'), fit_wrong)
    assert model.method_used == 'isotonic'
    stated = model.predict(prediction_features(test_probs, 'cls'))
    report = calibration_table(stated, test_wrong, n_bins=10)
    assert report['summary']['n_bins'] == 10
    for row in report['bins']:
        assert row['n'] == 400
        assert abs(row['gap']) < 0.1, row
    assert report['summary']['calibration_error'] < 0.05
    # Against observed outcomes, it may lose only a little calibration
    # relative to the oracle itself on the same held-out draws.
    oracle = calibration_table(test_oracle, test_wrong, n_bins=10)
    assert report['summary']['calibration_error'] <= oracle['summary']['calibration_error'] + 0.02
    # Raw 1 - max_prob is badly over-confident on the same data; the error
    # model is what makes the number honest.
    raw = calibration_table(1 - test_probs.max(axis=1), test_wrong)
    assert raw['summary']['calibration_error'] > 0.15
    assert report['summary']['auroc_error_detection'] > 0.75


def _truth_gap_medians(scale=1.0, seeds=range(10)):
    """Noise-free check against the truth: per decile of stated error, the
    gap between the stated error and the generator's true P(wrong) of the
    same held-out samples (no test-set sampling noise). Returns the medians,
    over `seeds`, of the worst-decile gap and the mean-decile gap.
    `scale` multiplies the stated error to simulate a mis-calibrated model."""
    worst, mean = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        fit_probs, fit_wrong = _overconfident(4000, rng)
        test_probs, _, truth = _overconfident(4000, rng, return_oracle=True)
        model = ErrorModel('cls').fit(prediction_features(fit_probs, 'cls'), fit_wrong)
        stated = scale * model.predict(prediction_features(test_probs, 'cls'))
        order = np.argsort(stated, kind='mergesort')
        gaps = np.array([abs(stated[idx].mean() - truth[idx].mean()) for idx in np.array_split(order, 10)])
        worst.append(gaps.max())
        mean.append(gaps.mean())
    return float(np.median(worst)), float(np.median(mean))


# Bounds come from the multi-seed spread, not one seed. Over 200 seeds a
# single seed's worst-decile gap has median 0.044 / p95 0.071 and its
# mean-decile gap median 0.016 / p95 0.022; 10-seed medians range
# 0.035-0.052 and 0.014-0.019 over 20 disjoint seed blocks.
TRUTH_GAP_MAX, TRUTH_GAP_MEAN = 0.07, 0.025


def test_stated_error_tracks_the_true_error_rate_over_seeds():
    worst, mean = _truth_gap_medians()
    assert worst < TRUTH_GAP_MAX, worst
    assert mean < TRUTH_GAP_MEAN, mean


def test_truth_gap_check_catches_a_miscalibrated_model():
    # Under-stating every error by 15% must fail the same check, clearly
    # (10-seed medians ~0.145 / ~0.08 over 20 seed blocks).
    worst, mean = _truth_gap_medians(scale=0.85)
    assert worst > 2 * TRUTH_GAP_MAX or mean > 2 * TRUTH_GAP_MEAN, (worst, mean)
    assert worst >= TRUTH_GAP_MAX and mean >= TRUTH_GAP_MEAN, (worst, mean)


def test_logistic_error_model_is_also_calibrated():
    rng = np.random.default_rng(1)
    fit_probs, fit_wrong = _overconfident(3000, rng)
    test_probs, test_wrong = _overconfident(3000, rng)
    model = ErrorModel('cls', method='logistic').fit(prediction_features(fit_probs, 'cls'), fit_wrong)
    assert model.method_used == 'logistic'
    report = calibration_table(model.predict(prediction_features(test_probs, 'cls')), test_wrong)
    assert report['summary']['calibration_error'] < 0.05


@pytest.mark.parametrize('method', ['isotonic', 'logistic'])
def test_stated_error_never_rises_with_confidence(method):
    rng = np.random.default_rng(2)
    probs, wrong = _overconfident(600, rng)
    features = ['max_prob'] if method == 'logistic' else None
    model = ErrorModel('cls', method=method, features=features).fit(prediction_features(probs, 'cls'), wrong)
    grid = np.linspace(0.34, 1.0, 200)
    grid_probs = np.column_stack([grid, (1 - grid) / 2, (1 - grid) / 2])
    stated = model.predict(prediction_features(grid_probs, 'cls'))
    assert np.all(np.diff(stated) <= 1e-12)
    assert np.all((stated >= 0) & (stated <= 1))


def test_pav_matches_scikit_learn_reference():
    from sklearn.isotonic import isotonic_regression
    rng = np.random.default_rng(3)
    for _ in range(5):
        y = rng.normal(size=60)
        w = rng.uniform(0.5, 2.0, size=60)
        np.testing.assert_allclose(isotonic_fit_values(y, w),
                                   isotonic_regression(y, sample_weight=w, increasing=True), atol=1e-10)


def test_tiny_fit_set_falls_back_to_base_rate():
    probs = np.array([[0.9, 0.1]] * 6 + [[0.55, 0.45]] * 4)
    wrong = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 0], dtype=float)
    model = ErrorModel('cls').fit(prediction_features(probs, 'cls'), wrong)
    assert model.method_used == 'constant'
    assert 'fit samples' in model.fit_info['fallback_reason']
    expected = (3 + 1) / (10 + 2)
    np.testing.assert_allclose(model.predict(prediction_features(probs, 'cls')), expected)
    assert model.fit_info['base_error'] == pytest.approx(0.3)


def test_fit_set_without_errors_falls_back_to_smoothed_base_rate():
    rng = np.random.default_rng(4)
    c = rng.uniform(0.5, 1.0, 200)
    probs = np.column_stack([c, 1 - c])
    model = ErrorModel('cls').fit(prediction_features(probs, 'cls'), np.zeros(200))
    assert model.method_used == 'constant'
    assert 'wrong predictions' in model.fit_info['fallback_reason']
    np.testing.assert_allclose(model.predict(prediction_features(probs, 'cls')), 1 / 202)


def test_regression_without_signal_predicts_mean_abs_error_and_with_signal_adapts():
    rng = np.random.default_rng(5)
    preds = rng.normal(size=500)
    sigma = rng.uniform(0.1, 2.0, 500)
    abs_err = np.abs(rng.normal(scale=sigma))
    plain = ErrorModel('reg').fit(prediction_features(preds, 'reg'), abs_err)
    assert plain.method_used == 'constant'
    np.testing.assert_allclose(plain.predict(prediction_features(preds, 'reg')), abs_err.mean())
    feats = prediction_features(preds, 'reg', extra={'sigma': sigma})
    adaptive = ErrorModel('reg', score_feature='sigma').fit(feats, abs_err)
    assert adaptive.method_used == 'isotonic'
    stated = adaptive.predict(feats)
    assert np.corrcoef(stated, sigma)[0, 1] > 0.9
    assert np.all(stated >= 0)


@pytest.mark.parametrize('method', ['auto', 'logistic', 'constant'])
def test_json_round_trip_gives_identical_predictions(tmp_path, method):
    rng = np.random.default_rng(6)
    probs, wrong = _overconfident(500, rng)
    feats = prediction_features(probs, 'cls')
    model = ErrorModel('cls', method=method).fit(feats, wrong, num_outputs=3)
    path = model.save(tmp_path / 'error_model.json')
    data = json.loads(path.read_text())  # plain JSON, no pickle
    assert data['format'] == 'deeptune-error-model'
    loaded = ErrorModel.load(path)
    assert np.array_equal(loaded.predict(feats), model.predict(feats))
    assert ErrorModel.from_json(model.to_json()).to_dict() == model.to_dict()


def test_predict_requires_its_features():
    rng = np.random.default_rng(7)
    probs, wrong = _overconfident(200, rng)
    model = ErrorModel('cls').fit(prediction_features(probs, 'cls'), wrong)
    with pytest.raises(ValueError, match='max_prob'):
        model.predict(pd.DataFrame({'margin': [0.1]}))


def test_calibration_table_edge_cases_and_report_files(tmp_path):
    empty = calibration_table([], [], n_bins=10)
    assert empty['summary']['n'] == 0 and empty['bins'] == []
    small = calibration_table([0.1, 0.2, 0.9], [0, 0, 1], n_bins=10)
    assert small['summary']['n_bins'] == 3
    assert small['summary']['mode'] == 'cls'
    rep = calibration_table(np.linspace(0, 1, 50), (np.arange(50) % 2).astype(float), n_bins=5)
    assert [b['n'] for b in rep['bins']] == [10] * 5
    json_path, csv_path = save_report(rep, tmp_path, extra={'split': 'test'})
    assert json.loads(json_path.read_text())['split'] == 'test'
    table = pd.read_csv(csv_path)
    assert len(table) == 5 and {'stated_mean', 'observed_mean', 'gap'} <= set(table.columns)


def test_find_and_load_error_model_next_to_checkpoint(tmp_path):
    ckpt = tmp_path / 'model_weights.pth'
    ckpt.write_bytes(b'')
    assert find_error_model(ckpt) is None and load_error_model_near(ckpt, 'cls') is None
    rng = np.random.default_rng(8)
    probs, wrong = _overconfident(100, rng)
    ErrorModel('cls').fit(prediction_features(probs, 'cls'), wrong, num_outputs=3).save(tmp_path / 'error_model.json')
    assert find_error_model(ckpt) == tmp_path / 'error_model.json'
    assert find_error_model(tmp_path) == tmp_path / 'error_model.json'
    assert load_error_model_near(ckpt, 'cls', num_outputs=3) is not None
    warnings = []
    assert load_error_model_near(ckpt, 'reg', warn=warnings.append) is None
    assert load_error_model_near(ckpt, 'cls', num_outputs=5, warn=warnings.append) is None
    (tmp_path / 'error_model.json').write_text('{broken')
    assert load_error_model_near(ckpt, 'cls', warn=warnings.append) is None
    assert len(warnings) == 3


def test_module_does_not_import_torch():
    code = ('import sys; import evaluators.error_model, inference.common; '
            'print("torch" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True,
                         timeout=120, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'

"""k-fold and .632+ logic, on arrays and a stand-in model; the pipeline test is in test_evaluation_pipeline_e2e.py."""
import json

import numpy as np
import pandas as pd
import pytest

from evaluators.video import validation as va


def _labels(n=60, classes=3):
    return np.arange(n) % classes


def _groups(n=60, per_group=3):
    return [f'g{i // per_group}' for i in range(n)]


def test_kfold_covers_each_row_once_and_is_stratified():
    labels = _labels()
    folds = va.kfold_splits(labels, 5)
    held = np.concatenate([h for _, h in folds])
    assert sorted(held.tolist()) == list(range(60))
    for fit, h in folds:
        assert not set(fit) & set(h)
        assert np.bincount(labels[h]).tolist() == [4, 4, 4]


def test_grouped_kfold_never_has_a_group_on_both_sides():
    labels = np.array([i % 2 for i in range(60)])
    groups = np.array(_groups())
    for mode in ('cls', 'reg'):
        for fit, held in va.kfold_splits(labels, 5, groups=groups, mode=mode):
            assert not set(groups[fit]) & set(groups[held])


def test_grouped_inner_validation_never_shares_a_group():
    labels = np.array([i % 2 for i in range(60)])
    groups = np.array(_groups())
    fit, _ = va.kfold_splits(labels, 5, groups=groups)[0]
    for mode in ('cls', 'reg'):
        train, val = va.inner_validation_split(labels, fit, groups, mode)
        assert len(val) and not set(groups[train]) & set(groups[val])
        assert sorted(np.concatenate([train, val]).tolist()) == sorted(fit.tolist())


def test_too_few_groups_or_clips_are_refused_with_a_reason():
    with pytest.raises(ValueError, match='at least 5 groups'):
        va.kfold_splits(_labels(12), 5, groups=_groups(12, per_group=6))
    with pytest.raises(ValueError, match='per class'):
        va.kfold_splits(np.array([0] * 10 + [1] * 2), 5)
    with pytest.raises(ValueError, match='k of at least 2'):
        va.kfold_splits(_labels(), 1)


def test_a_bootstrap_draws_whole_groups():
    labels = np.array([i % 2 for i in range(60)])
    groups = np.array(_groups())
    for in_bag, out_bag in va.bootstrap_resamples(labels, 30, groups=groups, seed=0):
        assert not set(groups[in_bag]) & set(groups[out_bag])
        counts = pd.Series(in_bag).value_counts()
        for g in set(groups[in_bag]):
            members = np.flatnonzero(groups == g)
            assert len(set(counts.reindex(members).tolist())) == 1  # all members drawn as often


def test_bootstrap_inner_validation_keeps_repeats_on_one_side():
    labels = _labels()
    in_bag, _ = va.bootstrap_resamples(labels, 1, seed=3)[0]
    train, val = va.inner_validation_split(labels, in_bag)
    assert not set(train) & set(val) and len(train) + len(val) == len(in_bag)


def test_632_plus_against_a_hand_worked_example():
    """err .1, Err1 .4, gamma .6. R = .3/.5 = .6. w = .632 / (1 - .368 * .6) = .632/.7792 = .811088...
    Err.632+ = (1-w)*.1 + w*.4 = .1 + .3 w = .3433265."""
    got = va.efron_632_plus(0.1, 0.4, 0.6)
    assert got['relative_overfitting'] == pytest.approx(0.6)
    assert got['weight'] == pytest.approx(0.632 / 0.7792)
    assert got['error_632_plus'] == pytest.approx(0.1 + 0.3 * 0.632 / 0.7792)
    assert got['error_632'] == pytest.approx(0.368 * 0.1 + 0.632 * 0.4)


def test_632_plus_edge_cases():
    # Err1 at or below the apparent error: R = 0, plain .632.
    got = va.efron_632_plus(0.3, 0.2, 0.5)
    assert got['relative_overfitting'] == 0 and got['error_632_plus'] == pytest.approx(0.368 * 0.3 + 0.632 * 0.2)
    # Err1 above gamma is clipped to gamma: R = 1, w = 1, estimate = gamma.
    got = va.efron_632_plus(0.0, 0.7, 0.5)
    assert got['weight'] == pytest.approx(1.0) and got['error_632_plus'] == pytest.approx(0.5)
    # gamma not above the apparent error.
    assert va.efron_632_plus(0.5, 0.6, 0.5)['relative_overfitting'] == 0


def test_no_information_error_by_hand():
    # p = (.5, .5), q = (.75, .25): .5 * .25 + .5 * .75 = .5
    assert va.no_information_error([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(.5)
    # p = (1, 0) for class 0; q = (.5, .5): 1 * .5 = .5
    assert va.no_information_error([0, 0], [0, 1]) == pytest.approx(.5)
    y, f = np.array([1., 3.]), np.array([0., 4.])
    pairs = np.mean([(a - b) ** 2 for a in y for b in f])
    assert va.no_information_error(y, f, 'reg') == pytest.approx(pairs)


class Recorder:
    """A stand-in model: predicts each row's true class with a set accuracy, and logs the rows it was given."""

    def __init__(self, labels, classes=3, accuracy=.8, seed=0, reg=False):
        self.labels, self.classes, self.accuracy, self.reg = labels, classes, accuracy, reg
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def __call__(self, train_rows, val_rows, eval_rows, name):
        self.calls.append((name, np.array(train_rows), np.array(val_rows), np.array(eval_rows)))
        if self.reg:
            return va.FoldScores(self.labels[eval_rows] + self.rng.normal(0, .1, len(eval_rows)))
        y = self.labels[eval_rows]
        hit = self.rng.random(len(y)) < self.accuracy
        said = np.where(hit, y, (y + 1) % self.classes)
        p = np.full((len(y), self.classes), .1 / max(1, self.classes - 1))
        p[np.arange(len(y)), said] = .9
        return va.FoldScores(p, expected_error=np.full(len(y), .2))


def _run(tmp_path, scheme='kfold', **kw):
    labels = kw.pop('labels', _labels())
    recorder = kw.pop('recorder', Recorder(labels))
    summary = va.run_validation(scheme, labels, recorder, tmp_path, ids=[f'clip{i}' for i in range(len(labels))], ci_boot=50, **kw)
    return summary, recorder


def test_each_fold_is_a_fresh_call_with_disjoint_rows(tmp_path):
    summary, recorder = _run(tmp_path)
    assert [c[0] for c in recorder.calls] == [f'fold_{i}' for i in range(5)]
    for _, train, val, held in recorder.calls:
        assert not set(train) & set(val) and not set(train) & set(held) and not set(val) & set(held)
    assert sorted(np.concatenate([c[3] for c in recorder.calls]).tolist()) == list(range(60))
    assert summary['n_rows'] == 60 and summary['n_scored_rows'] == 60
    assert summary['across_folds']['accuracy']['n'] == 5


def test_rows_outside_the_pool_cannot_reach_a_fold(tmp_path):
    """The sealed test set: rows 1000+ are not passed in, so no fold sees them, and assignments.csv is the record."""
    labels = _labels()
    row_ids = np.arange(60) + 100
    va.run_validation('kfold', labels, Recorder(labels), tmp_path, row_ids=row_ids, ci_boot=20)
    assignments = pd.read_csv(tmp_path / va.ASSIGNMENTS_FILENAME)
    assert set(assignments['row_id']) == set(row_ids.tolist())
    assert set(assignments['role']) == {'train', 'validation', 'heldout'}
    assert assignments[assignments['role'] == 'heldout']['row_id'].is_unique


def test_oof_file_format_is_stable(tmp_path):
    groups = _groups()
    _run(tmp_path, groups=groups)
    table = va.load_oof_predictions(tmp_path)
    assert list(table.columns) == ['row_id', 'fold', 'video_id', 'target', 'group', 'n_models', 'prediction', 'confidence', 'correct',
                                   'expected_error', 'prob_0', 'prob_1', 'prob_2']
    assert table['row_id'].tolist() == list(range(60)) and table['video_id'][7] == 'clip7' and table['group'][7] == groups[7]
    assert table['target'].tolist() == _labels().tolist() and table['expected_error'].eq(.2).all()
    summary = json.loads((tmp_path / va.SUMMARY_FILENAME).read_text())
    assert summary['schema_version'] == va.OOF_SCHEMA_VERSION == 1


def test_loader_refuses_a_table_that_is_not_out_of_fold(tmp_path):
    pd.DataFrame({'row_id': [0], 'target': [1]}).to_csv(tmp_path / 'x.csv')
    with pytest.raises(ValueError, match='not an out-of-fold'):
        va.load_oof_predictions(tmp_path / 'x.csv')


def test_summary_intervals_resample_groups_and_say_so(tmp_path):
    summary, _ = _run(tmp_path, groups=_groups())
    interval = summary['out_of_fold']['intervals']['accuracy']
    assert interval['unit'] == 'group' and interval['n_units'] == 20
    assert interval['low'] <= summary['out_of_fold']['metrics']['accuracy'] <= interval['high']


def test_strong_and_chance_models_are_told_apart(tmp_path):
    labels = _labels(120)
    strong, _ = _run(tmp_path / 's', labels=labels, recorder=Recorder(labels, accuracy=.95))
    chance, _ = _run(tmp_path / 'c', labels=labels, recorder=Recorder(labels, accuracy=1 / 3))
    assert strong['out_of_fold']['intervals']['accuracy']['low'] > chance['out_of_fold']['intervals']['accuracy']['high']


def test_oof_calibration_is_cross_fitted_and_lowers_ece(tmp_path):
    summary, _ = _run(tmp_path, labels=_labels(300), recorder=Recorder(_labels(300), accuracy=.6))
    calibration = summary['calibration']
    assert calibration['after']['ece'] < calibration['before']['ece']  # stated .9, right 60% of the time
    assert len(calibration['temperature_per_fold']) == 5 and 'other folds' in calibration['fit_rows']


def test_oof_error_model_states_the_error_it_sees_on_rows_it_did_not_fit(tmp_path):
    """Error depends on confidence: the deciles of cross-fitted statements must follow it."""
    n = 2000
    rng = np.random.default_rng(0)
    confidence = rng.uniform(.5, 1, n)
    labels = rng.integers(0, 2, n)
    right = rng.random(n) < confidence
    said = np.where(right, labels, 1 - labels)
    p = np.zeros((n, 2))
    p[np.arange(n), said] = confidence
    p[np.arange(n), 1 - said] = 1 - confidence
    oof = pd.DataFrame({'row_id': range(n), 'group': None, 'correct': right, 'prediction': said, 'prob_0': p[:, 0], 'prob_1': p[:, 1]})
    out = va.fit_oof_error_model(oof, 'cls', tmp_path)
    assert out['method'] == 'isotonic'
    assert out['held_out_summary']['calibration_error'] < .05 and out['held_out_summary']['bins_within_95ci'] >= 8
    table = json.loads((tmp_path / 'error_calibration_oof.json').read_text())
    assert table['split'] == 'out-of-fold, cross-fitted'
    assert (tmp_path / 'error_model_oof.json').is_file()


def test_bootstrap_632_plus_end_to_end_with_a_stand_in(tmp_path):
    labels = _labels()
    apparent = Recorder(labels, accuracy=1.0)
    summary, recorder = _run(tmp_path, 'bootstrap632', n_boot=8, groups=_groups(), apparent_fit_predict=apparent)
    assert len(recorder.calls) == 8 and apparent.calls[0][0] == 'apparent'
    out = summary['bootstrap632']
    assert out['apparent_error'] == 0 and out['resampled_units'] == 'groups' and out['loss'] == '0-1 error'
    expected = va.efron_632_plus(0, out['oob_error'], out['no_information_error'])['error_632_plus']
    assert out['error_632_plus'] == pytest.approx(expected)
    long = pd.read_csv(tmp_path / 'oob_predictions_all_resamples.csv')
    # Err^(1) by its definition, computed again from the long table.
    assert out['oob_error'] == pytest.approx((1 - long.groupby('row_id')['correct'].mean()).mean())
    oof = va.load_oof_predictions(tmp_path)
    assert oof['row_id'].is_unique and oof['n_models'].sum() == len(long)
    assert oof['prob_0'].sum() + oof['prob_1'].sum() + oof['prob_2'].sum() == pytest.approx(len(oof))


def test_bootstrap_without_a_model_on_all_rows_says_so(tmp_path):
    summary, _ = _run(tmp_path, 'bootstrap632', n_boot=3)
    assert summary['bootstrap632']['error_632_plus'] is None and 'not computed' in summary['bootstrap632']['note']


def test_regression_kfold_and_632_plus(tmp_path):
    y = np.linspace(0, 5, 40)
    summary, _ = _run(tmp_path / 'k', labels=y, recorder=Recorder(y, reg=True), mode='reg')
    assert summary['across_folds']['mae']['mean'] < .2 and 'calibration' not in summary
    table = va.load_oof_predictions(tmp_path / 'k')
    assert 'abs_error' in table.columns and table['target'].iloc[3] == y[3]
    summary, _ = _run(tmp_path / 'b', 'bootstrap632', labels=y, recorder=Recorder(y, reg=True), mode='reg', n_boot=4,
                      apparent_fit_predict=Recorder(y, reg=True))
    assert summary['bootstrap632']['loss'] == 'squared error' and summary['bootstrap632']['error_632_plus'] < .1


def test_wrong_number_of_scores_is_an_error(tmp_path):
    labels = _labels()
    with pytest.raises(ValueError, match='expected'):
        va.run_validation('kfold', labels, lambda t, v, e, n: va.FoldScores(np.zeros((3, 3))), tmp_path, ci_boot=10)


def test_holdout_is_not_a_scheme_to_run():
    with pytest.raises(ValueError, match='Unknown validation scheme'):
        va.run_validation('holdout', _labels(), None, 'unused')

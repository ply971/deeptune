"""Abstention: hand-checked formulas, the file format, and the guarantees themselves over many seeds. numpy only, no model. See test_abstention_pipeline.py for the run."""
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from evaluators.abstention import (DEFAULT_DELTA, GRID_SIZE, NOT_SURE, UNIT_NOTES, AbstentionRule, apply_rule, build_report, carve_calibration_rows, clopper_pearson, conformal_rank,
                                   conformal_threshold, fit_abstention, group_worst_scores, groups_needed, load_rule_near, loose_below, overlapping_units, pipeline_issues,
                                   realised_coverage_interval, risk_p_values, risk_upper_bound, select_selective_threshold, selective_curve, threshold_grid, threshold_stats,
                                   write_report)
from evaluators.abstention_sim import DEPENDENT, INDEPENDENT, MOSTLY_WRONG, STRESS, Scenario, draw, true_risks

GRID = threshold_grid(2)
ALPHA = 0.10
DELTA = 0.10


# -- formulas, by hand -----------------------------------------------------------------------------------------------
def test_grid_is_fixed_ascending_and_answers_everything_first():
    assert GRID[0] == 0.0 and len(GRID) == GRID_SIZE and (np.diff(GRID) > 0).all() and GRID[-1] < 1
    assert np.array_equal(threshold_grid(2), threshold_grid(2))


def test_hoeffding_bentkus_p_value_matches_the_published_formula():
    # n = 20, 1.5 units of loss, alpha = 0.2. min(exp(-n h(r, a)), e P(Bin(n, a) <= ceil(n r))), arXiv 2110.01052 Eq. 2.
    r, a, n = 1.5 / 20, 0.2, 20
    h = r * math.log(r / a) + (1 - r) * math.log((1 - r) / (1 - a))
    hand = min(math.exp(-n * h), math.e * stats.binom.cdf(2, n, a))
    assert risk_p_values(20, 1.5, 0.2)[0] == pytest.approx(hand)


def test_single_clip_units_use_the_exact_binomial():
    assert risk_p_values(50, 2.0, 0.1, single_clip_units=True)[0] == pytest.approx(stats.binom.cdf(2, 50, 0.1))


def test_no_answered_unit_rejects_nothing():
    assert risk_p_values(0, 0.0, 0.5)[0] == 1.0
    assert risk_upper_bound(0, 0.0, 0.05) == 1.0


def test_integer_loss_is_not_counted_as_one_more_by_rounding():
    assert risk_p_values(30, 3.0000000000001, 0.2, True)[0] == pytest.approx(stats.binom.cdf(3, 30, 0.2))


def test_upper_bound_is_the_edge_of_rejection():
    bound = risk_upper_bound(100, 4.0, 0.01)
    assert 0.04 < bound < 1
    assert risk_p_values(100, 4.0, bound + 1e-6)[0] <= 0.01 < risk_p_values(100, 4.0, bound - 1e-3)[0]


def test_group_losses_by_hand():
    score = np.array([.9, .9, .2, .8, .8, .1])
    wrong = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
    units = np.array([0, 0, 0, 1, 1, 2])
    row = threshold_stats(score, wrong, units, [0.5]).iloc[0]
    assert row['answered_units'] == 2 and row['unit_loss_sum'] == pytest.approx(0.5 + 0.0)
    assert row['answered_clips'] == 4 and row['wrong_clips'] == 1 and row['units_without_silent_error'] == 2


def test_conformal_rank_and_threshold_by_hand():
    assert conformal_rank(9, 0.1) == 9  # ceil(10 * 0.9)
    assert conformal_rank(19, 0.1) == 18
    worst = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
    assert conformal_threshold(worst, 0.1) == pytest.approx(.9) and conformal_threshold(worst, 0.1) > .9
    assert conformal_threshold(worst[:5], 0.1) is None  # (5 + 1) * 0.1 < 1
    assert conformal_threshold(np.full(19, -np.inf), 0.1) == 0.0  # no group holds an error


def test_group_worst_score_ignores_right_clips():
    worst = group_worst_scores(np.array([.9, .3, .8]), np.array([0, 1, 0], dtype=bool), np.array([0, 0, 1]))
    assert worst[0] == pytest.approx(.3) and np.isneginf(worst[1])


def test_beta_interval_closed_form():
    out = realised_coverage_interval(99, 0.1)
    assert out['distribution'] == 'Beta(90, 10)' and out['n_effective'] == 99
    assert out['low'] == pytest.approx(stats.beta.ppf(.025, 90, 10)) and out['high'] == pytest.approx(stats.beta.ppf(.975, 90, 10))
    assert realised_coverage_interval(5, 0.1)['low'] is None


def test_clopper_pearson_against_scipy():
    low, high = clopper_pearson(7, 40)
    expected = stats.binomtest(7, 40).proportion_ci(0.95, method='exact')
    assert low == pytest.approx(expected.low) and high == pytest.approx(expected.high)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_loose_rule_tracks_the_target():
    values = {a: loose_below(a) for a in (0.2, 0.1, 0.05, 0.01)}
    assert values[0.2] < values[0.1] < values[0.05] < values[0.01]
    for a, n in values.items():
        assert abs(n - 15.4 * (1 - a) / a) / (15.4 * (1 - a) / a) < 0.35  # the normal approximation stated in the docstring
        assert stats.beta.ppf(.975, math.floor((n + 1) * a), n + 1 - math.floor((n + 1) * a)) <= 1.5 * a
        l = math.floor(n * a)
        assert l < 1 or stats.beta.ppf(.975, l, n - l) > 1.5 * a


def test_groups_needed_is_the_zero_error_floor():
    n = groups_needed(0.1, 0.05)
    assert risk_p_values(n, 0.0, 0.1)[0] <= 0.05 / GRID_SIZE
    assert risk_p_values(n - 1, 0.0, 0.1)[0] > 0.05 / GRID_SIZE
    assert risk_upper_bound(n, 0.0, 0.05 / GRID_SIZE) <= 0.1


# -- fit, statements, refusal ----------------------------------------------------------------------------------------
def _easy(n_groups=2000, seed=0, scenario=Scenario(clips_per_group=3, group_sd=0.5, offset=-1.5)):
    return draw(np.random.default_rng(seed), n_groups, scenario)


@pytest.mark.parametrize('split', ['test', 'Validation', 'val', 'train'])
def test_refuses_to_fit_on_a_named_split(split):
    score, wrong, groups = _easy(50)
    with pytest.raises(ValueError, match='Refusing to calibrate'):
        fit_abstention(score, wrong, groups, target_error=0.1, split=split)


@pytest.mark.parametrize('kw', [{'target_error': 0}, {'target_error': 1}, {'target_error': .1, 'delta': 0}, {'target_error': .1, 'guarantee': 'accuracy'}])
def test_bad_settings_are_refused(kw):
    score, wrong, groups = _easy(50)
    with pytest.raises(ValueError):
        fit_abstention(score, wrong, groups, **kw)


def test_missing_scores_and_lengths_are_refused():
    with pytest.raises(ValueError, match='finite'):
        fit_abstention(np.array([np.nan, .5]), np.array([True, False]), ['a', 'b'], target_error=.1)
    with pytest.raises(ValueError):
        fit_abstention(np.array([.5]), np.array([True, False]), target_error=.1)


def test_met_target_states_the_bound_and_the_interval():
    score, wrong, groups = _easy()
    info = fit_abstention(score, wrong, groups, target_error=0.1, num_classes=2).info
    assert info['status'] == 'met' and info['guarantee_holds'] and info['unit'] == 'group' and info['n_units'] == 2000 and info['guarantee_level'] == 'group'
    assert info['risk_upper_bound'] <= 0.1
    interval = info['realised_coverage']
    assert interval['n_effective'] == 2000 and interval['low'] <= interval['point'] <= interval['high'] and interval['exact_beta_form'] is False
    assert info['effective_n'] <= 2000 and not info['loose'] and 'is loose' not in info['headline']
    assert 'probability at least 0.95' in info['guarantee_text']
    assert 'per-class error' in ' '.join(info['not_guaranteed'])
    assert info['temperature_applied'] is False
    assert info['threshold'] in GRID


def test_target_out_of_reach_abstains_on_everything_and_says_so():
    score, wrong, groups = _easy(5)
    rule = fit_abstention(score, wrong, groups, target_error=0.01)
    info = rule.info
    assert info['status'] == 'target_not_met' and info['threshold'] is None and info['guarantee_holds'] is False
    assert not rule.answers(score).any() and NOT_SURE in info['headline'] and info['guarantee_text'].startswith('None.')
    assert info['groups_needed_at_least'] == groups_needed(0.01, DEFAULT_DELTA)
    assert info['answer_rate_on_calibration'] == 0.0
    frame = pd.DataFrame({'confidence': score, 'predicted_label': 'x'})
    summary = apply_rule(rule, frame, {})
    assert (frame['decision'] == NOT_SURE).all() and summary['answered'] == 0 and summary['guarantee_holds'] is False


def test_silent_error_too_few_groups_abstains():
    score, wrong, groups = _easy(5)
    info = fit_abstention(score, wrong, groups, target_error=0.05, guarantee='silent_error').info
    assert info['status'] == 'target_not_met' and info['groups_needed_at_least'] == 19


def test_few_answered_groups_say_loose():
    score, wrong, groups = _easy(90, scenario=Scenario(clips_per_group=2, group_sd=0.1, offset=-6))
    info = fit_abstention(score, wrong, groups, target_error=0.1).info
    assert info['status'] == 'met' and info['loose'] and info['headline'].startswith('Guarantee is loose') and info['effective_n'] < info['loose_below_effective_n']


def test_a_constant_score_is_called_out():
    info = fit_abstention(np.full(40, 0.7), np.zeros(40, dtype=bool), [f'g{i}' for i in range(40)], target_error=0.3, score_name='expected_error').info
    assert any('same expected_error' in warning for warning in info['warnings'])


def test_group_weighted_error_can_be_far_from_clip_weighted_when_big_groups_err():
    # Nine one-clip groups all right, one nine-clip group all wrong: 0.1 by group, 0.5 by clip. This is why the guarantee names its weighting.
    groups = [*range(9), *[9] * 9]
    row = threshold_stats(np.full(18, .9), np.array([False] * 9 + [True] * 9), groups, [0.0]).iloc[0]
    assert row['unit_loss_sum'] / row['answered_units'] == pytest.approx(0.1)
    assert row['wrong_clips'] / row['answered_clips'] == pytest.approx(0.5)
    assert 'clip-weighted' in ' '.join(fit_abstention(np.full(18, .9), np.array([False] * 9 + [True] * 9), groups, target_error=.5).info['not_guaranteed'])


def test_no_group_column_says_clip_level_and_independent():
    score, wrong, _ = _easy(1000)
    info = fit_abstention(score, wrong, target_error=0.1).info
    assert info['unit'] == 'row' and info['guarantee_level'] == 'clip' and info['assumes_independent_clips'] is True
    assert 'clip level and assumes independent clips' in info['headline'] and UNIT_NOTES['row'] in info['warnings']
    frame = pd.DataFrame({'confidence': score, 'predicted_label': 'x'})
    summary = apply_rule(AbstentionRule(info), frame, {})
    assert summary['assumes_independent_clips'] is True and summary['guarantee_level'] == 'clip' and 'independent clips' in summary['headline']


def test_repeated_source_counts_as_the_group_and_says_so():
    score, wrong, groups = _easy(1000)
    info = fit_abstention(score, wrong, sources=[f'v{g}.mp4' for g in groups], target_error=0.1).info
    assert info['unit'] == 'source_video' and info['n_units'] == 1000 and info['assumes_independent_clips'] is False
    assert 'No group column' in info['headline']


def test_a_group_column_needs_no_clip_level_note():
    score, wrong, groups = _easy(1000)
    info = fit_abstention(score, wrong, groups, target_error=0.1).info
    assert info['assumes_independent_clips'] is False and not info['warnings'] and 'independent clips' not in info['headline']


def test_void_reasons_remove_the_guarantee_from_the_output():
    score, wrong, groups = _easy()
    rule = fit_abstention(score, wrong, groups, target_error=0.1, void_reasons=['3 calibration groups also appear in other splits.'])
    assert rule.info['guarantee_holds'] is False and rule.info['headline'].startswith('No guarantee') and rule.threshold is not None
    assert apply_rule(rule, pd.DataFrame({'confidence': score, 'predicted_label': 'x'}), {})['guarantee_holds'] is False


def test_rule_round_trip_gives_the_same_decisions(tmp_path):
    score, wrong, groups = _easy()
    rule = fit_abstention(score, wrong, groups, target_error=0.1, provenance={'weights_sha256': 'abc', 'pipeline': {'test_clips': 3}})
    path = rule.save(tmp_path / 'abstention.json')
    loaded = AbstentionRule.load(path)
    assert np.array_equal(loaded.answers(score), rule.answers(score)) and json.loads(path.read_text())['format'] == 'deeptune-abstention'


def test_a_newer_or_foreign_file_is_refused():
    with pytest.raises(ValueError):
        AbstentionRule.from_json('{"format": "other"}')
    with pytest.raises(ValueError, match='newer'):
        AbstentionRule.from_json('{"format": "deeptune-abstention", "version": 99}')


def test_load_near_refuses_other_weights_classes_and_corrupt_files(tmp_path):
    score, wrong, groups = _easy()
    checkpoint = tmp_path / 'model_weights.pth'
    assert load_rule_near(checkpoint) is None
    fit_abstention(score, wrong, groups, target_error=0.1, num_classes=2, provenance={'weights_sha256': 'abc'}).save(tmp_path / 'abstention.json')
    said = []
    assert load_rule_near(checkpoint, 2, 'abc', warn=said.append) is not None and not said
    assert load_rule_near(checkpoint, 2, 'other', warn=said.append) is None and 'different weights' in said[-1]
    assert load_rule_near(checkpoint, 5, 'abc', warn=said.append) is None and '5' in said[-1]
    (tmp_path / 'abstention.json').write_text('not json')
    assert load_rule_near(checkpoint, 2, 'abc', warn=said.append) is None and 'ignoring' in said[-1]


@pytest.mark.parametrize('change', [{'test_clips': 3}, {'test_crops': 3}, {'num_frames': 16}, {'temporal_sampling': 'dense'}])
def test_a_different_pipeline_warns_and_voids_the_guarantee(change):
    score, wrong, groups = _easy()
    fitted = {'test_clips': 1, 'test_crops': 1, 'num_frames': 8, 'temporal_sampling': 'uniform'}
    rule = fit_abstention(score, wrong, groups, target_error=0.1, provenance={'pipeline': fitted})
    assert pipeline_issues(rule, fitted) == []
    said = []
    frame = pd.DataFrame({'confidence': score, 'predicted_label': 'x'})
    summary = apply_rule(rule, frame, {**fitted, **change}, warn=said.append)
    assert summary['guarantee_holds'] is False and summary['pipeline_mismatches'] and said and 'does not hold' in said[0]
    assert summary['headline'].startswith('No guarantee') and 'decision' in frame.columns


def test_applying_a_temperature_would_void_the_guarantee():
    score, wrong, groups = _easy()
    rule = fit_abstention(score, wrong, groups, target_error=0.03)
    assert rule.threshold > 0
    assert pipeline_issues(rule, {}, temperature_applied=False) == []
    assert 'temperature' in pipeline_issues(rule, {}, temperature_applied=True)[0]
    # And it would change decisions: the threshold belongs to the scale it was fitted on.
    tau = rule.threshold
    logit = np.log(score / (1 - score))
    cooled = 1 / (1 + np.exp(-logit / 2))
    assert (rule.answers(cooled) != (score >= tau)).any()


def test_decision_column_uses_the_label_or_not_sure():
    rule = AbstentionRule({'format': 'deeptune-abstention', 'version': 1, 'threshold': 0.5, 'score': 'max_prob', 'target_error': .1, 'guarantee_kind': 'selective_risk',
                           'status': 'met', 'headline': 'x', 'guarantee_holds': True})
    frame = pd.DataFrame({'confidence': [.9, .2], 'predicted_label': ['cat', 'dog'], 'correct': [False, True]})
    summary = apply_rule(rule, frame, {})
    assert frame['decision'].tolist() == ['cat', NOT_SURE] and frame['answered'].tolist() == [True, False]
    assert summary['clip_error_among_answered'] == 1.0 and summary['not_sure'] == 1


def test_expected_error_score_is_optional_and_named():
    rule = AbstentionRule({'format': 'deeptune-abstention', 'version': 1, 'threshold': 0.7, 'score': 'expected_error', 'target_error': .1, 'guarantee_kind': 'selective_risk',
                           'status': 'met', 'headline': 'x', 'guarantee_holds': True})
    frame = pd.DataFrame({'confidence': [.9, .2], 'predicted_label': ['a', 'b']})
    said = []
    assert apply_rule(rule, frame, {}, warn=said.append) is None and 'expected_error' in said[0] and 'decision' not in frame
    frame['expected_error'] = [0.5, 0.1]  # the score is 1 - expected_error, not confidence
    apply_rule(rule, frame, {})
    assert frame['answered'].tolist() == [False, True]


# -- the calibration split -------------------------------------------------------------------------------------------
def test_carve_moves_whole_groups_and_keeps_every_row():
    frame = pd.DataFrame({'labels': np.tile([0, 1], 20), 'group': np.repeat(np.arange(10), 4)})
    train, calibration = carve_calibration_rows(frame, 3, frame['group'].to_numpy(), 0)
    assert not set(train['group']) & set(calibration['group']) and calibration['group'].nunique() == 3 and len(train) + len(calibration) == 40
    again, _ = carve_calibration_rows(frame, 3, frame['group'].to_numpy(), 0)
    assert train.equals(again)


def test_carve_refuses_to_empty_a_class_from_training():
    frame = pd.DataFrame({'labels': [0, 0, 1], 'group': [0, 1, 2]})
    with pytest.raises(ValueError, match='no training rows'):
        for seed in range(50):  # one seed picks the only group of class 1
            carve_calibration_rows(frame, 1, frame['group'].to_numpy(), seed)


def test_overlap_is_reported():
    assert overlapping_units(['a', 'b'], ['b', 'c'], ['d']) == ['b'] and overlapping_units(['a'], ['b']) == []


# -- the curve -------------------------------------------------------------------------------------------------------
def test_curve_by_hand_with_group_intervals():
    score = np.array([.99, .99, .6, .6, .55, .55])
    wrong = np.array([0, 0, 1, 0, 1, 1], dtype=bool)
    groups = ['a', 'a', 'b', 'b', 'c', 'c']
    curve = selective_curve(score, wrong, groups, grid=[0.0, 0.58, 0.9], n_boot=50)
    assert curve['coverage'].tolist() == pytest.approx([1, 4 / 6, 2 / 6])
    assert curve['error'].tolist() == pytest.approx([.5, .25, 0])
    assert curve['group_mean_error'].tolist() == pytest.approx([(0 + .5 + 1) / 3, .25, 0])
    assert (curve['error_low'] <= curve['error']).all() and (curve['error'] <= curve['error_high']).all()


def test_curve_resamples_groups_not_rows():
    score, wrong, groups = draw(np.random.default_rng(0), 30, Scenario(group_sd=2.0, score_mean=1.0))
    by_group = selective_curve(score, wrong, groups, grid=[0.0], n_boot=400)
    by_row = selective_curve(score, wrong, grid=[0.0], n_boot=400)
    assert (by_group['error_high'] - by_group['error_low'])[0] > 1.3 * (by_row['error_high'] - by_row['error_low'])[0]


def test_curve_drops_repeats_of_the_same_answered_set():
    curve = selective_curve(np.array([.9, .95]), np.array([0, 1], dtype=bool), grid=[.1, .2, .91, .99], n_boot=0)
    assert curve['answered_clips'].tolist() == [2, 1, 0]
    assert curve['error'].tolist()[2] is None or pd.isna(curve['error'].tolist()[2])


def test_report_writes_table_json_and_plot(tmp_path):
    pytest.importorskip('matplotlib')
    score, wrong, groups = _easy(400)
    rule = fit_abstention(score, wrong, groups, target_error=0.2)
    new_score, new_wrong, new_groups = _easy(200, seed=1)
    frame = pd.DataFrame({'confidence': new_score, 'correct': ~new_wrong, 'expected_error': 1 - new_score})
    report, curve = build_report(rule, frame, new_groups, n_boot=30, extra_warnings=['w'])
    path = write_report(report, curve, tmp_path)
    saved = json.loads(path.read_text())
    assert saved['target_error'] == 0.2 and saved['split'] == 'test' and saved['unit'] == 'group' and saved['n_units'] == 200 and saved['warnings'] == ['w']
    assert saved['at_fitted_threshold']['error_interval']['unit'] == 'group' and 'one draw' in saved['reading'].lower()
    assert saved['plots'] == 'written' and (tmp_path / 'selective_risk_curve.png').stat().st_size > 0
    assert 'mean_expected_error' in pd.read_csv(tmp_path / 'selective_risk_curve.csv').columns


def test_report_skips_the_plot_when_matplotlib_is_missing(tmp_path, monkeypatch, capsys):
    import evaluators.video.report as report_mod
    monkeypatch.setattr(report_mod, 'plotting_available', lambda: False)
    score, wrong, groups = _easy(100)
    report, curve = build_report(None, pd.DataFrame({'confidence': score, 'correct': ~wrong}), groups, n_boot=0)
    path = write_report(report, curve, tmp_path)
    assert 'skipped' in json.loads(path.read_text())['plots'] and not (tmp_path / 'selective_risk_curve.png').exists()
    assert (tmp_path / 'selective_risk_curve.csv').is_file() and 'matplotlib' in capsys.readouterr().out


def test_report_when_everything_abstains():
    score, wrong, groups = _easy(5)
    rule = fit_abstention(score, wrong, groups, target_error=0.01)
    report, _ = build_report(rule, pd.DataFrame({'confidence': score, 'correct': ~wrong}), groups, n_boot=0)
    assert report['at_fitted_threshold']['answered_clips'] == 0 and 'error' not in report['at_fitted_threshold'] and report['guarantee_holds'] is False


def test_module_is_torch_free():
    code = "import sys, evaluators.abstention, evaluators.abstention_sim, handlers.abstention_args; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)


# -- validity over many seeds ----------------------------------------------------------------------------------------
def _slack(rate, seeds):
    """Three binomial standard errors: room for the finite number of seeds."""
    return 3 * math.sqrt(rate * (1 - rate) / seeds)


@pytest.fixture(scope='module')
def populations():
    """Very large draws standing in for the truth at every grid threshold."""
    return {name: true_risks(*draw(np.random.default_rng(10_000), n, scenario), GRID)
            for name, (scenario, n) in {'independent': (INDEPENDENT, 400_000), 'dependent': (DEPENDENT, 60_000), 'stress': (STRESS, 60_000)}.items()}


def _failures(scenario, truth, n_groups, seeds, per_clip=False, truth_key='group', alpha=ALPHA, delta=DELTA):
    """Share of calibration draws whose picked threshold has true risk above the target, and the share of draws that picked one."""
    failed, chosen = 0, 0
    for seed in range(seeds):
        score, wrong, groups = draw(np.random.default_rng(seed), n_groups, scenario)
        units = np.arange(len(score)) if per_clip else groups
        single = per_clip or scenario.clips_per_group == 1
        tau = select_selective_threshold(threshold_stats(score, wrong, units, GRID), alpha, delta, single)
        if tau is not None:
            chosen += 1
            failed += truth[truth_key][np.searchsorted(GRID, tau)] > alpha
    return failed / seeds, chosen / seeds


def test_selective_risk_holds_on_exchangeable_data(populations):
    seeds = 400
    assert populations['independent']['group'][0] > ALPHA  # answering everything would miss the target
    rate, chosen = _failures(INDEPENDENT, populations['independent'], 1500, seeds)
    assert rate <= DELTA + _slack(DELTA, seeds)
    assert chosen > 0.9  # the check means little if it rarely answers


def test_the_exact_p_value_is_valid_and_not_vacuous_at_the_boundary():
    """One threshold, i.i.d. clips, true risk just above the target: a false claim should get through close to delta of the time, no more. This is the tight case Bonferroni hides."""
    alpha, delta, n, seeds = 0.1, 0.1, 300, 4000
    wrong_counts = np.random.default_rng(0).binomial(n, alpha + 0.001, seeds)
    rate = (stats.binom.cdf(wrong_counts, n, alpha) <= delta).mean()
    rate_from_code = (risk_p_values(np.full(seeds, n), wrong_counts.astype(float), alpha, True) <= delta).mean()
    assert rate_from_code == rate
    assert 0.5 * delta < rate_from_code <= delta + _slack(delta, seeds)


def test_the_hoeffding_bentkus_p_value_is_valid_for_fractional_group_losses():
    """False claims at the boundary, with group losses strictly between 0 and 1 and clips within a group dependent."""
    alpha, delta, n_groups, seeds = 0.1, 0.1, 100, 3000
    rng = np.random.default_rng(0)
    loss = rng.beta(0.2, 1.75, (seeds, n_groups))  # mean about 0.1026
    true_mean = 0.2 / 1.95
    assert true_mean > alpha
    rate = (risk_p_values(np.full(seeds, n_groups), loss.sum(axis=1), alpha) <= delta).mean()
    assert rate <= delta + _slack(delta, seeds)


def test_clip_counting_makes_false_claims_where_group_counting_does_not(populations):
    """The core claim. Every threshold's true error is above the target, so any answer is a false claim. Same data and rule; only the unit differs."""
    seeds = 800
    # The score is ignored by the error, so the risk is the same at every threshold; the answer-everything row estimates it best.
    truth = {'group': np.full(len(GRID), populations['stress']['group'][0])}
    assert truth['group'][0] > ALPHA
    grouped, _ = _failures(STRESS, truth, 60, seeds)
    naive, _ = _failures(STRESS, truth, 60, seeds, per_clip=True)
    assert grouped <= DELTA + _slack(DELTA, seeds)
    assert naive > DELTA + _slack(DELTA, seeds)
    assert naive > 5 * max(grouped, 1 / seeds)


def test_group_rule_stays_useful_with_dependent_clips_on_a_useful_score(populations):
    seeds = 200
    assert populations['dependent']['group'][0] > ALPHA
    rate, chosen = _failures(DEPENDENT, populations['dependent'], 600, seeds)
    assert rate <= DELTA + _slack(DELTA, seeds)
    assert chosen > 0.9


def test_clip_weighted_error_stays_close_for_equal_groups(populations):
    """With equal group sizes the group-weighted guarantee also describes the clip-weighted error reported beside it. Not a claim for unequal groups."""
    truth = populations['dependent']
    known = np.isfinite(truth['group']) & (GRID < 0.99)
    assert np.abs(truth['group'][known] - truth['clip'][known]).max() < 0.04


def test_no_bonferroni_would_pick_lower_thresholds():
    """Shows the correction does something: without it the same data answers at a lower threshold."""
    score, wrong, groups = draw(np.random.default_rng(0), 300, DEPENDENT)
    table = threshold_stats(score, wrong, groups, GRID)
    n, total = table['answered_units'][0], table['unit_loss_sum'][0]
    alpha = (risk_upper_bound(n, total, DELTA) + risk_upper_bound(n, total, DELTA / len(GRID))) / 2  # between the two bounds on answering everything
    corrected = select_selective_threshold(table, alpha, DELTA)
    uncorrected = select_selective_threshold(table, alpha, DELTA * len(GRID))
    assert uncorrected == 0.0 and (corrected is None or corrected > 0)


def _silent_coverage(n, alpha, seeds, scenario, truth_score, truth_wrong, truth_groups):
    worst_population = np.sort(group_worst_scores(truth_score, truth_wrong, truth_groups))
    realised = []
    for seed in range(seeds):
        score, wrong, groups = draw(np.random.default_rng(seed), n, scenario)
        tau = conformal_threshold(group_worst_scores(score, wrong, groups), alpha)
        realised.append(np.searchsorted(worst_population, tau, side='left') / len(worst_population))
    return np.array(realised)


NEARLY_ALL_GROUPS_ERR = Scenario(clips_per_group=8, group_sd=1.0, offset=3.0, score_mean=1.0)


@pytest.fixture(scope='module')
def silent_population():
    return draw(np.random.default_rng(20_000), 200_000, NEARLY_ALL_GROUPS_ERR)


def test_silent_error_marginal_coverage_is_at_nominal_and_not_far_above(silent_population):
    n, alpha, seeds = 99, 0.1, 1500
    assert np.isneginf(group_worst_scores(*silent_population)).mean() < 0.002  # nearly no ties, so the upper limit applies too
    realised = _silent_coverage(n, alpha, seeds, NEARLY_ALL_GROUPS_ERR, *silent_population)
    standard_error = realised.std() / math.sqrt(seeds)
    assert realised.mean() >= 1 - alpha - 3 * standard_error
    assert realised.mean() <= 1 - alpha + 1 / (n + 1) + 3 * standard_error


def test_silent_error_realised_coverage_follows_the_stated_beta(silent_population):
    n, alpha, seeds = 99, 0.1, 1500
    realised = _silent_coverage(n, alpha, seeds, NEARLY_ALL_GROUPS_ERR, *silent_population)
    interval = realised_coverage_interval(n, alpha)
    inside = ((realised >= interval['low']) & (realised <= interval['high'])).mean()
    assert 0.95 - _slack(0.95, seeds) <= inside <= 0.95 + _slack(0.95, seeds)
    assert stats.kstest(realised, 'beta', args=(n + 1 - math.floor((n + 1) * alpha), math.floor((n + 1) * alpha))).statistic < 0.05


def test_a_beta_interval_on_clip_counts_is_too_narrow_where_group_counts_are_not():
    """P-118 at the wrong unit: the same procedure at clip level, stated with n = clips, misses its interval far more than 5% of the time."""
    alpha, n_groups, seeds = 0.1, 100, 600
    population_score, population_wrong, _ = draw(np.random.default_rng(30_000), 100_000, MOSTLY_WRONG)
    silent = np.sort(np.where(population_wrong, population_score, -np.inf))
    n_clips = n_groups * MOSTLY_WRONG.clips_per_group
    claimed = realised_coverage_interval(n_clips, alpha)
    realised = []
    for seed in range(seeds):
        score, wrong, _ = draw(np.random.default_rng(seed), n_groups, MOSTLY_WRONG)
        tau = conformal_threshold(np.where(wrong, score, -np.inf), alpha)
        realised.append(np.searchsorted(silent, tau, side='left') / len(silent))
    realised = np.array(realised)
    assert np.unique(realised).size > seeds / 2  # not stuck on a tie
    outside = ((realised < claimed['low']) | (realised > claimed['high'])).mean()
    assert outside > 0.05 + 3 * _slack(0.05, seeds)


def test_calibrating_on_the_validation_set_that_chose_the_model_under_covers():
    """P-114: pick the best of many equal models by validation error. The picked model's validation errors are luckily few, so a threshold from them answers too much.
    A separate calibration split reaches nominal coverage on the same picks."""
    alpha, n, models, seeds = 0.1, 99, 30, 400
    scenario = Scenario(clips_per_group=1, group_sd=0.0, score_group_sd=0.0, offset=0.0)
    # All models are the same in truth: wrong is independent of the model, so the population worst-score law is known from one large draw.
    score, wrong, _ = draw(np.random.default_rng(40_000), 400_000, scenario)
    population = np.sort(np.where(wrong, score, -np.inf))
    on_validation, on_calibration = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        candidates = [draw(rng, n, scenario) for _ in range(models)]
        best = min(candidates, key=lambda c: c[1].mean())  # model selection on validation
        tau = conformal_threshold(group_worst_scores(*best), alpha)
        on_validation.append(np.searchsorted(population, tau, side='left') / len(population))
        fresh = draw(rng, n, scenario)
        tau = conformal_threshold(group_worst_scores(*fresh), alpha)
        on_calibration.append(np.searchsorted(population, tau, side='left') / len(population))
    margin = 3 * np.std(on_calibration) / math.sqrt(seeds)
    assert np.mean(on_calibration) >= 1 - alpha - margin
    assert np.mean(on_validation) < 1 - alpha - margin

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
    # V-053 item 9: for selective_risk the bound still holds with few answered groups, so the headline says "few units", not "loose guarantee".
    assert info['status'] == 'met' and info['loose'] and info['headline'].startswith('Few answered group units') and info['effective_n'] < info['loose_below_effective_n']
    assert 'Guarantee is loose' not in info['headline'] and info['guarantee_holds']


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


# -- V-053 fix round: what the review found missing ------------------------------------------------------------------
from evaluators.abstention import (METHODS, binding_issues, budget_message, certification_budget, file_fingerprint,  # noqa: E402
                                   mapping_fingerprint)


def _bound_rule(score_name='max_prob', **info):
    score, wrong, groups = _easy()
    provenance = {'weights_sha256': 'w1', 'label_mapping_sha256': mapping_fingerprint({'cat': 0, 'dog': 1}), 'error_model_sha256': 'e1', **info}
    return fit_abstention(score, wrong, groups, target_error=0.1, score_name=score_name, provenance=provenance), score


def _apply(rule, score, **binding):
    frame = pd.DataFrame({'confidence': score, 'expected_error': 1 - score, 'predicted_label': 'cat'})
    return apply_rule(rule, frame, {}, warn=lambda _: None, **binding), frame


def test_matching_weights_mapping_and_error_model_keep_the_guarantee():
    rule, score = _bound_rule('expected_error')
    assert rule.info['guarantee_holds']
    summary, frame = _apply(rule, score, weights_sha256='w1', label_mapping={'dog': 1, 'cat': 0}, error_model_sha256='e1')  # key order does not matter
    assert summary['pipeline_mismatches'] == [] and summary['guarantee_holds'] is True and not summary['headline'].startswith('No guarantee')
    assert binding_issues(rule, 'w1', {'cat': 0, 'dog': 1}, 'e1') == []


@pytest.mark.parametrize('binding, needle', [
    ({'weights_sha256': None}, 'weights'),
    ({'label_mapping': {'cat': 1, 'dog': 0}}, 'label mapping'),  # names swapped: the decision column would name the wrong class
    ({'error_model_sha256': 'e2'}, 'error model'),
    ({'error_model_sha256': None}, 'error model'),  # the rule was fitted with an error model that is now gone
])
def test_a_binding_mismatch_voids_the_guarantee_at_apply_time(binding, needle):
    """V-053 items 4 and 5: the label mapping and error_model.json are bound to the rule like the weights. Decisions are still written; the claim goes."""
    rule, score = _bound_rule('expected_error')
    given = {'weights_sha256': 'w1', 'label_mapping': {'cat': 0, 'dog': 1}, 'error_model_sha256': 'e1', **binding}
    summary, frame = _apply(rule, score, **given)
    assert summary['guarantee_holds'] is False and summary['headline'].startswith('No guarantee')
    assert any(issue.startswith(needle) for issue in summary['pipeline_mismatches']) and 'decision' in frame.columns


def test_a_rule_with_no_recorded_binding_cannot_claim_the_guarantee():
    score, wrong, groups = _easy()
    rule = fit_abstention(score, wrong, groups, target_error=0.1, score_name='expected_error')  # no provenance at all
    issues = binding_issues(rule, 'w1', {'cat': 0}, 'e1')
    assert {issue.split(':')[0] for issue in issues} == {'weights', 'label mapping', 'error model'}
    assert _apply(rule, score, weights_sha256='w1', label_mapping={'cat': 0}, error_model_sha256='e1')[0]['guarantee_holds'] is False


@pytest.mark.parametrize('weights_in_use', ['w1', None])
def test_a_rule_with_no_weights_hash_cannot_claim_the_guarantee(weights_in_use):
    """Re-check BI3: everything else is bound and matches; only the rule's weights hash is missing. That alone must void, and say why."""
    rule, score = _bound_rule('expected_error', weights_sha256=None)
    issues = binding_issues(rule, weights_in_use, {'cat': 0, 'dog': 1}, 'e1')
    assert issues == ['weights: the rule records no weights hash, so what it was fitted on cannot be verified']
    summary, _ = _apply(rule, score, weights_sha256=weights_in_use, label_mapping={'cat': 0, 'dog': 1}, error_model_sha256='e1')
    assert summary['guarantee_holds'] is False and 'records no weights hash' in summary['headline']


def test_csv_group_ids_that_look_numeric_still_match_the_runs_parquet(tmp_path):
    """Re-check finding 1: pandas reads '001' in a CSV as 1, which never matched '001' in the run's parquet, so the overlap went unseen."""
    from evaluators.video.abstention_fit import _void_reasons, text_units
    train = _split_file(tmp_path / 'train_split.parquet', ['001', '002', '003'])
    pd.DataFrame({'labels': [0, 1], 'group': ['001', '002']}).to_csv(tmp_path / 'calib.csv', index=False)
    pd.DataFrame({'labels': [0, 1], 'group': ['004', '005']}).to_csv(tmp_path / 'new.csv', index=False)
    assert text_units(tmp_path / 'calib.csv', 'group')[0] == ['001', '002']
    assert text_units(tmp_path / 'calib.csv', None)[0] == ['001', '002']  # the default 'group' column too
    assert '2 calibration groups also appear' in _void_reasons(tmp_path / 'calib.csv', 'group', [train])[0]
    assert _void_reasons(tmp_path / 'new.csv', 'group', [train]) == []
    # Integer ids in both files still match, as text.
    _split_file(tmp_path / 'int_train.parquet', [7, 8])
    pd.DataFrame({'labels': [0], 'group': [7]}).to_csv(tmp_path / 'int.csv', index=False)
    assert _void_reasons(tmp_path / 'int.csv', 'group', [tmp_path / 'int_train.parquet'])


def test_numeric_class_names_that_differ_from_indices_are_refused_unless_encoded(tmp_path):
    """Re-check finding 2: with {'1': 0, '10': 1, '2': 2} a label 1 may be the name '1' (index 0) or index 1. Guessing would mislabel rows and keep the claim."""
    from evaluators.video.abstention_fit import ambiguous_numeric_names, integer_labels, is_run_calibration_split
    mapping = {'1': 0, '10': 1, '2': 2}
    assert ambiguous_numeric_names(mapping) and not ambiguous_numeric_names({'0': 0, '1': 1}) and not ambiguous_numeric_names({'left': 0, 'right': 1})
    for values in (pd.Series([1, 2, 2]), pd.Series(['1', '10', '2'])):
        with pytest.raises(ValueError, match='could be a name or an index'):
            integer_labels(values, mapping, 3)
    assert integer_labels(pd.Series([1, 2, 2]), mapping, 3, encoded=True).tolist() == [1, 2, 2]
    assert integer_labels(pd.Series([1, 0]), {'0': 0, '1': 1}, 2).tolist() == [1, 0]  # names equal to indices: no ambiguity
    run = tmp_path / 'deeptune-x'
    (run / 'data_splits_1').mkdir(parents=True)
    checkpoint = run / 'trainval_output_1' / 'model_weights.pth'
    assert is_run_calibration_split(run / 'data_splits_1' / 'calibration_split.parquet', checkpoint)
    assert not is_run_calibration_split(tmp_path / 'calibration_split.parquet', checkpoint)
    assert not is_run_calibration_split(run / 'data_splits_1' / 'train_split.parquet', checkpoint)
    other = tmp_path / 'deeptune-y' / 'data_splits_1' / 'calibration_split.parquet'
    assert not is_run_calibration_split(other, checkpoint)


def test_the_error_model_is_bound_only_when_it_is_the_score():
    rule, score = _bound_rule('max_prob')
    assert binding_issues(rule, 'w1', None, 'something else') == []


def test_fingerprints_are_stable_and_tell_files_apart(tmp_path):
    assert mapping_fingerprint({'a': 0, 'b': 1}) == mapping_fingerprint({'b': 1, 'a': 0}) != mapping_fingerprint({'a': 1, 'b': 0})
    assert mapping_fingerprint(None) is None and file_fingerprint(None) is None
    (tmp_path / 'x.json').write_text('{"a": 1}')
    (tmp_path / 'y.json').write_text('{"a": 2}')
    assert file_fingerprint(tmp_path / 'x.json') != file_fingerprint(tmp_path / 'y.json')


def test_integer_labels_maps_names_and_refuses_bad_labels():
    """V-053 item 3: a wrong label reads as a wrong prediction, so bad calibration labels would decide the threshold silently."""
    from evaluators.video.abstention_fit import integer_labels
    mapping = {'left': 0, 'right': 1}
    assert integer_labels(pd.Series(['right', 'left', 'right']), mapping, 2).tolist() == [1, 0, 1]
    assert integer_labels(pd.Series([1, 0, 1.0]), mapping, 2).tolist() == [1, 0, 1]
    assert integer_labels(pd.Series(['1', '0']), None, 2).tolist() == [1, 0]
    refusals = [(pd.Series([0, None]), mapping, 'missing'), (pd.Series([0, 1.5]), mapping, 'whole class'), (pd.Series(['left', 'up']), mapping, 'does not have'),
                (pd.Series([0, 2]), mapping, 'outside'), (pd.Series([-1, 0]), mapping, 'outside'), (pd.Series(['left', 'right']), None, 'no label_mapping')]
    for values, given, needle in refusals:
        with pytest.raises(ValueError, match=needle):
            integer_labels(values, given, 2)


def _split_file(path, groups, sources=None, labels=None):
    n = len(groups) if groups is not None else len(sources) if sources is not None else 2
    frame = pd.DataFrame({'labels': labels if labels is not None else [i % 2 for i in range(n)]})
    if groups is not None:
        frame['group'] = groups
    if sources is not None:
        frame['videos'] = sources
    frame.to_parquet(path, index=False)
    return path


def test_void_reasons_need_something_to_compare_unless_disjoint_by_construction(tmp_path):
    """V-053 item 1: a standalone fit with nothing to compare cannot show disjointness, so it claims nothing. deeptune.py drew the rows itself."""
    from evaluators.video.abstention_fit import _void_reasons
    calibration = _split_file(tmp_path / 'calibration.parquet', ['a', 'b'])
    assert _void_reasons(calibration, 'group', [])[0].startswith('Disjointness unverified')
    assert _void_reasons(calibration, 'group', [], disjoint_by_construction=True) == []
    clean = _split_file(tmp_path / 'train.parquet', ['c', 'd'])
    assert _void_reasons(calibration, 'group', [clean]) == []
    overlapping = _split_file(tmp_path / 'test.parquet', ['b', 'e'])
    assert '1 calibration groups also appear' in _void_reasons(calibration, 'group', [clean, overlapping])[0]
    # No group column and no source path in either file (clip bytes): nothing to compare, so unverified.
    bare = _split_file(tmp_path / 'bare.parquet', None)
    other_bare = _split_file(tmp_path / 'other_bare.parquet', None)
    assert any(reason.startswith('Disjointness unverified') for reason in _void_reasons(bare, None, [other_bare]))
    # Source paths compare even when written relative to their own file.
    (tmp_path / 'sub').mkdir()
    by_path = _split_file(tmp_path / 'by_path.parquet', None, sources=['sub/v1.mp4', 'sub/v2.mp4'])
    shares = _split_file(tmp_path / 'sub' / 'shares.parquet', None, sources=['v2.mp4', 'v3.mp4'])
    assert any('source videos also appear' in reason for reason in _void_reasons(by_path, None, [shares]))


def test_run_split_files_finds_the_runs_own_splits(tmp_path):
    from evaluators.video.abstention_fit import run_label_mapping, run_split_files
    run = tmp_path / 'deeptune-x'
    (run / 'trainval_output_1').mkdir(parents=True)
    splits = run / 'data_splits_1'
    splits.mkdir()
    for name in ('train', 'val', 'test'):
        _split_file(splits / f'{name}_split.parquet', [name])
    (splits / 'label_mapping.json').write_text('{"left": 0, "right": 1}')
    checkpoint = run / 'trainval_output_1' / 'model_weights.pth'
    assert [path.name for path in run_split_files(checkpoint)] == ['train_split.parquet', 'val_split.parquet', 'test_split.parquet']
    assert run_label_mapping(checkpoint) == {'left': 0, 'right': 1}
    lonely = tmp_path / 'lonely' / 'w' / 'model_weights.pth'
    assert run_split_files(lonely) == [] and run_label_mapping(lonely) is None


def test_carving_updates_split_summary_and_prints_the_budget(tmp_path, capsys):
    """V-053 nice-to-have: split_summary.json no longer describes training before calibration left it. And the budget is said before training."""
    from evaluators.video.abstention_fit import carve_calibration_split
    groups = np.repeat([f'g{i}' for i in range(40)], 2)
    train = _split_file(tmp_path / 'train_split.parquet', groups, labels=np.repeat(np.arange(40) % 2, 2))
    (tmp_path / 'split_summary.json').write_text(json.dumps({'sample_counts': {'train': 80, 'val': 10, 'test': 10}, 'group_counts': {'train': 40, 'val': 5, 'test': 5}}))
    calibration = carve_calibration_split(train, 0.1, 0.7, 'group', 0, target_error=0.1, delta=0.05)
    summary = json.loads((tmp_path / 'split_summary.json').read_text())
    kept, carved = pd.read_parquet(train), pd.read_parquet(calibration)
    assert summary['sample_counts'] == {'train': len(kept), 'val': 10, 'test': 10, 'calibration': len(carved)}
    assert summary['group_counts'] == {'train': kept['group'].nunique(), 'val': 5, 'test': 5, 'calibration': carved['group'].nunique()}
    assert len(kept) + len(carved) == 80 and summary['calibration_taken_from'] == 'train'
    out = capsys.readouterr().out
    assert 'Certification budget' in out and 'cannot certify target 0.1' in out  # 6 groups: below the zero-error floor of 62


def test_certification_budget_is_the_edge_of_certification_by_hand():
    alpha, n, delta = 0.05, 250, 0.05
    budget = certification_budget(alpha, n, delta)

    def p_value(k):  # Hoeffding-Bentkus, arXiv 2110.01052 Eq. 2, by hand
        r = k / n
        h = (r * math.log(r / alpha) if r > 0 else 0.0) + (1 - r) * math.log((1 - r) / (1 - alpha))
        return min(math.exp(-n * h), math.e * stats.binom.cdf(k, n, alpha))

    assert budget is not None and budget / n < alpha
    assert p_value(budget) <= delta / GRID_SIZE < p_value(budget + 1)
    assert certification_budget(0.1, 10, 0.05) is None and certification_budget(0.1, 0, 0.05) is None
    message = budget_message(alpha, n, delta, 'selective_risk')
    assert 'at most' in message and f'{budget} wrong' in message
    assert 'at least 19' in budget_message(0.05, 10, delta, 'silent_error')


def test_method_names_say_learn_then_test_and_split_conformal():
    """V-053 item 11: the default is Learn then Test risk control, not conformal prediction."""
    score, wrong, groups = _easy()
    selective = fit_abstention(score, wrong, groups, target_error=0.1).info
    silent = fit_abstention(score, wrong, groups, target_error=0.1, guarantee='silent_error').info
    assert selective['method'] == METHODS['selective_risk'] and 'Learn then Test' in selective['method'] and 'conformal' not in selective['method']
    assert selective['guarantee_text'].startswith('Learn then Test') and 'conformal' not in selective['headline'].lower()
    assert silent['method'] == METHODS['silent_error'] and silent['guarantee_text'].startswith('Split conformal')
    applied = apply_rule(AbstentionRule(selective), pd.DataFrame({'confidence': score, 'predicted_label': 'x'}), {}, warn=lambda _: None)
    assert applied['method'] == selective['method']


def test_silent_error_with_few_groups_still_says_loose():
    """Item 9 changed only the selective_risk wording: for silent_error few groups do make the coverage itself vary widely."""
    score, wrong, groups = _easy(60, scenario=Scenario(clips_per_group=2, group_sd=0.1, offset=-1))
    info = fit_abstention(score, wrong, groups, target_error=0.1, guarantee='silent_error').info
    assert info['status'] == 'met' and info['loose'] and info['headline'].startswith('Guarantee is loose')


def _big_group_errs():
    """Nine one-clip groups all right, one nine-clip group all wrong: 0.1 by group, 0.5 by clip."""
    groups = [*range(9), *[9] * 9]
    frame = pd.DataFrame({'confidence': np.full(18, .9), 'correct': [True] * 9 + [False] * 9})
    return frame, groups


def _rule_at_zero(guarantee):
    return AbstentionRule({'format': 'deeptune-abstention', 'version': 1, 'threshold': 0.0, 'score': 'max_prob', 'target_error': .2, 'guarantee_kind': guarantee,
                           'status': 'met', 'headline': 'x', 'guarantee_holds': True})


@pytest.mark.parametrize('guarantee, column', [('selective_risk', 'group_mean_error'), ('silent_error', 'share_groups_with_answered_wrong_clip')])
def test_report_and_plot_use_the_guaranteed_quantity_not_clip_weighted_error(tmp_path, monkeypatch, guarantee, column):
    """V-053 item 7: the plot drew clip-weighted error against a group-weighted target. Here clip-weighted error is 0.5, above the 0.2 target,
    while the guaranteed quantity is 0.1, within it."""
    pytest.importorskip('matplotlib')
    import evaluators.video.report as report_mod
    drawn = {}
    finish = report_mod._finish

    def spy(figure, axes, *rest, **kw):
        drawn.update(axes=axes, ylabel=rest[2])
        return finish(figure, axes, *rest, **kw)

    monkeypatch.setattr(report_mod, '_finish', spy)
    frame, groups = _big_group_errs()
    report, curve = build_report(_rule_at_zero(guarantee), frame, groups, n_boot=20)
    chosen = report['at_fitted_threshold']
    assert chosen['error'] == pytest.approx(0.5) and chosen[column] == pytest.approx(0.1)
    assert chosen['guaranteed_quantity'] == column and chosen['point_within_target'] is True
    write_report(report, curve, tmp_path)
    axes = drawn['axes']
    marker = [c for c in axes.collections if c.get_label() == 'fitted threshold'][0]
    assert marker.get_offsets()[0][1] == pytest.approx(0.1)
    plotted = curve[curve['answered_clips'] > 0].sort_values('coverage')  # the plot leaves out thresholds that answer nothing
    labels = [line.get_label() for line in axes.get_lines()]
    guaranteed = [line for line in axes.get_lines() if '(guaranteed)' in line.get_label() or line.get_label().startswith('share of groups')]
    assert len(guaranteed) == 1, labels
    assert np.asarray(guaranteed[0].get_ydata(), dtype=float) == pytest.approx(plotted[column].to_numpy(dtype=float))
    if guarantee == 'silent_error':
        assert 'Share of groups' in drawn['ylabel'] and not any('each answered clip' in label for label in labels)


# -- V-053 item 8: validity through the public fit_abstention, and tight enough to catch a wrong delta ----------------
def _saturated_false_claim_rate(n, alpha, true_risk, delta, seeds, seed=0):
    """Every score is 1, above the whole grid, so every threshold answers every clip. Clips are i.i.d. with error true_risk > alpha, so any
    answer is a false claim. Returns the realised rate through fit_abstention and the exact rate the procedure at `delta` implies:
    P(binomial tail <= delta / grid) for K ~ Bin(n, true_risk), from scipy alone."""
    wrong = np.random.default_rng(seed).random((seeds, n)) < true_risk
    claims = sum(fit_abstention(np.ones(n), row, target_error=alpha, delta=delta).threshold is not None for row in wrong)
    k = np.arange(n + 1)
    passing = k[stats.binom.cdf(k, n, alpha) <= delta / GRID_SIZE]
    exact = float(stats.binom.cdf(passing.max(), n, true_risk)) if len(passing) else 0.0
    return claims / seeds, exact


def test_fit_abstention_false_claims_match_the_exact_rate_at_the_boundary():
    """Tightness, not only validity: the realised false-claim rate equals what the exact procedure at delta implies, within 3 standard errors.
    A fit that used delta / 8 (exact rate 0.0014) or delta / 2 (0.0070) claims far less often than 0.025, and fails here. Bonferroni's slack is out of play
    since every threshold answers the same clips. n = 240 puts a binomial atom close to delta / 32, so discreteness costs little."""
    n, alpha, true_risk, delta, seeds = 240, 0.1, 0.101, 0.9, 1500
    rate, exact = _saturated_false_claim_rate(n, alpha, true_risk, delta, seeds)
    assert exact > 0.75 * delta / GRID_SIZE  # near the boundary: the bound is not vacuous here
    assert rate <= delta
    assert abs(rate - exact) <= 3 * math.sqrt(exact * (1 - exact) / seeds)


@pytest.mark.seed_study
def test_fit_abstention_false_claims_match_the_exact_rate_at_the_boundary_long():
    """The same with 8000 other draws: 3 standard errors are then 0.005."""
    n, alpha, true_risk, delta, seeds = 240, 0.1, 0.101, 0.9, 8000
    rate, exact = _saturated_false_claim_rate(n, alpha, true_risk, delta, seeds, seed=1)
    assert abs(rate - exact) <= 3 * math.sqrt(exact * (1 - exact) / seeds)


@pytest.mark.parametrize('guarantee', ['selective_risk', 'silent_error'])
def test_the_grid_is_fixed_whatever_the_scores(guarantee):
    """A grid derived from the calibration scores (quantiles, or dropping thresholds above the top score) breaks the fixed-family Bonferroni argument.
    The fit table must hold the same thresholds, all of them, for any data."""
    low = _easy(300, seed=0, scenario=Scenario(clips_per_group=2, score_mean=-2.0, score_clip_sd=0.3))
    high = _easy(300, seed=1, scenario=Scenario(clips_per_group=2, score_mean=6.0, score_clip_sd=0.3))
    assert low[0].max() < 0.5 < high[0].min()
    fits = [fit_abstention(*data, target_error=0.2, guarantee=guarantee, num_classes=5).info for data in (low, high)]
    tables = [[row['threshold'] for row in info['fit_table']] for info in fits]
    assert tables[0] == tables[1] == threshold_grid(5).tolist()
    assert fits[0]['grid_size'] == fits[1]['grid_size'] == GRID_SIZE


def _public_failures(scenario, truth, n_groups, seeds, per_clip=False):
    """As _failures, through fit_abstention: groups=None makes every clip its own unit."""
    failed, chosen = 0, 0
    for seed in range(seeds):
        score, wrong, groups = draw(np.random.default_rng(seed), n_groups, scenario)
        tau = fit_abstention(score, wrong, None if per_clip else groups, target_error=ALPHA, delta=DELTA).threshold
        if tau is not None:
            chosen += 1
            failed += truth['group'][np.searchsorted(GRID, tau)] > ALPHA
    return failed / seeds, chosen / seeds


@pytest.mark.parametrize('name, scenario, n_groups, seeds', [('independent', INDEPENDENT, 1500, 60), ('dependent', DEPENDENT, 600, 120)])
def test_fit_abstention_selective_risk_holds_and_answers(populations, name, scenario, n_groups, seeds):
    assert populations[name]['group'][0] > ALPHA
    rate, chosen = _public_failures(scenario, populations[name], n_groups, seeds)
    assert rate <= DELTA + _slack(DELTA, seeds) and chosen > 0.9


@pytest.mark.seed_study
def test_fit_abstention_selective_risk_study(populations):
    """The validity checks above at full size through the public fit, and the clip-counting failure."""
    for name, scenario, n_groups, seeds in (('independent', INDEPENDENT, 1500, 400), ('dependent', DEPENDENT, 600, 400)):
        rate, chosen = _public_failures(scenario, populations[name], n_groups, seeds)
        assert rate <= DELTA + _slack(DELTA, seeds) and chosen > 0.9
    seeds = 800
    truth = {'group': np.full(len(GRID), populations['stress']['group'][0])}
    grouped, _ = _public_failures(STRESS, truth, 60, seeds)
    naive, _ = _public_failures(STRESS, truth, 60, seeds, per_clip=True)
    assert grouped <= DELTA + _slack(DELTA, seeds) and naive > DELTA + _slack(DELTA, seeds)


def _public_silent_coverage(n, alpha, seeds, truth_score, truth_wrong, truth_groups):
    worst_population = np.sort(group_worst_scores(truth_score, truth_wrong, truth_groups))
    realised = []
    for seed in range(seeds):
        score, wrong, groups = draw(np.random.default_rng(seed), n, NEARLY_ALL_GROUPS_ERR)
        tau = fit_abstention(score, wrong, groups, target_error=alpha, guarantee='silent_error').threshold
        realised.append(np.searchsorted(worst_population, tau, side='left') / len(worst_population))
    return np.array(realised)


def test_fit_abstention_silent_error_coverage_is_nominal(silent_population):
    n, alpha, seeds = 99, 0.1, 400
    realised = _public_silent_coverage(n, alpha, seeds, *silent_population)
    standard_error = realised.std() / math.sqrt(seeds)
    assert 1 - alpha - 3 * standard_error <= realised.mean() <= 1 - alpha + 1 / (n + 1) + 3 * standard_error


@pytest.mark.seed_study
def test_fit_abstention_silent_error_study(silent_population):
    n, alpha, seeds = 99, 0.1, 1500
    realised = _public_silent_coverage(n, alpha, seeds, *silent_population)
    standard_error = realised.std() / math.sqrt(seeds)
    assert 1 - alpha - 3 * standard_error <= realised.mean() <= 1 - alpha + 1 / (n + 1) + 3 * standard_error
    interval = realised_coverage_interval(n, alpha)
    inside = ((realised >= interval['low']) & (realised <= interval['high'])).mean()
    assert abs(inside - 0.95) <= _slack(0.95, seeds)


def test_run_label_mapping_uses_the_recorded_file_and_hash(tmp_path):
    """Item 6 (with R1): a run that recorded label_mapping_path and label_mapping_sha256 gets the file matching that hash, wherever it is;
    a changed file is refused, not read. Runs from before the record still use the data_splits_* folder."""
    from evaluators.video.abstention_fit import run_label_mapping
    from inference.common import label_mapping_digest
    mapping = {'left': 0, 'right': 1}
    elsewhere = tmp_path / 'kept' / 'label_mapping.json'
    elsewhere.parent.mkdir()
    elsewhere.write_text(json.dumps(mapping))
    run = tmp_path / 'deeptune-x'
    (run / 'trainval_output_1').mkdir(parents=True)
    (run / 'data_splits_1').mkdir()
    (run / 'data_splits_1' / 'label_mapping.json').write_text(json.dumps({'left': 1, 'right': 0}))  # nearby but not the run's: must not win
    checkpoint = run / 'trainval_output_1' / 'model_weights.pth'
    recorded = {'label_mapping_path': str(elsewhere), 'label_mapping_sha256': label_mapping_digest(mapping)}
    assert run_label_mapping(checkpoint, recorded) == mapping
    (run / 'trainval_output_1' / 'cli_arguments.json').write_text(json.dumps(recorded))
    assert run_label_mapping(checkpoint) == mapping  # read from cli_arguments.json beside the weights
    elsewhere.write_text(json.dumps({'left': 1, 'right': 0}))
    with pytest.raises(ValueError, match="Cannot verify this run's label mapping"):
        run_label_mapping(checkpoint, recorded)
    assert run_label_mapping(checkpoint, {}) == {'left': 1, 'right': 0}  # no record: the old search

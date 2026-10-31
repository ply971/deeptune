"""Abstention ("not sure") with a target error rate, calibrated on whole groups. Torch-free.

The model answers a clip when its score reaches a threshold fitted on a calibration split, and says "not sure"
otherwise. Nothing here promises a fixed accuracy: the model's accuracy decides how much it can answer, and that
can be nothing at all.

The unit
    Clips from one group (one source video, one UCF101 group) are dependent, so the group is the unit everywhere here.
    Calibration groups must be disjoint from train, validation and test. Validation chose the weights once early stopping
    exists, so its scores are not exchangeable with new data and it cannot calibrate (P-104, P-114, V-011).

Two guarantees, chosen by `guarantee`
    'selective_risk' (default). Assumes calibration groups and new groups are i.i.d. from one population, given the fitted model.
        For a threshold t, draw a new group that has at least one answered clip, then one of its answered clips uniformly.
        R(t) is the probability that this clip is wrong. The rule guarantees, over the draw of the calibration groups,

            P( R(t_hat) <= target_error ) >= 1 - delta.

        This is a high-probability bound on selective risk, and it is group-weighted: every answered group counts once.
        Method: Learn then Test (Angelopoulos, Bates, Candes, Jordan, Lei, arXiv 2110.01052). One hypothesis R(t) > target for
        each threshold on a grid fixed before seeing data; the Hoeffding-Bentkus p-value (their Prop. 1, Eq. 2) on the per-group
        losses; Bonferroni over the grid (their Prop. 2), which makes every rejected threshold valid together (their Thm. 1), so R(t)
        need not be monotone. Selective risk with abstention follows their Sec. 3.2. Here the mean is over the N(t) answered groups:
        given N(t) they are i.i.d. draws from the conditional law, so the p-value is valid given N(t) and so without it. The per-group
        loss is in [0, 1] whatever the dependence inside the group, which is why the clips need no independence.
        With one clip per unit the loss is a Bernoulli and the exact binomial tail is used, with no factor e. The risk-controlled
        rejection idea goes back to SGR (Geifman & El-Yaniv, NeurIPS 2017, arXiv 1705.08500). Its binary search is not used.

    'silent_error'. Assumes only that calibration groups and a new group are exchangeable, given the fitted model.
        A group's score a_g is the highest score among its wrong clips (minus infinity when none is wrong). The threshold sits just
        above the k-th smallest a_g, k = ceil((n + 1)(1 - target)): split conformal on one score per group. Then, marginally
        over the calibration groups and the new group,

            P( the new group has at least one answered wrong clip ) <= target_error.

        This says nothing about P(wrong | answered): a rule that answers 10% of clips can be wrong on half of those. It is conformal in
        the exact sense and is what the Beta statement below applies to. Reduction to one score per group is in the spirit of Dunn,
        Wasserman & Ramdas (JASA, arXiv 1809.07441), who study prediction sets for such two-layer data. Their subsampling and CDF pooling
        are not what is implemented.

Realised coverage (P-118)
    Coverage given one calibration set is random. For 'silent_error' it is Beta(n + 1 - l, l), l = floor((n + 1) * target) (Angelopoulos &
    Bates, arXiv 2107.07511, Sec. 3.2), with n = groups, exact when the a_g have no ties. Ties make the true coverage larger, so then
    only the lower end holds. For 'selective_risk' t_hat is picked from the grid rather than by rank, so that form does not apply. The
    share of new groups without an answered wrong clip gets a Clopper-Pearson interval (Beta quantiles) on n groups, Bonferroni-corrected over
    the grid so that it holds for the picked threshold.

What is not guaranteed
    Accuracy on any one clip, group or class. Per-class error. Clip-weighted error when error depends on group size (reported, not
    guaranteed). New clips from a group seen in train or calibration. Any distribution shift. Wrong labels in calibration. Any other
    `test_clips`, `test_crops`, sampling or weights than the fit's. The score need not be calibrated, and no temperature is assumed.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from evaluators.stats import bootstrap_ci, unit_labels

FORMAT = 'deeptune-abstention'
FORMAT_VERSION = 1
ABSTENTION_FILENAME = 'abstention.json'
FIT_TABLE_FILENAME = 'abstention_fit_table.csv'
REPORT_FILENAME = 'abstention_report.json'
CURVE_FILENAME = 'selective_risk_curve.csv'
PLOT_FILENAME = 'selective_risk_curve.png'
INFERENCE_SUMMARY_FILENAME = 'abstention_summary.json'
NOT_SURE = 'not sure'

GUARANTEES = ('selective_risk', 'silent_error')
SCORES = ('max_prob', 'expected_error')
DEFAULT_DELTA = 0.05
DEFAULT_CALIB_FRACTION = 0.1
GRID_SIZE = 32
SMALLEST_DISTANCE_FROM_ONE = 1e-6  # float32 softmax cannot resolve much closer to 1
INTERVAL_LEVEL = 0.95
LOOSE_FACTOR = 1.5  # "loose" when the realised miss rate can plausibly reach 1.5 x target: see loose_below
REFUSED_SPLITS = ('test', 'val', 'validation', 'train', 'training')

# The unit's level is said in every output. Without a group column nothing can detect clips cut from one recording (P-105, a later extension).
GUARANTEE_LEVELS = {'group': 'group', 'source_video': 'source video', 'row': 'clip'}
UNIT_NOTES = {
    'row': 'No group column and no repeated source video: this guarantee is at clip level and assumes independent clips. If clips were cut from one recording '
           '(pre-cut segments in separate files look like this) it does not hold, and nothing here can detect that. Add a group column naming each clip\'s source.',
    'source_video': 'No group column: clips that name the same source video are treated as one group. Clips cut from one recording into separate files would still count as '
                    'independent, and nothing here can detect that. Add a group column naming each clip\'s source.',
}

NOT_GUARANTEED = [
    'accuracy on any single clip, group or class',
    'per-class error: one class can be far above the target while the average holds',
    'clip-weighted error when error depends on group size (reported, not guaranteed)',
    'new clips from a group seen in training or calibration',
    'any distribution shift between the calibration groups and new data',
    'calibration data with wrong labels',
    'a different number of test clips or crops, other sampling, or other weights than the fit used',
]
SOURCES = {
    'selective_risk': 'Learn then Test, Angelopoulos, Bates, Candes, Jordan, Lei, arXiv 2110.01052: Hoeffding-Bentkus p-value (Prop. 1), Bonferroni (Prop. 2), Thm. 1, Sec. 3.2. Applied to per-group losses.',
    'silent_error': 'Split conformal on one score per group. Realised coverage: Angelopoulos & Bates, arXiv 2107.07511, Sec. 3.2.',
}


# -- grid -----------------------------------------------------------------------------------------------------------
def threshold_grid(num_classes: int = 2, size: int = GRID_SIZE) -> np.ndarray:
    """Ascending thresholds, fixed before any data is seen. Evenly spaced in log(1 - t), since softmax scores crowd near 1.
    The first one, 0, answers everything. A data-chosen grid would break the Bonferroni argument."""
    if size < 2:
        raise ValueError('The grid needs at least two thresholds.')
    largest = max(1.0 - 1.0 / max(int(num_classes), 2), 0.5)
    distances = np.geomspace(largest, SMALLEST_DISTANCE_FROM_ONE, size - 1)
    return np.concatenate([[0.0], 1.0 - distances])


# -- per-group statistics -------------------------------------------------------------------------------------------
def _clean(score, wrong, units) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    wrong = np.asarray(wrong, dtype=bool).reshape(-1)
    if len(score) == 0:
        raise ValueError('Abstention needs at least one calibration clip.')
    if len(wrong) != len(score) or len(units) != len(score):
        raise ValueError(f'{len(score)} scores, {len(wrong)} outcomes and {len(units)} units: they must match.')
    if not np.isfinite(score).all():
        raise ValueError('Abstention scores must be finite.')
    return score, wrong, pd.factorize(pd.Series(list(units)))[0]


def threshold_stats(score, wrong, units, grid) -> pd.DataFrame:
    """One row per threshold: answered clips and groups, and the sum of per-group losses (the mean wrong share among a group's answered clips)."""
    score, wrong, index = _clean(score, wrong, units)
    n_units = int(index.max()) + 1
    rows = []
    for tau in np.asarray(grid, dtype=np.float64):
        answered = score >= tau
        clips = np.bincount(index[answered], minlength=n_units)
        errors = np.bincount(index[answered & wrong], minlength=n_units)
        has = clips > 0
        rows.append((tau, int(has.sum()), float((errors[has] / clips[has]).sum()), int(clips.sum()), int(errors.sum()),
                     int((errors == 0).sum())))
    return pd.DataFrame(rows, columns=['threshold', 'answered_units', 'unit_loss_sum', 'answered_clips', 'wrong_clips', 'units_without_silent_error'])


def _kl(a: np.ndarray, b: float) -> np.ndarray:
    a = np.clip(a, 1e-300, 1 - 1e-16)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def risk_p_values(answered_units, loss_sum, alpha: float, single_clip_units: bool = False) -> np.ndarray:
    """p-values for H0: R > alpha. Hoeffding-Bentkus for losses in [0, 1]; the exact binomial tail when each unit is one clip, as the loss is then a
    Bernoulli. No answered unit gives p = 1. Valid for any dependence inside a unit."""
    from scipy import stats
    n = np.atleast_1d(np.asarray(answered_units, dtype=np.int64))
    total = np.atleast_1d(np.asarray(loss_sum, dtype=np.float64))
    k = np.ceil(np.round(total, 9))  # rounding stops 3.0000000001 from counting as 4 errors
    p = np.ones(len(n))
    some = n > 0
    if not some.any():
        return p
    tail = stats.binom.cdf(k[some], n[some], alpha)
    if single_clip_units:
        p[some] = tail
    else:
        hoeffding = np.exp(-n[some] * _kl(np.minimum(total[some] / n[some], alpha), alpha))
        p[some] = np.minimum(hoeffding, math.e * tail)
    return np.minimum(p, 1.0)


def risk_upper_bound(answered_units: int, loss_sum: float, delta: float, single_clip_units: bool = False) -> float:
    """The smallest alpha that the p-value rejects at level `delta`: an upper confidence bound on R. 1.0 with no answered unit."""
    if answered_units <= 0:
        return 1.0
    low, high = float(np.clip(loss_sum / answered_units, 0.0, 1.0)), 1.0
    if low >= 1.0:
        return 1.0
    if low > 0 and risk_p_values(answered_units, loss_sum, low, single_clip_units)[0] <= delta:
        return low
    if low == 0:
        low = 1e-12
    for _ in range(50):
        middle = (low + high) / 2
        if middle >= 1.0:
            break
        if risk_p_values(answered_units, loss_sum, middle, single_clip_units)[0] <= delta:
            high = middle
        else:
            low = middle
    return float(high)


def select_selective_threshold(stats_table: pd.DataFrame, alpha: float, delta: float, single_clip_units: bool = False) -> Optional[float]:
    """The lowest (most answering) threshold rejected at delta / grid size, else None."""
    p = risk_p_values(stats_table['answered_units'].to_numpy(), stats_table['unit_loss_sum'].to_numpy(), alpha, single_clip_units)
    passed = stats_table['threshold'].to_numpy()[p <= delta / len(stats_table)]
    return float(passed.min()) if len(passed) else None


def group_worst_scores(score, wrong, units) -> np.ndarray:
    """One score per group: the highest score among its wrong clips, -inf when it has none."""
    score, wrong, index = _clean(score, wrong, units)
    worst = np.full(int(index.max()) + 1, -np.inf)
    np.maximum.at(worst, index[wrong], score[wrong])
    return worst


def conformal_threshold(worst: np.ndarray, alpha: float) -> Optional[float]:
    """Threshold for 'silent_error', or None when (n + 1) * alpha < 1: then n groups cannot certify this target. Clips answer when score >= it."""
    worst = np.sort(np.asarray(worst, dtype=np.float64))
    n = len(worst)
    rank = conformal_rank(n, alpha)
    if rank > n:
        return None
    value = worst[rank - 1]
    return 0.0 if np.isneginf(value) else float(np.nextafter(value, np.inf))


def conformal_rank(n: int, alpha: float) -> int:
    return n + 1 - math.floor((n + 1) * alpha + 1e-12)


# -- how far one calibration set can land from the target -----------------------------------------------------------
def realised_coverage_interval(n: int, alpha: float, level: float = INTERVAL_LEVEL) -> dict:
    """Central interval of the coverage given one calibration set: Beta(n + 1 - l, l). None when l = 0, where no threshold exists."""
    from scipy import stats
    l = math.floor((n + 1) * alpha + 1e-12)
    out = {'distribution': f'Beta({n + 1 - l}, {l})', 'n_effective': int(n), 'level': level, 'low': None, 'high': None}
    if l >= 1:
        tail = (1 - level) / 2
        out.update(low=float(stats.beta.ppf(tail, n + 1 - l, l)), high=float(stats.beta.ppf(1 - tail, n + 1 - l, l)))
    return out


def clopper_pearson(k: int, n: int, level: float = INTERVAL_LEVEL) -> tuple[Optional[float], Optional[float]]:
    """Exact binomial interval, whose ends are Beta quantiles."""
    from scipy import stats
    if n <= 0:
        return None, None
    tail = (1 - level) / 2
    low = 0.0 if k <= 0 else float(stats.beta.ppf(tail, k, n - k + 1))
    high = 1.0 if k >= n else float(stats.beta.ppf(1 - tail, k + 1, n - k))
    return low, high


@lru_cache(maxsize=64)
def loose_below(alpha: float, level: float = INTERVAL_LEVEL, factor: float = LOOSE_FACTOR) -> int:
    """The effective n below which the output says "guarantee is loose".

    The rule: with fewer groups, the upper end of the central `level` interval of the realised miss rate, Beta(l, n + 1 - l), is above
    `factor` x target, so one calibration draw can plausibly miss half as much again as the plain target says. Tied to the target because
    the same n is ample at 20% and hopeless at 1%. At 95%, flooring included: 70 at 20%, 160 at 10%, 342 at 5%, 903 at 2%, 1810 at 1%.
    The smallest n from which the rule holds for every larger n, as flooring l makes it wiggle."""
    from scipy import stats
    limit = int(min(400_000, 40 / alpha))
    n = np.arange(1, limit + 1)
    l = np.floor((n + 1) * alpha + 1e-12)
    upper = np.ones(limit)
    valid = l >= 1
    upper[valid] = stats.beta.ppf(1 - (1 - level) / 2, l[valid], (n + 1 - l)[valid])
    failing = np.nonzero(upper > factor * alpha)[0]
    return int(n[failing[-1]] + 1) if len(failing) else 1


def groups_needed(alpha: float, delta: float, grid_size: int = GRID_SIZE) -> int:
    """Answered groups the selective guarantee needs at the very least, with none of them wrong at all."""
    return int(math.ceil(math.log(delta / grid_size) / math.log1p(-alpha)))  # with no loss, the Hoeffding term and the exact tail are both (1 - alpha)^n


# -- the fitted rule ------------------------------------------------------------------------------------------------
class AbstentionRule:
    """A fitted threshold and everything needed to say what it means. Stored as plain JSON."""

    def __init__(self, info: dict):
        self.info = info

    @property
    def threshold(self) -> Optional[float]:
        return self.info['threshold']

    @property
    def score_name(self) -> str:
        return self.info['score']

    def answers(self, score) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64)
        if self.threshold is None:
            return np.zeros(score.shape, dtype=bool)
        return score >= self.threshold

    def to_json(self) -> str:
        return json.dumps(self.info, indent=2)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding='utf-8')
        return path

    @classmethod
    def from_json(cls, text: str) -> 'AbstentionRule':
        info = json.loads(text)
        if not isinstance(info, dict) or info.get('format') != FORMAT:
            raise ValueError(f'Not a {FORMAT} file.')
        if int(info.get('version', 0)) > FORMAT_VERSION:
            raise ValueError(f"Abstention file version {info.get('version')} is newer than this code understands ({FORMAT_VERSION}).")
        return cls(info)

    @classmethod
    def load(cls, path) -> 'AbstentionRule':
        return cls.from_json(Path(path).read_text(encoding='utf-8'))


def pipeline_record(video_options, num_frames, pooling) -> dict:
    """Settings that change the scores, at fit and at inference. Workers and cache folders do not change them, so they are left out."""
    record = {'num_frames': num_frames, 'pooling': pooling}
    for name in ('test_clips', 'test_crops', 'temporal_sampling', 'clip_stride', 'clip_duration_sec', 'motion_source', 'clip_motion'):
        record[name] = getattr(video_options, name, None)
    resize = getattr(video_options, 'frame_resize', None)
    if resize is None:  # what the pixels actually went through
        resize = getattr(video_options, 'frame_cache_size', 0) if getattr(video_options, 'frame_cache', None) else 0
    record['frame_resize'] = resize
    return record


def score_values(frame: pd.DataFrame, score_name: str) -> np.ndarray:
    """Higher means answer sooner. expected_error is A4's stated error (error_model.json, fitted on validation, so independent of calibration).
    Isotonic expected_error is a stepwise function of max_prob, so it ranks no better until extra features exist."""
    if score_name == 'max_prob':
        return frame['confidence'].to_numpy(dtype=np.float64)
    if score_name == 'expected_error':
        if 'expected_error' not in frame.columns:
            raise ValueError("Score 'expected_error' needs error_model.json for these weights; none was applied. Use max_prob or retrain without --no_error_model.")
        return 1.0 - frame['expected_error'].to_numpy(dtype=np.float64)
    raise ValueError(f'Unknown abstention score {score_name!r}; choose from {SCORES}.')


def fit_abstention(score, wrong, groups=None, sources=None, *, target_error: float, delta: float = DEFAULT_DELTA, guarantee: str = 'selective_risk',
                   num_classes: int = 2, score_name: str = 'max_prob', split: str = 'calibration', provenance: Optional[dict] = None,
                   void_reasons: Sequence[str] = ()) -> AbstentionRule:
    """Fit on a group-disjoint calibration split. `groups` or `sources` name each clip's unit; with neither, every clip counts as its own unit, which
    assumes clips are independent and is said in the output. Refuses a split named like train, validation or test."""
    if not 0 < target_error < 1:
        raise ValueError(f'The target error must be between 0 and 1, got {target_error}.')
    if not 0 < delta < 1:
        raise ValueError(f'delta must be between 0 and 1, got {delta}.')
    if guarantee not in GUARANTEES:
        raise ValueError(f'Unknown guarantee {guarantee!r}; choose from {GUARANTEES}.')
    if str(split).lower() in REFUSED_SPLITS:
        raise ValueError(f"Refusing to calibrate on the {split!r} split. Training rows fitted the model, validation chose its weights, and test rows must stay sealed; "
                         'use a separate, group-disjoint calibration split (--calib_fraction).')
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    units, unit = unit_labels(len(score), groups, sources)
    score, wrong, index = _clean(score, wrong, units)
    n_units = int(index.max()) + 1
    single_clip = int(np.bincount(index).max()) == 1
    grid = threshold_grid(num_classes)
    table = threshold_stats(score, wrong, index, grid)
    bound_delta = delta / len(grid)
    table['simultaneous_upper_bound'] = [risk_upper_bound(n, s, bound_delta, single_clip) for n, s in
                                         zip(table['answered_units'], table['unit_loss_sum'])]
    table['group_mean_error'] = (table['unit_loss_sum'] / table['answered_units'].replace(0, np.nan))
    table['clip_error'] = (table['wrong_clips'] / table['answered_clips'].replace(0, np.nan))
    table['clip_coverage'] = table['answered_clips'] / len(score)
    table['p_value'] = risk_p_values(table['answered_units'].to_numpy(), table['unit_loss_sum'].to_numpy(), target_error, single_clip)

    worst = group_worst_scores(score, wrong, index)
    if guarantee == 'selective_risk':
        threshold = select_selective_threshold(table, target_error, delta, single_clip)
    else:
        threshold = conformal_threshold(worst, target_error)

    warnings = [UNIT_NOTES[unit]] if unit in UNIT_NOTES else []
    if len(np.unique(score)) == 1:
        warnings.append(f"Every calibration clip has the same {score_name}, so the rule answers all clips or none. With expected_error this means error_model.json is the constant fallback.")
    limit = loose_below(target_error)
    info = {'format': FORMAT, 'version': FORMAT_VERSION, 'target_error': float(target_error), 'guarantee_kind': guarantee, 'score': score_name,
            'threshold': threshold, 'split': split, 'unit': unit, 'guarantee_level': GUARANTEE_LEVELS[unit], 'assumes_independent_clips': unit == 'row',
            'n_units': n_units, 'n_clips': int(len(score)),
            'largest_group': int(np.bincount(index).max()), 'grid_size': len(grid), 'source': SOURCES[guarantee],
            'probabilities': 'uncalibrated softmax, averaged over test clips and crops; no temperature applied at fit or at inference',
            'temperature_applied': False, 'not_guaranteed': list(NOT_GUARANTEED), 'warnings': warnings, 'void_reasons': list(void_reasons),
            'num_classes': int(num_classes), 'loose_below_effective_n': limit, 'calibration_error_summary': {
                'clip_error_answering_everything': float(wrong.mean()), 'groups_with_an_error': int(np.isfinite(worst).sum())}}
    if guarantee == 'selective_risk':
        info['delta'] = float(delta)
    info.update(provenance or {})

    if threshold is None:
        info.update(_not_met(target_error, delta, guarantee, table, n_units, single_clip))
    elif guarantee == 'selective_risk':
        info.update(_selective_statement(target_error, delta, threshold, table, n_units, limit))
    else:
        info.update(_silent_statement(target_error, threshold, score, index, worst, n_units, limit))
    info['guarantee_holds'] = threshold is not None and not void_reasons
    info['headline'] = _headline(info)
    info['fit_table'] = [{k: _json_number(v) for k, v in row.items()} for row in table.to_dict('records')]
    return AbstentionRule(info)


def _json_number(value):
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _not_met(alpha, delta, guarantee, table, n_units, single_clip) -> dict:
    if guarantee == 'selective_risk':
        floor = float(table['simultaneous_upper_bound'].min())
        needed = groups_needed(alpha, delta, len(table))
        why = (f'No threshold could be shown to have error at or below {alpha:g} on {n_units} groups. The lowest target these data can support is about {floor:.3g}; '
               f'even with no wrong clip at all, at least {needed} answered groups are needed.')
        extra = {'lowest_supported_target': floor, 'groups_needed_at_least': needed}
    else:
        needed = math.ceil(1 / alpha) - 1
        why = f'{n_units} groups cannot certify {alpha:g}: split conformal needs at least {needed} groups.'
        extra = {'groups_needed_at_least': needed}
    return {'status': 'target_not_met', 'effective_n': n_units, 'loose': True, 'answer_rate_on_calibration': 0.0, 'realised_coverage': None,
            'guarantee_text': f'None. The rule abstains on everything, as {why}', 'reason': why, **extra}


def _selective_statement(alpha, delta, threshold, table, n_units, limit) -> dict:
    row = table.loc[table['threshold'] == threshold].iloc[0]
    answered = int(row['answered_units'])
    level = 1 - (1 - INTERVAL_LEVEL) / len(table)  # holds together over the grid, so for the picked threshold
    low, high = clopper_pearson(int(row['units_without_silent_error']), n_units, level)
    answer_low, answer_high = clopper_pearson(answered, n_units, level)
    text = (f'With probability at least {1 - delta:g} over the draw of the calibration groups, the error rate among answered clips is at most {alpha:g}, for new groups '
            'i.i.d. with the calibration groups: draw a new group with an answered clip, then one of its answered clips. Group-weighted, and not per class or under shift.')
    return {'status': 'met', 'effective_n': answered, 'loose': answered < limit, 'guarantee_text': text,
            'risk_upper_bound': float(row['simultaneous_upper_bound']), 'answer_rate_on_calibration': float(row['clip_coverage']),
            'group_answer_rate': {'n_effective': n_units, 'point': answered / n_units, 'low': answer_low, 'high': answer_high, 'level': level,
                                  'method': 'Clopper-Pearson, Bonferroni over the grid'},
            'realised_coverage': {'quantity': 'share of new groups with no answered wrong clip', 'n_effective': n_units,
                                  'point': float(row['units_without_silent_error']) / n_units, 'low': low, 'high': high, 'level': level,
                                  'method': 'Clopper-Pearson (Beta quantiles) on groups, Bonferroni over the grid, since the threshold is picked from the grid',
                                  'exact_beta_form': False}}


def _silent_statement(alpha, threshold, score, index, worst, n_units, limit) -> dict:
    interval = realised_coverage_interval(n_units, alpha)
    finite = worst[np.isfinite(worst)]
    ties = len(finite) < n_units or len(np.unique(finite)) < len(finite)
    text = (f'Marginally over the calibration groups and a new group exchangeable with them, the chance that the new group has any answered wrong clip is at most {alpha:g}. '
            'This is not an error rate among answered clips.')
    return {'status': 'met', 'effective_n': n_units, 'loose': n_units < limit, 'guarantee_text': text,
            'answer_rate_on_calibration': float((score >= threshold).mean()),
            'realised_coverage': {'quantity': 'share of new groups with no answered wrong clip, given this calibration set', **interval,
                                  'method': 'Beta distribution of split-conformal coverage on one score per group', 'exact_beta_form': not ties,
                                  'note': ('Tied group scores (several groups with no wrong clip count as ties): the true coverage is at least this, so only the lower end holds.'
                                           if ties else 'Group scores have no ties, so the form is exact.')}}


def _headline(info: dict) -> str:
    if info['status'] == 'target_not_met':
        return f"Target error {info['target_error']:g} cannot be met on this calibration data, so every clip gets \"{NOT_SURE}\". {info['reason']}"
    if info['void_reasons']:
        return 'No guarantee: ' + ' '.join(info['void_reasons'])
    limit = info['loose_below_effective_n']
    note = f" {UNIT_NOTES[info['unit']]}" if info['unit'] in UNIT_NOTES else ''
    level = info['guarantee_level']
    if info['loose']:
        return (f"Guarantee is loose: target error {info['target_error']:g} rests on only {info['effective_n']} effective {level} units (fewer than {limit}). "
                f"One calibration draw can land far from the target. {info['guarantee_text']}{note}")
    return f"Target error {info['target_error']:g} on {info['effective_n']} effective {level} units. {info['guarantee_text']}{note}"


# -- applying a rule to new clips -----------------------------------------------------------------------------------
def find_rule(checkpoint_path) -> Optional[Path]:
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    candidate = (checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent) / ABSTENTION_FILENAME
    return candidate if candidate.is_file() else None


def load_rule_near(checkpoint_path, num_classes: Optional[int] = None, weights_sha256: Optional[str] = None, warn=print) -> Optional[AbstentionRule]:
    """The run's rule. Unreadable, for other weights or for another class count: warn and return None, as a threshold of other weights means nothing."""
    path = find_rule(checkpoint_path)
    if path is None:
        return None
    try:
        rule = AbstentionRule.load(path)
    except (OSError, ValueError, KeyError) as exc:
        warn(f'Warning: ignoring {path}: {exc}. No abstention is applied.')
        return None
    fitted = rule.info.get('weights_sha256')
    if weights_sha256 is not None and fitted and fitted != weights_sha256:
        warn(f'Warning: {path} was fitted on different weights. No abstention is applied; fit it again for these weights.')
        return None
    if num_classes is not None and int(rule.info.get('num_classes', num_classes)) != int(num_classes):
        warn(f"Warning: {path} was fitted for {rule.info.get('num_classes')} classes, not {num_classes}. No abstention is applied.")
        return None
    return rule


def pipeline_issues(rule: AbstentionRule, pipeline: dict, temperature_applied: bool = False) -> list[str]:
    """What differs from the fit. The score's meaning depends on all of it, e.g. averaging clips lowers max_prob (V-016), so the guarantee does not carry over."""
    fitted = rule.info.get('pipeline') or {}
    issues = [f'{name}: fitted with {fitted[name]!r}, now {pipeline[name]!r}' for name in sorted(set(fitted) & set(pipeline))
              if str(fitted[name]) != str(pipeline[name])]
    if bool(temperature_applied) != bool(rule.info.get('temperature_applied', False)):
        issues.append(f"temperature: fitted with temperature_applied={rule.info.get('temperature_applied', False)}, now {temperature_applied}")
    return issues


def apply_rule(rule: AbstentionRule, frame: pd.DataFrame, pipeline: Optional[dict] = None, temperature_applied: bool = False, warn=print) -> Optional[dict]:
    """Adds `answered` and `decision` (the predicted label, or "not sure") to `frame` in place, and returns the summary to save. None when the score is missing."""
    try:
        score = score_values(frame, rule.score_name)
    except ValueError as exc:
        warn(f'Warning: no abstention is applied. {exc}')
        return None
    issues = pipeline_issues(rule, pipeline or {}, temperature_applied)
    for issue in issues:
        warn(f'Warning: this run differs from the abstention fit ({issue}). The target error guarantee does not hold for these outputs; fit it again with these settings.')
    answered = rule.answers(score)
    frame['answered'] = answered
    labels = frame['predicted_label'].astype(str) if 'predicted_label' in frame.columns else frame['predicted_class'].astype(str)
    frame['decision'] = labels.where(answered, NOT_SURE)
    holds = bool(rule.info.get('guarantee_holds')) and not issues
    summary = {'n_clips': int(len(frame)), 'answered': int(answered.sum()), 'not_sure': int((~answered).sum()),
               'answer_rate': float(answered.mean()) if len(frame) else None, 'threshold': rule.threshold, 'target_error': rule.info['target_error'],
               'guarantee_kind': rule.info['guarantee_kind'], 'status': rule.info['status'], 'guarantee_holds': holds, 'pipeline_mismatches': issues,
               'loose': rule.info.get('loose'), 'effective_n': rule.info.get('effective_n'), 'unit': rule.info.get('unit'), 'guarantee_level': rule.info.get('guarantee_level'),
               'assumes_independent_clips': rule.info.get('assumes_independent_clips'),
               'headline': rule.info['headline'] if not issues else 'No guarantee: this run differs from the abstention fit: ' + '; '.join(issues),
               'not_guaranteed': rule.info.get('not_guaranteed', [])}
    if 'correct' in frame.columns and frame['correct'].notna().any():
        known = frame.loc[answered, 'correct'].dropna().astype(bool)
        summary['clip_error_among_answered'] = None if known.empty else float(1 - known.mean())
        summary['clip_error_note'] = 'Observed on these clips, counted per clip. A check only when these are new groups from the calibration population.'
    return summary


# -- selective-risk curve -------------------------------------------------------------------------------------------
def selective_curve(score, wrong, groups=None, sources=None, *, grid=None, num_classes: int = 2, n_boot: int = 2000, seed: int = 0,
                    expected_error=None) -> pd.DataFrame:
    """Error against coverage on held-out clips, one row per distinct answered set, with intervals over resampled whole groups (B3's stats). A description, not a guarantee."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    wrong = np.asarray(wrong, dtype=bool).reshape(-1)
    n = len(score)
    grid = threshold_grid(num_classes) if grid is None else np.asarray(grid, dtype=np.float64)
    units, _ = unit_labels(n, groups, sources)
    stats_table = threshold_stats(score, wrong, units, grid)
    keep = ~stats_table['answered_clips'].duplicated(keep='first').to_numpy()
    rows = []
    for tau, row in zip(grid[keep], stats_table[keep].to_dict('records')):
        answered = score >= tau

        def coverage(idx, answered=answered):
            return float(answered[idx].mean())

        def error(idx, answered=answered):
            picked = answered[idx]
            return float(wrong[idx][picked].mean()) if picked.any() else float('nan')

        out = {'threshold': float(tau), 'answered_clips': int(row['answered_clips']), 'answered_units': int(row['answered_units']),
               'coverage': row['answered_clips'] / n, 'error': row['wrong_clips'] / row['answered_clips'] if row['answered_clips'] else None,
               'group_mean_error': row['unit_loss_sum'] / row['answered_units'] if row['answered_units'] else None}
        if expected_error is not None:
            out['mean_expected_error'] = float(np.asarray(expected_error, dtype=np.float64)[answered].mean()) if answered.any() else None
        if n_boot > 0:
            band_error = bootstrap_ci(error, n, groups, sources, n_boot=n_boot, seed=seed)
            band_coverage = bootstrap_ci(coverage, n, groups, sources, n_boot=n_boot, seed=seed)
            out.update(error_low=band_error['low'], error_high=band_error['high'], coverage_low=band_coverage['low'], coverage_high=band_coverage['high'])
        rows.append(out)
    return pd.DataFrame(rows)


def build_report(rule: Optional[AbstentionRule], frame: pd.DataFrame, groups=None, sources=None, *, split: str = 'test', n_boot: int = 2000, seed: int = 0,
                 extra_warnings: Sequence[str] = ()) -> tuple[dict, pd.DataFrame]:
    """The held-out check of a fitted rule: `frame` is predictions.csv (confidence, correct). Nothing is fitted here."""
    score_name = rule.score_name if rule is not None else 'max_prob'
    score = score_values(frame, score_name)
    wrong = ~frame['correct'].astype(bool).to_numpy()
    expected = frame['expected_error'].to_numpy(dtype=np.float64) if 'expected_error' in frame.columns else None
    num_classes = rule.info.get('num_classes', 2) if rule is not None else 2
    curve = selective_curve(score, wrong, groups, sources, num_classes=num_classes, n_boot=n_boot, seed=seed, expected_error=expected)
    _, unit = unit_labels(len(score), groups, sources)
    report = {'format': 'deeptune-abstention-report', 'version': FORMAT_VERSION, 'split': split, 'n_clips': int(len(frame)), 'unit': unit,
              'n_units': int(len(pd.unique(unit_labels(len(score), groups, sources)[0]))), 'n_boot': int(n_boot),
              'curve_covers': 'Group-resampled percentile intervals: which held-out groups were drawn, for this fitted model. This curve is a description, not the guarantee.',
              'warnings': list(extra_warnings)}
    if rule is not None:
        answered = rule.answers(score)
        chosen = {'threshold': rule.threshold, 'answered_clips': int(answered.sum()), 'coverage': float(answered.mean()), 'not_sure_clips': int((~answered).sum())}
        if answered.any():
            def error(idx):
                picked = answered[idx]
                return float(wrong[idx][picked].mean()) if picked.any() else float('nan')
            chosen['error'] = float(wrong[answered].mean())
            chosen['error_interval'] = bootstrap_ci(error, len(score), groups, sources, n_boot=n_boot, seed=seed) if n_boot > 0 else None
            stats_row = threshold_stats(score, wrong, unit_labels(len(score), groups, sources)[0], [rule.threshold]).iloc[0]
            chosen['group_mean_error'] = float(stats_row['unit_loss_sum'] / stats_row['answered_units'])
            if rule.info['guarantee_kind'] == 'selective_risk':
                chosen['point_within_target'] = chosen['group_mean_error'] <= rule.info['target_error']
        report.update(target_error=rule.info['target_error'], guarantee_kind=rule.info['guarantee_kind'], guarantee_level=rule.info.get('guarantee_level'), assumes_independent_clips=rule.info.get('assumes_independent_clips'),
                       status=rule.info['status'], guarantee_holds=rule.info['guarantee_holds'],
                      headline=rule.info['headline'], at_fitted_threshold=chosen,
                      reading='One held-out set is one draw. A point above the target does not by itself refute a guarantee that holds with probability 1 - delta, and a point below does not confirm it.')
    return report, curve


def plot_selective_curve(curve: pd.DataFrame, path, target: Optional[float] = None, chosen: Optional[dict] = None, split: str = 'test') -> None:
    from evaluators.video.report import _MUTED, _SERIES, _finish, _new_axes
    figure, axes = _new_axes()
    rows = curve[curve['error'].notna()].sort_values('coverage')
    if 'error_low' in rows.columns and rows['error_low'].notna().any():
        axes.fill_between(rows['coverage'], rows['error_low'], rows['error_high'], color=_SERIES[0], alpha=0.18, linewidth=0, label='95% interval, whole groups resampled')
    axes.plot(rows['coverage'], rows['error'], color=_SERIES[0], linewidth=2, marker='o', markersize=4, label=f'error among answered clips, {split}')
    if target is not None:
        axes.axhline(target, color=_MUTED, linewidth=1, linestyle=':', label=f'target error {target:g}')
    if chosen and chosen.get('answered_clips') and chosen.get('error') is not None:
        axes.scatter([chosen['coverage']], [chosen['error']], color=_SERIES[1], s=70, zorder=3, edgecolors='white', linewidths=1.5, label='fitted threshold')
    axes.set_xlim(0, 1.02)
    axes.set_ylim(bottom=0)
    _finish(figure, axes, 'Selective risk: error against coverage', 'Coverage (share of clips answered)', 'Error among answered clips', Path(path), legend=True)


def write_report(report: dict, curve: pd.DataFrame, out_dir) -> Path:
    from evaluators.video.report import plotting_available
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / CURVE_FILENAME, index=False)
    plots = plotting_available()
    report['plots'] = 'written' if plots else 'skipped: matplotlib could not be imported'
    if plots:
        plot_selective_curve(curve, out_dir / PLOT_FILENAME, report.get('target_error'), report.get('at_fitted_threshold'), report.get('split', 'test'))
    else:
        print('Warning: matplotlib could not be imported, so there is no selective-risk plot. selective_risk_curve.csv is complete.')
    path = out_dir / REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2, default=_json_number), encoding='utf-8')
    return path


def write_fit_table(rule: AbstentionRule, out_dir) -> Path:
    path = Path(out_dir) / FIT_TABLE_FILENAME
    pd.DataFrame(rule.info['fit_table']).to_csv(path, index=False)
    return path


# -- the calibration split ------------------------------------------------------------------------------------------
def carve_calibration_rows(frame: pd.DataFrame, n_calibration_units: int, units: np.ndarray, seed: Optional[int], label_column: str = 'labels') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw whole units uniformly at random. Not stratified: a uniform draw is what makes calibration units exchangeable with other units of the population, and the
    guarantee needs no class in calibration. Refuses when training would lose a class."""
    unique = pd.unique(units)
    if len(unique) < 2:
        raise ValueError('The calibration split needs at least two training groups to draw from.')
    if not 1 <= n_calibration_units < len(unique):
        raise ValueError(f'Cannot take {n_calibration_units} of {len(unique)} training groups for calibration.')
    picked = np.random.default_rng(seed).permutation(unique)[:n_calibration_units]
    mask = np.isin(units, picked)
    train, calibration = frame[~mask].reset_index(drop=True), frame[mask].reset_index(drop=True)
    if label_column in frame.columns:
        lost = sorted(map(str, set(frame[label_column]) - set(train[label_column])))
        if lost:
            raise ValueError(f'Taking the calibration split would leave no training rows for class(es) {lost}. Lower --calib_fraction or add groups.')
    return train, calibration


def overlapping_units(calibration_units, *other_units) -> list:
    """Units of the calibration split that another split also holds."""
    calibration = set(calibration_units)
    return sorted({str(unit) for others in other_units for unit in others if unit in calibration})

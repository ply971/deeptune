"""Confidence intervals for held-out results (P-117). Torch-free.

Clips from one source are dependent, so every interval here resamples whole units: the
`group` column when there is one, else the source video, else the row (which the result
says, and warns about). A resample keeps all the rows of each drawn unit.

What these intervals cover: the uncertainty from *which test units were drawn*, with the
model held fixed. They do not cover training variability (another seed or training set
would give another model); the fold spread in `evaluators/video/validation.py` speaks to
that part.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_BOOTSTRAPS = 2000
DEFAULT_LEVEL = 0.95
FEW_UNITS = 20  # percentile intervals are unreliable with fewer units than this
ROW_UNIT_WARNING = ("no group or repeated source video: resampled rows, which assumes clips are independent. "
                    "If several clips come from one source, this interval is too narrow; add a 'group' column.")
COVERS = 'test-sample uncertainty for this fitted model (which units were drawn); not training variability'


def unit_labels(n: int, groups: Optional[Sequence] = None, sources: Optional[Sequence] = None) -> tuple[np.ndarray, str]:
    """The unit each row belongs to, and its name: 'group', 'source_video' or 'row'."""
    if groups is not None:
        values = pd.Series(list(groups))
        if len(values) != n:
            raise ValueError(f'{len(values)} group values for {n} rows.')
        if values.isna().any():
            raise ValueError('Group values cannot be missing: a missing group has no unit to resample.')
        return pd.factorize(values)[0], 'group'
    if sources is not None:
        values = pd.Series(list(sources))
        if len(values) != n:
            raise ValueError(f'{len(values)} source values for {n} rows.')
        if values.notna().all() and values.nunique() < n:
            return pd.factorize(values)[0], 'source_video'
    return np.arange(n), 'row'


def _unit_rows(units: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(units, kind='mergesort')
    _, starts = np.unique(units[order], return_index=True)
    return np.split(order, starts[1:])


def _resampled_indices(rows: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])


def _interval(values: np.ndarray, point: float, level: float, n_units: int, unit: str, n_boot: int) -> dict:
    values = values[np.isfinite(values)]
    out = {'point': None if not np.isfinite(point) else float(point), 'level': level, 'unit': unit, 'n_units': int(n_units),
           'n_boot': int(n_boot), 'n_valid_resamples': int(len(values)), 'method': 'percentile', 'covers': COVERS}
    if len(values) < max(2, 0.5 * n_boot):  # a metric undefined on most resamples has no meaningful interval
        return {**out, 'low': None, 'high': None, 'warning': 'metric undefined on most resamples'}
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    out.update(low=float(low), high=float(high))
    warnings = []
    if unit == 'row':
        warnings.append(ROW_UNIT_WARNING)
    if n_units < FEW_UNITS:
        warnings.append(f'only {n_units} {unit} units: this interval is unstable')
    if low == high:
        warnings.append(f'zero width: every resample gave {low:.4g}. This does not mean certainty: a bootstrap cannot show errors the {n_units} units did not contain')
    if warnings:
        out['warning'] = '; '.join(warnings)
    return out


def bootstrap_ci(metric: Callable[[np.ndarray], float], n: int, groups=None, sources=None, n_boot: int = DEFAULT_BOOTSTRAPS,
                 level: float = DEFAULT_LEVEL, seed: int = 0) -> dict:
    """Percentile interval of `metric(row_indices)` over cluster-bootstrap resamples.

    `metric` receives row positions (repeats included) and returns a float, NaN where undefined.
    """
    units, unit = unit_labels(n, groups, sources)
    rows = _unit_rows(units)
    rng = np.random.default_rng(seed)
    values = np.array([metric(_resampled_indices(rows, rng)) for _ in range(int(n_boot))], dtype=np.float64)
    return _interval(values, float(metric(np.arange(n))), level, len(rows), unit, n_boot)


def paired_difference_ci(metric_a: Callable[[np.ndarray], float], metric_b: Callable[[np.ndarray], float], n: int, groups=None,
                         sources=None, n_boot: int = DEFAULT_BOOTSTRAPS, level: float = DEFAULT_LEVEL, seed: int = 0) -> dict:
    """Interval of metric_a - metric_b, both computed on the *same* resampled units of a shared test set.

    `zero_excluded` is the honest summary: it is not a p-value and does not correct for many comparisons."""
    units, unit = unit_labels(n, groups, sources)
    rows = _unit_rows(units)
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = _resampled_indices(rows, rng)
        values[i] = metric_a(idx) - metric_b(idx)
    every = np.arange(n)
    out = _interval(values, float(metric_a(every) - metric_b(every)), level, len(rows), unit, n_boot)
    out['covers'] = 'test-sample uncertainty of the paired difference between these two fitted models on one test set'
    out['zero_excluded'] = None if out['low'] is None else bool(out['low'] > 0 or out['high'] < 0)
    return out


def fold_summary(values: Sequence[float], level: float = DEFAULT_LEVEL) -> dict:
    """Mean, sample std and a Student-t interval over folds or resamples.

    Folds share training data, so this interval is known to be too narrow (no unbiased variance estimator
    exists: Bengio & Grandvalet, JMLR 2004). `std` describes the spread between folds. The interval is not a guarantee."""
    from scipy import stats
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    n = len(array)
    out = {'n': n, 'mean': float(array.mean()) if n else None, 'std': float(array.std(ddof=1)) if n > 1 else None,
           'fold_t_low': None, 'fold_t_high': None, 'level': level,
           'covers': 'std: the spread between folds. fold_t_low and fold_t_high treat folds as independent; they are not, so that interval is too narrow'}
    if n > 1:
        half = stats.t.ppf(0.5 + level / 2, n - 1) * out['std'] / np.sqrt(n)
        out.update(fold_t_low=float(out['mean'] - half), fold_t_high=float(out['mean'] + half))
    return out

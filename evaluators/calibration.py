"""Calibration of classifier probabilities: ECE, Brier, reliability bins, temperature scaling. Torch-free.

A temperature is fitted on validation or out-of-fold probabilities only, and reported on other data.
`calibration_report` needs both sets as separate arguments so that the split is on the record.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ECE_BINS = 15
TEMPERATURE_BOUNDS = (0.01, 100.0)
_EPS = 1e-12


def _checked(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    if p.ndim != 2 or len(p) != len(y) or not len(y):
        raise ValueError('Expected probabilities [N, K] and N labels, with N > 0.')
    if y.min() < 0 or y.max() >= p.shape[1]:
        raise ValueError(f'Labels must lie in 0..{p.shape[1] - 1}.')
    return p, y


def reliability_bins(probabilities, labels, n_bins: int = DEFAULT_ECE_BINS) -> list[dict]:
    """Equal-width bins over the top-class confidence, (lo, hi], with the first bin closed at 0."""
    p, y = _checked(probabilities, labels)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    edges = np.linspace(0, 1, int(n_bins) + 1)
    index = np.clip(np.searchsorted(edges, confidence, side='left') - 1, 0, int(n_bins) - 1)
    rows = []
    for i in range(int(n_bins)):
        mask = index == i
        count = int(mask.sum())
        rows.append({'bin': i + 1, 'low': float(edges[i]), 'high': float(edges[i + 1]), 'n': count,
                     'mean_confidence': float(confidence[mask].mean()) if count else None,
                     'accuracy': float(correct[mask].mean()) if count else None})
    return rows


def expected_calibration_error(probabilities, labels, n_bins: int = DEFAULT_ECE_BINS) -> float:
    """Top-label ECE: the count-weighted mean of |accuracy - confidence| over equal-width bins (Guo et al., ICML 2017)."""
    rows = reliability_bins(probabilities, labels, n_bins)
    total = sum(r['n'] for r in rows)
    return float(sum(r['n'] * abs(r['accuracy'] - r['mean_confidence']) for r in rows if r['n']) / total)


def brier_score(probabilities, labels) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2, range 0 to 2."""
    p, y = _checked(probabilities, labels)
    one_hot = np.eye(p.shape[1])[y]
    return float(((p - one_hot) ** 2).sum(axis=1).mean())


def negative_log_likelihood(probabilities, labels) -> float:
    p, y = _checked(probabilities, labels)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1)).mean())


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    """softmax(log p / T). The same as dividing the logits by T, as the log differs from them by a constant per sample."""
    if not temperature > 0:
        raise ValueError('The temperature must be positive.')
    logp = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1)) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    scaled = np.exp(logp)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, labels) -> float:
    """The T > 0 minimising the NLL on the data given. Pass validation or out-of-fold rows, never test rows."""
    from scipy.optimize import minimize_scalar
    p, y = _checked(probabilities, labels)
    logp = np.log(np.clip(p, _EPS, 1))

    def nll(log_t):
        scaled = logp / np.exp(log_t)
        scaled -= scaled.max(axis=1, keepdims=True)
        return float(-(scaled[np.arange(len(y)), y] - np.log(np.exp(scaled).sum(axis=1))).mean())

    result = minimize_scalar(nll, bounds=tuple(np.log(TEMPERATURE_BOUNDS)), method='bounded', options={'xatol': 1e-5})
    return float(np.exp(result.x))


def _scores(p, y, n_bins) -> dict:
    return {'ece': expected_calibration_error(p, y, n_bins), 'brier': brier_score(p, y), 'nll': negative_log_likelihood(p, y),
            'reliability': reliability_bins(p, y, n_bins)}


def calibration_report(eval_probabilities, eval_labels, fit_probabilities=None, fit_labels=None, fit_split: str | None = None,
                       eval_split: str = 'test', n_bins: int = DEFAULT_ECE_BINS, selection_used_fit_split=None) -> dict:
    """Before/after scores on the evaluation rows. Temperature scaling leaves the argmax unchanged, so accuracy does not move.

    Without a fit set the report says so and gives only the 'before' scores. A temperature is never fitted on the rows it
    is reported on."""
    p, y = _checked(eval_probabilities, eval_labels)
    report = {'eval_split': eval_split, 'n_eval': int(len(y)), 'n_bins': int(n_bins), 'before': _scores(p, y, n_bins)}
    if fit_probabilities is None:
        return {**report, 'temperature': None, 'fit_split': None, 'after': None,
                'note': 'not fitted: no validation or out-of-fold probabilities were available'}
    if fit_split in (None, eval_split):
        raise ValueError(f'A temperature must be fitted on another split than {eval_split!r}; got {fit_split!r}.')
    temperature = fit_temperature(fit_probabilities, fit_labels)
    return {**report, 'temperature': temperature, 'fit_split': fit_split, 'n_fit': int(len(fit_labels)),
            'fit_split_used_for_model_selection': selection_used_fit_split,
            'after': _scores(apply_temperature(p, temperature), y, n_bins)}

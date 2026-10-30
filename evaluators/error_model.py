"""Per-prediction error estimate ("adaptive assurance") for DeepTune.

Every prediction DeepTune makes can carry an ``expected_error``:

* classification: the probability that this prediction is wrong;
* regression: the expected absolute error of this prediction (target units).

The number comes from a small *error model* fitted on held-out predictions
(the validation split, at the end of training; later phases can fit it on
out-of-fold predictions instead). Its job is to map per-sample signals that
are available at prediction time -- max softmax probability, top-1/top-2
margin, normalised entropy, and any ``extra`` signals a later phase adds
(multi-clip TTA disagreement, embedding kNN distance to the training set,
evidential uncertainty, ...) -- onto the error rate actually observed for
predictions like it. ``calibration_table`` then checks the promise on data
the error model never saw: per decile of stated error, is the observed error
the same?

This module is deliberately torch-free (numpy + pandas only) so the desktop
GUI, which must never import torch, can read ``error_model.json`` and the
calibration reports directly. Models are stored as plain JSON (no pickle).

Methods
-------
``isotonic`` (default when there is enough data)
    Weighted pool-adjacent-violators (PAV) on one score feature
    (``max_prob`` for classification), so the stated error is monotone in
    confidence. Smoothing for small data: every isotonic block is shrunk
    toward the (Laplace-smoothed) base rate with ``prior_strength``
    pseudo-samples, the shrunk blocks are re-pooled so monotonicity holds,
    and predictions interpolate linearly between block centres (constant
    beyond the ends) instead of jumping in steps.
``logistic``
    L2-regularised logistic regression on several standardised features
    (classification only). Opt-in; useful once extra signals exist.
``constant``
    The base rate: the Laplace-smoothed error rate ``(errors + 1)/(n + 2)``
    for classification, the mean absolute error for regression. Used as the
    fallback whenever the fit set is too small to learn anything more:
    fewer than ``MIN_FIT_SAMPLES`` (30) samples, or -- for classification --
    fewer than ``MIN_OUTCOME_COUNT`` (5) wrong *or* fewer than 5 right
    predictions, or a regression run with no uncertainty signal.

Regression note: a regression network gives no built-in confidence, so
without an ``extra`` signal (passed as a column and named with
``score_feature``) the error model can only predict the mean absolute error
for every sample. That is still honest, just not adaptive.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

FORMAT = 'deeptune-error-model'
FORMAT_VERSION = 1
ERROR_MODEL_FILENAME = 'error_model.json'
ERROR_MODEL_FIT_FILENAME = 'error_model_fit.json'
CALIBRATION_STEM = 'error_calibration'

MIN_FIT_SAMPLES = 30
MIN_OUTCOME_COUNT = 5
PRIOR_STRENGTH = 2.0
LOGISTIC_L2 = 1.0

CLASSIFICATION_FEATURES = ('max_prob', 'margin', 'entropy')
# Direction of each built-in feature: True when a larger value means a
# larger error. Extra features default to "larger = more uncertain".
KNOWN_INCREASING = {'max_prob': False, 'margin': False, 'entropy': True}

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Opt-out switch (``--no_error_model``). cli.py sets it once per process; a
# Trainer can also override it per instance with ``trainer.fit_error_model``.
# ---------------------------------------------------------------------------
_FIT_BY_DEFAULT = True


def set_fit_by_default(enabled: bool) -> None:
    """Process-wide default for fitting the error model after training."""
    global _FIT_BY_DEFAULT
    _FIT_BY_DEFAULT = bool(enabled)


def fit_by_default() -> bool:
    return _FIT_BY_DEFAULT


# ---------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def prediction_features(scores, mode: str, extra=None, from_logits: bool = False) -> pd.DataFrame:
    """Per-sample features the error model reads.

    Args:
        scores: classification -> ``[N, K]`` class probabilities (or logits
            with ``from_logits=True``); regression -> ``[N]`` or ``[N, D]``
            model outputs.
        mode: ``'cls'`` or ``'reg'``.
        extra: optional per-sample extra signals (DataFrame, dict of arrays,
            or ``[N, M]`` array -> columns ``extra_0..``), appended as
            columns. Must have N rows.

    Returns:
        classification: ``max_prob``, ``margin`` (top1 - top2) and
        ``entropy`` (normalised to [0, 1] by log K); regression:
        ``predicted_value`` (mean over output dims). Plus ``extra`` columns.
    """
    if mode not in ('cls', 'reg'):
        raise ValueError("mode must be 'cls' or 'reg'.")
    arr = np.asarray(scores, dtype=np.float64)
    if mode == 'cls':
        if arr.ndim != 2 or arr.shape[1] < 1:
            raise ValueError('Classification scores must be shaped [N, num_classes].')
        probs = softmax(arr) if from_logits else arr
        k = probs.shape[1]
        ordered = np.sort(probs, axis=1)[:, ::-1]
        top1 = ordered[:, 0]
        top2 = ordered[:, 1] if k > 1 else np.zeros(len(probs))
        clipped = np.clip(probs, 1e-12, 1.0)
        entropy = -(probs * np.log(clipped)).sum(axis=1)
        entropy = entropy / math.log(k) if k > 1 else np.zeros(len(probs))
        frame = pd.DataFrame({'max_prob': top1, 'margin': top1 - top2,
                              'entropy': np.clip(entropy, 0.0, 1.0)})
    else:
        values = arr.reshape(len(arr), -1) if arr.ndim else arr.reshape(1, 1)
        frame = pd.DataFrame({'predicted_value': values.mean(axis=1)})
    if extra is not None:
        extra_frame = _as_frame(extra, len(frame))
        clash = set(extra_frame.columns) & set(frame.columns)
        if clash:
            raise ValueError(f'Extra feature names clash with built-in features: {sorted(clash)}')
        frame = pd.concat([frame, extra_frame.reset_index(drop=True)], axis=1)
    return frame


def _as_frame(extra, n: int) -> pd.DataFrame:
    if isinstance(extra, pd.DataFrame):
        frame = extra.copy()
    elif isinstance(extra, dict):
        frame = pd.DataFrame({str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in extra.items()})
    else:
        values = np.asarray(extra, dtype=np.float64)
        values = values.reshape(len(values), -1)
        frame = pd.DataFrame(values, columns=[f'extra_{i}' for i in range(values.shape[1])])
    if len(frame) != n:
        raise ValueError(f'extra has {len(frame)} rows, expected {n}.')
    frame.columns = [str(c) for c in frame.columns]
    return frame


def error_targets(scores, labels, mode: str) -> np.ndarray:
    """What the error model learns: ``is_wrong`` (0/1, argmax != label) for
    classification, absolute error (mean over output dims) for regression."""
    arr = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels)
    if mode == 'cls':
        return (arr.argmax(axis=1) != lab.reshape(-1).astype(np.int64)).astype(np.float64)
    values = arr.reshape(len(arr), -1)
    truth = lab.astype(np.float64).reshape(len(arr), -1)
    return np.abs(values - truth).mean(axis=1)


# ---------------------------------------------------------------------------
# Isotonic regression (weighted PAV), numpy only
# ---------------------------------------------------------------------------
def _pav(y: np.ndarray, w: np.ndarray):
    """Weighted pool-adjacent-violators for a non-decreasing fit of ``y``
    (already ordered by x). Returns (block_values, block_starts, block_ends,
    block_weights); block i covers ``y[starts[i]:ends[i]]``."""
    values, weights, starts, ends = [], [], [], []
    for i, (yi, wi) in enumerate(zip(y, w)):
        values.append(float(yi)); weights.append(float(wi)); starts.append(i); ends.append(i + 1)
        while len(values) > 1 and values[-2] > values[-1]:
            w_new = weights[-2] + weights[-1]
            v_new = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w_new
            values[-2:] = [v_new]; weights[-2:] = [w_new]
            ends[-2:] = [ends[-1]]; starts.pop()
    return np.array(values), np.array(starts), np.array(ends), np.array(weights)


def isotonic_fit_values(y, w=None) -> np.ndarray:
    """Per-element isotonic (non-decreasing) fit, for testing against
    scikit-learn's reference implementation."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    values, starts, ends, _ = _pav(y, w)
    out = np.empty_like(y)
    for v, s, e in zip(values, starts, ends):
        out[s:e] = v
    return out


def _fit_isotonic_knots(x: np.ndarray, y: np.ndarray, prior: float, prior_strength: float):
    """Monotone (non-decreasing in x) smoothed knots: PAV over unique x,
    shrink each block toward ``prior``, re-pool, and return block centres
    (count-weighted mean x) and values."""
    order = np.argsort(x, kind='mergesort')
    xs, ys = x[order], y[order]
    ux, inverse, counts = np.unique(xs, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ys)
    means = sums / counts
    values, starts, ends, weights = _pav(means, counts.astype(np.float64))
    centres = np.array([np.average(ux[s:e], weights=counts[s:e]) for s, e in zip(starts, ends)])
    shrunk = (values * weights + prior_strength * prior) / (weights + prior_strength)
    # Shrinking small blocks harder can break monotonicity; pool again.
    v2, s2, e2, _ = _pav(shrunk, weights + prior_strength)
    knots_x = np.array([np.average(centres[s:e], weights=weights[s:e]) for s, e in zip(s2, e2)])
    return knots_x, v2


# ---------------------------------------------------------------------------
# The error model
# ---------------------------------------------------------------------------
class ErrorModel:
    """Maps per-sample prediction features to an expected error.

    Args:
        mode: ``'cls'`` (target = is_wrong) or ``'reg'`` (target = abs error).
        method: ``'auto'`` (isotonic on ``score_feature`` when possible,
            otherwise constant), ``'isotonic'``, ``'logistic'`` (cls only,
            multi-feature) or ``'constant'``. Too-small fit sets always fall
            back to ``'constant'``; ``fit_info['fallback_reason']`` says why.
        score_feature: the 1-D score for isotonic; defaults to ``max_prob``
            for classification and to none (-> constant) for regression.
        features: columns used by ``'logistic'`` (default: max_prob, margin,
            entropy plus any extra columns present at fit time).
        increasing: ``'auto'`` (built-in direction for known features, larger
            = more error for extras) or a bool.
    """

    def __init__(self, mode: str, method: str = 'auto', score_feature: Optional[str] = None,
                 features: Optional[Sequence[str]] = None, increasing='auto',
                 min_samples: int = MIN_FIT_SAMPLES, min_outcome_count: int = MIN_OUTCOME_COUNT,
                 prior_strength: float = PRIOR_STRENGTH, l2: float = LOGISTIC_L2):
        if mode not in ('cls', 'reg'):
            raise ValueError("mode must be 'cls' or 'reg'.")
        if method not in ('auto', 'isotonic', 'logistic', 'constant'):
            raise ValueError("method must be 'auto', 'isotonic', 'logistic' or 'constant'.")
        if method == 'logistic' and mode != 'cls':
            raise ValueError('The logistic error model is for classification only.')
        self.mode = mode
        self.method = method
        self.score_feature = score_feature if score_feature is not None else ('max_prob' if mode == 'cls' else None)
        self.features = list(features) if features is not None else None
        self.increasing = increasing
        self.min_samples = int(min_samples)
        self.min_outcome_count = int(min_outcome_count)
        self.prior_strength = float(prior_strength)
        self.l2 = float(l2)
        self.method_used: Optional[str] = None
        self.constant: Optional[float] = None
        self.knots_x: Optional[np.ndarray] = None
        self.knots_y: Optional[np.ndarray] = None
        self.direction_increasing: Optional[bool] = None
        self.logistic: Optional[dict] = None
        self.fit_info: dict = {}

    # -- fitting ------------------------------------------------------------
    def fit(self, features: pd.DataFrame, target, num_outputs: Optional[int] = None) -> 'ErrorModel':
        """Fit on held-out predictions (validation or out-of-fold, never
        test). ``num_outputs`` (number of classes / output dims) is recorded
        so a mismatched model is not applied to a different network."""
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(features) != len(target):
            raise ValueError('features and target must have the same number of rows.')
        finite = np.isfinite(target)
        n = int(finite.sum())
        y = target[finite]
        frame = features.reset_index(drop=True).loc[finite].reset_index(drop=True)
        if self.mode == 'cls':
            if not np.isin(y, (0.0, 1.0)).all():
                raise ValueError('Classification error targets must be 0 (right) or 1 (wrong).')
            n_err = int(y.sum())
            base = float(y.mean()) if n else float('nan')
            fallback = (n_err + 1.0) / (n + 2.0)  # Laplace-smoothed error rate
        else:
            if (y < 0).any():
                raise ValueError('Regression error targets must be absolute errors (>= 0).')
            n_err = None
            base = float(y.mean()) if n else float('nan')
            fallback = base if n else 0.0
        self.constant = float(fallback)
        self.fit_info = {'n_samples': n, 'base_error': base, 'fallback_error': float(fallback),
                         'requested_method': self.method, 'fallback_reason': None,
                         'num_outputs': None if num_outputs is None else int(num_outputs)}
        if self.mode == 'cls':
            self.fit_info['n_errors'] = n_err

        reason = self._fallback_reason(n, y, n_err, frame)
        if reason is not None:
            self.method_used = 'constant'
            self.fit_info['fallback_reason'] = reason
        elif self.method == 'logistic':
            self._fit_logistic(frame, y)
        else:
            self._fit_isotonic(frame, y)
        self.fit_info['method'] = self.method_used
        self.fit_info['score_feature'] = self.score_feature if self.method_used == 'isotonic' else None
        return self

    def _fallback_reason(self, n, y, n_err, frame) -> Optional[str]:
        if self.method == 'constant':
            return 'constant method requested'
        if n < self.min_samples:
            return f'only {n} fit samples (< {self.min_samples})'
        if self.mode == 'cls':
            if n_err < self.min_outcome_count:
                return f'only {n_err} wrong predictions in the fit set (< {self.min_outcome_count})'
            if n - n_err < self.min_outcome_count:
                return f'only {n - n_err} correct predictions in the fit set (< {self.min_outcome_count})'
        if self.method == 'logistic':
            return None
        if self.score_feature is None:
            return 'no uncertainty signal for regression (pass an extra feature as score_feature)'
        if self.score_feature not in frame.columns:
            raise ValueError(f'score_feature {self.score_feature!r} not in features {list(frame.columns)}')
        x = frame[self.score_feature].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            return f'non-finite values in {self.score_feature}'
        if np.ptp(x) == 0:
            return f'{self.score_feature} is constant in the fit set'
        if self.mode == 'reg' and np.ptp(y) == 0:
            return 'absolute error is constant in the fit set'
        return None

    def _direction(self) -> bool:
        if isinstance(self.increasing, bool):
            return self.increasing
        return KNOWN_INCREASING.get(self.score_feature, True)

    def _fit_isotonic(self, frame: pd.DataFrame, y: np.ndarray) -> None:
        x = frame[self.score_feature].to_numpy(dtype=np.float64)
        inc = self._direction()
        xt = x if inc else -x
        knots_x, knots_y = _fit_isotonic_knots(xt, y, self.constant, self.prior_strength)
        self.direction_increasing = bool(inc)
        self.knots_x, self.knots_y = knots_x, knots_y
        self.method_used = 'isotonic'
        self.fit_info['n_knots'] = int(len(knots_x))

    def _fit_logistic(self, frame: pd.DataFrame, y: np.ndarray) -> None:
        names = self.features or [c for c in frame.columns if c != 'predicted_value']
        missing = [c for c in names if c not in frame.columns]
        if missing:
            raise ValueError(f'Missing logistic features: {missing}')
        X = frame[names].to_numpy(dtype=np.float64)
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - mean) / scale])
        beta = np.zeros(Z.shape[1])
        beta[0] = math.log(self.constant / (1.0 - self.constant))
        penalty = np.full(Z.shape[1], self.l2); penalty[0] = 0.0
        for _ in range(100):  # Newton / IRLS with an L2 penalty on the slopes
            p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
            grad = Z.T @ (p - y) + penalty * beta
            hess = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(penalty) + 1e-9 * np.eye(len(beta))
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.features = list(names)
        self.logistic = {'mean': mean.tolist(), 'scale': scale.tolist(),
                         'intercept': float(beta[0]), 'coef': beta[1:].tolist()}
        self.method_used = 'logistic'

    # -- prediction ---------------------------------------------------------
    def required_features(self) -> list:
        if self.method_used == 'isotonic':
            return [self.score_feature]
        if self.method_used == 'logistic':
            return list(self.features)
        return []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Expected error per row: P(wrong) in [0, 1] for classification,
        expected |error| >= 0 for regression. Rows with a missing/non-finite
        required feature get the fallback (base-rate) value."""
        if self.method_used is None:
            raise RuntimeError('ErrorModel is not fitted.')
        n = len(features)
        missing = [c for c in self.required_features() if c not in features.columns]
        if missing:
            raise ValueError(f'Features missing for the error model: {missing}')
        out = np.full(n, self.constant, dtype=np.float64)
        if self.method_used == 'isotonic':
            x = features[self.score_feature].to_numpy(dtype=np.float64)
            ok = np.isfinite(x)
            xt = x[ok] if self.direction_increasing else -x[ok]
            out[ok] = np.interp(xt, self.knots_x, self.knots_y)
        elif self.method_used == 'logistic':
            X = features[self.features].to_numpy(dtype=np.float64)
            ok = np.isfinite(X).all(axis=1)
            z = (X[ok] - np.asarray(self.logistic['mean'])) / np.asarray(self.logistic['scale'])
            logit = self.logistic['intercept'] + z @ np.asarray(self.logistic['coef'])
            out[ok] = 1.0 / (1.0 + np.exp(-logit))
        return np.clip(out, 0.0, 1.0) if self.mode == 'cls' else np.maximum(out, 0.0)

    # -- (de)serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        if self.method_used is None:
            raise RuntimeError('ErrorModel is not fitted.')
        return {
            'format': FORMAT, 'version': FORMAT_VERSION, 'mode': self.mode,
            'target': 'probability prediction is wrong' if self.mode == 'cls' else 'expected absolute error',
            'method': self.method_used, 'requested_method': self.method,
            'score_feature': self.score_feature, 'features': self.features,
            'direction_increasing': self.direction_increasing,
            'knots_x': None if self.knots_x is None else [float(v) for v in self.knots_x],
            'knots_y': None if self.knots_y is None else [float(v) for v in self.knots_y],
            'logistic': self.logistic, 'constant': self.constant,
            'settings': {'min_samples': self.min_samples, 'min_outcome_count': self.min_outcome_count,
                         'prior_strength': self.prior_strength, 'l2': self.l2},
            'fit_info': self.fit_info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ErrorModel':
        if data.get('format') != FORMAT:
            raise ValueError('Not a DeepTune error model file.')
        if int(data.get('version', 0)) > FORMAT_VERSION:
            raise ValueError(f"Error model format version {data.get('version')} is newer than this DeepTune supports.")
        settings = data.get('settings', {})
        model = cls(data['mode'], method=data.get('requested_method', 'auto'),
                    score_feature=data.get('score_feature'), features=data.get('features'),
                    **{k: settings[k] for k in ('min_samples', 'min_outcome_count', 'prior_strength', 'l2') if k in settings})
        model.method_used = data['method']
        model.constant = float(data['constant'])
        model.direction_increasing = data.get('direction_increasing')
        if data.get('knots_x') is not None:
            model.knots_x = np.asarray(data['knots_x'], dtype=np.float64)
            model.knots_y = np.asarray(data['knots_y'], dtype=np.float64)
        model.logistic = data.get('logistic')
        model.fit_info = data.get('fit_info', {})
        return model

    def to_json(self) -> str:
        return json.dumps(_jsonable(self.to_dict()), indent=2)

    @classmethod
    def from_json(cls, text: str) -> 'ErrorModel':
        return cls.from_dict(json.loads(text))

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path: PathLike) -> 'ErrorModel':
        return cls.from_json(Path(path).read_text(encoding='utf-8-sig'))


def _jsonable(value):
    """NaN/inf are not valid JSON; write them as null."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Locating the error model of a run
# ---------------------------------------------------------------------------
def find_error_model(checkpoint_path: Optional[PathLike]) -> Optional[Path]:
    """``error_model.json`` next to a checkpoint file (or inside a checkpoint
    directory), or None. Old runs simply have none."""
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    candidate = (path if path.is_dir() else path.parent) / ERROR_MODEL_FILENAME
    return candidate if candidate.is_file() else None


def load_error_model_near(checkpoint_path: Optional[PathLike], mode: str,
                          num_outputs: Optional[int] = None, path: Optional[PathLike] = None,
                          warn=print) -> Optional[ErrorModel]:
    """Load the run's error model if there is a usable one; otherwise None
    (with a warning when a file exists but cannot be used)."""
    found = Path(path) if path is not None else find_error_model(checkpoint_path)
    if found is None or not found.is_file():
        return None
    try:
        model = ErrorModel.load(found)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warn(f'Ignoring unreadable error model {found}: {exc}')
        return None
    if model.mode != mode:
        warn(f'Ignoring error model {found}: it was fitted for mode {model.mode!r}, not {mode!r}.')
        return None
    fitted_k = model.fit_info.get('num_outputs')
    if num_outputs is not None and fitted_k is not None and int(fitted_k) != int(num_outputs):
        warn(f'Ignoring error model {found}: fitted for {fitted_k} outputs, model has {num_outputs}.')
        return None
    return model


# ---------------------------------------------------------------------------
# Is the stated error trustworthy? Calibration on held-out data
# ---------------------------------------------------------------------------
def _wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _auroc(score: np.ndarray, positive: np.ndarray) -> Optional[float]:
    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(score).rank(method='average').to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_table(expected_error, observed, n_bins: int = 10, mode: Optional[str] = None) -> dict:
    """Stated vs observed error per equal-count bin (deciles by default).

    Args:
        expected_error: what the error model said for each held-out sample.
        observed: what happened -- 1/0 wrong/right (classification) or the
            absolute error (regression).
        mode: ``'cls'``/``'reg'``; inferred (0/1 observations -> cls) if None.

    Returns ``{'summary': {...}, 'bins': [...]}``. Each bin has its sample
    count, stated range and mean, observed mean, gap (observed - stated) and
    a 95% interval for the observed mean (Wilson for classification, normal
    approximation for regression); ``within_95ci`` says whether the stated
    mean falls inside it. ``summary.calibration_error`` is the count-weighted
    mean |gap| (an ECE over equal-mass bins), in error units.
    """
    stated = np.asarray(expected_error, dtype=np.float64).reshape(-1)
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    if len(stated) != len(obs):
        raise ValueError('expected_error and observed must have the same length.')
    keep = np.isfinite(stated) & np.isfinite(obs)
    stated, obs = stated[keep], obs[keep]
    if mode is None:
        mode = 'cls' if np.isin(obs, (0.0, 1.0)).all() else 'reg'
    n = len(stated)
    bins = []
    if n:
        order = np.argsort(stated, kind='mergesort')
        for i, idx in enumerate(np.array_split(order, min(int(n_bins), n)), start=1):
            s, o = stated[idx], obs[idx]
            m = len(idx)
            if mode == 'cls':
                lo, hi = _wilson(float(o.sum()), m)
            else:
                half = 1.96 * (o.std(ddof=1) / math.sqrt(m)) if m > 1 else float('nan')
                lo, hi = float(o.mean() - half), float(o.mean() + half)
            bins.append({
                'bin': i, 'n': int(m),
                'stated_min': float(s.min()), 'stated_max': float(s.max()),
                'stated_mean': float(s.mean()), 'observed_mean': float(o.mean()),
                'gap': float(o.mean() - s.mean()),
                'observed_ci_low': lo, 'observed_ci_high': hi,
                'within_95ci': bool(lo <= s.mean() <= hi) if math.isfinite(lo) else None,
            })
    summary = {
        'mode': mode, 'n': int(n), 'n_bins': len(bins),
        'mean_stated': float(stated.mean()) if n else None,
        'mean_observed': float(obs.mean()) if n else None,
        'calibration_error': float(sum(b['n'] * abs(b['gap']) for b in bins) / n) if n else None,
        'max_abs_gap': float(max(abs(b['gap']) for b in bins)) if bins else None,
        'bins_within_95ci': int(sum(1 for b in bins if b['within_95ci'])),
    }
    if mode == 'cls' and n:
        summary['brier'] = float(np.mean((stated - obs) ** 2))
        summary['auroc_error_detection'] = _auroc(stated, obs)
    return {'summary': summary, 'bins': bins}


def metrics_summary(report: dict, model: Optional[ErrorModel] = None) -> dict:
    """Flat, small summary for ``full_metrics.json`` / ``inference_metrics.json``."""
    summary = dict(report['summary'])
    if model is not None:
        summary['method'] = model.method_used
        summary['fit_samples'] = model.fit_info.get('n_samples')
    return {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}


def save_report(report: dict, out_dir: PathLike, stem: str = CALIBRATION_STEM,
                extra: Optional[dict] = None) -> tuple:
    """Write ``<stem>.json`` (summary + bins [+ extra]) and ``<stem>.csv``
    (one row per bin). Returns the two paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    if extra:
        payload.update(extra)
    json_path = out_dir / f'{stem}.json'
    csv_path = out_dir / f'{stem}.csv'
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding='utf-8')
    columns = ['bin', 'n', 'stated_min', 'stated_max', 'stated_mean', 'observed_mean', 'gap',
               'observed_ci_low', 'observed_ci_high', 'within_95ci']
    pd.DataFrame(report['bins'], columns=columns).to_csv(csv_path, index=False)
    return json_path, csv_path


def observed_errors(predictions: pd.DataFrame, mode: str) -> Optional[np.ndarray]:
    """Observed per-row error from a predictions table (``correct`` for
    classification, ``abs_error`` for regression), or None if unlabelled."""
    if mode == 'cls' and 'correct' in predictions.columns:
        known = predictions['correct'].map(lambda v: np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else float(not bool(v)))
        return known.to_numpy(dtype=np.float64)
    if mode == 'reg' and 'abs_error' in predictions.columns:
        return pd.to_numeric(predictions['abs_error'], errors='coerce').to_numpy(dtype=np.float64)
    return None

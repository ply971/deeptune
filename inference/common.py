"""Shared helpers across every modality's inference module."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def summarize_predictions(predictions: pd.DataFrame, mode: str) -> Optional[dict]:
    """Aggregate accuracy (classification) or MAE/RMSE (regression) across
    `predictions`, or None if it has no ground truth to compare against."""
    if mode == 'cls' and 'correct' in predictions.columns:
        known = predictions['correct'].dropna()
        if known.empty:
            return None
        return {'accuracy': float(known.mean()), 'num_labeled': int(len(known)), 'num_samples': int(len(predictions))}
    if mode == 'reg' and 'abs_error' in predictions.columns:
        known = predictions['abs_error'].dropna()
        if known.empty:
            return None
        errors = known.astype(float)
        return {'mae': float(errors.mean()), 'rmse': float((errors ** 2).mean() ** 0.5),
                'num_labeled': int(len(known)), 'num_samples': int(len(predictions))}
    return None


# --- per-prediction error estimate (A4) ---
def attach_expected_error(result: pd.DataFrame, mode: str, checkpoint_path, scores) -> bool:
    """Adds an ``expected_error`` column to `result` (P(wrong) for
    classification, expected |error| for regression) when the run has an
    ``error_model.json`` next to `checkpoint_path`. `scores` are the class
    probabilities [N, K] (cls) or raw outputs [N] / [N, D] (reg), in the
    same row order as `result`. Old runs without the file: no column, no
    error. Returns whether the column was added. Torch-free."""
    import numpy as np
    from evaluators.error_model import load_error_model_near, prediction_features
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(result) or not len(scores):
        return False
    model = load_error_model_near(checkpoint_path, mode, num_outputs=scores.reshape(len(scores), -1).shape[1])
    if model is None:
        return False
    result['expected_error'] = model.predict(prediction_features(scores, mode))
    return True


def error_calibration_report(predictions: pd.DataFrame, mode: str) -> Optional[dict]:
    """Stated vs observed error per decile for labelled predictions that
    carry ``expected_error`` (see evaluators.error_model.calibration_table),
    or None when either is missing."""
    import numpy as np
    from evaluators.error_model import calibration_table, observed_errors
    if 'expected_error' not in predictions.columns:
        return None
    observed = observed_errors(predictions, mode)
    if observed is None or not np.isfinite(observed).any():
        return None
    return calibration_table(predictions['expected_error'].to_numpy(dtype=float), observed, mode=mode)

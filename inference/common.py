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
def attach_expected_error(result: pd.DataFrame, mode: str, checkpoint_path, scores,
                          state_dict=None) -> bool:
    """Adds an ``expected_error`` column to `result` (P(wrong) for
    classification, expected |error| for regression) when the run has an
    ``error_model.json`` next to `checkpoint_path`. `scores` are the class
    probabilities [N, K] (cls) or raw outputs [N] / [N, D] (reg), in the
    same row order as `result`. `state_dict` (the loaded weights) is
    fingerprinted so an error model fitted on other weights is refused with
    a warning. Old runs without the file: no column, no error. Returns
    whether the column was added. Torch-free."""
    import numpy as np
    from evaluators.error_model import load_error_model_near, prediction_features, weights_fingerprint
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(result) or not len(scores):
        return False
    model = load_error_model_near(checkpoint_path, mode, num_outputs=scores.reshape(len(scores), -1).shape[1],
                                  weights_sha256=None if state_dict is None else weights_fingerprint(state_dict))
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


# --- abstention (R2) ---
def attach_abstention(result: pd.DataFrame, checkpoint_path, num_classes: int, state_dict, pipeline: dict, warn=print, label_mapping: Optional[dict] = None) -> Optional[dict]:
    """Adds `answered` and `decision` ("not sure" or the predicted label) when the run has an abstention.json for these weights, and
    returns the summary for abstention_summary.json. Old runs: no columns, no error. Torch-free."""
    from evaluators.abstention import apply_rule, file_fingerprint, load_rule_near
    from evaluators.error_model import find_error_model, weights_fingerprint
    weights_sha256 = None if state_dict is None else weights_fingerprint(state_dict)
    rule = load_rule_near(checkpoint_path, num_classes, weights_sha256, warn=warn)
    if rule is None:
        return None
    summary = apply_rule(rule, result, pipeline, warn=warn, weights_sha256=weights_sha256, label_mapping=label_mapping,
                         error_model_sha256=file_fingerprint(find_error_model(checkpoint_path)))
    if summary is not None:
        warn(f"Abstention: {summary['answered']} of {summary['n_clips']} clips answered, {summary['not_sure']} \"not sure\". {summary['headline']}")
    return summary

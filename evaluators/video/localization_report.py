"""Temporal localisation metrics (Phase 5, task 6). Torch-free.

ActivityNet-style detection mAP at tIoU 0.1 to 0.9, with the THUMOS'14 averages CtxPL reports (0.1-0.5, 0.1-0.7, all),
per-class AP, classification mAP over video-level scores, and mean boundary error. The AP follows the rules of
`compute_average_precision_detection` and `interpolated_prec_rec` in the ActivityNet evaluation code, as used by CtxPL
(Hussain & Gondra, CRV 2026, MIT) in `eval/eval_detection.py`:

- predictions are taken by decreasing score; each may claim the free ground truth with the highest tIoU, at or above the
  threshold; a prediction whose candidates are all claimed, or below the threshold, is a false positive;
- a prediction in a video with no ground truth of its class is a false positive;
- segments that only touch have tIoU 0, so they do not match at any positive threshold;
- mAP averages over classes with at least one ground truth; predictions of other classes are ignored.

Tables use the reference's columns: `video-id, t-start, t-end, label`, plus `score` for predictions.

Task 7, evidential uncertainty: `validate_uncertainty` checks stated uncertainty against observed error per decile, on
held-out data. CtxPL's outputs do not exist in DeepTune yet (Phase 4), so this has only run on synthetic arrays.

    python -m evaluators.video.localization_report --ground_truth gt_segments.csv --predictions segments.csv --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from evaluators.error_model import _auroc, calibration_table

DEFAULT_TIOU_THRESHOLDS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1).tolist())
AVERAGE_RANGES = {'avg_0.1_0.5': (0.1, 0.5), 'avg_0.1_0.7': (0.1, 0.7), 'avg_0.1_0.9': (0.1, 0.9)}
SEGMENT_COLUMNS = ('video-id', 't-start', 't-end')
_UNCERTAINTY_MAX_GAP = 0.05  # largest count-weighted |stated - observed| still called calibrated
_UNCERTAINTY_MIN_ROWS = 100  # a floor, not a working size: see WORKING_ROWS_NOTE
WORKING_ROWS_NOTE = 'Calibrated data needs about 500 to 800 rows to pass this check reliably, as the weighted gap of a smaller sample is mostly noise.'
_UNCERTAINTY_MAX_BIN_GAP = 0.10  # a decile further off than this must at least have the stated value inside its joint interval


def segment_iou(segment, candidates) -> np.ndarray:
    """tIoU of one [start, end] with candidates [N, 2]. Touching segments give 0."""
    candidates = np.asarray(candidates, dtype=np.float64)
    start = np.maximum(segment[0], candidates[:, 0])
    end = np.minimum(segment[1], candidates[:, 1])
    intersection = np.clip(end - start, 0, None)
    union = (candidates[:, 1] - candidates[:, 0]) + (segment[1] - segment[0]) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def interpolated_ap(precision, recall) -> float:
    """Area under the precision envelope, summed where recall changes (VOC 2011 onwards)."""
    envelope = np.maximum.accumulate(np.concatenate([[0.0], precision, [0.0]])[::-1])[::-1]
    points = np.concatenate([[0.0], recall, [1.0]])
    changes = np.flatnonzero(points[1:] != points[:-1]) + 1
    return float(((points[changes] - points[changes - 1]) * envelope[changes]).sum())


def _check_columns(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f'The {name} table lacks the columns {missing}.')


def class_average_precision(ground_truth: pd.DataFrame, prediction: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_TIOU_THRESHOLDS) -> np.ndarray:
    """AP per threshold, for one class. No predictions gives zeros. No ground truth is the caller's problem: AP is undefined."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if prediction.empty:
        return np.zeros(len(thresholds))
    if ground_truth.empty:
        raise ValueError('AP is undefined for a class with no ground truth.')
    order = prediction['score'].to_numpy().argsort()[::-1]  # the reference's order, so ties break the same way
    videos = prediction['video-id'].to_numpy()[order]
    segments = prediction[['t-start', 't-end']].to_numpy(dtype=np.float64)[order]
    truths = {video: part[['t-start', 't-end']].to_numpy(dtype=np.float64) for video, part in ground_truth.groupby('video-id')}
    taken = {(video, j): np.zeros(len(thresholds), dtype=bool) for video, part in truths.items() for j in range(len(part))}
    true_positive = np.zeros((len(thresholds), len(order)), dtype=bool)
    for i, (video, segment) in enumerate(zip(videos, segments)):
        candidates = truths.get(video)
        if candidates is None:
            continue
        overlaps = segment_iou(segment, candidates)
        ranked = overlaps.argsort()[::-1]
        for t, threshold in enumerate(thresholds):
            for j in ranked:
                if overlaps[j] < threshold:
                    break
                if not taken[(video, j)][t]:
                    taken[(video, j)][t] = True
                    true_positive[t, i] = True
                    break
    hits = np.cumsum(true_positive, axis=1, dtype=np.float64)
    misses = np.cumsum(~true_positive, axis=1, dtype=np.float64)
    recall = hits / len(ground_truth)
    precision = hits / (hits + misses)
    return np.array([interpolated_ap(precision[t], recall[t]) for t in range(len(thresholds))])


def detection_map(ground_truth: pd.DataFrame, prediction: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_TIOU_THRESHOLDS,
                  class_names: Optional[dict] = None) -> dict:
    """mAP per threshold, the averages, and per-class AP. Fractions in [0, 1]; CtxPL's tables show them in percent."""
    _check_columns(ground_truth, (*SEGMENT_COLUMNS, 'label'), 'ground truth')
    _check_columns(prediction, (*SEGMENT_COLUMNS, 'label', 'score'), 'prediction')
    thresholds = tuple(float(t) for t in thresholds)
    labels = sorted(ground_truth['label'].unique().tolist())
    if not labels:
        raise ValueError('Detection mAP needs at least one ground-truth segment.')
    ignored = sorted(set(prediction['label'].unique().tolist()) - set(labels))
    table = np.vstack([class_average_precision(ground_truth[ground_truth['label'] == label],
                                               prediction[prediction['label'] == label], thresholds) for label in labels])
    per_threshold = table.mean(axis=0)
    averages = {}
    for name, (low, high) in AVERAGE_RANGES.items():
        keep = [i for i, t in enumerate(thresholds) if low - 1e-9 <= t <= high + 1e-9]
        averages[name] = float(per_threshold[keep].mean()) if keep else None
    return {'thresholds': list(thresholds), 'map_per_threshold': {f'{t:g}': float(v) for t, v in zip(thresholds, per_threshold)},
            'averages': averages, 'n_classes': len(labels), 'n_ground_truth': int(len(ground_truth)), 'n_predictions': int(len(prediction)),
            'ignored_prediction_labels': [str(v) for v in ignored],
            'per_class_ap': {str((class_names or {}).get(label, label)): {f'{t:g}': float(v) for t, v in zip(thresholds, row)}
                             for label, row in zip(labels, table)}}


def classification_map(scores, labels) -> dict:
    """Video-level mAP: non-interpolated AP per class over videos, as CtxPL's `getAP`. `scores`, `labels`: [videos, classes].

    Classes without a positive video have no AP and are left out (the reference divides by zero there)."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError('Expected scores and labels of one shape, [videos, classes].')
    per_class = []
    for k in range(scores.shape[1]):
        positive = labels[:, k] == 1
        if not positive.any():
            per_class.append(None)
            continue
        ranked = positive[np.argsort(-scores[:, k], kind='stable')]
        precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
        per_class.append(float(precision[ranked].sum() / ranked.sum()))
    defined = [v for v in per_class if v is not None]
    return {'map': float(np.mean(defined)) if defined else None, 'per_class_ap': per_class}


def match_segments(ground_truth: pd.DataFrame, prediction: pd.DataFrame, tiou: float = 0.5) -> pd.DataFrame:
    """Each prediction, with the rule AP uses, and its match: `matched`, `tiou`, `gt_start`, `gt_end`. Same row order as given."""
    out = prediction.reset_index(drop=True).copy()
    out['matched'], out['tiou'], out['gt_start'], out['gt_end'] = False, 0.0, np.nan, np.nan
    for label, rows in out.groupby('label'):
        truth_rows = ground_truth[ground_truth['label'] == label]
        truths = {video: part[['t-start', 't-end']].to_numpy(dtype=np.float64) for video, part in truth_rows.groupby('video-id')}
        free = {video: np.ones(len(part), dtype=bool) for video, part in truths.items()}
        for index in rows.index[rows['score'].to_numpy().argsort()[::-1]]:
            video = out.at[index, 'video-id']
            if video not in truths:
                continue
            overlaps = segment_iou(out.loc[index, ['t-start', 't-end']].to_numpy(dtype=np.float64), truths[video])
            for j in overlaps.argsort()[::-1]:
                if overlaps[j] < tiou:
                    break
                if free[video][j]:
                    free[video][j] = False
                    out.loc[index, ['matched', 'tiou', 'gt_start', 'gt_end']] = [True, float(overlaps[j]), *truths[video][j]]
                    break
    out['matched'] = out['matched'].astype(bool)
    return out


def boundary_error(ground_truth: pd.DataFrame, prediction: pd.DataFrame, tiou: float = 0.5) -> dict:
    """Mean |start error| and |end error| in seconds over predictions matched at `tiou`. Unmatched ones are not counted, so
    read this with the mAP: it says how accurate the boundaries of the found segments are, not how many were found."""
    matched = match_segments(ground_truth, prediction, tiou)
    matched = matched[matched['matched']]
    if matched.empty:
        return {'tiou': tiou, 'n_matched': 0, 'start_seconds': None, 'end_seconds': None, 'mean_seconds': None}
    start = np.abs(matched['t-start'].to_numpy(dtype=np.float64) - matched['gt_start'].to_numpy(dtype=np.float64))
    end = np.abs(matched['t-end'].to_numpy(dtype=np.float64) - matched['gt_end'].to_numpy(dtype=np.float64))
    return {'tiou': tiou, 'n_matched': int(len(matched)), 'start_seconds': float(start.mean()), 'end_seconds': float(end.mean()),
            'mean_seconds': float((start.mean() + end.mean()) / 2)}


def localization_report(ground_truth: pd.DataFrame, prediction: pd.DataFrame, video_scores=None, video_labels=None,
                        thresholds: Sequence[float] = DEFAULT_TIOU_THRESHOLDS, boundary_tiou: float = 0.5) -> dict:
    report = {'detection': detection_map(ground_truth, prediction, thresholds), 'boundary_error': boundary_error(ground_truth, prediction, boundary_tiou)}
    report['classification'] = classification_map(video_scores, video_labels) if video_scores is not None else None
    return report


# -- task 7: is evidential uncertainty an error estimate? ------------------------------------------------------------
def _gap_if_calibrated(stated: np.ndarray, n_bins: int, draws: int = 400, seed: int = 0) -> float:
    """95th percentile of the weighted gap when errors really occur at the stated rates, over the same deciles as the table."""
    rng = np.random.default_rng(seed)
    totals = np.zeros(draws)
    for rows in np.array_split(np.argsort(stated, kind='mergesort'), min(n_bins, len(stated))):
        chance = stated[rows]
        totals += np.abs((rng.random((draws, len(rows))) < chance).sum(axis=1) - chance.sum())
    return float(np.percentile(totals / len(stated), 95))


def _outside_joint_interval(rows: dict, n_bins: int) -> bool:
    """Is the stated mean outside this decile's Wilson interval, Bonferroni-corrected so that all deciles hold together at about 95%?"""
    from scipy.stats import norm
    z = norm.ppf(1 - 0.05 / (2 * n_bins))
    n, rate = rows['n'], rows['observed_mean']
    bottom = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / bottom
    half = z * np.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / bottom
    return not centre - half <= rows['stated_mean'] <= centre + half


def validate_uncertainty(uncertainty, observed_error, n_bins: int = 10, max_gap: float = _UNCERTAINTY_MAX_GAP, unit: Optional[str] = None,
                         split: Optional[str] = None, min_rows: int = _UNCERTAINTY_MIN_ROWS, max_bin_gap: float = _UNCERTAINTY_MAX_BIN_GAP) -> dict:
    """Stated uncertainty (in [0, 1]) against observed errors (1 wrong, 0 right) per decile, on held-out rows.

    `calibrated` needs all of: at least `min_rows` rows; a count-weighted gap of at most `max_gap`; and in every decile a gap
    of at most `max_bin_gap`, or the stated value inside the decile's interval (Wilson, Bonferroni-corrected so that the deciles
    hold jointly at about 95%). `calibration_failures` says which failed.

    Failing is not the same as being miscalibrated. The weighted gap of a small sample is mostly noise: truly calibrated data
    with uniform uncertainties passes about 1% of the time at 100 rows, 35% at 300 and 92% at 800. So `verdict` separates three
    outcomes: 'calibrated' (passed), 'miscalibrated' (a decile is outside its joint interval, or the weighted gap is above the
    95th percentile of what calibrated data of this size gives, `gap_if_calibrated_p95`), and 'inconclusive' (it failed, but the
    data are consistent with being calibrated). Only 'calibrated' may be attached as an error estimate.

    `ranks_errors` is the weaker property, that more `ranks_errors` is the weaker property, that more
    uncertain rows go wrong more often (AUROC above 0.5 and a positive Spearman correlation). A rank-only signal may still
    be mapped to an error rate by fitting an `ErrorModel` on held-out rows; it must not be stated as a probability as is."""
    from scipy.stats import spearmanr
    stated = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_error, dtype=np.float64).reshape(-1)
    if len(stated) != len(observed) or not len(stated):
        raise ValueError('Expected one observed error per stated uncertainty, and at least one row.')
    if not np.isin(observed, (0.0, 1.0)).all():
        raise ValueError('Observed errors must be 0 or 1.')
    if not np.isfinite(stated).all():
        raise ValueError('Uncertainty must be finite: rows without a value cannot be checked, and would still count in n.')
    if ((stated < 0) | (stated > 1)).any():
        raise ValueError('Uncertainty must lie in [0, 1].')
    table = calibration_table(stated, observed, n_bins=n_bins, mode='cls')
    gap = table['summary']['calibration_error']
    failures = []
    if len(stated) < min_rows:
        failures.append(f'{len(stated)} rows, under the minimum of {min_rows}')
    if gap > max_gap:
        failures.append(f'weighted gap {gap:.3f} is above {max_gap:g}')
    off = [b for b in table['bins'] if abs(b['gap']) > max_bin_gap and _outside_joint_interval(b, len(table['bins']))]
    if off:
        failures.append(f"{len(off)} decile(s) off by more than {max_bin_gap:g} with the stated value outside their joint 95% interval, worst {max(abs(b['gap']) for b in off):.3f}")
    null_gap = _gap_if_calibrated(stated, n_bins)
    verdict = 'calibrated' if not failures else 'miscalibrated' if off or gap > null_gap else 'inconclusive'
    both = len(set(observed.tolist())) == 2
    correlation = float(spearmanr(stated, observed).statistic) if both and np.ptp(stated) > 0 else None
    auroc = _auroc(stated, observed.astype(bool))
    return {'n': int(len(stated)), 'unit': unit, 'split': split, 'calibration_error': gap, 'max_gap_allowed': max_gap, 'min_rows': min_rows,
            'max_bin_gap_allowed': max_bin_gap, 'calibrated': not failures, 'calibration_failures': failures, 'verdict': verdict, 'gap_if_calibrated_p95': null_gap,
            'verdict_note': WORKING_ROWS_NOTE if verdict == 'inconclusive' else None, 'spearman': correlation, 'auroc_error_detection': auroc,
            'ranks_errors': bool(auroc is not None and auroc > 0.5 and correlation is not None and correlation > 0),
            'table': table}


def segment_false_positives(ground_truth: pd.DataFrame, prediction: pd.DataFrame, tiou: float = 0.5) -> np.ndarray:
    """1 for a predicted segment that matches no ground truth at `tiou`; aligned with `prediction`'s rows."""
    return (~match_segments(ground_truth, prediction, tiou)['matched'].to_numpy()).astype(np.float64)


def validate_segment_uncertainty(ground_truth: pd.DataFrame, prediction: pd.DataFrame, uncertainty, tiou: float = 0.5, split: str = 'validation') -> dict:
    """Per segment: does stated uncertainty follow the rate of false positives at `tiou`, on held-out rows named `split`?"""
    result = validate_uncertainty(uncertainty, segment_false_positives(ground_truth, prediction, tiou), unit='segment', split=split)
    return {**result, 'error': f'false positive at tIoU {tiou:g}'}


def video_label_errors(scores, labels) -> np.ndarray:
    """Per video: 1 when the top-scoring class is not one of its labels. `scores`, `labels`: [videos, classes]."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError('Expected scores and labels of one shape, [videos, classes].')
    return (labels[np.arange(len(scores)), scores.argmax(axis=1)] != 1).astype(np.float64)


def validate_video_uncertainty(uncertainty, scores, labels, split: str = 'validation') -> dict:
    """Per video: does stated uncertainty follow the rate of video-level label errors, on held-out rows named `split`?"""
    return {**validate_uncertainty(uncertainty, video_label_errors(scores, labels), unit='video', split=split), 'error': 'top class not among the labels'}


def attach_expected_error(table: pd.DataFrame, uncertainty, validation: dict, *, applied_to: str, unit: str = 'segment') -> pd.DataFrame:
    """Adds `expected_error` from the uncertainty only if `validation` found it calibrated; otherwise the table is unchanged.

    `applied_to` names the split of `table`, and `unit` what its rows are ('segment' or 'video'). `validation` must be a check
    of that same unit on another split: a check on the rows it is applied to says nothing about new rows, and a video-level
    check says nothing about segments."""
    if validation.get('unit') != unit:
        raise ValueError(f"The uncertainty was validated per {validation.get('unit')}; it cannot state an error per {unit}.")
    if not validation.get('split'):
        raise ValueError('The check does not name the rows it was made on, so it cannot be told apart from these rows.')
    if validation['split'] == applied_to:
        raise ValueError(f'The uncertainty was validated on the {applied_to} rows themselves; validate it on other held-out rows.')
    values = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if len(values) != len(table) or not np.isfinite(values).all():
        raise ValueError('Expected one finite uncertainty per row.')
    out = table.copy()
    if validation.get('calibrated'):
        out['expected_error'] = values
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Detection mAP, per-class AP and boundary error of predicted segments.')
    p.add_argument('--ground_truth', type=Path, required=True, help='CSV with video-id, t-start, t-end, label (a benchmark gt_segments.csv has these).')
    p.add_argument('--predictions', type=Path, required=True, help='CSV with video-id, t-start, t-end, label, score.')
    p.add_argument('--out', type=Path, required=True, help='Folder for localization_report.json and per_class_ap.csv.')
    p.add_argument('--boundary_tiou', type=float, default=0.5, help='tIoU at which predictions count as matched for the boundary error (default 0.5).')
    p.add_argument('--uncertainty_column', help='Column of --predictions holding per-segment uncertainty in [0, 1]: also checks it against false positives.')
    p.add_argument('--split', default='validation', help='Name of these held-out rows, recorded with the uncertainty check (default validation).')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    truth, prediction = pd.read_csv(args.ground_truth), pd.read_csv(args.predictions)
    report = localization_report(truth, prediction, boundary_tiou=args.boundary_tiou)
    if args.uncertainty_column:
        _check_columns(prediction, (args.uncertainty_column,), 'prediction')
        report['uncertainty'] = validate_segment_uncertainty(truth, prediction, prediction[args.uncertainty_column], args.boundary_tiou, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'localization_report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(report['detection']['per_class_ap']).T.rename_axis('class').to_csv(args.out / 'per_class_ap.csv')
    print(path)
    return 0


if __name__ == '__main__':
    sys.exit(main())

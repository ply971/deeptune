"""Full metric report for video runs (Phase 5). Torch-free, so the GUI may import it.

Classification: confusion matrix, accuracy, balanced accuracy, macro/weighted F1, per-class sensitivity, specificity,
PPV, NPV, one-vs-rest ROC AUC and PR-AUC, calibration before and after temperature scaling. Regression: MAE, RMSE, R2
and a residual plot. Intervals resample whole groups or videos (`evaluators/stats.py`), and each states what it covers.

    python -m evaluators.video.report --eval_dir RUN/eval_output_x --test_df RUN/data_splits_x/test_split.parquet
    python -m evaluators.video.report compare --a RUN_A/eval_output_x --b RUN_B/eval_output_y --test_df test_split.parquet

Undefined values (a zero denominator, a class with no positives) are null in JSON and empty in CSV, never 0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from evaluators.calibration import calibration_report
from evaluators.stats import DEFAULT_BOOTSTRAPS, bootstrap_ci, paired_difference_ci

REPORT_FILENAME = 'evaluation_report.json'
REPORT_SCHEMA_VERSION = 1
SCORES_FILENAME = 'probabilities.npy'
MAX_ANNOTATED_CLASSES = 20
MAX_ROC_SERIES = 8
_SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
_INK, _MUTED, _GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _ratio(top: float, bottom: float) -> Optional[float]:
    return None if bottom == 0 else float(top / bottom)


def _none_if_nan(value) -> Optional[float]:
    return None if value is None or not math.isfinite(value) else float(value)


def confusion(y, pred, num_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(y) * num_classes + np.asarray(pred), minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_table(y, pred, num_classes: int) -> list[dict]:
    """Sensitivity, specificity, PPV and NPV of each class, one against the rest. None where the denominator is 0."""
    matrix = confusion(y, pred, num_classes)
    total = matrix.sum()
    rows = []
    for k in range(num_classes):
        tp = matrix[k, k]
        fn = matrix[k].sum() - tp
        fp = matrix[:, k].sum() - tp
        tn = total - tp - fn - fp
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        rows.append({'class': k, 'support': int(tp + fn), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
                     'sensitivity': _ratio(tp, tp + fn), 'specificity': _ratio(tn, tn + fp), 'ppv': _ratio(tp, tp + fp),
                     'npv': _ratio(tn, tn + fn), 'f1': f1})
    return rows


def _confusion_scores(matrix: np.ndarray) -> dict:
    """Accuracy and the F1 family, by the same rules as scikit-learn: classes absent from both the truth and the
    predictions do not count, balanced accuracy averages over classes in the truth, a 0/0 F1 is 0."""
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = np.diag(matrix).astype(np.float64)
    present = (support + predicted) > 0
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    in_truth = support > 0
    return {'accuracy': float(tp.sum() / matrix.sum()),
            'balanced_accuracy': float((tp[in_truth] / support[in_truth]).mean()),
            'macro_f1': float(f1[present].mean()), 'weighted_f1': float((f1 * support).sum() / support.sum())}


def _class_aucs(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """One-vs-rest ROC AUC per class by the rank formula (Mann-Whitney), NaN for a class without positives or negatives."""
    from scipy.stats import rankdata
    positive = np.eye(probabilities.shape[1], dtype=bool)[y]
    ranks = rankdata(probabilities, axis=0)
    n_pos = positive.sum(axis=0).astype(np.float64)
    n_neg = len(y) - n_pos
    valid = (n_pos > 0) & (n_neg > 0)
    out = np.full(probabilities.shape[1], np.nan)
    out[valid] = ((ranks * positive).sum(axis=0)[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) / (n_pos[valid] * n_neg[valid])
    return out


def _macro(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')


def classification_metrics(y, probabilities) -> dict:
    y = np.asarray(y).reshape(-1).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    matrix = confusion(y, probabilities.argmax(axis=1), probabilities.shape[1])
    return {**_confusion_scores(matrix), 'macro_auc': _none_if_nan(_macro(_class_aucs(y, probabilities)))}


def regression_metrics(y, prediction) -> dict:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    residual = prediction - y
    total = ((y - y.mean()) ** 2).sum()
    return {'mae': float(np.abs(residual).mean()), 'rmse': float(np.sqrt((residual ** 2).mean())),
            'r2': None if len(y) < 2 or total == 0 else float(1 - (residual ** 2).sum() / total)}


def _cls_intervals(y, p, n, groups, sources, n_boot, seed) -> dict:
    def metric(name):
        def compute(idx):
            values = classification_metrics(y[idx], p[idx])[name] if name == 'macro_auc' else \
                _confusion_scores(confusion(y[idx], pred[idx], p.shape[1]))[name]
            return float('nan') if values is None else values
        return compute
    pred = p.argmax(axis=1)
    return {name: bootstrap_ci(metric(name), n, groups, sources, n_boot, seed=seed)
            for name in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auc')}


def _reg_intervals(y, prediction, n, groups, sources, n_boot, seed) -> dict:
    def metric(name):
        def compute(idx):
            value = regression_metrics(y[idx], prediction[idx])[name]
            return float('nan') if value is None else value
        return compute
    return {name: bootstrap_ci(metric(name), n, groups, sources, n_boot, seed=seed) for name in ('mae', 'rmse', 'r2')}


def roc_points(y, probabilities) -> pd.DataFrame:
    """Long table: class, fpr, tpr, threshold. Classes without positives or negatives have no curve."""
    from sklearn.metrics import roc_curve
    rows = []
    for k in range(probabilities.shape[1]):
        truth = np.asarray(y) == k
        if truth.all() or not truth.any():
            continue
        fpr, tpr, thresholds = roc_curve(truth, probabilities[:, k])
        rows.append(pd.DataFrame({'class': k, 'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['class', 'fpr', 'tpr', 'threshold'])


def _class_table(y, probabilities, class_names) -> list[dict]:
    from sklearn.metrics import average_precision_score
    num_classes = probabilities.shape[1]
    rows = per_class_table(y, probabilities.argmax(axis=1), num_classes)
    aucs = _class_aucs(y, probabilities)
    for k, row in enumerate(rows):
        row['name'] = class_names[k] if class_names is not None and k < len(class_names) else str(k)
        row['roc_auc'] = _none_if_nan(aucs[k])
        truth = y == k
        row['pr_auc'] = float(average_precision_score(truth, probabilities[:, k])) if truth.any() else None
    return rows


def _chance(y, train_labels, mode) -> Optional[dict]:
    """What a model that learned only the training labels scores on these rows."""
    if train_labels is None:
        return None
    train_labels = np.asarray(train_labels)
    if mode == 'cls':
        values, counts = np.unique(train_labels, return_counts=True)
        majority = int(values[counts.argmax()])
        constant = np.full(len(y), majority)
        size = int(max(np.max(y), majority)) + 1
        scores = _confusion_scores(confusion(y, constant, size))
        return {'predicts': f'training majority class {majority}', **{k: scores[k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1')}}
    constant = np.full(len(y), float(train_labels.mean()))
    return {'predicts': f'training mean {float(train_labels.mean()):g}', **regression_metrics(y, constant)}


def build_report(mode: str, y, scores, groups=None, sources=None, fit_scores=None, fit_labels=None, fit_split=None,
                 eval_split: str = 'test', class_names: Optional[Sequence[str]] = None, train_labels=None,
                 n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = 0, selection_used_fit_split=None) -> dict:
    """The whole report as a dict. `scores`: probabilities [N, K] for 'cls', predictions [N] for 'reg'."""
    y = np.asarray(y)
    n = len(y)
    if not n:
        raise ValueError('Cannot report on an empty set.')
    report = {'schema_version': REPORT_SCHEMA_VERSION, 'mode': mode, 'split': eval_split, 'n': int(n),
              'bootstrap_seed': int(seed)}
    if mode == 'cls':
        y = y.reshape(-1).astype(np.int64)
        p = np.asarray(scores, dtype=np.float64)
        matrix = confusion(y, p.argmax(axis=1), p.shape[1])
        report['metrics'] = classification_metrics(y, p)
        report['confusion_matrix'] = matrix.tolist()
        report['per_class'] = _class_table(y, p, class_names)
        report['macro_pr_auc'] = _none_if_nan(_macro(np.array([np.nan if r['pr_auc'] is None else r['pr_auc'] for r in report['per_class']])))
        report['intervals'] = _cls_intervals(y, p, n, groups, sources, n_boot, seed)
        report['calibration'] = calibration_report(p, y, fit_scores, fit_labels, fit_split, eval_split,
                                                   selection_used_fit_split=selection_used_fit_split)
    else:
        y = y.astype(np.float64).reshape(-1)
        prediction = np.asarray(scores, dtype=np.float64).reshape(-1)
        report['metrics'] = regression_metrics(y, prediction)
        report['intervals'] = _reg_intervals(y, prediction, n, groups, sources, n_boot, seed)
        report['calibration'] = {'note': 'not applicable to regression; see expected_error in predictions.csv'}
    report['chance'] = _chance(y, train_labels, mode)
    any_interval = next(iter(report['intervals'].values()))
    report['resampling_unit'] = any_interval['unit']
    report['n_units'] = any_interval['n_units']
    report['warnings'] = sorted({v['warning'] for v in report['intervals'].values() if v.get('warning')})
    return report


def _new_axes(width=5.0, height=4.0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width, height), dpi=150)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(_MUTED)
    axes.tick_params(colors=_MUTED, labelsize=8)
    axes.grid(color=_GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    return figure, axes


def _finish(figure, axes, title, xlabel, ylabel, path: Path, legend=False) -> None:
    axes.set_title(title, color=_INK, fontsize=10, loc='left')
    axes.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    if legend:
        axes.legend(frameon=False, fontsize=8, labelcolor=_MUTED)
    figure.tight_layout()
    figure.savefig(path)


def plot_confusion(matrix, names, path: Path) -> None:
    matrix = np.asarray(matrix)
    size = min(9.0, 3.0 + 0.3 * len(matrix))
    figure, axes = _new_axes(size, size)
    axes.grid(False)
    axes.imshow(matrix, cmap='Blues', vmin=0)
    if len(matrix) <= MAX_ANNOTATED_CLASSES:
        ticks = range(len(matrix))
        axes.set_xticks(list(ticks), names, rotation=90, fontsize=7)
        axes.set_yticks(list(ticks), names, fontsize=7)
        cut = matrix.max() / 2
        for i in ticks:
            for j in ticks:
                axes.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7,
                          color='#ffffff' if matrix[i, j] > cut else _INK)
    _finish(figure, axes, 'Confusion matrix (counts)', 'Predicted class', 'True class', path)


def plot_roc(points: pd.DataFrame, table: list[dict], path: Path) -> None:
    figure, axes = _new_axes()
    axes.plot([0, 1], [0, 1], color=_MUTED, linewidth=1, linestyle=':', label='chance')
    classes = sorted(points['class'].unique()) if len(points) else []
    if len(classes) <= MAX_ROC_SERIES:
        for slot, k in enumerate(classes):
            part = points[points['class'] == k]
            axes.plot(part['fpr'], part['tpr'], color=_SERIES[slot], linewidth=2,
                      label=f"{table[k]['name']} (AUC {table[k]['roc_auc']:.3f})")
    elif classes:  # more classes than distinct colours: the average, with every curve in roc_curves.csv
        grid = np.linspace(0, 1, 201)
        curves = [np.interp(grid, part['fpr'], part['tpr']) for _, part in points.groupby('class')]
        axes.plot(grid, np.mean(curves, axis=0), color=_SERIES[0], linewidth=2, label=f'average of {len(classes)} classes')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    _finish(figure, axes, 'ROC, one class against the rest', 'False positive rate', 'True positive rate', path, legend=True)


def plot_reliability(calibration: dict, path: Path) -> None:
    figure, axes = _new_axes()
    axes.plot([0, 1], [0, 1], color=_MUTED, linewidth=1, linestyle=':', label='perfect')
    series = [('before', _SERIES[0], f"before (ECE {calibration['before']['ece']:.3f})")]
    if calibration.get('after'):
        series.append(('after', _SERIES[1], f"after T={calibration['temperature']:.2f} (ECE {calibration['after']['ece']:.3f})"))
    for key, colour, label in series:
        rows = [r for r in calibration[key]['reliability'] if r['n']]
        axes.plot([r['mean_confidence'] for r in rows], [r['accuracy'] for r in rows], color=colour, linewidth=2,
                  marker='o', markersize=5, label=label)
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    _finish(figure, axes, f"Reliability on {calibration['eval_split']}", 'Mean confidence', 'Accuracy', path, legend=True)


def plot_residuals(y, prediction, path: Path) -> None:
    figure, axes = _new_axes()
    axes.axhline(0, color=_MUTED, linewidth=1, linestyle=':')
    axes.scatter(prediction, np.asarray(prediction) - np.asarray(y), s=14, color=_SERIES[0], alpha=0.7, linewidths=0)
    _finish(figure, axes, 'Residuals', 'Prediction', 'Prediction minus truth', path)


def _reliability_rows(calibration: dict) -> pd.DataFrame:
    rows = []
    for key in ('before', 'after'):
        for row in (calibration.get(key) or {}).get('reliability', []):
            rows.append({'scores': key, **row})
    return pd.DataFrame(rows)


def write_report(report: dict, out_dir, y=None, scores=None) -> Path:
    """evaluation_report.json, its CSV tables, and PNGs when `y` and `scores` are given."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_FILENAME
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame([{'metric': name, **{k: v[k] for k in ('point', 'low', 'high', 'level', 'unit', 'n_units', 'n_boot')}}
                  for name, v in report['intervals'].items()]).to_csv(out_dir / 'metric_intervals.csv', index=False)
    if report['mode'] == 'cls':
        table = report['per_class']
        names = [row['name'] for row in table]
        pd.DataFrame(table).to_csv(out_dir / 'per_class_metrics.csv', index=False)
        pd.DataFrame(report['confusion_matrix'], index=names, columns=names).to_csv(out_dir / 'confusion_matrix.csv')
        _reliability_rows(report['calibration']).to_csv(out_dir / 'reliability.csv', index=False)
        plot_confusion(report['confusion_matrix'], names, out_dir / 'confusion_matrix.png')
        plot_reliability(report['calibration'], out_dir / 'reliability_diagram.png')
        if scores is not None:
            points = roc_points(np.asarray(y).reshape(-1).astype(np.int64), np.asarray(scores))
            points.to_csv(out_dir / 'roc_curves.csv', index=False)
            plot_roc(points, table, out_dir / 'roc_curves.png')
    elif scores is not None:
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        prediction = np.asarray(scores, dtype=np.float64).reshape(-1)
        pd.DataFrame({'truth': y, 'prediction': prediction, 'residual': prediction - y}).to_csv(out_dir / 'residuals.csv', index=False)
        plot_residuals(y, prediction, out_dir / 'residuals.png')
    return path


# -- reading what the evaluation wrote -------------------------------------------------------------------------------
def load_eval_scores(eval_dir, mode: Optional[str] = None) -> tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    """(mode, truth, scores, predictions.csv table) from a folder TestTrainer wrote. Probabilities come from
    probabilities.npy, which holds every class; predictions.csv leaves them out above 50 classes."""
    eval_dir = Path(eval_dir)
    table = pd.read_csv(eval_dir / 'predictions.csv')
    mode = mode or ('cls' if 'true_label' in table.columns else 'reg')
    if mode == 'reg':
        if 'true_value' not in table.columns:
            raise ValueError(f'{eval_dir}: multi-output regression has no single prediction column to report on.')
        return mode, table['true_value'].to_numpy(), table['predicted_value'].to_numpy(), table
    truth = table['true_label'].to_numpy()
    npy = eval_dir / SCORES_FILENAME
    if npy.is_file():
        return mode, truth, np.load(npy), table
    columns = sorted((c for c in table.columns if c.startswith('prob_')), key=lambda c: int(c.split('_')[1]))
    if not columns:
        raise ValueError(f'{eval_dir} has neither {SCORES_FILENAME} nor prob_<k> columns, so there are no probabilities.')
    return mode, truth, table[columns].to_numpy(), table


def split_units(split_path, grouper: Optional[str] = None) -> tuple[Optional[list], Optional[list]]:
    """(group values, source paths) of a split, row by row. Bytes storage has no source path."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if split_path is None:
        return None, None
    split_path = Path(split_path)
    if split_path.suffix.lower() != '.parquet':
        frame = pd.read_csv(split_path)
        names = list(frame.columns)
        read = lambda column: frame[column]  # noqa: E731
    else:
        schema = pq.read_schema(split_path)
        names = schema.names
        read = lambda column: pd.read_parquet(split_path, columns=[column])[column]  # noqa: E731
        if 'videos' in names and not (pa.types.is_string(schema.field('videos').type) or pa.types.is_large_string(schema.field('videos').type)):
            names = [name for name in names if name != 'videos']  # clip bytes are no source path, and may be large
    column = grouper or ('group' if 'group' in names else None)
    groups = read(column).tolist() if column and column in names else None
    sources = None
    if 'videos' in names:
        values = read('videos')
        if len(values) and isinstance(values.iloc[0], str):
            sources = values.tolist()
    return groups, sources


def _class_names(mapping_path) -> Optional[list[str]]:
    """Class names by index, from split_dataset's label_mapping.json (name -> index or index -> name)."""
    if mapping_path is None or not Path(mapping_path).is_file():
        return None
    mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    pairs = {}
    for key, value in mapping.items():
        if isinstance(value, int):
            pairs[value] = str(key)
        elif str(key).lstrip('-').isdigit():
            pairs[int(key)] = str(value)
    return [pairs.get(k, str(k)) for k in range(max(pairs) + 1)] if pairs else None


def report_eval_dir(eval_dir, test_df=None, grouper=None, fit_dir=None, fit_split=None, mapping_path=None, train_df=None,
                    n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = 0, selection_used_fit_split=None) -> Path:
    """Writes the report of one evaluation folder into that folder."""
    mode, truth, scores, _ = load_eval_scores(eval_dir)
    groups, sources = split_units(test_df, grouper)
    fit_scores = fit_truth = None
    if fit_dir is not None and mode == 'cls':
        _, fit_truth, fit_scores, _ = load_eval_scores(fit_dir, mode)
    train_labels = pd.read_parquet(train_df, columns=['labels'])['labels'].to_numpy() if train_df is not None else None
    report = build_report(mode, truth, scores, groups, sources, fit_scores, fit_truth, fit_split, class_names=_class_names(mapping_path),
                          train_labels=train_labels, n_boot=n_boot, seed=seed, selection_used_fit_split=selection_used_fit_split)
    return write_report(report, eval_dir, truth, scores)


def selection_used_validation(run_dir) -> Optional[bool]:
    """Did validation choose the weights? From T1's training_run.json next to the checkpoint; None when unrecorded."""
    try:
        record = json.loads((Path(run_dir) / 'training_run.json').read_text(encoding='utf-8'))
        return record['selection']['validation_used_for_model_selection']
    except (OSError, ValueError, KeyError, TypeError):
        return None


# -- paired comparison of two models on one test set -----------------------------------------------------------------
_COMPARE_METRICS = {'cls': ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auc'), 'reg': ('mae', 'rmse', 'r2')}


def compare_models(mode: str, y, scores_a, scores_b, groups=None, sources=None, n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = 0) -> dict:
    """Paired per-unit difference A - B, both resampled together. Two separate intervals would ignore that the models were
    scored on the same clips, which usually makes the paired one much narrower."""
    y = np.asarray(y)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    if len(scores_a) != len(y) or len(scores_b) != len(y):
        raise ValueError('Both models must have one prediction per test row.')

    def metric(scores, name):
        def compute(idx):
            value = (classification_metrics(y[idx], scores[idx]) if mode == 'cls' else regression_metrics(y[idx], scores[idx]))[name]
            return float('nan') if value is None else value
        return compute
    return {'mode': mode, 'n': int(len(y)), 'difference': 'A minus B',
            'metrics': {name: paired_difference_ci(metric(scores_a, name), metric(scores_b, name), len(y), groups, sources, n_boot, seed=seed)
                        for name in _COMPARE_METRICS[mode]}}


def compare_eval_dirs(dir_a, dir_b, test_df=None, grouper=None, n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = 0) -> dict:
    """Refuses two folders that were not scored on the same rows in the same order."""
    mode, truth_a, scores_a, table_a = load_eval_scores(dir_a)
    mode_b, truth_b, scores_b, table_b = load_eval_scores(dir_b)
    if mode != mode_b or len(truth_a) != len(truth_b) or not np.array_equal(truth_a, truth_b):
        raise ValueError('These two evaluations were not made on one test set: their rows or labels differ.')
    groups, sources = split_units(test_df, grouper)
    return compare_models(mode, truth_a, scores_a, scores_b, groups, sources, n_boot, seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Metric report of an evaluation folder, or a paired comparison of two (first word: compare).')
    p.add_argument('--eval_dir', type=Path, help='Folder with predictions.csv.')
    p.add_argument('--a', type=Path, help='compare: first evaluation folder.')
    p.add_argument('--b', type=Path, help='compare: second evaluation folder.')
    p.add_argument('--test_df', type=Path, help='The evaluated split; its group column sets the resampling unit.')
    p.add_argument('--grouper', help="Group column of --test_df (default 'group').")
    p.add_argument('--fit_dir', type=Path, help='Evaluation folder of the validation or out-of-fold rows to fit the temperature on.')
    p.add_argument('--fit_split', help='Name of the --fit_dir rows, e.g. validation. Must not be test.')
    p.add_argument('--mapping', type=Path, help='label_mapping.json, for class names.')
    p.add_argument('--train_df', type=Path, help='Training split, for the chance row.')
    p.add_argument('--n_boot', type=int, default=DEFAULT_BOOTSTRAPS, help=f'Bootstrap resamples (default {DEFAULT_BOOTSTRAPS}).')
    p.add_argument('--seed', type=int, default=0, help='Bootstrap seed (default 0).')
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    comparing = bool(argv) and argv[0] == 'compare'
    args = build_parser().parse_args(argv[1:] if comparing else argv)
    if comparing:
        if not (args.a and args.b):
            raise SystemExit('compare needs --a and --b.')
        print(json.dumps(compare_eval_dirs(args.a, args.b, args.test_df, args.grouper, args.n_boot, args.seed), indent=2))
        return 0
    if not args.eval_dir:
        raise SystemExit('--eval_dir is required.')
    if args.fit_dir and not args.fit_split:
        raise SystemExit('--fit_dir needs --fit_split, to record which rows the temperature was fitted on.')
    print(report_eval_dir(args.eval_dir, args.test_df, args.grouper, args.fit_dir, args.fit_split, args.mapping, args.train_df,
                          args.n_boot, args.seed))
    return 0


if __name__ == '__main__':
    sys.exit(main())

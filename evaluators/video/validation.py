"""Validation schemes: stratified / grouped k-fold and Efron's .632+ bootstrap (Phase 5, task 1). Torch-free.

The caller passes only the rows it may use (train + validation); the sealed test set never reaches this module. Each fold
or resample trains a fresh model through `fit_predict(train_rows, val_rows, eval_rows, name)`, which returns one score per
evaluation row. Validation rows (for the Trainer's logging, best-epoch selection and error model) are split from the fit
rows, so scored rows are out-of-sample also when validation selects the checkpoint. Whole groups move together everywhere.

A fold model trains on (k-1)/k of the rows, so these estimates lean pessimistic for a model trained on them all.

`oof_predictions.csv` is a stable format (`OOF_SCHEMA_VERSION`, board INTERFACES), read by the label-issue finder and active
learning; use `load_oof_predictions`.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from evaluators.calibration import apply_temperature, brier_score, expected_calibration_error, fit_temperature, negative_log_likelihood
from evaluators.stats import DEFAULT_BOOTSTRAPS, bootstrap_ci, fold_summary, unit_labels
from evaluators.video.report import classification_metrics, regression_metrics

OOF_SCHEMA_VERSION = 1
OOF_FILENAME = 'oof_predictions.csv'
SUMMARY_FILENAME = 'cv_summary.json'
ASSIGNMENTS_FILENAME = 'assignments.csv'
FOLD_METRICS_FILENAME = 'fold_metrics.csv'
INNER_VALIDATION_FRACTION = 0.1
SCHEMES = ('holdout', 'kfold', 'bootstrap632')
# The README repeats these words; a test holds the two together.
KFOLD_CAVEATS = ('Each fold model trains on fewer rows than the final model, with 10% of its rows held out for checkpoint selection, so these estimates lean pessimistic for the final model.',)
BOOTSTRAP_CAVEATS = ("Group resampling is an approximation: the .632 weight is derived for independent units, and groups of unequal size make it rougher.",
                     'Resample models train on about 57% distinct units (0.632 x 0.9), with held-out checkpoint selection.',
                     'The apparent model trains on 100% of the rows and selects on its own training rows, so err and Err1 come from differently trained models.')
ENSEMBLE_NOTE = ('Probabilities averaged over the resamples that did not see each row, about 37% of them. This describes an ensemble, which '
                 'usually scores above any one model. It is not the estimate: use out_of_fold and bootstrap632.')
ERROR_MODEL_BOOTSTRAP_NOTE = ('Not fitted in bootstrap mode: the averaged out-of-bag probabilities describe an ensemble, not one model. '
                              'Use --validation kfold, or the error model of the final run, fitted on validation.')


# -- splitting -------------------------------------------------------------------------------------------------------
def _as_groups(n: int, groups) -> Optional[np.ndarray]:
    if groups is None:
        return None
    values = pd.Series(list(groups))
    if len(values) != n or values.isna().any():
        raise ValueError('Every row needs a group value.')
    return pd.factorize(values)[0]


def kfold_splits(labels, k: int = 5, groups=None, mode: str = 'cls', seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """(fit positions, held-out positions) per fold: stratified for classification, grouped when `groups` is given."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
    labels = np.asarray(labels)
    n = len(labels)
    k = int(k)
    if k < 2:
        raise ValueError('k-fold needs k of at least 2.')
    ids = _as_groups(n, groups)
    if ids is not None:
        n_groups = len(np.unique(ids))
        if n_groups < k:
            raise ValueError(f'{k}-fold needs at least {k} groups; found {n_groups}. Lower --k or add groups.')
        if mode == 'cls':
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(n), labels, groups=ids))
    elif mode == 'cls':
        values, counts = np.unique(labels, return_counts=True)
        short = values[counts < k]
        if len(short):
            raise ValueError(f'{k}-fold needs at least {k} clips per class; too few in classes {short.tolist()[:10]}. Lower --k or add clips.')
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), labels))
    else:
        if n < k:
            raise ValueError(f'{k}-fold needs at least {k} rows; found {n}.')
        splits = list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n)))
    return [(np.sort(fit), np.sort(held)) for fit, held in splits]


def inner_validation_split(labels, positions, groups=None, mode: str = 'cls', seed: int = 42,
                           fraction: float = INNER_VALIDATION_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    """Split `positions` into (train, validation), keeping groups whole. Repeated positions (a bootstrap sample) stay on
    one side, since a row on both sides would make validation partly training data."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    labels = np.asarray(labels)
    positions = np.asarray(positions)
    ids = _as_groups(len(labels), groups)
    units = ids[positions] if ids is not None else positions
    if len(np.unique(units)) < 2:
        raise ValueError('Too few distinct rows or groups to set validation rows aside. Add data, or lower --k.')
    parts = max(2, min(int(round(1 / fraction)), len(np.unique(units))))
    kept = None
    if mode == 'cls':
        # The stratified splitter refuses more parts than the largest class has rows, so a small dataset sets more aside.
        largest = int(np.unique(labels[positions], return_counts=True)[1].max())
        if largest >= 2:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='The least populated class', category=UserWarning)
                # A class may have too few members for every part: the split then leaves it out of some, which is only a logging cost.
                kept = next(StratifiedGroupKFold(n_splits=min(parts, largest), shuffle=True, random_state=seed).split(positions, labels[positions], groups=units))
            if not len(kept[0]) or not len(kept[1]):
                kept = None
    if kept is None:  # regression, or one row per class: nothing to stratify on
        kept = next(GroupShuffleSplit(n_splits=1, test_size=1 / parts, random_state=seed).split(positions, groups=units))
    return positions[kept[0]], positions[kept[1]]


def bootstrap_resamples(labels, n_boot: int = 50, groups=None, mode: str = 'cls', seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """(in-bag positions with repeats, out-of-bag positions) per resample. Draws whole groups (P-117): a clip-level bootstrap
    would put clips of one group both in and out of the bag. Draws with an empty out-of-bag set, or with no training example
    of every class seen in the data's bag, are redrawn."""
    labels = np.asarray(labels)
    n = len(labels)
    ids = _as_groups(n, groups)
    units = ids if ids is not None else np.arange(n)
    names = np.unique(units)
    if len(names) < 2:
        raise ValueError('The bootstrap needs at least two rows or groups.')
    rows = {name: np.flatnonzero(units == name) for name in names}
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < int(n_boot):
        attempts += 1
        if attempts > 100 * int(n_boot):
            raise ValueError('Could not draw bootstrap samples holding at least two classes and an out-of-bag row. Add data.')
        drawn = rng.choice(names, size=len(names), replace=True)
        in_bag = np.concatenate([rows[name] for name in drawn])
        out_bag = np.setdiff1d(np.arange(n), in_bag)
        if not len(out_bag) or (mode == 'cls' and len(np.unique(labels[in_bag])) < 2):
            continue
        out.append((np.sort(in_bag), out_bag))
    return out


# -- .632+ -----------------------------------------------------------------------------------------------------------
def no_information_error(truth, predictions, mode: str = 'cls') -> float:
    """Efron & Tibshirani's gamma: the error if truth and prediction were independent.
    Classification: sum_k p_k (1 - q_k). Regression (squared error): the mean of (y_i - f_j)^2 over all pairs."""
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    if mode == 'cls':
        classes = np.union1d(truth, predictions)
        p = np.array([(truth == c).mean() for c in classes])
        q = np.array([(predictions == c).mean() for c in classes])
        return float((p * (1 - q)).sum())
    y, f = truth.astype(np.float64), predictions.astype(np.float64)
    return float((y ** 2).mean() + (f ** 2).mean() - 2 * y.mean() * f.mean())


def efron_632_plus(apparent_error: float, oob_error: float, no_information: float) -> dict:
    """Eq. 32 of Efron & Tibshirani, JASA 1997. `oob_error` is the leave-one-out bootstrap error Err^(1).
    Err1' = min(Err1, gamma); R' = (Err1' - err) / (gamma - err) when Err1 > err and gamma > err, else 0;
    Err.632 = .368 err + .632 Err1, with the unclipped Err1;
    Err.632+ = Err.632 + (Err1' - err) * (.368 * .632 * R') / (1 - .368 * R')."""
    err, err1, gamma = float(apparent_error), float(oob_error), float(no_information)
    clipped = min(err1, gamma)
    rate = (clipped - err) / (gamma - err) if err1 > err and gamma > err else 0.0
    rate = float(np.clip(rate, 0, 1))
    weight = 0.632 / (1 - 0.368 * rate)
    error_632 = 0.368 * err + 0.632 * err1
    estimate = error_632 + (clipped - err) * (0.368 * 0.632 * rate) / (1 - 0.368 * rate)
    return {'apparent_error': err, 'oob_error': err1, 'no_information_error': gamma, 'relative_overfitting': rate, 'weight': weight,
            'error_632': error_632, 'error_632_plus': float(estimate)}


# -- the runner ------------------------------------------------------------------------------------------------------
@dataclass
class FoldScores:
    """What `fit_predict` returns. `scores`: probabilities [N, K] or predictions [N], in `eval_rows` order."""
    scores: np.ndarray
    expected_error: Optional[np.ndarray] = None


FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, str], FoldScores]


def _metrics(mode, truth, scores) -> dict:
    return classification_metrics(truth, scores) if mode == 'cls' else regression_metrics(truth, scores)


def _errors(mode, truth, scores) -> np.ndarray:
    return (scores.argmax(axis=1) != truth).astype(np.float64) if mode == 'cls' else (scores.reshape(-1) - truth) ** 2


def _check_scores(result: FoldScores, n: int, mode: str, name: str) -> np.ndarray:
    scores = np.asarray(result.scores, dtype=np.float64)
    if len(scores) != n or (mode == 'cls' and scores.ndim != 2):
        raise ValueError(f'{name}: expected {n} scores; got shape {scores.shape}.')
    return scores if mode == 'cls' else scores.reshape(-1)


def _table(ids: Sequence, labels, groups, row_ids) -> pd.DataFrame:
    table = pd.DataFrame({'row_id': row_ids, 'video_id': list(ids), 'target': labels})
    table['group'] = list(groups) if groups is not None else None
    return table


def _oof_frame(base: pd.DataFrame, rows: np.ndarray, mode: str, scores: np.ndarray, name, models: Optional[np.ndarray] = None,
               expected_error: Optional[np.ndarray] = None) -> pd.DataFrame:
    frame = base.iloc[rows].reset_index(drop=True)
    frame.insert(1, 'fold', name)
    frame['n_models'] = 1 if models is None else models
    if mode == 'cls':
        frame['target'] = frame['target'].astype(np.int64)
        frame['prediction'] = scores.argmax(axis=1)
        frame['confidence'] = scores.max(axis=1)
        frame['correct'] = frame['prediction'] == frame['target']
    else:
        frame['target'] = frame['target'].astype(np.float64)
        frame['prediction'] = scores
        frame['abs_error'] = np.abs(scores - frame['target'])
    frame['expected_error'] = expected_error if expected_error is not None else np.nan
    if mode == 'cls':
        probabilities = pd.DataFrame(scores, columns=[f'prob_{k}' for k in range(scores.shape[1])])
        frame = pd.concat([frame, probabilities], axis=1)
    return frame


def _oof_calibration(truth, scores, folds) -> dict:
    """Before/after on out-of-fold rows. Each fold's rows use a temperature fitted on the other folds, so none is scored
    with a temperature fitted on itself. `temperature_all_folds` is the one to use on new data."""
    after = np.empty_like(scores)
    used = {}
    for fold in np.unique(folds):
        here = folds == fold
        used[str(fold)] = fit_temperature(scores[~here], truth[~here])
        after[here] = apply_temperature(scores[here], used[str(fold)])
    return {'eval_rows': 'out-of-fold', 'fit_rows': 'the other folds, for each fold', 'temperature_per_fold': used,
            'temperature_all_folds': fit_temperature(scores, truth),
            'before': {'ece': expected_calibration_error(scores, truth), 'brier': brier_score(scores, truth), 'nll': negative_log_likelihood(scores, truth)},
            'after': {'ece': expected_calibration_error(after, truth), 'brier': brier_score(after, truth), 'nll': negative_log_likelihood(after, truth)},
            'caveat': 'pools probabilities of several fold models, which were not trained on all the rows',
            'applied_at_inference': False}


def _per_resample_scores(long: pd.DataFrame, mode: str, fold_table: pd.DataFrame, names, groups, sources, n_boot, seed) -> dict:
    """The bootstrap's headline: each metric scored on each resample's own out-of-bag rows, then averaged, as Err^(1) is.

    The intervals resample units over each row's mean loss across the models that did not see it, so accuracy's interval is
    about 1 - Err^(1). Metrics that do not decompose per row (F1, AUC) have no interval here: their spread is in across_folds."""
    means = {name: fold_summary(fold_table[name].tolist())['mean'] for name in names}
    grouped = long.groupby('row_id', sort=True)
    truth = grouped['target'].first().to_numpy()
    if mode == 'cls':
        hit = grouped['correct'].mean().to_numpy(dtype=np.float64)

        def balanced(idx):
            return float(np.mean([hit[idx][truth[idx] == c].mean() for c in np.unique(truth[idx])]))
        metrics = {'accuracy': lambda idx: float(hit[idx].mean()), 'balanced_accuracy': balanced}
    else:
        difference = long['prediction'] - long['target']
        absolute = difference.abs().groupby(long['row_id'], sort=True).mean().to_numpy()
        squared = (difference ** 2).groupby(long['row_id'], sort=True).mean().to_numpy()

        def r2(idx):
            total = ((truth[idx] - truth[idx].mean()) ** 2).sum()
            return float('nan') if total == 0 else float(1 - squared[idx].sum() / total)
        metrics = {'mae': lambda idx: float(absolute[idx].mean()), 'rmse': lambda idx: float(np.sqrt(squared[idx].mean())), 'r2': r2}
    intervals = {name: bootstrap_ci(metric, len(truth), groups, sources, n_boot, seed=seed) for name, metric in metrics.items()}
    for interval in intervals.values():
        interval['covers'] = 'which units were drawn, over each row\'s mean loss across the resample models; not training variability'
    return {'metrics': means, 'intervals': intervals, 'scored': 'per resample, on its own out-of-bag rows, then averaged, matching oob_error'}


def _oof_intervals(mode, truth, scores, groups, sources, n_boot, seed) -> dict:
    def metric(name):
        def compute(idx):
            value = _metrics(mode, truth[idx], scores[idx])[name]
            return float('nan') if value is None else value
        return compute
    names = ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auc') if mode == 'cls' else ('mae', 'rmse', 'r2')
    return {name: bootstrap_ci(metric(name), len(truth), groups, sources, n_boot, seed=seed) for name in names}


def fit_oof_error_model(oof: pd.DataFrame, mode: str, out_dir, seed: int = 42) -> dict:
    """Task 4: the error model fitted on out-of-fold predictions, validated on rows it did not see.

    Validation is cross-fitted: half of the units fit a model that states an error for the other half, and the other way
    round; the deciles compare those statements with what happened. The saved `error_model_oof.json` is then fitted on all
    rows. It describes fold models, trained on fewer rows than a final model, so a final model may be a little better than
    it states. It is not applied automatically: the run's `error_model.json` remains the one fitted on validation."""
    from evaluators.error_model import ErrorModel, calibration_table, prediction_features, save_report
    out_dir = Path(out_dir)
    probability_columns = sorted((c for c in oof.columns if c.startswith('prob_')), key=lambda c: int(c.split('_')[1]))
    scores = oof[probability_columns].to_numpy() if mode == 'cls' else oof['prediction'].to_numpy()
    target = (~oof['correct'].to_numpy()).astype(np.float64) if mode == 'cls' else oof['abs_error'].to_numpy()
    num_outputs = len(probability_columns) if mode == 'cls' else 1
    features = prediction_features(scores, mode)
    units, _ = unit_labels(len(oof), None if oof['group'].isna().any() else oof['group'].tolist())
    names = np.random.default_rng(seed).permutation(np.unique(units))
    half = np.isin(units, names[:len(names) // 2])
    stated = np.full(len(oof), np.nan)
    if half.any() and (~half).any():
        for fit_side in (half, ~half):
            model = ErrorModel(mode).fit(features[fit_side].reset_index(drop=True), target[fit_side], num_outputs=num_outputs)
            stated[~fit_side] = model.predict(features[~fit_side].reset_index(drop=True))
    final = ErrorModel(mode).fit(features, target, num_outputs=num_outputs)
    final.fit_info.update(split='out-of-fold', validated_by='cross-fitting on two halves of the units')
    final.save(out_dir / 'error_model_oof.json')
    report = calibration_table(stated, target, mode=mode)
    save_report(report, out_dir, 'error_calibration_oof', extra={'split': 'out-of-fold, cross-fitted', 'method': final.method_used})
    return {'method': final.method_used, 'fallback_reason': final.fit_info.get('fallback_reason'), 'held_out_summary': report['summary']}


def run_validation(scheme: str, labels, fit_predict: FitPredict, out_dir, mode: str = 'cls', ids: Optional[Sequence] = None, groups=None,
                   sources=None, k: int = 5, n_boot: int = 50, seed: int = 42, ci_boot: int = DEFAULT_BOOTSTRAPS,
                   row_ids: Optional[Sequence] = None, apparent_fit_predict: Optional[FitPredict] = None,
                   group_source: Optional[str] = None) -> dict:
    """Runs the scheme on these rows and writes cv_summary.json, fold_metrics.csv, assignments.csv and oof_predictions.csv."""
    if scheme not in ('kfold', 'bootstrap632'):
        raise ValueError(f'Unknown validation scheme {scheme!r}.')
    labels = np.asarray(labels)
    n = len(labels)
    truth = labels.astype(np.int64) if mode == 'cls' else labels.astype(np.float64)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_ids = np.arange(n) if row_ids is None else np.asarray(row_ids)
    base = _table(ids if ids is not None else [str(i) for i in row_ids], truth, groups, row_ids)
    splits = kfold_splits(labels, k, groups, mode, seed) if scheme == 'kfold' else bootstrap_resamples(labels, n_boot, groups, mode, seed)
    assignment_rows, fold_rows, frames = [], [], []
    for i, (fit, held) in enumerate(splits):
        name = f"{'fold' if scheme == 'kfold' else 'resample'}_{i}"
        train_rows, val_rows = inner_validation_split(labels, fit, groups, mode, seed + i)
        for role, positions in (('train', train_rows), ('validation', val_rows), ('heldout', held)):
            assignment_rows.append(pd.DataFrame({'fold': i, 'role': role, 'row_id': row_ids[positions]}))
        result = fit_predict(train_rows, val_rows, held, name)
        scores = _check_scores(result, len(held), mode, name)
        fold_rows.append({'fold': i, 'n_train': int(len(train_rows)), 'n_validation': int(len(val_rows)), 'n_heldout': int(len(held)), **_metrics(mode, truth[held], scores)})
        frames.append(_oof_frame(base, held, mode, scores, i, expected_error=result.expected_error))
    pd.concat(assignment_rows, ignore_index=True).to_csv(out_dir / ASSIGNMENTS_FILENAME, index=False)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(out_dir / FOLD_METRICS_FILENAME, index=False)
    names = [c for c in fold_table.columns if c not in ('fold', 'n_train', 'n_validation', 'n_heldout')]
    long = pd.concat(frames, ignore_index=True)
    if scheme == 'bootstrap632':
        long.to_csv(out_dir / 'oob_predictions_all_resamples.csv', index=False)
        oof = _mean_over_resamples(long, mode)
    else:
        oof = long.sort_values('row_id', kind='stable').reset_index(drop=True)
    oof.attrs['schema_version'] = OOF_SCHEMA_VERSION
    oof.to_csv(out_dir / OOF_FILENAME, index=False)
    oof_truth = oof['target'].to_numpy()
    oof_scores = oof[[c for c in oof.columns if c.startswith('prob_')]].to_numpy() if mode == 'cls' else oof['prediction'].to_numpy()
    positions = pd.Series(np.arange(n), index=row_ids).loc[oof['row_id']].to_numpy()
    oof_groups = None if groups is None else np.asarray(list(groups), dtype=object)[positions].tolist()
    oof_sources = None if sources is None else np.asarray(list(sources), dtype=object)[positions].tolist()
    summary = {'schema_version': OOF_SCHEMA_VERSION, 'scheme': scheme, 'mode': mode, 'seed': int(seed), 'n_rows': int(n), 'n_scored_rows': int(len(oof)),
               'grouped': groups is not None, 'group_source': (group_source or 'column') if groups is not None else None,
               'k': int(k) if scheme == 'kfold' else None, 'n_boot': int(n_boot) if scheme != 'kfold' else None,
               'caveats': list(KFOLD_CAVEATS if scheme == 'kfold' else BOOTSTRAP_CAVEATS),
               'rows_used': 'train + validation rows; the sealed test rows are not given to this scheme',
               'per_fold': fold_rows, 'across_folds': {name: fold_summary(fold_table[name].tolist()) for name in names},
               'how_to_read': ('out_of_fold.intervals: uncertainty from which units were drawn, for models of this kind. across_folds: '
                               'spread between models; its t interval is too narrow, as folds share data.')}
    if scheme == 'kfold':
        summary['out_of_fold'] = {'metrics': _metrics(mode, oof_truth, oof_scores), 'scored': 'all out-of-fold rows pooled',
                                  'intervals': _oof_intervals(mode, oof_truth, oof_scores, oof_groups, oof_sources, ci_boot, seed)}
        summary['error_model'] = fit_oof_error_model(oof, mode, out_dir, seed)
    else:
        summary['out_of_fold'] = _per_resample_scores(long, mode, fold_table, names, oof_groups, oof_sources, ci_boot, seed)
        summary['oob_ensemble'] = {'metrics': _metrics(mode, oof_truth, oof_scores), 'note': ENSEMBLE_NOTE}
        summary['error_model'] = {'method': None, 'note': ERROR_MODEL_BOOTSTRAP_NOTE}
    if mode == 'cls':
        summary['calibration'] = _oof_calibration(oof_truth, oof_scores, oof['fold'].to_numpy()) if scheme == 'kfold' else None
    if scheme == 'bootstrap632':
        summary['bootstrap632'] = _bootstrap_summary(labels, truth, long, mode, n, groups, seed, apparent_fit_predict, oof)
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2, default=_json_default), encoding='utf-8')
    return summary


def _json_default(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f'{type(value).__name__} is not JSON serialisable.')


def _mean_over_resamples(long: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per pool row: the average of the models that did not see it, so P-107 and P-109 can read one format.
    These are ensemble probabilities (see ENSEMBLE_NOTE), not one model's; `n_models` says how many were averaged."""
    probability_columns = sorted((c for c in long.columns if c.startswith('prob_')), key=lambda c: int(c.split('_')[1]))
    numeric = probability_columns if mode == 'cls' else ['prediction']
    keep = ['row_id', 'video_id', 'target', 'group']
    grouped = long.groupby('row_id', sort=True)
    frame = grouped[keep[1:]].first().reset_index()
    frame.insert(1, 'fold', -1)
    frame['n_models'] = grouped.size().to_numpy()
    means = grouped[numeric].mean().reset_index(drop=True)
    if mode == 'cls':
        scores = means.to_numpy()
        frame['prediction'] = scores.argmax(axis=1)
        frame['confidence'] = scores.max(axis=1)
        frame['correct'] = frame['prediction'] == frame['target']
    else:
        frame['prediction'] = means['prediction']
        frame['abs_error'] = np.abs(frame['prediction'] - frame['target'])
    frame['expected_error'] = np.nan
    return pd.concat([frame, means], axis=1) if mode == 'cls' else frame


def _bootstrap_summary(labels, truth, long, mode, n, groups, seed, apparent_fit_predict, oof) -> dict:
    # Err^(1): per row, its mean loss over the models that did not see it; then the mean over rows.
    if mode == 'cls':
        long = long.assign(_loss=(~long['correct']).astype(np.float64))
    else:
        long = long.assign(_loss=(long['prediction'] - long['target']) ** 2)
    oob_error = float(long.groupby('row_id')['_loss'].mean().mean())
    out = {'loss': '0-1 error' if mode == 'cls' else 'squared error', 'resampled_units': 'groups' if groups is not None else 'rows',
           'rows_never_out_of_bag': int(n - len(oof)), 'oob_error': oob_error, 'error_632_plus': None}
    if apparent_fit_predict is None:
        out['note'] = 'no model on all the rows was given, so .632+ was not computed'
        return out
    every = np.arange(n)
    train_rows, val_rows = every, every  # the apparent error is on the rows trained on, by definition; see BOOTSTRAP_CAVEATS
    result = apparent_fit_predict(train_rows, val_rows, every, 'apparent')
    scores = _check_scores(result, n, mode, 'apparent')
    apparent = float(_errors(mode, truth, scores).mean())
    predicted = scores.argmax(axis=1) if mode == 'cls' else scores
    out.update(efron_632_plus(apparent, oob_error, no_information_error(truth, predicted, mode)))
    return out


def load_oof_predictions(path, require_schema: int = OOF_SCHEMA_VERSION) -> pd.DataFrame:
    """Reads and checks an `oof_predictions.csv`. `path`: the file or its folder. Every row is out of sample by construction."""
    path = Path(path)
    if path.is_dir():
        path = path / OOF_FILENAME
    summary_path = path.parent / SUMMARY_FILENAME
    if summary_path.is_file():
        version = json.loads(summary_path.read_text(encoding='utf-8')).get('schema_version')
        if version != require_schema:
            raise ValueError(f'{path}: schema version {version}, this code reads {require_schema}.')
    table = pd.read_csv(path)
    missing = [c for c in ('row_id', 'fold', 'video_id', 'group', 'target', 'prediction', 'n_models', 'expected_error') if c not in table.columns]
    if missing:
        raise ValueError(f'{path} is not an out-of-fold table: it lacks {missing}.')
    if table['row_id'].duplicated().any():
        raise ValueError(f'{path} has more than one row for some row_id.')
    return table

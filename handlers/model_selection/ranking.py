"""Reading candidate results and ranking them.

A candidate's score is the mean of its metric over the k folds of grouped cross-validation on the development rows
(`cv_summary.json`, written by evaluators/video/validation.py). The leaderboard also shows the fold standard deviation and,
where B3 provides one, the group-bootstrap interval of the pooled out-of-fold metric.

The paired difference of the top two uses their out-of-fold predictions on the same rows: both candidates ran the same
folds (same development file and seed), so each row was scored by one model of each. `evaluators.video.report.compare_models`
resamples whole groups for both at once. The interval covers which units were drawn, not training variability, and it is
not corrected for having picked these two out of many.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from handlers.model_selection.grid import AXES

CANDIDATE_FIELDS = tuple(AXES.values())

# metric -> +1 if higher is better, -1 if lower is better
METRICS = {'cls': {'accuracy': 1, 'balanced_accuracy': 1, 'macro_f1': 1, 'weighted_f1': 1, 'macro_auc': 1},
           'reg': {'mae': -1, 'rmse': -1, 'r2': 1}}
DEFAULT_METRIC = {'cls': 'accuracy', 'reg': 'mae'}
WINNERS_CURSE = ("The winner's cross-validation score is optimistically biased: it was chosen as the best of several "
                 "noisy scores on the same rows (the winner's curse). The sealed-test score is not, as no choice looked at it.")


def find_cv_summary(run_dir: Path) -> Optional[Path]:
    found = sorted(Path(run_dir).glob('*/validation_kfold/cv_summary.json'), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def read_cv_result(run_dir: Path, out_root: Path) -> dict:
    """What the ledger keeps of a finished cross-validation: every fold's metrics and the pooled intervals."""
    path = find_cv_summary(run_dir)
    if path is None:
        raise FileNotFoundError(f'No validation_kfold/cv_summary.json under {run_dir}.')
    summary = json.loads(path.read_text(encoding='utf-8'))
    intervals = summary.get('out_of_fold', {}).get('intervals', {})
    oof = pd.read_csv(path.parent / 'oof_predictions.csv', usecols=['row_id', 'fold', 'video_id', 'target'])
    return {'folds': summary['per_fold'],
            'out_of_fold': summary.get('out_of_fold', {}).get('metrics', {}),
            'intervals': {name: {key: value.get(key) for key in ('low', 'high', 'level', 'unit', 'n_units', 'warning')}
                          for name, value in intervals.items()},
            'grouped': summary.get('grouped'), 'k': summary.get('k'), 'fold_signature': fold_signature(oof),
            'oof_path': str((path.parent / 'oof_predictions.csv').relative_to(out_root))}


def fold_signature(oof: pd.DataFrame) -> str:
    """A hash of which row went to which fold. Candidates are only comparable, and only paired, on identical folds."""
    table = oof[['row_id', 'fold', 'video_id', 'target']].sort_values('row_id', kind='stable').astype(str)
    return hashlib.sha256(table.to_csv(index=False).encode('utf-8')).hexdigest()[:16]


def fold_mismatch(entries: list) -> Optional[str]:
    """Why these finished candidates cannot be ranked against each other, or None."""
    signatures = {}
    for entry in entries:
        signatures.setdefault(entry.get('fold_signature'), []).append(entry.get('label', entry['key']))
    if len(signatures) <= 1:
        return None
    parts = '; '.join(f'{len(labels)} on folds {signature} (e.g. {labels[0]})' for signature, labels in signatures.items())
    return (f'the finished candidates were not cross-validated on the same folds ({parts}). Their scores come from different '
            'splits of the development rows, so ranking or pairing them would compare unlike things.')


def fold_stats(entry: dict, metric: str) -> tuple[Optional[float], Optional[float]]:
    values = [fold.get(metric) for fold in entry.get('folds', [])]
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def _augmentation_column(config: dict) -> str:
    """What the augmentation column says: the value, or that the search's rule left it out (a frozen backbone; R3b)."""
    from handlers.model_selection.grid import augmentation_applies
    value = config.get('augmentation', 'off')
    try:
        applies = augmentation_applies(config.get('train_mode', 'frozen'))
    except ValueError:
        applies = True
    return value if applies else 'not searched (frozen-backbone rule)'


def leaderboard(entries: list, metric: str, mode: str) -> pd.DataFrame:
    """One row per candidate and protocol. Done candidates of the ranking protocol (cv) first, best first, with a rank;
    the rest after, with their status and reason: screening and tuning results of other protocols (protocol column) are
    listed without a rank, as their scores are not comparable with cv scores. Ties keep grid order."""
    if metric not in METRICS[mode]:
        raise ValueError(f'Unknown metric {metric!r} for {mode}; choose one of {sorted(METRICS[mode])}.')
    direction = METRICS[mode][metric]
    rows = []
    for order, entry in enumerate(entries):
        mean, std = fold_stats(entry, metric) if entry['status'] == 'done' else (None, None)
        interval = entry.get('intervals', {}).get(metric, {}) if entry['status'] == 'done' else {}
        config = {name: entry['config'].get(name) for name in CANDIDATE_FIELDS}
        config['augmentation'] = _augmentation_column(entry['config'])
        rows.append({'key': entry['key'], 'candidate': entry.get('label', ''), **config, 'status': entry['status'],
                     f'cv_{metric}_mean': mean, f'cv_{metric}_fold_sd': std, 'folds': len(entry.get('folds', [])),
                     f'oof_{metric}_low': interval.get('low'), f'oof_{metric}_high': interval.get('high'),
                     'interval_unit': interval.get('unit'), 'trainable_params': entry.get('footprint', {}).get('trainable_params'),
                     'protocol': entry.get('protocol', 'cv'), 'stage': entry.get('stage', ''),
                     'seconds': entry.get('seconds'), 'reason': entry.get('reason', ''), '_order': order})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    ranked = table[(table['status'] == 'done') & table[f'cv_{metric}_mean'].notna() & (table['protocol'] == 'cv')].copy()
    ranked['_score'] = ranked[f'cv_{metric}_mean'] * direction
    ranked = ranked.sort_values(['_score', '_order'], ascending=[False, True], kind='stable').drop(columns='_score')
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    rest = table.drop(index=ranked.index).sort_values('_order', kind='stable')
    rest.insert(0, 'rank', None)
    return pd.concat([ranked, rest], ignore_index=True).drop(columns='_order')


def _oof_scores(path: Path, mode: str) -> pd.DataFrame:
    from evaluators.video.validation import load_oof_predictions
    return load_oof_predictions(path).sort_values('row_id', kind='stable').reset_index(drop=True)


def paired_difference(out_root: Path, entry_a: dict, entry_b: dict, mode: str, metric: str, n_boot: int, seed: int) -> dict:
    """Winner minus runner-up on the pooled out-of-fold rows, group-bootstrapped together."""
    from evaluators.video.report import compare_models
    a, b = (_oof_scores(Path(out_root) / entry['oof_path'], mode) for entry in (entry_a, entry_b))
    same = len(a) == len(b) and a['row_id'].equals(b['row_id']) and a['target'].equals(b['target']) and \
        a['video_id'].astype(str).equals(b['video_id'].astype(str))
    if not same:
        raise ValueError('The two candidates were not cross-validated on the same rows, so they cannot be compared pairwise.')
    if not a['fold'].astype(str).equals(b['fold'].astype(str)):
        raise ValueError('The two candidates were not cross-validated on the same folds, so they cannot be compared pairwise.')

    def scores(table):
        if mode == 'cls':
            columns = sorted((c for c in table.columns if c.startswith('prob_')), key=lambda c: int(c.split('_')[1]))
            return table[columns].to_numpy()
        return table['prediction'].to_numpy()
    groups = None if a['group'].isna().any() else a['group'].tolist()
    result = compare_models(mode, a['target'].to_numpy(), scores(a), scores(b), groups, None, n_boot, seed)
    chosen = result['metrics'][metric]
    better = METRICS[mode][metric]
    return {'metric': metric, 'difference': 'winner minus runner-up', 'rows': 'pooled out-of-fold development rows',
            'point': chosen['point'], 'low': chosen['low'], 'high': chosen['high'], 'level': chosen['level'], 'unit': chosen['unit'],
            'n_units': chosen['n_units'], 'n_boot': chosen['n_boot'], 'zero_excluded': chosen['zero_excluded'],
            'higher_is_better': better > 0, 'warning': chosen.get('warning'),
            'covers': 'which development units were drawn, for these two sets of fold models; not training variability, '
                      'and not corrected for choosing the top two of many'}


def read_final_result(run_dir: Path) -> dict:
    """The sealed-test evaluation of the final run: evaluation_report.json of its eval_output_* folder."""
    found = sorted(Path(run_dir).glob('*/eval_output_*/evaluation_report.json'), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f'No eval_output_*/evaluation_report.json under {run_dir}.')
    report = json.loads(found[-1].read_text(encoding='utf-8'))
    splits = sorted(Path(run_dir).glob('*/data_splits_*/split_summary.json'), key=lambda p: p.stat().st_mtime)
    validation_seed = json.loads(splits[-1].read_text(encoding='utf-8')).get('validation_seed') if splits else None
    return {'n': report.get('n'), 'metrics': report.get('metrics', {}), 'validation_seed': validation_seed,
            'intervals': {name: {key: value.get(key) for key in ('low', 'high', 'level', 'unit', 'n_units', 'warning')}
                          for name, value in report.get('intervals', {}).items()},
            'report': str(found[-1])}

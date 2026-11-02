"""The sealed test split of a model search: made once, before anything trains, and recorded.

The input rows are split into `dev.parquet` and `sealed_test.parquet` by the first fold of the same grouped, stratified
k-fold that the evaluation suite uses (`evaluators.video.validation.kfold_splits`), with 1/test_fraction parts. Whole
groups move together: the `group` column (or --grouper), else the source video when clips share one, else the row.

Every candidate run gets `dev.parquet` only. `sealed_test.parquet` is read once, by the winner's final run.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEV_FILE, SEALED_FILE, SPLIT_FILE, ASSIGNMENT_FILE = 'dev.parquet', 'sealed_test.parquet', 'split.json', 'split_assignment.csv'


def _units(table: pa.Table, grouper: Optional[str]) -> tuple[Optional[list], str]:
    """Group values per row and their source: the group column, else repeated source videos, else none (rows)."""
    from evaluators.video.run_validation import _holds_paths
    from handlers.split_dataset import resolve_grouper
    column = resolve_grouper(table.column_names, grouper, 'video')
    if column:
        if column not in table.column_names:
            raise ValueError(f'Grouper column {column!r} is not in the data.')
        values = table.column(column).to_pylist()
        if any(value is None for value in values):
            raise ValueError(f'Grouper column {column!r} has missing values.')
        return values, f'column {column!r}'
    if _holds_paths(table):
        videos = table.column('videos').to_pylist()
        if len(set(videos)) < len(videos):
            return videos, 'source video (no group column, and clips share source videos)'
    return None, 'row (no group column: clips are assumed independent)'


def make_sealed_split(input_path: Path, data_dir: Path, target: str = 'labels', grouper: Optional[str] = None, mode: str = 'cls',
                      test_fraction: float = 0.2, seed: int = 42) -> dict:
    """Write dev.parquet, sealed_test.parquet, split.json and split_assignment.csv to `data_dir`. Returns the record."""
    from evaluators.video.run_validation import _absolute_videos
    from evaluators.video.validation import kfold_splits
    from trainers.training_basics import data_fingerprint
    if not 0 < test_fraction <= 0.5:
        raise ValueError(f'--test_fraction must be in (0, 0.5], got {test_fraction}.')
    input_path, data_dir = Path(input_path), Path(data_dir)
    table = _absolute_videos(pq.read_table(input_path, memory_map=True), input_path)
    if target not in table.column_names:
        raise ValueError(f'Target column {target!r} is not in {input_path}.')
    labels = table.column(target).to_pylist()
    if any(value is None for value in labels):
        raise ValueError(f'Target column {target!r} has missing values.')
    labels = np.asarray([str(v) for v in labels]) if mode == 'cls' else np.asarray(labels, dtype=np.float64)
    groups, unit = _units(table, grouper)
    parts = max(2, int(round(1 / test_fraction)))
    try:
        dev_rows, test_rows = kfold_splits(labels, parts, groups, mode, seed)[0]
    except ValueError as exc:
        raise ValueError(f'Could not set a sealed test split of about {test_fraction:.0%} aside: {exc}') from None
    if mode == 'cls':
        missing = sorted(set(labels[test_rows]) - set(labels[dev_rows]))
        if missing:
            raise ValueError(f'The sealed test split holds classes that no development row has: {missing[:10]}. Add data.')
    if groups is not None:
        shared = set(np.asarray(groups, dtype=object)[dev_rows]) & set(np.asarray(groups, dtype=object)[test_rows])
        if shared:  # kfold_splits keeps groups whole; this guards the guarantee itself
            raise AssertionError(f'{len(shared)} groups are in both the development rows and the sealed test rows.')
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array(dev_rows, type=pa.int64())), data_dir / DEV_FILE)
    pq.write_table(table.take(pa.array(test_rows, type=pa.int64())), data_dir / SEALED_FILE)
    part = np.full(table.num_rows, 'dev', dtype=object)
    part[test_rows] = 'sealed_test'
    pd.DataFrame({'input_row': np.arange(table.num_rows), 'part': part}).to_csv(data_dir / ASSIGNMENT_FILE, index=False)

    def describe(rows):
        out = {'rows': int(len(rows))}
        if groups is not None:
            out['groups'] = int(len(set(np.asarray(groups, dtype=object)[rows])))
        if mode == 'cls':
            values, counts = np.unique(labels[rows], return_counts=True)
            out['class_counts'] = {str(v): int(c) for v, c in zip(values, counts)}
        return out
    record = {'method': f'first fold of a {"stratified " if mode == "cls" else ""}grouped {parts}-fold split '
                        '(evaluators.video.validation.kfold_splits)',
              'seed': int(seed), 'test_fraction_requested': float(test_fraction), 'unit': unit,
              'input': str(input_path.resolve()), 'input_fingerprint': data_fingerprint(input_path),
              'dev': {**describe(dev_rows), 'file': DEV_FILE, 'fingerprint': data_fingerprint(data_dir / DEV_FILE)},
              'sealed_test': {**describe(test_rows), 'file': SEALED_FILE, 'fingerprint': data_fingerprint(data_dir / SEALED_FILE)},
              'assignment': ASSIGNMENT_FILE,
              'rule': 'Candidates are trained and scored on dev rows only. The sealed test rows are scored once, by the winner.'}
    (data_dir / SPLIT_FILE).write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def check_split_files(data_dir: Path, record: dict) -> Optional[str]:
    """Why the split files on disk no longer match their record, or None."""
    from trainers.training_basics import data_fingerprint
    for part in ('dev', 'sealed_test'):
        path = Path(data_dir) / record[part]['file']
        if data_fingerprint(path) != record[part]['fingerprint']:
            return f'{path} is missing or changed since the split was made.'
    return None

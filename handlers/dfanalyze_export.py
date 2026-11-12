"""Assembles the df-analyze export (Phase 5, task 5). Torch-free; embed/video/dfanalyze.py computes the blocks.

The rules that keep this free of leakage:
- The only fitted step is PCA, fitted on the rows of `fit_split` and applied unchanged to all the others.
- Group ids, split names and video ids are not features. All are in `dfanalyze_ids.csv` (and `dfanalyze.parquet`, whose use is
  to find the rows of a df-analyze holdout again). When groups exist the CSV has a `group` column, and the command passes it to
  df-analyze's --grouper, which keeps groups whole and removes the column from X (upstream
  src/df_analyze/preprocessing/prepare.py). `split` and `video_id` join the CSV only with `ids`, and the command drops them.
- A group or video in two splits is refused.
- `dfanalyze.csv` defaults to rows a trained model did not see, since df-analyze makes its own 60/40 split of whatever it is
  given. A holdout holding rows the model trained on would read too high. PCA's training rows are not in it then. A pretrained
  model saw no rows, so the caller defaults to all of them, and the export warns if `--dfanalyze_dims` then puts PCA's rows in.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

TARGET_COLUMN = 'target'
ID_COLUMNS = ('split', 'video_id')
ROWS_PER_FEATURE_WANTED = 10
EXPLORATORY_NOTE = "These rows already fed DeepTune's report, so this is exploratory, not a second sealed test."
UNKNOWN_ROWS_NOTE = ('Unknown whether these rows fed a DeepTune report: this export was not given that record. '
                     'Check that before treating df-analyze results on them as a sealed test.')
MOTION_STATISTICS = ('motion_mean', 'motion_std', 'motion_max', 'motion_p90')
MANIFEST_FILENAME = 'dfanalyze_export.json'


@dataclass
class SplitBlocks:
    """One split's rows. `embedding`: [N, D], the mean over clips. `embedding_std`: [N, D] over the same clips, or None."""
    target: np.ndarray
    video_ids: Sequence[str]
    embedding: np.ndarray
    embedding_std: Optional[np.ndarray] = None
    motion: Optional[pd.DataFrame] = None
    groups: Optional[Sequence] = None


def curve_statistics(curve) -> dict:
    """Summary of a per-frame motion curve for one range. A curve's first value compares with a frame outside the range
    (or is 0 at the start of the file), so it is left out. A single frame has no motion: all zeros."""
    values = np.asarray(curve, dtype=np.float64)[1:]
    if not len(values):
        return dict.fromkeys(MOTION_STATISTICS, 0.0)
    return {'motion_mean': float(values.mean()), 'motion_std': float(values.std()), 'motion_max': float(values.max()),
            'motion_p90': float(np.percentile(values, 90))}


def audit_splits(parts: dict[str, SplitBlocks]) -> None:
    """Refuses a video or a group that is in two splits."""
    names = list(parts)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = set(parts[first].video_ids) & set(parts[second].video_ids)
            if shared:
                raise ValueError(f'Split leakage: videos {sorted(map(str, shared))[:5]} are in both {first} and {second}.')
            if parts[first].groups is not None and parts[second].groups is not None:
                shared = set(parts[first].groups) & set(parts[second].groups)
                if shared:
                    raise ValueError(f'Split leakage: groups {sorted(map(str, shared))[:5]} are in both {first} and {second}.')


class BlockPCA:
    """PCA fitted on one matrix: centring and projection only, stored so that it can be checked and reapplied."""

    def __init__(self, fit_matrix: np.ndarray, dims: int):
        from sklearn.decomposition import PCA
        fit_matrix = np.asarray(fit_matrix, dtype=np.float64)
        if len(fit_matrix) < 2:
            raise ValueError(f'PCA needs at least 2 training rows; found {len(fit_matrix)}.')
        self.requested = int(dims)
        self.used = min(self.requested, fit_matrix.shape[0], fit_matrix.shape[1])
        model = PCA(n_components=self.used, svd_solver='full').fit(fit_matrix)
        self.mean, self.components = model.mean_, model.components_
        self.explained = np.nan_to_num(model.explained_variance_ratio_)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (np.asarray(matrix, dtype=np.float64) - self.mean) @ self.components.T


def _named(matrix: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(matrix, dtype=np.float32), columns=[f'{prefix}_{i}' for i in range(matrix.shape[1])])


def _check_finite(parts: dict[str, SplitBlocks]) -> None:
    for name, part in parts.items():
        for block in (part.embedding, part.embedding_std):
            if block is not None and not np.isfinite(block).all():
                raise ValueError(f'The {name} split has embeddings that are not finite; df-analyze would quietly impute them.')


def command_for(csv_path, mode: str, with_ids: bool, has_groups: bool, outdir: str = './dfanalyze_results') -> str:
    """The df-analyze command, from its docs/arguments.md. df-analyze is not a dependency, and this was not run here."""
    if mode == 'cls':
        task = '--mode=classify --classifiers lgbm rf lr sgd knn mlp dummy'
    else:
        task = '--mode=regress --regressors knn lgbm elastic sgd mlp dummy'
    command = f'python df-analyze.py --df "{csv_path}" --target {TARGET_COLUMN} {task} --embed-select none linear lgbm --outdir {outdir}'
    if has_groups:
        command += ' --grouper group'
    if with_ids:
        command += ' --drops ' + ','.join(ID_COLUMNS)
    return command


def write_dfanalyze_export(parts: dict[str, SplitBlocks], out_dir, mode: str, dims: Optional[int] = None, extras: Sequence[str] = (),
                           rows: Sequence[str] = ('test',), fit_split: str = 'train', model_saw_fit_split: bool = True,
                           provenance: Optional[dict] = None) -> dict:
    """Writes the files; returns what `dfanalyze_export.json` holds. `rows`: the splits of `dfanalyze.csv`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extras = tuple(extras)
    missing = [name for name in rows if name not in parts]
    if missing:
        raise ValueError(f'dfanalyze.csv should hold {missing}, but no such split was embedded: {list(parts)}.')
    _check_finite(parts)
    audit_splits(parts)
    has_std = any(part.embedding_std is not None for part in parts.values())
    pca = pca_std = None
    if dims is not None:
        if fit_split not in parts:
            raise ValueError(f'--dfanalyze_dims fits PCA on the training rows, and the {fit_split!r} split was not given.')
        pca = BlockPCA(parts[fit_split].embedding, dims)
        if has_std:
            pca_std = BlockPCA(parts[fit_split].embedding_std, dims)

    has_groups = any(part.groups is not None for part in parts.values())
    tables, ids = {}, []
    for name, part in parts.items():
        blocks = [_named(pca.transform(part.embedding), 'pca') if pca else _named(part.embedding, 'emb')]
        if part.embedding_std is not None:
            blocks.append(_named(pca_std.transform(part.embedding_std), 'pca_std') if pca_std else _named(part.embedding_std, 'emb_std'))
        if part.motion is not None:
            blocks.append(part.motion.reset_index(drop=True).astype(np.float32))
        features = pd.concat(blocks, axis=1)
        features[TARGET_COLUMN] = np.asarray(part.target)
        identity = pd.DataFrame({'split': name, 'video_id': list(part.video_ids),
                                 'group': list(part.groups) if part.groups is not None else None})
        tables[name] = (features, identity)
        ids.append(identity.assign(row_id=np.arange(len(identity)))[['row_id', 'split', 'video_id', 'group']])
    pd.concat(ids, ignore_index=True).to_csv(out_dir / 'dfanalyze_ids.csv', index=False)

    with_ids = 'ids' in extras
    feature_columns = [c for c in next(iter(tables.values()))[0].columns if c != TARGET_COLUMN]
    for name, (features, identity) in tables.items():
        _csv_frame(features, identity, with_ids, has_groups).to_csv(out_dir / f'dfanalyze_{name}.csv', index=False)
    main_csv = pd.concat([_csv_frame(*tables[name], with_ids, has_groups) for name in rows], ignore_index=True)
    main_csv.to_csv(out_dir / 'dfanalyze.csv', index=False)
    pd.concat([pd.concat(list(tables[name]), axis=1) for name in rows], ignore_index=True).to_parquet(out_dir / 'dfanalyze.parquet', index=False)

    warnings = []
    if has_groups and main_csv['group'].isna().any():
        warnings.append(f"{int(main_csv['group'].isna().sum())} of {len(main_csv)} rows have no group, but the command passes --grouper group. "
                        'Rows with no group have no unit to keep together, and df-analyze may treat them unpredictably. Fill in the group column.')
    if len(main_csv) < ROWS_PER_FEATURE_WANTED * len(feature_columns):
        warnings.append(f'dfanalyze.csv has {len(main_csv)} rows for {len(feature_columns)} features, under {ROWS_PER_FEATURE_WANTED} rows per feature, '
                        'and df-analyze holds out 40% of them. Expect unstable selection and scores. Consider a smaller --dfanalyze_dims or more data.')
    if dims is not None and fit_split in rows:
        warnings.append(f'dfanalyze.csv includes the {fit_split} rows PCA was fitted on, so some of df-analyze\'s holdout took part in that fit. '
                        'PCA does not see the labels, but read the holdout scores with this in mind.')
    if model_saw_fit_split and fit_split in rows:
        warnings.append(f'dfanalyze.csv includes the {fit_split} rows the model trained on. Embeddings of those rows fit better than '
                        "new rows do, so df-analyze's holdout scores read too high. Use --dfanalyze_rows test for an honest estimate.")
    if dims is not None and pca.used < pca.requested:
        warnings.append(f'--dfanalyze_dims {pca.requested} is more than the {fit_split} rows allow: kept {pca.used} components.')
    if dims is None and len(feature_columns) > 100:
        warnings.append(f'{len(feature_columns)} features: df-analyze advises under about 100. Consider --dfanalyze_dims.')

    manifest = {'command': command_for((out_dir / 'dfanalyze.csv').resolve(), mode, with_ids, has_groups), 'command_was_run_here': False,
                'target': TARGET_COLUMN, 'mode': mode, 'rows_in_dfanalyze_csv': list(rows), 'extras': list(extras),
                'n_rows': {name: int(len(part.target)) for name, part in parts.items()}, 'n_features': len(feature_columns),
                'feature_columns_first_last': [feature_columns[0], feature_columns[-1]],
                'ids_in_csv': [*(ID_COLUMNS if with_ids else ()), *(('group',) if has_groups else ())], 'ids_file': 'dfanalyze_ids.csv',
                'status_of_these_rows': EXPLORATORY_NOTE if 'test' in rows else UNKNOWN_ROWS_NOTE,
                'pca': None, 'warnings': warnings, **(provenance or {})}
    if pca is not None:
        arrays = {'mean': pca.mean, 'components': pca.components, 'explained_variance_ratio': pca.explained}
        if pca_std is not None:
            arrays.update(std_mean=pca_std.mean, std_components=pca_std.components, std_explained_variance_ratio=pca_std.explained)
        np.savez(out_dir / 'dfanalyze_pca.npz', **arrays)
        fitted_on = sorted(map(str, parts[fit_split].video_ids))
        manifest['pca'] = {'fit_split': fit_split, 'n_fit_rows': len(fitted_on), 'fit_video_ids_sha256': hashlib.sha256('\n'.join(fitted_on).encode()).hexdigest(),
                           'requested': pca.requested, 'used': pca.used, 'explained_variance': float(pca.explained.sum()), 'file': 'dfanalyze_pca.npz'}
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for warning in warnings:
        print(f'df-analyze export: {warning}')
    print(f"df-analyze export is in {out_dir}. To analyse it:\n  {manifest['command']}")
    return manifest


def _csv_frame(features: pd.DataFrame, identity: pd.DataFrame, with_ids: bool, has_groups: bool) -> pd.DataFrame:
    columns = [*(ID_COLUMNS if with_ids else ()), *(('group',) if has_groups else ())]
    return pd.concat([features, identity[columns]], axis=1) if columns else features

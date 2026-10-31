"""Runs `--validation kfold|bootstrap632` through the real trainer, and writes the final model's report (Phase 5).

Only the train and validation splits are read here. The test split never reaches `run_video_validation`, so no fold can
see it; `tests/test_evaluation_pipeline_e2e.py` checks that on the rows each fold was given. `deeptune.py` scores the test
split once, after this, with the usual final model.

Each fold trains a fresh model with the run's own settings. With a fixed seed the folds start from the same initial weights,
so their spread comes from the data alone and leaves out initialisation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evaluators.stats import DEFAULT_BOOTSTRAPS
from evaluators.video.report import load_eval_scores, report_eval_dir, selection_used_validation
from evaluators.video.validation import FoldScores, run_validation

FIT_SPLIT_FOLDER = 'calibration_fit'


def _holds_paths(table: pa.Table) -> bool:
    if 'videos' not in table.column_names:
        return False
    kind = table.schema.field('videos').type
    return pa.types.is_string(kind) or pa.types.is_large_string(kind)


def _absolute_videos(table: pa.Table, split_path: Path) -> pa.Table:
    """Relative paths mean relative to their split file, and fold files live elsewhere."""
    if not _holds_paths(table):
        return table
    index = table.schema.get_field_index('videos')
    root = split_path.resolve().parent
    values = [value if value is None or Path(value).is_absolute() else str(root / value) for value in table.column(index).to_pylist()]
    return table.set_column(index, 'videos', pa.array(values, type=table.schema.field(index).type))


def read_pool(train_path, val_path) -> pa.Table:
    """The rows this scheme may use, train first. Memory-mapped, so clip bytes are not all read into memory."""
    tables = [_absolute_videos(pq.read_table(path, memory_map=True), Path(path)) for path in (train_path, val_path)]
    return pa.concat_tables(tables, promote_options='default')


def _column(table: pa.Table, name):
    return table.column(name).to_pylist() if name and name in table.column_names else None


def _video_ids(table: pa.Table) -> list[str]:
    ids = _column(table, 'video_id')
    if ids is not None:
        return [str(value) for value in ids]
    paths = _column(table, 'videos') if _holds_paths(table) else None
    return [Path(path).name for path in paths] if paths else [str(i) for i in range(table.num_rows)]


def run_video_validation(args, train_path, val_path, out, train_fn, evaluate_fn, train_config: dict, eval_config: dict, grouper=None) -> dict:
    """k-fold or .632+ on the train + validation rows. `train_fn` and `evaluate_fn` are trainers/video/train.py's `train` and
    evaluators/video/evaluate.py's `evaluate`; `*_config` are their keyword arguments apart from paths, out and args."""
    pool = read_pool(train_path, val_path)
    scheme = args.validation
    root = Path(out) / f'validation_{scheme}'
    mode = train_config['mode']
    count = args.k if scheme == 'kfold' else args.n_boot + 1
    print(f'--validation {scheme}: training {count} extra models on {pool.num_rows} rows. The test split is not used.')

    def fit_predict(train_rows, val_rows, eval_rows, name, error_model=True) -> FoldScores:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        paths = {role: folder / f'{role}.parquet' for role in ('train', 'val', 'eval')}
        for role, rows in (('train', train_rows), ('val', val_rows), ('eval', eval_rows)):
            pq.write_table(pool.take(pa.array(np.asarray(rows, dtype=np.int64))), paths[role])
        fold_args = copy.copy(args)  # a parameter budget writes its choice into args: one fold must not pass it to the next
        fold_args.no_error_model = args.no_error_model or not error_model
        weights = train_fn(train_df=paths['train'], val_df=paths['val'], out=folder, args=fold_args, **train_config)
        config = dict(eval_config)
        if train_config.get('trainable_params') is not None and fold_args.embed_size is not None:
            config['embed_size'] = fold_args.embed_size
        evaluate_fn(eval_df=paths['eval'], out=folder, model_weights=weights, args=fold_args, **config)
        eval_dir = next(folder.glob('eval_output_*'))
        _, _, scores, table = load_eval_scores(eval_dir, mode)
        expected = table['expected_error'].to_numpy() if 'expected_error' in table.columns else None
        if not args.keep_fold_models:
            for path in [*paths.values(), Path(weights), *Path(weights).parent.glob('*.pth'), *Path(weights).parent.glob('*.pt')]:
                path.unlink(missing_ok=True)
        return FoldScores(scores, expected_error=expected)

    def apparent(train_rows, val_rows, eval_rows, name) -> FoldScores:
        return fit_predict(train_rows, val_rows, eval_rows, name, error_model=False)  # its validation rows are its training rows

    videos = _column(pool, 'videos') if _holds_paths(pool) else None
    groups, group_source = _column(pool, grouper), 'column'
    if groups is None and videos is not None and len(set(videos)) < len(videos):
        # Clips of one source would straddle folds, while the intervals already treat the source as the unit.
        groups, group_source = videos, 'source_video'
        print(f'--validation {scheme}: no group column, but {pool.num_rows} rows come from {len(set(videos))} source videos. Using the source video as the group.')
    summary = run_validation(scheme, np.asarray(_column(pool, 'labels')), fit_predict, root, mode=mode, ids=_video_ids(pool),
                             groups=groups, group_source=group_source, sources=videos, k=args.k, n_boot=args.n_boot,
                             seed=_validation_seed(args), ci_boot=args.report_boot or DEFAULT_BOOTSTRAPS,
                             apparent_fit_predict=apparent if scheme == 'bootstrap632' else None)
    args.save_args(root)
    print(f'Validation results are in {root}')
    return summary


def _validation_seed(args) -> int:
    seed = getattr(args, 'seed', None)
    return 42 if seed is None else int(seed)


def write_final_report(args, run_dir, weights, test_path, val_path, train_path, mapping_path, evaluate_fn, eval_config: dict, grouper=None,
                       extra_warnings=()) -> Path:
    """evaluation_report.json for the test evaluation in `run_dir`. For classification the validation rows are scored too, with
    the same test-time settings, and only they fit the temperature. No temperature is ever fitted on test rows."""
    run_dir = Path(run_dir)
    eval_dir = next(path for path in sorted(run_dir.glob('eval_output_*')) if path.is_dir())
    fit_dir = fit_split = None
    if eval_config['mode'] == 'cls':
        folder = run_dir / FIT_SPLIT_FOLDER
        fit_args = copy.copy(args)
        evaluate_fn(eval_df=val_path, out=folder, model_weights=weights, args=fit_args, **eval_config)
        fit_dir, fit_split = next(folder.glob('eval_output_*')), 'validation'
    path = report_eval_dir(eval_dir, test_path, grouper, fit_dir, fit_split, mapping_path, train_path, n_boot=args.report_boot,
                           selection_used_fit_split=selection_used_validation(Path(weights).parent), extra_warnings=extra_warnings)
    print(f'The evaluation report is in {path}')
    return path

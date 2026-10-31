"""Ensembles: average class probabilities across trained runs.

Two ways in, one rule. ``ensemble_runs`` runs each trained model on new data, each with the settings its run recorded,
so members may differ in backbone, frames, clips and preprocessing. ``ensemble_tables`` combines finished
``predictions.csv`` files, for runs that have no pixel path, such as ``train_feature_head.py`` probes.

Probabilities are averaged, not logits: a member's confidence counts for at most 1. The members' ``expected_error``
columns are not carried over, because each error model was fitted on one model's probabilities and says nothing about
the average. ``member_disagreement`` is the same measure as ``clip_disagreement``: the mean total-variation distance
between each member's probabilities and the average, 0 when they all agree. For regression it is the members' standard
deviation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

_PROBABILITY = re.compile(r'^probability_(\d+)$')
ENSEMBLE_RECORD = 'ensemble.json'


def probability_columns(table: pd.DataFrame) -> list[str]:
    """``probability_<k>`` columns in class order, whatever order the file has them in."""
    found = sorted((int(m.group(1)), m.group(0)) for m in map(_PROBABILITY.match, map(str, table.columns)) if m)
    if [k for k, _ in found] != list(range(len(found))):
        raise ValueError(f'Probability columns must run from probability_0 without gaps, found {[n for _, n in found]}.')
    return [name for _, name in found]


def _key_column(tables: Sequence[pd.DataFrame]) -> str:
    for key in ('video_id', 'sample_id'):
        if all(key in table.columns for table in tables):
            return key
    raise ValueError('Every member needs a video_id or sample_id column to line its rows up with the others.')


def _aligned(tables: Sequence[pd.DataFrame], key: str) -> list[pd.DataFrame]:
    """The members in the first member's row order. Refuses anything but the same rows in all of them."""
    first = tables[0][key].astype(str)
    aligned = []
    for index, table in enumerate(tables):
        ids = table[key].astype(str)
        if ids.duplicated().any():
            raise ValueError(f'Member {index} repeats {key} {ids[ids.duplicated()].iloc[0]!r}.')
        if set(ids) != set(first):
            different = sorted(set(ids) ^ set(first))
            raise ValueError(f'Member {index} predicted other rows than member 0: {len(different)} differ, for example {different[:3]}.')
        aligned.append(table.assign(**{key: ids}).set_index(key).loc[first.tolist()].reset_index())
    return aligned


def _truth_column(table: pd.DataFrame) -> Optional[str]:
    return next((name for name in ('true_label', 'true_value', 'target') if name in table.columns), None)


def _check_same_truth(tables: Sequence[pd.DataFrame], column: str) -> None:
    for index, table in enumerate(tables[1:], start=1):
        if column not in table.columns or table[column].astype(str).tolist() != tables[0][column].astype(str).tolist():
            raise ValueError(f'Member {index} has other {column} values than member 0, so they were not scored on the same labels.')


def combine_tables(tables: Sequence[pd.DataFrame], mode: str = 'cls', label_mapping: Optional[dict] = None) -> pd.DataFrame:
    """The ensemble's predictions, from member tables with ``probability_<k>`` columns (cls) or values (reg)."""
    if len(tables) < 2:
        raise ValueError('An ensemble needs at least two members.')
    key = _key_column(tables)
    tables = _aligned(tables, key)
    truth = _truth_column(tables[0])
    if truth is not None:
        _check_same_truth(tables, truth)
    result = pd.DataFrame({key: tables[0][key]})
    if mode == 'cls':
        columns = [probability_columns(table) for table in tables]
        if not columns[0] or any(c != columns[0] for c in columns):
            raise ValueError(f'Members must have the same classes; their probability columns differ or are missing: {[len(c) for c in columns]}.')
        stack = np.stack([table[columns[0]].to_numpy(dtype=np.float64) for table in tables])  # [members, rows, classes]
        average = stack.mean(axis=0)
        predicted = average.argmax(axis=1)
        result['predicted_class'] = predicted
        names = {int(index): name for name, index in (label_mapping or {}).items()}
        result['predicted_label'] = [names.get(int(p), str(int(p))) for p in predicted]
        result['confidence'] = average.max(axis=1)
        for k, name in enumerate(columns[0]):
            result[name] = average[:, k]
        result['member_disagreement'] = 0.5 * np.abs(stack - average).sum(axis=2).mean(axis=0)
        if truth == 'target':
            result['target'] = tables[0][truth].to_numpy()
            result['correct'] = result['target'].astype(float).to_numpy() == predicted
        elif truth is not None:
            result['true_label'] = tables[0][truth].to_numpy()
            result['correct'] = [str(t) == str(p) for t, p in zip(result['true_label'], result['predicted_label'])]
        return result
    values = [next((table[c] for c in ('predicted_value', 'prediction') if c in table.columns), None) for table in tables]
    if any(v is None for v in values):
        raise ValueError('Every regression member needs a predicted_value or prediction column.')
    stack = np.stack([v.to_numpy(dtype=np.float64) for v in values])
    result['predicted_value'] = stack.mean(axis=0)
    result['member_disagreement'] = stack.std(axis=0)
    if truth is not None:
        result['true_value'] = tables[0][truth].to_numpy()
        result['abs_error'] = np.abs(result['true_value'].astype(float) - result['predicted_value'])
    return result


def _write(result: pd.DataFrame, out: Path, record: dict, mode: str) -> pd.DataFrame:
    from inference.common import summarize_predictions
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / 'predictions.csv', index=False)
    metrics = summarize_predictions(result, mode)
    if metrics is not None:
        (out / 'inference_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f'Ensemble of {len(record["members"])}: {metrics}')
    (out / ENSEMBLE_RECORD).write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Wrote {len(result)} prediction(s) to {out / "predictions.csv"}, members and weight licences to {out / ENSEMBLE_RECORD}')
    return result


def resolve_checkpoint(run: Path) -> Path:
    """A weights file, or a folder holding exactly one ``model_weights.pth``."""
    run = Path(run)
    if run.is_file():
        return run
    found = sorted(run.rglob('model_weights.pth')) if run.is_dir() else []
    if len(found) != 1:
        raise ValueError(f'{run}: expected a weights file or a folder with one model_weights.pth, found {len(found)}.')
    return found[0]


def find_label_mapping(checkpoint: Path) -> Optional[dict]:
    """The run's ``data_splits_*/label_mapping.json``, if it can be found near its checkpoint."""
    for folder in list(Path(checkpoint).parents)[:3]:
        for path in sorted(folder.glob('data_splits_*/label_mapping.json')):
            return json.loads(path.read_text(encoding='utf-8-sig'))
    return None


def _predict_member(checkpoint: Path, recorded: dict, df: pd.DataFrame, batch_size: int, label_mapping: Optional[dict]) -> pd.DataFrame:
    """One member through the same code as a single-model infer.py, rebuilt from what its run recorded."""
    from datasets.video_datasets import video_options_from_args
    from handlers.video_backbones import accept_weights_licences
    from infer import apply_recorded_video_settings, build_parser
    from inference.video import predict_video
    argv = ['--modality', 'video', '--checkpoint', str(checkpoint), '--data', '.', '--out', '.']
    args = build_parser().parse_args(argv)
    apply_recorded_video_settings(args, argv)
    accept_weights_licences(recorded.get('accept_weights_licence'))
    extra = {} if args.lora_rank is None else {'lora_rank': args.lora_rank}
    return predict_video(df, checkpoint, recorded['model_version'], int(recorded.get('num_classes') or 2),
                         int(recorded.get('added_layers') or 2), int(recorded.get('embed_size') or 1000),
                         bool(recorded.get('freeze_backbone', False)), recorded.get('mode') or 'cls', bool(recorded.get('use_peft', False)),
                         batch_size, args.num_frames, args.pooling, label_mapping, video_options=video_options_from_args(args),
                         with_probabilities=True, **extra)


def ensemble_runs(runs: Sequence[Path], data: Path, out: Path, batch_size: int = 16, label_mapping_path: Optional[Path] = None,
                  accept_licences: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Average trained runs' predictions on new data. A run with no recorded settings, or members that disagree on the
    task, the number of classes or the meaning of a class index, are refused before any model loads."""
    from argparse import Namespace

    from handlers.video_backbones import accept_weights_licences
    from infer import _load_new_data, _recorded_run_arguments
    accept_weights_licences(accept_licences)
    checkpoints = [resolve_checkpoint(run) for run in runs]
    recorded = [_recorded_run_arguments(c) for c in checkpoints]
    if len(checkpoints) < 2:
        raise ValueError('An ensemble needs at least two runs.')
    for checkpoint, values in zip(checkpoints, recorded):
        if not values.get('model_version'):
            raise ValueError(f'{checkpoint}: no cli_arguments.json beside it, so its model cannot be rebuilt.')
        if values.get('modality', 'video') != 'video':
            raise ValueError(f'{checkpoint} is a {values["modality"]} run; ensembles are for video runs.')
    modes = {values.get('mode') or 'cls' for values in recorded}
    classes = {int(values.get('num_classes') or 2) for values in recorded}
    if len(modes) > 1 or (modes == {'cls'} and len(classes) > 1):
        raise ValueError(f'Members must solve the same task; found modes {sorted(modes)} and class counts {sorted(classes)}.')
    mode = modes.pop()
    mappings = [find_label_mapping(c) for c in checkpoints]
    known = [m for m in mappings if m is not None]
    if any(m != known[0] for m in known):
        raise ValueError('Members map class names to indices differently, so averaging would mix classes: '
                         f'{[m for m in known]}. Retrain them on one split.')
    if label_mapping_path is not None:
        given = json.loads(Path(label_mapping_path).read_text(encoding='utf-8-sig'))
        if known and given != known[0]:
            raise ValueError(f'--label_mapping differs from the mapping the runs recorded: {known[0]}.')
        mapping = given
    else:
        mapping = known[0] if known else None
    df, _ = _load_new_data(Namespace(modality='video', data=Path(data)))
    tables = []
    for checkpoint, values in zip(checkpoints, recorded):
        print(f'Ensemble member {len(tables) + 1} of {len(checkpoints)}: {values["model_version"]}, {checkpoint}')
        tables.append(_predict_member(checkpoint, values, df, batch_size, mapping))
        _release_gpu_memory()
    result = combine_tables(tables, mode, mapping)
    record = {'method': 'mean of class probabilities' if mode == 'cls' else 'mean of predicted values', 'mode': mode, 'data': str(data),
              'members': [{'checkpoint': str(c), 'model_version': v['model_version'], 'pooling': v.get('pooling'),
                           'num_frames': v.get('num_frames'), 'test_clips': v.get('test_clips'), 'test_crops': v.get('test_crops'),
                           'weights_licence': _member_licence(v)} for c, v in zip(checkpoints, recorded)],
              'expected_error': "not given: each member's error model describes that member alone"}
    return _write(result, out, record, mode)


def _member_licence(recorded: dict):
    if recorded.get('weights_licence'):
        return recorded['weights_licence']
    from src.video.feature_backbones import run_weight_licences  # run from before licences were recorded
    return run_weight_licences(recorded['model_version'])


def _release_gpu_memory() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensemble_tables(paths: Sequence[Path], out: Path, mode: str = 'cls', label_mapping_path: Optional[Path] = None) -> pd.DataFrame:
    """Average finished ``predictions.csv`` files. Each must have per-class probabilities and the same rows."""
    tables = [pd.read_csv(path) for path in paths]
    mapping = json.loads(Path(label_mapping_path).read_text(encoding='utf-8-sig')) if label_mapping_path else None
    result = combine_tables(tables, mode, mapping)
    members = []
    for path in paths:
        arguments = Path(path).parent / 'cli_arguments.json'
        recorded = json.loads(arguments.read_text(encoding='utf-8-sig')) if arguments.is_file() else {}
        members.append({'predictions': str(path), 'weights_licence': recorded.get('weights_licence', 'not recorded next to this file')})
    record = {'method': 'mean of class probabilities' if mode == 'cls' else 'mean of predicted values', 'mode': mode, 'members': members,
              'expected_error': "not given: each member's error model describes that member alone"}
    return _write(result, out, record, mode)

"""Fits abstention.json for a trained video model on a separate calibration split, and reports the rule on held-out predictions.

deeptune.py calls `carve_calibration_split`, `fit_on_calibration_split` and `report_on_eval_dir`. To fit a finished run later:
    python -m evaluators.video.abstention_fit --checkpoint <run>/model_weights.pth --calib_df <groups the run never saw> --target_error 0.05
The run's own data_splits_* folder is found from the checkpoint and checked for overlap; --other_df adds more files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluators.abstention import (ABSTENTION_FILENAME, DEFAULT_DELTA, GUARANTEES, SCORES, AbstentionRule, build_report, carve_calibration_rows, fit_abstention,
                                   budget_message, file_fingerprint, mapping_fingerprint, pipeline_record, score_values, write_fit_table, write_report)
from evaluators.abstention import UNIT_NOTES, find_rule
from evaluators.stats import unit_labels

CALIBRATION_FILENAME = 'calibration_split.parquet'
CALIBRATION_SUMMARY_FILENAME = 'calibration_split_summary.json'


def carve_calibration_split(train_path, calib_fraction: float, train_fraction: float, grouper: Optional[str], seed: Optional[int], target_error: Optional[float] = None,
                            delta: float = DEFAULT_DELTA, guarantee: str = 'selective_risk') -> Path:
    """Moves whole units from the training split into calibration_split.parquet beside it. The validation and test files stay as they were; split_summary.json is updated."""
    from evaluators.video.report import split_units
    train_path = Path(train_path)
    if calib_fraction >= train_fraction:
        raise ValueError(f'--calib_fraction {calib_fraction} must be below the training share {train_fraction}: it is taken from the training split.')
    frame = pd.read_parquet(train_path)
    groups, sources = split_units(train_path, grouper)
    units, unit = unit_labels(len(frame), groups, sources)
    n_units = len(np.unique(units))
    wanted = max(1, int(round(n_units * calib_fraction / train_fraction)))
    train, calibration = carve_calibration_rows(frame, min(wanted, n_units - 1), units, seed)
    path = train_path.parent / CALIBRATION_FILENAME
    calibration.to_parquet(path, index=False)
    train.to_parquet(train_path, index=False)
    summary = {'calib_fraction': calib_fraction, 'taken_from': 'train', 'unit': unit, 'seed': seed, 'train_rows': len(train), 'calibration_rows': len(calibration),
               'calibration_units': min(wanted, n_units - 1), 'units_before': n_units, 'method': 'whole units drawn uniformly at random, not stratified'}
    (train_path.parent / CALIBRATION_SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _update_split_summary(train_path.parent, train, calibration, grouper)
    print(f'Calibration split: {len(calibration)} clips from {summary["calibration_units"]} {unit} units, taken from the training split. Training keeps {len(train)} clips.')
    if unit != 'group':
        print(f'Warning: {UNIT_NOTES[unit]}')
    if target_error is not None:
        print(budget_message(target_error, summary['calibration_units'], delta, guarantee, unit))
    return path


def _update_split_summary(split_dir: Path, train: pd.DataFrame, calibration: pd.DataFrame, grouper: Optional[str]) -> None:
    """split_summary.json describes the split before calibration was taken out of train, so bring its counts up to date."""
    path = split_dir / 'split_summary.json'
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    counts = summary.setdefault('sample_counts', {})
    counts.update(train=len(train), calibration=len(calibration))
    column = grouper or summary.get('grouper')  # the split may have found the group column itself
    if column and 'group_counts' in summary and column in train.columns:
        summary['group_counts'].update(train=int(train[column].nunique()), calibration=int(calibration[column].nunique()))
    summary['calibration_taken_from'] = 'train'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')


def _resolved_sources(path, values) -> Optional[set]:
    if values is None:
        return None
    base = Path(path).parent
    return {str((Path(value) if Path(value).is_absolute() else base / value).resolve()) for value in values}


def text_units(path, grouper) -> tuple[Optional[list], Optional[list]]:
    """split_units with group ids as text. A CSV read by pandas turns '001' into 1, which then never matches '001' in the run's parquet, so overlap would
    go unseen. Here a CSV's group column is read as str, and every group id is compared as str."""
    from evaluators.video.report import split_units
    groups, sources = split_units(path, grouper)
    if groups is not None and Path(path).suffix.lower() != '.parquet':
        column = grouper or 'group'
        groups = pd.read_csv(path, usecols=[column], dtype={column: str}, keep_default_na=False)[column].tolist()
    return (None if groups is None else [str(group) for group in groups]), sources


def _void_reasons(calib_path, grouper, other_paths, disjoint_by_construction: bool = False) -> list[str]:
    """Why these rows cannot be shown to make a clean calibration set. The fit still writes its table, and says it carries no guarantee.
    With `disjoint_by_construction` (deeptune.py drew them itself) rows that hold nothing to compare are fine; otherwise that leaves disjointness unverified."""
    split_units = text_units
    other_paths = list(other_paths)
    if not other_paths and not disjoint_by_construction:
        return ["Disjointness unverified: no data_splits folder was found beside the run and no --other_df was given, so nothing shows the model never saw these rows."]
    groups, sources = split_units(calib_path, grouper)
    calib_groups, calib_sources = set(groups) if groups is not None else None, _resolved_sources(calib_path, sources)
    shared_groups, shared_sources, unverified = set(), set(), []
    for path in other_paths:
        other_groups, other_sources = split_units(path, grouper)
        compared = False
        if calib_groups is not None and other_groups is not None:
            shared_groups |= calib_groups & set(other_groups)
            compared = True
        if calib_sources is not None and other_sources is not None:
            shared_sources |= calib_sources & _resolved_sources(path, other_sources)
            compared = True
        if not compared:
            unverified.append(Path(path).name)
    reasons = []
    if shared_groups:
        reasons.append(f'{len(shared_groups)} calibration groups also appear in other splits, for example {sorted(map(str, shared_groups))[:3]}.')
    if shared_sources:
        reasons.append(f'{len(shared_sources)} calibration source videos also appear in other splits, for example {sorted(shared_sources)[:1]}.')
    if unverified and not disjoint_by_construction:
        reasons.append(f'Disjointness unverified: no shared group column or source path to compare with {", ".join(unverified)} (clip bytes with no group column hold neither).')
    return reasons


def run_split_files(checkpoint) -> list[Path]:
    """The train, validation and test files of the run that holds these weights: <run>/trainval_output_*/weights and <run>/data_splits_*/."""
    run_dir = Path(checkpoint).resolve().parent.parent
    return [path for folder in sorted(run_dir.glob('data_splits_*')) for name in ('train', 'val', 'test') for path in [folder / f'{name}_split.parquet'] if path.is_file()]


def run_label_mapping(checkpoint) -> Optional[dict]:
    """The run's {class name: index}, or None."""
    for folder in sorted(Path(checkpoint).resolve().parent.parent.glob('data_splits_*')):
        path = folder / 'label_mapping.json'
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8-sig'))
    return None


def ambiguous_numeric_names(mapping: Optional[dict]) -> bool:
    """Class names that are all numbers, some differing from their index (LabelEncoder on '1', '10', '2' gives {'1': 0, '10': 1, '2': 2}): a label 1 may be either."""
    return bool(mapping) and all(str(name).lstrip('-').isdigit() for name in mapping) and any(str(index) != str(name) for name, index in mapping.items())


def is_run_calibration_split(calib_df, checkpoint) -> bool:
    """The calibration_split.parquet deeptune.py carved for this run, whose labels it already encoded as indices."""
    path = Path(calib_df).resolve()
    return (path.name == CALIBRATION_FILENAME and path.parent.name.startswith('data_splits_')
            and path.parent.parent == Path(checkpoint).resolve().parent.parent)


def integer_labels(values: pd.Series, mapping: Optional[dict], num_classes: int, source='calibration', encoded: bool = False) -> np.ndarray:
    """Class indices in [0, num_classes), or a refusal. A wrong label reads as a wrong prediction, so bad labels would silently decide the threshold.
    `encoded`: the labels are known to be indices already (the run's own calibration split)."""
    if values.isna().any():
        raise ValueError(f'{source} labels: {int(values.isna().sum())} are missing.')
    numeric = pd.to_numeric(values, errors='coerce')
    if numeric.notna().all():
        labels = numeric.to_numpy(dtype=np.float64)
        if (labels != np.round(labels)).any():
            raise ValueError(f'{source} labels must be whole class indices, found {sorted(set(labels[labels != np.round(labels)]))[:5]}.')
        labels = labels.astype(np.int64)
        if not encoded and ambiguous_numeric_names(mapping):  # refused, not guessed: reading names as indices would mislabel rows and still claim the guarantee
            raise ValueError(f"This run's class names are numbers that differ from their indices ({dict(list(mapping.items())[:3])}...), so a {source} label like "
                             f"{int(labels[0])} could be a name or an index. Use the run's own calibration_split.parquet, whose labels deeptune.py encoded, "
                             'or calibrate a run whose class names are not numbers.')
    else:
        if not mapping:
            raise ValueError(f'{source} labels hold class names, but the run has no label_mapping.json to read them with.')
        names = values.astype(str)
        unknown = sorted(set(names) - set(map(str, mapping)))
        if unknown:
            raise ValueError(f'{source} labels name classes this run does not have: {unknown[:5]}.')
        lookup = {str(name): int(index) for name, index in mapping.items()}
        labels = names.map(lookup).to_numpy(dtype=np.int64)
    outside = sorted(set(labels[(labels < 0) | (labels >= num_classes)].tolist()))
    if outside:
        raise ValueError(f'{source} labels {outside[:5]} are outside the model\'s {num_classes} classes, [0, {num_classes}).')
    return labels


def _read_with_paths(path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path)
    if 'videos' in frame.columns and len(frame) and isinstance(frame['videos'].iloc[0], str):
        frame['videos'] = [value if Path(value).is_absolute() else str(path.parent / value) for value in frame['videos']]
    return frame


def fit_on_calibration_split(calib_df, checkpoint, *, target_error: float, delta: float = DEFAULT_DELTA, guarantee: str = 'selective_risk', score_name: str = 'max_prob',
                             model_version: str, num_classes: int, added_layers: int, embed_size: int, freeze_backbone: bool, use_peft: bool, batch_size: int, num_frames: int,
                             pooling: str, video_options, grouper: Optional[str] = None, other_splits=(), lora_rank: Optional[int] = None, split: str = 'calibration',
                             void_reasons=(), disjoint_by_construction: bool = False) -> AbstentionRule:
    """One pass over the calibration rows through inference's own path, so scores match what infer.py computes, including --test_clips and --test_crops. Saves next to the checkpoint."""
    import torch
    from evaluators.error_model import find_error_model, weights_fingerprint
    from inference.video import predict_video
    from options import DEVICE
    checkpoint = Path(checkpoint)
    frame = _read_with_paths(calib_df)
    if 'labels' not in frame.columns:
        raise ValueError(f'{calib_df} has no labels column, so it cannot calibrate anything.')
    mapping = run_label_mapping(checkpoint)
    labels = integer_labels(frame['labels'], mapping, num_classes, encoded=is_run_calibration_split(calib_df, checkpoint))
    frame['labels'] = labels
    predictions = predict_video(frame, checkpoint, model_version, num_classes, added_layers, embed_size, freeze_backbone, 'cls', use_peft, batch_size, num_frames, pooling,
                                None, video_options=video_options, lora_rank=lora_rank, abstain=False)
    wrong = predictions['predicted_class'].to_numpy() != labels[:len(predictions)]
    groups, sources = text_units(calib_df, grouper)
    from trainers.training_basics import data_fingerprint
    provenance = {'weights_sha256': weights_fingerprint(torch.load(checkpoint, map_location=DEVICE, weights_only=True)),
                  'pipeline': pipeline_record(video_options, num_frames, pooling), 'calibration_file': data_fingerprint(calib_df),
                  'selection': _selection_record(checkpoint.parent), 'label_mapping_sha256': mapping_fingerprint(mapping),
                  'error_model_sha256': file_fingerprint(find_error_model(checkpoint))}
    rule = fit_abstention(score_values(predictions, score_name), wrong, groups, sources, target_error=target_error, delta=delta, guarantee=guarantee, num_classes=num_classes,
                          score_name=score_name, split=split, provenance=provenance,
                          void_reasons=[*void_reasons, *_void_reasons(calib_df, grouper, other_splits, disjoint_by_construction)])
    path = rule.save(checkpoint.parent / ABSTENTION_FILENAME)
    write_fit_table(rule, checkpoint.parent)
    print(f'Abstention rule saved to {path}: {rule.info["headline"]}')
    for warning in rule.info['warnings']:
        print(f'Warning: {warning}')
    return rule


def _selection_record(run_dir: Path) -> Optional[dict]:
    """T1's record of which weights validation chose. Validation used for selection is why calibration is a separate split."""
    try:
        record = json.loads((run_dir / 'error_model_fit.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return {name: record.get(name) for name in ('validation_used_for_model_selection', 'selection_metric', 'best_epoch')}


def report_on_eval_dir(rule: AbstentionRule, eval_dir, test_df, grouper=None, n_boot: int = 2000, calib_df=None) -> Path:
    """abstention_report.json, the curve and its plot, in the evaluation folder. Reads what evaluation wrote. Nothing is fitted on these rows."""
    from evaluators.video.report import shared_source_warning, split_units
    eval_dir = Path(eval_dir)
    table = pd.read_csv(eval_dir / 'predictions.csv')
    groups, sources = split_units(test_df, grouper)
    warning = shared_source_warning(test_df, calib_df) if calib_df is not None else None
    report, curve = build_report(rule, table, groups, sources, split='test', n_boot=n_boot, extra_warnings=[warning] if warning else [])
    answered = rule.answers(score_values(table, rule.score_name))
    predicted = table['predicted_class'].astype(str)
    decisions = table.assign(answered=answered, decision=predicted.where(answered, 'not sure'))
    decisions.to_csv(eval_dir / 'predictions_abstention.csv', index=False)
    path = write_report(report, curve, eval_dir)
    print(f'The selective-risk report is in {path}')
    return path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Fit "not sure" abstention for a finished video run, on groups it never saw.', allow_abbrev=False)
    p.add_argument('--checkpoint', required=True, type=Path, help="The run's model_weights.pth. cli_arguments.json beside it supplies the model settings.")
    p.add_argument('--calib_df', required=True, type=Path, help='Calibration rows with labels: Parquet or CSV. Their groups must be new to the model.')
    p.add_argument('--target_error', required=True, type=float, help='The target, between 0 and 1.')
    p.add_argument('--abstain_delta', type=float, default=DEFAULT_DELTA, help=f'Failure chance over the calibration draw (default {DEFAULT_DELTA}).')
    p.add_argument('--abstain_guarantee', choices=GUARANTEES, default='selective_risk', help='What the target bounds (default selective_risk).')
    p.add_argument('--abstain_score', choices=SCORES, default='max_prob', help='What ranks clips (default max_prob).')
    p.add_argument('--grouper', default=None, help="The group column. Default: 'group' when the file has one.")
    p.add_argument('--other_df', nargs='*', type=Path, default=[], help="""More files to check against, on top of the run's own data_splits_* folder, which is always checked. Overlap voids the guarantee, and so does having nothing to compare with.""")
    p.add_argument('--split_name', default='calibration', help="What these rows are. 'validation', 'test' and 'train' are refused.")
    p.add_argument('--batch_size', type=int, default=16, help='Batch size (default 16).')
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from datasets.video_datasets import video_options_from_args
    from infer import _required_stub, apply_recorded_video_settings, build_parser
    recorded = json.loads((args.checkpoint.parent / 'cli_arguments.json').read_text(encoding='utf-8-sig'))
    settings = build_parser().parse_args(_required_stub() + ['--checkpoint', str(args.checkpoint)])
    apply_recorded_video_settings(settings, ['--checkpoint', str(args.checkpoint)])
    grouper = args.grouper or recorded.get('grouper') or None  # split_units falls back to a 'group' column
    if find_rule(args.checkpoint) is not None:
        print(f'Warning: overwriting the abstention rule beside {args.checkpoint}. Refitting after seeing results is selection, which the guarantee does not cover.')
    run_dir = Path(args.checkpoint).resolve().parent.parent
    if any(path.is_dir() for path in run_dir.glob('eval_output_*')):
        print(f'Warning: {run_dir} already has test results (eval_output_*). A rule fitted after they were seen is not independent of them if they influenced this target, score or split. '
              'Any abstention_report.json there still describes the earlier rule.')
    other_splits = list(dict.fromkeys([*args.other_df, *run_split_files(args.checkpoint)]))
    void_reasons = ["The run used --resume_allow_data_change, so its weights may have trained on these rows."] if recorded.get('resume_allow_data_change') else []
    fit_on_calibration_split(args.calib_df, args.checkpoint, target_error=args.target_error, delta=args.abstain_delta, guarantee=args.abstain_guarantee, score_name=args.abstain_score,
                             model_version=recorded['model_version'], num_classes=recorded['num_classes'], added_layers=recorded.get('added_layers') or 2,
                             embed_size=recorded.get('embed_size') or 1000, freeze_backbone=bool(recorded.get('freeze_backbone')), use_peft=bool(recorded.get('use_peft')),
                             batch_size=args.batch_size, num_frames=settings.num_frames, pooling=settings.pooling, video_options=video_options_from_args(settings), grouper=grouper,
                             other_splits=other_splits, lora_rank=settings.lora_rank, split=args.split_name, void_reasons=void_reasons)
    return 0


if __name__ == '__main__':
    sys.exit(main())

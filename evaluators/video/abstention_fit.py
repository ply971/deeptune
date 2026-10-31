"""Fits abstention.json for a trained video model on a separate calibration split, and reports the rule on held-out predictions.

deeptune.py calls `carve_calibration_split`, `fit_on_calibration_split` and `report_on_eval_dir`. To fit a finished run later:
    python -m evaluators.video.abstention_fit --checkpoint <run>/model_weights.pth --calib_df <groups the run never saw> --target_error 0.05
        --other_df train_split.parquet val_split.parquet test_split.parquet
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
                                   overlapping_units, pipeline_record, score_values, write_fit_table, write_report)
from evaluators.abstention import UNIT_NOTES
from evaluators.stats import unit_labels

CALIBRATION_FILENAME = 'calibration_split.parquet'
CALIBRATION_SUMMARY_FILENAME = 'calibration_split_summary.json'


def carve_calibration_split(train_path, calib_fraction: float, train_fraction: float, grouper: Optional[str], seed: Optional[int]) -> Path:
    """Moves whole units from the training split into calibration_split.parquet beside it. The validation and test files stay as they were."""
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
    print(f'Calibration split: {len(calibration)} clips from {summary["calibration_units"]} {unit} units, taken from the training split. Training keeps {len(train)} clips.')
    if unit != 'group':
        print(f'Warning: {UNIT_NOTES[unit]}')
    return path


def _void_reasons(calib_path, grouper, other_paths) -> list[str]:
    """Why these rows do not make a clean calibration set. The fit still writes its table, and says it carries no guarantee."""
    from evaluators.video.report import shared_source_warning, split_units
    reasons = []
    groups = split_units(calib_path, grouper)[0]
    if groups is not None:
        shared = overlapping_units(groups, *(split_units(path, grouper)[0] or [] for path in other_paths))
        if shared:
            reasons.append(f'{len(shared)} calibration groups also appear in other splits, for example {shared[:3]}.')
    warning = shared_source_warning(calib_path, *other_paths) if other_paths else None
    if warning:
        reasons.append('Calibration rows share source videos with other splits: ' + warning.split('.')[0] + '.')
    return reasons


def _read_with_paths(path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path)
    if 'videos' in frame.columns and len(frame) and isinstance(frame['videos'].iloc[0], str):
        frame['videos'] = [value if Path(value).is_absolute() else str(path.parent / value) for value in frame['videos']]
    return frame


def fit_on_calibration_split(calib_df, checkpoint, *, target_error: float, delta: float = DEFAULT_DELTA, guarantee: str = 'selective_risk', score_name: str = 'max_prob',
                             model_version: str, num_classes: int, added_layers: int, embed_size: int, freeze_backbone: bool, use_peft: bool, batch_size: int, num_frames: int,
                             pooling: str, video_options, grouper: Optional[str] = None, other_splits=(), lora_rank: Optional[int] = None, split: str = 'calibration') -> AbstentionRule:
    """One pass over the calibration rows through inference's own path, so scores match what infer.py computes, including --test_clips and --test_crops. Saves next to the checkpoint."""
    import torch
    from evaluators.error_model import weights_fingerprint
    from evaluators.video.report import split_units
    from inference.video import predict_video
    from options import DEVICE
    checkpoint = Path(checkpoint)
    frame = _read_with_paths(calib_df)
    if 'labels' not in frame.columns:
        raise ValueError(f'{calib_df} has no labels column, so it cannot calibrate anything.')
    predictions = predict_video(frame, checkpoint, model_version, num_classes, added_layers, embed_size, freeze_backbone, 'cls', use_peft, batch_size, num_frames, pooling,
                                None, video_options=video_options, lora_rank=lora_rank, abstain=False)
    wrong = predictions['predicted_class'].to_numpy() != frame['labels'].to_numpy()[:len(predictions)]
    groups, sources = split_units(calib_df, grouper)
    from trainers.training_basics import data_fingerprint
    provenance = {'weights_sha256': weights_fingerprint(torch.load(checkpoint, map_location=DEVICE, weights_only=True)),
                  'pipeline': pipeline_record(video_options, num_frames, pooling), 'calibration_file': data_fingerprint(calib_df),
                  'selection': _selection_record(checkpoint.parent)}
    rule = fit_abstention(score_values(predictions, score_name), wrong, groups, sources, target_error=target_error, delta=delta, guarantee=guarantee, num_classes=num_classes,
                          score_name=score_name, split=split, provenance=provenance, void_reasons=_void_reasons(calib_df, grouper, other_splits))
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
    p.add_argument('--other_df', nargs='*', type=Path, default=[], help='Train, validation and test files: overlap with them voids the guarantee, and the output says so.')
    p.add_argument('--split_name', default='calibration', help="What these rows are. 'validation', 'test' and 'train' are refused.")
    p.add_argument('--batch_size', type=int, default=16, help='Batch size (default 16).')
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from datasets.video_datasets import video_options_from_args
    from handlers.split_dataset import VIDEO_GROUP_COLUMN
    from infer import _required_stub, apply_recorded_video_settings, build_parser
    recorded = json.loads((args.checkpoint.parent / 'cli_arguments.json').read_text(encoding='utf-8-sig'))
    settings = build_parser().parse_args(_required_stub() + ['--checkpoint', str(args.checkpoint)])
    apply_recorded_video_settings(settings, ['--checkpoint', str(args.checkpoint)])
    grouper = args.grouper or recorded.get('grouper') or None
    if grouper is None and args.calib_df.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        grouper = VIDEO_GROUP_COLUMN if VIDEO_GROUP_COLUMN in pq.read_schema(args.calib_df).names else None
    fit_on_calibration_split(args.calib_df, args.checkpoint, target_error=args.target_error, delta=args.abstain_delta, guarantee=args.abstain_guarantee, score_name=args.abstain_score,
                             model_version=recorded['model_version'], num_classes=recorded['num_classes'], added_layers=recorded.get('added_layers') or 2,
                             embed_size=recorded.get('embed_size') or 1000, freeze_backbone=bool(recorded.get('freeze_backbone')), use_peft=bool(recorded.get('use_peft')),
                             batch_size=args.batch_size, num_frames=settings.num_frames, pooling=settings.pooling, video_options=video_options_from_args(settings), grouper=grouper,
                             other_splits=args.other_df, lora_rank=settings.lora_rank, split=args.split_name)
    return 0


if __name__ == '__main__':
    sys.exit(main())

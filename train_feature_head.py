"""Train a head on cached snippet features (P-102), through the shared Trainer.

    python extract_features.py --data clips.csv --out features --cache_tokens 8x4x4
    python train_feature_head.py --features features --train_df train.csv --val_df val.csv --out probe --num_classes 5

The heads are in ``src/video/feature_heads.py``; no pixels are decoded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train a head (attentive or linear probe) on cached snippet features.')
    p.add_argument('--features', required=True, help='Feature folder written by extract_features.py.')
    p.add_argument('--train_df', required=True, help='Training rows: CSV/XLSX/parquet with videos (or video_id) and labels.')
    p.add_argument('--val_df', required=True, help='Validation rows.')
    p.add_argument('--test_df', default=None, help='Optional test rows, disjoint from train and val; writes metrics.json and per-video predictions.csv.')
    p.add_argument('--out', required=True, type=Path, help='Output folder.')
    p.add_argument('--head', choices=['attentive_probe', 'linear_probe'], default='attentive_probe', help='Head (default attentive_probe).')
    p.add_argument('--streams', nargs='+', choices=['rgb', 'motion'], default=None,
                   help='Streams to concatenate (default: all in the folder).')
    p.add_argument('--target', default='labels', help='Numeric label column (default labels; e.g. label_idx).')
    p.add_argument('--mode', choices=['cls', 'reg'], default='cls', help='Classification or regression (default cls).')
    p.add_argument('--num_classes', type=int, default=2, help='Classes when --mode cls (default 2).')
    p.add_argument('--num_snippets', type=int, default=8,
                   help='Snippets per video: evenly spaced at test, one random per segment in training (default 8).')
    p.add_argument('--batch_size', type=int, default=16, help='Batch size (default 16).')
    p.add_argument('--num_epochs', type=int, default=10, help='Epochs (default 10).')
    p.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (default 1e-3).')
    p.add_argument('--probe_width', type=int, default=256, help='attentive_probe: token width (default 256).')
    p.add_argument('--probe_depth', type=int, default=1, help='attentive_probe: cross-attention blocks, 1 to 4 (default 1).')
    p.add_argument('--probe_heads', type=int, default=4, help='attentive_probe: attention heads (default 4).')
    p.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (default 0: cached features load fast).')
    p.add_argument('--fixed-seed', dest='fixed_seed', action='store_true', help='Use fixed seed 42.')
    return p


def _loader(dataset, batch_size, shuffle, workers):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def audit_splits(splits):
    """Refuse splits that share a feature entry, a source video (so two trims of one video count) or a `group`."""
    import itertools
    import pandas as pd
    for (left_name, left), (right_name, right) in itertools.combinations(splits.items(), 2):
        for field in ('video_ids', 'sources', 'groups'):
            a, b = getattr(left, field), getattr(right, field)
            if a is None or b is None:
                continue
            key = str.casefold if field == 'video_ids' else str  # Clip and clip are one file on NTFS
            shared = {key(str(v)) for v in a if pd.notna(v)} & {key(str(v)) for v in b if pd.notna(v)}
            if shared:
                raise ValueError(f'Split leakage: {left_name} and {right_name} share {field}: '
                                 f'{sorted(shared)[:3]}. Use disjoint videos and groups.')
    grouped = all(d.groups is not None for d in splits.values())
    if not grouped:
        print("WARNING: not every split has a 'group' column, so clips of one person or scene could still be on both "
              'sides; only ids and source videos were checked.')
    return {'checked_splits': list(splits), 'video_and_source_overlap': False,
            'group_check': 'checked' if grouped else 'incomplete: group column missing from one or more splits'}


def _chance_row(truth, train_labels, mode):
    """The score of a model that learned only the training labels: the majority class, or the mean for regression."""
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    truth, train_labels = np.asarray(truth), np.asarray(train_labels)
    if mode == 'cls':
        values, counts = np.unique(train_labels, return_counts=True)
        majority = values[counts.argmax()]
        constant = np.full(len(truth), majority)
        return {'predicts': f'training majority class {int(majority)}', 'accuracy': accuracy_score(truth, constant),
                'balanced_accuracy': balanced_accuracy_score(truth, constant),
                'macro_f1': f1_score(truth, constant, average='macro', zero_division=0)}
    constant = np.full(len(truth), float(train_labels.mean()))
    return {'predicts': f'training mean {float(train_labels.mean()):g}', 'mse': mean_squared_error(truth, constant),
            'mae': mean_absolute_error(truth, constant), 'r2': r2_score(truth, constant) if len(truth) > 1 else None}


def evaluate_head(model, loader, mode, predictions_path=None, train_labels=None):
    """Metrics and per-video predictions; `train_labels` adds the `chance` row to compare them with."""
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                                 mean_squared_error, r2_score)
    from options import DEVICE
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(DEVICE)).cpu()
            truth.extend(labels.numpy().reshape(-1).tolist())
            scores.append(outputs.softmax(dim=1).numpy() if mode == 'cls' else outputs.numpy().reshape(-1))
    if not truth:
        raise ValueError('Cannot evaluate an empty feature dataset.')
    scores = np.concatenate(scores)
    prediction = scores.argmax(axis=1) if mode == 'cls' else scores
    rows = {'video_id': loader.dataset.video_ids, 'target': truth, 'prediction': prediction}
    if getattr(loader.dataset, 'groups', None) is not None:
        rows['group'] = loader.dataset.groups
    if mode == 'cls':
        metrics = {'accuracy': accuracy_score(truth, prediction),
                   'balanced_accuracy': balanced_accuracy_score(truth, prediction),
                   'macro_f1': f1_score(truth, prediction, average='macro', zero_division=0),
                   'confusion_matrix': confusion_matrix(truth, prediction, labels=range(scores.shape[1])).tolist()}
        rows.update({f'probability_{i}': scores[:, i] for i in range(scores.shape[1])})
    else:
        mse = mean_squared_error(truth, prediction)
        metrics = {'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': mean_absolute_error(truth, prediction),
                   'r2': r2_score(truth, prediction) if len(truth) > 1 else None}
    if predictions_path is not None:
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
    if train_labels is not None:
        metrics['chance'] = _chance_row(truth, train_labels, mode)
    return {**metrics, 'n': len(truth)}


def feature_provenance(dataset):
    """What the probe's features were made with, for its run output."""
    streams = dataset.store.info()['streams']
    provenance = {'feature_config_hash': {name: streams[name].get('config_hash') for name in dataset.streams if name in streams}}
    # Every set of weights behind these features, from the sidecars extraction wrote.
    sidecars = {name: dataset.store.read_sidecar(name, dataset.video_ids[0]) or {} for name in dataset.streams}
    provenance['weights_licence'] = {name: sidecar.get('weights_licence') for name, sidecar in sidecars.items()}
    provenance['weights_revision'] = {name: sidecar.get('weights_revision') for name, sidecar in sidecars.items()}
    if 'motion' in dataset.streams:
        sidecar = dataset.store.read_sidecar('motion', dataset.video_ids[0]) or {}
        provenance['motion_weights_trained_on_motion'] = sidecar.get('motion_weights_trained_on_motion')
        if provenance['motion_weights_trained_on_motion'] is False:
            print('NOTE: the motion stream runs through RGB filters that were not trained on motion.')
    return provenance


def train(args: argparse.Namespace) -> Path:
    from datasets.feature_dataset import CachedFeatureDataset
    from src.video.feature_heads import build_feature_head
    from trainers.vision.trainer import Trainer
    from utils import set_seed
    seed = set_seed(args.fixed_seed)
    common = dict(features_dir=args.features, streams=args.streams, num_snippets=args.num_snippets, target=args.target)
    train_set = CachedFeatureDataset(args.train_df, train=True, **common)
    val_set = CachedFeatureDataset(args.val_df, train=False, **common)
    test_set = CachedFeatureDataset(args.test_df, train=False, **common) if args.test_df else None
    splits = {'train': train_set, 'val': val_set}
    if test_set is not None:
        splits['test'] = test_set
    audit = audit_splits(splits)
    options = dict(width=args.probe_width, depth=args.probe_depth, num_heads=args.probe_heads) if args.head == 'attentive_probe' else {}
    model = build_feature_head(args.head, train_set.feature_dim(), args.num_classes, args.mode, **options)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'split_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    trainer = Trainer(model, train_loader=_loader(train_set, args.batch_size, True, args.num_workers),
                      val_loader=_loader(val_set, args.batch_size, False, args.num_workers),
                      learning_rate=args.learning_rate, mode=args.mode, num_epochs=args.num_epochs, output_dir=args.out)
    trainer.fit_error_model = False
    trainer.train()
    weights = args.out / 'model_weights.pth'
    trainer.saveModel(path=str(weights))
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    arguments.update(input_dim=train_set.feature_dim(), seed=seed, **feature_provenance(train_set))
    (args.out / 'cli_arguments.json').write_text(json.dumps(arguments, indent=2), encoding='utf-8')
    if args.test_df:
        metrics = evaluate_head(model, _loader(test_set, args.batch_size, False, 0), args.mode,
                                predictions_path=args.out / 'predictions.csv', train_labels=train_set.labels)
        (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f'Test: {metrics}')
    return weights


def main(argv=None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == '__main__':
    sys.exit(main())

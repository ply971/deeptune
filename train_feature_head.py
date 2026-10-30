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
    p.add_argument('--test_df', default=None, help='Optional test rows; the final metric goes to metrics.json.')
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


def evaluate_head(model, loader, mode):
    import torch
    from options import DEVICE
    model.eval()
    right, squared, total = 0, 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(DEVICE)).cpu()
            total += len(labels)
            if mode == 'cls':
                right += int((outputs.argmax(dim=1) == labels).sum())
            else:
                squared += float(((outputs.view(-1) - labels.float()) ** 2).sum())
    return {'accuracy': right / total, 'n': total} if mode == 'cls' else {'mse': squared / total, 'n': total}


def train(args: argparse.Namespace) -> Path:
    from datasets.feature_dataset import CachedFeatureDataset
    from src.video.feature_heads import build_feature_head
    from trainers.vision.trainer import Trainer
    from utils import set_seed
    set_seed(args.fixed_seed)
    common = dict(features_dir=args.features, streams=args.streams, num_snippets=args.num_snippets, target=args.target)
    train_set = CachedFeatureDataset(args.train_df, train=True, **common)
    val_set = CachedFeatureDataset(args.val_df, train=False, **common)
    options = dict(width=args.probe_width, depth=args.probe_depth, num_heads=args.probe_heads) if args.head == 'attentive_probe' else {}
    model = build_feature_head(args.head, train_set.feature_dim(), args.num_classes, args.mode, **options)
    args.out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_loader=_loader(train_set, args.batch_size, True, args.num_workers),
                      val_loader=_loader(val_set, args.batch_size, False, args.num_workers),
                      learning_rate=args.learning_rate, mode=args.mode, num_epochs=args.num_epochs, output_dir=args.out)
    trainer.fit_error_model = False
    trainer.train()
    weights = args.out / 'model_weights.pth'
    trainer.saveModel(path=str(weights))
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    arguments['input_dim'] = train_set.feature_dim()
    (args.out / 'cli_arguments.json').write_text(json.dumps(arguments, indent=2), encoding='utf-8')
    if args.test_df:
        test_set = CachedFeatureDataset(args.test_df, train=False, **common)
        metrics = evaluate_head(model, _loader(test_set, args.batch_size, False, 0), args.mode)
        (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f'Test: {metrics}')
    return weights


def main(argv=None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == '__main__':
    sys.exit(main())

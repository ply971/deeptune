"""Average class probabilities across trained video runs (inference/ensemble.py).

    python ensemble.py --runs RUN_A RUN_B --data clips/ --out results
    python ensemble.py --predictions probe_a/predictions.csv probe_b/predictions.csv --out results

``--runs`` runs each model on new data with its recorded settings. ``--predictions`` combines finished tables, the way
to ensemble ``train_feature_head.py`` probes, whose rows line up by ``video_id``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Average class probabilities across trained video runs.')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--runs', nargs='+', type=Path, metavar='RUN', help='Trained runs: weights files or folders holding one model_weights.pth, with cli_arguments.json beside it.')
    source.add_argument('--predictions', nargs='+', type=Path, metavar='CSV', help='Finished predictions.csv files with probability_<k> columns and the same rows.')
    p.add_argument('--out', required=True, type=Path, help='Output folder: predictions.csv, ensemble.json, and inference_metrics.json when there are labels.')
    p.add_argument('--data', type=Path, default=None, help='With --runs: a folder of clips, one clip, or a CSV/XLSX list of paths.')
    p.add_argument('--batch_size', type=int, default=16, help='With --runs: batch size for each member (default 16).')
    p.add_argument('--mode', choices=['cls', 'reg'], default='cls', help='With --predictions: classification or regression (default cls). --runs reads it from the runs.')
    p.add_argument('--label_mapping', type=Path, default=None, help="A run's data_splits_*/label_mapping.json, for class names.")
    p.add_argument('--allow_unverified_labels', action='store_true',
                   help='Average members even if what their class indices mean cannot be verified. ensemble.json says so.')
    p.add_argument('--accept_weights_licence', nargs='+', default=None, metavar='LICENCE',
                   help='With --runs: accept the licence of opt-in weights by name. Runs that recorded their acceptance need no repeat.')
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from inference.ensemble import ensemble_runs, ensemble_tables
    if args.runs:
        if args.data is None:
            parser.error('--runs needs --data.')
        ensemble_runs(args.runs, args.data, args.out, args.batch_size, args.label_mapping, args.accept_weights_licence, args.allow_unverified_labels)
    else:
        ensemble_tables(args.predictions, args.out, args.mode, args.label_mapping, args.allow_unverified_labels)
    return 0


if __name__ == '__main__':
    sys.exit(main())

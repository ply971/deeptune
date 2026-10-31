"""How often do the frame-comparing models pass the direction oracle? Reruns tests/models/test_temporal_oracle.py's own
data, training and checks over fixed dataset and model seeds, so one lucky seed cannot stand in for the model (V-046).

The seeds and settings below were fixed before the first run. They are not tuned on the results: this measures the
settings the oracle already uses, on the same held-out clips those settings were chosen on. So the numbers are wiring
evidence, not an architecture ranking and not a sample-complexity claim.

One job at a time, CPU only, no DataLoader workers:

    $env:CUDA_VISIBLE_DEVICES='-1'; $env:OMP_NUM_THREADS='4'
    .venv\\Scripts\\python.exe scripts\\study_temporal_oracle_seeds.py --study cnn_rnn --out .artifacts\\b2_seeds\\cnn_rnn.jsonl

Each run is appended to --out as one JSON line as soon as it finishes, and a finished run is not repeated.
"""
import argparse
import json
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'tests' / 'models')]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.video.tiny_backbone import TinyResNet  # noqa: E402
from test_temporal_oracle import (  # noqa: E402
    MOTION_AWARE_MIN, REVERSED_MAX, SLOW_SPEEDS, TT_BUDGET, TT_POOLING, Task, _accuracy, _fit, _permutations,
    passes_motion_oracle)

MODEL_SEEDS = (0, 1, 2, 3, 4)


def _runs(pooling, n_train, dataset_seeds=(0, 1), model_seeds=MODEL_SEEDS, budget=None, speed_range=(1.0, 2.0), grid=2):
    for dataset_seed in dataset_seeds:
        for model_seed in model_seeds:
            yield dict(pooling=pooling, n_train=n_train, dataset_seed=dataset_seed, model_seed=model_seed,
                       budget=budget, speed_range=list(speed_range), grid=grid)


STUDIES = {
    # V-046 item 1. The 240-clip runs repeat the critic's check; grid=1 is the global pool the old docstring compared with.
    'cnn_rnn': lambda: [*_runs('cnn_rnn', 240), *_runs('cnn_rnn', 800),
                        *_runs('cnn_rnn', 240, dataset_seeds=(0,), grid=1), *_runs('cnn_rnn', 800, dataset_seeds=(0,), grid=1)],
    # V-046 item 2: the settings of test_temporal_transformer_beats_chance_given_enough_clips.
    'temporal_transformer': lambda: list(_runs(TT_POOLING, 1500, budget=list(TT_BUDGET))),
    # The coordinator's decision on late_fusion: the half-speed task is its done-when, and the default speed is bounded.
    'late_fusion': lambda: [*_runs('late_fusion', 800, speed_range=SLOW_SPEEDS), *_runs('late_fusion', 800)],
}


@contextmanager
def _grid(size):
    original = TinyResNet.GRID
    TinyResNet.GRID = size
    try:
        yield
    finally:
        TinyResNet.GRID = original


def _key(run):
    return json.dumps({name: run[name] for name in ('pooling', 'n_train', 'dataset_seed', 'model_seed', 'speed_range', 'grid')},
                      sort_keys=True)


def _measure(run, scratch):
    started = time.time()
    data = Path(tempfile.mkdtemp(dir=scratch))
    task = Task(data, 32, run['n_train'], speed_range=tuple(run['speed_range']), seed=run['dataset_seed'])
    x_train, y_train = task.split('train')
    with _grid(run['grid']):
        model = _fit(run['pooling'], x_train, y_train, seed=run['model_seed'], budget=run['budget'] and tuple(run['budget']))
    x, y = task.split('test')
    accuracy = _accuracy(model, x, y)
    reversed_accuracy = _accuracy(model, x.flip(1), y)
    shuffled = [_accuracy(model, x[:, order], y) for order in _permutations()]
    if run['pooling'] == 'late_fusion':  # its oracle has no shuffle check: only the first and last frames matter to it
        passes = accuracy >= MOTION_AWARE_MIN and reversed_accuracy <= REVERSED_MAX
    else:
        passes = passes_motion_oracle(accuracy, reversed_accuracy, shuffled)
    return {**run, 'train_accuracy': round(_accuracy(model, x_train, y_train), 4), 'accuracy': round(accuracy, 4),
            'reversed': round(reversed_accuracy, 4), 'shuffled': [round(v, 4) for v in shuffled],
            'passes': passes, 'seconds': round(time.time() - started, 1)}


def summarise(records):
    groups = {}
    for record in records:
        key = (record['pooling'], record['n_train'], tuple(record['speed_range']), record['grid'], record['dataset_seed'])
        groups.setdefault(key, []).append(record)
    print('pooling | clips | speed | grid | dataset | held-out per model seed | mean | passed')
    for (pooling, clips, speed, grid, dataset), group in sorted(groups.items(), key=str):
        scores = [r['accuracy'] for r in sorted(group, key=lambda r: r['model_seed'])]
        print(f"{pooling} | {clips} | {speed[0]}-{speed[1]} | {grid} | {dataset} | {' '.join(f'{v:.3f}' for v in scores)} | "
              f"{np.mean(scores):.3f} | {sum(r['passes'] for r in group)}/{len(group)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--study', required=True, choices=sorted(STUDIES))
    parser.add_argument('--out', required=True, type=Path, help='JSON lines file; finished runs in it are not repeated')
    args = parser.parse_args()
    if torch.cuda.is_available():
        parser.error("CPU only: set CUDA_VISIBLE_DEVICES to -1 first.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.exists() else []
    finished = {_key(record) for record in records}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as scratch:
        for run in STUDIES[args.study]():
            if _key(run) in finished:
                continue
            record = _measure(run, scratch)
            records.append(record)
            with args.out.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    summarise(records)


if __name__ == '__main__':
    main()

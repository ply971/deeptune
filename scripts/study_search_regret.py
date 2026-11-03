"""Regret study of select_model.py's strategies on a simulated grid (CPU, no training, a few minutes).

For each seed: exhaustive cross-validates every candidate of a small grid (the reference); smart, smart without its
stage-3 re-check, random and bayes then search the same development folds. A strategy's regret is exhaustive's best CV
score minus exhaustive's CV score of the strategy's pick: how much worse its choice is, measured the same way.
random and bayes get the same number of trainings smart used on that seed.

The candidates' qualities come from handlers/model_selection/simulate.py's `interaction_surface`: full fine-tuning is the
best setting at the low learning rate and the worst at the high one, and the base learning rate is the high one, so
tuning one group at a time from the base settings is misled. These are simulated scores, not results on real data; the
real-data comparison is a GPU gate (docs/video-plan/reports/R3b.md).

    python scripts/study_search_regret.py --seeds 0 1 2 3 4 5 6 7 --out .artifacts/r3b/regret
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = ['--backbones', 'resnet_tiny', 'siglip_tiny', 'vjepa2_tiny', '--train_modes', 'frozen', 'lora:8', 'full',
        '--learning_rates', '1e-3', '1e-4', '--augmentations', 'off', 'aug']
VARIANTS = {'smart': ['--search', 'smart'], 'smart_no_recheck': ['--search', 'smart', '--recheck_models', '0'],
            'random': ['--search', 'random'], 'bayes': ['--search', 'bayes']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Regret of the smart, random and bayes searches against exhaustive, on a simulated grid.')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(8)), help='Seeds: data split, folds, surface jitter, sampling (default 0-7).')
    parser.add_argument('--out', type=Path, default=Path('.artifacts/r3b/regret'), help='Work folder (default .artifacts/r3b/regret).')
    parser.add_argument('--no_bayes', action='store_true', help='Leave bayes out (when Optuna is not installed).')
    args = parser.parse_args(argv)
    import pandas as pd
    from handlers.model_selection.simulate import regret_study, summarize
    variants = {name: flags for name, flags in VARIANTS.items() if not (args.no_bayes and name == 'bayes')}
    rows = regret_study(args.out, args.seeds, GRID, variants=variants)
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / 'regret_rows.csv', index=False)
    summary = summarize(rows)
    summary.to_csv(args.out / 'regret_summary.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.to_string(index=False))
    print(f'Rows: {args.out / "regret_rows.csv"}. Simulated scores (simulate.interaction_surface), not real data.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

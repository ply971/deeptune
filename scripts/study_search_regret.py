"""Regret study of select_model.py's strategies on simulated grids (CPU, no training, about 15 minutes for 8 seeds).

For each surface and seed: exhaustive cross-validates every candidate of a small grid (the reference); the other variants
then search the same development folds. A variant's regret is exhaustive's best CV score minus exhaustive's CV score of
the variant's pick (how much worse its choice is, measured the same way); true_quality_regret is the same on the
surface's noiseless quality. random and bayes get the same number of trainings smart used on that seed (screens
included, which favours them a little: a probe costs less than a cross-validation).

Surfaces (handlers/model_selection/simulate.py; simulated scores, not results on real data):
  interaction        full fine-tuning best at the low learning rate, worst at the high (base) one; LoRA also prefers
                     the low rate. Written by the R3b builder.
  A_lora_high_rate   the ml-critic's: as above, but LoRA prefers the high rate.
  B_screen_misleads  the ml-critic's: frozen features rank the backbones opposite to fine-tuning.
The variants include smart as it shipped first in R3b's stage 1 (frozen view only) and without stage 3, to show what each
part of the fix does.

    python scripts/study_search_regret.py --seeds 0 1 2 3 4 5 6 7 --out .artifacts/r3b/regret
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = ['--backbones', 'resnet_tiny', 'siglip_tiny', 'vjepa2_tiny', '--train_modes', 'frozen', 'lora:8', 'full',
        '--learning_rates', '1e-3', '1e-4', '--augmentations', 'off', 'aug']
VARIANTS = {'smart': ['--search', 'smart'],
            'smart_frozen_view_only': ['--search', 'smart', '--screen_views', 'frozen'],
            'smart_no_recheck': ['--search', 'smart', '--recheck_models', '0'],
            'random': ['--search', 'random'], 'bayes': ['--search', 'bayes']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Regret of the smart, random and bayes searches against exhaustive, on simulated grids.')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(8)), help='Seeds: data split, folds, surface jitter, sampling (default 0-7).')
    parser.add_argument('--surfaces', nargs='+', default=None, help='Surfaces to run (default: all of simulate.SURFACES).')
    parser.add_argument('--out', type=Path, default=Path('.artifacts/r3b/regret'), help='Work folder (default .artifacts/r3b/regret).')
    parser.add_argument('--no_bayes', action='store_true', help='Leave bayes out (when Optuna is not installed).')
    args = parser.parse_args(argv)
    import pandas as pd
    from handlers.model_selection.simulate import SURFACES, regret_study, summarize
    variants = {name: flags for name, flags in VARIANTS.items() if not (args.no_bayes and name == 'bayes')}
    args.out.mkdir(parents=True, exist_ok=True)
    tables = []
    for name in args.surfaces or list(SURFACES):
        rows = regret_study(args.out / name, args.seeds, GRID, variants=variants, surface_fn=SURFACES[name])
        pd.DataFrame(rows).assign(surface=name).to_csv(args.out / f'regret_rows_{name}.csv', index=False)
        tables.append(summarize(rows).assign(surface=name))
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(args.out / 'regret_summary.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 20, 'display.float_format', '{:.3f}'.format):
        print(summary.to_string(index=False))
    print(f'Rows and summary: {args.out}. Simulated scores (simulate.SURFACES), not real data.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

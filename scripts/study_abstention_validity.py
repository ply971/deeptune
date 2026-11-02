"""How often abstention's guarantees hold on synthetic scores, over many seeds. numpy only, CPU, no model and no video, so it says nothing about real data.

    python scripts/study_abstention_validity.py --seeds 2000

Prints the tables in docs/video-plan/reports/R2.md.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluators.abstention import (conformal_threshold, group_worst_scores, loose_below, realised_coverage_interval, select_selective_threshold, threshold_grid,  # noqa: E402
                                   threshold_stats)
from evaluators.abstention_sim import DEPENDENT, INDEPENDENT, MOSTLY_WRONG, STRESS, Scenario, draw, true_risks  # noqa: E402

GRID = threshold_grid(2)
FLAT_IID = Scenario(clips_per_group=1, group_sd=0.0, score_group_sd=0.0, offset=-2.0, informative=False)  # true error 0.119 at every threshold
HEAVY_GROUPS = Scenario(group_sd=3.0, score_mean=3.0, score_clip_sd=2.0)
SILENT = Scenario(clips_per_group=8, group_sd=1.0, offset=3.0, score_mean=1.0)


def selective_row(name, scenario, n_groups, seeds, alpha, delta, flat):
    population = true_risks(*draw(np.random.default_rng(10_000), 100_000 if scenario.clips_per_group > 1 else 800_000, scenario), GRID)
    truth = np.full(len(GRID), population['group'][0]) if flat else population['group']
    out = [name, n_groups, scenario.clips_per_group, f'{population["group"][0]:.3f}']
    for per_clip in (False, True):
        failures = picked = 0
        answered = []
        for seed in range(seeds):
            score, wrong, groups = draw(np.random.default_rng(seed), n_groups, scenario)
            units = np.arange(len(score)) if per_clip else groups
            tau = select_selective_threshold(threshold_stats(score, wrong, units, GRID), alpha, delta, per_clip or scenario.clips_per_group == 1)
            if tau is not None:
                picked += 1
                failures += truth[np.searchsorted(GRID, tau)] > alpha
                answered.append((score >= tau).mean())
        out += [f'{failures / seeds:.3f}', f'{picked / seeds:.2f}', f'{np.mean(answered):.2f}' if answered else '-']
    return out


def silent_row(n, alpha, seeds):
    population = np.sort(group_worst_scores(*draw(np.random.default_rng(20_000), 200_000, SILENT)))
    realised = np.empty(seeds)
    for seed in range(seeds):
        tau = conformal_threshold(group_worst_scores(*draw(np.random.default_rng(seed), n, SILENT)), alpha)
        realised[seed] = np.searchsorted(population, tau, side='left') / len(population) if tau is not None else 1.0
    interval = realised_coverage_interval(n, alpha)
    inside = ((realised >= interval['low']) & (realised <= interval['high'])).mean()
    return [n, alpha, f'{realised.mean():.4f}', f'{1 - alpha:.3f}', f'{1 - alpha + 1 / (n + 1):.3f}', f'{interval["low"]:.3f} to {interval["high"]:.3f}', f'{inside:.3f}']


def clip_beta_row(n_groups, alpha, seeds):
    score, wrong, _ = draw(np.random.default_rng(30_000), 100_000, MOSTLY_WRONG)
    population = np.sort(np.where(wrong, score, -np.inf))
    n_clips = n_groups * MOSTLY_WRONG.clips_per_group
    claimed = realised_coverage_interval(n_clips, alpha)
    realised = np.empty(seeds)
    for seed in range(seeds):
        score, wrong, _ = draw(np.random.default_rng(seed), n_groups, MOSTLY_WRONG)
        tau = conformal_threshold(np.where(wrong, score, -np.inf), alpha)
        realised[seed] = np.searchsorted(population, tau, side='left') / len(population)
    return [n_groups, n_clips, f'{claimed["low"]:.3f} to {claimed["high"]:.3f}', f'{np.percentile(realised, 2.5):.3f} to {np.percentile(realised, 97.5):.3f}',
            f'{((realised >= claimed["low"]) & (realised <= claimed["high"])).mean():.3f}']


def table(header, rows):
    print('| ' + ' | '.join(map(str, header)) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(map(str, row)) + ' |')
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=500, help='Calibration draws per row (default 500).')
    args = parser.parse_args()
    alpha, delta = 0.10, 0.10

    print(f'Selective risk, target {alpha}, delta {delta}, {args.seeds} calibration draws. "false claims" must stay at or below delta.\n')
    table(['case', 'groups', 'clips per group', 'error answering all', 'groups: false claims', 'picked', 'answered', 'clips as units: false claims', 'picked', 'answered'],
          [selective_row('i.i.d., useful score', INDEPENDENT, 1500, args.seeds, alpha, delta, False),
           selective_row('i.i.d., no threshold can meet it', FLAT_IID, 500, args.seeds, alpha, delta, True),
           selective_row('dependent, useful score', DEPENDENT, 600, args.seeds, alpha, delta, False),
           selective_row('strongly dependent, useful score', HEAVY_GROUPS, 600, args.seeds, alpha, delta, False),
           selective_row('stress: whole groups err, no threshold can meet it', STRESS, 60, args.seeds, alpha, delta, True)])

    print('Silent error (split conformal on one score per group). Mean coverage should lie between nominal and nominal + 1/(n+1).\n')
    table(['groups', 'target', 'mean coverage', 'nominal', 'upper limit', 'Beta 95% interval', 'realised inside it'],
          [silent_row(n, a, args.seeds) for n, a in ((99, 0.1), (299, 0.1), (199, 0.05))])

    print('The Beta interval stated on clip counts, at clip level, with dependent clips.\n')
    table(['groups', 'clips', 'interval claimed with n = clips', 'realised central 95%', 'realised inside the claim (should be 0.95)'],
          [clip_beta_row(n, alpha, args.seeds) for n in (50, 100, 300)])

    print('Effective n below which the output says "guarantee is loose".\n')
    table(['target', 'loose below'], [[a, loose_below(a)] for a in (0.2, 0.1, 0.05, 0.02, 0.01)])
    print(f'(Normal approximation: {", ".join(f"{a}: {math.ceil(15.4 * (1 - a) / a)}" for a in (0.2, 0.1, 0.05, 0.02, 0.01))}.)')


if __name__ == '__main__':
    main()

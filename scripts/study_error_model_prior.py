"""Reproduce the choice of evaluators.error_model.PRIOR_STRENGTH.

The isotonic error model shrinks each block toward the base rate with
`prior_strength` pseudo-samples. This script scores several values on
synthetic data that is NOT the unit-test fixture:

* 5 classes; max probability m ~ U(0.4, 1) in column 0; the other four
  shares are (1 - m) * Dirichlet(1), capped at m (any excess moves to the
  smallest share), so m is always the arg-max;
* three true error curves: 1 - m**1.8, 1 - m (a calibrated model), and a
  step 0.02 + 0.5 * [m < 0.6];
* the score is the mean |stated error - true P(wrong)| on held-out draws,
  i.e. distance to the truth with no test-set sampling noise. Lower is
  better.

CPU only (numpy; never imports torch), seeded, deterministic. The default is
small (seconds). The A4 report's table is exactly
`--seeds 30 --n_test 10000 --fit_sizes 50 200 1000 4000`.

    python scripts/study_error_model_prior.py
    python scripts/study_error_model_prior.py --seeds 30 --n_test 10000 --fit_sizes 50 200 1000 4000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluators.error_model import ErrorModel, prediction_features  # noqa: E402

NUM_CLASSES = 5
CURVES = {
    '1 - m^1.8': lambda m: 1 - m ** 1.8,
    '1 - m': lambda m: 1 - m,
    'step 0.02 + 0.5*[m<0.6]': lambda m: 0.02 + 0.5 * (m < 0.6),
}


def synthetic_probs(n: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.uniform(0.4, 1.0, n)
    rest = rng.dirichlet(np.ones(NUM_CLASSES - 1), n) * (1 - m)[:, None]
    for _ in range(NUM_CLASSES - 1):
        excess = np.clip(rest - m[:, None], 0, None)
        rest = rest - excess
        rest[np.arange(n), rest.argmin(axis=1)] += excess.sum(axis=1)
    probs = np.column_stack([m, rest])
    assert np.allclose(probs.max(axis=1), m) and np.allclose(probs.sum(axis=1), 1)
    return probs


def distance_to_truth(curve, prior: float, n_fit: int, n_test: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    fit_probs = synthetic_probs(n_fit, rng)
    test_probs = synthetic_probs(n_test, rng)
    fit_wrong = (rng.uniform(size=n_fit) < curve(fit_probs[:, 0])).astype(float)
    stated = ErrorModel('cls', prior_strength=prior).fit(
        prediction_features(fit_probs, 'cls'), fit_wrong).predict(prediction_features(test_probs, 'cls'))
    return float(np.mean(np.abs(stated - curve(test_probs[:, 0]))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--priors', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument('--fit_sizes', type=int, nargs='+', default=[50, 200, 1000])
    p.add_argument('--seeds', type=int, default=10, help='seeds per (curve, fit size)')
    p.add_argument('--n_test', type=int, default=5000)
    p.add_argument('--first_seed', type=int, default=100)
    args = p.parse_args(argv)

    header = '| curve, n_fit | ' + ' | '.join(f'prior {v:g}' for v in args.priors) + ' |'
    print(header)
    print('|' + '---|' * (len(args.priors) + 1))
    wins = {v: 0 for v in args.priors}
    for name, curve in CURVES.items():
        for n_fit in args.fit_sizes:
            scores = [np.mean([distance_to_truth(curve, prior, n_fit, args.n_test, args.first_seed + s)
                               for s in range(args.seeds)]) for prior in args.priors]
            best = int(np.argmin(scores))
            wins[args.priors[best]] += 1
            cells = [f'**{v:.4f}**' if i == best else f'{v:.4f}' for i, v in enumerate(scores)]
            print(f'| {name}, {n_fit} | ' + ' | '.join(cells) + ' |', flush=True)
    print('best-in-row counts: ' + ', '.join(f'prior {k:g}: {v}' for k, v in wins.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())

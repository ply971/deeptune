"""The time estimate printed before a search trains anything. It is rough, and says so.

Basis: a timed pilot. The cheapest candidate runs through deeptune.py exactly like a real candidate, but for 1 epoch on
2 folds. Its wall time W (process start, data loading, 2 fold trainings of one epoch, their evaluations) gives
s = W / 2 seconds per fold-epoch, overhead included. A candidate c is then estimated at

    s x cost(c) / cost(pilot) x [(k-1)/k] / [1/2] x k x epochs

cost() is the preflight's relative cost of one training step (checks.Footprint.cost: forward FLOPs, plus twice that in
proportion to the share of parameters that train; parameters x frames when FLOPs could not be counted; 1 when the
preflight was off). The [(k-1)/k] / [1/2] factor scales the rows a fold trains on. The final refit trains once on all
development rows, estimated as the dearest candidate's fold training scaled by k/(k-1).

It leans high: per-process overhead is scaled like compute, and early stopping can only shorten the epochs. It leans low
when a model's speed is limited by memory traffic rather than FLOPs, and FLOP counts miss some custom kernels.
"""
from __future__ import annotations

from typing import Optional

LARGE_RUN_TRAININGS = 50  # above this many trainings, a search with no time estimate needs --yes


def candidate_seconds(pilot_seconds: float, cost: Optional[float], pilot_cost: Optional[float], k: int, epochs: int) -> float:
    ratio = cost / pilot_cost if cost and pilot_cost else 1.0
    per_fold_epoch = pilot_seconds / 2
    return per_fold_epoch * ratio * ((k - 1) / k) / 0.5 * k * epochs


def search_seconds(pilot_seconds: float, costs: list, pilot_cost: Optional[float], k: int, epochs: int, final_needed: bool) -> float:
    per_candidate = [candidate_seconds(pilot_seconds, cost, pilot_cost, k, epochs) for cost in costs]
    total = sum(per_candidate)
    if final_needed and per_candidate:
        total += max(per_candidate) / k * (k / (k - 1))  # one fold training of the dearest, on all rows
    return total


def describe_hours(seconds: Optional[float]) -> str:
    if seconds is None:
        return 'unknown'
    hours = seconds / 3600
    return f'about {hours:.1f} h' if hours >= 1 else f'about {seconds / 60:.0f} min'

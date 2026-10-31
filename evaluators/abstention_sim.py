"""Synthetic scores for checking abstention's guarantees: no model, no video. Torch-free. Used by tests/test_abstention.py and scripts/study_abstention_validity.py.

A group draws a hidden difficulty u ~ N(0, group_sd^2) and a shared score level. Each clip draws a score around that level and is wrong with probability
    expit(logit(1 - score) + offset + u), or expit(offset + u) when the score is not informative.
The score does not see u, so with group_sd > 0 the clips of a group err together and clip-level counting overstates the evidence. group_sd = 0 with one clip per group is the
i.i.d. case. Groups are i.i.d., so the assumption of the group-level guarantees holds by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    clips_per_group: int = 8
    group_sd: float = 2.0
    offset: float = 0.0
    score_group_sd: float = 0.5
    score_clip_sd: float = 1.0
    score_mean: float = 2.0  # on the logit scale
    informative: bool = True  # False: error ignores the score, so no threshold helps (the hardest case for a false claim)


# In the first two, answering everything misses a 0.10 target (about 0.16), so the threshold has work to do.
INDEPENDENT = Scenario(clips_per_group=1, group_sd=0.0, score_group_sd=0.0, score_mean=2.0)
DEPENDENT = Scenario(group_sd=2.0, score_mean=3.0, score_clip_sd=1.5)  # a useful score, with clips of a group erring together
# About half the clips are wrong, so silent-error thresholds sit in the continuous part of the scores rather than on the tie at "no wrong clip".
MOSTLY_WRONG = Scenario(group_sd=3.0, offset=1.0, score_mean=1.0)
# Whole groups are right or wrong, the true error (about 0.117) is just above a 0.10 target at every threshold, and the score does not help: a claim of 0.10 is always false here.
STRESS = Scenario(clips_per_group=16, group_sd=8.0, offset=-9.8, informative=False)


def draw(rng: np.random.Generator, n_groups: int, scenario: Scenario = DEPENDENT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(score, wrong, group index), clip by clip."""
    size = scenario.clips_per_group
    hidden = rng.normal(0.0, scenario.group_sd, n_groups) if scenario.group_sd else np.zeros(n_groups)
    level = rng.normal(scenario.score_mean, scenario.score_group_sd, n_groups) if scenario.score_group_sd else np.full(n_groups, scenario.score_mean)
    groups = np.repeat(np.arange(n_groups), size)
    logit_score = level[groups] + rng.normal(0.0, scenario.score_clip_sd, n_groups * size)
    score = 1.0 / (1.0 + np.exp(-logit_score))
    ranking = logit_score if scenario.informative else 0.0
    p_wrong = 1.0 / (1.0 + np.exp(ranking - scenario.offset - hidden[groups]))
    return score, rng.random(n_groups * size) < p_wrong, groups


def true_risks(score: np.ndarray, wrong: np.ndarray, groups: np.ndarray, grid: np.ndarray) -> dict:
    """Per threshold on a very large draw, standing in for the population: group-weighted selective risk, clip-weighted selective risk, the share of groups with an answered wrong
    clip, and the share of clips that are answered and wrong."""
    from evaluators.abstention import threshold_stats
    table = threshold_stats(score, wrong, groups, grid)
    answered = table['answered_units'].to_numpy(dtype=np.float64)
    clips = table['answered_clips'].to_numpy(dtype=np.float64)
    n_groups = int(groups.max()) + 1
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'group': table['unit_loss_sum'].to_numpy() / answered, 'clip': table['wrong_clips'].to_numpy() / clips,
                'silent_group': 1.0 - table['units_without_silent_error'].to_numpy() / n_groups, 'silent_clip': table['wrong_clips'].to_numpy() / len(score)}

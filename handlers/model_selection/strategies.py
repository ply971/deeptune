"""Search strategies of select_model.py: which candidates to cross-validate, under which protocol, in which order.

A strategy never runs anything and never sees a file. It yields batches of jobs (jobs.py) made by the context's `job()`,
and reads finished results back through `ctx.entry()` / `ctx.score()`: development-row cross-validation only. The engine
(search.py) skips jobs the ledger already holds, so a strategy that makes the same decisions from the same results resumes
by replaying itself. All of them are deterministic for a given seed and ledger.

exhaustive  every runnable candidate of the grid, in grid order (R3a).
smart       the staged search (smart.py), the default.
random      a seeded random order of the grid, without replacement; budget --n_trials and/or --max_hours.
bayes       Optuna's TPE sampler over the grid's axes as categorical choices, one trial at a time; same budgets.
            Optuna is optional and imported only here, inside the strategy.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

from handlers.model_selection.grid import AXES, Candidate

INSTALL_OPTUNA = ('--search bayes needs Optuna, which DeepTune does not install by default. Install it with '
                  '`pip install optuna` (or `pip install "DeepTune[search]"`), or use --search smart, random or exhaustive.')


def backbone_family(name: str) -> str:
    """The family a backbone belongs to, for smart's first screen: one small model per family."""
    from handlers.video_backbones import hub_backbone
    info = hub_backbone(name)
    if info is not None:
        return info.family
    if name.startswith('swin3d'):
        return 'swin3d'
    if name in ('r3d_18', 'mc3_18', 'r2plus1d_18'):
        return '3d_cnn'
    if name.startswith('mvit'):
        return 'mvit'
    for family in ('resnet', 'densenet', 'efficientnet', 'vgg', 'convnext', 'swin', 'vit'):  # feature_backbones._FRAME_FAMILIES
        if name.startswith(family):
            return family
    match = re.match(r'[a-zA-Z]+', name)
    return match.group(0).lower() if match else name


@dataclass
class Plan:
    """For the run count and the time estimate: (checked item, epochs or None) per planned job. Adaptive strategies give an
    upper bound with representative items, and say how it is counted in `note`."""
    jobs: list = field(default_factory=list)
    note: str = ''


class Strategy:
    name = ''
    adaptive = True  # False: the plan is exact, and --max_hours is a start gate rather than a running budget

    def __init__(self, options=None):
        self.o = options
        self.strategy_options = dict(getattr(options, 'strategy_options', None) or {})

    def option(self, name, default=None):
        value = self.strategy_options.get(name)
        return default if value is None else value

    def settings(self) -> dict:
        return {}

    def check(self, ctx):
        """Why this strategy refuses to start on these candidates (before anything trains), or None."""
        return None

    def plan(self, ctx) -> Plan:
        raise NotImplementedError

    def jobs(self, ctx) -> Iterator[list]:
        raise NotImplementedError


class ExhaustiveStrategy(Strategy):
    """Every candidate of the grid, in grid order."""
    name = 'exhaustive'
    adaptive = False

    def plan(self, ctx) -> Plan:
        return Plan([(item, None) for item in ctx.runnable], 'every runnable candidate')

    def jobs(self, ctx) -> Iterator[list]:
        yield [ctx.job(item, stage='exhaustive', why='every candidate of the grid') for item in ctx.runnable]


def _seed(*parts) -> int:
    """A stable 32-bit seed from parts (Python's hash() of a string changes between processes)."""
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


class RandomStrategy(Strategy):
    """A seeded permutation of the whole grid (in grid order, refused and skipped candidates included, so the order does not
    shift when a check changes), keeping the runnable ones, until --n_trials trainings have finished or the budget ends.
    Resume replays the same permutation: finished candidates come back from the ledger and the sequence goes on."""
    name = 'random'

    def settings(self) -> dict:
        return {'n_trials': self.option('n_trials'), 'seed': self.o.settings.seed if self.o else None,
                'order': 'a seeded permutation of the grid, without replacement'}

    def order(self, ctx) -> list:
        import numpy as np
        rng = np.random.default_rng(_seed('random', ctx.settings.seed))
        return [ctx.checked[i] for i in rng.permutation(len(ctx.checked)) if ctx.checked[i].status == 'runnable']

    def plan(self, ctx) -> Plan:
        order = self.order(ctx)
        n = self.option('n_trials')
        chosen = order[:n] if n else order
        return Plan([(item, None) for item in chosen], f'{len(chosen)} random draws' + ('' if n else ' (no --n_trials: until --max_hours or the grid ends)'))

    def jobs(self, ctx) -> Iterator[list]:
        n, trained = self.option('n_trials'), 0
        for draw, item in enumerate(self.order(ctx), start=1):
            if n and trained >= n:
                ctx.decide('random', f'stopped after --n_trials {n} trainings.')
                return
            job = ctx.job(item, stage='random', why=f'random draw {draw} (seed {ctx.settings.seed})')
            yield [job]
            if ctx.entry(job) is not None:
                trained += 1
        ctx.decide('random', f'every runnable candidate of the grid was drawn ({trained} trainings).')


class BayesStrategy(Strategy):
    """Optuna's TPE sampler over the grid's axes (categorical choices), one trial at a time.

    Optuna only ever sees development cross-validation means of the ranking protocol. Its study is not stored: each step
    builds a fresh study from the trials this strategy has run so far (their scores come from the ledger) and a sampler
    seeded from (--seed, step, attempt). So an interrupted search, rerun, replays the same proposals and goes on exactly as
    an uninterrupted one would, and the ledger stays the only record. A proposal that is not runnable (refused, skipped,
    excluded) or already tried is asked again with the next attempt seed; after `MAX_ATTEMPTS` the step takes the next
    untried candidate of a seeded permutation, and records that it did."""
    name = 'bayes'
    MAX_ATTEMPTS = 25

    @staticmethod
    def require_optuna():
        try:
            import optuna  # noqa: F401
        except ImportError:
            raise ImportError(INSTALL_OPTUNA) from None
        return optuna

    def startup(self) -> int:
        n = self.option('n_trials')
        default = max(3, n // 4) if n else 5
        return int(self.option('bayes_startup', default))

    def settings(self) -> dict:
        return {'n_trials': self.option('n_trials'), 'seed': self.o.settings.seed if self.o else None,
                'sampler': 'optuna TPESampler(multivariate=True), rebuilt from the ledger at every step',
                'n_startup_trials': self.startup()}

    def plan(self, ctx) -> Plan:
        n = self.option('n_trials')
        chosen = ctx.runnable[:n] if n else ctx.runnable
        return Plan([(item, None) for item in chosen], f'{len(chosen)} trials' + ('' if n else ' (no --n_trials: until --max_hours or the grid ends)'))

    def _space(self, ctx) -> dict:
        """Axis field -> its values as text, for the axes with more than one value."""
        return {AXES[name]: [str(v) for v in getattr(ctx.grid, name)] for name in AXES if len(getattr(ctx.grid, name)) > 1}

    def _candidate(self, ctx, params: dict) -> Candidate:
        base = {AXES[name]: getattr(ctx.grid, name)[0] for name in AXES}
        for name in AXES:
            axis = AXES[name]
            if axis in params:
                base[axis] = next(v for v in getattr(ctx.grid, name) if str(v) == params[axis])
        return Candidate(**base).replace()

    def _params(self, ctx, candidate: Candidate, space: dict) -> dict:
        return {axis: str(getattr(candidate, axis)) for axis in space}

    def jobs(self, ctx) -> Iterator[list]:
        optuna = self.require_optuna()
        import warnings
        from optuna.distributions import CategoricalDistribution
        from optuna.samplers import TPESampler
        from optuna.trial import TrialState, create_trial
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        space = self._space(ctx)
        distributions = {axis: CategoricalDistribution(values) for axis, values in space.items()}
        direction = 'maximize' if ctx.direction > 0 else 'minimize'
        n, seed = self.option('n_trials'), ctx.settings.seed
        history, tried = [], set()  # [(params, value or None)], candidate keys
        fallback = RandomStrategy(self.o).order(ctx)
        runnable_keys = {item.key for item in ctx.runnable}
        step = 0
        while len(tried) < len(runnable_keys) and not (n and step >= n):
            item, how = None, ''
            if distributions:
                for attempt in range(self.MAX_ATTEMPTS):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')  # experimental-feature warnings of multivariate TPE
                        study = optuna.create_study(direction=direction, sampler=TPESampler(
                            seed=_seed('bayes', seed, step, attempt), multivariate=True, n_startup_trials=self.startup()))
                        for params, value in history:
                            study.add_trial(create_trial(params=params, distributions=distributions, value=value,
                                                         state=TrialState.COMPLETE if value is not None else TrialState.FAIL))
                        proposal = study.ask(distributions).params
                    found = ctx.item(self._candidate(ctx, proposal))
                    if found is not None and found.status == 'runnable' and found.key not in tried:
                        item, how = found, f'TPE proposal (step {step + 1}, attempt {attempt + 1})'
                        break
            if item is None:
                item = next(candidate for candidate in fallback if candidate.key not in tried)
                how = (f'step {step + 1}: TPE proposed only tried or unrunnable candidates in {self.MAX_ATTEMPTS} attempts, so the next '
                       'untried candidate of a seeded permutation was taken' if distributions else
                       f'step {step + 1}: the grid has one value per axis')
            tried.add(item.key)
            job = ctx.job(item, stage='bayes', why=how)
            yield [job]
            entry = ctx.entry(job)
            if entry is None:  # stopped by the budget: the engine closes this generator
                return
            history.append((self._params(ctx, item.candidate, space), ctx.score(job)))
            step += 1
        ctx.decide('bayes', f'stopped after {step} trials' + (f' (--n_trials {n})' if n and step >= n else ' (every runnable candidate was tried)')
                   + '.')


# smart.py imports this module, so search.py adds SmartStrategy to these.
BASE_STRATEGIES = {'exhaustive': ExhaustiveStrategy, 'random': RandomStrategy, 'bayes': BayesStrategy}

"""`--search smart`, the default: the user's staged search (PLAN.md 4b, R3).

Stage 0 (the data profile, profile.py) runs in the engine for every strategy. Then:

1  Screen, cheaply, on the development folds.
   1a  One small model per backbone family (the fewest parameters by the preflight), at the base settings. The best
       --top_families families go on. Skipped when there are no more families than that.
   1b  Every backbone of those families, at the base settings. The best --top_backbones go on. Skipped likewise.
   Protocol: `probe` (cached frozen features + an attentive probe, probe.py) when every screened candidate is a frozen
   backbone the feature extractor takes and --screen is auto or probe; otherwise `screen` (deeptune.py, --screen_epochs
   epochs). Both protocols run on the same folds as the ranking, but their scores are never ranked against cv scores.
2  Tune one group at a time on each surviving backbone, from the base settings: the temporal head (video_arch with
   pooling), what trains (train mode), the learning rate, the frames, then augmentation. Each group tries every value of
   its axes with the others at the best so far, and keeps the best. Protocol: the ranking protocol (cv: --epochs, with
   early stopping), or `tune` when --tune_epochs differs from --epochs.
3  Re-check combinations, since one group at a time misses interactions: on the --recheck_models best tuned models, every
   pair of tuned groups is set to their runner-up values together. If one of these beats the model's tuned best, it is
   adopted and one more one-group-at-a-time pass runs from it.
4  Final: the --final_top best by grouped cross-validation under the ranking protocol, on the same folds. When tuning ran
   under the ranking protocol these runs already exist (the same folds and seed would give the same numbers), so nothing
   trains again; with --tune_epochs they are cross-validated now. The engine then ranks, and the winner alone goes to the
   sealed test set, exactly as in exhaustive mode.

Base settings: the first value of every axis, except that `frozen` and augmentation `off` are preferred when they are in
the grid, and --fix values win. Every choice is recorded with its reason (ctx.decide: printed, stages.json,
selection.json). Every stage selects on the development folds the final ranking also uses, so its choices are
optimistically biased like the winner's score; the sealed test is the honest number.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Iterator, Optional

from handlers.model_selection.grid import AXES, COLLAPSE_REASON, Candidate, augmentation_applies
from handlers.model_selection.jobs import Protocol
from handlers.model_selection.strategies import Plan, Strategy, backbone_family

GROUPS = (('temporal head', ('video_arch', 'pooling')), ('what trains', ('train_mode',)), ('learning rate', ('learning_rate',)),
          ('frames', ('num_frames',)), ('augmentation', ('augmentation',)))
FIELDS = tuple(AXES.values())
GRID_AXIS = {field: axis for axis, field in AXES.items()}
DEFAULTS = {'screen': 'auto', 'screen_epochs': 2, 'top_families': 2, 'top_backbones': 3, 'tune_epochs': None,
            'recheck_models': 2, 'final_top': 3}


def _show(candidate: Candidate, fields) -> str:
    return ', '.join(f'{field}={getattr(candidate, field)}' for field in fields)


class SmartStrategy(Strategy):
    name = 'smart'

    def knob(self, name):
        return self.option(name, DEFAULTS[name])

    def settings(self) -> dict:
        return {name: self.knob(name) for name in DEFAULTS} | {'groups': [name for name, _ in GROUPS]}

    # -- protocols -------------------------------------------------------------------------------------------------------
    def _tune_protocol(self, ctx) -> Protocol:
        epochs = self.knob('tune_epochs')
        return ctx.ranking if epochs in (None, ctx.settings.epochs) else Protocol('tune', epochs=int(epochs))

    def _screen_protocol(self, ctx, items: list) -> Protocol:
        cv = Protocol('screen', epochs=int(self.knob('screen_epochs')))
        choice = self.knob('screen')
        if choice == 'cv':
            return cv
        problems = []
        if not ctx.probe_available:
            problems.append('no feature-probe runner is available here')
        else:
            for item in items:
                problem = ctx.probe_problem(item.candidate)
                if problem:
                    problems.append(f'{item.candidate.backbone}: {problem}')
        if problems:
            ctx.decide('screen protocol', f'short cross-validation runs ({cv.epochs} epochs) instead of cached-feature probes, because '
                       + '; '.join(problems) + '.')
            return cv
        ctx.decide('screen protocol', 'cached frozen features + attentive probe: every screened candidate is a frozen backbone the '
                   'feature extractor takes. Probe scores rank backbones within the screen only.')
        return Protocol('probe', method='feature_probe', details=(('probe', 'attentive_probe'),))

    # -- candidates ------------------------------------------------------------------------------------------------------
    def _base(self, ctx, backbone: str) -> Candidate:
        values = {}
        for field in FIELDS[1:]:
            options = list(getattr(ctx.grid, GRID_AXIS[field]))
            fixed = ctx.fix.get(field)
            chosen = options[0]
            if fixed:
                chosen = next((v for v in options if v in fixed), chosen)
            elif field == 'train_mode' and 'frozen' in options:
                chosen = 'frozen'
            elif field == 'augmentation' and 'off' in options:
                chosen = 'off'
            values[field] = chosen
        return Candidate(backbone=backbone, **values).replace()

    def _nearest(self, ctx, candidate: Candidate):
        """The candidate itself if runnable, else the runnable one of its backbone that differs on the fewest axes."""
        item = ctx.item(candidate)
        if item is not None and item.status == 'runnable':
            return item
        options = [(sum(getattr(other.candidate, f) != getattr(candidate, f) for f in FIELDS), index, other)
                   for index, other in enumerate(ctx.runnable) if other.candidate.backbone == candidate.backbone]
        return min(options, key=lambda option: option[:2])[2] if options else None

    def _ranking_text(self, ctx, ranked: list, jobs: list) -> str:
        done = {id(job) for job, _ in ranked}
        parts = [ctx.describe(job) for job, _ in ranked] + [ctx.describe(job) for job in jobs if id(job) not in done]
        return '; '.join(parts)

    @staticmethod
    def _rows(ctx, jobs: list) -> list:
        return [{'candidate': job.candidate.label(), 'score': ctx.score(job), 'status': (ctx.entry(job) or {}).get('status', 'not run')}
                for job in jobs]

    # -- plan ------------------------------------------------------------------------------------------------------------
    def plan(self, ctx) -> Plan:
        backbones = self._backbones(ctx, record=False)
        if not backbones:
            return Plan([], 'nothing is runnable')
        items = {b: self._nearest(ctx, self._base(ctx, b)) for b in backbones}
        families = {}
        for b in backbones:
            families.setdefault(backbone_family(b), []).append(b)
        top_f, top_b = int(self.knob('top_families')), int(self.knob('top_backbones'))
        screen_epochs, tune = int(self.knob('screen_epochs')), self._tune_protocol(ctx)
        jobs, parts = [], []
        if len(families) > top_f:
            jobs += [(items[bs[0]], screen_epochs) for bs in families.values()]
            parts.append(f'{len(families)} family screens')
        kept = sorted((len(bs) for bs in families.values()), reverse=True)[:top_f]
        if sum(kept) > top_b:
            jobs += [(items[b], screen_epochs) for b in backbones[:sum(kept)]]
            parts.append(f'{sum(kept)} backbone screens')
        survivors = backbones[:min(top_b, len(backbones))]
        sizes = [self._group_size(ctx, fields) for _, fields in GROUPS]
        sizes = [size for size in sizes if size > 1]
        per_model = 1 + sum(sizes)
        jobs += [(items[b], tune.epochs) for b in survivors for _ in range(per_model)]
        parts.append(f'{len(survivors)} tuned models x (1 + {" + ".join(map(str, sizes)) or "0"}) group values')
        rechecks = min(int(self.knob('recheck_models')), len(survivors))
        recheck_jobs = rechecks * (comb(len(sizes), 2) + sum(sizes))
        jobs += [(items[survivors[0]], tune.epochs)] * recheck_jobs
        parts.append(f'{rechecks} re-checked x ({comb(len(sizes), 2)} pairs + {sum(sizes)} for one more pass)')
        if not tune.ranking:
            jobs += [(items[survivors[0]], None)] * int(self.knob('final_top'))
            parts.append(f'{self.knob("final_top")} final cross-validations')
        return Plan(jobs, ' + '.join(parts))

    def _group_size(self, ctx, fields) -> int:
        size = 1
        for field in fields:
            if field not in ctx.fix:
                size *= len(getattr(ctx.grid, GRID_AXIS[field]))
        return size

    def _backbones(self, ctx, record: bool = True) -> list:
        """Grid backbones that have a runnable candidate, in grid order."""
        runnable = {item.candidate.backbone for item in ctx.runnable}
        kept = [b for b in ctx.grid.backbones if b in runnable]
        if record and len(kept) < len(ctx.grid.backbones):
            ctx.decide('screen', 'left out, as none of their candidates is runnable (refused, skipped, over a budget or excluded): '
                       + ', '.join(b for b in ctx.grid.backbones if b not in runnable) + '.')
        return kept

    # -- the stages ------------------------------------------------------------------------------------------------------
    def jobs(self, ctx) -> Iterator[list]:
        backbones = self._backbones(ctx)
        if not backbones:
            ctx.decide('screen', 'nothing to search: no candidate is runnable.')
            return
        other = sorted(set(ctx.include) - {'backbone', 'family'})
        if other:
            ctx.decide('overrides', f'--include {", ".join(other)}: every value of a tuned axis is tried in stage 2 anyway; '
                       '--fix pins an axis instead.')
        tune = self._tune_protocol(ctx)
        self._ranking_jobs = []
        survivors = yield from self._screen(ctx, backbones)
        tuned = []
        for backbone in survivors:
            result = yield from self._tune(ctx, backbone, tune)
            if result is not None:
                tuned.append(result)
        yield from self._recheck(ctx, tuned, tune)
        yield from self._final(ctx, tuned, tune)

    def _screen(self, ctx, backbones: list):
        top_f, top_b = int(self.knob('top_families')), int(self.knob('top_backbones'))
        include_b, include_f = ctx.include.get('backbone', set()), ctx.include.get('family', set())
        items = {b: self._nearest(ctx, self._base(ctx, b)) for b in backbones}
        families = {}
        for b in backbones:
            families.setdefault(backbone_family(b), []).append(b)
        needs_1a, needs_1b = len(families) > top_f, None
        protocol = None
        if needs_1a or len(backbones) > top_b:
            protocol = self._screen_protocol(ctx, list(items.values()))

        # 1a: one small model per family
        if needs_1a:
            def size(b):
                return items[b].footprint.get('total_params') or float('inf'), backbones.index(b)
            reps = {family: min(members, key=size) for family, members in families.items()}
            jobs = {family: ctx.job(items[b], protocol, 'screen families', why=f'the smallest {family} model, to screen its family')
                    for family, b in reps.items()}
            yield list(jobs.values())
            ranked = ctx.rank(list(jobs.values()))
            order = [job.candidate.backbone for job, _ in ranked]
            kept = [backbone_family(b) for b in order[:top_f]]
            forced = [f for f in families if f not in kept and (f in include_f or any(b in include_b for b in families[f]))]
            dropped = [f for f in families if f not in kept + forced]
            ctx.decide('screen families', f'kept {", ".join(kept) or "none"}' + (f', plus {", ".join(forced)} (--include)' if forced else '')
                       + f'; dropped {", ".join(dropped) or "none"}. By {protocol.describe(ctx.settings.epochs)} on the development '
                       f'folds: {self._ranking_text(ctx, ranked, list(jobs.values()))}.', ranking=self._rows(ctx, list(jobs.values())))
            families = {f: families[f] for f in kept + forced}
        else:
            ctx.decide('screen families', f'{len(families)} famil{"y" if len(families) == 1 else "ies"} ({", ".join(families)}), not more '
                       f'than --top_families {top_f}: all kept, nothing to screen.')

        # 1b: every backbone of the kept families
        members = [b for family in families for b in families[family]]
        if len(members) > top_b:
            jobs = [ctx.job(items[b], protocol, 'screen backbones', why=f'a backbone of the {backbone_family(b)} family') for b in members]
            yield jobs
            ranked = ctx.rank(jobs)
            kept = [job.candidate.backbone for job, _ in ranked[:top_b]]
            forced = [b for b in members if b not in kept and b in include_b]
            ctx.decide('screen backbones', f'kept {", ".join(kept) or "none"}' + (f', plus {", ".join(forced)} (--include)' if forced else '')
                       + f' (--top_backbones {top_b}). By {protocol.describe(ctx.settings.epochs)}: {self._ranking_text(ctx, ranked, jobs)}.',
                       ranking=self._rows(ctx, jobs))
            return kept + forced
        ctx.decide('screen backbones', f'{len(members)} backbone{"s" if len(members) != 1 else ""} ({", ".join(members)}), not more than '
                   f'--top_backbones {top_b}: all go to tuning.')
        return members

    def _variants(self, ctx, current: Candidate, fields) -> list:
        axes = [[getattr(current, f)] if f in ctx.fix else list(getattr(ctx.grid, GRID_AXIS[f])) for f in fields]
        seen, out = set(), []
        for combination in itertools.product(*axes):
            candidate = current.replace(**dict(zip(fields, combination)))
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out

    def _sweep(self, ctx, current: Candidate, group: str, fields, protocol: Protocol, backbone: str, stage: str):
        """Try every value of one group with the rest at `current`. Returns (best candidate, score, [(candidate, score)])."""
        jobs, missing = [], []
        for candidate in self._variants(ctx, current, fields):
            item = ctx.item(candidate)
            if item is None or item.status != 'runnable':
                missing.append(f'{_show(candidate, fields)} ({item.status + ": " + item.reason if item else "not in the grid"})')
                continue
            jobs.append(ctx.job(item, protocol, stage, why=f'{group} for {backbone}, the other settings at the best so far'))
        if len(jobs) <= 1:
            why = ''
            if 'augmentation' in fields and not augmentation_applies(current.train_mode) and len(ctx.grid.augmentations) > 1:
                why = f' ({COLLAPSE_REASON})'
            elif all(f in ctx.fix for f in fields):
                why = ' (fixed by --fix)'
            elif all(len(getattr(ctx.grid, GRID_AXIS[f])) == 1 for f in fields):
                return current, None, None  # nothing to say: the grid has one value
            ctx.decide(f'{stage} ({backbone})', f'kept {_show(current, fields)}: only one runnable value{why}'
                       + (f'; not runnable: {"; ".join(missing)}' if missing else '') + '.')
            if jobs:
                yield jobs
            return current, None, None
        yield jobs
        if protocol.ranking:
            self._ranking_jobs += jobs
        ranked = ctx.rank(jobs)
        if not ranked:
            ctx.decide(f'{stage} ({backbone})', f'no value of {group} finished; kept {_show(current, fields)}.', ranking=self._rows(ctx, jobs))
            return current, None, []
        best_job, best = ranked[0]
        ctx.decide(f'{stage} ({backbone})', f'chose {_show(best_job.candidate, fields)}: {self._ranking_text(ctx, ranked, jobs)}'
                   + (f'; not runnable: {"; ".join(missing)}' if missing else '') + '.', ranking=self._rows(ctx, jobs))
        return best_job.candidate, best, [(job.candidate, score) for job, score in ranked]

    def _tune(self, ctx, backbone: str, protocol: Protocol):
        start = self._nearest(ctx, self._base(ctx, backbone))
        if start is None:
            return None
        first = ctx.job(start, protocol, 'tune', why=f'the starting point for {backbone}')
        yield [first]
        if protocol.ranking:
            self._ranking_jobs.append(first)
        current, score, groups = start.candidate, ctx.score(first), {}
        for group, fields in GROUPS:
            chosen, best, ranked = yield from self._sweep(ctx, current, group, fields, protocol, backbone, f'tune {group}')
            if best is not None:
                current, score = chosen, best
            if ranked and len(ranked) > 1:
                groups[group] = ranked
        return {'backbone': backbone, 'best': current, 'score': score, 'groups': groups}

    def _recheck(self, ctx, tuned: list, protocol: Protocol):
        direction = ctx.direction
        order = sorted((t for t in tuned if t['score'] is not None), key=lambda t: -t['score'] * direction)
        for model in order[:int(self.knob('recheck_models'))]:
            best_item = ctx.item(model['best'])
            best_job = ctx.job(best_item, protocol, 'recheck', why='the tuned best')
            groups = list(model['groups'].items())
            jobs = []
            for (group_a, ranked_a), (group_b, ranked_b) in itertools.combinations(groups, 2):
                fields_a, fields_b = dict(GROUPS)[group_a], dict(GROUPS)[group_b]
                runner_a, runner_b = ranked_a[1][0], ranked_b[1][0]
                values = {**{f: getattr(runner_a, f) for f in fields_a}, **{f: getattr(runner_b, f) for f in fields_b}}
                candidate = model['best'].replace(**values)
                item = ctx.item(candidate)
                if item is None or item.status != 'runnable' or candidate == model['best']:
                    continue
                jobs.append(ctx.job(item, protocol, 'recheck pairs', why=f'{group_a} runner-up ({_show(runner_a, fields_a)}) together with '
                                                                        f'{group_b} runner-up ({_show(runner_b, fields_b)})'))
            if not jobs:
                ctx.decide(f'recheck ({model["backbone"]})', 'no pair of tuned groups with a runnable runner-up combination; nothing to re-check.')
                continue
            yield jobs
            if protocol.ranking:
                self._ranking_jobs += jobs
            ranked = ctx.rank([best_job] + jobs)
            if not ranked or ranked[0][0] is best_job or ranked[0][1] == model['score']:
                ctx.decide(f'recheck ({model["backbone"]})', f'the one-group-at-a-time choice held against {len(jobs)} runner-up pairs: '
                           f'{self._ranking_text(ctx, ranked, [best_job] + jobs)}.', ranking=self._rows(ctx, [best_job] + jobs))
                continue
            current, score = ranked[0][0].candidate, ranked[0][1]
            ctx.decide(f'recheck ({model["backbone"]})', f'an interaction: {current.label()} beat the tuned best {model["best"].label()} '
                       f'({score:.4f} against {model["score"]:.4f}); one more one-group-at-a-time pass runs from it.',
                       ranking=self._rows(ctx, [best_job] + jobs))
            for group, fields in GROUPS:
                chosen, best, _ = yield from self._sweep(ctx, current, group, fields, protocol, model['backbone'], f'recheck {group}')
                if best is not None:
                    current, score = chosen, best
            model['best'], model['score'] = current, score

    def _final(self, ctx, tuned: list, protocol: Protocol):
        top = int(self.knob('final_top'))
        if protocol.ranking:
            unique = list({job.key: job for job in self._ranking_jobs}.values())
            ranked = ctx.rank(unique)[:top]
            ctx.decide('final', f'the top {len(ranked)} by grouped cross-validation on the same folds: '
                       + '; '.join(ctx.describe(job) for job, _ in ranked) + '. They ran under the ranking protocol while tuning, '
                       'so they are not trained again (the same folds and seed would give the same numbers). The best of them '
                       'is refitted and scored once on the sealed test set.', ranking=self._rows(ctx, [job for job, _ in ranked]))
            return
        direction = ctx.direction
        best = sorted((t for t in tuned if t['score'] is not None), key=lambda t: -t['score'] * direction)
        candidates = []
        for model in best:
            if model['best'] not in candidates:
                candidates.append(model['best'])
        jobs = [ctx.job(ctx.item(c), ctx.ranking, 'final', why=f'among the top {top} by tuning ({protocol.name} protocol); now under the '
                                                               'ranking protocol') for c in candidates[:top]]
        if jobs:
            yield jobs
        ctx.decide('final', f'cross-validated the top {len(jobs)} under the ranking protocol: '
                   + '; '.join(ctx.describe(job) for job in jobs) + '. The best of them is refitted and scored once on the sealed test set.',
                   ranking=self._rows(ctx, jobs))

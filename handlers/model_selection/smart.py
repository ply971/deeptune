"""`--search smart`, the default: the user's staged search (PLAN.md 4b, R3).

Stage 0 (the data profile, profile.py) runs in the engine for every strategy. Then:

1  Screen, cheaply, on the development folds, in up to two views:
     frozen view     the base settings with a frozen backbone (when `frozen` is in the grid). Protocol `probe` (cached
                     frozen features + an attentive probe, probe.py) when --screen is auto or probe and every screened
                     backbone can be probed; else `screen` (deeptune.py, --screen_epochs epochs).
     trainable view  the base settings with the first trainable train mode of the grid (lora:R, full or partial:N), when
                     there is one. Protocol `screen` (short deeptune.py runs; a probe cannot stand for weights that train).
   Frozen features can rank backbones in another order than fine-tuning does, and which of the two wins is only known
   after tuning, so each view keeps its own top N and the survivors are the union (up to twice N when the views disagree,
   N when they agree). --screen_views frozen or trainable keeps one view only.
   1a  One small model per backbone family (the fewest parameters by the preflight). Each view's best --top_families
       families go on. Skipped when there are no more families than that.
   1b  Every backbone of those families. Each view's best --top_backbones go on. Skipped likewise.
   The two views' scores are never ranked against each other, nor against cv scores.
2  Tune one group at a time on each surviving backbone, from the base settings: the temporal head (video_arch with
   pooling), what trains together with the learning rate, the frames, then augmentation. Train mode and learning rate are
   one group because the rate that suits a train mode depends on it (full fine-tuning usually wants a lower rate than
   LoRA or a head), the most common interaction in fine-tuning. Each group tries every combination of its axes with the
   rest at the best so far, and keeps the best. Protocol: the ranking protocol (cv: --epochs, with early stopping), or
   `tune` when --tune_epochs differs from --epochs.
3  Re-check pairs of groups on the --recheck_models best tuned models, since one group at a time misses interactions
   between groups. For a pair whose tried values multiply to at most --recheck_full_product (6), every combination of
   their values is tried with the rest at the tuned best; for a larger pair, only the two runner-up values together. If one
   of these beats the tuned best, it is adopted and one more one-group-at-a-time pass runs from it. It catches exactly the
   combinations it tries: an interaction among three groups, or between a large pair's other values, can still be missed.
4  Final: the --final_top best by grouped cross-validation under the ranking protocol, on the same folds. When tuning ran
   under the ranking protocol, those tuning runs are the ranking runs themselves, so nothing trains again; with
   --tune_epochs the --final_top best tune runs (all of them, across backbones) are cross-validated again. The engine then
   ranks every ranking-protocol entry of the --out folder (runs of another strategy in the same folder included) and the
   winner alone goes to the sealed test set, exactly as in exhaustive mode.

Base settings: the first value of every axis, except that `frozen` and augmentation `off` are preferred when they are in
the grid, and --fix values win. Every choice is recorded with its reason (ctx.decide: printed, stages.json,
selection.json). Every stage selects on the development folds the final ranking also uses, so its choices are
optimistically biased like the winner's score; the sealed test is the honest number.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Iterator, Optional

from handlers.model_selection.grid import AXES, COLLAPSE_REASON, Candidate, augmentation_applies, parse_train_mode
from handlers.model_selection.jobs import Protocol
from handlers.model_selection.strategies import Plan, Strategy, backbone_family

GROUPS = (('temporal head', ('video_arch', 'pooling')), ('what trains and learning rate', ('train_mode', 'learning_rate')),
          ('frames', ('num_frames',)), ('augmentation', ('augmentation',)))
FIELDS = tuple(AXES.values())
GRID_AXIS = {field: axis for axis, field in AXES.items()}
DEFAULTS = {'screen': 'auto', 'screen_views': 'auto', 'screen_epochs': 2, 'top_families': 2, 'top_backbones': 3,
            'tune_epochs': None, 'recheck_models': 2, 'recheck_full_product': 6, 'final_top': 3}
SMART_OPTIONS = tuple(DEFAULTS) + ('include',)


def _show(candidate: Candidate, fields) -> str:
    return ', '.join(f'{field}={getattr(candidate, field)}' for field in fields)


def _frozen(train_mode: str) -> bool:
    return parse_train_mode(train_mode)[0] == 'frozen'


class SmartStrategy(Strategy):
    name = 'smart'

    def knob(self, name):
        return self.option(name, DEFAULTS[name])

    def settings(self) -> dict:
        return {name: self.knob(name) for name in DEFAULTS} | {'groups': [name for name, _ in GROUPS]}

    # -- protocols and views ---------------------------------------------------------------------------------------------
    def _tune_protocol(self, ctx) -> Protocol:
        epochs = self.knob('tune_epochs')
        return ctx.ranking if epochs in (None, ctx.settings.epochs) else Protocol('tune', epochs=int(epochs))

    @staticmethod
    def probe_protocol() -> Protocol:
        from handlers.model_selection.probe import PROBE_EPOCHS, PROBE_LR
        return Protocol('probe', method='feature_probe',
                        details=(('probe', 'attentive_probe'), ('probe_epochs', PROBE_EPOCHS), ('probe_lr', PROBE_LR)))

    def _view_modes(self, ctx) -> list:
        """[(view, train mode of its base settings)]: frozen, trainable, or both, from the grid and --screen_views."""
        modes = [m for m in ctx.grid.train_modes if 'train_mode' not in ctx.fix or m in ctx.fix['train_mode']]
        frozen = next((m for m in modes if _frozen(m)), None)
        trainable = next((m for m in modes if not _frozen(m)), None)
        wanted = self.knob('screen_views')
        views = []
        if frozen and wanted in ('auto', 'frozen'):
            views.append(('frozen', frozen))
        if trainable and wanted in ('auto', 'trainable'):
            views.append(('trainable', trainable))
        return views or ([('frozen', frozen)] if frozen else [('trainable', trainable)])

    def _probe_problems(self, ctx, items: list) -> list:
        if not ctx.probe_available:
            return ['no feature-probe runner is available here']
        return [f'{item.candidate.backbone}: {problem}' for item in items for problem in [ctx.probe_problem(item.candidate)] if problem]

    def _screen_protocol(self, ctx, view: str, items: list) -> Protocol:
        cv = Protocol('screen', epochs=int(self.knob('screen_epochs')))
        stage = f'screen protocol ({view} view)'
        if view == 'trainable':
            ctx.decide(stage, f'short cross-validation runs ({cv.epochs} epochs, --screen_epochs): weights train in this view, which '
                       'cached frozen features cannot stand for.')
            return cv
        choice = self.knob('screen')
        if choice == 'cv':
            ctx.decide(stage, f'short cross-validation runs ({cv.epochs} epochs), as --screen cv asks.')
            return cv
        problems = self._probe_problems(ctx, items)
        if problems:  # --screen probe was refused before anything trained (check); this is auto
            ctx.decide(stage, f'short cross-validation runs ({cv.epochs} epochs) instead of cached-feature probes, because '
                       + '; '.join(problems) + '.')
            return cv
        ctx.decide(stage, 'cached frozen features + attentive probe: every screened candidate is a frozen backbone the feature '
                   'extractor takes. Probe scores rank backbones within this view only.')
        return self.probe_protocol()

    def check(self, ctx) -> Optional[str]:
        """Refuse before anything trains: an explicit --screen probe that cannot run."""
        if self.knob('screen') != 'probe':
            return None
        backbones = self._backbones(ctx, record=False)
        views = dict(self._view_modes(ctx))
        if 'frozen' not in views:
            return '--screen probe screens frozen backbones, and the grid (after --fix) has no frozen train mode.'
        items = [item for b in backbones for item in [self._nearest(ctx, self._base(ctx, b, views['frozen']))] if item is not None]
        problems = self._probe_problems(ctx, items)
        if problems:
            return '--screen probe cannot run: ' + '; '.join(problems) + '. Use --screen auto or cv.'
        return None

    # -- candidates ------------------------------------------------------------------------------------------------------
    def _base(self, ctx, backbone: str, train_mode: Optional[str] = None) -> Candidate:
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
        if train_mode is not None:
            values['train_mode'] = train_mode
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
        views = len(self._view_modes(ctx))
        top_f, top_b = int(self.knob('top_families')), int(self.knob('top_backbones'))
        screen_epochs, tune = int(self.knob('screen_epochs')), self._tune_protocol(ctx)
        forced = {b for b in backbones if b in ctx.include.get('backbone', set())
                  or backbone_family(b) in ctx.include.get('family', set())}
        jobs, parts = [], []
        if len(families) > top_f:
            jobs += [(items[bs[0]], screen_epochs) for bs in families.values()] * views
            parts.append(f'{len(families)} family screens x {views} view(s)')
        kept = sorted((len(bs) for bs in families.values()), reverse=True)[:min(len(families), top_f * views)]
        if sum(kept) > top_b:
            jobs += [(items[b], screen_epochs) for b in backbones[:sum(kept)]] * views
            parts.append(f'{sum(kept)} backbone screens x {views} view(s)')
        survivors = list(dict.fromkeys(backbones[:min(top_b * views, len(backbones))] + sorted(forced)))
        sizes = [self._group_size(ctx, fields) for _, fields in GROUPS]
        sizes = [size for size in sizes if size > 1]
        per_model = 1 + sum(sizes)
        jobs += [(items[b], tune.epochs) for b in survivors for _ in range(per_model)]
        parts.append(f'{len(survivors)} tuned models (up to {top_b} per view, plus --include) x (1 + {" + ".join(map(str, sizes)) or "0"}) '
                     'group values')
        threshold = int(self.knob('recheck_full_product'))
        pair_runs = sum(a * b if a * b <= threshold else 1 for a, b in itertools.combinations(sizes, 2))
        rechecks = min(int(self.knob('recheck_models')), len(survivors))
        jobs += [(items[survivors[0]], tune.epochs)] * (rechecks * (pair_runs + sum(sizes)))
        parts.append(f'{rechecks} re-checked x ({pair_runs} pair combinations + {sum(sizes)} for one more pass)')
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
        self._ranking_jobs, self._tune_jobs = [], []
        survivors = yield from self._screen(ctx, backbones)
        tuned = []
        for backbone in survivors:
            result = yield from self._tune(ctx, backbone, tune)
            if result is not None:
                tuned.append(result)
        yield from self._recheck(ctx, tuned, tune)
        yield from self._final(ctx, tune)

    def _screen_round(self, ctx, stage: str, members: dict, views: list, top: int, keep_as):
        """One screening round over `members` (a distinct label -> the backbone screened) in every view: each view keeps its own top
        `top`; returns the union, in the order of the views. `keep_as` maps a ranked backbone to the name kept."""
        kept = []
        for view, protocol, items in views:
            jobs = [ctx.job(items[b], protocol, stage, why=f'{name}, {view} view') for name, b in members.items()]
            yield jobs
            ranked = ctx.rank(jobs)
            chosen = [keep_as(job.candidate.backbone) for job, _ in ranked[:top]]
            ctx.decide(f'{stage} ({view} view)', f'best {top}: {", ".join(chosen) or "none"}. By {protocol.describe(ctx.settings.epochs)} '
                       f'on the development folds: {self._ranking_text(ctx, ranked, jobs)}.', ranking=self._rows(ctx, jobs))
            kept += [name for name in chosen if name not in kept]
        return kept

    def _screen(self, ctx, backbones: list):
        top_f, top_b = int(self.knob('top_families')), int(self.knob('top_backbones'))
        include_b, include_f = ctx.include.get('backbone', set()), ctx.include.get('family', set())
        families = {}
        for b in backbones:
            families.setdefault(backbone_family(b), []).append(b)
        members = [b for family in families for b in families[family]]
        if len(families) <= top_f and len(members) <= top_b:
            ctx.decide('screen', f'{len(families)} famil{"y" if len(families) == 1 else "ies"} and {len(members)} backbone'
                       f'{"s" if len(members) != 1 else ""} ({", ".join(members)}), not more than --top_families {top_f} and '
                       f'--top_backbones {top_b}: all go to tuning, nothing to screen.')
            return members
        views = []
        for view, mode in self._view_modes(ctx):
            items = {b: self._nearest(ctx, self._base(ctx, b, mode)) for b in backbones}
            items = {b: item for b, item in items.items() if item is not None}
            views.append((view, self._screen_protocol(ctx, view, list(items.values())), items))
        if len(views) > 1:
            ctx.decide('screen views', 'frozen and trainable views each keep their own best, and the survivors are their union: frozen '
                       'features may rank backbones in another order than fine-tuning, and tuning decides which wins.')

        # 1a: one small model per family
        if len(families) > top_f:
            def size(b):
                return views[0][2][b].footprint.get('total_params') or float('inf'), backbones.index(b)
            reps = {f'the smallest {family} model': min(bs, key=size) for family, bs in families.items()}
            kept = yield from self._screen_round(ctx, 'screen families', reps, views, top_f, backbone_family)
            forced = [f for f in families if f not in kept and (f in include_f or any(b in include_b for b in families[f]))]
            dropped = [f for f in families if f not in kept + forced]
            ctx.decide('screen families', f'kept {", ".join(kept) or "none"}' + (f', plus {", ".join(forced)} (--include)' if forced else '')
                       + f'; dropped {", ".join(dropped) or "none"}.')
            families = {f: families[f] for f in kept + forced}
        else:
            ctx.decide('screen families', f'{len(families)} famil{"y" if len(families) == 1 else "ies"} ({", ".join(families)}), not more '
                       f'than --top_families {top_f}: all kept, nothing to screen.')

        # 1b: every backbone of the kept families
        members = [b for family in families for b in families[family]]
        if len(members) > top_b:
            kept = yield from self._screen_round(ctx, 'screen backbones', {f'{b}, a {backbone_family(b)} backbone': b for b in members},
                                                 views, top_b, lambda b: b)
            forced = [b for b in members if b not in kept and b in include_b]
            ctx.decide('screen backbones', f'kept {", ".join(kept) or "none"}' + (f', plus {", ".join(forced)} (--include)' if forced else '')
                       + f' (--top_backbones {top_b} per view).')
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

    def _track(self, jobs: list, protocol: Protocol) -> None:
        (self._ranking_jobs if protocol.ranking else self._tune_jobs).extend(jobs)

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
                self._track(jobs, protocol)
            return current, None, None
        yield jobs
        self._track(jobs, protocol)
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
        self._track([first], protocol)
        current, score, groups = start.candidate, ctx.score(first), {}
        for group, fields in GROUPS:
            chosen, best, ranked = yield from self._sweep(ctx, current, group, fields, protocol, backbone, f'tune {group}')
            if best is not None:
                current, score = chosen, best
            if ranked and len(ranked) > 1:
                groups[group] = ranked
        return {'backbone': backbone, 'best': current, 'score': score, 'groups': groups}

    def _pair_values(self, ranked: list, fields) -> list:
        """The distinct values (tuples over `fields`) a group's sweep finished, best first."""
        return list(dict.fromkeys(tuple(getattr(candidate, f) for f in fields) for candidate, _ in ranked))

    def _recheck(self, ctx, tuned: list, protocol: Protocol):
        direction, threshold = ctx.direction, int(self.knob('recheck_full_product'))
        order = sorted((t for t in tuned if t['score'] is not None), key=lambda t: -t['score'] * direction)
        for model in order[:int(self.knob('recheck_models'))]:
            best_item = ctx.item(model['best'])
            best_job = ctx.job(best_item, protocol, 'recheck', why='the tuned best')
            jobs, seen, how = [], {model['best']}, []
            for (group_a, ranked_a), (group_b, ranked_b) in itertools.combinations(model['groups'].items(), 2):
                fields_a, fields_b = dict(GROUPS)[group_a], dict(GROUPS)[group_b]
                values_a, values_b = self._pair_values(ranked_a, fields_a), self._pair_values(ranked_b, fields_b)
                if len(values_a) * len(values_b) <= threshold:
                    pairs, what = list(itertools.product(values_a, values_b)), 'every combination'
                else:
                    pairs, what = [(values_a[1], values_b[1])], 'the runner-up values together'
                how.append(f'{group_a} x {group_b}: {what} ({len(values_a)} x {len(values_b)} values)')
                for value_a, value_b in pairs:
                    candidate = model['best'].replace(**dict(zip(fields_a, value_a)), **dict(zip(fields_b, value_b)))
                    item = ctx.item(candidate)
                    if candidate in seen or item is None or item.status != 'runnable':
                        continue
                    seen.add(candidate)
                    jobs.append(ctx.job(item, protocol, 'recheck pairs', why=f'{group_a} ({_show(candidate, fields_a)}) together with '
                                                                            f'{group_b} ({_show(candidate, fields_b)}): {what}'))
            if not jobs:
                ctx.decide(f'recheck ({model["backbone"]})', 'no pair of tuned groups gives a new runnable combination; nothing to re-check'
                           + (f' ({"; ".join(how)}).' if how else '.'))
                continue
            yield jobs
            self._track(jobs, protocol)
            ranked = ctx.rank([best_job] + jobs)
            checked = f'{len(jobs)} combinations ({"; ".join(how)})'
            if not ranked or ranked[0][0] is best_job or ranked[0][1] == model['score']:
                ctx.decide(f'recheck ({model["backbone"]})', f'the one-group-at-a-time choice held against {checked}: '
                           f'{self._ranking_text(ctx, ranked, [best_job] + jobs)}.', ranking=self._rows(ctx, [best_job] + jobs))
                continue
            current, score = ranked[0][0].candidate, ranked[0][1]
            ctx.decide(f'recheck ({model["backbone"]})', f'an interaction: {current.label()} beat the tuned best {model["best"].label()} '
                       f'({score:.4f} against {model["score"]:.4f}) among {checked}; one more one-group-at-a-time pass runs from it.',
                       ranking=self._rows(ctx, [best_job] + jobs))
            for group, fields in GROUPS:
                chosen, best, _ = yield from self._sweep(ctx, current, group, fields, protocol, model['backbone'], f'recheck {group}')
                if best is not None:
                    current, score = chosen, best
            model['best'], model['score'] = current, score

    def _final(self, ctx, protocol: Protocol):
        top = int(self.knob('final_top'))
        if protocol.ranking:
            unique = list({job.key: job for job in self._ranking_jobs}.values())
            ranked = ctx.rank(unique)[:top]
            ctx.decide('final', f'the top {len(ranked)} of this search by grouped cross-validation on the same folds: '
                       + '; '.join(ctx.describe(job) for job, _ in ranked) + '. These are the ranking runs themselves (tuning ran '
                       'under the ranking protocol), so nothing trains again. The engine ranks every ranking-protocol entry of '
                       'this --out folder, runs of another strategy in it included, refits the best and scores it once on the '
                       'sealed test set.', ranking=self._rows(ctx, [job for job, _ in ranked]))
            return
        unique = list({job.key: job for job in self._tune_jobs}.values())
        ranked = ctx.rank(unique)
        candidates = list(dict.fromkeys(job.candidate for job, _ in ranked))[:top]
        jobs = [ctx.job(ctx.item(c), ctx.ranking, 'final', why=f'among the top {top} of all {len(unique)} tune runs ({protocol.name} '
                                                               'protocol); now under the ranking protocol') for c in candidates]
        if jobs:
            yield jobs
        ctx.decide('final', f'cross-validated the top {len(jobs)} of all {len(unique)} tune runs under the ranking protocol: '
                   + '; '.join(ctx.describe(job) for job in jobs) + '. The engine ranks every ranking-protocol entry of this --out '
                   'folder, refits the best and scores it once on the sealed test set.', ranking=self._rows(ctx, jobs))

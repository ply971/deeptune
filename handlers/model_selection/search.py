"""The model search: split once, check every candidate, estimate, train, rank, refit the winner, score it once.

Order of work, and what each step may read:
1. The sealed split (sealed.py) is made once, before anything else, and recorded. A resumed search reuses it.
2. Every candidate is checked by deeptune.py's parser and, with the preflight on, built once to count and size it.
3. The run count and a time estimate are printed (estimate.py). --dry_run stops here.
4. Candidates train through deeptune.py with --cv_only on dev.parquet: grouped k-fold on development rows only.
   Each finished candidate becomes one ledger line.
5. Ranking reads the ledger and the out-of-fold files of development rows only.
6. Only then does the winner train once on all development rows and score once on sealed_test.parquet (--test_df).

Strategies (strategies.py, smart.py) decide which candidates run, under which protocol, and in which order: exhaustive,
smart (the default), random and bayes. A strategy only ever names candidates and protocols (jobs.py); the engine below
builds every command line itself, always with `role='cv'` on dev.parquet, and keeps the ledger. Strategies read results
back through `StrategyContext`, which serves development-row cross-validation scores only. So a new strategy has no way to
reach the sealed test file, and its runs are checked, recorded, resumed and fold-checked like R3a's.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import pandas as pd

from handlers.model_selection.estimate import LARGE_RUN_TRAININGS, describe_hours, jobs_seconds, search_seconds
from handlers.model_selection.grid import (AXES, Candidate, Grid, RunSettings, augmentation_applies, candidate_argv,
                                           extra_augmentation_flags)
from handlers.model_selection.jobs import Job, Protocol
from handlers.model_selection.ledger import FINISHED, LEDGER_FILE, Ledger, candidate_key
from handlers.model_selection.ranking import (DEFAULT_METRIC, METRICS, WINNERS_CURSE, fold_mismatch, fold_stats, leaderboard, paired_difference,
                                              read_cv_result, read_final_result)
from handlers.model_selection.runner import RunResult, SubprocessRunner
from handlers.model_selection.sealed import DEV_FILE, SEALED_FILE, check_split_files, make_sealed_split

MANIFEST_FILE, FINAL_FILE, PILOT_FILE = 'search.json', 'final.json', 'pilot.json'
CANDIDATES_FILE, LEADERBOARD_FILE, SELECTION_FILE = 'candidates.csv', 'leaderboard.csv', 'selection.json'
STAGES_FILE, PROFILE_FILE = 'stages.json', 'profile.json'
EXIT_OK, EXIT_NO_RESULT, EXIT_REFUSED, EXIT_NEEDS_YES, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
IMBALANCE_RATIO = 1.5  # largest / smallest class above this: accuracy gets a warning
STAGE_BIAS = ('Every stage that chose these candidates (screening, tuning, re-checking or the sampler) looked at the same '
              'development folds as the final ranking, so the stages\' choices are optimistically biased too, not only the '
              "winner's score. The sealed-test score is the honest number: no stage and no ranking step read it.")
OVERRIDE_KEYS = tuple(AXES.values())  # backbone, video_arch, pooling, train_mode, learning_rate, num_frames, augmentation


def _normal_value(axis: str, value):
    if axis == 'learning_rate':
        return float(value)
    if axis == 'num_frames':
        return value if str(value) == 'default' else int(value)
    return str(value)


def parse_overrides(words: list, flag: str, extra_keys: tuple = ()) -> dict:
    """['backbone=r3d_18', 'learning_rate=1e-4'] -> {'backbone': {'r3d_18'}, 'learning_rate': {0.0001}}."""
    found = {}
    for word in words or []:
        key, sep, value = str(word).partition('=')
        key, value = key.strip(), value.strip()
        if not sep or not value or key not in OVERRIDE_KEYS + tuple(extra_keys):
            raise ValueError(f'{flag} {word!r}: use key=value with a key from {", ".join(OVERRIDE_KEYS + tuple(extra_keys))}.')
        try:
            found.setdefault(key, set()).add(value if key == 'family' else _normal_value(key, value))
        except ValueError:
            raise ValueError(f'{flag} {word!r}: {value!r} is not a valid {key}.') from None
    return found


def matches(candidate: Candidate, overrides: dict) -> bool:
    """True when the candidate has one of the given values on every axis named. Augmentation on a frozen backbone does
    not apply, so a frozen candidate matches any augmentation."""
    for key, values in overrides.items():
        if key == 'family':
            from handlers.model_selection.strategies import backbone_family
            if backbone_family(candidate.backbone) not in values:
                return False
        elif key == 'augmentation' and not augmentation_applies(candidate.train_mode):
            continue
        elif _normal_value(key, getattr(candidate, key)) not in values:
            return False
    return True


@dataclass
class SearchOptions:
    df: Path
    out: Path
    grid: Grid
    settings: RunSettings
    strategy: str = 'smart'
    metric: Optional[str] = None
    include: dict = field(default_factory=dict)  # smart only: values kept past screening cuts
    exclude: dict = field(default_factory=dict)  # candidates matching any of these are skipped
    fix: dict = field(default_factory=dict)  # candidates without these values are skipped
    strategy_options: dict = field(default_factory=dict)  # per-strategy knobs (strategies.py, smart.py)
    profile_clips: int = 16  # clips the data profile opens for length and motion (0: none)
    raw_data: bool = False
    test_fraction: float = 0.2
    max_trainable_params: Optional[int] = None
    gpu_memory_gb: Optional[float] = None
    preflight: str = 'full'  # full | params | off
    pilot: bool = True
    dry_run: bool = False
    estimate_only: bool = False
    yes: bool = False
    max_hours: float = 8.0
    allow_grid_change: bool = False
    retry_failed: bool = False
    rescore_sealed_test: bool = False


@dataclass
class Checked:
    """A candidate after the checks: status 'runnable', 'refused' or 'skipped'."""
    candidate: Candidate
    key: str
    argv: list
    status: str
    reason: str = ''
    footprint: dict = field(default_factory=dict)
    cost: Optional[float] = None


from handlers.model_selection.smart import SmartStrategy  # noqa: E402  (strategies import nothing from this module)
from handlers.model_selection.strategies import BASE_STRATEGIES  # noqa: E402

STRATEGIES = {**BASE_STRATEGIES, 'smart': SmartStrategy}


class StrategyContext:
    """What a strategy may see and ask for. Every job it gets runs `role='cv'` on dev.parquet; every score it reads is a
    mean over the development folds. It has no handle on the sealed test file, the final run or the runner."""

    def __init__(self, search: 'ModelSearch', checked: list):
        self._search = search
        self.checked = checked
        self.runnable = [item for item in checked if item.status == 'runnable']
        self._by_candidate = {item.candidate: item for item in checked}
        self.grid, self.settings, self.mode, self.metric = search.o.grid, search.o.settings, search.mode, search.metric
        self.direction = METRICS[self.mode][self.metric]
        self.include, self.fix = search.o.include, search.o.fix
        self.ranking = Protocol()
        self.decisions: list = []
        # The cached-feature probe needs real clips: a test's fake runner offers it only if it has a `probe` method.
        self.probe_available = search.probe_runner is not None or isinstance(search.runner, SubprocessRunner)
        self._probe_check = search.probe_check

    def probe_problem(self, candidate: Candidate) -> Optional[str]:
        """Why this candidate cannot be screened by the cached-feature probe, or None."""
        if self._probe_check is not None:
            return self._probe_check(candidate, self.settings)
        from handlers.model_selection.probe import probe_problem
        return probe_problem(candidate, self.settings)

    # -- candidates and jobs ---------------------------------------------------------------------------------------------
    def item(self, candidate: Candidate):
        """The checked candidate (any status), after the augmentation collapse; None if it is not in the grid."""
        return self._by_candidate.get(candidate.replace())

    def job(self, item, protocol: Optional[Protocol] = None, stage: str = '', why: str = '') -> Job:
        protocol = protocol or self.ranking
        search = self._search
        key = item.key if protocol.ranking else candidate_key(item.candidate.config(), search.dev_fp,
                                                              protocol.key_settings(self.settings.training()))
        argv = candidate_argv(item.candidate, self.settings, search.dev, search.out / 'runs' / key, 'cv', epochs=protocol.epochs)
        return Job(item, protocol, stage, why, key, argv)

    # -- results ---------------------------------------------------------------------------------------------------------
    def entry(self, job: Job) -> Optional[dict]:
        """The job's ledger entry once it has finished (done, failed or did not fit), else None."""
        entry = self._search.live_ledger.get(job.key)
        return entry if entry is not None and _finished(entry) else None

    def score(self, job: Job) -> Optional[float]:
        """Mean metric over the development folds, or None if the job has no finished result."""
        entry = self.entry(job)
        return fold_stats(entry, self.metric)[0] if entry is not None and entry['status'] == 'done' else None

    def rank(self, jobs: list) -> list:
        """[(job, score)] of the finished ones, best first; ties keep the given order. Only compare jobs of one protocol."""
        if len({job.protocol for job in jobs}) > 1:
            raise ValueError('Jobs of different protocols cannot be ranked together.')
        scored = [(job, self.score(job)) for job in jobs]
        scored = [(job, score) for job, score in scored if score is not None]
        return sorted(scored, key=lambda pair: -pair[1] * self.direction)

    def finished_count(self, jobs: list) -> int:
        return sum(self.entry(job) is not None for job in jobs)

    def describe(self, job: Job) -> str:
        entry, score = self.entry(job), self.score(job)
        if score is not None:
            return f'{job.candidate.label()} ({self.metric} {score:.4f})'
        return f'{job.candidate.label()} ({entry["status"] if entry else "not run"})'

    def decide(self, stage: str, text: str, **data) -> None:
        """A stage's choice and its reason: printed now, kept for stages.json and selection.json."""
        self.decisions.append({'stage': stage, 'decision': text, **_json_normal(data)})
        self._search.say(f'  [{stage}] {text}')
        _write_json(self._search.out / STAGES_FILE, self.decisions)


def code_version() -> dict:
    """The git commit of this code, and whether tracked files had changed (T1's run_environment)."""
    from trainers.training_basics import run_environment
    environment = run_environment()
    return {'commit': environment['commit'], 'commit_dirty': environment['commit_dirty']}


def _finished(entry: dict) -> bool:
    """Trained or tried. A check's decision (refused, skipped, a preflight crash) is made again on every run."""
    return entry['status'] in FINISHED and entry.get('stage') != 'checks'


def _json_normal(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _write_json(path: Path, value) -> None:
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def _raw_fingerprint(path: Path) -> str:
    from trainers.training_basics import data_fingerprint
    path = Path(path)
    if path.is_file():
        return data_fingerprint(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob('*') if p.is_file()):
        stat = item.stat()
        digest.update(f'{item.relative_to(path).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


class ModelSearch:
    def __init__(self, options: SearchOptions, runner: Optional[Callable] = None, footprint_fn: Optional[Callable] = None,
                 check_fn: Optional[Callable] = None, say: Callable[[str], None] = None, probe_runner: Optional[Callable] = None):
        from handlers.model_selection.checks import check_argv, measure_footprint
        self.o = options
        self.runner = runner or SubprocessRunner()
        # The cached-feature probe (smart's screening). A test runner that also has a `probe` method serves both.
        self.probe_runner = probe_runner or getattr(runner, 'probe', None)
        # Which candidates it can screen: its own `probe_problem` if it has one (a simulation), else the real extractor's rules.
        self.probe_check = getattr(probe_runner, 'probe_problem', None) or getattr(runner, 'probe_problem', None)
        self.footprint_fn = footprint_fn or measure_footprint
        self.check_fn = check_fn or check_argv
        self.say = say or (lambda text: print(text, flush=True))
        if options.strategy not in STRATEGIES:
            raise ValueError(f'Unknown search strategy {options.strategy!r}; choose one of {sorted(STRATEGIES)}.')
        self.strategy = STRATEGIES[options.strategy](options)
        self.dev: Optional[Path] = None
        self.dev_fp: Optional[str] = None
        self.live_ledger: dict = {}
        self.stopped: Optional[str] = None
        self.profile: Optional[dict] = None
        self.context: Optional[StrategyContext] = None
        self.mode = options.settings.mode
        self.metric = options.metric or DEFAULT_METRIC[self.mode]
        if self.metric not in METRICS[self.mode]:
            raise ValueError(f'--metric {self.metric} does not apply to {self.mode}; choose one of {sorted(METRICS[self.mode])}.')
        self.out = Path(options.out).resolve()
        self.data_dir = self.out / 'data'
        self.ledger = Ledger(self.out / LEDGER_FILE)
        self.code: Optional[dict] = None
        self.warnings: list = []

    def warn(self, text: str) -> None:
        """Printed now, and kept for selection.json."""
        if text not in self.warnings:
            self.warnings.append(text)
        self.say(f'Warning: {text}')

    # -- data and resume -------------------------------------------------------------------------------------------------
    def _input(self) -> tuple[Path, str]:
        from trainers.training_basics import data_fingerprint
        source = Path(self.o.df)
        if not source.exists():
            raise FileNotFoundError(f'--df does not exist: {source}')
        if not self.o.raw_data:
            if source.suffix.lower() != '.parquet' or not source.is_file():
                raise ValueError('--df must be a .parquet file, or pass --raw-data for a clip folder or a CSV/XLSX list.')
            return source.resolve(), data_fingerprint(source)
        converted = self.data_dir / 'input.parquet'
        if not converted.is_file():
            from handlers.raw_to_parquet_dataset import raw_to_parquet
            made = raw_to_parquet(dataset_dir=source, out=self.data_dir / 'raw_conversion', modality='video', video_storage='paths')
            self.data_dir.mkdir(parents=True, exist_ok=True)
            Path(made).replace(converted)
            shutil.rmtree(self.data_dir / 'raw_conversion', ignore_errors=True)
        return converted, _raw_fingerprint(source)

    def _manifest(self, input_path: Path, input_fp: str) -> tuple[Optional[dict], Optional[str]]:
        """(manifest, None), or (None, why this out folder cannot be resumed with these inputs)."""
        path = self.out / MANIFEST_FILE
        settings, grid = _json_normal(self.o.settings.training()), _json_normal(self.o.grid.to_dict())
        if not path.is_file():
            if self.ledger.path.is_file():
                return None, f'{self.out} has a ledger but no {MANIFEST_FILE}; it is not a search folder this code can resume.'
            self.say(f'Setting the sealed test set aside (about {self.o.test_fraction:.0%}, seed {self.o.settings.seed}) before anything trains.')
            split = make_sealed_split(input_path, self.data_dir, self.o.settings.target, self.o.settings.grouper, self.mode,
                                      self.o.test_fraction, self.o.settings.seed)
            sealed_part = split['sealed_test']
            self.say(f'Sealed test set: {sealed_part["rows"]} clips' + (f' in {sealed_part["groups"]} groups' if 'groups' in sealed_part else '')
                     + f'; development rows: {split["dev"]["rows"]}; unit: {split["unit"]}.')
            manifest = {'schema_version': 1, 'input': str(Path(self.o.df).resolve()), 'input_fingerprint': input_fp,
                        'raw_data': self.o.raw_data, 'test_fraction': self.o.test_fraction, 'split': split, 'settings': settings,
                        'grid': grid, 'grid_history': [], 'strategy': self.o.strategy}
            _write_json(path, manifest)
            return manifest, None
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['grid'].setdefault('augmentations', ['off'])  # R3a folders had no augmentation axis: it was off
        if manifest['input_fingerprint'] != input_fp:
            return None, ('the dataset changed since this search started: its sealed test split was drawn from the old file. '
                          'Use a new --out for the new data.')
        if manifest['test_fraction'] != self.o.test_fraction:
            return None, f'--test_fraction was {manifest["test_fraction"]} when this search started; the sealed split cannot change.'
        if manifest['settings'] != settings:
            changed = sorted(name for name in set(settings) | set(manifest['settings']) if settings.get(name) != manifest['settings'].get(name))
            return None, (f'these settings differ from the ones this search started with: {changed}. Earlier results were trained with '
                          'the old values. Use the old values, or a new --out.')
        problem = check_split_files(self.data_dir, manifest['split'])
        if problem:
            return None, problem
        if manifest['grid'] != grid:
            if not self.o.allow_grid_change:
                return None, ('the grid differs from the one this search started with. Pass --allow_grid_change to keep the results '
                              'of candidates still in the grid, train the new ones, and leave the dropped ones out of the leaderboard.')
            manifest['grid_history'].append(manifest['grid'])
            manifest['grid'] = grid
            _write_json(path, manifest)
            self.say('The grid changed (--allow_grid_change): results of candidates still in it are kept.')
        if manifest.get('strategy') != self.o.strategy:
            manifest.setdefault('strategy_history', []).append(manifest.get('strategy'))
            manifest['strategy'] = self.o.strategy
            _write_json(path, manifest)
            self.say(f'This folder was searched with --search {manifest["strategy_history"][-1]} before; finished candidates are reused, '
                     'and the sealed test set still scores only one winner unless --rescore_sealed_test.')
        return manifest, None

    # -- checks ----------------------------------------------------------------------------------------------------------
    def _num_classes(self, dev: Path) -> int:
        if self.o.settings.num_classes is not None:
            return int(self.o.settings.num_classes)
        if self.mode != 'cls':
            return 1
        return int(pd.read_parquet(dev, columns=[self.o.settings.target])[self.o.settings.target].nunique())

    def _gpu_bytes(self) -> Optional[float]:
        return None if self.o.gpu_memory_gb is None else self.o.gpu_memory_gb * 1e9

    def check_candidates(self, dev: Path, dev_fp: str) -> list:
        settings, training = self.o.settings, self.o.settings.training()
        num_classes = self._num_classes(dev)
        builds, checked = {}, []
        for candidate in self.o.grid.expand():
            key = candidate_key(candidate.config(), dev_fp, training)
            argv = candidate_argv(candidate, settings, dev, self.out / 'runs' / key, 'cv')
            override = self._override_reason(candidate)
            if override:
                checked.append(Checked(candidate, key, argv, 'skipped', override))
                continue
            args, reason = self.check_fn(argv)
            if reason:
                checked.append(Checked(candidate, key, argv, 'refused', f'deeptune.py: {reason}'))
                continue
            if self.o.preflight == 'off':
                checked.append(Checked(candidate, key, argv, 'runnable'))
                continue
            build = candidate.build_key()
            if build not in builds:
                try:
                    builds[build] = self.footprint_fn(args, num_classes, self.o.preflight == 'full')
                except Exception as exc:  # the trainer's own builder refused this combination
                    builds[build] = exc
            footprint = builds[build]
            if isinstance(footprint, ValueError):  # the builder's refusal of an invalid combination
                checked.append(Checked(candidate, key, argv, 'refused', f'model build: {type(footprint).__name__}: {footprint}'))
                continue
            if isinstance(footprint, Exception):  # a crash, e.g. a failed download: not the candidate's fault, tried again next run
                checked.append(Checked(candidate, key, argv, 'failed', f'preflight: {type(footprint).__name__}: {footprint}'))
                continue
            frames = int(args.num_frames)
            item = Checked(candidate, key, argv, 'runnable', footprint=footprint.to_dict(), cost=footprint.cost(frames))
            item.footprint['estimated_gpu_gb'] = round(footprint.memory_bytes(settings.batch_size) / 1e9, 2)
            if self.o.max_trainable_params is not None and footprint.trainable_params > self.o.max_trainable_params:
                item.status, item.reason = 'skipped', (f'{footprint.trainable_params:,} trainable parameters, over the budget of '
                                                      f'{self.o.max_trainable_params:,} (--max_trainable_params)')
            elif self._gpu_bytes() is not None and footprint.memory_bytes(settings.batch_size) > self._gpu_bytes():
                item.status, item.reason = 'skipped', (f'estimated {footprint.memory_bytes(settings.batch_size) / 1e9:.1f} GB of GPU memory '
                                                      f'at batch size {settings.batch_size}, over {self.o.gpu_memory_gb:g} GB (rough estimate, '
                                                      'see handlers/model_selection/checks.py)')
            checked.append(item)
        return checked

    def _override_reason(self, candidate: Candidate) -> str:
        for key, values in self.o.exclude.items():
            if matches(candidate, {key: values}):
                return f'excluded by --exclude {key}={"|".join(map(str, sorted(values, key=str)))}'
        for key, values in self.o.fix.items():
            if not matches(candidate, {key: values}):
                return f'not the value fixed by --fix {key}={"|".join(map(str, sorted(values, key=str)))}'
        return ''

    def _record_decisions(self, checked: list, ledger: dict) -> None:
        """Refused and skipped candidates get a ledger line, once per new decision, so none is silently dropped."""
        for item in checked:
            if item.status in ('refused', 'skipped', 'failed'):
                last = ledger.get(item.key)
                if last is None or (not _finished(last) and (last['status'], last.get('reason')) != (item.status, item.reason)):
                    ledger[item.key] = self.ledger.append(self._entry(item, item.status, reason=item.reason, stage='checks'))
        rows = [{'key': item.key, 'candidate': item.candidate.label(), **item.candidate.config(), 'check': item.status,
                 'reason': item.reason, **{f'preflight_{k}': v for k, v in item.footprint.items()}} for item in checked]
        pd.DataFrame(rows).to_csv(self.out / CANDIDATES_FILE, index=False)

    def _entry(self, item: Checked, status: str, **extra) -> dict:
        return {'key': item.key, 'label': item.candidate.label(), 'config': item.candidate.config(), 'status': status,
                'footprint': item.footprint, 'code': self.code, **extra}

    # -- estimate --------------------------------------------------------------------------------------------------------
    def _pilot(self, pending: list, dev: Path) -> Optional[dict]:
        path = self.out / PILOT_FILE
        if path.is_file():
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('status') == 'ok':
                return record
        if self.o.dry_run or not self.o.pilot or not pending:
            return None
        cheapest = min(pending, key=lambda item: item.cost if item.cost is not None else float('inf'))
        run_dir = self.out / 'pilot' / cheapest.key
        shutil.rmtree(run_dir, ignore_errors=True)
        argv = candidate_argv(cheapest.candidate, self.o.settings, dev, run_dir, 'cv', epochs=1, k=2)
        self.say(f'Timing a pilot before anything else trains: {cheapest.candidate.label()}, 1 epoch on 2 folds of the development rows '
                 '(--no_pilot skips it).')
        result = self.runner(argv, run_dir, 'pilot')
        record = {'key': cheapest.key, 'label': cheapest.candidate.label(), 'status': result.status, 'seconds': result.seconds,
                  'cost': cheapest.cost, 'reason': result.reason, 'what': '1 epoch on 2 folds, one deeptune.py process'}
        _write_json(path, record)
        if result.status != 'ok':
            self.say(f'The pilot did not finish ({result.status}: {result.reason}); the time estimate is unknown.')
            return None
        return record

    # -- warnings --------------------------------------------------------------------------------------------------------
    def _imbalance_warning(self, split: dict) -> None:
        counts = split['dev'].get('class_counts') or {}
        if self.mode != 'cls' or len(counts) < 2 or self.metric != 'accuracy':
            return
        ratio = max(counts.values()) / min(counts.values())
        if ratio > IMBALANCE_RATIO:
            self.warn(f'the classes are imbalanced (the largest has {ratio:.1f}x the clips of the smallest in the development rows), '
                      'so accuracy rewards predicting the large classes. Consider --metric balanced_accuracy.')

    def _code_warning(self, ledger: dict) -> None:
        other = self._other_commits(ledger.values())
        if other:
            self.warn(f'{len(other)} finished candidates were trained by other code versions (commits '
                      f'{", ".join(sorted({str(c)[:12] for c in other.values()}))}) than this one ({str(self.code.get("commit"))[:12]}). '
                      'Their scores may not be comparable with new ones.')

    def _other_commits(self, entries) -> dict:
        """key -> commit, for finished candidates whose recorded commit differs from the current one."""
        current = (self.code or {}).get('commit')
        return {entry['key']: (entry.get('code') or {}).get('commit') for entry in entries
                if _finished(entry) and (entry.get('code') or {}).get('commit') != current}

    # -- the search ------------------------------------------------------------------------------------------------------
    def run(self) -> int:
        o, k = self.o, self.o.settings.k
        self.out.mkdir(parents=True, exist_ok=True)
        try:
            input_path, input_fp = self._input()
            manifest, problem = self._manifest(input_path, input_fp)
        except (OSError, ValueError) as exc:
            self.say(f'Refused: {exc}')
            return EXIT_REFUSED
        if problem:
            self.say(f'Refused: {problem}')
            return EXIT_REFUSED
        split = manifest['split']
        dev, sealed = self.data_dir / DEV_FILE, self.data_dir / SEALED_FILE
        ledger = self.ledger.load()
        self.code = code_version()
        for warning in split.get('warnings', []):
            self.warn(warning)
        self._imbalance_warning(split)
        self._code_warning(ledger)
        self._augmentation_warning()
        self.dev, self.dev_fp, self.live_ledger = dev, split['dev']['fingerprint'], ledger

        self.say(f'Search {self.strategy.name}: {o.grid.formula()}.')
        self.profile = self._profile(dev, split)
        checked = self.check_candidates(dev, split['dev']['fingerprint'])
        self._record_decisions(checked, ledger)
        ctx = self.context = StrategyContext(self, checked)
        runnable = ctx.runnable
        retry = ('failed', 'did_not_fit') if o.retry_failed else ()
        finished = {key for key, entry in ledger.items() if _finished(entry) and entry['status'] not in retry}
        plan = self.strategy.plan(ctx)
        exact = not self.strategy.adaptive
        planned = [item for item, _ in plan.jobs]
        pending_jobs = [(item, epochs) for item, epochs in plan.jobs if not exact or item.key not in finished]
        pending = [item for item, _ in pending_jobs]
        final_record = self._final_record()
        final_needed = not (final_record and final_record.get('status') == 'done' and not pending)
        trainings = len(pending) * k + (1 if final_needed else 0)
        counts = {status: sum(item.status == status for item in checked) for status in ('runnable', 'refused', 'skipped', 'failed')}
        already = len(planned) - len(pending) if exact else sum(item.key in finished for item in runnable)
        self.say(f'Candidates: {len(checked)}; refused by checks: {counts["refused"]}; skipped (budget or memory): {counts["skipped"]}; '
                 f'preflight failed: {counts["failed"]}; '
                 f'runnable: {counts["runnable"]}, of which {already} already finished in the ledger.')
        for item in checked:
            if item.status != 'runnable':
                self.say(f'  {item.status}: {item.candidate.label()}: {item.reason}')
        for folded in o.grid.collapsed():
            self.say(f'  folded: {folded["asked"]} runs as {folded["runs_as"]}: {folded["reason"]}')
        if exact:
            self.say(f'Trainings to run: {len(pending)} candidates x k={k} folds' + (' + 1 final refit' if final_needed else '') + f' = {trainings}.')
        else:
            self.say(f'Trainings to run: at most {len(pending)} candidate runs x k={k} folds' + (' + 1 final refit' if final_needed else '')
                     + f' = {trainings} (an upper bound, before any reuse from the ledger: {plan.note}).')

        pilot_pool = [item for item in pending if item.key not in finished] if pending else []
        pilot = self._pilot(pilot_pool, dev) if pilot_pool else None
        seconds = None
        if pilot is not None:
            if exact:
                seconds = search_seconds(pilot['seconds'], [item.cost for item in pending], pilot.get('cost'), k, o.settings.epochs, final_needed)
            else:
                seconds = jobs_seconds(pilot['seconds'], [(item.cost, epochs or o.settings.epochs) for item, epochs in pending_jobs],
                                       pilot.get('cost'), k, o.settings.epochs, final_needed)
            basis = (f'pilot {pilot["label"]}: {pilot["seconds"]:.0f} s for 1 epoch on 2 folds, scaled per candidate by its '
                     f'{"FLOP-based cost" if o.preflight == "full" else "parameter count" if o.preflight == "params" else "(no preflight: equal) cost"}, '
                     f'rows per fold, k and {o.settings.epochs} epochs')
            self.say(f'Time estimate: {"at most " if not exact else ""}{describe_hours(seconds)} (rough; basis: {basis}. '
                     'Early stopping can only shorten it).')
        elif pending:
            self.say('Time estimate: unknown (' + ('a dry run trains nothing, not even the pilot' if o.dry_run else 'no pilot timing') + ').')
        if not exact:
            self.say(f'Budget: --max_hours {o.max_hours:g}: no new training starts once the trainings recorded in the ledger add up to it; '
                     'what finished is then ranked.')
        if o.dry_run:
            self.say('--dry_run: nothing was trained.')
            return EXIT_OK
        if o.estimate_only:
            self.say('--estimate_only: stopping after the estimate.')
            return EXIT_OK
        if pending and not o.yes and exact:
            if seconds is not None and seconds / 3600 > o.max_hours:
                self.say(f'Refused: the estimate ({describe_hours(seconds)}) is over --max_hours {o.max_hours:g}. '
                         'Shrink the grid, or pass --yes to run it anyway.')
                return EXIT_NEEDS_YES
            if seconds is None and trainings > LARGE_RUN_TRAININGS:
                self.say(f'Refused: {trainings} trainings with no time estimate (over {LARGE_RUN_TRAININGS}). Pass --yes to run it anyway.')
                return EXIT_NEEDS_YES

        try:
            self._run_jobs(ctx, len(pending), exact)
        except KeyboardInterrupt:
            done = sum(_finished(entry) for entry in self.ledger.load().values())
            self.say(f'Interrupted. The ledger holds {done} finished candidates; the one that was running has no line and runs again. '
                     'Rerun the same command to resume.')
            return EXIT_INTERRUPTED
        return self._conclude(checked, manifest, dev, sealed)

    def _used_seconds(self) -> float:
        return sum(float(entry.get('seconds') or 0) for entry in self.live_ledger.values() if _finished(entry))

    def _run_jobs(self, ctx: StrategyContext, total: int, exact: bool) -> None:
        """Run the strategy's jobs in its order. A job already finished in the ledger is not run again (resume), unless
        --retry_failed and it failed; the strategy reads its result either way. Adaptive strategies stop at --max_hours."""
        retry = ('failed', 'did_not_fit') if self.o.retry_failed else ()
        retried, position = set(), 0
        budget = None if exact else self.o.max_hours * 3600
        jobs = self.strategy.jobs(ctx)
        try:
            for batch in jobs:
                for job in batch:
                    entry = self.live_ledger.get(job.key)
                    if entry is not None and _finished(entry) and not (entry['status'] in retry and job.key not in retried):
                        continue
                    if budget is not None and self._used_seconds() >= budget:
                        self.stopped = (f'stopped by --max_hours {self.o.max_hours:g}: the trainings recorded in the ledger took '
                                        f'{self._used_seconds() / 3600:.2f} h; candidates the {self.strategy.name} search would have run '
                                        'next were not run')
                        self.say(f'Budget reached: {self.stopped}.')
                        return
                    retried.add(job.key)
                    position += 1
                    self._run_job(job, f'[{position}/{total}]' if exact else f'[{position} of at most {total}] {job.stage}:')
        finally:
            jobs.close()

    def _probe_info(self) -> dict:
        return {'dev': self.dev, 'settings': self.o.settings, 'features_root': self.out / 'features', 'mode': self.mode,
                'num_classes': self._num_classes(self.dev)}

    def _run_job(self, job: Job, prefix: str) -> None:
        item = job.item
        run_dir = self.out / 'runs' / job.key
        shutil.rmtree(run_dir, ignore_errors=True)  # an unfinished run left by an interruption
        self.say(f'{prefix} {item.candidate.label()}' + ('' if job.protocol.ranking else f' ({job.protocol.name} protocol)'))
        if job.protocol.method == 'feature_probe':
            if self.probe_runner is None:
                from handlers.model_selection.probe import FeatureProbeRunner
                self.probe_runner = FeatureProbeRunner(env=getattr(self.runner, 'env', None))
            result: RunResult = self.probe_runner(job, run_dir, self._probe_info())
        else:
            result = self.runner(job.argv, run_dir, 'cv')
        entry = self._entry(item, result.status, key=job.key, seconds=round(result.seconds, 1), reason=result.reason,
                            run_dir=str(run_dir.relative_to(self.out)), stage=job.stage or self.strategy.name, why=job.why,
                            protocol=job.protocol.name, method=job.protocol.method,
                            epochs=self.o.settings.epochs if job.protocol.epochs is None else job.protocol.epochs)
        if result.status == 'ok':
            try:
                entry.update(read_cv_result(run_dir, self.out), status='done')
            except (OSError, ValueError, KeyError) as exc:
                entry.update(status='failed', reason=f'finished, but its results could not be read: {exc}')
        mean, _ = fold_stats(entry, self.metric) if entry['status'] == 'done' else (None, None)
        self.say(f'    {entry["status"]}' + (f': mean CV {self.metric} {mean:.4f}' if mean is not None else f': {entry["reason"]}'))
        self.live_ledger[job.key] = self.ledger.append(entry)

    def _augmentation_warning(self) -> None:
        flags = extra_augmentation_flags(self.o.settings.deeptune_args)
        if flags:
            self.warn(f'augmentation flags in --deeptune_args ({", ".join(flags)}) reach every run, frozen backbones included, and the '
                      "leaderboard's augmentation column cannot show them. Use --augmentations to search augmentation instead.")

    def _profile(self, dev: Path, split: dict) -> Optional[dict]:
        from handlers.model_selection.profile import print_profile, profile_development
        try:
            profile = profile_development(dev, split, self.o.settings, self.o.grid, self.mode, self.o.profile_clips)
        except Exception as exc:  # the profile informs; it must never cost the search
            self.say(f'Data profile: not made ({type(exc).__name__}: {exc}).')
            return {'error': f'{type(exc).__name__}: {exc}'}
        _write_json(self.out / PROFILE_FILE, profile)
        print_profile(profile, self.say)
        return profile

    def _final_record(self) -> Optional[dict]:
        path = self.out / FINAL_FILE
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    # -- ranking, the final refit and the record -------------------------------------------------------------------------
    def _conclude(self, checked: list, manifest: dict, dev: Path, sealed: Path) -> int:
        o = self.o
        ledger = self.ledger.load()
        if self.strategy.adaptive:
            entries = [ledger.get(item.key) or self._entry(item, item.status if item.status != 'runnable' else 'not_run',
                                                           reason=item.reason or f'not chosen by the {self.strategy.name} search')
                       for item in checked]
        else:
            entries = [ledger.get(item.key) or self._entry(item, item.status if item.status != 'runnable' else 'skipped',
                                                           reason=item.reason or 'not run') for item in checked]
        configs = {json.dumps(item.candidate.config(), sort_keys=True) for item in checked}
        other_protocols = [entry for entry in ledger.values() if entry.get('protocol', 'cv') != 'cv' and _finished(entry)
                           and json.dumps(entry['config'], sort_keys=True) in configs]
        table = leaderboard(entries + other_protocols, self.metric, self.mode)
        table.to_csv(self.out / LEADERBOARD_FILE, index=False)
        ranked = [entry for entry in sorted((e for e in entries if e['status'] == 'done' and fold_stats(e, self.metric)[0] is not None),
                                            key=lambda e: list(table['key']).index(e['key']))]
        selection = self._selection_base(manifest, checked, entries)
        # Same folds at every stage: screening and tuning results are checked too, not only the ranked ones.
        mismatch = fold_mismatch(ranked + [entry for entry in other_protocols if entry['status'] == 'done'])
        if mismatch:
            selection['winner'] = None
            selection['reason'] = f'Refused to rank: {mismatch} The sealed test set was not used.'
            _write_json(self.out / SELECTION_FILE, selection)
            self.say(selection['reason'])
            return EXIT_REFUSED
        if not ranked:
            selection['winner'] = None
            selection['reason'] = ('No candidate finished under the ranking protocol, so there is no winner and the sealed test set was '
                                   'not used.' + (f' The search was {self.stopped}.' if self.stopped else ''))
            _write_json(self.out / SELECTION_FILE, selection)
            self.say(selection['reason'])
            return EXIT_NO_RESULT
        winner = ranked[0]
        by_key = {item.key: item for item in checked}
        selection['winner'] = self._describe(winner)
        if len(ranked) > 1:
            runner_up = ranked[1]
            selection['runner_up'] = self._describe(runner_up)
            try:
                selection['paired_difference'] = paired_difference(self.out, winner, runner_up, self.mode, self.metric, o.settings.boot,
                                                                   o.settings.seed)
            except (OSError, ValueError, KeyError) as exc:
                selection['paired_difference'] = {'error': str(exc)}
        else:
            selection['runner_up'] = None
            selection['paired_difference'] = {'note': 'only one candidate finished, so there is nothing to compare'}
        selection['reason'] = self._reason(selection)

        final = self._final(by_key[winner['key']], dev, sealed)
        selection['final'] = {key: value for key, value in final.items() if key != 'test'}
        selection['test'] = final.get('test')
        selection['seeds']['refit_validation'] = (final.get('test') or {}).get('validation_seed')
        selection['sealed_test_uses'] = final.get('sealed_test_uses', 0)
        if selection['sealed_test_uses'] > 1:
            selection['notes'].append(f'The sealed test set has scored {selection["sealed_test_uses"]} different winners of this folder, '
                                      'so its score is no longer an untouched estimate.')
        _write_json(self.out / SELECTION_FILE, selection)
        self._summary(selection, table)
        return EXIT_OK if final.get('status') in ('done', 'not_scored') else EXIT_NO_RESULT

    def _describe(self, entry: dict) -> dict:
        mean, sd = fold_stats(entry, self.metric)
        return {'key': entry['key'], 'label': entry['label'], 'config': entry['config'], f'cv_{self.metric}_mean': mean,
                f'cv_{self.metric}_fold_sd': sd, 'folds': len(entry.get('folds', [])),
                f'oof_{self.metric}_interval': entry.get('intervals', {}).get(self.metric), 'footprint': entry.get('footprint', {}),
                'run_dir': entry.get('run_dir')}

    def _reason(self, selection: dict) -> str:
        better = 'highest' if METRICS[self.mode][self.metric] > 0 else 'lowest'
        winner = selection['winner']
        text = (f'Highest-ranked by the {better} mean {self.metric} over the {winner["folds"]} cross-validation folds of the development '
                f'rows ({winner[f"cv_{self.metric}_mean"]:.4f}).')
        runner_up, paired = selection.get('runner_up'), selection.get('paired_difference') or {}
        if runner_up and paired.get('low') is not None:
            verdict = ('the interval excludes 0' if paired['zero_excluded'] else
                       'the interval includes 0, so on this data the two are not distinguishable')
            text += (f' Runner-up {runner_up["label"]} ({runner_up[f"cv_{self.metric}_mean"]:.4f}); paired difference on the pooled '
                     f'out-of-fold rows {paired["point"]:+.4f} [{paired["low"]:+.4f}, {paired["high"]:+.4f}], {verdict}. '
                     'The top two were picked on these same rows, so this interval flatters their gap: it is not a test, and it is '
                     'not corrected for the selection.')
        return text

    def _final(self, item: Checked, dev: Path, sealed: Path) -> dict:
        """Refit the winner on every development row and score it once on the sealed test set."""
        previous = self._final_record()
        uses = (previous or {}).get('sealed_test_uses', 0)
        if previous and previous.get('key') == item.key and previous.get('status') == 'done':
            return previous
        if previous and previous.get('test') is not None and previous.get('key') != item.key and not self.o.rescore_sealed_test:
            reason = (f'The sealed test set already scored an earlier winner ({previous.get("label")}), so it is not used again for '
                      f'{item.candidate.label()}. Pass --rescore_sealed_test to do so knowingly.')
            self.say(reason)
            return {'key': item.key, 'label': item.candidate.label(), 'status': 'not_scored', 'reason': reason, 'sealed_test_uses': uses,
                    'earlier_winner': {name: previous.get(name) for name in ('key', 'label', 'test')}}
        run_dir = self.out / 'final' / item.key
        shutil.rmtree(run_dir, ignore_errors=True)
        argv = candidate_argv(item.candidate, self.o.settings, dev, run_dir, 'final', test_df=sealed)
        self.say(f'Final refit of the winner on all development rows, then one score on the sealed test set: {item.candidate.label()}')
        result = self.runner(argv, run_dir, 'final')
        record = {'key': item.key, 'label': item.candidate.label(), 'status': 'done' if result.status == 'ok' else result.status,
                  'seconds': round(result.seconds, 1), 'reason': result.reason, 'run_dir': str(run_dir.relative_to(self.out)),
                  'sealed_test_uses': uses}
        if result.status == 'ok':
            try:
                record['test'] = read_final_result(run_dir)
                record['sealed_test_uses'] = uses + 1
            except (OSError, ValueError) as exc:
                record.update(status='failed', reason=f'finished, but its test report could not be read: {exc}')
        _write_json(self.out / FINAL_FILE, record)
        return record

    def _selection_base(self, manifest: dict, checked: list, entries: list) -> dict:
        statuses = ('done', 'failed', 'did_not_fit', 'refused', 'skipped') + (('not_run',) if self.strategy.adaptive else ())
        counts = {status: sum(entry['status'] == status for entry in entries) for status in statuses}
        backbones = sorted({entry['config']['backbone'] for entry in entries if entry['status'] == 'done'})
        trainings = [entry for entry in self.ledger.load().values() if _finished(entry)]
        notes_extra = [STAGE_BIAS] if self.strategy.adaptive else []
        if any(entry.get('protocol', 'cv') != 'cv' for entry in trainings):
            notes_extra.append('Screening and tuning scores (protocols screen, tune, probe) are on other protocols than the ranking '
                               "protocol (cv): the leaderboard lists them without a rank, and they never compete with cv scores.")
        return {'schema_version': 2, 'strategy': self.strategy.name, 'strategy_settings': self.strategy.settings(),
                'grid': self.o.grid.to_dict(), 'grid_formula': self.o.grid.formula(), 'augmentation_collapsed': self.o.grid.collapsed(),
                'overrides': {name: {key: sorted(map(str, values)) for key, values in getattr(self.o, name).items()}
                              for name in ('include', 'exclude', 'fix')},
                'stages': self.context.decisions if self.context else [], 'profile': self.profile,
                'budget': {'max_hours': self.o.max_hours, 'applies': 'as a start gate' if not self.strategy.adaptive else 'as a running budget',
                           'trainings_recorded': len(trainings), 'hours_recorded': round(self._used_seconds() / 3600, 4),
                           'stopped_early': self.stopped},
                'settings': asdict(self.o.settings),
                'seeds': {'split': self.o.settings.seed, 'folds': self.o.settings.seed, 'training': self.o.settings.seed,
                          'fold_inner_validation': f'{self.o.settings.seed} + fold index (evaluators/video/validation.py)',
                          'refit_validation': None},
                'code': {**(self.code or {}), 'candidates_from_other_commits': self._other_commits(entries)},
                'warnings': self.warnings,
                'split': manifest['split'], 'metric': self.metric, 'higher_is_better': METRICS[self.mode][self.metric] > 0,
                'counts': counts, 'weights_licences': self._licences(backbones),
                'notes': [WINNERS_CURSE,
                          'Cross-validation scores are on development rows only; the test score is on the sealed rows, which no '
                          'candidate and no ranking step read. Neither is a promise of accuracy on other data.',
                          'Fold models train on (k-1)/k of the development rows, with 10% of those held out for checkpoint selection.',
                          *notes_extra]}

    @staticmethod
    def _licences(backbones: list) -> dict:
        try:
            from src.video.feature_backbones import run_weight_licence
        except Exception as exc:  # the record should not cost the search
            return {backbone: f'unknown ({exc})' for backbone in backbones}
        return {backbone: run_weight_licence(backbone) for backbone in backbones}

    def _summary(self, selection: dict, table: pd.DataFrame) -> None:
        metric, winner, counts = self.metric, selection['winner'], selection['counts']
        self.say('')
        others = sum(1 for entry in self.ledger.load().values() if _finished(entry) and entry.get('protocol', 'cv') != 'cv')
        self.say(f'Model selection ({self.strategy.name}): ' + ', '.join(f'{v} {k.replace("_", " ")}' for k, v in counts.items())
                 + (f'; plus {others} screening or tuning runs on other protocols (unranked)' if others > 0 else '') + '.')
        interval = winner.get(f'oof_{metric}_interval') or {}
        sd = winner[f'cv_{metric}_fold_sd']
        self.say(f'Winner: {winner["label"]}')
        self.say(f'  cross-validation {metric}: {winner[f"cv_{metric}_mean"]:.4f}' + (f' (fold SD {sd:.4f})' if sd is not None else '')
                 + (f', pooled out-of-fold {interval.get("level", 0.95):.0%} interval [{interval["low"]:.4f}, {interval["high"]:.4f}] '
                    f'by {interval.get("unit")}' if interval.get('low') is not None else ''))
        self.say(f'  why: {selection["reason"]}')
        test = selection.get('test')
        if test:
            point, bounds = test['metrics'].get(metric), test['intervals'].get(metric, {})
            self.say(f'  sealed test ({test["n"]} clips, scored once): {metric} {point:.4f}'
                     + (f' [{bounds["low"]:.4f}, {bounds["high"]:.4f}]' if bounds.get('low') is not None else ''))
        else:
            self.say(f'  sealed test: not scored ({selection["final"].get("status")}: {selection["final"].get("reason", "")})')
        self.say(f'  {WINNERS_CURSE}')
        if self.strategy.adaptive:
            self.say(f'  {STAGE_BIAS}')
            if self.stopped:
                self.say(f'  The search was {self.stopped}.')
        for warning in selection.get('warnings', []):
            self.say(f'  Warning: {warning}')
        files = [self.out / LEADERBOARD_FILE, self.out / SELECTION_FILE, self.out / LEDGER_FILE]
        if self.context and self.context.decisions:
            files.append(self.out / STAGES_FILE)
        self.say('Files: ' + ', '.join(map(str, files)))

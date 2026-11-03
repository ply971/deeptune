"""Simulated searches with known answers: the regret study of R3b, and the tests of every strategy.

`SurfaceRunner` trains nothing. It stands in for deeptune.py --cv_only (and for the feature probe and the final refit):
it reads the development file it is given, and writes the same files through the real
`evaluators.video.validation.run_validation`, with predictions right on a share of the rows equal to a known "quality" of
the candidate: `surface(config)`. So the engine, the ledger, the fold check, the ranking and the sealed test all run as in
production, and the true best candidate is known. The share is laid out with a low-discrepancy sequence, so the measured
accuracy is the quality to within about one row.

`regret_study` runs exhaustive (the reference: its CV score of every candidate), then each other strategy, on the same
data and folds, and reports how far each strategy's pick falls from the true best, by exhaustive's CV, and how many
trainings it used. `INTERACTION_SURFACE` has an interaction between what trains and the learning rate, so a search that
tunes one group at a time from the base settings goes wrong unless it re-checks combinations (smart's stage 3).

Nothing here is used by a real search.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from handlers.model_selection.checks import Footprint
from handlers.model_selection.grid import DEFAULT, Candidate
from handlers.model_selection.runner import RunResult

PHI = 0.6180339887


def _flag(argv, name, default=None):
    argv = [str(a) for a in argv]
    return argv[argv.index(name) + 1] if name in argv else default


def _content(path) -> str:
    """The development file's content hash: runs of one search folder can be reused by another on the same rows."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_from_argv(argv: list) -> Candidate:
    """The candidate a deeptune.py command line trains (the inverse of grid.candidate_argv, for the axes)."""
    argv = [str(a) for a in argv]
    if '--freeze-backbone' in argv:
        mode = 'frozen'
    elif '--use-peft' in argv:
        mode = f'lora:{_flag(argv, "--lora_rank") or _flag(argv, "--embed_size")}'
    elif '--trainable_params' in argv:
        mode = f'partial:{_flag(argv, "--trainable_params")}'
    else:
        mode = 'full'
    augmentation = 'off'
    if '--video_augment' in argv:
        augmentation = 'aug_mix' if '--mixup_alpha' in argv else 'aug'
        augmentation += '_flip' if '--video_flip' in argv else ''
    frames = _flag(argv, '--num_frames')
    return Candidate(backbone=_flag(argv, '--model_version'), video_arch=_flag(argv, '--video_arch', DEFAULT),
                     pooling=_flag(argv, '--pooling', DEFAULT), train_mode=mode, learning_rate=float(_flag(argv, '--learning_rate')),
                     num_frames=int(frames) if frames is not None else DEFAULT, augmentation=augmentation)


def jitter(*parts, scale: float = 0.02) -> float:
    """A fixed pseudo-random number in [-scale, scale] for these parts."""
    value = int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little') / 2 ** 32
    return (2 * value - 1) * scale


def interaction_surface(seed: int = 0) -> Callable[[Candidate], float]:
    """Known qualities with an interaction: full fine-tuning is best at a low learning rate and worst at a high one, and the
    base learning rate is the high one. Augmentation helps where weights train (it does not apply to frozen backbones)."""
    backbone = {'resnet_tiny': 0.55, 'siglip_tiny': 0.50, 'vjepa2_tiny': 0.45, 'dinov3_tiny': 0.40, 'siglip2_tiny': 0.47}
    mode_rate = {('frozen', 1e-3): 0.0, ('frozen', 1e-4): -0.05, ('lora:8', 1e-3): -0.03, ('lora:8', 1e-4): 0.08,
                 ('full', 1e-3): -0.20, ('full', 1e-4): 0.20}
    augmentation = {'off': 0.0, 'aug': 0.04, 'aug_mix': 0.06}

    def surface(c: Candidate) -> float:
        q = backbone.get(c.backbone, 0.4) + mode_rate.get((c.train_mode, c.learning_rate), -0.1)
        q += augmentation.get(c.augmentation, 0.0) if c.train_mode != 'frozen' else 0.0
        if c.num_frames not in (DEFAULT, 8):
            q += 0.02 if int(c.num_frames) > 8 else -0.02
        return float(np.clip(q + jitter('surface', seed, json.dumps(c.config(), sort_keys=True)), 0.05, 0.98))
    return surface


@dataclass
class SurfaceRunner:
    """A runner (and probe runner) whose candidates score `surface(candidate)`. Screening runs (fewer epochs) and probes
    score with extra noise, so cheap screens are informative but not exact."""
    surface: Callable[[Candidate], float]
    seconds: float = 60.0
    ci_boot: int = 30
    screen_noise: float = 0.04
    calls: list = field(default_factory=list)
    reads: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    def _quality(self, candidate: Candidate, cheap: str = '') -> float:
        q = self.surface(candidate)
        if cheap:
            q += jitter(cheap, json.dumps(candidate.config(), sort_keys=True), scale=self.screen_noise)
        return float(np.clip(q, 0.02, 0.99))

    @staticmethod
    def _correct(n: int, quality: float, salt: float) -> np.ndarray:
        return ((np.arange(n) * PHI + salt) % 1.0) < quality

    @staticmethod
    def _probabilities(labels, correct, classes):
        predicted = np.where(correct, labels, (labels + 1) % classes)
        scores = np.full((len(labels), classes), 0.1 / (classes - 1))
        scores[np.arange(len(labels)), predicted] = 0.9
        return scores

    def _cross_validate(self, dev: Path, out: Path, quality: float, salt: float, k: int, seed: int, group: Optional[str]):
        from evaluators.video.validation import FoldScores, run_validation
        frame = pd.read_parquet(dev)
        self.reads.append(str(Path(dev).resolve()))
        codes, names = pd.factorize(frame['labels'], sort=True)
        right = self._correct(len(frame), quality, salt)

        def fit_predict(train_rows, val_rows, eval_rows, name):
            return FoldScores(self._probabilities(codes[eval_rows], right[eval_rows], len(names)))
        groups = frame[group].tolist() if group and group in frame else None
        run_validation('kfold', codes, fit_predict, out / 'sim-x' / 'validation_kfold', mode='cls', ids=frame['videos'].tolist(),
                       groups=groups, k=k, seed=seed, ci_boot=self.ci_boot)

    def _cached(self, signature: str, run_dir: Path, make: Callable[[], None]) -> None:
        if signature in self._cache and Path(self._cache[signature]).is_dir():
            shutil.copytree(self._cache[signature], run_dir / 'sim-x')
            return
        make()
        self._cache[signature] = str(run_dir / 'sim-x')

    def __call__(self, argv, run_dir, role):
        argv, run_dir = [str(a) for a in argv], Path(run_dir)
        self.calls.append((role, argv))
        candidate = config_from_argv(argv)
        if role == 'final':
            from evaluators.video.report import build_report
            test = pd.read_parquet(_flag(argv, '--test_df'))
            self.reads.append(str(Path(_flag(argv, '--test_df')).resolve()))
            codes, names = pd.factorize(test['labels'], sort=True)
            scores = self._probabilities(codes, self._correct(len(test), self._quality(candidate), 0.3), len(names))
            report = build_report('cls', codes, scores, test['group'].tolist() if 'group' in test else None, n_boot=self.ci_boot)
            folder = run_dir / 'sim-x' / 'eval_output_sim'
            folder.mkdir(parents=True)
            (folder / 'evaluation_report.json').write_text(json.dumps(report, default=float), encoding='utf-8')
            return RunResult('ok', self.seconds)
        epochs, k = int(_flag(argv, '--num_epochs')), int(_flag(argv, '--k'))
        cheap = f'epochs{epochs}' if role == 'pilot' or epochs < self.full_epochs(argv) else ''
        quality = self._quality(candidate, cheap)
        signature = json.dumps([_content(_flag(argv, '--df')), candidate.config(), epochs, k, _flag(argv, '--seed'), cheap], sort_keys=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._cached(signature, run_dir, lambda: self._cross_validate(Path(_flag(argv, '--df')), run_dir, quality, 0.0, k,
                                                                     int(_flag(argv, '--seed')), _flag(argv, '--grouper', 'group')))
        return RunResult('ok', self.seconds * epochs / max(1, self.full_epochs(argv)))

    full_epochs_value: int = 0

    def full_epochs(self, argv) -> int:
        return self.full_epochs_value or int(_flag(argv, '--num_epochs'))

    @staticmethod
    def probe_problem(candidate, settings):
        """The simulated probe takes any frozen backbone (the real one: probe.probe_problem)."""
        return None if candidate.train_mode == 'frozen' else f'its backbone trains ({candidate.train_mode})'

    def probe(self, job, run_dir, info):
        """The cached-feature probe's stand-in: the frozen candidate's quality, with screening noise, on the same folds."""
        run_dir = Path(run_dir)
        self.calls.append(('probe', [job.candidate.label()]))
        settings = info['settings']
        quality = self._quality(job.candidate, 'probe')
        signature = json.dumps(['probe', _content(info['dev']), job.candidate.config(), settings.k, settings.seed], sort_keys=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._cached(signature, run_dir, lambda: self._cross_validate(Path(info['dev']), run_dir, quality, 0.0, settings.k, settings.seed,
                                                                     settings.grouper or 'group'))
        return RunResult('ok', self.seconds / 10)


def quick_check(argv):
    """Accept every command line and give the footprint function what it needs, without deeptune.py's parser (fast)."""
    frames = _flag(argv, '--num_frames')
    return types.SimpleNamespace(model_version=_flag(argv, '--model_version'), num_frames=int(frames) if frames else 8,
                                 freeze_backbone='--freeze-backbone' in argv, use_peft='--use-peft' in argv), None


SIM_PARAMS = {'resnet_tiny': 300_000, 'siglip_tiny': 2_000_000, 'vjepa2_tiny': 800_000, 'dinov3_tiny': 600_000, 'siglip2_tiny': 1_500_000}


def sim_footprint(args, num_classes, measure):
    total = SIM_PARAMS.get(args.model_version, 1_000_000)
    trainable = 20_000 if args.freeze_backbone else total // 10 if args.use_peft else total
    return Footprint(total, trainable, activation_bytes_per_clip=total * 10, forward_flops_per_clip=total * 100)


def make_dataset(root: Path, per_class: int = 15, classes=('left', 'right', 'up'), seed: int = 0) -> Path:
    """Clip paths need not exist: nothing here decodes video. Groups of 3 clips."""
    root.mkdir(parents=True, exist_ok=True)
    rows = [{'videos': str((root / f'{label}_{i}.mp4').resolve()), 'labels': label, 'group': f'{label}-src{i // 3}'}
            for label in classes for i in range(per_class)]
    path = root / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def run_search(argv: list, runner: SurfaceRunner, quiet: bool = True) -> int:
    import select_model
    from handlers.model_selection.search import ModelSearch
    parser = select_model.build_parser()
    args = parser.parse_args(argv)
    options = select_model.options_from_args(args)
    search = ModelSearch(options, runner=runner, footprint_fn=sim_footprint, check_fn=quick_check,
                         say=(lambda text: None) if quiet else None)
    return search.run()


def _ledger(out: Path) -> list:
    path = Path(out) / 'ledger.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.is_file() else []


DEFAULT_VARIANTS = {'smart': ['--search', 'smart'], 'random': ['--search', 'random'], 'bayes': ['--search', 'bayes']}


def regret_study(work: Path, seeds, grid_args: list, variants: Optional[dict] = None, budget_from: str = 'smart',
                 surface_fn=interaction_surface, k: int = 3) -> list:
    """One row per (seed, variant): its pick, its regret against exhaustive's best (by exhaustive's CV of every candidate),
    its trainings. `variants` maps a name to select_model.py flags. Variants that sample (random, bayes) without their own
    --n_trials get the trainings `budget_from` used on that seed, screens included (a matched budget, generous to them, as
    a probe costs less than a cross-validation)."""
    variants = dict(variants or DEFAULT_VARIANTS)
    from handlers.model_selection.ranking import fold_stats
    rows = []
    for seed in seeds:
        root = Path(work) / f'seed{seed}'
        data = make_dataset(root / 'data', seed=seed)
        surface = surface_fn(seed)
        runner = SurfaceRunner(surface)
        common = ['--df', str(data), '--k', str(k), '--epochs', '4', '--boot', '30', '--gpu_memory_gb', 'none', '--no_pilot',
                  '--profile_clips', '0', '--seed', str(seed), '--yes', *grid_args]
        runner.full_epochs_value = 4
        assert run_search(['--search', 'exhaustive', '--out', str(root / 'exhaustive'), *common], runner) == 0
        truth = {json.dumps(e['config'], sort_keys=True): fold_stats(e, 'accuracy')[0]
                 for e in _ledger(root / 'exhaustive') if e['status'] == 'done' and e.get('protocol', 'cv') == 'cv'}
        best_config = max(truth, key=truth.get)
        rows.append({'seed': seed, 'strategy': 'exhaustive', 'pick': best_config, 'pick_cv': truth[best_config], 'best_cv': truth[best_config],
                     'regret': 0.0, 'cv_runs': len(truth), 'screen_runs': 0, 'true_quality_regret': 0.0,
                     'picked_true_best': True})
        budget = None
        for strategy, flags in variants.items():
            out = root / strategy
            argv = [*flags, '--out', str(out), *common]
            sampled = any(flags[i:i + 2] in (['--search', 'random'], ['--search', 'bayes']) for i in range(len(flags)))
            if sampled and budget is not None and '--n_trials' not in argv:
                argv += ['--n_trials', str(budget)]
            assert run_search(argv, runner) == 0, strategy
            selection = json.loads((out / 'selection.json').read_text(encoding='utf-8'))
            pick = json.dumps(selection['winner']['config'], sort_keys=True)
            entries = [e for e in _ledger(out) if e.get('stage') != 'checks' and e['status'] in ('done', 'failed', 'did_not_fit')]
            cv_runs = len({e['key'] for e in entries if e.get('protocol', 'cv') == 'cv'})
            screen_runs = len({e['key'] for e in entries if e.get('protocol', 'cv') != 'cv'})
            picked = Candidate(**selection['winner']['config'])
            best = Candidate(**json.loads(best_config))
            rows.append({'seed': seed, 'strategy': strategy, 'pick': pick, 'pick_cv': truth[pick], 'best_cv': truth[best_config],
                         'regret': truth[best_config] - truth[pick], 'cv_runs': cv_runs, 'screen_runs': screen_runs,
                         'true_quality_regret': surface(best) - surface(picked), 'picked_true_best': pick == best_config})
            if strategy == budget_from:
                budget = cv_runs + screen_runs
    return rows


def summarize(rows: list) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    return table.groupby('strategy', sort=False).agg(seeds=('seed', 'nunique'), mean_regret=('regret', 'mean'), max_regret=('regret', 'max'),
                                                     picked_best=('picked_true_best', 'mean'), mean_cv_runs=('cv_runs', 'mean'),
                                                     mean_screen_runs=('screen_runs', 'mean')).reset_index()

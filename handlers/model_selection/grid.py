"""The search grid of select_model.py: axes, candidates, and each candidate's deeptune.py command line.

A candidate is one combination of the axes below. The grid is their Cartesian product, so its size is the product of the
axis lengths; `Grid.formula` prints it. Any axis may hold the value ``default``, which leaves that deeptune.py flag out.

Train modes (what trains):
    frozen        --freeze-backbone: only the new head trains
    full          every parameter trains (V-JEPA 2 and VideoMAE, frozen by default, get --unfreeze_backbone)
    lora:R        --use-peft at LoRA rank R (--lora_rank R on native video models; on 2D backbones the rank is also the
                  head's width, so it is passed as --embed_size R, as deeptune.py documents)
    partial:N     --trainable_params N: the head, then the top backbone layers while they fit in N parameters (A1)
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from trainers.param_count import parse_param_count

DEFAULT = 'default'
TRAIN_MODE_FORMS = 'frozen, full, lora:R or partial:COUNT'
# Axis name in the grid file and on the command line -> Candidate field.
AXES = {'backbones': 'backbone', 'video_archs': 'video_arch', 'poolings': 'pooling', 'train_modes': 'train_mode',
        'learning_rates': 'learning_rate', 'num_frames': 'num_frames'}
DEFAULT_AXES = {'backbones': None, 'video_archs': [DEFAULT], 'poolings': [DEFAULT], 'train_modes': ['frozen'],
                'learning_rates': [1e-4], 'num_frames': [DEFAULT]}


def parse_train_mode(text: str) -> tuple[str, Optional[int]]:
    """'frozen' -> ('frozen', None); 'lora:8' -> ('lora', 8); 'partial:100k' -> ('partial', 100000)."""
    name, _, value = str(text).strip().partition(':')
    if name in ('frozen', 'full') and not value:
        return name, None
    if name == 'lora' and value:
        rank = int(value)
        if rank < 1:
            raise ValueError(f'lora rank must be at least 1, got {rank}.')
        return name, rank
    if name == 'partial' and value:
        return name, parse_param_count(value)
    raise ValueError(f'Unknown train mode {text!r}; use {TRAIN_MODE_FORMS}.')


def _learning_rate(value) -> float:
    rate = float(value)
    if not rate > 0:
        raise ValueError(f'Learning rates must be above 0, got {value!r}.')
    return rate


def _frames(value):
    if value == DEFAULT:
        return DEFAULT
    frames = int(value)
    if frames < 1:
        raise ValueError(f'num_frames must be at least 1, got {value!r}.')
    return frames


@dataclass(frozen=True)
class Candidate:
    backbone: str
    video_arch: str = DEFAULT
    pooling: str = DEFAULT
    train_mode: str = 'frozen'
    learning_rate: float = 1e-4
    num_frames: object = DEFAULT

    def config(self) -> dict:
        """Canonical and JSON-ready: the identity of this candidate in the ledger."""
        return asdict(self)

    def label(self) -> str:
        parts = [self.backbone, self.video_arch, self.pooling, self.train_mode, f'lr={self.learning_rate:g}',
                 f'{self.num_frames} frames' if self.num_frames != DEFAULT else DEFAULT]
        return ' | '.join(str(part) for part in parts)

    def build_key(self) -> str:
        """Everything that shapes the model, so candidates differing only in learning rate share one preflight build."""
        config = self.config()
        config.pop('learning_rate')
        return json.dumps(config, sort_keys=True)


@dataclass
class Grid:
    backbones: list
    video_archs: list = field(default_factory=lambda: [DEFAULT])
    poolings: list = field(default_factory=lambda: [DEFAULT])
    train_modes: list = field(default_factory=lambda: ['frozen'])
    learning_rates: list = field(default_factory=lambda: [1e-4])
    num_frames: list = field(default_factory=lambda: [DEFAULT])

    def __post_init__(self):
        for name in AXES:
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f'Grid axis {name} needs at least one value.')
            if len(set(map(str, values))) != len(values):
                raise ValueError(f'Grid axis {name} repeats a value: {values}.')
        for mode in self.train_modes:
            parse_train_mode(mode)
        self.learning_rates = [_learning_rate(v) for v in self.learning_rates]
        self.num_frames = [_frames(v) for v in self.num_frames]
        self.backbones, self.video_archs, self.poolings, self.train_modes = (
            [str(v) for v in values] for values in (self.backbones, self.video_archs, self.poolings, self.train_modes))

    @classmethod
    def from_sources(cls, grid_file: Optional[Path] = None, **flags) -> 'Grid':
        """Axes from a JSON file (keys as in AXES), then from flags; a flag given replaces the file's axis."""
        values = {}
        if grid_file is not None:
            loaded = json.loads(Path(grid_file).read_text(encoding='utf-8'))
            unknown = sorted(set(loaded) - set(AXES))
            if unknown:
                raise ValueError(f'{grid_file}: unknown grid keys {unknown}; the axes are {sorted(AXES)}.')
            values.update({key: value if isinstance(value, list) else [value] for key, value in loaded.items()})
        values.update({key: value for key, value in flags.items() if value is not None})
        if not values.get('backbones'):
            raise ValueError('The grid needs backbones: --backbones, or "backbones" in --grid.')
        return cls(**values)

    def to_dict(self) -> dict:
        return {name: list(getattr(self, name)) for name in AXES}

    def size(self) -> int:
        size = 1
        for name in AXES:
            size *= len(getattr(self, name))
        return size

    def formula(self) -> str:
        return ' x '.join(f'{len(getattr(self, name))} {name}' for name in AXES) + f' = {self.size()} candidates'

    def expand(self) -> list[Candidate]:
        """Every combination, in a fixed order (backbones vary slowest)."""
        axes = [getattr(self, name) for name in AXES]
        return [Candidate(*combination) for combination in itertools.product(*axes)]


@dataclass
class RunSettings:
    """What every candidate run shares. `training()` is the part that changes a trained model, kept in the ledger key."""
    mode: str = 'cls'
    num_classes: Optional[int] = None
    target: str = 'labels'
    grouper: Optional[str] = None
    k: int = 5
    seed: int = 42
    epochs: int = 10
    early_stopping_patience: int = 3
    batch_size: int = 16
    num_workers: int = 0
    boot: int = 2000
    accept_weights_licence: list = field(default_factory=list)
    deeptune_args: list = field(default_factory=list)

    def training(self) -> dict:
        settings = asdict(self)
        settings.pop('boot')  # bootstrap resamples change intervals, not models
        return settings


# deeptune.py destinations that --deeptune_args may not set. The data, the split, the folds and the seed decide what the
# sealed test means; the rest are set per candidate by the grid axes or by select_model.py's own flags, so an extra flag
# would silently replace them (argparse keeps the last value).
RESERVED_DESTS = {
    'df': 'select_model gives every candidate the development rows only', 'test_df': 'only the final refit gets the sealed test file',
    'cv_only': 'set per run', 'validation': 'set per run', 'k': 'use --k', 'n_boot': 'set per run', 'seed': 'use --seed',
    'fixed_seed': 'set per run', 'raw_data': 'use --raw-data on select_model', 'modality': 'video only', 'out': 'set per run',
    'mode': 'use --mode', 'model_version': 'a grid axis (--backbones)', 'config': 'a config file could set any of these',
    'video_arch': 'a grid axis (--video_archs)', 'pooling': 'a grid axis (--poolings)', 'num_frames': 'a grid axis (--num_frames)',
    'learning_rate': 'a grid axis (--learning_rates)', 'freeze_backbone': 'a grid axis (--train_modes)',
    'use_peft': 'a grid axis (--train_modes)', 'lora_rank': 'a grid axis (--train_modes)', 'embed_size': 'set by --train_modes lora:R',
    'trainable_params': 'a grid axis (--train_modes partial:N)', 'unfreeze_backbone': 'a grid axis (--train_modes full)',
    'num_epochs': 'use --epochs', 'batch_size': 'use --batch_size', 'num_workers': 'use --num_workers',
    'early_stopping_patience': 'use --early_stopping_patience', 'target': 'use --target', 'num_classes': 'use --num_classes',
    'grouper': 'use --grouper', 'accept_weights_licence': 'use --accept_weights_licence', 'report_boot': 'use --boot',
}
_NUMBER = re.compile(r'^-\d+(\.\d*)?([eE][-+]?\d+)?$|^-\.\d+$')


def _deeptune_parser():
    import contextlib
    import io
    from cli import DeepTuneVisionOptions
    from utils import RunType
    with contextlib.redirect_stdout(io.StringIO()):
        return DeepTuneVisionOptions(RunType.ONECALL, args=['--modality', 'video', '--df', 'x.parquet', '--out', 'x']).parser


def extra_args_problems(extra: list) -> list[str]:
    """Why these extra deeptune.py words cannot be passed to every run. Each word that starts with a dash is matched to a
    deeptune.py flag the way argparse matches it (exact, else a unique prefix), so `--d` or `--see` are caught too."""
    parser = _deeptune_parser()
    options = parser._option_string_actions  # argparse's own table of option strings
    problems = []
    for word in map(str, extra):
        if not word.startswith('-') or _NUMBER.match(word):
            continue
        name = word.split('=', 1)[0]
        if ' ' in name:
            problems.append(f'{word!r}: give each flag and value as separate words after --deeptune_args, e.g. --deeptune_args --precision bf16.')
            continue
        action = options.get(name)
        if action is None:
            matches = {options[s] for s in options if s.startswith(name) and name.startswith('--')} if parser.allow_abbrev else set()
            if len(matches) != 1:
                problems.append(f'{word!r} is ' + ('an ambiguous abbreviation of deeptune.py flags.' if matches else 'not a deeptune.py flag.'))
                continue
            action = matches.pop()
        if action.dest in RESERVED_DESTS:
            problems.append(f'{word!r} sets deeptune.py\'s {action.dest}, which select_model.py controls: {RESERVED_DESTS[action.dest]}.')
    return problems


def train_mode_flags(train_mode: str, backbone: str) -> list[str]:
    """The deeptune.py flags of a train mode on this backbone. A translation only: deeptune.py checks the result."""
    from handlers.video_backbones import hub_backbone
    name, value = parse_train_mode(train_mode)
    if name == 'frozen':
        return ['--freeze-backbone']
    if name == 'full':
        info = hub_backbone(backbone)
        return ['--unfreeze_backbone'] if info is not None and info.frozen_by_default else []
    if name == 'partial':
        return ['--trainable_params', str(value)]
    from utils import is_native_video_architecture  # imports torch, so only here
    if is_native_video_architecture(backbone):
        return ['--use-peft', '--lora_rank', str(value)]
    return ['--use-peft', '--embed_size', str(value)]


def candidate_argv(candidate: Candidate, settings: RunSettings, df, out, role: str = 'cv', test_df=None,
                   epochs: Optional[int] = None, k: Optional[int] = None) -> list[str]:
    """deeptune.py's command line for one candidate.

    role 'cv': grouped k-fold on every row of `df` (--cv_only), with no test split and no final model.
    role 'final': train on `df` and score once on `test_df` (--test_df). Only the search's last step uses it.
    `epochs` and `k` override the settings (the timing pilot runs 1 epoch on 2 folds)."""
    if role not in ('cv', 'final'):
        raise ValueError(f'Unknown role {role!r}.')
    epochs = settings.epochs if epochs is None else epochs
    k = settings.k if k is None else k
    argv = ['--modality', 'video', '--df', str(df), '--out', str(out), '--mode', settings.mode,
            '--model_version', candidate.backbone, '--num_epochs', str(epochs), '--learning_rate', repr(float(candidate.learning_rate)),
            '--batch_size', str(settings.batch_size), '--num_workers', str(settings.num_workers),
            '--fixed-seed', '--seed', str(settings.seed)]
    if settings.early_stopping_patience and epochs > 1:
        argv += ['--early_stopping_patience', str(settings.early_stopping_patience)]
    if settings.target != 'labels':
        argv += ['--target', settings.target]
    if settings.num_classes is not None:
        argv += ['--num_classes', str(settings.num_classes)]
    if settings.grouper:
        argv += ['--grouper', settings.grouper]
    if candidate.video_arch != DEFAULT:
        argv += ['--video_arch', candidate.video_arch]
    if candidate.pooling != DEFAULT:
        argv += ['--pooling', candidate.pooling]
    if candidate.num_frames != DEFAULT:
        argv += ['--num_frames', str(candidate.num_frames)]
    argv += train_mode_flags(candidate.train_mode, candidate.backbone)
    if settings.accept_weights_licence:
        argv += ['--accept_weights_licence', *settings.accept_weights_licence]
    argv += list(settings.deeptune_args)
    if role == 'cv':
        argv += ['--validation', 'kfold', '--k', str(k), '--cv_only', '--report_boot', str(settings.boot)]
    else:
        if test_df is None:
            raise ValueError('The final run needs the sealed test file.')
        argv += ['--test_df', str(test_df), '--report_boot', str(settings.boot)]
    return argv

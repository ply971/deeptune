"""Training-basics CLI flags and config files (images and video).

Shared by cli.py (deeptune.py and the train scripts). Torch-free, so the desktop
GUI can import it too (it must never import torch). The training side lives in
trainers/training_basics.py and trainers/vision/trainer.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass
class TrainingOptions:
    """Everything the shared Trainer needs beyond its constructor arguments.

    The defaults are today's behaviour: same loss, same optimizer, same weights.
    """
    precision: str = 'no'
    seed: Optional[int] = None
    resume: Optional[str] = None
    resume_allow_data_change: bool = False
    checkpoint_every: int = 0
    select_best: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    selection_metric: str = 'val_loss'
    lr_scheduler: str = 'none'
    warmup_epochs: float = 0.0
    lr_step_size: Optional[int] = None
    lr_gamma: float = 0.1
    min_lr: float = 0.0
    grad_accum_steps: int = 1
    experiment_logger: str = 'none'
    experiment_name: Optional[str] = None
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mix_prob: float = 1.0
    layer_decay: float = 1.0
    ema_decay: float = 0.0
    freeze_bn: str = 'off'

    @classmethod
    def from_args(cls, args) -> 'TrainingOptions':
        """Read the options recorded on a parsed-arguments object. Only real
        instance attributes count, so an object without these flags (an old
        Namespace, a test double) gets the defaults."""
        recorded = vars(args) if hasattr(args, '__dict__') else {}
        values = {f.name: recorded[f.name] for f in fields(cls) if f.name in recorded}
        if values.get('resume') is not None:
            values['resume'] = str(values['resume'])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def selects_best(self) -> bool:
        return self.select_best or self.early_stopping_patience > 0

    @property
    def checkpoints_or_resumes(self) -> bool:
        return self.checkpoint_every > 0 or self.resume is not None

    def non_default(self) -> list[str]:
        """Names of the options that differ from the defaults."""
        return [f.name for f in fields(self) if getattr(self, f.name) != f.default]

    def issues(self, mode: Optional[str] = None) -> list[str]:
        """Problems with this combination of options, as user-facing messages."""
        found = []
        if self.precision not in PRECISION_CHOICES:
            found.append(f"--precision must be one of {', '.join(PRECISION_CHOICES)}.")
        if self.lr_scheduler not in LR_SCHEDULER_CHOICES:
            found.append(f"--lr_scheduler must be one of {', '.join(LR_SCHEDULER_CHOICES)}.")
        if self.selection_metric not in SELECTION_METRIC_CHOICES:
            found.append(f"--selection_metric must be one of {', '.join(SELECTION_METRIC_CHOICES)}.")
        if self.experiment_logger not in EXPERIMENT_LOGGER_CHOICES:
            found.append(f"--experiment_logger must be one of {', '.join(EXPERIMENT_LOGGER_CHOICES)}.")
        if self.seed is not None and self.seed < 0:
            found.append('--seed must not be negative.')
        if self.freeze_bn not in FREEZE_BN_CHOICES:
            found.append(f"--freeze_bn must be one of {', '.join(FREEZE_BN_CHOICES)}.")
        if self.grad_accum_steps < 1:
            found.append('--grad_accum_steps must be at least 1.')
        if self.checkpoint_every < 0:
            found.append('--checkpoint_every must be 0 (off) or more.')
        if self.early_stopping_patience < 0:
            found.append('--early_stopping_patience must be 0 (off) or more.')
        if self.early_stopping_min_delta < 0:
            found.append('--early_stopping_min_delta must not be negative.')
        if self.warmup_epochs < 0:
            found.append('--warmup_epochs must not be negative.')
        if self.lr_step_size is not None and self.lr_step_size < 1:
            found.append('--lr_step_size must be at least 1.')
        if not 0 < self.lr_gamma <= 1:
            found.append('--lr_gamma must be in (0, 1].')
        if self.min_lr < 0:
            found.append('--min_lr must not be negative.')
        if not 0 <= self.label_smoothing < 1:
            found.append('--label_smoothing must be in [0, 1).')
        if self.mixup_alpha < 0 or self.cutmix_alpha < 0:
            found.append('--mixup_alpha and --cutmix_alpha must not be negative.')
        if not 0 < self.mix_prob <= 1:
            found.append('--mix_prob must be in (0, 1].')
        if not 0 < self.layer_decay <= 1:
            found.append('--layer_decay must be in (0, 1]; 1 turns it off.')
        if not 0 <= self.ema_decay < 1:
            found.append('--ema_decay must be in [0, 1); 0 turns it off.')
        if self.resume is not None and not Path(self.resume).is_file():
            found.append(f'--resume: no checkpoint file at {self.resume}.')
        if mode == 'reg':
            if self.label_smoothing > 0:
                found.append('--label_smoothing is for classification; it has no meaning with --mode reg.')
            if self.mixup_alpha > 0 or self.cutmix_alpha > 0:
                found.append('--mixup_alpha / --cutmix_alpha are for classification; they are not available with --mode reg.')
            if self.selection_metric == 'val_accuracy':
                found.append('--selection_metric val_accuracy needs classification; use val_loss with --mode reg.')
        return found

    def validate(self, mode: Optional[str] = None) -> 'TrainingOptions':
        found = self.issues(mode)
        if found:
            raise ValueError(' '.join(found))
        return self


PRECISION_CHOICES = ('no', 'bf16', 'fp16')
LR_SCHEDULER_CHOICES = ('none', 'cosine', 'step')
SELECTION_METRIC_CHOICES = ('val_loss', 'val_accuracy')
EXPERIMENT_LOGGER_CHOICES = ('none', 'mlflow', 'wandb')
FREEZE_BN_CHOICES = ('off', 'on', 'auto')

TRAINING_BASICS_ARG_NAMES = tuple(f.name for f in fields(TrainingOptions))


def add_training_basics_arguments(parser):
    """Mixed precision, resume, early stopping, schedulers and the expert recipe.
    Every default keeps the behaviour from before these flags existed."""
    g = parser.add_argument_group('training basics (images and video)')
    g.add_argument('--precision', choices=PRECISION_CHOICES, default='no',
                   help='Mixed precision on CUDA: bf16 or fp16 (fp16 uses a GradScaler). Default no. On CPU it does nothing and says so in the log.')
    g.add_argument('--seed', type=int, default=None,
                   help="""Seed python, numpy, torch, CUDA and DataLoader workers with this value. Recorded in training_run.json.
Default: as before (--fixed-seed = 42, otherwise a random seed, which is still recorded).
In deeptune.py it also fixes the data split, so changing --seed changes training and not the test set.""")
    g.add_argument('--checkpoint_every', type=int, default=0, metavar='N',
                   help="""Write last_checkpoint.pt every N epochs and after the last one (default 0 = off): model, optimizer, scheduler,
scaler, EMA, epoch, best-so-far and RNG state. Workers are then restarted each epoch (no persistent_workers),
because the random state of a persistent worker cannot be restored.""")
    g.add_argument('--resume', type=str, default=None, metavar='CHECKPOINT',
                   help="""Continue from a last_checkpoint.pt at the start of the next epoch. Use the same flags as the first run, with
--num_epochs as the total. The run stops if the train/val data differ from the checkpoint's.""")
    g.add_argument('--resume_allow_data_change', action='store_true',
                   help='With --resume: continue even if the train/val data differ from the checkpoint (a warm start, not a resume).')
    g.add_argument('--select_best', action='store_true',
                   help="""Keep the weights with the best validation score and reload them after training (best_model_weights.pth).
The validation split then becomes a selection set, and training_run.json and error_model_fit.json say so.""")
    g.add_argument('--early_stopping_patience', type=int, default=0, metavar='EPOCHS',
                   help='Stop after this many epochs without improvement and reload the best weights (implies --select_best). Default 0 = off.')
    g.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                   help='Smallest change in the selection metric that counts as an improvement (default 0).')
    g.add_argument('--selection_metric', choices=SELECTION_METRIC_CHOICES, default='val_loss',
                   help='What --select_best and early stopping compare (default val_loss). Validation loss never includes label smoothing or mixup.')
    g.add_argument('--lr_scheduler', choices=LR_SCHEDULER_CHOICES, default='none',
                   help='Learning-rate schedule, updated at every optimizer step: cosine to --min_lr, or step decay. Default none.')
    g.add_argument('--warmup_epochs', type=float, default=0.0,
                   help='Linear warmup at the start of training, in epochs; fractions are allowed (default 0). Also works with --lr_scheduler none.')
    g.add_argument('--lr_step_size', type=int, default=None, metavar='EPOCHS',
                   help='With --lr_scheduler step: multiply the learning rate by --lr_gamma every this many epochs (default: a third of the epochs).')
    g.add_argument('--lr_gamma', type=float, default=0.1, help='Decay factor for --lr_scheduler step (default 0.1).')
    g.add_argument('--min_lr', type=float, default=0.0, help='Final learning rate for --lr_scheduler cosine (default 0).')
    g.add_argument('--grad_accum_steps', type=int, default=1, metavar='N',
                   help="""Update the weights every N batches (default 1). Gradients are averaged over all samples of the window, and a
shorter last window is still applied. BatchNorm still sees the small batches.""")
    g.add_argument('--freeze_bn', choices=FREEZE_BN_CHOICES, default='off',
                   help="""Backbone BatchNorm during training. off (default): as before, it updates its statistics from each batch.
on: eval mode and frozen parameters, keeping the pretrained statistics.
auto: on when --batch_size is under 16 and the backbone has BatchNorm and pretrained weights; otherwise off.
The cutoff of 16 is a starting guess. The choice is logged and saved in training_run.json.""")
    g.add_argument('--experiment_logger', choices=EXPERIMENT_LOGGER_CHOICES, default='none',
                   help='Also log to MLflow or W&B if already installed (DeepTune does not install them). training_metrics.csv is always written.')
    g.add_argument('--experiment_name', type=str, default=None, help='Run name for --experiment_logger.')
    g.add_argument('--label_smoothing', type=float, default=0.0,
                   help='Label smoothing for the training loss, classification only (default 0).')
    g.add_argument('--mixup_alpha', type=float, default=0.0,
                   help='Mixup Beta(alpha, alpha), classification only (default 0 = off). All frames of a clip share one mixing weight.')
    g.add_argument('--cutmix_alpha', type=float, default=0.0,
                   help='CutMix Beta(alpha, alpha), classification only (default 0 = off). All frames of a clip share one box. With mixup too, each batch picks one.')
    g.add_argument('--mix_prob', type=float, default=1.0, help='Share of batches that get mixup or CutMix (default 1).')
    g.add_argument('--layer_decay', type=float, default=1.0,
                   help="""Layer-wise learning-rate decay: each trainable block one step further from the output gets lr * this value.
Default 1 = off. Depth follows module registration order, which is approximate for models that register out of order.""")
    g.add_argument('--ema_decay', type=float, default=0.0,
                   help='Keep an exponential moving average of the weights, e.g. 0.999 (default 0 = off). Validation, selection and the saved weights then use the average.')
    return g


# --- config files ---------------------------------------------------------
def load_config_file(path) -> dict:
    """Read a JSON or YAML mapping of flag names to values."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'--config: no such file: {path}')
    text = path.read_text(encoding='utf-8')
    suffix = path.suffix.lower()
    if suffix == '.json':
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f'--config: {path} is not valid JSON: {exc}') from exc
    elif suffix in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(f'--config: PyYAML is not installed, so {path} cannot be read. Use a .json config.') from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f'--config: {path} is not valid YAML: {exc}') from exc
    else:
        raise ValueError(f'--config: use a .json, .yaml or .yml file, not {path.name}.')
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f'--config: {path} must hold a mapping of flag names to values.')
    return data


def _convert_config_value(action, key, value):
    if isinstance(value, dict):
        raise ValueError(f'--config: {key} must not be a mapping.')
    if action.nargs in ('+', '*') or (isinstance(action.nargs, int) and action.nargs > 0):
        if not isinstance(value, list):
            value = [value]
        return [_convert_scalar(action, key, item) for item in value]
    if isinstance(value, list):
        raise ValueError(f'--config: {key} takes one value, not a list.')
    return _convert_scalar(action, key, value)


def _convert_scalar(action, key, value):
    if action.nargs == 0:  # store_true and friends
        if not isinstance(value, bool):
            raise ValueError(f'--config: {key} must be true or false, not {value!r}.')
        return value
    if action.type is not None:
        try:
            value = action.type(str(value) if not isinstance(value, str) else value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'--config: invalid value for {key}: {value!r} ({exc}).') from exc
    if action.choices is not None and value not in action.choices:
        raise ValueError(f'--config: {key} must be one of {list(action.choices)}, not {value!r}.')
    return value


def apply_config_defaults(parser, argv=None) -> Optional[str]:
    """If ``--config`` is on the command line, make the file's values the
    parser's defaults, so that flags given on the command line still win.

    Keys are flag names without the dashes (``--freeze-backbone`` is
    ``freeze_backbone`` or ``freeze-backbone``). Values get the same type and
    choice checks as the command line. An unknown key is an error, because a
    silently ignored typo would change an experiment; the exception is a
    training-basics key on a command without those flags, so that one file can
    serve both a train and an eval command. Returns the config path or None.
    """
    from argparse import ArgumentParser
    pre = ArgumentParser(add_help=False)
    pre.add_argument('--config')
    known, _ = pre.parse_known_args(argv)
    if not known.config:
        return None
    try:
        data = load_config_file(known.config)
        by_dest = {action.dest: action for action in parser._actions}
        values = {}
        for raw_key, value in data.items():
            key = str(raw_key).lstrip('-').replace('-', '_')
            action = by_dest.get(key)
            if action is None or key in ('help', 'config'):
                if key in TRAINING_BASICS_ARG_NAMES:
                    continue
                raise ValueError(f'--config: unknown key {raw_key!r} in {known.config}.')
            if value is None:
                continue
            values[key] = _convert_config_value(action, raw_key, value)
            action.required = False
    except ValueError as exc:
        parser.error(str(exc))
    parser.set_defaults(**values)
    return known.config

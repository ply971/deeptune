"""Building blocks of the shared Trainer: mixed precision, RNG capture, LR
schedules, layer-wise LR decay, EMA, mixup/CutMix, data fingerprints, peak
memory and experiment logging. Model-agnostic: nothing here looks at a model
family, only at parameter names and tensor shapes.

Clips are ``[B, T, C, H, W]`` and images ``[B, C, H, W]``. Mixing treats the last
two dimensions as space, so every frame of a clip gets the same weight and box.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from trainers.param_budget import parameter_group


# --- mixed precision -------------------------------------------------------
def resolve_precision(precision: str, device, bf16_supported=None):
    """What ``--precision`` means on this device.

    Returns ``(autocast dtype or None, use GradScaler?, log line)``. Autocast
    runs on CUDA only; on any other device the flag does nothing and says so.
    """
    device_type = getattr(device, 'type', str(device))
    if precision == 'no':
        return None, False, None
    if device_type != 'cuda':
        return None, False, (f'--precision {precision} has no effect on {device_type}: '
                             'training runs in full precision (float32).')
    if precision == 'bf16':
        supported = torch.cuda.is_bf16_supported() if bf16_supported is None else bf16_supported
        if not supported:
            return None, False, ('--precision bf16 is not supported by this GPU: training runs in full '
                                 'precision (float32). Try --precision fp16.')
        return torch.bfloat16, False, 'Mixed precision: bf16 autocast, no GradScaler.'
    if precision == 'fp16':
        return torch.float16, True, 'Mixed precision: fp16 autocast with GradScaler.'
    raise ValueError(f'Unknown precision: {precision}')


def autocast_context(dtype):
    return torch.autocast('cuda', dtype=dtype) if dtype is not None else contextlib.nullcontext()


# --- seeding and RNG state ---------------------------------------------------
def seed_everything(seed: int) -> None:
    """Python, numpy, torch and all CUDA generators, plus PYTHONHASHSEED for child processes."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: python and numpy follow the worker's torch seed."""
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def install_worker_seeding(loader) -> None:
    """Add :func:`seed_worker` to a loader that has workers and no init function of its own."""
    if getattr(loader, 'num_workers', 0) > 0 and getattr(loader, 'worker_init_fn', None) is None:
        loader.worker_init_fn = seed_worker


def get_rng_state() -> dict:
    """All generators as tensors and plain Python values, so that a checkpoint
    can be read back with ``torch.load(weights_only=True)``."""
    kind, key, pos, has_gauss, cached = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': {'kind': str(kind), 'key': torch.from_numpy(key.astype(np.int64)), 'pos': int(pos),
                  'has_gauss': int(has_gauss), 'cached': float(cached)},
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict, warn=print) -> None:
    random.setstate(_as_python_state(state['python']))
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['kind'], numpy_state['key'].numpy().astype(np.uint32), numpy_state['pos'],
                         numpy_state['has_gauss'], numpy_state['cached']))
    torch.set_rng_state(state['torch'].cpu())
    saved = state.get('cuda')
    if torch.cuda.is_available():
        if saved is not None and len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved])
        else:
            warn('The checkpoint holds no CUDA random state for this machine; dropout and other '
                 'GPU randomness will differ from an uninterrupted run.')


def _as_python_state(state):
    version, internal, gauss = state
    return version, tuple(internal), gauss


# --- BatchNorm at small batch sizes ---------------------------------------------
FREEZE_BN_AUTO_BELOW = 16  # a starting guess (P-116), to be checked on real data


def is_pretrained_version(model_version: str) -> bool:
    """False for the 'dummy*' test versions, the only ones DeepTune builds with weights=None."""
    return not str(model_version).startswith('dummy')


def backbone_batchnorm_layers(model: nn.Module) -> list:
    """The BatchNorm layers of the pretrained backbone. A BN in a new head has
    no pretrained statistics to keep, so it stays out."""
    return [module for name, module in model.named_modules()
            if isinstance(module, _BatchNorm) and parameter_group(name + '.') == 'backbone']


def decide_freeze_bn(setting: str, physical_batch, n_layers: int, pretrained: bool) -> tuple[bool, str]:
    """Whether to freeze backbone BatchNorm, and why, as a sentence for the log."""
    if setting == 'off':
        return False, '--freeze_bn off: BatchNorm updates its statistics from each batch.'
    if n_layers == 0:
        return False, f'--freeze_bn {setting}: the backbone has no BatchNorm layers, nothing to freeze.'
    if setting == 'on':
        if not pretrained:
            return True, ('--freeze_bn on: freezing BatchNorm WITHOUT pretrained weights, so its statistics stay at '
                          'their initial values and are never learned.')
        return True, '--freeze_bn on: BatchNorm keeps its pretrained statistics.'
    if not pretrained:
        return False, 'Freeze_bn auto: off, the backbone has no pretrained weights, so the statistics must be learned.'
    if physical_batch is None:
        return False, 'Freeze_bn auto: off, the loader has no fixed batch size to judge.'
    if physical_batch < FREEZE_BN_AUTO_BELOW:
        return True, (f'Freeze_bn auto: on, batch size {physical_batch} is under {FREEZE_BN_AUTO_BELOW}, so BatchNorm '
                      f'keeps its pretrained statistics.')
    return False, f'Freeze_bn auto: off, batch size {physical_batch} is at least {FREEZE_BN_AUTO_BELOW}.'


# --- learning-rate schedules ---------------------------------------------------
def lr_multiplier(update: int, total_updates: int, warmup_updates: int = 0, kind: str = 'none',
                  step_updates: int | None = None, gamma: float = 0.1, min_ratio: float = 0.0) -> float:
    """Factor on the base learning rate for optimizer update number ``update`` (from 0).

    Warmup rises linearly to 1, reached at update ``warmup_updates``. Cosine goes
    from 1 to ``min_ratio`` over the remaining updates. Step multiplies by
    ``gamma`` every ``step_updates`` updates, counted after warmup.
    """
    if warmup_updates > 0 and update < warmup_updates:
        return (update + 1) / warmup_updates
    after = update - warmup_updates
    if kind == 'cosine':
        span = max(1, total_updates - warmup_updates)
        progress = min(1.0, after / span)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    if kind == 'step':
        return gamma ** (after // max(1, step_updates or 1))
    return 1.0


def build_scheduler(optimizer, options, num_epochs: int, updates_per_epoch: int, base_lr: float):
    """A LambdaLR stepped once per optimizer update, or None when nothing is
    scheduled. The multiplier is a pure function of the update count, so the
    scheduler's state is just that count."""
    warmup_updates = int(round(options.warmup_epochs * updates_per_epoch))
    if options.lr_scheduler == 'none' and warmup_updates == 0:
        return None
    total_updates = num_epochs * updates_per_epoch
    if warmup_updates >= total_updates:
        raise ValueError(f'--warmup_epochs {options.warmup_epochs} leaves no epochs to train after warmup '
                         f'(--num_epochs {num_epochs}).')
    step_epochs = options.lr_step_size or max(1, num_epochs // 3)
    step_updates = step_epochs * updates_per_epoch
    min_ratio = options.min_lr / base_lr if base_lr > 0 else 0.0
    kind, gamma = options.lr_scheduler, options.lr_gamma

    def multiplier(update):
        return lr_multiplier(update, total_updates, warmup_updates, kind, step_updates, gamma, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


# --- layer-wise LR decay -------------------------------------------------------
_WRAPPER_PARTS = {'peftmodel', 'base_model', 'model', 'backbone', 'base_layer', 'default'}


def layer_key(name: str) -> str:
    """Block a parameter belongs to: its module path up to the first part with
    a digit, e.g. ``layer2.0`` from ``model.layer2.0.conv1.0.0.weight``,
    ``encoder.layers.encoder_layer_3`` or ``features.3.1``. LoRA adapters count
    as part of the layer they adapt."""
    module_parts = name.split('.')[:-1]
    parts = []
    for part in module_parts:
        if part.startswith('lora_'):
            break
        if part not in _WRAPPER_PARTS:
            parts.append(part)
    if not parts:
        return name
    for index, part in enumerate(parts):
        if any(ch.isdigit() for ch in part):
            end = index + 1
            while end < len(parts) and parts[end].isdigit():
                end += 1
            return '.'.join(parts[:end])
    return '.'.join(parts)


def layer_wise_param_groups(model: nn.Module, base_lr: float, decay: float) -> list[dict]:
    """Optimizer groups with lr = base_lr * decay ** (distance from the output).

    The new head always gets ``base_lr``. The other trainable parameters form
    blocks by :func:`layer_key`, ordered by first appearance in registration
    order (the usual input-to-output order, but approximate for models that
    register out of order). Frozen parameters do not count towards the depth.
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = [p for n, p in trainable if parameter_group(n) == 'head']
    blocks: dict[str, list] = {}
    for name, param in trainable:
        if parameter_group(name) != 'head':
            blocks.setdefault(layer_key(name), []).append(param)
    depth = len(blocks)
    groups = [{'params': params, 'lr': base_lr * decay ** (depth - index), 'layer': key}
              for index, (key, params) in enumerate(blocks.items())]
    if head:
        groups.append({'params': head, 'lr': base_lr, 'layer': 'head'})
    return groups


# --- exponential moving average ----------------------------------------------
class ModelEma:
    """Moving average of the trainable parameters and float buffers. Frozen
    parameters are not copied, which matters on an 8 GB GPU; other buffers
    (e.g. num_batches_tracked) are copied as they are."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.updates = 0
        self.shadow = {name: tensor.detach().clone() for name, tensor in self._tracked(model)}

    @staticmethod
    def _tracked(model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield name, param
        yield from model.named_buffers()

    def current_decay(self) -> float:
        """Ramp-up from Tarvainen & Valpola 2017, so early averages are not
        dominated by the random initial weights in short runs."""
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.current_decay()
        self.updates += 1
        for name, tensor in self._tracked(model):
            shadow = self.shadow[name]
            if tensor.is_floating_point():
                shadow.lerp_(tensor.detach(), 1 - decay)
            else:
                shadow.copy_(tensor)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, tensor in self._tracked(model):
            tensor.copy_(self.shadow[name])

    @contextlib.contextmanager
    def swapped_in(self, model: nn.Module):
        """The model holds the averaged weights inside this block and its own afterwards."""
        backup = {name: tensor.detach().clone() for name, tensor in self._tracked(model)}
        try:
            self.copy_to(model)
            yield
        finally:
            with torch.no_grad():
                for name, tensor in self._tracked(model):
                    tensor.copy_(backup[name])

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'updates': self.updates,
                'shadow': {name: tensor.cpu() for name, tensor in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        missing = set(self.shadow) ^ set(state['shadow'])
        if missing:
            raise ValueError(f'EMA state does not fit this model; {len(missing)} tensors differ, e.g. {sorted(missing)[:3]}')
        self.updates = state['updates']
        for name, tensor in state['shadow'].items():
            self.shadow[name].copy_(tensor)


# --- mixup / CutMix ------------------------------------------------------------
def _sample_beta(alpha: float) -> float:
    return torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha))).sample().item()


def _cutmix_box(height: int, width: int, lam: float):
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    center_y = int(torch.randint(0, height, (1,)).item())
    center_x = int(torch.randint(0, width, (1,)).item())
    y0, y1 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
    x0, x1 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
    return y0, y1, x0, x1


def mix_batch(inputs: torch.Tensor, labels: torch.Tensor, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0,
              mix_prob: float = 1.0):
    """Mixup or CutMix across a batch of images or clips.

    Returns ``(inputs, other labels, weight of the original labels)``. The other
    labels are None when this batch stays unmixed. One weight and one box apply
    to all frames of a clip, so motion inside the clip is kept. All randomness
    comes from torch's global generator, which checkpoints restore.
    """
    if (mixup_alpha <= 0 and cutmix_alpha <= 0) or (mix_prob < 1 and torch.rand(1).item() >= mix_prob):
        return inputs, None, 1.0
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or torch.rand(1).item() < 0.5)
    lam = _sample_beta(cutmix_alpha if use_cutmix else mixup_alpha)
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    if use_cutmix:
        height, width = inputs.shape[-2:]
        y0, y1, x0, x1 = _cutmix_box(height, width, lam)
        mixed = inputs.clone()
        mixed[..., y0:y1, x0:x1] = inputs[permutation][..., y0:y1, x0:x1]
        lam = 1.0 - (y1 - y0) * (x1 - x0) / (height * width)  # the box may be clipped at the border
    else:
        mixed = lam * inputs + (1.0 - lam) * inputs[permutation]
    return mixed, labels[permutation], lam


# --- data identity -----------------------------------------------------------
def data_fingerprint(path) -> str | None:
    """Cheap identity of a data file: its size and first and last MiB. Video
    parquet files can be gigabytes, and a different split changes both the
    rows and the footer. None when there is no such file."""
    path = Path(path)
    if not path.is_file():
        return None
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with open(path, 'rb') as stream:
        digest.update(stream.read(1 << 20))
        if size > 1 << 20:
            stream.seek(max(1 << 20, size - (1 << 20)))
            digest.update(stream.read(1 << 20))
    return digest.hexdigest()


# --- memory ----------------------------------------------------------------------
def peak_memory_mb(device) -> dict:
    """Peak memory since the last :func:`reset_peak_memory`. GPU values are None
    without CUDA, and the process value is None without psutil."""
    use_cuda = getattr(device, 'type', str(device)) == 'cuda'
    peak = {
        'peak_gpu_allocated_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if use_cuda else None,
        'peak_gpu_reserved_mb': torch.cuda.max_memory_reserved() / 2 ** 20 if use_cuda else None,
        'peak_process_rss_mb': None,
    }
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak['peak_process_rss_mb'] = getattr(info, 'peak_wset', info.rss) / 2 ** 20
    except ImportError:
        pass
    return peak


def reset_peak_memory(device) -> None:
    if getattr(device, 'type', str(device)) == 'cuda':
        torch.cuda.reset_peak_memory_stats()


# --- experiment logging ------------------------------------------------------
METRICS_FILENAME = 'training_metrics.csv'


class ExperimentLogger:
    """Per-epoch metrics: always a CSV, plus MLflow or W&B when asked for and
    already installed. A logging failure never stops a training run."""

    def __init__(self, kind: str, output_dir, run_name=None, params=None, rows=None, warn=print):
        self.kind = kind
        self.warn = warn
        self.path = Path(output_dir) / METRICS_FILENAME if output_dir else None
        self.rows = list(rows or [])
        self.backend = None
        self._columns = []
        if kind == 'none':
            return
        try:
            module = importlib.import_module(kind)
        except ImportError:
            warn(f'--experiment_logger {kind}: {kind} is not installed; logging to {METRICS_FILENAME} only.')
            return
        try:
            flat = {key: value for key, value in (params or {}).items() if value is not None}
            if kind == 'mlflow':
                module.start_run(run_name=run_name)
                module.log_params(flat)
            elif kind == 'wandb':
                self._wandb_run = module.init(project='deeptune', name=run_name, config=flat,
                                              dir=str(output_dir) if output_dir else None)
            self.backend = module
        except Exception as exc:  # no login, no server: the run itself must go on
            warn(f'--experiment_logger {kind} could not start ({exc}); logging to {METRICS_FILENAME} only.')

    def log_epoch(self, row: dict) -> None:
        self.rows.append(row)
        self._write_csv()
        if self.backend is None:
            return
        numbers = {k: float(v) for k, v in row.items()
                   if k != 'epoch' and isinstance(v, (int, float)) and not isinstance(v, bool)}
        try:
            if self.kind == 'mlflow':
                self.backend.log_metrics(numbers, step=row['epoch'])
            else:
                self._wandb_run.log(numbers, step=row['epoch'])
        except Exception as exc:
            self.warn(f'--experiment_logger {self.kind} failed ({exc}); logging to {METRICS_FILENAME} only from here on.')
            self.backend = None

    def _write_csv(self) -> None:
        if self.path is None:
            return
        columns = list(self.rows[0])
        for row in self.rows[1:]:
            columns.extend(key for key in row if key not in columns)
        with open(self.path, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self) -> None:
        if self.backend is None:
            return
        try:
            if self.kind == 'mlflow':
                self.backend.end_run()
            else:
                self._wandb_run.finish()
        except Exception as exc:
            self.warn(f'--experiment_logger {self.kind} did not close cleanly: {exc}')
        self.backend = None


# Options that change the training trajectory. If one differs between a
# checkpoint and the resuming run, the run is no longer "the same run continued".
_NOT_TRAJECTORY = {'resume', 'resume_allow_data_change', 'checkpoint_every', 'experiment_logger', 'experiment_name',
                   'seed'}


def trajectory_differences(saved: dict, current: dict) -> list[str]:
    """``name: saved -> current`` for everything that would change how training goes on."""
    found = []
    for key in sorted(set(saved) | set(current)):
        if key not in _NOT_TRAJECTORY and saved.get(key) != current.get(key):
            found.append(f'{key}: {saved.get(key)} -> {current.get(key)}')
    return found


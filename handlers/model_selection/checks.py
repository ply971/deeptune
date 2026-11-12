"""Checking candidates before anything trains.

1. `check_argv`: the candidate's deeptune.py command line goes through deeptune.py's own parser and checks
   (`deeptune.parse_args`), so a refusal carries deeptune.py's reason and no rule is written twice.
2. `measure_footprint` (the preflight): the model is built once on the CPU with the trainer's own builder
   (`trainers.video.train.build_training_model`), which also catches combinations only the builder refuses. It counts
   total and trainable parameters, and one forward pass of one clip measures the activations kept for the backward pass
   and the forward FLOPs. Building loads the backbone's weights, from the local cache or by downloading them once.

The memory estimate (documented, rough, fp32): weights 4 B per parameter, plus gradients and the two Adam moments,
12 B per trainable parameter, plus the measured activations of one clip times the batch size, times 1.2 for the
allocator's overhead, plus 0.5 GB for the CUDA context. Mixed precision and activation reuse make real use lower; cuDNN
workspaces make it higher. A candidate above the GPU's memory is skipped, and an out-of-memory error at run time is still
recorded as "did not fit".
"""
from __future__ import annotations

import contextlib
import copy
import gc
import io
from dataclasses import asdict, dataclass
from typing import Optional

CUDA_CONTEXT_BYTES = 0.5e9
ALLOCATOR_OVERHEAD = 1.2


@dataclass
class Footprint:
    total_params: int
    trainable_params: int
    activation_bytes_per_clip: Optional[int] = None
    forward_flops_per_clip: Optional[int] = None
    note: str = ''

    def memory_bytes(self, batch_size: int) -> float:
        static = 4 * self.total_params + 12 * self.trainable_params
        activations = (self.activation_bytes_per_clip or 0) * batch_size * ALLOCATOR_OVERHEAD
        return static * ALLOCATOR_OVERHEAD + activations + CUDA_CONTEXT_BYTES

    def cost(self, num_frames: int) -> float:
        """Relative cost of one training step, for scaling the timed pilot: forward FLOPs, plus about twice that again
        for the backward pass in proportion to the share of parameters that train. Without FLOPs: parameters x frames."""
        share = self.trainable_params / self.total_params if self.total_params else 1.0
        base = self.forward_flops_per_clip if self.forward_flops_per_clip else self.total_params * max(1, int(num_frames))
        return float(base) * (1 + 2 * share)

    def to_dict(self) -> dict:
        return asdict(self)


def check_argv(argv: list) -> tuple[Optional[object], Optional[str]]:
    """(parsed args, None) when deeptune.py accepts the command line, else (None, its reason)."""
    import deeptune
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            args, _, _ = deeptune.parse_args([str(a) for a in argv])
    except SystemExit:
        text = stderr.getvalue().strip()
        reason = text.split('error:', 1)[1].strip() if 'error:' in text else (text.splitlines()[-1] if text else 'refused')
        return None, ' '.join(reason.split())
    except Exception as exc:  # a checker that raises instead of using the parser
        return None, f'{type(exc).__name__}: {exc}'
    return args, None


def _clip_shape(model_version: str, num_frames: int) -> tuple:
    """[T, C, H, W] of one clip after the dataset's transform for this model."""
    import torch
    from datasets.video_datasets import native_video_transform
    transform = native_video_transform(model_version)
    if transform is not None:
        return tuple(transform(torch.zeros(num_frames, 3, 240, 320, dtype=torch.uint8)).shape)
    from helpers import transformations
    size = next(t.size for t in transformations.transforms if hasattr(t, 'size'))
    size = (size, size) if isinstance(size, int) else tuple(size)
    return (num_frames, 3, *size)


def _measure_forward(model, clip_shape: tuple) -> tuple[Optional[int], Optional[int], str]:
    """(saved-for-backward bytes, forward FLOPs, note) of one clip in training mode. The model is moved to the meta device
    first, so shapes are computed without memory or time: a large model on a long clip would not fit in the laptop's RAM."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    seen, saved = set(), [0]
    try:
        model.to('meta')
        parameters = {id(p) for p in model.parameters()}
    except Exception as exc:
        return None, None, f'activations not measured ({type(exc).__name__}: {exc})'

    def pack(tensor):
        if not isinstance(tensor, torch.nn.Parameter) and id(tensor) not in parameters and id(tensor) not in seen:
            seen.add(id(tensor))  # the graph holds every packed tensor, so ids stay unique during the pass
            saved[0] += tensor.numel() * tensor.element_size()
        return tensor

    model.train()
    clip = torch.zeros(1, *clip_shape, device='meta')
    try:
        with FlopCounterMode(display=False) as flops, torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(clip)
        return saved[0], int(flops.get_total_flops()) or None, ''
    except Exception as exc:  # e.g. two_stream needs its motion input: the parameter part of the estimate still holds
        return None, None, f'activations not measured ({type(exc).__name__}: {exc})'


def measure_footprint(args, num_classes: int, measure_activations: bool = True) -> Footprint:
    """Build the candidate as `train` would and measure it. Raises what the builder raises (the candidate is then refused)."""
    import deeptune
    import torch
    from trainers.param_budget import summarize_parameters
    from trainers.video.train import build_training_model
    args = copy.copy(args)
    added_layers = args.added_layers or deeptune.defaults['added_layers']
    embed_size = args.embed_size if args.embed_size is not None else deeptune.defaults['embed_size']
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_training_model(args, args.model_version, num_classes, added_layers, embed_size, args.freeze_backbone,
                                     args.mode or 'cls', args.use_peft, args.pooling, args.trainable_params)
    try:
        summary = summarize_parameters(model)
        footprint = Footprint(summary['total'], summary['trainable'])
        if measure_activations:
            footprint.activation_bytes_per_clip, footprint.forward_flops_per_clip, footprint.note = _measure_forward(
                model, _clip_shape(args.model_version, args.num_frames))
        return footprint
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

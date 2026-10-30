"""Multi-clip test-time averaging (``--test_clips N``).

With ``num_clips = N > 1`` the video datasets return ``[N, T, C, H, W]`` per
video, so a batch is ``[B, N, T, C, H, W]``. :class:`MultiClipModel` wraps any
DeepTune video model, runs the ``B * N`` clips through it and averages:

* classification: the **softmax probabilities** are averaged and returned
  as ``log(mean p)``. Every consumer applies softmax / argmax / cross-entropy
  to these "logits", and ``softmax(log p) == p``, so TestTrainer, inference
  and the per-prediction error model (A4) all see the averaged probabilities;
* regression: outputs are averaged;
* ``extract_embed=True``: embeddings are averaged.

A ``[B, T, C, H, W]`` input passes straight through, so the wrapper is a
no-op for single-clip runs. ``state_dict`` / ``load_state_dict`` delegate to
the wrapped model, so checkpoints are unchanged.

``clip_disagreement`` is an extra per-video signal (0 = all clips agree):
for classification the mean total-variation distance between each clip's
probabilities and the average; for regression the standard deviation of
the clip outputs. Inference writes it as the ``clip_disagreement`` column.
"""
from __future__ import annotations

import torch
from torch import nn


def multi_clip_scores(model, x: torch.Tensor, mode: str, extract_embed: bool = False):
    """Returns (averaged output, per-clip output ``[B, N, ...]`` or None)."""
    if x.dim() != 6:
        out = model(x, extract_embed=True) if extract_embed else model(x)
        return out, None
    batch, clips = x.shape[:2]
    flat = x.reshape(batch * clips, *x.shape[2:])
    out = model(flat, extract_embed=True) if extract_embed else model(flat)
    out = out.reshape(batch, clips, *out.shape[1:])
    if extract_embed or mode != 'cls':
        return out.mean(dim=1), out
    probs = torch.softmax(out.float(), dim=-1)
    return probs.mean(dim=1), probs


def clip_disagreement(per_clip: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-video disagreement ``[B]`` from per-clip probabilities / outputs ``[B, N, K]``."""
    if mode == 'cls':
        mean = per_clip.mean(dim=1, keepdim=True)
        return 0.5 * (per_clip - mean).abs().sum(dim=-1).mean(dim=1)
    return per_clip.reshape(per_clip.shape[0], per_clip.shape[1], -1).std(dim=1, unbiased=False).mean(dim=-1)


class MultiClipModel(nn.Module):
    def __init__(self, model: nn.Module, mode: str = 'cls'):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x, extract_embed: bool = False):
        if x.dim() != 6:
            return self.model(x, extract_embed=True) if extract_embed else self.model(x)
        averaged, _ = multi_clip_scores(self.model, x, self.mode, extract_embed)
        if extract_embed or self.mode != 'cls':
            return averaged
        return torch.log(averaged.clamp_min(1e-12))

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)


def wrap_for_test_clips(model: nn.Module, test_clips: int, mode: str) -> nn.Module:
    return MultiClipModel(model, mode) if int(test_clips) > 1 else model


def error_model_clips_mismatch(checkpoint_path, test_clips: int, warn=print) -> bool:
    """Warn (and return True) when the run's error_model.json was fitted on a
    different number of clips per video than `test_clips`. Runs from before
    this check recorded nothing and were fitted on one clip."""
    import json
    from evaluators.error_model import find_error_model
    found = find_error_model(checkpoint_path)
    if found is None:
        return False
    try:
        fitted = int((json.loads(found.read_text(encoding='utf-8')).get('fit_info') or {}).get('test_clips') or 1)
    except (OSError, ValueError, TypeError):
        return False
    if fitted == int(test_clips):
        return False
    warn(f'Warning: the error model ({found}) was fitted on {fitted} clip(s) per video, but this run averages '
         f'{test_clips}; expected_error may be miscalibrated. Use --test_clips {fitted} to match it.')
    return True

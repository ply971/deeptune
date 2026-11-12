"""Heads that train on cached snippet features instead of pixels (P-102).

Inputs are what :class:`datasets.feature_dataset.CachedFeatureDataset` yields:
``[B, K, D]`` for cached vectors or ``[B, K, t, h, w, D]`` for token grids (K
snippets per video). Both heads follow DeepTune's model contract,
``forward(x, extract_embed=False)`` returning ``[B, num_classes]`` or
``[B, output_dim]``, so the shared ``Trainer`` trains them unchanged.

* ``linear_probe``: mean of all tokens, then a linear layer.
* ``attentive_probe``: a learnable query reads all tokens through 1-4
  cross-attention blocks, as in the V-JEPA evaluation protocol (Bardes et al.,
  2024; Assran et al., arXiv 2506.09985), then a linear layer. Tokens get a
  learned projection of sin/cos coordinates per grid axis, because CNN tokens
  carry no position of their own; it works for any grid shape.

New heads are added to ``FEATURE_HEADS``; the trainers need no special case.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

_POSITION_BANDS = 4
_POSITION_AXES = 4  # snippet, t, h, w


def _check_task(task_type: str) -> None:
    if task_type not in ('cls', 'reg'):
        raise ValueError("task_type must be 'cls' or 'reg'.")


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """Any ``[B, ..., D]`` -> ``[B, tokens, D]``."""
    if x.ndim < 2:
        raise ValueError(f'Expected features shaped [B, ..., D], got {tuple(x.shape)}.')
    return x.unsqueeze(1) if x.ndim == 2 else x.flatten(1, -2)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, task_type: str = 'cls', output_dim: int = 1):
        super().__init__()
        _check_task(task_type)
        self.head = nn.Linear(input_dim, num_classes if task_type == 'cls' else output_dim)

    def forward(self, x: torch.Tensor, extract_embed: bool = False) -> torch.Tensor:
        pooled = _flatten_tokens(x).mean(dim=1)
        return pooled if extract_embed else self.head(pooled)


class _CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_query = nn.LayerNorm(width)
        self.norm_tokens = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        memory = self.norm_tokens(tokens)
        attended, _ = self.attention(self.norm_query(query), memory, memory, need_weights=False)
        query = query + attended
        return query + self.mlp(self.norm_mlp(query))


class AttentiveProbe(nn.Module):
    """A learnable query cross-attends to every cached token (see the module docstring)."""

    def __init__(self, input_dim: int, num_classes: int, task_type: str = 'cls', output_dim: int = 1, width: int = 256,
                 depth: int = 1, num_heads: int = 4, dropout: float = 0.0, use_position: bool = True):
        super().__init__()
        _check_task(task_type)
        if not 1 <= depth <= 4:
            raise ValueError('The attentive probe uses 1 to 4 cross-attention blocks.')
        if width % num_heads:
            raise ValueError(f'probe width {width} must be divisible by num_heads {num_heads}.')
        self.project = nn.Linear(input_dim, width)
        self.position = nn.Linear(2 * _POSITION_BANDS * _POSITION_AXES, width) if use_position else None
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.blocks = nn.ModuleList(_CrossAttentionBlock(width, num_heads, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, num_classes if task_type == 'cls' else output_dim)

    def _position_features(self, grid: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        """``[tokens, 2 * bands * axes]``; missing axes stay at coordinate 0."""
        axes = [torch.linspace(0, 1, size, device=like.device, dtype=like.dtype) if size > 1
                else torch.zeros(1, device=like.device, dtype=like.dtype) for size in grid]
        axes = axes[-_POSITION_AXES:]
        mesh = torch.meshgrid(*axes, indexing='ij') if axes else ()
        coords = torch.stack([m.reshape(-1) for m in mesh], dim=-1) if axes else torch.zeros(1, 0, device=like.device, dtype=like.dtype)
        padding = torch.zeros(coords.shape[0], _POSITION_AXES - coords.shape[1], device=like.device, dtype=like.dtype)
        coords = torch.cat([coords, padding], dim=-1)
        scales = (2.0 ** torch.arange(_POSITION_BANDS, device=like.device, dtype=like.dtype)) * math.pi
        angles = coords.unsqueeze(-1) * scales
        return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)

    def forward(self, x: torch.Tensor, extract_embed: bool = False) -> torch.Tensor:
        tokens = self.project(_flatten_tokens(x))
        if self.position is not None:
            tokens = tokens + self.position(self._position_features(tuple(x.shape[1:-1]), tokens))
        query = self.query.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, tokens)
        pooled = self.norm(query).squeeze(1)
        return pooled if extract_embed else self.head(pooled)


FEATURE_HEADS = {'linear_probe': LinearProbe, 'attentive_probe': AttentiveProbe}


def build_feature_head(name: str, input_dim: int, num_classes: int, mode: str = 'cls', **options) -> nn.Module:
    """Build a head by name. `options` go to its constructor (width, depth, num_heads, ...)."""
    if name not in FEATURE_HEADS:
        raise ValueError(f'Unknown feature head {name!r}; choose one of {", ".join(FEATURE_HEADS)}.')
    return FEATURE_HEADS[name](input_dim, num_classes, task_type=mode, **options)

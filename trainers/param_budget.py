"""Trainable-parameter accounting and budgets for DeepTune's vision/video models.

Answers "how many of this model's parameters are actually learned?" and lets
the user cap that number (``--trainable_params 100k``):

* ``summarize_parameters`` / ``format_summary`` report total vs. trainable
  parameters, split into the pretrained backbone, LoRA adapters and the new
  head DeepTune adds (``fc1``/``fc2``, frame-pool ``attention``).
* ``apply_top_down_budget`` (full fine-tuning models) freezes everything,
  keeps the head trainable and then unfreezes the backbone from the output
  side down while the budget allows.
* ``choose_lora_rank`` (PEFT models) picks the largest LoRA rank whose
  trainable count fits the budget.

Layout assumed (see ``src/vision/*.py``, ``src/video/*.py``): the pretrained
network lives in ``self.model`` (and, for PEFT classes, is also wrapped as
``self.peftmodel``); everything else with parameters is DeepTune's head.
``VideoFramePoolClassifier`` nests that layout under ``backbone.``.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from trainers.param_count import ParamCountError, param_count_issue, parse_param_count

__all__ = [
    "ParamCountError",
    "parse_param_count",
    "param_count_issue",
    "parameter_group",
    "summarize_parameters",
    "format_summary",
    "apply_top_down_budget",
    "lora_rank_line",
    "choose_lora_rank",
    "build_with_budget",
]

PRETRAINED_PREFIXES = ("model.", "peftmodel.")
WRAPPER_PREFIXES = ("backbone.",)
LORA_MARKER = "lora_"


def parameter_group(name: str) -> str:
    """'lora', 'backbone' or 'head' for a parameter name from named_parameters()."""
    if LORA_MARKER in name:
        return "lora"
    stripped = name
    for prefix in WRAPPER_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    if stripped.startswith(PRETRAINED_PREFIXES):
        return "backbone"
    return "head"


def summarize_parameters(model: nn.Module) -> dict:
    """Total / trainable / frozen parameter counts, plus a per-group breakdown.

    Shared tensors are counted once (``named_parameters`` de-duplicates them).
    """
    breakdown = {group: {"total": 0, "trainable": 0} for group in ("backbone", "lora", "head")}
    for name, param in model.named_parameters():
        group = breakdown[parameter_group(name)]
        group["total"] += param.numel()
        if param.requires_grad:
            group["trainable"] += param.numel()
    total = sum(group["total"] for group in breakdown.values())
    trainable = sum(group["trainable"] for group in breakdown.values())
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": (100.0 * trainable / total) if total else 0.0,
        "breakdown": breakdown,
    }


def format_summary(summary: dict) -> str:
    """One line in the style of peft's ``print_trainable_parameters``."""
    return (
        f"trainable params: {summary['trainable']:,} || all params: {summary['total']:,} "
        f"|| trainable%: {summary['trainable_pct']:.4f}"
    )


def apply_top_down_budget(model: nn.Module, budget: int) -> dict:
    """Make at most ``budget`` parameters trainable, head first, then the
    backbone from the output side down.

    Everything is frozen, the head is always unfrozen, then backbone tensors
    are unfrozen in reverse registration order (closest to the output first)
    while they still fit. It stops at the first tensor that does not fit, so
    the trainable part of the backbone is one contiguous top block. The model's
    shape is not changed.

    Raises ValueError when the head alone is larger than the budget.
    """
    budget = parse_param_count(budget)
    named = list(model.named_parameters())
    head = [(n, p) for n, p in named if parameter_group(n) == "head"]
    backbone = [(n, p) for n, p in named if parameter_group(n) != "head"]

    head_size = sum(p.numel() for _, p in head)
    if head_size > budget:
        raise ValueError(
            f"The new head alone has {head_size:,} parameters, more than the trainable-parameter "
            f"budget of {budget:,}. Raise the budget, or shrink the head with --added_layers 1 "
            f"or a smaller --embed_size."
        )

    for _, param in named:
        param.requires_grad = False
    for _, param in head:
        param.requires_grad = True

    used = head_size
    unfrozen = 0
    for _, param in reversed(backbone):
        if used + param.numel() > budget:
            break
        param.requires_grad = True
        used += param.numel()
        unfrozen += 1

    summary = summarize_parameters(model)
    summary["budget"] = budget
    summary["unfrozen_backbone_tensors"] = unfrozen
    summary["backbone_tensors"] = len(backbone)
    return summary


def _rank_scaled_numel(name: str, shape: tuple, rank: int) -> int:
    """numel / rank for a tensor whose size grows linearly with the LoRA rank,
    else 0. LoRA tensors always scale with r; head tensors scale with it when
    one of their dims is r (in PEFT classes embed_size is both the LoRA rank
    and the head's hidden width)."""
    if rank not in shape:
        return 0
    numel = 1
    for dim in shape:
        numel *= dim
    return numel // rank


def lora_rank_line(model: nn.Module, rank: int) -> tuple[int, int]:
    """(slope, intercept) of trainable-count = slope * r + intercept, read
    from the trainable tensors of a PEFT model built with LoRA rank ``rank``."""
    slope = 0
    trainable = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable += param.numel()
        if parameter_group(name) in ("lora", "head"):
            slope += _rank_scaled_numel(name, tuple(param.shape), rank)
    return slope, trainable - slope * rank


def _largest_rank(slope: int, intercept: int, budget: int) -> int:
    return (budget - intercept) // slope if slope > 0 else 0


def choose_lora_rank(build_fn: Callable[[int], nn.Module], budget: int, start_rank: int) -> tuple[nn.Module, int]:
    """Largest LoRA rank r >= 1 whose model has at most ``budget`` trainable
    parameters. ``build_fn(r)`` must build the PEFT model with rank r.

    The trainable count is affine in r (LoRA A/B tensors and the head's hidden
    layer grow linearly with it), so one build at ``start_rank`` gives the
    slope and intercept (``lora_rank_line``) and the rank is solved for. The
    model is then rebuilt at that rank and its real count compared with the
    prediction. If they differ (a head dimension that merely happened to equal
    r), the line is refitted exactly from the two real builds and the model
    rebuilt once more. The returned model's real trainable count is always
    checked: an over-budget model is never returned (RuntimeError).

    Returns (model, rank). Raises ValueError when even r = 1 does not fit.
    """
    budget = parse_param_count(budget)
    start_rank = max(1, int(start_rank or 1))

    def too_small(minimum: int) -> ValueError:
        return ValueError(
            f"Even LoRA rank 1 needs {minimum:,} trainable parameters, more than the budget of "
            f"{budget:,}. Raise --trainable_params to at least {minimum:,} or use --added_layers 1."
        )

    model = build_fn(start_rank)
    first_count = summarize_parameters(model)["trainable"]
    slope, intercept = lora_rank_line(model, start_rank)

    if slope <= 0:  # nothing depends on r (should not happen for DeepTune's PEFT classes)
        if first_count > budget:
            raise ValueError(
                f"This PEFT model has {first_count:,} trainable parameters whatever the LoRA rank, "
                f"more than the budget of {budget:,}. Raise the budget or use --added_layers 1."
            )
        return model, start_rank

    rank = max(1, _largest_rank(slope, intercept, budget))
    count = first_count
    if rank != start_rank:
        model = build_fn(rank)
        count = summarize_parameters(model)["trainable"]
        if count != slope * rank + intercept:
            # The shape-based line was off: refit it exactly from the two real builds.
            slope = (count - first_count) // (rank - start_rank)
            intercept = count - slope * rank
            refit = max(1, _largest_rank(slope, intercept, budget)) if slope > 0 else rank
            if refit != rank:
                rank = refit
                model = build_fn(rank)
                count = summarize_parameters(model)["trainable"]

    if count > budget:
        if rank == 1:
            raise too_small(count)
        raise RuntimeError(
            f"Could not find a LoRA rank within the budget of {budget:,}: the model built at rank "
            f"{rank} has {count:,} trainable parameters."
        )
    return model, rank


def build_with_budget(
    build_fn: Callable[[int], nn.Module],
    budget: int,
    use_peft: bool,
    embed_size: int,
    freeze_backbone: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, int]:
    """Build a model that respects a trainable-parameter budget.

    ``build_fn(embed_size)`` builds the model. With PEFT the LoRA rank (which
    DeepTune ties to embed_size) is chosen by ``choose_lora_rank``; without it
    the model is built at ``embed_size`` and ``apply_top_down_budget`` decides
    what trains. Returns (model, embed_size actually used) - callers must
    record that embed_size so evaluation/inference rebuild the same shapes.
    """
    budget = parse_param_count(budget)
    if freeze_backbone:
        log("Note: --trainable_params overrides --freeze-backbone; the budget decides which "
            "backbone parameters train.")
    if use_peft:
        model, embed_size = choose_lora_rank(build_fn, budget, embed_size)
        log(f"Trainable-parameter budget {budget:,}: using LoRA rank r = {embed_size} "
            f"(also the head's hidden width / embed_size).")
    else:
        model = build_fn(embed_size)
        summary = apply_top_down_budget(model, budget)
        log(f"Trainable-parameter budget {budget:,}: head plus the top "
            f"{summary['unfrozen_backbone_tensors']} of {summary['backbone_tensors']} backbone "
            f"tensors are trainable.")
    log(format_summary(summarize_parameters(model)))
    return model, embed_size

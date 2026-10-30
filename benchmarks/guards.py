"""Safety checks for augmentations that would silently corrupt motion labels.

A horizontal flip turns a clip moving right into one moving left, so on a
motion-direction dataset it swaps the labels of half the training data. Every
benchmark records ``hflip_safe`` / ``vflip_safe`` in its ``dataset.json``;
augmentation code (Phase 1 clip augmentation, the benchmark runner) calls
:func:`check_flip_augmentation` before enabling a flip.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path


class UnsafeAugmentationError(ValueError):
    """Raised when an augmentation would change the meaning of the labels."""


def _is_benchmark_metadata(meta) -> bool:
    """Only Phase 2 benchmark metadata (a dict with ``task`` and the flip flags)
    counts; a user's own ``dataset.json`` with other content does not."""
    return isinstance(meta, dict) and "task" in meta and ("hflip_safe" in meta or "vflip_safe" in meta)


def _find_metadata(source: Path | str | dict) -> dict | None:
    """Benchmark metadata for ``source``, or ``None`` for ordinary data. A
    ``dataset.json`` that is unreadable, not JSON, or not benchmark metadata is
    treated as ordinary data, never as an error."""
    if isinstance(source, dict):
        return source if _is_benchmark_metadata(source) else None
    path = Path(source)
    candidates = [path] if path.suffix == ".json" else [path / "dataset.json", path.parent / "dataset.json"]
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            meta = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # unreadable / invalid JSON / bad encoding
            continue
        if _is_benchmark_metadata(meta):
            return meta
    return None


def check_flip_augmentation(source: Path | str | dict, hflip: bool = False, vflip: bool = False,
                            strict: bool = True) -> None:
    """Refuse (or warn about) flips the dataset declares unsafe.

    Args:
        source: benchmark folder, its ``dataset.json``, a manifest inside it,
            or the already-loaded metadata dict.
        hflip / vflip: whether horizontal / vertical flip augmentation is on.
        strict: raise :class:`UnsafeAugmentationError` (default); if False,
            emit a ``UserWarning`` instead.

    Datasets without benchmark metadata (ordinary user data) pass silently.
    """
    meta = _find_metadata(source)
    if meta is None:
        return
    problems = []
    if hflip and meta.get("hflip_safe") is False:
        problems.append("horizontal flip")
    if vflip and meta.get("vflip_safe") is False:
        problems.append("vertical flip")
    if not problems:
        return
    message = (f"{' and '.join(problems).capitalize()} augmentation is unsafe on the "
               f"'{meta.get('task', 'unknown')}' benchmark: its labels depend on the direction of motion, "
               "so flipping a clip would swap its class. Turn the flip off for this dataset.")
    if strict:
        raise UnsafeAugmentationError(message)
    warnings.warn(message, UserWarning, stacklevel=2)

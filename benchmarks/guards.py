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


def _find_metadata(source: Path | str | dict) -> dict | None:
    if isinstance(source, dict):
        return source
    path = Path(source)
    candidates = [path] if path.suffix == ".json" else [path / "dataset.json", path.parent / "dataset.json"]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
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

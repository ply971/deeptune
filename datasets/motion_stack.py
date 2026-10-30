"""Motion fields for the two-stream model's temporal stream. Torch-free.

``load_motion_field`` wraps ``VideoReader.get_motion_vectors`` (conventions in datasets/video_io.py: pixels per frame
of the stored frame, later minus earlier, +dx right, +dy down; I-frames and frame 0 are zero) and adds:

* the display rotation, so the field matches what ``get_frames`` returns;
* an optional disk cache keyed by the video's fingerprint and every parameter.

``normalise_motion_field`` divides each component by ``MOTION_FULL_SCALE`` of the frame size: moving 5 % of the frame's
width per frame gives dx = 1 at any resolution, so the value stays valid when frames are resized for the model.
``clip_motion_field`` then saturates at [-1, 1]. No mean is subtracted, because the mean flow is the signal when the
camera is still.

Sources: ``farneback`` (OpenCV, works on every codec) and ``mv`` (codec motion vectors; raises
``MotionVectorsUnavailable`` when a file exports none, and needs clips of about 48 px or more, see
docs/video-plan/benchmark.md).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

MOTION_FULL_SCALE = 0.05
MOTION_BLOCK_SIZE = 4
_METHODS = {'farneback': 'farneback', 'mv': 'codec'}


def rotate_motion_field(field: np.ndarray, rotation: int) -> np.ndarray:
    """Turn a ``[..., 2, H, W]`` field by a display rotation (counter-clockwise degrees, as ``video_io._rotate``):
    the grid turns with the frame and so do the vectors (y points down, so one turn maps (dx, dy) to (dy, -dx))."""
    turns = int(round(rotation / 90.0)) % 4
    if not turns:
        return field
    turned = np.rot90(field, k=turns, axes=(-2, -1))
    dx, dy = turned[..., 0, :, :], turned[..., 1, :, :]
    vectors = {1: (dy, -dx), 2: (-dx, -dy), 3: (-dy, dx)}[turns]
    return np.stack(vectors, axis=-3).astype(np.float32)


def motion_cache_path(cache_dir, fingerprint: str, indices, source: str, block_size: int) -> Path:
    key = hashlib.sha1(f'{fingerprint}|{source}|{block_size}|{np.asarray(indices).tolist()}'.encode()).hexdigest()
    return Path(cache_dir) / key[:2] / f'{key}.npy'


def load_motion_field(reader, indices, source: str, block_size: int = MOTION_BLOCK_SIZE,
                      cache_dir: Optional[str] = None) -> np.ndarray:
    """Motion of the frames at `indices` from a ``VideoReader``: float32 ``[T, 2, Hb, Wb]`` in pixels per frame,
    display orientation. Indices past the end use the last frame, like ``get_frames(clamp_to_end=True)``."""
    if source not in _METHODS:
        raise ValueError(f"Unknown motion source {source!r}; choose one of {', '.join(_METHODS)}.")
    meta = reader.metadata()
    indices = np.minimum(np.asarray(indices, dtype=np.int64), meta.frame_count - 1)
    path = None
    if cache_dir:
        path = motion_cache_path(cache_dir, reader.fingerprint(), indices, source, block_size)
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):  # no entry yet, or an unreadable one: recompute
            pass
    field = reader.get_motion_vectors(indices, block_size=block_size, method=_METHODS[source]).field
    field = rotate_motion_field(field, meta.rotation)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f'{path.name}.{os.getpid()}.part')
        with open(partial, 'wb') as handle:
            np.save(handle, field, allow_pickle=False)
        os.replace(partial, path)
    return field


def normalise_motion_field(field: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pixels per frame of a ``height`` x ``width`` frame -> unit-free and resolution-independent (not yet clipped)."""
    scale = np.array([width, height], dtype=np.float32).reshape(2, 1, 1) * MOTION_FULL_SCALE
    return field / scale


def clip_motion_field(field: np.ndarray) -> np.ndarray:
    return np.clip(field, -1.0, 1.0)

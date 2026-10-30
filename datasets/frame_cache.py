"""Decoded-frame cache (``--frame_cache DIR``, Phase 1 task 7). Torch-free.

Decoding is usually the slowest part of an epoch, so the sampled, resized
uint8 frames of a clip are written once and memory-mapped afterwards.

Layout: one ``<DIR>/<key[:2]>/<key>.npy`` per clip (uint8 ``[T, H, W, 3]``),
opened with ``np.load(mmap_mode='r')``; several DataLoader workers therefore
share the OS page cache. Motion curves for ``--temporal_sampling motion`` live
next to them (float64 ``[frames]``), because computing one decodes the whole
clip.

**Key** (:func:`frame_cache_key`): the source fingerprint (``clip_key(i)``:
sha1 of the bytes, or of path + size + mtime), every sampling parameter, the
exact frame indices, the resize target and the resolved decoder (backends can
differ by a few grey levels), plus a format version.

**What is cached** (:func:`clip_sampling_is_cacheable`). A cached sample is
only valid if the same frames would be drawn again:

===========================================  ==========
clip sampling                                cached?
===========================================  ==========
evaluation / embedding / inference, any      yes (deterministic)
training, ``uniform``, whole clip            yes (deterministic)
training, ``segment`` / ``dense`` / ``motion``  no (random per epoch)
training, ``--clip_duration_sec``            no (random window position)
training, ``--video_augment``                no (temporal jitter, speed)
===========================================  ==========

Random clips bypass the cache, with one logged warning, instead of silently
reusing one draw for every epoch; their indices would never repeat, so caching
them would also just fill the disk. They still get the same resize, so the
training and validation pipelines match. Spatial augmentation runs after the
cache, on the resized frames, and stays random per epoch.

**Crash safety.** An entry is written to a unique ``*.part`` file, flushed to
disk, and moved into place with ``os.replace``, so readers see the complete old
state or the complete new one, even with concurrent writers or a killed
process. Reads also check dtype, shape and exact file size; anything invalid
counts as a miss and is rewritten. A file that is only locked (Windows refuses
to open one while it is being replaced) is a miss too, and is left in place. The cache has no size limit: delete the
folder to reclaim the space (it is always safe to do so between runs).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

LOGGER = logging.getLogger('deeptune.video')

CACHE_FORMAT = 1
DEFAULT_FRAME_CACHE_SIZE = 256
STALE_PART_SECONDS = 24 * 3600
_REPLACE_ATTEMPTS = 5


def retry_while_locked(action: Callable) -> None:
    """Run `action`, retrying briefly on PermissionError: on Windows a file another thread or
    process has open cannot be replaced or deleted for a moment. Raises if it stays locked."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            action()
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.02 * (attempt + 1))


def atomic_write(path, write: Callable) -> None:
    """Call ``write(stream)`` on a temporary file, flush it to disk, then move it to `path`.

    A crash at any point leaves `path` untouched (at worst a ``*.part`` file,
    which no reader ever opens). On Windows ``os.replace`` fails while another
    process has the target open, so it is retried briefly before raising.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part')
    try:
        with open(temporary, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        retry_while_locked(lambda: os.replace(temporary, path))
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def atomic_save_npy(path, array: np.ndarray) -> None:
    atomic_write(path, lambda stream: np.save(stream, array, allow_pickle=False))


def valid_npy_memmap(path) -> np.memmap:
    """Memory-map a ``.npy`` file, or raise ValueError/OSError if it is truncated or padded."""
    mapped = np.load(path, mmap_mode='r', allow_pickle=False)
    if not isinstance(mapped, np.memmap):  # a zero-size array is returned unmapped
        raise ValueError('not a memory-mappable array')
    if os.path.getsize(path) != mapped.offset + mapped.nbytes:
        raise ValueError('file size does not match the array header')
    return mapped


def clip_sampling_is_cacheable(spec, train: bool) -> bool:
    """True when the same indices come back every time (the table in the module docstring)."""
    if not train:
        return True
    return (spec.strategy == 'uniform' and spec.clip_duration_sec is None and not spec.temporal_jitter
            and not spec.speed_perturb)


def frame_cache_key(fingerprint: str, spec, indices, size: int, backend: str) -> str:
    """Stable key of one cached clip: changes with the file, any sampling parameter, the frames, the size or the decoder."""
    payload = {'format': CACHE_FORMAT, 'kind': 'frames', 'source': fingerprint, 'spec': asdict(spec),
               'size': int(size), 'backend': backend,
               'indices': hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def motion_curve_key(fingerprint: str, method: str, backend: str) -> str:
    payload = {'format': CACHE_FORMAT, 'kind': 'motion_curve', 'source': fingerprint, 'method': method, 'backend': backend}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resize_short_side(frames: np.ndarray, size: int) -> np.ndarray:
    """Shrink uint8 ``[T, H, W, 3]`` frames so the short side is `size`; never enlarges. ``size=0`` keeps them."""
    if not size or not len(frames):
        return frames
    height, width = frames.shape[1:3]
    scale = size / min(height, width)
    if scale >= 1:
        return frames
    import cv2
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.stack([cv2.resize(frame, target, interpolation=cv2.INTER_AREA) for frame in frames])


class FrameCache:
    """Reads and writes cache entries. Holds no file handles, so it pickles cheaply into DataLoader workers.

    Cache trouble (a full disk, a read-only folder) never stops a run: frames
    are served from the decoder and the problem is logged once per process.
    """

    def __init__(self, root, size: int = DEFAULT_FRAME_CACHE_SIZE):
        if int(size) < 0:
            raise ValueError('frame_cache_size must be 0 (keep the decoded size) or positive.')
        self.root = Path(root).expanduser()
        self.size = int(size)
        self._warned = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._sweep_stale_parts()
        except OSError as exc:
            self._warn(f'cannot use {self.root}: {exc}')

    def _warn(self, message: str):
        if not self._warned:
            self._warned = True
            LOGGER.warning('Frame cache: %s. Continuing without caching these clips.', message)

    def _sweep_stale_parts(self):
        """Delete temporary files left by killed processes (old enough that no writer can still own them)."""
        cutoff = time.time() - STALE_PART_SECONDS
        for part in self.root.rglob('*.part'):
            try:
                if part.stat().st_mtime < cutoff:
                    part.unlink()
            except OSError:
                pass

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f'{key}.npy'

    def resize(self, frames: np.ndarray) -> np.ndarray:
        return resize_short_side(frames, self.size)

    def _read(self, key: str, valid: Callable[[np.ndarray], bool]) -> Optional[np.ndarray]:
        path = self.path(key)
        try:
            mapped = valid_npy_memmap(path)
        except (FileNotFoundError, PermissionError):  # a locked file is not a damaged one: keep it
            return None
        except (ValueError, OSError, EOFError) as exc:
            self._drop_invalid(path, exc)
            return None
        if not valid(mapped):
            del mapped
            self._drop_invalid(path, 'unexpected dtype or shape')
            return None
        # A private copy: torch tensors need writable memory, and an open map
        # would keep the file locked on Windows.
        array = np.array(mapped)
        del mapped
        return array

    def _drop_invalid(self, path: Path, reason) -> None:
        LOGGER.warning('Frame cache: ignoring invalid entry %s (%s); it will be rewritten.', path.name, reason)
        try:
            path.unlink()
        except OSError:
            pass

    def _write(self, key: str, array: np.ndarray) -> None:
        path = self.path(key)
        try:
            atomic_save_npy(path, array)
        except OSError as exc:
            if path.is_file():  # another worker finished the same entry and still has it open
                return
            self._warn(f'cannot write {path}: {exc}')

    def get_frames(self, key: str, num_frames: int) -> Optional[np.ndarray]:
        """The cached uint8 ``[T, H, W, 3]`` frames, or None on a miss (or an invalid entry)."""
        return self._read(key, lambda a: a.dtype == np.uint8 and a.ndim == 4 and a.shape[0] == num_frames and a.shape[3] == 3)

    def put_frames(self, key: str, frames: np.ndarray) -> None:
        self._write(key, np.ascontiguousarray(frames))

    def get_curve(self, key: str) -> Optional[np.ndarray]:
        return self._read(key, lambda a: a.dtype == np.float64 and a.ndim == 1 and len(a) > 0)

    def put_curve(self, key: str, curve: np.ndarray) -> None:
        self._write(key, np.asarray(curve, dtype=np.float64))

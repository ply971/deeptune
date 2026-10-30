"""On-disk layout of cached snippet features (Phase 1 task 8, P-102). Torch-free.

Written by ``extract_features.py``; read here by the localisation head, the
attentive probe (``datasets/feature_dataset.py``) and anything else::

    <root>/features.json              format, per-stream config, dims, `split`
    <root>/cli_arguments.json         the extraction flags
    <root>/failed.json                videos that failed in the last run (if any)
    <root>/rgb/<video_id>.npy         float16 [N, D_rgb], or [N, t, h, w, D_rgb]
    <root>/rgb/<video_id>.json        sidecar (below)
    <root>/motion/<video_id>.npy      float16 [N, D_motion], or [N, t, h, w, D_motion]
    <root>/motion/<video_id>.json

:func:`load_snippet_features` returns the streams concatenated on the last
axis, ``[N, D_rgb + D_motion]``, with ``split = {'rgb': (0, D_rgb), 'motion':
(D_rgb, D_rgb + D_motion)}``.

**Snippets** (:func:`snippet_count`). A video of ``F`` frames with snippet
length ``S`` and stride ``R`` gives snippets starting at ``k * R``::

    N = min(1 + ceil(max(F - S, 0) / R), ceil(F / R))

The first term is the usual ``ceil((F - S) / R) + 1``: every frame is covered
and a last, incomplete snippet is padded by repeating the final frame (its
real length is the sidecar's ``last_snippet_valid_frames``). The ``min`` only
matters when ``R > S``: it stops a snippet from starting after the last frame.
Snippet ``k`` starts at ``start_frame + k * R``, which is ``k * R / fps``
seconds into the range when the frame rate is constant.

**Sidecar** keys: ``fps, snippet_frames, stride, backbone, weights,
weights_licence, stream`` plus ``video_id, source`` (absolute path, or ``<N bytes>``)``, source_fingerprint,
config_hash, frame_count, start_frame, end_frame, duration_sec, variable_fps,
num_snippets, last_snippet_valid_frames, shape, dim, token_grid, dtype,
float16, seconds, bytes, source_range, field_of_view``. The motion stream adds
``motion_source, intra_fraction, motion_normalisation, motion_gain,
motion_clip_pixels, motion_encoding, motion_weights_trained_on_motion``; a
variable-frame-rate video adds
``snippet_start_sec``.

**float16** (:func:`to_float16`). A cast can overflow to inf (above 65504) and
loses precision (about 3 significant digits; values under 6e-8 become 0), so
any non-finite value, before or after the cast, rejects the whole video
rather than storing it, and the worst rounding error is recorded in the
sidecar's ``float16``.

**Commit order.** The array is written atomically first, the sidecar last, so
an interrupted extraction leaves an entry that fails :meth:`FeatureStore.is_complete`
and is simply extracted again. An entry only counts as complete if its source
fingerprint and config hash match and the array has the recorded shape.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from datasets.frame_cache import atomic_save_npy, atomic_write, retry_while_locked, valid_npy_memmap

LOGGER = logging.getLogger('deeptune.video')

FORMAT_VERSION = 1
# Bumped when a stream's features change meaning. Motion: 2 = dx, dy, magnitude instead of one repeated filter;
# 3 = the same crop of the frame as rgb, and an explicit gain.
PREPROCESSING_VERSION = {'rgb': 2, 'motion': 3}
STREAMS = ('rgb', 'motion')  # concatenation order
INFO_FILE = 'features.json'
FAILED_FILE = 'failed.json'
# Below the smallest normal float16 (2 ** -14) the relative precision drops quickly.
FLOAT16_MIN_NORMAL = float(np.finfo(np.float16).tiny)


class FeatureValueError(ValueError):
    """Features that float16 cannot represent (NaN, inf, overflow, or everything rounding to zero)."""


class FeatureConfigError(ValueError):
    """A feature folder that already holds features extracted with other settings."""


# ---------------------------------------------------------------------------
# Snippet arithmetic
# ---------------------------------------------------------------------------
def _check_snippet_args(snippet_frames: int, stride: int) -> None:
    if snippet_frames < 1 or stride < 1:
        raise ValueError('snippet_frames and snippet_stride must be at least 1.')


def snippet_count(frames: int, snippet_frames: int, stride: int) -> int:
    """Number of snippets in a video of `frames` frames (formula: module docstring)."""
    _check_snippet_args(snippet_frames, stride)
    if frames < 1:
        return 0
    return min(1 + math.ceil(max(frames - snippet_frames, 0) / stride), math.ceil(frames / stride))


def snippet_starts(frames: int, snippet_frames: int, stride: int) -> np.ndarray:
    """First frame (relative to the range) of every snippet, int64 ``[N]``."""
    return np.arange(snippet_count(frames, snippet_frames, stride), dtype=np.int64) * stride


def parse_token_grid(text) -> Optional[tuple[int, int, int]]:
    """``--cache_tokens``: 'none' -> None (one vector per snippet); '8' -> (8, 1, 1); '8x4x4' -> (8, 4, 4)."""
    value = str(text).strip().lower()
    if value in ('', 'none'):
        return None
    parts = value.split('x')
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        numbers = []
    if len(numbers) not in (1, 3) or min(numbers, default=0) < 1:
        raise ValueError(f"Token grid {text!r}: use 'none', 'T' or 'TxHxW' with positive integers, e.g. 8x4x4.")
    return (numbers[0], 1, 1) if len(numbers) == 1 else tuple(numbers)


# ---------------------------------------------------------------------------
# Video ids
# ---------------------------------------------------------------------------
_UNSAFE_ID = re.compile(r'[^\w.\-]+')


def safe_video_id(value) -> str:
    cleaned = _UNSAFE_ID.sub('_', str(value)).strip('._')
    if not cleaned:
        raise ValueError(f'Video id {value!r} has no usable characters.')
    return cleaned


def same_path(left, right) -> bool:
    """Whether two paths name one file. Existing files are asked directly, which also settles letter case, links and
    8.3 names as the file system sees them; otherwise the spellings are compared, ignoring case only on Windows."""
    try:
        if os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


def video_id_for_path(path, start_sec=None, end_sec=None) -> str:
    """Id of a manifest row without a ``video_id`` column. It depends only on that
    row, so a train split and the full manifest agree on it."""
    name = os.path.normcase(os.path.abspath(str(path)))
    trim = '' if start_sec is None and end_sec is None else f'|{start_sec}|{end_sec}'
    digest = hashlib.sha1(f'{name}{trim}'.encode()).hexdigest()[:8]
    return f'{safe_video_id(Path(str(path)).stem)}-{digest}'


# ---------------------------------------------------------------------------
# float16
# ---------------------------------------------------------------------------
def to_float16(features, label: str = 'features') -> tuple[np.ndarray, dict]:
    """Cast to float16, refusing anything it cannot represent. Returns (array, stats for the sidecar)."""
    source = np.asarray(features, dtype=np.float32)
    bad = int(np.size(source) - np.count_nonzero(np.isfinite(source)))
    if bad:
        raise FeatureValueError(f'{label}: {bad} of {source.size} values are NaN or infinite.')
    with np.errstate(over='ignore', under='ignore'):
        converted = source.astype(np.float16)
    overflowed = int(np.size(converted) - np.count_nonzero(np.isfinite(converted)))
    max_abs = float(np.abs(source).max()) if source.size else 0.0
    if overflowed:
        raise FeatureValueError(f'{label}: {overflowed} values overflow float16 (largest magnitude {max_abs:.6g}; '
                                f'float16 stops at {float(np.finfo(np.float16).max):.0f}).')
    back = converted.astype(np.float32)
    nonzero = source != 0
    flushed = int(np.count_nonzero(nonzero & (back == 0)))
    if source.size and flushed == int(np.count_nonzero(nonzero)) and flushed:
        raise FeatureValueError(f'{label}: every nonzero value rounds to 0 in float16 (largest magnitude {max_abs:.6g}).')
    error = float(np.abs(source - back).max()) if source.size else 0.0
    relative = error / max_abs if max_abs else 0.0
    stats = {'max_abs': max_abs, 'max_abs_error': error, 'max_error_relative_to_max': relative,
             'flushed_to_zero': flushed, 'subnormal_range_only': bool(0 < max_abs < FLOAT16_MIN_NORMAL)}
    if stats['subnormal_range_only']:
        LOGGER.warning('%s: all values are below %.3g, so float16 keeps little precision (largest magnitude %.3g).',
                       label, FLOAT16_MIN_NORMAL, max_abs)
    return converted, stats


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def config_hash(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json_atomic(path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, default=str).encode('utf-8')
    atomic_write(path, lambda stream: stream.write(text))


def _read_json(path) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class SnippetFeatures:
    """One video's features. `features` is ``[N, D_total]`` or ``[N, t, h, w, D_total]`` float16;
    `split` gives each stream's ``(start, end)`` on the last axis; `meta` holds each stream's sidecar."""
    video_id: str
    features: np.ndarray
    split: dict[str, tuple[int, int]]
    meta: dict[str, dict] = field(default_factory=dict)

    def stream(self, name: str) -> np.ndarray:
        start, end = self.split[name]
        return self.features[..., start:end]


def check_streams_agree(video_id: str, meta: dict[str, dict]) -> None:
    """Refuse to treat streams as one video unless their sidecars (`meta`, by stream) describe the same frames."""
    prints = {name: side['source_fingerprint'] for name, side in meta.items() if 'source_fingerprint' in side}
    if len(set(prints.values())) > 1:
        sources = {name: meta[name].get('source') for name in prints}
        raise ValueError(f'Video {video_id!r}: the streams come from different videos, or from a file that has changed '
                         f'since: {sources}. Run extract_features.py again with all the streams.')
    ranges = {name: side['source_range'] for name, side in meta.items() if 'source_range' in side}
    if len({json.dumps(value, sort_keys=True) for value in ranges.values()}) > 1:
        raise ValueError(f'Video {video_id!r}: the streams come from different trims of the video: {ranges}. '
                         'Run extract_features.py again with all the streams.')
    views = {name: side['field_of_view'] for name, side in meta.items() if 'field_of_view' in side}
    if len({json.dumps(value) for value in views.values()}) > 1:
        raise ValueError(f'Video {video_id!r}: the streams see different parts of the frame: {views}. Extract them with '
                         'the same --model_version and --feature_input_size, or load one stream at a time.')


class FeatureStore:
    """A feature folder. Pickles as its path, so DataLoader workers do not copy anything."""

    def __init__(self, root):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------
    def array_path(self, stream: str, video_id: str) -> Path:
        return self.root / stream / f'{video_id}.npy'

    def sidecar_path(self, stream: str, video_id: str) -> Path:
        return self.root / stream / f'{video_id}.json'

    # -- folder-level info ----------------------------------------------------
    def info(self) -> dict:
        return _read_json(self.root / INFO_FILE) or {'format_version': FORMAT_VERSION, 'streams': {}, 'split': {}}

    def check_config(self, stream: str, config: dict) -> None:
        """Refuse to mix settings: one folder, one config per stream."""
        known = self.info()['streams'].get(stream)
        if known and known.get('config_hash') != config_hash(config):
            differing = sorted(k for k in set(known['config']) | set(config)
                               if known['config'].get(k) != config.get(k))
            raise FeatureConfigError(f'{self.root} already holds {stream!r} features with other settings '
                                     f'(differs in: {", ".join(differing)}). Choose another --out folder or delete it.')

    def check_motion_is_current(self, streams: Sequence[str]) -> None:
        """Refuse to read motion features from before the current preprocessing: they cannot be trusted."""
        known = self.info()['streams'].get('motion')
        if 'motion' not in streams or not known:
            return
        version = (known.get('config') or {}).get('preprocessing_version', 0)
        if version < PREPROCESSING_VERSION['motion']:
            raise FeatureConfigError(f'{self.root} holds motion features from preprocessing version {version}; the current '
                                     f"one is {PREPROCESSING_VERSION['motion']}. Run extract_features.py again into a new folder.")

    def register_stream(self, stream: str, config: dict, dim: int, token_grid) -> None:
        info = self.info()
        info['format_version'] = FORMAT_VERSION
        info['streams'][stream] = {'config': config, 'config_hash': config_hash(config), 'dim': int(dim),
                                   'token_grid': list(token_grid) if token_grid else None}
        split, start = {}, 0
        for name in STREAMS:
            if name in info['streams']:
                end = start + int(info['streams'][name]['dim'])
                split[name] = [start, end]
                start = end
        info['split'] = split
        write_json_atomic(self.root / INFO_FILE, info)

    def streams(self) -> list[str]:
        known = self.info()['streams']
        return [name for name in STREAMS if name in known]

    # -- entries ---------------------------------------------------------------
    def read_sidecar(self, stream: str, video_id: str) -> Optional[dict]:
        return _read_json(self.sidecar_path(stream, video_id))

    def is_complete(self, stream: str, video_id: str, fingerprint: str, config_digest: str,
                    source_range: Optional[dict] = None) -> bool:
        """True only for a whole, matching entry: this is what makes extraction resumable."""
        sidecar = self.read_sidecar(stream, video_id)
        if not sidecar or sidecar.get('source_fingerprint') != fingerprint or sidecar.get('config_hash') != config_digest:
            return False
        if source_range is not None and sidecar.get('source_range') != source_range:
            return False
        try:
            mapped = valid_npy_memmap(self.array_path(stream, video_id))
        except (ValueError, OSError, EOFError):
            return False
        complete = mapped.dtype == np.float16 and list(mapped.shape) == list(sidecar.get('shape') or [])
        del mapped
        return complete

    def write(self, stream: str, video_id: str, features: np.ndarray, sidecar: dict) -> None:
        if features.dtype != np.float16:
            raise TypeError('Snippet features are stored as float16; use to_float16 first.')
        side_path = self.sidecar_path(stream, video_id)
        try:  # never let a stale sidecar describe the new array
            retry_while_locked(side_path.unlink)
        except FileNotFoundError:
            pass
        atomic_save_npy(self.array_path(stream, video_id), features)
        size = self.array_path(stream, video_id).stat().st_size
        write_json_atomic(side_path, dict(sidecar, shape=list(features.shape), dtype='float16', bytes=size))

    def video_ids(self, stream: Optional[str] = None) -> list[str]:
        """Videos with a sidecar in `stream` (default: in every stream of this folder)."""
        names = [stream] if stream else self.streams()
        found = None
        for name in names:
            ids = {path.stem for path in (self.root / name).glob('*.json')}
            found = ids if found is None else found & ids
        return sorted(found or ())

    def load(self, video_id: str, streams: Optional[Sequence[str]] = None, indices=None,
             mmap: bool = True) -> SnippetFeatures:
        """One video's streams concatenated on the last axis. `indices` picks snippets before
        concatenating; `mmap=False` returns writable arrays in memory."""
        names = [name for name in STREAMS if name in (streams or self.streams())]
        if streams and len(names) != len(set(streams)):
            raise ValueError(f'Unknown stream in {list(streams)}; choose from {", ".join(STREAMS)}.')
        if not names:
            raise FileNotFoundError(f'{self.root} has no extracted features ({INFO_FILE} is missing or empty).')
        arrays, split, meta, start = [], {}, {}, 0
        for name in names:
            path = self.array_path(name, video_id)
            if not path.is_file():
                raise FileNotFoundError(f'No {name!r} features for video {video_id!r} in {self.root}.')
            array = np.load(path, mmap_mode='r' if mmap else None, allow_pickle=False)
            arrays.append(array)
            split[name] = (start, start + int(array.shape[-1]))
            start += int(array.shape[-1])
            meta[name] = self.read_sidecar(name, video_id) or {}
        check_streams_agree(video_id, meta)
        if len({a.shape[:-1] for a in arrays}) > 1:
            shapes = {name: a.shape for name, a in zip(names, arrays)}
            raise ValueError(f'Video {video_id!r}: streams cannot be concatenated, their shapes differ: {shapes}. '
                             'Token grids must match; extract them with the same --cache_tokens and --snippet_* '
                             'flags, or load one stream at a time.')
        if indices is not None:
            arrays = [a[np.asarray(indices)] for a in arrays]
        features = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=-1)
        return SnippetFeatures(video_id, features, split, meta)


def load_snippet_features(root, video_id: str, streams: Optional[Sequence[str]] = None, mmap: bool = True) -> SnippetFeatures:
    """``[N, D_rgb + D_motion]`` (or one stream) plus the recorded split: see :class:`SnippetFeatures`.
    Refuses motion features older than the current preprocessing."""
    store = FeatureStore(root)
    loaded = store.load(video_id, streams=streams, mmap=mmap)
    store.check_motion_is_current(list(loaded.split))
    return loaded

"""Video datasets: lazy, path- or bytes-based, with temporal sampling, clip
augmentation and multi-clip test-time sampling.

Two storage formats, detected automatically (``detect_video_storage``):

* **paths** (new default for ``--raw-data`` ingestion): the ``videos`` column
  holds file paths (absolute, or relative to the manifest's folder). Nothing
  is read until ``__getitem__`` -- a 100k-clip manifest costs a few MB.
  -> :class:`VideoDataset`.
* **bytes** (DeepTune's original format, still fully supported): the
  ``videos`` column holds whole encoded clips. -> :class:`ParquetVideoDataset`.
  Large parquet files, and any bytes dataset handed to DataLoader workers,
  are moved to a memory-mapped Arrow IPC file, so worker processes receive a
  file path instead of a copy of every clip (G15).

Optional manifest columns: ``labels``, ``group`` (kept together by the
splitter), ``start_sec`` / ``end_sec`` (sample only that part of the video);
all extra columns are passed through as a third item, as before.

Each sample is ``(clip, label[, extras])`` with ``clip`` shaped
``[T, C, H, W]`` -- or ``[N, T, C, H, W]`` when ``num_clips = N > 1``
(``--test_clips``; see ``src/video/multi_clip.py`` for the averaging).

Hooks for the frame cache (Phase 1 task 7, a later agent): every decode goes
through ``_decode_frames(i, indices)``; ``clip_key(i)`` gives a stable key
(file fingerprint). A cache subclasses / wraps ``_decode_frames``.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from PIL import Image
from torch.utils.data import Dataset

from datasets.video_augment import ClipAugment, ClipAugmentConfig
from datasets.video_io import (BACKEND_CHOICES, CURVE_METHODS, VideoDecodeError, VideoReader, describe_source,
                               source_fingerprint)
from datasets.video_sampling import STRATEGIES, SamplingSpec, sample_clip_indices

LOGGER = logging.getLogger('deeptune.video')

STORAGE_CHOICES = ('paths', 'bytes')
# Bytes parquet files above this size are memory-mapped instead of loaded.
LAZY_PARQUET_BYTES = 256 * 1024 * 1024
RESERVED_COLUMNS = ('videos', 'labels')


class VideoPresetTransform:
    """Apply the native backbone's Kinetics preprocessing to a whole clip."""
    def __init__(self, model_version: str):
        from src.video.video3d import _KINETICS400_WEIGHTS
        self.transform = _KINETICS400_WEIGHTS[model_version].transforms()

    def __call__(self, clip):
        # torchvision returns C,T,H,W; DeepTune datasets expose T,C,H,W.
        return self.transform(clip).permute(1, 0, 2, 3)


def native_video_transform(model_version: str):
    from src.video.video3d import VIDEO3D_ARCHITECTURES
    return VideoPresetTransform(model_version) if model_version in VIDEO3D_ARCHITECTURES else None


def collate_video_samples(samples):
    """The default collator rejects None labels on embedding-only datasets,
    and gives an unreadable error when clips differ in size."""
    from torch.utils.data import default_collate
    shapes = {tuple(sample[0].shape) for sample in samples}
    if len(shapes) > 1:
        raise ValueError(f'Clips in one batch have different shapes {sorted(shapes)}. Use a resizing transform '
                         '(the default one resizes every frame) when videos have mixed resolutions.')
    clips = default_collate([sample[0] for sample in samples])
    labels = None if all(sample[1] is None for sample in samples) else default_collate([sample[1] for sample in samples])
    if len(samples[0]) == 3:
        return clips, labels, default_collate([sample[2] for sample in samples])
    return clips, labels


# ---------------------------------------------------------------------------
# Options shared by train / eval / embed / inference
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoDataOptions:
    """Every video data-layer setting. All of them are CLI flags recorded in
    cli_arguments.json (see cli.py) so inference rebuilds the same pipeline."""
    video_backend: str = 'auto'
    temporal_sampling: str = 'uniform'
    clip_stride: Optional[int] = None
    clip_duration_sec: Optional[float] = None
    motion_source: str = 'auto'
    video_augment: bool = False
    video_flip: bool = False
    video_speed_perturb: float = 0.0
    test_clips: int = 1
    num_workers: object = 'auto'

    def __post_init__(self):
        if self.video_backend not in BACKEND_CHOICES:
            raise ValueError(f'video_backend must be one of {", ".join(BACKEND_CHOICES)}.')
        if self.temporal_sampling not in STRATEGIES:
            raise ValueError(f'temporal_sampling must be one of {", ".join(STRATEGIES)}.')
        if self.motion_source not in CURVE_METHODS:
            raise ValueError(f'motion_source must be one of {", ".join(CURVE_METHODS)}.')
        if int(self.test_clips) < 1:
            raise ValueError('test_clips must be at least 1.')
        if self.video_flip and not self.video_augment:
            raise ValueError('--video_flip only applies together with --video_augment.')

    def sampling_spec(self, num_frames: int, train: bool) -> SamplingSpec:
        return SamplingSpec(strategy=self.temporal_sampling, num_frames=num_frames, clip_stride=self.clip_stride,
                            clip_duration_sec=self.clip_duration_sec,
                            temporal_jitter=bool(train and self.video_augment),
                            speed_perturb=float(self.video_speed_perturb) if (train and self.video_augment) else 0.0)

    def augment_config(self) -> ClipAugmentConfig:
        return ClipAugmentConfig(enabled=bool(self.video_augment), flip=bool(self.video_flip),
                                 speed_perturb=float(self.video_speed_perturb))


_OPTION_TYPES = {
    'video_backend': str, 'temporal_sampling': str, 'clip_stride': int, 'clip_duration_sec': float,
    'motion_source': str, 'video_augment': bool, 'video_flip': bool, 'video_speed_perturb': float,
    'test_clips': int, 'num_workers': (int, str),
}


def video_options_from_args(args=None, **overrides) -> VideoDataOptions:
    """Read VideoDataOptions from a DeepTuneVisionOptions / argparse Namespace /
    dict, ignoring anything missing or not a plain value (old saved runs,
    test doubles), then apply `overrides`."""
    values = {}
    for name, kind in _OPTION_TYPES.items():
        value = args.get(name) if isinstance(args, dict) else getattr(args, name, None) if args is not None else None
        if value is None or isinstance(value, bool) and kind is not bool:
            continue
        if isinstance(value, kind if isinstance(kind, tuple) else (kind,)) or (kind is float and isinstance(value, int)):
            values[name] = value
    values.update({k: v for k, v in overrides.items() if v is not None})
    return VideoDataOptions(**values)


# ---------------------------------------------------------------------------
# Bytes stores
# ---------------------------------------------------------------------------
class _MemoryBytes:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


SPILL_PREFIX = 'deeptune-clips-'
STALE_SPILL_SECONDS = 24 * 3600
_PENDING_UNLINK: set = set()


def _retry_pending_unlinks():
    for path in list(_PENDING_UNLINK):
        try:
            os.unlink(path)
            _PENDING_UNLINK.discard(path)
        except FileNotFoundError:
            _PENDING_UNLINK.discard(path)
        except OSError:
            pass


def _try_unlink(path):
    """Remove a spilled clip file. On Windows this fails while a (persistent)
    worker still maps it; then retry at interpreter exit, and failing that the
    next run's sweep (`_sweep_stale_spills`) removes it."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        if not _PENDING_UNLINK:
            import atexit
            atexit.register(_retry_pending_unlinks)
        _PENDING_UNLINK.add(str(path))


def _sweep_stale_spills(folder=None, max_age=STALE_SPILL_SECONDS) -> int:
    """Delete spilled clip files older than `max_age` left behind by crashed or killed runs."""
    import time
    folder = Path(folder or tempfile.gettempdir())
    removed = 0
    for path in folder.glob(f'{SPILL_PREFIX}*.arrow'):
        try:
            if time.time() - path.stat().st_mtime > max_age:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed


class ArrowBytesStore:
    """Encoded clips in an uncompressed Arrow IPC file, memory-mapped.

    Pickles as its path only, so DataLoader workers map the same file
    (shared OS page cache) instead of each receiving every clip's bytes.
    """

    def __init__(self, path: Path, length: int):
        self.path = str(path)
        self.length = int(length)
        self._column = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        return {'path': self.path, 'length': self.length}

    def __setstate__(self, state):
        self.path, self.length, self._column = state['path'], state['length'], None

    def _open(self):
        import pyarrow as pa
        source = pa.memory_map(self.path, 'r')
        self._column = pa.ipc.open_file(source).read_all().column('videos')

    def __getitem__(self, index):
        if self._column is None:
            self._open()
        value = self._column[int(index)].as_py()
        return value

    @classmethod
    def from_batches(cls, batches, path: Path) -> 'ArrowBytesStore':
        import pyarrow as pa
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([('videos', pa.large_binary())])
        tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.part')
        rows = 0
        with pa.OSFile(str(tmp), 'wb') as sink, pa.ipc.new_file(sink, schema) as writer:
            for values in batches:
                array = pa.array(values, type=pa.large_binary())
                writer.write_batch(pa.record_batch([array], schema=schema))
                rows += len(array)
        os.replace(tmp, path)
        return cls(path, rows)


def _parquet_cache_path(parquet_file: Path) -> Path:
    key = source_fingerprint(parquet_file)[:12]
    folder = parquet_file.parent / '.deeptune_video_cache'
    try:
        folder.mkdir(exist_ok=True)
        probe = folder / '.write_test'
        probe.write_bytes(b'')
        probe.unlink()
    except OSError:
        folder = Path(tempfile.gettempdir()) / 'deeptune_video_cache'
        folder.mkdir(exist_ok=True)
    return folder / f'{parquet_file.stem}-{key}.videos.arrow'


# ---------------------------------------------------------------------------
# Storage detection
# ---------------------------------------------------------------------------
def detect_video_storage(source) -> str:
    """'paths' or 'bytes' for a DataFrame or a parquet/CSV/XLSX file, without reading clip data."""
    if isinstance(source, DataFrame):
        if 'videos' not in source.columns:
            raise ValueError("Video datasets must have a 'videos' column (file paths or clip bytes).")
        for value in source['videos']:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            return 'bytes' if isinstance(value, (bytes, bytearray, memoryview)) else 'paths'
        return 'bytes'
    path = Path(source)
    if path.suffix.lower() in ('.csv', '.xlsx'):
        return 'paths'
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    if 'videos' not in schema.names:
        raise ValueError(f"{path.name} has no 'videos' column (file paths or clip bytes).")
    kind = schema.field('videos').type
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return 'paths'
    return 'bytes'


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class _VideoClipDataset(Dataset):
    """Shared sampling / decoding / augmentation logic; subclasses provide `_source(i)`."""

    def __init__(self, df: DataFrame, transform=None, num_frames: int = 8, clip_transform=None, *,
                 train: bool = False, options: Optional[VideoDataOptions] = None, num_clips: int = 1):
        if isinstance(num_frames, bool) or not isinstance(num_frames, (int, np.integer)) or num_frames < 1:
            raise ValueError('num_frames must be a positive integer.')
        if 'videos' not in df.columns:
            raise ValueError("Video datasets must have a 'videos' column containing clip bytes or file paths.")
        self.transform = transform
        self.clip_transform = clip_transform
        self.num_frames = int(num_frames)
        self.train = bool(train)
        self.options = options or VideoDataOptions()
        self.num_clips = int(num_clips)
        if self.num_clips < 1:
            raise ValueError('num_clips must be at least 1.')
        self.spec = self.options.sampling_spec(self.num_frames, self.train)
        self.augment = ClipAugment(self.options.augment_config()) if self.train and self.options.video_augment else None
        self.has_labels = 'labels' in df.columns
        self.labels = df['labels'].to_numpy() if self.has_labels else None
        self.float_labels = bool(self.has_labels and pd.api.types.is_float_dtype(df['labels']))
        if self.has_labels and len(df) and not (pd.api.types.is_numeric_dtype(df['labels'])
                                                or pd.api.types.is_bool_dtype(df['labels'])):
            raise ValueError('Video labels must be numeric (class indices or regression targets); deeptune.py '
                             'encodes class names when it splits the dataset.')
        self.data = df.drop(columns=['videos'])  # labels + passthrough columns; never clip bytes
        self._extra_columns = [c for c in df.columns if c not in RESERVED_COLUMNS]
        self._starts = self._time_column(df, 'start_sec')
        self._ends = self._time_column(df, 'end_sec')
        self._motion_cache: dict = {}
        self._warned: set = set()

    @staticmethod
    def _time_column(df, name):
        if name not in df.columns:
            return None
        values = pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f'{name} must not be negative.')
        return values

    # -- subclass API ------------------------------------------------------
    def _source(self, i: int):
        raise NotImplementedError

    def prepare_for_workers(self):
        """Called before DataLoader workers start; bytes datasets move clips out of the pickled state."""

    def __len__(self):
        return len(self.data)

    # -- hooks for the frame cache (task 7) -------------------------------
    def clip_key(self, i: int) -> str:
        return source_fingerprint(self._source(i))

    def _decode_frames(self, i: int, reader: VideoReader, indices: np.ndarray) -> np.ndarray:
        return reader.get_frames(indices, clamp_to_end=True)

    # -- sampling ----------------------------------------------------------
    def _rng(self):
        if not self.train:
            return np.random.default_rng(0)
        # Drawn from torch so --fixed-seed and per-worker seeding apply.
        return np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,)).item()))

    def _motion(self, i: int, reader: VideoReader):
        if i not in self._motion_cache:
            self._motion_cache[i] = reader.motion_curve(self.options.motion_source)
        return self._motion_cache[i]

    def sample_indices(self, i: int, reader: VideoReader) -> np.ndarray:
        meta = reader.metadata()
        start = end = None
        if self._starts is not None and np.isfinite(self._starts[i]):
            start = reader.time_to_index(self._starts[i])
        if self._ends is not None and np.isfinite(self._ends[i]):
            end = reader.time_to_end_index(self._ends[i])
            if start is not None and end <= start:
                raise ValueError(f'end_sec ({self._ends[i]}) must be after start_sec ({self._starts[i]}).')
        motion = self._motion(i, reader) if self.spec.needs_motion else None
        indices = sample_clip_indices(meta.frame_count, self.spec, train=self.train, rng=self._rng(),
                                      num_clips=self.num_clips, fps=meta.fps, start_frame=start or 0,
                                      end_frame=end, motion=motion)
        span = (end or meta.frame_count) - (start or 0)
        needed = self.num_frames if self.spec.strategy != 'dense' else (self.num_frames - 1) * self.spec.stride + 1
        if span < needed and i not in self._warned:
            self._warned.add(i)
            LOGGER.warning('Video row %d has %d frames in range but the clip needs %d; frames are repeated '
                           '(the last frame pads dense clips).', i, span, needed)
        return indices

    # -- items -------------------------------------------------------------
    def _to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # uint8 [T, C, H, W]
        if self.augment is not None:
            clip, _ = self.augment(clip)
        if self.clip_transform is not None:
            return self.clip_transform(clip)
        if self.transform:
            return torch.stack([self.transform(Image.fromarray(frame.permute(1, 2, 0).numpy())) for frame in clip])
        return clip.float() / 255.0

    def load_clip(self, i: int) -> torch.Tensor:
        source = self._source(i)
        with VideoReader(source, backend=self.options.video_backend) as reader:
            indices = self.sample_indices(i, reader)
            frames = self._decode_frames(i, reader, indices.reshape(-1))
        frames = frames.reshape(self.num_clips, self.num_frames, *frames.shape[1:])
        clips = [self._to_tensor(clip) for clip in frames]
        return clips[0] if self.num_clips == 1 else torch.stack(clips)

    def __getitem__(self, idx):
        try:
            clip = self.load_clip(idx)
        except (ValueError, IndexError) as exc:
            raise ValueError(f'Video row {idx}: {exc}') from exc
        label = None
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.float32 if self.float_labels else torch.long)
        if self._extra_columns:
            extras = self.data.iloc[idx][self._extra_columns].to_dict()
            return clip, label, extras
        return clip, label


class ParquetVideoDataset(_VideoClipDataset):
    """
    Loads the videos and labels from a parquet file we feed to the model.

    Each video clip is stored as raw encoded bytes (the same convention
    DeepTune uses for images) and decoded on the fly -- in memory with PyAV,
    no temporary file. ``num_frames`` frames are sampled per clip
    (``--temporal_sampling``, default uniform over the whole clip, exactly
    as before), so every sample has the same shape regardless of the clip's
    length or frame rate.

    Attributes:
        data (pd.DataFrame): labels and passthrough columns (no clip bytes).
        transform: per-frame (PIL) transform, DeepTune's ImageNet-style one by default.
        clip_transform: whole-clip transform (native 3D backbones' Kinetics preset).
        num_frames (int): frames sampled per clip.
        labels (np.ndarray): the labels, if present.
    """

    def __init__(self, df: DataFrame, transform=None, num_frames: int = 8, clip_transform=None, **kwargs):
        super().__init__(df, transform, num_frames, clip_transform, **kwargs)
        self._store = _MemoryBytes(df['videos'].to_numpy())
        self._finalizer = None

    @classmethod
    def from_parquet(cls, parquet_file, transform=None, num_frames: int = 8, clip_transform=None, **kwargs):
        """Open a parquet split of either storage format (detected); path-based
        parquet files give a :class:`VideoDataset`."""
        return open_video_dataset(parquet_file, transform=transform, num_frames=num_frames,
                                  clip_transform=clip_transform, **kwargs)

    @classmethod
    def _lazy_from_parquet(cls, parquet_file: Path, transform, num_frames, clip_transform, **kwargs):
        import pyarrow.parquet as pq
        names = pq.read_schema(parquet_file).names
        meta = pd.read_parquet(parquet_file, columns=[c for c in names if c != 'videos'])
        cache = _parquet_cache_path(parquet_file)
        if cache.is_file():
            store = ArrowBytesStore(cache, len(meta))
        else:
            reader = pq.ParquetFile(parquet_file)
            batches = (batch.column(0).to_pylist() for batch in reader.iter_batches(columns=['videos'], batch_size=64))
            store = ArrowBytesStore.from_batches(batches, cache)
        if len(store) != len(meta):
            raise ValueError(f'{parquet_file.name}: cached clip store has {len(store)} rows, expected {len(meta)}.')
        placeholder = meta.copy()
        placeholder.insert(0, 'videos', None)
        dataset = cls.__new__(cls)
        _VideoClipDataset.__init__(dataset, placeholder, transform, num_frames, clip_transform, **kwargs)
        dataset._store, dataset._finalizer = store, None
        return dataset

    def _source(self, i: int):
        value = self._store[i]
        if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
            raise VideoDecodeError('Each video must contain nonempty encoded clip bytes.')
        return value

    def prepare_for_workers(self):
        if not isinstance(self._store, _MemoryBytes) or not len(self._store):
            return
        _sweep_stale_spills()
        handle = tempfile.NamedTemporaryFile(prefix=SPILL_PREFIX, suffix='.arrow', delete=False)
        handle.close()
        values = list(self._store.values)
        self._store = ArrowBytesStore.from_batches((values[i:i + 64] for i in range(0, len(values), 64)), Path(handle.name))
        self._finalizer = weakref.finalize(self, _try_unlink, handle.name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_finalizer'] = None
        return state


class VideoDataset(_VideoClipDataset):
    """Path-based video dataset: a manifest of file paths, decoded lazily.

    `df` has a ``videos`` column of paths (absolute, or relative to `root`)
    plus optional ``labels``, ``group``, ``start_sec``, ``end_sec`` and any
    passthrough columns. Construction reads no clip data (only an existence
    check per path, which is a stat, not a read).
    """

    def __init__(self, df: DataFrame, transform=None, num_frames: int = 8, clip_transform=None, *,
                 root=None, check_files: bool = True, **kwargs):
        super().__init__(df, transform, num_frames, clip_transform, **kwargs)
        base = Path(root) if root is not None else None
        paths = []
        for row, value in enumerate(df['videos']):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Video row {row}: no file path.')
            path = Path(value).expanduser()
            if base is not None and not path.is_absolute():
                path = base / path
            paths.append(str(path))
        if check_files:
            missing = [p for p in paths if not os.path.isfile(p)]
            if missing:
                raise ValueError(f'{len(missing)} of {len(paths)} video files are missing, e.g. {missing[0]}')
        self.paths = np.asarray(paths, dtype=object)

    @classmethod
    def from_manifest(cls, manifest, transform=None, num_frames: int = 8, clip_transform=None, **kwargs):
        from handlers.video_manifest import read_video_manifest
        return cls(read_video_manifest(manifest), transform, num_frames, clip_transform, **kwargs)

    def _source(self, i: int):
        return self.paths[i]


def open_video_dataset(source, transform=None, num_frames: int = 8, clip_transform=None, **kwargs):
    """Build the right dataset for a DataFrame or a parquet/CSV/XLSX file of either storage format."""
    if isinstance(source, DataFrame):
        storage = detect_video_storage(source)
        cls = VideoDataset if storage == 'paths' else ParquetVideoDataset
        return cls(source, transform, num_frames, clip_transform, **kwargs)
    path = Path(source)
    if path.suffix.lower() in ('.csv', '.xlsx'):
        return VideoDataset.from_manifest(path, transform, num_frames, clip_transform, **kwargs)
    if detect_video_storage(path) == 'paths':
        return VideoDataset(pd.read_parquet(path), transform, num_frames, clip_transform, root=path.parent, **kwargs)
    if path.stat().st_size > LAZY_PARQUET_BYTES:
        return ParquetVideoDataset._lazy_from_parquet(path, transform, num_frames, clip_transform, **kwargs)
    return ParquetVideoDataset(pd.read_parquet(path), transform, num_frames, clip_transform, **kwargs)


VAL_WORKERS_WINDOWS = 2


def video_loader_kwargs(dataset, batch_size: int, shuffle: bool, num_workers='auto', role: Optional[str] = None) -> dict:
    """DataLoader keyword arguments for a video dataset: worker count (see
    options.resolve_num_workers), pinned memory on CUDA, the None-label-safe
    collator, and bytes moved out of worker pickles.

    `role` (default: 'train' when shuffling, else 'once'):
    'train' -- persistent workers (iterated every epoch);
    'val' -- iterated every epoch too, so persistent, but on Windows capped at
    VAL_WORKERS_WINDOWS workers, so train + val keep at most 4 + 2 spawned
    processes alive instead of 8 (A2 V-019);
    'once' -- evaluation, embedding, inference, error-model fit: iterated once,
    so not persistent (the pool exits as soon as the pass ends)."""
    import sys
    from options import DEVICE, resolve_num_workers
    role = role or ('train' if shuffle else 'once')
    if role not in ('train', 'val', 'once'):
        raise ValueError("role must be 'train', 'val' or 'once'.")
    try:
        size = len(dataset)
    except TypeError:
        size = 0
    workers = resolve_num_workers(num_workers, size)
    if role == 'val' and sys.platform.startswith('win') and isinstance(num_workers, str):
        workers = min(workers, VAL_WORKERS_WINDOWS)
    if workers > 0:
        prepare = getattr(dataset, 'prepare_for_workers', None)
        if callable(prepare):
            prepare()
    return dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=DEVICE.type == 'cuda',
                persistent_workers=workers > 0 and role != 'once', collate_fn=collate_video_samples)

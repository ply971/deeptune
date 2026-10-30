"""Train and evaluate on cached snippet features instead of pixels (P-102).

:func:`open_feature_source` reads a manifest (CSV/XLSX), a parquet split of
either storage format, or a DataFrame, and maps every row to its feature
``video_id``: the ``video_id`` column if there is one, else a name derived
from the file path (``datasets.snippet_features.video_id_for_path``), else, for
clip bytes, the sha1 of the bytes (so such rows are read once).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.snippet_features import FeatureStore, safe_video_id, video_id_for_path
from datasets.video_datasets import ParquetVideoDataset, VideoDataset, detect_video_storage, open_video_dataset


def _finite_or_none(values, i: int):
    return None if values is None or not np.isfinite(values[i]) else float(values[i])


def open_feature_source(source):
    """(video dataset, its non-clip columns). Labels may be class names here: nothing is trained on this dataset."""
    root = None
    if isinstance(source, pd.DataFrame):
        table = source
    else:
        path = Path(source)
        root = path.parent
        if path.suffix.lower() == '.csv':
            table = pd.read_csv(path)
        elif path.suffix.lower() == '.xlsx':
            table = pd.read_excel(path)
        elif detect_video_storage(path) == 'paths':
            table = pd.read_parquet(path)
        else:  # bytes parquet: the dataset memory-maps large files instead of loading them
            dataset = open_video_dataset(path, num_frames=1)
            return dataset, dataset.data
    columns = table.drop(columns=['videos'], errors='ignore')
    clips = table.drop(columns=['labels'], errors='ignore')
    if detect_video_storage(table) == 'paths':
        return VideoDataset(clips, num_frames=1, root=root), columns
    return ParquetVideoDataset(clips, num_frames=1), columns


def row_video_id(dataset, columns: pd.DataFrame, i: int) -> str:
    if 'video_id' in columns.columns and pd.notna(columns['video_id'].iloc[i]):
        return safe_video_id(columns['video_id'].iloc[i])
    if isinstance(dataset, VideoDataset):
        return video_id_for_path(dataset.paths[i], _finite_or_none(dataset._starts, i), _finite_or_none(dataset._ends, i))
    return dataset.clip_key(i)[:16]


def snippet_indices(available: int, wanted: int, train: bool, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """`wanted` snippet positions out of `available`: evenly spaced, or one random per segment when
    training. Short videos repeat positions, so every sample has the same shape."""
    if available < 1 or wanted < 1:
        raise ValueError('Need at least one snippet.')
    if not train:
        return np.linspace(0, available - 1, wanted).round().astype(np.int64)
    edges = np.linspace(0, available, wanted + 1)
    low = np.floor(edges[:-1]).astype(np.int64)
    high = np.maximum(low + 1, np.floor(edges[1:]).astype(np.int64))
    return rng.integers(low, high)


class CachedFeatureDataset(Dataset):
    """``(features, label)`` per video, features float32 ``[K, D]`` or ``[K, t, h, w, D]``.

    `source`: the rows to use (for example a train split). `streams`: default
    every stream in the folder, concatenated on the last axis. `num_snippets`:
    K. Construction checks that every row was extracted and reads no features.
    """

    def __init__(self, source, features_dir, streams: Optional[Sequence[str]] = None, num_snippets: int = 8,
                 train: bool = False, target: str = 'labels'):
        if num_snippets < 1:
            raise ValueError('num_snippets must be at least 1.')
        self.store = FeatureStore(features_dir)
        self.streams = list(streams) if streams else self.store.streams()
        if not self.streams:
            raise FileNotFoundError(f'{features_dir} has no extracted features; run extract_features.py first.')
        self.num_snippets = int(num_snippets)
        self.train = bool(train)
        dataset, columns = open_feature_source(source)
        if target not in columns.columns:
            raise ValueError(f'The feature dataset needs a {target!r} column with the labels.')
        if not pd.api.types.is_numeric_dtype(columns[target]):
            raise ValueError(f'Column {target!r} must be numeric (class indices or regression targets); got '
                             f'{columns[target].dtype}. Use an encoded column such as label_idx (--target).')
        self.labels = columns[target].to_numpy()
        self.float_labels = bool(pd.api.types.is_float_dtype(columns[target]))
        self.video_ids = [row_video_id(dataset, columns, i) for i in range(len(dataset))]
        missing = [(stream, vid) for vid in self.video_ids for stream in self.streams
                   if not self.store.array_path(stream, vid).is_file()]
        if missing:
            raise ValueError(f'{len(missing)} of {len(self.video_ids) * len(self.streams)} feature files are missing in '
                             f'{features_dir} (first: {missing[0][0]}/{missing[0][1]}). Run extract_features.py on these videos.')

    def __len__(self):
        return len(self.video_ids)

    def _rng(self):
        return np.random.default_rng(int(torch.randint(0, 2 ** 62, (1,)).item())) if self.train else None

    def __getitem__(self, i):
        vid = self.video_ids[i]
        first = np.load(self.store.array_path(self.streams[0], vid), mmap_mode='r', allow_pickle=False)
        available = len(first)
        del first
        indices = snippet_indices(available, self.num_snippets, self.train, self._rng())
        features = np.array(self.store.load(vid, self.streams, indices=indices).features, dtype=np.float32)
        label = torch.tensor(self.labels[i], dtype=torch.float32 if self.float_labels else torch.long)
        return torch.from_numpy(features), label

    def feature_dim(self) -> int:
        """D of the concatenated streams, without reading any features."""
        streams = self.store.info()['streams']
        return sum(int(streams[name]['dim']) for name in self.streams)

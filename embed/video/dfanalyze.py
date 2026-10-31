"""Clip embeddings and their extras for the df-analyze export; handlers/dfanalyze_export.py writes the files.

Every split is sampled deterministically (`train=False`), with no augmentation, including the training rows. A feature must
be computed the same way on every row, or a model could tell the splits apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.feature_dataset import row_video_id
from datasets.video_datasets import native_video_transform, open_video_dataset, video_loader_kwargs, video_options_from_args
from datasets.video_io import VideoReader
from embed.video.embed import load_embedding_model
from handlers.dfanalyze_export import SplitBlocks, curve_statistics, write_dfanalyze_export
from helpers import transformations
from options import DEVICE, PIN_MEM, UNIQUE_ID
from src.video.multi_clip import multi_clip_scores
from src.video.multi_crop import wrap_for_test_crops


def motion_statistics(dataset, i: int, backend: str) -> dict:
    """Statistics of the frame-difference curve, on the row's start_sec to end_sec range. Always `framediff`: 'auto' takes
    codec motion vectors where it can, and a column must not change units from one video to the next."""
    with VideoReader(dataset._source(i), backend=backend) as reader:
        start = end = None
        if dataset._starts is not None and np.isfinite(dataset._starts[i]):
            start = reader.time_to_index(dataset._starts[i])
        if dataset._ends is not None and np.isfinite(dataset._ends[i]):
            end = reader.time_to_end_index(dataset._ends[i])
        curve = reader.motion_curve('framediff')
    return curve_statistics(curve[start:end])


def embed_split(model, path, model_version, num_frames, options, clips, batch_size, grouper=None, want_std=False, want_motion=False,
                device=DEVICE) -> SplitBlocks:
    """One split's blocks, in the split's row order. `model` takes one clip batch, `[B, T, C, H, W]`."""
    dataset = open_video_dataset(path, transform=transformations, num_frames=num_frames, clip_transform=native_video_transform(model_version),
                                 train=False, options=options, num_clips=clips)
    if not len(dataset):
        raise ValueError(f'{path} has no rows.')
    if not dataset.has_labels:
        raise ValueError(f"{path} has no 'labels' column; df-analyze needs a target.")
    means, stds = [], []
    loader = DataLoader(dataset, **video_loader_kwargs(dataset, batch_size, False, options.num_workers))
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'df-analyze export: {Path(path).stem}'):
            data = batch[0].to(device, non_blocking=PIN_MEM)
            mean, per_clip = multi_clip_scores(model, data, 'cls', extract_embed=True)
            means.append(mean.reshape(len(data), -1).cpu().numpy())
            if want_std:
                stds.append(per_clip.reshape(*per_clip.shape[:2], -1).std(dim=1, unbiased=False).cpu().numpy())
    columns = dataset.data
    column = grouper if grouper and grouper in columns.columns else None
    motion = None
    if want_motion:
        motion = pd.DataFrame([motion_statistics(dataset, i, options.video_backend) for i in tqdm(range(len(dataset)), desc='motion statistics')])
    return SplitBlocks(target=np.asarray(dataset.labels), video_ids=[row_video_id(dataset, columns, i) for i in range(len(dataset))],
                       embedding=np.vstack(means), embedding_std=np.vstack(stds) if want_std else None, motion=motion,
                       groups=columns[column].tolist() if column else None)


def export_from_args(args, splits: dict, model_weights, use_case: str, model_version: str, added_layers, embed_size, num_classes, mode,
                     out, model_str: str, batch_size: int, num_frames: int, pooling: str, grouper=None, main_split: str = 'test',
                     device=DEVICE) -> Path:
    """Embeds `splits` ({name: file}) with one model and writes the export. `main_split`: the rows of dfanalyze.csv by default."""
    options = video_options_from_args(args)
    extras = tuple(args.dfanalyze_extras or ())
    tta = 'tta' in extras
    clips = args.dfanalyze_clips if tta else options.test_clips
    model = load_embedding_model(model_weights, use_case, model_version, added_layers, embed_size, num_classes, args, mode, pooling)
    model = wrap_for_test_crops(model, options.test_crops, 'cls').to(device).eval()
    parts = {name: embed_split(model, path, model_version, num_frames, options, clips, batch_size, grouper, want_std=tta, want_motion='motion' in extras, device=device)
             for name, path in splits.items()}
    if mode is None:  # an embedding run may not name it: --mode is only needed to rebuild a head
        mode = 'reg' if np.issubdtype(next(iter(parts.values())).target.dtype, np.floating) else 'cls'
    export_dir = Path(out) / f'dfanalyze_export_{model_str}_{UNIQUE_ID}'
    rows = list(parts) if args.dfanalyze_rows == 'all' else [main_split]
    provenance = {'embedding_source': getattr(args, 'embedding_source', None), 'clips_per_video': int(clips), 'crops_per_clip': int(options.test_crops),
                  'sampling': 'deterministic test-time sampling on every split, no augmentation',
                  'motion_method': 'framediff' if 'motion' in extras else None, 'source_files': {name: str(path) for name, path in splits.items()}}
    write_dfanalyze_export(parts, export_dir, mode, args.dfanalyze_dims, extras, rows, 'train', use_case != 'pretrained', provenance)
    args.save_args(export_dir)
    return export_dir

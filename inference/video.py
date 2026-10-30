"""Inference for DeepTune's video modality. Mirrors inference/vision.py --
video models built by utils.build_video_model expose the same
forward(x) -> [B, num_classes] API as the image models, whether they're a
native spatio-temporal architecture or a frame-sampling wrapper around a 2D
backbone -- but need ParquetVideoDataset's clip decoding/sampling and
build_video_model's model construction instead of images' equivalents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.video_datasets import (VideoDataOptions, native_video_transform, open_video_dataset, video_loader_kwargs)
from helpers import transformations
from inference.common import attach_expected_error
from options import DEVICE, PIN_MEM
from src.video.multi_clip import clip_disagreement, multi_clip_scores
from utils import build_video_model


def predict_video(
    df: pd.DataFrame,
    checkpoint_path: Path,
    model_version: str,
    num_classes: int,
    added_layers: int,
    embed_size: int,
    freeze_backbone: bool,
    mode: str,
    use_peft: bool,
    batch_size: int,
    num_frames: int = 8,
    pooling: str = 'mean',
    label_mapping: Optional[dict] = None,
    video_options: Optional[VideoDataOptions] = None,
) -> pd.DataFrame:
    """Runs `checkpoint_path`'s model over every row of `df` (a 'videos'
    column of file paths or clip bytes, detected automatically, + optional
    'labels'). Returns the same shape of predictions DataFrame as
    inference.vision.predict_images.

    `video_options` carries the run's recorded data-layer settings
    (sampling, backend, --test_clips, workers). With test_clips > 1 the
    clip softmax probabilities are averaged *before* the error model sees
    them, and a ``clip_disagreement`` column is added (0 = all clips agree)."""
    video_options = video_options or VideoDataOptions()
    model = build_video_model(
        model_version=model_version, num_classes=num_classes, added_layers=added_layers,
        embed_size=embed_size, freeze_backbone=freeze_backbone, mode=mode,
        use_peft=use_peft, pooling=pooling,
    )
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # See inference.vision.predict_images's matching comment: ParquetVideoDataset
    # needs 'labels' to already be numeric (or absent) to convert it to a
    # tensor, but inference.data's raw-folder loader produces class *names*
    # -- dropped here, and matched back in afterwards by name instead.
    raw_labels = df['labels'].tolist() if 'labels' in df.columns else None
    dataset = open_video_dataset(df.drop(columns=['labels']) if raw_labels is not None else df,
                                 transform=transformations, num_frames=num_frames,
                                 clip_transform=native_video_transform(model_version),
                                 options=video_options, num_clips=video_options.test_clips)
    loader = DataLoader(dataset, **video_loader_kwargs(dataset, batch_size, False, video_options.num_workers))

    sample_ids = df['__sample_id'].tolist() if '__sample_id' in df.columns else list(range(len(df)))
    index_to_name = {v: k for k, v in (label_mapping or {}).items()}

    predicted, confidences, scores, disagreement = [], [], [], []
    with torch.no_grad():
        for inputs, _labels, *_ in loader:
            inputs = inputs.to(DEVICE, non_blocking=PIN_MEM)
            outputs, per_clip = multi_clip_scores(model, inputs, mode)
            if per_clip is not None:
                disagreement.extend(clip_disagreement(per_clip.float(), mode).cpu().tolist())
            if mode == 'cls':
                # multi_clip_scores already returns averaged probabilities for [B, N, ...] input.
                probs = outputs if per_clip is not None else torch.softmax(outputs, dim=1)
                conf, pred = probs.max(dim=1)
                predicted.extend(pred.cpu().tolist())
                confidences.extend(conf.cpu().tolist())
                scores.append(probs.detach().float().cpu())
            else:
                predicted.extend(outputs.detach().cpu().view(-1).tolist())
                confidences.extend([None] * outputs.size(0))
                scores.append(outputs.detach().float().cpu())

    result = pd.DataFrame({'sample_id': sample_ids[:len(predicted)]})
    if mode == 'cls':
        result['predicted_class'] = predicted
        result['predicted_label'] = [index_to_name.get(int(p), str(int(p))) for p in predicted]
        result['confidence'] = confidences
    else:
        result['predicted_value'] = predicted
    if disagreement:
        result['clip_disagreement'] = disagreement[:len(predicted)]

    if raw_labels is not None:
        raw_labels = raw_labels[:len(predicted)]
        if mode == 'cls':
            result['true_label'] = raw_labels
            result['correct'] = [str(t) == str(p) for t, p in zip(raw_labels, result['predicted_label'])]
        else:
            result['true_value'] = raw_labels
            result['abs_error'] = [abs(float(t) - float(p)) for t, p in zip(raw_labels, predicted)]
    # Per-prediction error estimate (A4): only when the run saved error_model.json.
    if scores:
        attach_expected_error(result, mode, checkpoint_path, torch.cat(scores).numpy())
    return result

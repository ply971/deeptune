"""Decode once, extract features once: cached snippet features for every video.

    python extract_features.py --data clips.csv --out features --model_version r3d_18 --streams rgb motion

Each video is split into snippets; a frozen backbone runs on each snippet once;
the result is written per video. Finished videos are skipped, so an interrupted
run resumes where it stopped. The folder layout, snippet formula and float16
checks are documented in ``datasets/snippet_features.py``; flags are in
``handlers/video_args.py``.

Streams:

* ``rgb``: decoded frames through ``--model_version`` (any DeepTune video or
  frame backbone), with that backbone's usual preprocessing.
* ``motion``: dx, dy per frame through ``--motion_backbone`` with its first
  convolution inflated to 2 channels. The field comes from
  ``VideoReader.get_motion_vectors`` (codec vectors, Farneback flow as the
  fallback), is rotated to the display orientation, resized to the input size,
  rescaled to input pixels per frame, clipped to +-20 and divided by 20, as is
  usual for flow inputs. Frames added by padding have zero motion.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from datasets.snippet_features import (FAILED_FILE, FORMAT_VERSION, STREAMS, FeatureStore, config_hash, parse_token_grid,
                                       snippet_count, snippet_starts, to_float16, write_json_atomic)
from handlers.video_args import SNIPPET_FEATURE_ARG_NAMES, add_snippet_feature_arguments

LOGGER = logging.getLogger('deeptune.video')

MOTION_CLIP_PIXELS = 20.0
MOTION_NORMALISATION = f'input pixels per frame, clipped to +-{MOTION_CLIP_PIXELS:g}, divided by {MOTION_CLIP_PIXELS:g}'
BLOCK_SIZE = 8
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
KINETICS_MEAN, KINETICS_STD = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract cached snippet features from videos with a frozen backbone.')
    add_snippet_feature_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# Snippets from a decoder
# ---------------------------------------------------------------------------
def iter_rgb_snippets(frames, snippet_frames: int, stride: int):
    """Yield ``(uint8 [S, H, W, 3], valid_frames)`` from an iterator over a video's frames, streaming:
    at most `snippet_frames` frames are held. Counts follow ``snippet_count``."""
    buffer, start, seen, last = [], 0, 0, None
    for frame in frames:
        index, last = seen, frame
        seen += 1
        if index >= start:
            buffer.append((index, frame))
        while seen >= start + snippet_frames:
            yield np.stack([f for i, f in buffer if start <= i < start + snippet_frames]), snippet_frames
            start += stride
            buffer = [(i, f) for i, f in buffer if i >= start]
    emitted = start // stride
    for k in range(emitted, snippet_count(seen, snippet_frames, stride)):
        start = k * stride
        kept = [f for i, f in buffer if start <= i < start + snippet_frames]
        padded = kept + [last] * (snippet_frames - len(kept))
        yield np.stack(padded), len(kept)


def rotate_motion_field(field_: np.ndarray, rotation: int) -> np.ndarray:
    """Turn a ``[T, 2, h, w]`` stored-orientation field into the display orientation; rotation is
    counter-clockwise, as for ``VideoReader`` frames. Vectors turn with the grid: 90 degrees maps
    (dx, dy) to (dy, -dx)."""
    turns = int(round(rotation / 90.0)) % 4
    if not turns:
        return field_
    grid = np.rot90(field_, k=turns, axes=(2, 3))
    dx, dy = grid[:, 0], grid[:, 1]
    if turns == 1:
        vectors = (dy, -dx)
    elif turns == 2:
        vectors = (-dx, -dy)
    else:
        vectors = (-dy, dx)
    return np.stack(vectors, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class _RgbPreprocess:
    """uint8 ``[S, H, W, 3]`` -> float ``[S, 3, H', W']`` the way training preprocesses this backbone,
    or a plain resize + normalise when `input_size` overrides the size."""

    def __init__(self, version: str, native: bool, input_size: Optional[int]):
        import torch
        self.torch = torch
        self.input_size = input_size
        self.mean = torch.tensor(KINETICS_MEAN if native else IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(KINETICS_STD if native else IMAGENET_STD).view(1, 3, 1, 1)
        self.clip_transform = self.frame_transform = None
        if input_size is None:
            from datasets.video_datasets import native_video_transform
            self.clip_transform = native_video_transform(version)
            if self.clip_transform is None:
                from helpers import transformations
                self.frame_transform = transformations

    def __call__(self, frames: np.ndarray):
        torch = self.torch
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
        if self.clip_transform is not None:
            return self.clip_transform(clip)
        if self.frame_transform is not None:
            from PIL import Image
            return torch.stack([self.frame_transform(Image.fromarray(frame)) for frame in frames])
        clip = torch.nn.functional.interpolate(clip.float() / 255.0, size=(self.input_size, self.input_size),
                                               mode='bilinear', align_corners=False, antialias=True)
        return (clip - self.mean) / self.std


def _motion_to_tensor(motion: np.ndarray, display_size: tuple[int, int], input_size: int):
    """``[S, 2, h, w]`` pixels/frame at the display size -> float ``[S, 2, input, input]`` in [-1, 1]."""
    import torch
    width, height = display_size
    tensor = torch.from_numpy(np.ascontiguousarray(motion))
    tensor = torch.nn.functional.interpolate(tensor, size=(input_size, input_size), mode='bilinear', align_corners=False)
    scale = torch.tensor([input_size / width, input_size / height], dtype=tensor.dtype).view(1, 2, 1, 1)
    return (tensor * scale).clamp(-MOTION_CLIP_PIXELS, MOTION_CLIP_PIXELS) / MOTION_CLIP_PIXELS


# ---------------------------------------------------------------------------
# One stream
# ---------------------------------------------------------------------------
class _StreamExtractor:
    """Builds its backbone on first use, so a run with nothing to do loads no model and decodes nothing."""

    def __init__(self, stream: str, config: dict, requested_grid, batch_size: int, device):
        self.stream = stream
        self.config = config
        self.digest = config_hash(config)
        self.requested_grid = requested_grid
        self.batch_size = batch_size
        self.device = device
        self.backbone = None
        self.preprocess = None
        self.stored_grid = None

    def _ensure_backbone(self):
        if self.backbone is not None:
            return
        from src.video.feature_backbones import SnippetBackbone
        channels = 3 if self.stream == 'rgb' else 2
        backbone = SnippetBackbone(self.config['backbone'], pretrained=self.config['weights'] == 'pretrained',
                                   in_channels=channels)
        if backbone.weights_name is None:
            LOGGER.warning('The %s stream uses random weights (%s): useful for tests and timing, not for accuracy.',
                           self.stream, self.config['backbone'])
        self.backbone = backbone.to(self.device)
        self.input_size = self.config['input_size'] or backbone.default_input_size
        if self.stream == 'rgb':
            self.preprocess = _RgbPreprocess(self.config['backbone'], backbone.native, self.config['input_size'])

    def _forward(self, batch: list):
        import torch
        from src.video.feature_backbones import pool_token_grid, resolve_token_grid
        with torch.inference_mode():
            grid = self.backbone.token_grid(torch.stack(batch).to(self.device))
            self.stored_grid = resolve_token_grid(self.requested_grid, tuple(grid.shape[1:4]))
            pooled = pool_token_grid(grid, self.stored_grid)
        return pooled.float().cpu().numpy()

    def _run_batches(self, tensors) -> np.ndarray:
        outputs, batch = [], []
        for tensor in tensors:
            batch.append(tensor)
            if len(batch) == self.batch_size:
                outputs.append(self._forward(batch))
                batch = []
        if batch:
            outputs.append(self._forward(batch))
        if not outputs:
            from datasets.video_io import VideoDecodeError
            raise VideoDecodeError('The video has no decodable frames in the requested range.')
        return np.concatenate(outputs)

    def extract_rgb(self, reader, lo: int, hi: Optional[int]) -> tuple[np.ndarray, dict]:
        self._ensure_backbone()
        snippet_frames, stride = self.config['snippet_frames'], self.config['stride']
        info = {'frame_count': 0, 'last_snippet_valid_frames': 0}

        def frames():
            for index, frame in enumerate(reader.iter_frames()):
                if index < lo:
                    continue
                if hi is not None and index >= hi:
                    break
                info['frame_count'] += 1
                yield frame

        def tensors():
            for snippet, valid in iter_rgb_snippets(frames(), snippet_frames, stride):
                info['last_snippet_valid_frames'] = valid
                yield self.preprocess(snippet)

        return self._run_batches(tensors()), info

    def extract_motion(self, reader, lo: int, frame_count: int) -> tuple[np.ndarray, dict]:
        self._ensure_backbone()
        snippet_frames, stride = self.config['snippet_frames'], self.config['stride']
        meta = reader.metadata()
        starts = snippet_starts(frame_count, snippet_frames, stride)
        end = min(lo + frame_count, meta.frame_count)
        info = {'frame_count': frame_count, 'motion_normalisation': MOTION_NORMALISATION, 'motion_source': None,
                'last_snippet_valid_frames': int(min(snippet_frames, frame_count - starts[-1]))}
        intra = [0, 0]

        def tensors():
            for chunk in range(0, len(starts), self.batch_size):
                chosen = starts[chunk:chunk + self.batch_size]
                wanted = (lo + chosen[:, None] + np.arange(snippet_frames)).reshape(-1)
                inside = wanted < end
                motion = reader.get_motion_vectors(wanted[inside], block_size=BLOCK_SIZE, method=self.config['motion_method'])
                info['motion_source'] = motion.source
                intra[0] += int(np.count_nonzero(motion.is_intra))
                intra[1] += len(motion.is_intra)
                field_ = rotate_motion_field(motion.field, meta.rotation)
                filled = np.zeros((len(wanted), *field_.shape[1:]), dtype=np.float32)  # padded frames: no motion
                filled[inside] = field_
                for snippet in filled.reshape(len(chosen), snippet_frames, *field_.shape[1:]):
                    yield _motion_to_tensor(snippet, (meta.width, meta.height), self.input_size)

        features = self._run_batches(tensors())
        info['intra_fraction'] = intra[0] / max(1, intra[1])
        return features, info


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------
@dataclass
class ExtractionReport:
    extracted: dict = field(default_factory=lambda: {name: 0 for name in STREAMS})
    skipped: dict = field(default_factory=lambda: {name: 0 for name in STREAMS})
    failed: list = field(default_factory=list)
    seconds: float = 0.0
    bytes_written: int = 0


def _frame_range(reader, start_sec, end_sec) -> tuple[int, Optional[int]]:
    lo = reader.time_to_index(start_sec) if start_sec is not None else 0
    hi = reader.time_to_end_index(end_sec) if end_sec is not None else None
    if hi is not None and hi <= lo:
        raise ValueError(f'end_sec ({end_sec}) must be after start_sec ({start_sec}).')
    return lo, hi


def _range_frame_count(reader, lo: int, hi: Optional[int]) -> int:
    meta = reader.metadata()
    total = meta.frame_count if meta.frame_count_exact else sum(1 for _ in reader.iter_frames(width=32, gray=True))
    return (total if hi is None else min(hi, total)) - lo


def _snippet_times(reader, lo: int, starts: np.ndarray) -> Optional[list]:
    if not reader.metadata().variable_fps:
        return None
    times = reader.frame_times()
    if times is None or not len(times):
        return None
    return [float(times[min(int(lo + s), len(times) - 1)]) for s in starts]


def extract_features(data, out, streams: Sequence[str] = ('rgb',), model_version: str = 'r3d_18',
                     motion_backbone: str = 'r3d_18', feature_weights: str = 'pretrained', snippet_frames: int = 16,
                     snippet_stride: Optional[int] = None, cache_tokens: str = 'none', batch_size: int = 8,
                     motion_method: str = 'auto', feature_input_size: Optional[int] = None, video_backend: str = 'auto',
                     device=None) -> ExtractionReport:
    """Extract every requested stream of every video in `data` (manifest, parquet or DataFrame) into `out`."""
    from datasets.feature_dataset import open_feature_source, row_video_id
    from datasets.video_io import MotionVectorsUnavailable, VideoReader, describe_source, source_fingerprint
    if device is None:
        from options import DEVICE as device
    streams = [name for name in STREAMS if name in streams]
    if not streams:
        raise ValueError(f'Choose at least one stream from {", ".join(STREAMS)}.')
    stride = snippet_stride or snippet_frames
    grid = parse_token_grid(cache_tokens)
    store = FeatureStore(out)
    extractors = {}
    for stream in streams:
        config = {'format': FORMAT_VERSION, 'stream': stream, 'snippet_frames': snippet_frames, 'stride': stride,
                  'backbone': model_version if stream == 'rgb' else motion_backbone, 'weights': feature_weights,
                  'input_size': feature_input_size, 'token_grid': list(grid) if grid else None,
                  'video_backend': video_backend, 'motion_method': motion_method if stream == 'motion' else None}
        store.check_config(stream, config)
        extractors[stream] = _StreamExtractor(stream, config, grid, batch_size, device)
    dataset, columns = open_feature_source(data)
    report = ExtractionReport()
    began = time.perf_counter()
    seen: dict = {}
    registered = set()

    def extract_one(stream, reader, vid, source, fingerprint, lo, hi, frames_known):
        extractor, clock = extractors[stream], time.perf_counter()
        if stream == 'rgb':
            features, info = extractor.extract_rgb(reader, lo, hi)
        else:
            count = frames_known if frames_known is not None else _range_frame_count(reader, lo, hi)
            features, info = extractor.extract_motion(reader, lo, count)
        stored, stats = to_float16(features, f'{vid} ({stream})')
        count = info['frame_count']
        expected = snippet_count(count, snippet_frames, stride)
        if len(stored) != expected:  # a bug here would silently misalign features and time
            raise RuntimeError(f'{vid}: {len(stored)} snippets from {count} frames, the formula gives {expected}.')
        meta = reader.metadata()
        sidecar = {
            'format_version': FORMAT_VERSION, 'video_id': vid, 'stream': stream, 'source': describe_source(source),
            'source_fingerprint': fingerprint, 'config_hash': extractor.digest, 'fps': meta.fps, 'frame_count': count,
            'start_frame': lo, 'end_frame': lo + count, 'duration_sec': count / meta.fps if meta.fps else None,
            'variable_fps': bool(meta.variable_fps), 'snippet_frames': snippet_frames, 'stride': stride,
            'num_snippets': len(stored), 'last_snippet_valid_frames': info['last_snippet_valid_frames'],
            'backbone': extractor.config['backbone'], 'weights': extractor.backbone.weights_name,
            'weights_licence': extractor.backbone.weights_licence, 'input_size': extractor.input_size,
            'dim': int(stored.shape[-1]), 'token_grid': list(extractor.stored_grid) if extractor.stored_grid else None,
            'float16': stats, 'seconds': time.perf_counter() - clock,
        }
        if stream == 'motion':
            sidecar.update(motion_source=info['motion_source'], intra_fraction=info['intra_fraction'],
                           motion_normalisation=info['motion_normalisation'])
        times = _snippet_times(reader, lo, snippet_starts(count, snippet_frames, stride))
        if times is not None:
            sidecar['snippet_start_sec'] = times
        if stream not in registered:
            store.register_stream(stream, extractor.config, stored.shape[-1], extractor.stored_grid)
            registered.add(stream)
        store.write(stream, vid, stored, sidecar)
        report.extracted[stream] += 1
        report.bytes_written += store.array_path(stream, vid).stat().st_size
        return count

    for i in range(len(dataset)):
        vid = row_video_id(dataset, columns, i)
        source = dataset._source(i)
        fingerprint = source_fingerprint(source)
        if vid in seen:
            if seen[vid] != fingerprint:
                raise ValueError(f'Two different videos have the id {vid!r}; video_id values must be unique.')
            continue
        seen[vid] = fingerprint
        todo = [s for s in streams if not store.is_complete(s, vid, fingerprint, extractors[s].digest)]
        for stream in streams:
            if stream not in todo:
                report.skipped[stream] += 1
        if not todo:
            continue
        frames_known = None
        rgb_sidecar = store.read_sidecar('rgb', vid)  # the frames actually decoded, so both streams have the same N
        if rgb_sidecar and rgb_sidecar.get('source_fingerprint') == fingerprint:
            frames_known = rgb_sidecar.get('frame_count')
        try:
            with VideoReader(source, backend=video_backend) as reader:
                lo, hi = _frame_range(reader, _value(dataset._starts, i), _value(dataset._ends, i))
                for stream in todo:
                    count = extract_one(stream, reader, vid, source, fingerprint, lo, hi, frames_known)
                    frames_known = count if stream == 'rgb' else frames_known
        except (ValueError, IndexError, MotionVectorsUnavailable) as exc:  # includes VideoDecodeError, FeatureValueError
            LOGGER.error('Video %s failed: %s', vid, exc)
            report.failed.append({'video_id': vid, 'source': describe_source(source), 'error': str(exc)})
    report.seconds = time.perf_counter() - began
    failed_path = Path(out) / FAILED_FILE
    if report.failed:
        write_json_atomic(failed_path, {'failed': report.failed})
    elif failed_path.exists():
        failed_path.unlink()
    return report


def _value(values, i: int):
    return None if values is None or not np.isfinite(values[i]) else float(values[i])


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parse_token_grid(args.cache_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(levelname)s | %(message)s')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(args.out) / 'cli_arguments.json', {name: getattr(args, name) for name in SNIPPET_FEATURE_ARG_NAMES})
    report = extract_features(args.data, args.out, streams=args.streams, model_version=args.model_version,
                              motion_backbone=args.motion_backbone, feature_weights=args.feature_weights,
                              snippet_frames=args.snippet_frames, snippet_stride=args.snippet_stride,
                              cache_tokens=args.cache_tokens, batch_size=args.batch_size, motion_method=args.motion_method,
                              feature_input_size=args.feature_input_size, video_backend=args.video_backend)
    print(json.dumps({'extracted': report.extracted, 'skipped': report.skipped, 'failed': len(report.failed),
                      'seconds': round(report.seconds, 2), 'megabytes_written': round(report.bytes_written / 1e6, 3)}))
    if report.failed:
        print(f'{len(report.failed)} video(s) failed; see {Path(args.out) / FAILED_FILE}. Run again to retry them.')
    return 1 if report.failed else 0


if __name__ == '__main__':
    sys.exit(main())

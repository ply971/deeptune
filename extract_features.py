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
* ``motion``: dx, dy and magnitude per frame through the unchanged RGB filters
  of ``--motion_backbone`` (not trained on motion). The field comes from
  ``VideoReader.get_motion_vectors`` (codec vectors, Farneback flow as the
  fallback), is rotated to the display orientation, cut to the part of the frame
  the rgb stream keeps (native presets centre-crop), resized to the input size,
  rescaled to input pixels per frame, clipped to +-20 and multiplied by
  ``--motion_gain``. Frames added by padding have zero motion.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from datasets.snippet_features import (FAILED_FILE, FORMAT_VERSION, PREPROCESSING_VERSION, STREAMS, FeatureConfigError,
                                       FeatureStore, config_hash, parse_token_grid, snippet_count, snippet_starts, to_float16,
                                       write_json_atomic)
from handlers.video_args import SNIPPET_FEATURE_ARG_NAMES, add_snippet_feature_arguments

LOGGER = logging.getLogger('deeptune.video')

MOTION_CLIP_PIXELS = 20.0
MOTION_GAIN = 1.0 / MOTION_CLIP_PIXELS
FULL_FRAME = (0.0, 1.0, 0.0, 1.0)
BLOCK_SIZE = 8
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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
    or a plain resize + normalise with this backbone's own statistics when `input_size` overrides the size."""

    def __init__(self, version: str, native: bool, input_size: Optional[int]):
        import torch
        self.torch = torch
        self.input_size = input_size
        from src.video.video3d import _KINETICS400_WEIGHTS
        # Each native model has its own statistics (r3d 0.43/0.22, mvit 0.45/0.225, swin3d ImageNet). dummy_video3d is r3d_18.
        preset = _KINETICS400_WEIGHTS[version if version in _KINETICS400_WEIGHTS else 'r3d_18'].transforms() if native else None
        self.mean = torch.tensor(preset.mean if preset else IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(preset.std if preset else IMAGENET_STD).view(1, 3, 1, 1)
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


def crop_window(version: Optional[str], input_size: Optional[int], width: int, height: int) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1), as fractions of the display frame: the part `_RgbPreprocess` keeps for this backbone.
    A size override and frame backbones squash the whole frame; native presets resize, then centre-crop."""
    from src.video.video3d import _KINETICS400_WEIGHTS, VIDEO3D_ARCHITECTURES
    if input_size is not None or version not in VIDEO3D_ARCHITECTURES:
        return FULL_FRAME
    preset = _KINETICS400_WEIGHTS[version].transforms()
    crop_h, crop_w = preset.crop_size
    if len(preset.resize_size) == 2:
        resized_h, resized_w = preset.resize_size
    elif width <= height:  # one number is the short side, with torchvision's rounding
        resized_w, resized_h = preset.resize_size[0], int(preset.resize_size[0] * height / width)
    else:
        resized_h, resized_w = preset.resize_size[0], int(preset.resize_size[0] * width / height)
    top, left = int(round((resized_h - crop_h) / 2.0)), int(round((resized_w - crop_w) / 2.0))
    return left / resized_w, (left + crop_w) / resized_w, top / resized_h, (top + crop_h) / resized_h


def _motion_to_tensor(motion: np.ndarray, display_size: tuple[int, int], input_size: int, window=FULL_FRAME, gain: float = MOTION_GAIN):
    """``[S, 2, h, w]`` pixels/frame at the display size -> float ``[S, 2, input, input]``: `window` of the frame
    (see `crop_window`) at the input size, in input pixels per frame, clipped, times `gain`."""
    import torch
    width, height = display_size
    x0, x1, y0, y1 = window
    tensor = torch.from_numpy(np.ascontiguousarray(motion))
    centres = (torch.arange(input_size, dtype=tensor.dtype) + 0.5) / input_size
    grid_y, grid_x = torch.meshgrid(2 * (y0 + (y1 - y0) * centres) - 1, 2 * (x0 + (x1 - x0) * centres) - 1, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(len(tensor), -1, -1, -1)
    tensor = torch.nn.functional.grid_sample(tensor, grid, mode='bilinear', padding_mode='border', align_corners=False)
    scale = torch.tensor([input_size / (width * (x1 - x0)), input_size / (height * (y1 - y0))], dtype=tensor.dtype).view(1, 2, 1, 1)
    return (tensor * scale).clamp(-MOTION_CLIP_PIXELS, MOTION_CLIP_PIXELS) * gain


def check_feature_input(stream: str, version: str, pretrained: bool, input_size: Optional[int], snippet_frames: int) -> None:
    """Fail before decoding when a model with a fixed input could never run: inside the model this is an
    AssertionError or a shape RuntimeError that ends the run and says nothing about the flags."""
    from src.video.feature_backbones import fixed_input, frame_family
    size, frames = fixed_input(version, pretrained)
    # Without an override rgb frame backbones get the shared 224 transform; native presets and motion get the model's own size.
    fed = input_size or (224 if stream == 'rgb' and frame_family(version) else None)
    weights = 'its pretrained weights' if pretrained else 'no weights'
    if size is not None and fed is not None and fed != size:
        raise FeatureConfigError(f'{version} with {weights} only accepts {size}x{size} input, but the {stream} stream would '
                                 f'feed {fed}x{fed}. Use --feature_input_size {size}.')
    if frames is not None and snippet_frames != frames:
        raise FeatureConfigError(f'{version} only accepts snippets of exactly {frames} frames, not {snippet_frames}. '
                                 f'Use --snippet_frames {frames}.')


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
                tensor = self.preprocess(snippet)
                self.input_size = int(tensor.shape[-1])
                yield tensor

        return self._run_batches(tensors()), info

    def extract_motion(self, reader, lo: int, frame_count: int) -> tuple[np.ndarray, dict]:
        self._ensure_backbone()
        snippet_frames, stride = self.config['snippet_frames'], self.config['stride']
        meta = reader.metadata()
        starts = snippet_starts(frame_count, snippet_frames, stride)
        end = min(lo + frame_count, meta.frame_count)
        gain = self.config.get('motion_gain', MOTION_GAIN)
        info = {'frame_count': frame_count, 'motion_source': None,
                'motion_normalisation': f'input pixels per frame, clipped to +-{MOTION_CLIP_PIXELS:g}, times {gain:g}',
                'last_snippet_valid_frames': int(min(snippet_frames, frame_count - starts[-1]))}
        intra = [0, 0]
        codec_meta = None

        def tensors():
            nonlocal codec_meta
            for chunk in range(0, len(starts), self.batch_size):
                chosen = starts[chunk:chunk + self.batch_size]
                wanted = (lo + chosen[:, None] + np.arange(snippet_frames)).reshape(-1)
                inside = wanted < end
                motion = reader.get_motion_vectors(wanted[inside], block_size=BLOCK_SIZE, method=self.config['motion_method'])
                info['motion_source'] = motion.source
                intra[0] += int(np.count_nonzero(motion.is_intra))
                intra[1] += len(motion.is_intra)
                # Codec vectors always use PyAV's stored orientation. OpenCV and
                # TorchCodec metadata can omit the display rotation entirely.
                motion_meta = meta
                if motion.source == 'codec' and reader.backend != 'pyav':
                    if codec_meta is None:
                        from datasets.video_io import VideoReader
                        with VideoReader(reader.source, backend='pyav') as probe:
                            codec_meta = probe.metadata()
                    motion_meta = codec_meta
                field_ = rotate_motion_field(motion.field, motion_meta.rotation)
                filled = np.zeros((len(wanted), *field_.shape[1:]), dtype=np.float32)  # padded frames: no motion
                filled[inside] = field_
                for snippet in filled.reshape(len(chosen), snippet_frames, *field_.shape[1:]):
                    window = crop_window(self.config.get('geometry_backbone'), self.config.get('input_size'),
                                         motion_meta.width, motion_meta.height)
                    yield _motion_to_tensor(snippet, (motion_meta.width, motion_meta.height), self.input_size, window, gain)

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
                     device=None, record_arguments: Optional[dict] = None, motion_gain: float = MOTION_GAIN) -> ExtractionReport:
    """Extract every requested stream of every video in `data` (manifest, parquet or DataFrame) into `out`.

    `record_arguments` is written to ``cli_arguments.json`` once the settings and rows have been accepted, so a
    refused run leaves the record of the folder's real settings alone and an interrupted one still has it."""
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
        backbone = model_version if stream == 'rgb' else motion_backbone
        check_feature_input(stream, backbone, feature_weights == 'pretrained', feature_input_size, snippet_frames)
        config = {'format': FORMAT_VERSION, 'stream': stream, 'snippet_frames': snippet_frames, 'stride': stride,
                  'backbone': backbone, 'weights': feature_weights,
                  'input_size': feature_input_size, 'token_grid': list(grid) if grid else None,
                  'video_backend': video_backend, 'motion_method': motion_method if stream == 'motion' else None,
                  'preprocessing_version': PREPROCESSING_VERSION[stream]}
        if stream == 'motion':  # motion cuts the frame as the rgb backbone does, whatever its own backbone
            config.update(geometry_backbone=model_version, motion_gain=motion_gain, motion_clip_pixels=MOTION_CLIP_PIXELS)
        store.check_config(stream, config)
        extractors[stream] = _StreamExtractor(stream, config, grid, batch_size, device)
    dataset, columns = open_feature_source(data)
    report = ExtractionReport()
    began = time.perf_counter()
    registered = set()

    def extract_one(stream, reader, vid, source, fingerprint, source_range, lo, hi, frames_known):
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
            'format_version': FORMAT_VERSION, 'video_id': vid, 'stream': stream,
            'source': os.path.abspath(source) if isinstance(source, (str, os.PathLike)) else describe_source(source),
            'source_fingerprint': fingerprint, 'config_hash': extractor.digest, 'fps': meta.fps, 'frame_count': count,
            'source_range': source_range,
            'start_frame': lo, 'end_frame': lo + count, 'duration_sec': count / meta.fps if meta.fps else None,
            'variable_fps': bool(meta.variable_fps), 'snippet_frames': snippet_frames, 'stride': stride,
            'num_snippets': len(stored), 'last_snippet_valid_frames': info['last_snippet_valid_frames'],
            'backbone': extractor.config['backbone'], 'weights': extractor.backbone.weights_name,
            'weights_licence': extractor.backbone.weights_licence, 'input_size': extractor.input_size,
            'field_of_view': list(crop_window(extractor.config.get('geometry_backbone', extractor.config['backbone']),
                                              extractor.config['input_size'], meta.width, meta.height)),
            'dim': int(stored.shape[-1]), 'token_grid': list(extractor.stored_grid) if extractor.stored_grid else None,
            'float16': stats, 'seconds': time.perf_counter() - clock,
        }
        if stream == 'motion':
            sidecar.update(motion_source=info['motion_source'], intra_fraction=info['intra_fraction'],
                           motion_normalisation=info['motion_normalisation'], motion_gain=motion_gain,
                           motion_clip_pixels=MOTION_CLIP_PIXELS,
                           motion_encoding='dx, dy, magnitude through RGB filters',
                           motion_weights_trained_on_motion=False)
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

    # Checked before anything is decoded: one id cannot stand for two clips, wherever the clash is in the manifest.
    # Case folded, because NTFS and APFS would put Clip and clip in one file.
    # Every stream's folder, not only the requested ones: load would otherwise join rgb of Clip to motion of clip.
    on_disk = [{path.stem.casefold(): path.stem for path in (Path(out) / name).glob('*.json')} for name in STREAMS]
    rows: dict = {}
    for i in range(len(dataset)):
        vid = row_video_id(dataset, columns, i)
        identity = (source_fingerprint(dataset._source(i)), {'start_sec': _value(dataset._starts, i), 'end_sec': _value(dataset._ends, i)})
        known = rows.setdefault(vid.casefold(), (vid, i, *identity))
        if known[0] != vid:
            raise ValueError(f'The ids {known[0]!r} and {vid!r} differ only by case, and would share one feature file on '
                             'some file systems; video_id values must be unique ignoring case.')
        if known[2:] != identity:
            raise ValueError(f'Two different videos or trims have the id {vid!r}; video_id values must be unique.')
        for known_ids in on_disk:
            stored = known_ids.get(vid.casefold(), vid)
            if stored != vid:
                raise ValueError(f'{vid!r} would overwrite or mix with the features of {stored!r} in {out}; '
                                 'the ids differ only by case.')
    if record_arguments is not None:
        write_json_atomic(Path(out) / 'cli_arguments.json', record_arguments)

    for vid, i, fingerprint, source_range in rows.values():
        source = dataset._source(i)
        todo = [s for s in streams if not store.is_complete(s, vid, fingerprint, extractors[s].digest, source_range)]
        for stream in streams:
            if stream not in todo:
                report.skipped[stream] += 1
        if not todo:
            continue
        frames_known = None
        rgb_sidecar = store.read_sidecar('rgb', vid)  # the frames actually decoded, so both streams have the same N
        if (rgb_sidecar and rgb_sidecar.get('source_fingerprint') == fingerprint
                and rgb_sidecar.get('source_range') == source_range):
            frames_known = rgb_sidecar.get('frame_count')
        try:
            with VideoReader(source, backend=video_backend) as reader:
                lo, hi = _frame_range(reader, _value(dataset._starts, i), _value(dataset._ends, i))
                for stream in todo:
                    count = extract_one(stream, reader, vid, source, fingerprint, source_range, lo, hi, frames_known)
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
    report = extract_features(args.data, args.out, streams=args.streams, model_version=args.model_version,
                              motion_backbone=args.motion_backbone, feature_weights=args.feature_weights,
                              snippet_frames=args.snippet_frames, snippet_stride=args.snippet_stride,
                              cache_tokens=args.cache_tokens, batch_size=args.batch_size, motion_method=args.motion_method,
                              feature_input_size=args.feature_input_size, video_backend=args.video_backend, motion_gain=args.motion_gain,
                              record_arguments={name: getattr(args, name) for name in SNIPPET_FEATURE_ARG_NAMES})
    print(json.dumps({'extracted': report.extracted, 'skipped': report.skipped, 'failed': len(report.failed),
                      'seconds': round(report.seconds, 2), 'megabytes_written': round(report.bytes_written / 1e6, 3)}))
    if report.failed:
        print(f'{len(report.failed)} video(s) failed; see {Path(args.out) / FAILED_FILE}. Run again to retry them.')
    return 1 if report.failed else 0


if __name__ == '__main__':
    sys.exit(main())

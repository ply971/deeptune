"""Video decoding for DeepTune: one reader API over several backends.

``VideoReader(path_or_bytes)`` gives

* ``metadata()`` -> :class:`VideoMetadata` (fps, frame_count, duration, width,
  height, codec, ...),
* ``get_frames(indices)`` -> uint8 ``[T, H, W, 3]`` RGB, the **exact** frames
  at those 0-based presentation-order indices,
* ``get_motion_vectors(indices)`` -> :class:`MotionField`, a dense per-block
  motion field ``[T, 2, ceil(H/b), ceil(W/b)]`` read from the codec
  (CoViAR, Wu et al., CVPR 2018), with an automatic Farneback fallback,
* ``motion_curve()`` -> per-frame motion magnitude (for MGSampler).

Backends, in ``auto`` priority order:

1. ``torchcodec`` -- only if the package imports (optional, not in
   requirements.txt). Exact seeking (``seek_mode='exact'``), in-memory bytes.
2. ``pyav`` -- the default. In-memory bytes via ``io.BytesIO`` (no temp file),
   frame-exact reads (sequential decode for short spans; keyframe seek +
   presentation-timestamp matching for sparse reads in long videos), display
   rotation applied, and the only backend that exposes codec motion vectors.
3. ``opencv`` -- fallback. Always reads sequentially (``grab`` to skip,
   ``retrieve`` to keep) because ``CAP_PROP_POS_FRAMES`` seeking is not
   frame-exact on compressed video. Bytes need a temporary file.

Motion-vector conventions (see ``VideoReader.get_motion_vectors``):

* channel 0 = dx, channel 1 = dy, in **pixels of the stored frame**;
* sign = displacement of the content from the earlier frame to the later one
  (positive dx = moved right, positive dy = moved down) -- the same
  convention as Farneback optical flow ``prev -> next``;
* by default divided by the display-order distance to the reference anchor
  (previous / next I or P frame), i.e. "pixels per frame". Exact for
  single-reference encoders (MPEG-4 Part 2, x264 ``refs=1``); approximate for
  multi-reference H.264/HEVC, because FFmpeg does not export which reference
  picture each block used;
* I-frames (and frames without any vector) are zeros with ``is_intra=True``;
  blocks without a vector (intra blocks) are zero;
* ``block_size`` b (default 8, multiple of 4): vectors are painted on a
  4-pixel grid (H.264's smallest partition), then average-pooled to b.

This module does not import torch (the desktop GUI must not import torch, and
benchmark/probe code stays light).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np

LOGGER = logging.getLogger('deeptune.video')

BACKENDS = ('torchcodec', 'pyav', 'opencv')
BACKEND_CHOICES = ('auto', *BACKENDS)
MOTION_METHODS = ('auto', 'codec', 'farneback')
CURVE_METHODS = ('auto', 'codec', 'framediff')

# Reads whose highest index is below this are decoded sequentially from the
# start (always exact, and cheaper than a seek for short clips).
SEQUENTIAL_READ_LIMIT = 300
_ANCHORS = ('I', 'P', 'SI', 'SP')

Source = Union[str, os.PathLike, bytes, bytearray, memoryview]


class VideoDecodeError(ValueError):
    """A clip could not be opened or decoded (missing, corrupt, empty, unsupported)."""


class MotionVectorsUnavailable(VideoDecodeError):
    """The codec exports no motion vectors (e.g. an all-intra file)."""


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    duration: float
    width: int   # display width (after rotation)
    height: int  # display height (after rotation)
    codec: str
    backend: str
    frame_count_exact: bool = True
    variable_fps: bool = False
    rotation: int = 0  # counter-clockwise degrees applied to decoded frames

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MotionField:
    """Per-block motion for the requested frames.

    field: float32 ``[T, 2, ceil(H/b), ceil(W/b)]`` (dx, dy), stored orientation.
    is_intra: bool ``[T]`` -- True where the frame has no motion (I-frame, frame 0).
    source: ``'codec'`` or ``'farneback'``.
    """
    field: np.ndarray
    is_intra: np.ndarray
    source: str
    block_size: int
    per_frame: bool

    def __len__(self):
        return len(self.field)


def available_backends() -> list[str]:
    """Backends that import in this environment, in priority order."""
    found = []
    for name, module in (('torchcodec', 'torchcodec.decoders'), ('pyav', 'av'), ('opencv', 'cv2')):
        try:
            __import__(module)
            found.append(name)
        except Exception:
            pass
    return found


def resolve_backend(name: str = 'auto') -> str:
    name = (name or 'auto').lower()
    if name not in BACKEND_CHOICES:
        raise ValueError(f'Unknown video backend {name!r}; choose one of {", ".join(BACKEND_CHOICES)}.')
    found = available_backends()
    if name == 'auto':
        if not found:
            raise RuntimeError('No video decoder is installed. Install PyAV (pip install av) or opencv-python.')
        return found[0]
    if name not in found:
        hint = {'torchcodec': 'pip install torchcodec', 'pyav': 'pip install av', 'opencv': 'pip install opencv-python'}[name]
        raise RuntimeError(f'The {name} video backend is not installed ({hint}).')
    return name


def source_fingerprint(source: Source) -> str:
    """Stable cache key: sha1 of the bytes, or of resolved path + size + mtime."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return hashlib.sha1(bytes(source)).hexdigest()
    path = Path(source)
    stat = path.stat()
    return hashlib.sha1(f'{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()


def describe_source(source) -> str:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return f'<{len(source)} bytes>'
    return str(source)


def _check_source(source: Source):
    if isinstance(source, (bytes, bytearray, memoryview)):
        if len(source) == 0:
            raise VideoDecodeError('The video is empty (0 bytes).')
        return bytes(source) if isinstance(source, memoryview) else source
    if isinstance(source, (str, os.PathLike)):
        if not str(source).strip():
            raise VideoDecodeError('No video path was given.')
        path = Path(source)
        if not path.is_file():
            raise VideoDecodeError(f'Video file not found: {path}')
        if path.stat().st_size == 0:
            raise VideoDecodeError(f'The video file is empty (0 bytes): {path}')
        return str(path)
    raise VideoDecodeError(f'Expected a video path or encoded bytes, got {type(source).__name__}.')


def _as_index_array(indices) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(array) and array.min() < 0:
        raise IndexError('Frame indices must be >= 0.')
    return array


def _rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply a display rotation (counter-clockwise degrees, FFmpeg/PyAV convention)."""
    turns = int(round(rotation / 90.0)) % 4
    return np.ascontiguousarray(np.rot90(frame, k=turns)) if turns else frame


class VideoReader:
    """Frame-exact reader over a file path or in-memory encoded bytes.

    ``backend='auto'`` picks torchcodec > pyav > opencv. Opening is lazy;
    use as a context manager or call ``close()``.
    """

    def __init__(self, source: Source, backend: str = 'auto'):
        self.source = _check_source(source)
        self.backend = resolve_backend(backend)
        impl = {'torchcodec': _TorchCodecBackend, 'pyav': _PyAVBackend, 'opencv': _OpenCVBackend}[self.backend]
        self._impl = impl(self.source)
        self._metadata: Optional[VideoMetadata] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        impl = getattr(self, '_impl', None)
        if impl is not None:
            impl.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _wrap(self, action: str, exc: Exception) -> VideoDecodeError:
        return VideoDecodeError(f'Could not {action} {describe_source(self.source)}: {exc}')

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            try:
                self._metadata = self._impl.metadata()
            except VideoDecodeError:
                raise
            except Exception as exc:
                raise self._wrap('read', exc) from exc
            if self._metadata.variable_fps:
                LOGGER.warning('%s has a variable frame rate; frames are indexed in presentation order '
                               'and times come from the stream timestamps.', describe_source(self.source))
        return self._metadata

    def fingerprint(self) -> str:
        return source_fingerprint(self.source)

    def frame_times(self) -> Optional[np.ndarray]:
        """Presentation time (s, from the first frame) of every frame, when known exactly."""
        self.metadata()
        return self._impl.frame_times()

    def time_to_index(self, seconds: float) -> int:
        """First frame shown at or after `seconds`, clamped to the clip."""
        meta = self.metadata()
        times = self.frame_times()
        if times is not None and len(times):
            return int(min(np.searchsorted(times, float(seconds) - 1e-6), len(times) - 1))
        if meta.fps <= 0:
            raise VideoDecodeError(f'{describe_source(self.source)} reports no frame rate; cannot convert seconds to frames.')
        return int(min(max(0, round(float(seconds) * meta.fps)), max(0, meta.frame_count - 1)))

    def time_to_end_index(self, seconds: float) -> int:
        """Exclusive end index for a range ending at `seconds` (frames shown before it)."""
        meta = self.metadata()
        times = self.frame_times()
        if times is not None and len(times):
            return int(max(1, min(np.searchsorted(times, float(seconds) - 1e-6), len(times))))
        if meta.fps <= 0:
            raise VideoDecodeError(f'{describe_source(self.source)} reports no frame rate; cannot convert seconds to frames.')
        return int(max(1, min(round(float(seconds) * meta.fps), meta.frame_count)))

    # -- frames ------------------------------------------------------------
    def get_frames(self, indices: Sequence[int], clamp_to_end: bool = False) -> np.ndarray:
        """uint8 ``[T, H, W, 3]`` RGB frames at `indices` (any order, repeats allowed).

        Indices past the last decodable frame raise ``IndexError`` unless
        `clamp_to_end` is set; then they repeat the last frame (logged).
        Frames whose resolution changes mid-stream are resized to the size of
        the first returned frame (logged).
        """
        wanted = _as_index_array(indices)
        meta = self.metadata()
        if not len(wanted):
            return np.zeros((0, meta.height, meta.width, 3), dtype=np.uint8)
        targets = np.unique(wanted)
        try:
            decoded = self._impl.read(targets)
        except (VideoDecodeError, IndexError):
            raise
        except Exception as exc:
            raise self._wrap('decode', exc) from exc
        if not decoded:
            raise VideoDecodeError(f'{describe_source(self.source)} contains no decodable frames.')
        last = max(decoded)
        missing = [int(t) for t in targets if int(t) not in decoded]
        if missing:
            inside = [t for t in missing if t < last]
            if inside or not clamp_to_end:
                raise IndexError(f'{describe_source(self.source)}: frame(s) {missing[:5]} out of range '
                                 f'({len(decoded)} of the requested frames decoded, last index {last}).')
            LOGGER.warning('%s: only %d frames are decodable; repeating the last frame for %d requested index(es).',
                           describe_source(self.source), last + 1, len(missing))
        out, base_shape = [], None
        for index in wanted:
            frame = decoded.get(int(index), decoded[last])
            if meta.rotation:
                frame = _rotate(frame, meta.rotation)
            if base_shape is None:
                base_shape = frame.shape
            elif frame.shape != base_shape:
                import cv2
                LOGGER.warning('%s changes resolution mid-stream (%s -> %s); resizing to the first frame.',
                               describe_source(self.source), frame.shape[:2], base_shape[:2])
                frame = cv2.resize(frame, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_AREA)
            out.append(frame)
        return np.stack(out).astype(np.uint8, copy=False)

    def iter_frames(self, width: Optional[int] = None, gray: bool = False) -> Iterator[np.ndarray]:
        """Every frame in order (optionally downscaled to `width`, or grey). Streams; holds one frame."""
        meta = self.metadata()
        for frame in self._impl.iter_frames(width=width, gray=gray):
            yield _rotate(frame, meta.rotation) if meta.rotation else frame

    # -- motion ------------------------------------------------------------
    def get_motion_vectors(self, indices: Sequence[int], block_size: int = 8, method: str = 'auto',
                           per_frame: bool = True) -> MotionField:
        """Dense motion field for the frames at `indices` (conventions: module docstring).

        method: ``'codec'`` -- PyAV ``flags2=+export_mvs``; raises
        :class:`MotionVectorsUnavailable` when the codec exports none (e.g.
        MJPEG, FFV1, all-intra files); ``'farneback'`` -- OpenCV optical flow
        from frame t-1 to t, averaged per block; ``'auto'`` -- codec, falling
        back to Farneback with a logged warning.
        """
        if method not in MOTION_METHODS:
            raise ValueError(f'Unknown motion method {method!r}; choose one of {", ".join(MOTION_METHODS)}.')
        if block_size < 4 or block_size % 4:
            raise ValueError('block_size must be a positive multiple of 4.')
        wanted = _as_index_array(indices)
        meta = self.metadata()
        if method in ('auto', 'codec'):
            try:
                return codec_motion_vectors(self.source, wanted, block_size, per_frame)
            except MotionVectorsUnavailable:
                if method == 'codec':
                    raise
                LOGGER.warning('%s: codec %s exports no motion vectors; using Farneback optical flow instead.',
                               describe_source(self.source), meta.codec)
        return self._farneback(wanted, block_size)

    def _farneback(self, wanted: np.ndarray, block_size: int) -> MotionField:
        previous = np.maximum(wanted - 1, 0)
        frames = self.get_frames(np.concatenate([previous, wanted]), clamp_to_end=True)
        prev_frames, cur_frames = frames[:len(wanted)], frames[len(wanted):]
        if self.metadata().rotation:
            # Report in the stored orientation, like codec vectors.
            undo = -self.metadata().rotation
            prev_frames = np.stack([_rotate(f, undo) for f in prev_frames]) if len(wanted) else prev_frames
            cur_frames = np.stack([_rotate(f, undo) for f in cur_frames]) if len(wanted) else cur_frames
        fields, intra = [], []
        for t, (a, b) in enumerate(zip(prev_frames, cur_frames)):
            if wanted[t] == 0:
                h, w = a.shape[:2]
                fields.append(np.zeros((2, -(-h // block_size), -(-w // block_size)), dtype=np.float32))
                intra.append(True)
            else:
                fields.append(block_average(farneback_flow(a, b), block_size))
                intra.append(False)
        field = np.stack(fields) if fields else np.zeros((0, 2, 0, 0), np.float32)
        return MotionField(field, np.asarray(intra, dtype=bool), 'farneback', block_size, True)

    def motion_curve(self, method: str = 'auto') -> np.ndarray:
        """Per-frame motion magnitude ``m[t]`` (``m[0] = 0``), one value per frame.

        ``codec``: mean |MV| per frame (PyAV only; I-frames repeat the
        previous value). ``framediff``: mean absolute difference of
        consecutive 64-pixel-wide grey frames, as in MGSampler. ``auto``:
        codec, else framediff. Decodes the whole clip once.
        """
        if method not in CURVE_METHODS:
            raise ValueError(f"Motion curve method must be one of {', '.join(CURVE_METHODS)}.")
        if method in ('auto', 'codec'):
            if self.backend == 'pyav' or method == 'codec':
                try:
                    return codec_motion_curve(self.source)
                except MotionVectorsUnavailable:
                    if method == 'codec':
                        raise
        values, previous = [], None
        for frame in self.iter_frames(width=64, gray=True):
            frame = frame.astype(np.float32)
            values.append(0.0 if previous is None or previous.shape != frame.shape
                          else float(np.abs(frame - previous).mean()))
            previous = frame
        if not values:
            raise VideoDecodeError(f'{describe_source(self.source)} contains no decodable frames.')
        return np.asarray(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Optical flow / block helpers (Farneback fallback, augmentation, tests)
# ---------------------------------------------------------------------------
def farneback_flow(previous_rgb: np.ndarray, current_rgb: np.ndarray) -> np.ndarray:
    """Dense Farneback flow ``[H, W, 2]`` (dx, dy) from `previous_rgb` to `current_rgb`."""
    import cv2
    a = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def block_average(flow: np.ndarray, block_size: int) -> np.ndarray:
    """Average an ``[H, W, 2]`` flow into ``[2, ceil(H/b), ceil(W/b)]`` blocks."""
    h, w = flow.shape[:2]
    gh, gw = -(-h // block_size), -(-w // block_size)
    padded = np.zeros((gh * block_size, gw * block_size, 2), dtype=np.float32)
    padded[:h, :w] = flow
    weight = np.zeros((gh * block_size, gw * block_size), dtype=np.float32)
    weight[:h, :w] = 1
    sums = padded.reshape(gh, block_size, gw, block_size, 2).sum(axis=(1, 3))
    counts = weight.reshape(gh, block_size, gw, block_size).sum(axis=(1, 3))
    return np.moveaxis(sums / np.maximum(counts, 1)[..., None], -1, 0).astype(np.float32)


def flip_motion_field(field: np.ndarray) -> np.ndarray:
    """Horizontal flip of a ``[..., 2, H, W]`` motion field: mirror the grid and negate dx."""
    flipped = np.array(field[..., ::-1], dtype=np.float32, copy=True)
    flipped[..., 0, :, :] *= -1
    return flipped


# ---------------------------------------------------------------------------
# PyAV backend (default) + codec motion vectors
# ---------------------------------------------------------------------------
def _open_av(source):
    import av
    try:
        container = av.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source, mode='r')
    except Exception as exc:
        raise VideoDecodeError(f'Could not open {describe_source(source)}: the file may be corrupt or its format '
                               f'unsupported ({exc}).') from exc
    if not container.streams.video:
        container.close()
        raise VideoDecodeError(f'{describe_source(source)} has no video stream.')
    return container


_PICT_TYPES = {0: 'NONE', 1: 'I', 2: 'P', 3: 'B', 4: 'S', 5: 'SI', 6: 'SP', 7: 'BI'}


def _pict_type(frame) -> str:
    kind = frame.pict_type
    if hasattr(kind, 'name'):
        return kind.name
    try:
        return _PICT_TYPES.get(int(kind), str(kind))
    except (TypeError, ValueError):
        return str(kind)


def reencode_for_motion_vectors(source, destination, gop: int = 12, quality: int = 4) -> Path:
    """Transcode a clip to MPEG-4 Part 2 with P-frames only and one reference.

    This is CoViAR's preprocessing: with a single reference picture and no
    B-frames, every codec vector spans exactly one frame, so motion vectors
    are exact "pixels per frame". H.264/HEVC files with several reference
    frames (x264's default is 3) give vectors whose span FFmpeg does not
    report. Returns `destination`.
    """
    import av
    destination = Path(destination)
    container = _open_av(_check_source(source))
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or 25
        out = av.open(str(destination), 'w')
        try:
            target = out.add_stream('mpeg4', rate=rate)
            ctx = stream.codec_context
            target.width, target.height = ctx.width - ctx.width % 2, ctx.height - ctx.height % 2
            target.pix_fmt = 'yuv420p'
            target.options = {'g': str(gop), 'bf': '0', 'qmin': str(quality), 'qmax': str(quality + 2)}
            for number, frame in enumerate(container.decode(stream)):
                frame = frame.reformat(width=target.width, height=target.height, format='yuv420p')
                frame.pts = number
                frame.time_base = target.codec_context.time_base
                for packet in target.encode(frame):
                    out.mux(packet)
            for packet in target.encode():
                out.mux(packet)
        finally:
            out.close()
    finally:
        container.close()
    return destination


class _PyAVBackend:
    def __init__(self, source):
        self.source = source
        self._pts: Optional[np.ndarray] = None  # sorted presentation timestamps, one per frame
        self._keys: np.ndarray = np.zeros(0, dtype=np.int64)  # frame indices of keyframes
        self._time_base = None
        self._meta: Optional[VideoMetadata] = None

    def close(self):
        pass

    def metadata(self) -> VideoMetadata:
        if self._meta is not None:
            return self._meta
        container = _open_av(self.source)
        try:
            stream = container.streams.video[0]
            self._time_base = stream.time_base
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            fps = float(rate) if rate else 0.0
            pts, key_pts, packets = [], [], 0
            for packet in container.demux(stream):
                if packet.size:
                    packets += 1
                    if packet.pts is not None:
                        pts.append(packet.pts)
                        if packet.is_keyframe:
                            key_pts.append(packet.pts)
            if packets == 0:
                raise VideoDecodeError(f'{describe_source(self.source)} contains no video frames (zero length).')
            unique = np.unique(np.asarray(pts, dtype=np.int64))
            # Timestamps index frames exactly only if every packet has its own one.
            self._pts = unique if len(unique) == packets else None
            if self._pts is not None:
                self._keys = np.unique(np.searchsorted(self._pts, np.asarray(key_pts, dtype=np.int64)))
            variable, duration = False, (packets / fps if fps else 0.0)
            if self._pts is not None and stream.time_base is not None:
                times = (self._pts - self._pts[0]) * float(stream.time_base)
                steps = np.diff(times)
                if len(steps) >= 2 and steps.mean() > 0:
                    variable = bool(steps.std() > 0.05 * steps.mean())
                step = float(np.median(steps)) if len(steps) else (1.0 / fps if fps else 0.0)
                duration = float(times[-1] + step)
            # The first decoded frame gives the true size and display rotation.
            container.seek(0)
            first = next(iter(container.decode(stream)), None)
            if first is None:
                raise VideoDecodeError(f'{describe_source(self.source)} contains no decodable frames.')
            rotation = int(round(getattr(first, 'rotation', 0) or 0)) % 360
            width, height = int(first.width), int(first.height)
            if rotation in (90, 270):
                width, height = height, width
            self._meta = VideoMetadata(
                fps=fps, frame_count=packets, duration=duration, width=width, height=height,
                codec=str(stream.codec_context.name), backend='pyav', frame_count_exact=self._pts is not None,
                variable_fps=variable, rotation=rotation)
        finally:
            container.close()
        return self._meta

    def frame_times(self):
        if self._pts is None or self._time_base is None:
            return None
        return (self._pts - self._pts[0]) * float(self._time_base)

    def read(self, targets: np.ndarray) -> dict:
        return _av_iterate(self, targets, want_frames=True, want_mvs=False)[0]

    def iter_frames(self, width: Optional[int] = None, gray: bool = False):
        container = _open_av(self.source)
        try:
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for frame in container.decode(stream):
                fmt = 'gray' if gray else 'rgb24'
                if width:
                    height = max(2, int(round(frame.height * width / max(1, frame.width))) // 2 * 2)
                    yield frame.reformat(width=width, height=height, format=fmt).to_ndarray()
                else:
                    yield frame.to_ndarray(format=fmt)
        finally:
            container.close()


def _av_iterate(backend: _PyAVBackend, targets, want_frames: bool, want_mvs: bool, lookahead: int = 0):
    """Decode the frames at sorted unique `targets` (presentation order).

    Returns (frames {index: rgb}, mvs {index: structured array or None},
    kinds {index: 'I'|'P'|'B'|... for every decoded frame}). Short spans and
    files without usable timestamps decode sequentially from frame 0,
    counting frames (always exact). Sparse reads in long videos seek to the
    keyframe before a target and identify frames by presentation timestamp.
    `lookahead` keeps decoding that many frames past the last target (to find
    the next anchor frame for motion-vector normalisation).
    """
    meta = backend.metadata()
    frames, mvs, kinds = {}, {}, {}
    targets = [int(t) for t in targets]
    if not targets:
        return frames, mvs, kinds
    target_set = set(targets)
    stop_at = targets[-1] + lookahead
    tail = []  # the most recently decoded (index, frame), for clamping past the end

    def keep(index, frame):
        kinds[index] = _pict_type(frame)
        tail[:] = [(index, frame)]
        if index in target_set:
            if want_frames:
                frames[index] = frame.to_ndarray(format='rgb24')
            if want_mvs:
                side = frame.side_data.get('MOTION_VECTORS')
                mvs[index] = side.to_ndarray() if side is not None else None

    def open_stream():
        container = _open_av(backend.source)
        stream = container.streams.video[0]
        if want_mvs:
            stream.codec_context.options = {'flags2': '+export_mvs'}
        else:
            stream.thread_type = 'AUTO'
        return container, stream

    def add_tail():
        # Targets past the real end: hand back the true last frame so callers can repeat it.
        if want_frames and tail and tail[0][0] not in frames and tail[0][0] < targets[-1]:
            frames[tail[0][0]] = tail[0][1].to_ndarray(format='rgb24')

    def sequential():
        container, stream = open_stream()
        try:
            for index, frame in enumerate(container.decode(stream)):
                keep(index, frame)
                if index >= stop_at:
                    break
            add_tail()
        finally:
            container.close()

    pts_index = backend._pts
    if pts_index is None or targets[-1] < SEQUENTIAL_READ_LIMIT or meta.frame_count <= SEQUENTIAL_READ_LIMIT:
        sequential()
        return frames, mvs, kinds

    pts_to_index = {int(p): i for i, p in enumerate(pts_index)}
    keys = backend._keys

    def key_before(index):
        # Last keyframe at or before `index` (0 when unknown): where a seek would land.
        at = int(np.searchsorted(keys, index, side='right')) - 1
        return int(keys[at]) if at >= 0 else 0

    container, stream = open_stream()
    try:
        cursor, position, decoder = 0, -1, None
        while cursor < len(targets) and targets[cursor] < len(pts_index):
            target = targets[cursor]
            if decoder is None or key_before(target) > position:
                # Seek only when a later keyframe lets us skip frames; otherwise decode on.
                container.seek(int(pts_index[target]), stream=stream, backward=True, any_frame=False)
                decoder = container.decode(stream)
            reseek = False
            for frame in decoder:
                index = pts_to_index.get(int(frame.pts)) if frame.pts is not None else None
                if index is None:
                    continue
                position = index
                keep(index, frame)
                advanced = False
                while cursor < len(targets) and targets[cursor] <= index:
                    cursor += 1
                    advanced = True
                if cursor >= len(targets):
                    if index >= stop_at:
                        break
                    continue
                # Re-seek only after reaching a target (guarantees progress) and only
                # when the next target sits behind a keyframe we have not reached.
                if advanced and key_before(targets[cursor]) > index:
                    reseek = True
                    break
            if not reseek:
                break  # every target handled, or the stream ended
        add_tail()
    finally:
        container.close()
    if any(t not in kinds for t in targets if t < len(pts_index)):
        # A seek landed past a target (odd index/timestamps): exact sequential re-read.
        LOGGER.debug('%s: seek missed frames; re-reading sequentially.', describe_source(backend.source))
        sequential()
    return frames, mvs, kinds


def _anchor_distances(index: int, kinds: dict) -> tuple[int, int]:
    """Display-order distance from `index` to the previous and the next I/P anchor."""
    past = future = 1
    for d in range(1, 64):
        if index - d < 0:
            break
        if kinds.get(index - d) in _ANCHORS:
            past = d
            break
    for d in range(1, 64):
        kind = kinds.get(index + d)
        if kind is None:
            break
        if kind in _ANCHORS:
            future = d
            break
    return past, future


def rasterize_motion_vectors(vectors: np.ndarray, width: int, height: int, block_size: int,
                             past_distance: int = 1, future_distance: int = 1) -> np.ndarray:
    """Turn FFmpeg's per-block vectors into a dense ``[2, ceil(H/b), ceil(W/b)]`` field.

    FFmpeg's ``AVMotionVector`` says ``src = dst + motion / motion_scale``
    where (dst_x, dst_y) is the block centre in this frame and (src_x, src_y)
    its match in the reference. We store "later minus earlier" displacement:
    ``dst - src`` for past references (``source < 0``) and ``src - dst`` for
    future ones, divided by the reference distance. Vectors are painted on a
    4-pixel grid (overlaps, e.g. the two halves of a bi-predicted block, are
    averaged), then average-pooled to `block_size`.
    """
    cell = 4
    stored_w, stored_h = width, height
    gh, gw = -(-stored_h // cell), -(-stored_w // cell)
    sums = np.zeros((2, gh, gw), dtype=np.float64)
    counts = np.zeros((gh, gw), dtype=np.float64)
    if vectors is not None and len(vectors):
        past = vectors['source'] < 0
        sign = np.where(past, 1.0, -1.0)
        distance = np.where(past, float(max(1, past_distance)), float(max(1, future_distance)))
        scale = vectors['motion_scale'].astype(np.float64)
        dx = sign * (-vectors['motion_x'] / np.maximum(scale, 1)) / distance
        dy = sign * (-vectors['motion_y'] / np.maximum(scale, 1)) / distance
        bw = np.maximum(vectors['w'].astype(np.int64), cell)
        bh = np.maximum(vectors['h'].astype(np.int64), cell)
        x0 = (vectors['dst_x'].astype(np.int64) - bw // 2) // cell
        y0 = (vectors['dst_y'].astype(np.int64) - bh // 2) // cell
        for size in np.unique(np.stack([bw, bh], 1), axis=0):
            sel = (bw == size[0]) & (bh == size[1])
            ox, oy = np.meshgrid(np.arange(size[0] // cell), np.arange(size[1] // cell))
            xs = (x0[sel][:, None] + ox.reshape(1, -1)).reshape(-1)
            ys = (y0[sel][:, None] + oy.reshape(1, -1)).reshape(-1)
            vx, vy = np.repeat(dx[sel], ox.size), np.repeat(dy[sel], ox.size)
            ok = (xs >= 0) & (xs < gw) & (ys >= 0) & (ys < gh)
            np.add.at(sums[0], (ys[ok], xs[ok]), vx[ok])
            np.add.at(sums[1], (ys[ok], xs[ok]), vy[ok])
            np.add.at(counts, (ys[ok], xs[ok]), 1.0)
    dense = sums / np.maximum(counts, 1)[None]
    factor = block_size // cell
    oh, ow = -(-stored_h // block_size), -(-stored_w // block_size)
    padded = np.zeros((2, oh * factor, ow * factor))
    padded[:, :gh, :gw] = dense
    return padded.reshape(2, oh, factor, ow, factor).mean(axis=(2, 4)).astype(np.float32)


def _stored_size(meta: VideoMetadata) -> tuple[int, int]:
    if meta.rotation in (90, 270):
        return meta.height, meta.width
    return meta.width, meta.height


def codec_motion_vectors(source, indices, block_size: int = 8, per_frame: bool = True) -> MotionField:
    """Codec motion vectors via PyAV (``flags2=+export_mvs``) for `indices`."""
    try:
        import av  # noqa: F401
    except ImportError as exc:
        raise MotionVectorsUnavailable('Codec motion vectors need PyAV (pip install av).') from exc
    indices = _as_index_array(indices)
    backend = _PyAVBackend(_check_source(source))
    meta = backend.metadata()
    targets = np.unique(indices)
    if len(targets) and targets[-1] >= meta.frame_count:
        raise IndexError(f'{describe_source(source)}: frame {int(targets[-1])} is out of range ({meta.frame_count} frames).')
    _, vectors, kinds = _av_iterate(backend, targets, want_frames=False, want_mvs=True, lookahead=16)
    width, height = _stored_size(meta)
    if not any(v is not None and len(v) for v in vectors.values()) and not _codec_has_motion(backend):
        raise MotionVectorsUnavailable(f'{describe_source(source)} ({meta.codec}) exports no motion vectors.')
    gh, gw = -(-height // block_size), -(-width // block_size)
    fields, intra = [], []
    for index in indices:
        vec = vectors.get(int(index))
        if vec is None or not len(vec):
            fields.append(np.zeros((2, gh, gw), dtype=np.float32))
            intra.append(True)
            continue
        past, future = _anchor_distances(int(index), kinds) if per_frame else (1, 1)
        fields.append(rasterize_motion_vectors(vec, width, height, block_size, past, future))
        intra.append(False)
    field = np.stack(fields) if fields else np.zeros((0, 2, gh, gw), np.float32)
    return MotionField(field, np.asarray(intra, dtype=bool), 'codec', block_size, per_frame)


def _codec_has_motion(backend: _PyAVBackend, probe: int = 24) -> bool:
    """True if any of the first `probe` frames carries motion vectors."""
    targets = np.arange(min(probe, backend.metadata().frame_count))
    _, vectors, _ = _av_iterate(backend, targets, want_frames=False, want_mvs=True)
    return any(v is not None and len(v) for v in vectors.values())


def codec_motion_curve(source) -> np.ndarray:
    """Mean |MV| (pixels/frame) per frame over the whole clip; I-frames repeat the previous value."""
    backend = _PyAVBackend(_check_source(source))
    meta = backend.metadata()
    _, vectors, kinds = _av_iterate(backend, np.arange(meta.frame_count), want_frames=False, want_mvs=True)
    if not any(v is not None and len(v) for v in vectors.values()):
        raise MotionVectorsUnavailable(f'{describe_source(source)} ({meta.codec}) exports no motion vectors.')
    width, height = _stored_size(meta)
    decoded = max(kinds) + 1 if kinds else 0
    curve = np.zeros(max(decoded, 1), dtype=np.float64)
    previous = 0.0
    for index in range(1, decoded):
        vec = vectors.get(index)
        if vec is not None and len(vec):
            past, future = _anchor_distances(index, kinds)
            field = rasterize_motion_vectors(vec, width, height, 16, past, future)
            previous = float(np.hypot(field[0], field[1]).mean())
        curve[index] = previous
    return curve


# ---------------------------------------------------------------------------
# TorchCodec backend (optional; skipped cleanly when not installed)
# ---------------------------------------------------------------------------
class _TorchCodecBackend:
    def __init__(self, source):
        from torchcodec.decoders import VideoDecoder
        self.source = source
        try:
            self._decoder = VideoDecoder(source, seek_mode='exact', device='cpu')
        except Exception as exc:
            raise VideoDecodeError(f'Could not open {describe_source(source)} with TorchCodec ({exc}).') from exc

    def close(self):
        self._decoder = None

    def metadata(self) -> VideoMetadata:
        m = self._decoder.metadata
        count = int(m.num_frames or 0)
        if count == 0:
            raise VideoDecodeError(f'{describe_source(self.source)} contains no video frames (zero length).')
        fps = float(m.average_fps or 0.0)
        duration = float(m.duration_seconds or (count / fps if fps else 0.0))
        return VideoMetadata(fps=fps, frame_count=count, duration=duration, width=int(m.width), height=int(m.height),
                             codec=str(m.codec), backend='torchcodec')

    def frame_times(self):
        return None

    def read(self, targets: np.ndarray) -> dict:
        count = self.metadata().frame_count
        valid = [int(t) for t in targets if t < count]
        if not valid:
            return {}
        data = self._decoder.get_frames_at(indices=valid).data.permute(0, 2, 3, 1).contiguous().numpy()
        return {index: data[i] for i, index in enumerate(valid)}

    def iter_frames(self, width: Optional[int] = None, gray: bool = False):
        import cv2
        count = self.metadata().frame_count
        for start in range(0, count, 32):
            for frame in self.read(np.arange(start, min(count, start + 32))).values():
                yield _resize_convert(frame, width, gray, cv2)


def _resize_convert(rgb: np.ndarray, width: Optional[int], gray: bool, cv2):
    if width:
        height = max(2, int(round(rgb.shape[0] * width / max(1, rgb.shape[1]))))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray else rgb


# ---------------------------------------------------------------------------
# OpenCV backend (fallback; sequential reads only)
# ---------------------------------------------------------------------------
class _OpenCVBackend:
    def __init__(self, source):
        self.source = source
        self._temp = None
        if isinstance(source, (bytes, bytearray)):
            # OpenCV cannot decode from memory: one temp file per reader, removed on close().
            handle = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            handle.write(source)
            handle.close()
            self._temp = self._path = handle.name
        else:
            self._path = source

    def close(self):
        if self._temp is not None:
            try:
                os.unlink(self._temp)
            except OSError:
                pass
            self._temp = None

    def _capture(self):
        import cv2
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            cap.release()
            raise VideoDecodeError(f'Could not open {describe_source(self.source)}: the clip may be corrupt '
                                   'or its codec unsupported.')
        return cap

    def metadata(self) -> VideoMetadata:
        import cv2
        cap = self._capture()
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            codec = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip('\x00 ') or 'unknown'
            exact = count > 0
            ok, first = cap.read()
            if not ok:
                raise VideoDecodeError(f'{describe_source(self.source)} contains no decodable frames (zero length).')
            height, width = first.shape[:2]  # OpenCV applies the display rotation itself
            if count <= 0:
                count = 1
                while cap.grab():
                    count += 1
        finally:
            cap.release()
        return VideoMetadata(fps=fps, frame_count=count, duration=count / fps if fps else 0.0, width=width,
                             height=height, codec=codec, backend='opencv', frame_count_exact=exact)

    def frame_times(self):
        return None

    def read(self, targets: np.ndarray) -> dict:
        import cv2
        cap = self._capture()
        wanted = set(int(t) for t in targets)
        last = int(max(wanted))
        out = {}
        decoded = 0
        try:
            for index in range(last + 1):
                if not cap.grab():
                    break
                decoded = index + 1
                if index in wanted:
                    ok, frame = cap.retrieve()
                    if ok:
                        out[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        if 0 < decoded <= last and decoded - 1 not in out:
            # The reported frame count was too high: fetch the true last frame for clamping.
            out.update(self.read(np.asarray([decoded - 1])))
        return out

    def iter_frames(self, width: Optional[int] = None, gray: bool = False):
        import cv2
        cap = self._capture()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield _resize_convert(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width, gray, cv2)
        finally:
            cap.release()


def probe_metadata(source: Source, backend: str = 'auto') -> VideoMetadata:
    with VideoReader(source, backend=backend) as reader:
        return reader.metadata()

"""Tiny synthetic clips for tests and scripts/bench_video_io.py (not training data).

* ``index_coded_clip``: every frame's pixel values encode its own index, so a
  decoder's output can be checked for **exact** frame reads.
* ``translating_clip``: a smooth random texture moving by a known (dx, dy)
  pixels per frame (optionally static for the first frames), to check
  motion-vector sign / magnitude and motion-guided sampling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Index code: frame i -> bottom half grey level LOW_BASE + LOW_STEP * (i % 20),
# top-left quarter level HIGH_STEP * ((i // 20) % 20). Steps of 12 survive
# lossy coding of flat colours with a wide margin. Indices up to 399.
LOW_BASE, LOW_STEP, HIGH_STEP, DIGITS = 8, 12, 12, 20


def index_coded_frame(index: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), LOW_BASE + LOW_STEP * (index % DIGITS), np.uint8)
    frame[: height // 2, : width // 2] = HIGH_STEP * ((index // DIGITS) % DIGITS)
    return frame


def decode_frame_index(frame: np.ndarray) -> int:
    height, width = frame.shape[:2]
    low = frame[height // 2 + 2:, :].mean()
    high = frame[2: height // 2 - 2, 2: width // 2 - 2].mean()
    return int(round(high / HIGH_STEP)) * DIGITS + int(round((low - LOW_BASE) / LOW_STEP))


def write_clip(path, frames, fps: int = 25, codec: str = 'libx264', options: Optional[dict] = None,
               pix_fmt: str = 'yuv420p', rotation: Optional[int] = None) -> Path:
    """Encode RGB uint8 frames with PyAV."""
    import av
    path = Path(path)
    frames = list(frames)
    height, width = frames[0].shape[:2]
    container = av.open(str(path), 'w')
    try:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, pix_fmt
        if options:
            stream.options = dict(options)
        if rotation is not None:
            stream.set_display_rotation(rotation)
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path


def trim_with_edit_list(source, destination, hidden_frames: int = 3) -> Path:
    """Remux `source` (no re-encode) so its first `hidden_frames` frames sit before
    the MP4 edit list's start, like an ``ffmpeg -ss ... -c copy`` trim: the demuxer
    flags those packets ``is_discard`` and the decoder never shows them."""
    import av
    destination = Path(destination)
    source_container = av.open(str(source))
    try:
        stream = source_container.streams.video[0]
        packets = [packet for packet in source_container.demux(stream) if packet.size]
        times = sorted(packet.pts for packet in packets)
        shift = hidden_frames * (times[1] - times[0])
        output = av.open(str(destination), 'w')
        try:
            target = output.add_stream_from_template(stream)
            for packet in packets:
                packet.pts -= shift
                packet.dts -= shift
                packet.stream = target
                output.mux(packet)
        finally:
            output.close()
    finally:
        source_container.close()
    return destination


def index_coded_clip(path, num_frames: int, width: int = 32, height: int = 32, **kwargs) -> Path:
    return write_clip(path, (index_coded_frame(i, width, height) for i in range(num_frames)), **kwargs)


def translating_clip(path, num_frames: int, dx: int, dy: int, size: int = 64, static_frames: int = 0,
                     seed: int = 0, **kwargs) -> Path:
    """Content moves +dx (right) / +dy (down) pixels per frame after `static_frames` still frames."""
    import cv2
    margin = abs(dx) * num_frames + abs(dy) * num_frames + 8
    rng = np.random.default_rng(seed)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (size + 2 * margin, size + 2 * margin, 3), dtype=np.uint8), (5, 5), 0)

    def frame(t):
        k = max(0, t - static_frames)
        x0, y0 = margin - dx * k, margin - dy * k
        return texture[y0:y0 + size, x0:x0 + size]

    return write_clip(path, (frame(t) for t in range(num_frames)), **kwargs)

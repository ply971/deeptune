"""Cheap hand-made features used to check that the benchmarks are honest.

These are *probes*, not models: they show what information a clip carries.
- ``read_frames``: decode every frame of an MP4 with OpenCV (RGB uint8).
- ``farneback_flow``: dense optical flow (Farneback) between consecutive frames.
- ``mean_flow_features``: per-frame-pair mean horizontal (and vertical) flow.
- ``codec_mv_features``: the same from the codec's own motion vectors (needs
  PyAV; returns ``None`` when it is not installed or the stream has no MVs).
- ``appearance_features``: what a single frame (or a frame average) looks like.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_frames(path: Path) -> np.ndarray:
    """Decode all frames of a video file to uint8 RGB [T, H, W, 3]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {path}.")
    frames = []
    try:
        ok, frame = cap.read()
        while ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ok, frame = cap.read()
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}.")
    return np.stack(frames)


def farneback_flow(frames_rgb: np.ndarray) -> np.ndarray:
    """Farneback flow between consecutive frames: float32 [T-1, H, W, 2] (u, v)."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
    flows = [
        cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None,
                                     pyr_scale=0.5, levels=3, winsize=5, iterations=5,
                                     poly_n=5, poly_sigma=1.1, flags=0)
        for i in range(len(gray) - 1)
    ]
    return np.stack(flows).astype(np.float32)


def mean_flow_features(frames_rgb: np.ndarray, vertical: bool = False) -> np.ndarray:
    """Mean horizontal flow of each consecutive frame pair -> [T-1] (or [2(T-1)]).

    This mirrors the course slide's probe (Lucas-Kanade, 11 numbers for 12
    frames) with dense Farneback flow averaged over the whole frame.
    """
    flow = farneback_flow(frames_rgb)
    u = flow[..., 0].mean(axis=(1, 2))
    if not vertical:
        return u
    return np.concatenate([u, flow[..., 1].mean(axis=(1, 2))])


def flow_summary_features(frames_rgb: np.ndarray) -> np.ndarray:
    """Order-free summary of horizontal motion (6 numbers).

    ``[mean u, mean |u|, std u, consistency, speed, speed_any]`` where u is the
    per-frame-pair mean horizontal flow, ``consistency = |sum u| / sum |u|``
    (1 for a steady drift, near 0 for a back-and-forth oscillation), ``speed``
    is the median over frame pairs of the 97th percentile of |u| over pixels
    (the moving object's speed, insensitive to its size) and ``speed_any`` the
    same for the flow magnitude. Used for the speed and periodic probes, where
    the phase of the motion is random so per-frame features are not aligned.
    """
    flow = farneback_flow(frames_rgb)
    u = flow[..., 0].mean(axis=(1, 2))
    abs_u = np.abs(flow[..., 0]).reshape(len(flow), -1)
    mag = np.hypot(flow[..., 0], flow[..., 1]).reshape(len(flow), -1)
    consistency = np.abs(u.sum()) / max(float(np.abs(u).sum()), 1e-6)
    return np.array([u.mean(), np.abs(u).mean(), u.std(), consistency,
                     np.median(np.percentile(abs_u, 97, axis=1)),
                     np.median(np.percentile(mag, 97, axis=1))], np.float32)


def appearance_features(frame_rgb: np.ndarray, grid: int = 16) -> np.ndarray:
    """Raw-pixel appearance of one frame (or a frame average), downsampled.

    Returns a flat float vector: the frame resized to grid x grid x 3 plus a
    4-bin-per-channel colour histogram. Enough for a linear model to read the
    blob's position, colour and the background texture.
    """
    f = frame_rgb.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    small = cv2.resize(f, (grid, grid), interpolation=cv2.INTER_AREA).reshape(-1)
    hist = np.concatenate([np.histogram(f[..., c], bins=4, range=(0, 1))[0] / f[..., c].size for c in range(3)])
    return np.concatenate([small, hist]).astype(np.float32)


def spectrum_features(frame_rgb: np.ndarray, mask: np.ndarray | None = None, bins: int = 8) -> np.ndarray:
    """Translation-invariant texture descriptor of a frame's background.

    Pixels under ``mask`` (e.g. the moving blob) are replaced with the mean of
    the remaining pixels before the 2D FFT, so the blob does not contribute.
    Returns the log power spectrum pooled into ``bins`` x ``bins`` cells plus the
    mean colour of the unmasked pixels.
    """
    f = frame_rgb.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    gray = f.mean(axis=2)
    keep = np.ones(gray.shape, bool) if mask is None else ~mask.astype(bool)
    fill = gray[keep].mean() if keep.any() else 0.5
    gray = np.where(keep, gray, fill)
    power = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2)
    pooled = cv2.resize(power.astype(np.float32), (bins, bins), interpolation=cv2.INTER_AREA).reshape(-1)
    colour = f[keep].mean(axis=0) if keep.any() else np.full(3, 0.5, np.float32)
    return np.concatenate([pooled, colour]).astype(np.float32)


def codec_mv_features(path: Path) -> np.ndarray | None:
    """Mean horizontal codec motion per frame (P/B frames), via PyAV ``+export_mvs``.

    Returns ``None`` if PyAV is not installed or the decoder exports no motion
    vectors (e.g. an intra-only stream). Each motion vector's displacement is
    ``(dst - src)`` for a past reference (``source < 0``) and ``-(dst - src)``
    for a future reference, giving the forward motion of the block.
    Untested on machines without PyAV (Phase 1 installs it).
    """
    try:
        import av  # noqa: F401 - optional dependency
    except ImportError:
        return None
    feats = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.options = {"flags2": "+export_mvs"}
        for frame in container.decode(stream):
            side = frame.side_data.get("MOTION_VECTORS") if hasattr(frame.side_data, "get") else None
            if side is None:
                continue
            mvs = side.to_ndarray()
            if len(mvs) == 0:
                feats.append(0.0)
                continue
            dx = (mvs["dst_x"] - mvs["src_x"]).astype(np.float32)
            dx = np.where(mvs["source"] < 0, dx, -dx)
            area = (mvs["w"] * mvs["h"]).astype(np.float32)
            feats.append(float((dx * area).sum() / area.sum()))
    return np.asarray(feats, np.float32) if feats else None

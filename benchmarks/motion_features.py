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


# Grey-level change (0-255) between consecutive frames above which a pixel
# counts as "changing". Codec noise on a static background stays well below
# it (measured max error 15 on a still mp4v clip); the blob's edges and
# texture are far above it.
CHANGE_THRESHOLD = 40


def flow_summary_features(frames_rgb: np.ndarray) -> np.ndarray:
    """Order-free summary of horizontal motion (6 numbers).

    ``[mean u, mean |u|, std u, consistency, speed, speed_mean]`` where u is
    the per-frame-pair mean horizontal flow and ``consistency = |sum u| /
    sum |u|`` (1 for a steady drift, near 0 for a back-and-forth oscillation).
    ``speed`` / ``speed_mean`` are the median / mean over frame pairs of the
    median |u| over *changing* pixels (grey-level change > CHANGE_THRESHOLD):
    the moving object's speed, independent of its size. Gating matters on
    small frames: Farneback spreads flow far beyond the object along
    background stripes (aperture problem), so whole-frame percentiles are
    unreliable. Used for the speed and periodic probes, where the phase of the
    motion is random so per-frame features are not aligned.
    """
    flow = farneback_flow(frames_rgb)
    u = flow[..., 0].mean(axis=(1, 2))
    abs_u = np.abs(flow[..., 0]).reshape(len(flow), -1)
    grey = frames_rgb.astype(np.float32).mean(axis=-1)
    changing = (np.abs(np.diff(grey, axis=0)) > CHANGE_THRESHOLD).reshape(len(flow), -1)
    gated = np.array([np.median(abs_u[i][changing[i]]) if changing[i].sum() > 3 else 0.0
                      for i in range(len(flow))])
    consistency = np.abs(u.sum()) / max(float(np.abs(u).sum()), 1e-6)
    return np.array([u.mean(), np.abs(u).mean(), u.std(), consistency,
                     np.median(gated), gated.mean()], np.float32)


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
    """Mean horizontal codec motion of each inter frame, via PyAV ``+export_mvs``.

    FFmpeg exports one motion vector per (sub-)block of each P/B frame: the
    block at ``dst`` in the current frame is predicted from ``src = dst +
    motion / motion_scale`` in the reference frame (past if ``source < 0``).
    The forward horizontal motion of a block is therefore ``-motion_x /
    motion_scale`` for a past reference and ``+motion_x / motion_scale`` for a
    future one; blocks are weighted by area. Frames with no exported vectors
    (I-frames) are skipped; intra-coded blocks inside P-frames export none.

    Returns ``None`` if PyAV is not installed or no frame carries vectors.
    """
    try:
        import av
    except ImportError:
        return None
    feats = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.options = {"flags2": "+export_mvs"}
        for frame in container.decode(stream):
            side = frame.side_data.get("MOTION_VECTORS")
            if side is None:
                continue
            mvs = side.to_ndarray()
            if len(mvs) == 0:
                feats.append(0.0)
                continue
            scale = np.maximum(mvs["motion_scale"].astype(np.float64), 1.0)
            forward = mvs["motion_x"].astype(np.float64) / scale
            forward = np.where(mvs["source"] < 0, -forward, forward)
            # w/h are uint8: widen before multiplying (16 * 16 overflows).
            area = mvs["w"].astype(np.float64) * mvs["h"].astype(np.float64)
            feats.append(float((forward * area).sum() / area.sum()))
    return np.asarray(feats, np.float32) if feats else None

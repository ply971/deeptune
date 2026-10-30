"""Rendering primitives for the synthetic video benchmarks.

Everything here is pure NumPy/OpenCV and deterministic given a
``numpy.random.Generator``. Frames are float32 RGB in [0, 1] while rendering
and uint8 RGB when written.

The world is a torus: a blob that leaves the right edge re-enters on the left
(and likewise vertically). That makes the blob position in *any* single frame
uniformly distributed whatever its velocity, so a single frame carries no
information about the direction of motion (see ``synthetic_motion``).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Background texture families. The scene-bias task ties a family to a class.
TEXTURE_FAMILIES = ("stripes", "dots", "checker", "noise")

# A pixel whose red channel exceeds both others by this much is "blob red".
# Backgrounds and distractors are rendered with chroma well below it, so the
# blob can be detected in decoded frames without access to the generator.
BLOB_REDNESS_THRESHOLD = 0.25
_BACKGROUND_MAX_CHROMA = 0.06


def _smooth_noise(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """Low-frequency noise in [0, 1] (bicubic upsampled random grid)."""
    grid = rng.random((cells + 1, cells + 1)).astype(np.float32)
    up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    up -= up.min()
    return up / max(float(up.max()), 1e-6)


def _muted_color(rng: np.random.Generator, value_range=(0.25, 0.75)) -> np.ndarray:
    """A low-saturation RGB colour (grey plus a small chroma offset)."""
    value = rng.uniform(*value_range)
    chroma = rng.uniform(-_BACKGROUND_MAX_CHROMA, _BACKGROUND_MAX_CHROMA, size=3)
    return np.clip(value + chroma, 0.0, 1.0).astype(np.float32)


def make_background(rng: np.random.Generator, size: int, family: str) -> np.ndarray:
    """Static textured background of a given family, shape [size, size, 3]."""
    if family not in TEXTURE_FAMILIES:
        raise ValueError(f"Unknown texture family {family!r}; choose from {TEXTURE_FAMILIES}.")
    h = w = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    if family == "stripes":
        theta = rng.uniform(0, np.pi)
        freq = rng.uniform(2.5, 5.0) / size  # cycles per pixel
        phase = rng.uniform(0, 2 * np.pi)
        pattern = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xs * np.cos(theta) + ys * np.sin(theta)) + phase)
    elif family == "dots":
        pattern = np.zeros((h, w), np.float32)
        n_dots = int(rng.integers(8, 16) * (size / 32) ** 2)
        for _ in range(n_dots):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            r = rng.uniform(1.0, 2.2) * size / 32
            pattern = np.maximum(pattern, np.clip(r + 0.5 - np.hypot(xs - cx, ys - cy), 0, 1))
    elif family == "checker":
        cell = rng.uniform(3.0, 6.0) * size / 32
        ox, oy = rng.uniform(0, cell, size=2)
        pattern = (((xs + ox) // cell + (ys + oy) // cell) % 2).astype(np.float32)
        pattern = cv2.GaussianBlur(pattern, (0, 0), 0.6)
    else:  # noise
        pattern = _smooth_noise(rng, h, w, cells=int(rng.integers(3, 6)))
    c1 = _muted_color(rng, (0.15, 0.45))
    c2 = _muted_color(rng, (0.55, 0.85))
    return (c1 * (1 - pattern[..., None]) + c2 * pattern[..., None]).astype(np.float32)


def make_blob_patch(rng: np.random.Generator, radius: float) -> np.ndarray:
    """Square texture patch for the moving blob: saturated warm (red) colours."""
    side = int(np.ceil(2 * radius)) + 4
    # Fine, high-contrast texture (cells about 3 px): optical flow needs texture
    # *inside* the blob, otherwise only its rim moves and flow under-estimates
    # the speed (seen with a smooth blob: GT-masked Farneback vs speed r = 0.6).
    pattern = _smooth_noise(rng, side, side, cells=max(3, side // 3))
    pattern = np.clip((pattern - 0.5) * 1.6 + 0.5, 0.0, 1.0)
    # Red-dominant colours (R - max(G, B) >= 0.35 for both ends, so every blob
    # pixel is "blob red") with a large brightness difference between them. The
    # blob is easy to find in decoded frames and never confused with the muted
    # background or the grey distractor.
    dark = np.array([rng.uniform(0.40, 0.55), rng.uniform(0.00, 0.05), rng.uniform(0.00, 0.05)], np.float32)
    bright = np.array([rng.uniform(0.95, 1.00), rng.uniform(0.40, 0.55), rng.uniform(0.25, 0.40)], np.float32)
    return (dark * (1 - pattern[..., None]) + bright * pattern[..., None]).astype(np.float32)


def _wrapped_offsets(size: int, center: float) -> np.ndarray:
    """Signed offsets of each pixel coordinate from ``center`` on a ring of ``size``."""
    coords = np.arange(size, dtype=np.float64)
    return (coords - center + size / 2.0) % size - size / 2.0


def paint_blob(frame: np.ndarray, patch: np.ndarray, cx: float, cy: float, radius: float):
    """Paint a textured disc centred at (cx, cy) on a torus, in place.

    The texture moves rigidly with the blob (sub-pixel accurate, bilinear).
    Returns the boolean mask of pixels covered by the blob (alpha > 0.5).
    """
    h, w = frame.shape[:2]
    dx = _wrapped_offsets(w, cx)[None, :].repeat(h, 0)
    dy = _wrapped_offsets(h, cy)[:, None].repeat(w, 1)
    alpha = np.clip(radius + 0.5 - np.hypot(dx, dy), 0.0, 1.0).astype(np.float32)
    centre = (patch.shape[0] - 1) / 2.0
    tex = cv2.remap(patch, (dx + centre).astype(np.float32), (dy + centre).astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame *= 1 - alpha[..., None]
    frame += tex * alpha[..., None]
    return alpha > 0.5


def paint_square(frame: np.ndarray, x0: float, y0: float, side: float, color: np.ndarray) -> None:
    """Paint a flat, anti-aliased axis-aligned square (the distractor), in place."""
    h, w = frame.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    ax = np.clip(np.minimum(xs + 1 - x0, x0 + side - xs), 0, 1)
    ay = np.clip(np.minimum(ys + 1 - y0, y0 + side - ys), 0, 1)
    alpha = (ay[:, None] * ax[None, :])[..., None]
    frame *= 1 - alpha
    frame += color * alpha


def to_uint8(frames: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)


def redness(frames_rgb_uint8: np.ndarray) -> np.ndarray:
    """Per-pixel red dominance in [-1, 1]: R - max(G, B) (inputs uint8 RGB)."""
    f = frames_rgb_uint8.astype(np.float32) / 255.0
    return f[..., 0] - np.maximum(f[..., 1], f[..., 2])


# ---------------------------------------------------------------------------
# MP4 writing
# ---------------------------------------------------------------------------

_CODEC_CACHE: dict[str, str] = {}


class _SilencedStderr:
    """Temporarily silence native (C-level) stderr: FFmpeg/OpenCV log there."""

    def __enter__(self):
        try:
            self._saved = os.dup(2)
            self._null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._null, 2)
        except OSError:
            self._saved = None
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.dup2(self._saved, 2)
            os.close(self._saved)
            os.close(self._null)
        return False


def _probe_fourcc(fourcc: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="deeptune-codec-") as folder:
        path = os.path.join(folder, "probe.mp4")
        with _SilencedStderr():
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 10, (16, 16))
            ok = writer.isOpened()
            if ok:
                for i in range(3):
                    writer.write(np.full((16, 16, 3), 40 * i, np.uint8))
            writer.release()
        if not ok or not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        cap = cv2.VideoCapture(path)
        ok, _ = cap.read()
        cap.release()
        return bool(ok)


def resolve_codec(codec: str = "auto") -> str:
    """Pick the FourCC to write MP4s with.

    ``"auto"`` prefers H.264 (``avc1``) - its bitstream carries motion vectors
    that Phase 1 can export - and falls back to MPEG-4 Part 2 (``mp4v``), which
    also stores motion vectors. OpenCV wheels usually lack an H.264 *encoder*
    (OpenH264 DLL missing), so on most machines the result is ``mp4v``. The
    chosen codec is recorded in every dataset's metadata.
    """
    if codec != "auto":
        if len(codec) != 4:
            raise ValueError(f"codec must be 'auto' or a 4-character FourCC, got {codec!r}.")
        return codec
    if "auto" not in _CODEC_CACHE:
        _CODEC_CACHE["auto"] = "avc1" if _probe_fourcc("avc1") else "mp4v"
    return _CODEC_CACHE["auto"]


def write_mp4(path: Path, frames_rgb: np.ndarray, fps: float, fourcc: str) -> None:
    """Write uint8 RGB frames [T, H, W, 3] to an MP4 file with OpenCV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames_rgb.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a {fourcc!r} video writer for {path}.")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

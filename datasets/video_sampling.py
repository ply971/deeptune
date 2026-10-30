"""Temporal sampling: which frame indices make up a clip.

Strategies (``--temporal_sampling``):

* ``uniform`` (default, DeepTune's original behaviour): ``num_frames``
  indices evenly spaced over the window, ``linspace(lo, hi - 1, n)``
  truncated to integers -- identical to the old ``ParquetVideoDataset``.
* ``segment`` (TSN, Wang et al., ECCV 2016): the window is cut into
  ``num_frames`` equal segments; training takes a random frame per segment,
  evaluation the centre frame.
* ``dense``: ``num_frames`` consecutive frames ``clip_stride`` apart (a
  fixed-length window, e.g. 16 frames x stride 2). Training picks a random
  window start; evaluation the centred window (several evenly spaced
  windows with ``num_clips > 1``). Later phases use it for sliding windows.
* ``motion`` (MGSampler, Zhi et al., ICCV 2021): build the cumulative motion
  curve ``C(t) = sum_{s<=t} m[s] / sum m`` (``m`` from codec motion vectors or
  frame differences, see ``VideoReader.motion_curve``) and sample uniformly
  along it, so frames are denser where things move. Training draws one
  random point per quantile bin, evaluation the bin centres.

``clip_duration_sec`` limits the window to that many seconds (training: a
random window position; evaluation: centred / evenly spaced). Temporal jitter
(training, ``--video_augment``) shortens the window to a random 80-100 % of
the available span at a random offset; speed perturbation scales the
window span (and the dense stride) by a random factor in
``[1 - s, 1 + s]``.

Clips shorter than needed are padded by repeating the last frame (dense) or
by repeated indices (uniform/segment/motion) -- the caller logs it.
Everything here is pure numpy; no decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

STRATEGIES = ('uniform', 'segment', 'dense', 'motion')


@dataclass(frozen=True)
class SamplingSpec:
    strategy: str = 'uniform'
    num_frames: int = 8
    clip_stride: Optional[int] = None        # dense only; default 1
    clip_duration_sec: Optional[float] = None
    temporal_jitter: bool = False            # training only
    speed_perturb: float = 0.0               # training only; 0 = off

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f'Unknown temporal sampling {self.strategy!r}; choose one of {", ".join(STRATEGIES)}.')
        if isinstance(self.num_frames, bool) or not isinstance(self.num_frames, (int, np.integer)) or self.num_frames < 1:
            raise ValueError('num_frames must be a positive integer.')
        if self.clip_stride is not None and int(self.clip_stride) < 1:
            raise ValueError('clip_stride must be at least 1.')
        if self.clip_duration_sec is not None and not float(self.clip_duration_sec) > 0:
            raise ValueError('clip_duration_sec must be positive.')
        if not 0 <= float(self.speed_perturb) < 1:
            raise ValueError('speed_perturb must be in [0, 1).')

    @property
    def needs_motion(self) -> bool:
        return self.strategy == 'motion'

    @property
    def stride(self) -> int:
        return int(self.clip_stride or 1)


def _window_length(spec: SamplingSpec, region: int, fps: float, train: bool, rng) -> tuple[int, float]:
    """Window length (frames) and dense stride after speed perturbation."""
    speed = 1.0
    if train and spec.speed_perturb > 0:
        speed = float(rng.uniform(1 - spec.speed_perturb, 1 + spec.speed_perturb))
    stride = spec.stride * speed
    if spec.strategy == 'dense':
        length = int(np.floor((spec.num_frames - 1) * stride)) + 1
    elif spec.clip_duration_sec is not None:
        if not fps or fps <= 0:
            raise ValueError('clip_duration_sec needs the clip frame rate, but this video reports none.')
        length = int(round(float(spec.clip_duration_sec) * fps * speed))
    else:
        length = int(round(region * min(1.0, speed)))
    if train and spec.temporal_jitter and spec.strategy != 'dense' and length >= region:
        length = int(round(region * rng.uniform(0.8, 1.0)))
    return max(1, min(length, region)), stride


def _window_starts(region: int, length: int, train: bool, num_clips: int, rng) -> np.ndarray:
    room = region - length
    if train:
        return np.asarray([int(rng.integers(0, room + 1)) for _ in range(num_clips)])
    if num_clips == 1:
        return np.asarray([room // 2])
    return np.round(np.linspace(0, room, num_clips)).astype(int)


def _offset(clip: int, num_clips: int) -> float:
    """Fractional position inside a segment for evaluation clip `clip` of `num_clips`."""
    return (clip + 0.5) / num_clips


def sample_clip_indices(frame_count: int, spec: SamplingSpec, train: bool = False, rng=None, num_clips: int = 1,
                        fps: float = 0.0, start_frame: int = 0, end_frame: Optional[int] = None,
                        motion: Optional[np.ndarray] = None) -> np.ndarray:
    """Frame indices ``[num_clips, num_frames]`` (int64) for one video.

    `frame_count`: frames in the video. `start_frame`/`end_frame`: optional
    sub-range (from a manifest's start_sec/end_sec). `motion`: per-frame
    motion magnitudes (length `frame_count`), required for ``motion``.
    `rng`: ``numpy.random.Generator`` for training randomness.
    """
    if frame_count < 1:
        raise ValueError('The video has no frames to sample.')
    if num_clips < 1:
        raise ValueError('num_clips must be at least 1.')
    rng = rng if rng is not None else np.random.default_rng(0)
    lo_region = int(min(max(0, start_frame), frame_count - 1))
    hi_region = int(frame_count if end_frame is None else min(max(end_frame, lo_region + 1), frame_count))
    region = hi_region - lo_region
    n = spec.num_frames
    clips = []
    for clip in range(num_clips):
        length, stride = _window_length(spec, region, fps, train, rng)
        if train:
            start = int(_window_starts(region, length, True, 1, rng)[0])
        else:
            start = int(_window_starts(region, length, False, num_clips, rng)[clip])
        lo = lo_region + start
        whole = length >= region and not train
        if spec.strategy == 'uniform':
            if num_clips == 1 or not whole:
                idx = lo + np.linspace(0, length - 1, n).astype(np.int64)
            else:  # several clips over the same span: evenly offset grids
                idx = lo + np.floor((np.arange(n) + _offset(clip, num_clips)) * length / n).astype(np.int64)
        elif spec.strategy == 'segment':
            edges = np.linspace(0, length, n + 1)
            if train:
                low = np.floor(edges[:-1]).astype(np.int64)
                high = np.maximum(low + 1, np.floor(edges[1:]).astype(np.int64))
                idx = lo + rng.integers(low, high)
            else:
                frac = _offset(clip, num_clips) if whole else 0.5
                idx = lo + np.floor(edges[:-1] + frac * (edges[1:] - edges[:-1])).astype(np.int64)
        elif spec.strategy == 'dense':
            idx = lo + np.floor(np.arange(n) * stride).astype(np.int64)
        else:  # motion (MGSampler)
            if motion is None:
                raise ValueError("The 'motion' sampling strategy needs a motion curve.")
            curve = np.asarray(motion, dtype=np.float64).reshape(-1)
            if len(curve) < frame_count:
                curve = np.concatenate([curve, np.full(frame_count - len(curve), curve[-1] if len(curve) else 0.0)])
            window = np.clip(curve[lo:lo + length], 0, None) + 1e-6 * max(1.0, float(np.abs(curve).max()))
            cumulative = np.cumsum(window) / window.sum()
            if train:
                q = (np.arange(n) + rng.uniform(0, 1, n)) / n
            else:
                q = (np.arange(n) + (_offset(clip, num_clips) if whole else 0.5)) / n
            idx = lo + np.minimum(np.searchsorted(cumulative, q), length - 1).astype(np.int64)
        clips.append(np.clip(idx, lo_region, hi_region - 1))
    return np.stack(clips).astype(np.int64)



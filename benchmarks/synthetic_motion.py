"""Seeded synthetic video benchmarks: the test oracle for motion, localisation
and background reliance (Phase 2).

Five tasks, each written as MP4 files + a CSV manifest + ground-truth JSON:

``direction``
    The course slide's experiment ("when motion is the whole signal"): a
    textured blob drifts over a static textured background; the two classes
    (``left``/``right``) differ *only* in the sign of the horizontal velocity.
    The trajectory is centred on the middle frame and the world wraps around
    (torus), so the blob position in any single frame is uniform whatever the
    class, and the frame average is identical for both signs.
``speed``
    Same scene, random direction; regression target = speed in px/frame.
``periodic``
    ``oscillate`` (sinusoidal back-and-forth) vs ``drift`` (constant velocity)
    with matched mean speed; random phase.
``untrimmed``
    Long videos (hundreds of frames) with background-only stretches (static or
    with a grey distractor square that moves but is not an action) and 1..k
    action instances of known class, start and end. Ground truth is written
    ActivityNet/CtxPL style (``video-id, t-start, t-end, label``) and as a
    video-level multi-hot label - the only label a weakly-supervised model may
    train on.
``scene_bias``
    Classes are motion directions; each class also has a background texture
    family. In ``train`` (and the biased ``val``) the family matches the class
    in ``bias`` of the clips; in ``test`` family and class are independent
    (every class x family combination equally often). Protocol inspired by
    Choi et al., "Why Can't I Dance in the Mall?" (NeurIPS 2019).

Determinism: every clip is rendered from its own ``numpy`` generator seeded by
``(seed, task, split, index)``, so the same arguments always produce the same
pixels, independent of how many other clips are generated.

Output layout (``out/``)::

    videos/<video_id>.mp4
    manifest.csv            all clips; videos, labels, group, split, video_id, label_idx, ...
    train.csv, test.csv     (and val.csv) same columns, one split each
    dataset.json            task, seed, params, codec, classes, per-video ground truth
    masks.npz               per-video blob masks (np.packbits of bool [T, H, W])
    gt_segments.csv         untrimmed only: video-id, t-start, t-end, label
    gt_activitynet.json     untrimmed only: ActivityNet-style "database"

CLI::

    python -m benchmarks.synthetic_motion --task direction --out .artifacts/bench/direction
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.render import (
    TEXTURE_FAMILIES,
    make_background,
    make_blob_patch,
    paint_blob,
    paint_square,
    resolve_codec,
    to_uint8,
    write_mp4,
)

TASKS = ("direction", "speed", "periodic", "untrimmed", "scene_bias")
_TASK_CODES = {name: i + 1 for i, name in enumerate(TASKS)}
_SPLIT_CODES = {"train": 0, "val": 1, "test": 2}

# Motion patterns that define the classes of the untrimmed task.
UNTRIMMED_PATTERNS = ("right", "left", "oscillate", "up", "down", "circle")
# Motion patterns that define the classes of the scene-bias task (one
# background family per class, in TEXTURE_FAMILIES order).
SCENE_BIAS_PATTERNS = ("right", "left", "down", "up")

FORMAT_VERSION = 1


def _rng(seed: int, task: str, *keys: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), _TASK_CODES[task], *[int(k) for k in keys]])


def _scaled(range_px_at_32: tuple[float, float], size: int) -> tuple[float, float]:
    """Speed ranges are specified for 32x32 frames and scale with the frame size."""
    return (range_px_at_32[0] * size / 32.0, range_px_at_32[1] * size / 32.0)


def _check_common(num_frames: int, size: int) -> None:
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2.")
    if size < 16 or size % 2:
        raise ValueError("size must be an even number >= 16 (MPEG-4 needs even dimensions).")


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkDataset:
    """What a generator produced (all paths absolute)."""
    task: str
    root: Path
    manifest: Path
    metadata: Path
    masks: Path
    splits: dict[str, Path]
    codec: str
    classes: list[str]
    extra: dict[str, Path] = field(default_factory=dict)

    def load_metadata(self) -> dict:
        return json.loads(self.metadata.read_text(encoding="utf-8"))

    def load_manifest(self, split: str | None = None) -> pd.DataFrame:
        return pd.read_csv(self.splits[split] if split else self.manifest)


class _Writer:
    def __init__(self, out: Path, task: str, seed: int, params: dict, codec: str, fps: float,
                 size: int, classes: list[str] | None, hflip_safe: bool, vflip_safe: bool):
        self.root = Path(out).expanduser().resolve()
        (self.root / "videos").mkdir(parents=True, exist_ok=True)
        self.task, self.fps, self.size = task, float(fps), size
        self.codec = resolve_codec(codec)
        self.rows: list[dict] = []
        self.masks: dict[str, np.ndarray] = {}
        self.meta = {
            "format_version": FORMAT_VERSION,
            "task": task,
            "seed": int(seed),
            "params": params,
            "codec": self.codec,
            "container": "mp4",
            "fps": self.fps,
            "size": [size, size],
            "classes": classes,
            # Horizontal flip swaps left/right motion: it must stay off for tasks
            # whose labels depend on direction (see benchmarks.guards).
            "hflip_safe": hflip_safe,
            "vflip_safe": vflip_safe,
            "videos": {},
        }

    def add(self, video_id: str, frames: np.ndarray, masks: np.ndarray, row: dict, info: dict) -> None:
        rel = f"videos/{video_id}.mp4"
        write_mp4(self.root / rel, frames, self.fps, self.codec)
        self.masks[video_id] = np.packbits(masks.reshape(-1))
        self.rows.append({"videos": rel, **row, "video_id": video_id})
        self.meta["videos"][video_id] = {
            "file": rel,
            "num_frames": int(frames.shape[0]),
            "duration_sec": round(frames.shape[0] / self.fps, 6),
            "mask_shape": list(masks.shape),
            **info,
        }

    def finish(self, classes: list[str]) -> BenchmarkDataset:
        frame = pd.DataFrame(self.rows)
        manifest = self.root / "manifest.csv"
        frame.to_csv(manifest, index=False)
        splits = {}
        for split in ("train", "val", "test"):
            part = frame[frame["split"] == split]
            if len(part):
                splits[split] = self.root / f"{split}.csv"
                part.to_csv(splits[split], index=False)
        metadata = self.root / "dataset.json"
        metadata.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        masks = self.root / "masks.npz"
        np.savez_compressed(masks, **self.masks)
        return BenchmarkDataset(self.task, self.root, manifest, metadata, masks, splits, self.codec, classes)


def load_blob_masks(root: Path, video_id: str) -> np.ndarray:
    """Boolean blob masks [T, H, W] of one video (True = moving blob / action)."""
    root = Path(root)
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    shape = meta["videos"][video_id]["mask_shape"]
    with np.load(root / "masks.npz") as packed:
        bits = packed[video_id]
    return np.unpackbits(bits, count=int(np.prod(shape))).astype(bool).reshape(shape)


# ---------------------------------------------------------------------------
# Trimmed clips (direction / speed / periodic / scene_bias)
# ---------------------------------------------------------------------------

def _render_track(rng: np.random.Generator, background: np.ndarray, centers: np.ndarray,
                  radius: float) -> tuple[np.ndarray, np.ndarray]:
    patch = make_blob_patch(rng, radius)
    frames, masks = [], []
    for cx, cy in centers:
        frame = background.copy()
        masks.append(paint_blob(frame, patch, float(cx), float(cy), radius))
        frames.append(frame)
    return to_uint8(np.stack(frames)), np.stack(masks)


def _centered_times(num_frames: int) -> np.ndarray:
    """Frame times relative to the clip middle, symmetric: t - (T-1)/2."""
    return np.arange(num_frames, dtype=np.float64) - (num_frames - 1) / 2.0


def _linear_track(rng, size, num_frames, vx, vy) -> np.ndarray:
    cx, cy = rng.uniform(0, size, size=2)
    t = _centered_times(num_frames)
    return np.stack([(cx + vx * t) % size, (cy + vy * t) % size], axis=1)


def _radius(rng, size) -> float:
    return float(rng.uniform(0.15, 0.21) * size)


def _round_track(centers: np.ndarray) -> list:
    return np.round(centers, 4).tolist()


def _split_plan(n_train: int, n_test: int, n_val: int = 0):
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for index in range(count):
            yield split, index


def make_direction_dataset(out: Path, n_train: int = 700, n_test: int = 400, num_frames: int = 12,
                           size: int = 64, seed: int = 0, speed_range=(1.0, 2.0), fps: float = 10,
                           clips_per_scene: int = 2, codec: str = "auto") -> BenchmarkDataset:
    """The slide's direction task: classes differ only in the sign of vx.

    Each *scene* (a background instance, the ``group``) hosts
    ``clips_per_scene`` clips with alternating classes, so a scene's
    background is shared by both classes; train and test scenes are disjoint.
    Speed |vx| ~ U(speed_range) px/frame (given for 32x32, scaled with size),
    a small class-independent vy ~ U(-0.25, 0.25)|vx|.
    """
    _check_common(num_frames, size)
    classes = ["left", "right"]
    params = dict(n_train=n_train, n_test=n_test, num_frames=num_frames, size=size,
                  speed_range_px_per_frame=list(_scaled(speed_range, size)), clips_per_scene=clips_per_scene)
    writer = _Writer(out, "direction", seed, params, codec, fps, size, classes, hflip_safe=False, vflip_safe=True)
    lo, hi = _scaled(speed_range, size)
    for split, index in _split_plan(n_train, n_test):
        scene = index // clips_per_scene
        scene_rng = _rng(seed, "direction", _SPLIT_CODES[split], 1, scene)
        family = TEXTURE_FAMILIES[int(scene_rng.integers(len(TEXTURE_FAMILIES)))]
        background = make_background(scene_rng, size, family)
        rng = _rng(seed, "direction", _SPLIT_CODES[split], 0, index)
        label_idx = index % 2
        speed = rng.uniform(lo, hi)
        vx = speed if classes[label_idx] == "right" else -speed
        vy = rng.uniform(-0.25, 0.25) * speed
        centers = _linear_track(rng, size, num_frames, vx, vy)
        frames, masks = _render_track(rng, background, centers, _radius(rng, size))
        vid = f"direction_{split}_{index:05d}"
        group = f"{split}-scene{scene:05d}"
        writer.add(vid, frames, masks,
                   dict(labels=classes[label_idx], label_idx=label_idx, group=group, split=split,
                        background_family=family),
                   dict(split=split, group=group, label=classes[label_idx], label_idx=label_idx,
                        background_family=family, velocity=[round(vx, 6), round(vy, 6)],
                        centers=_round_track(centers)))
    return writer.finish(classes)


def make_speed_dataset(out: Path, n_train: int = 700, n_test: int = 400, num_frames: int = 12,
                       size: int = 64, seed: int = 0, speed_range=(0.5, 3.0), fps: float = 10,
                       clips_per_scene: int = 2, codec: str = "auto") -> BenchmarkDataset:
    """Speed regression (``mode=reg``): label = |vx| in px/frame, random sign."""
    _check_common(num_frames, size)
    lo, hi = _scaled(speed_range, size)
    params = dict(n_train=n_train, n_test=n_test, num_frames=num_frames, size=size,
                  speed_range_px_per_frame=[lo, hi], clips_per_scene=clips_per_scene)
    writer = _Writer(out, "speed", seed, params, codec, fps, size, None, hflip_safe=True, vflip_safe=True)
    for split, index in _split_plan(n_train, n_test):
        scene = index // clips_per_scene
        scene_rng = _rng(seed, "speed", _SPLIT_CODES[split], 1, scene)
        family = TEXTURE_FAMILIES[int(scene_rng.integers(len(TEXTURE_FAMILIES)))]
        background = make_background(scene_rng, size, family)
        rng = _rng(seed, "speed", _SPLIT_CODES[split], 0, index)
        speed = float(rng.uniform(lo, hi))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        vx, vy = sign * speed, 0.0
        centers = _linear_track(rng, size, num_frames, vx, vy)
        frames, masks = _render_track(rng, background, centers, _radius(rng, size))
        vid = f"speed_{split}_{index:05d}"
        group = f"{split}-scene{scene:05d}"
        writer.add(vid, frames, masks,
                   dict(labels=round(speed, 6), group=group, split=split, background_family=family),
                   dict(split=split, group=group, label=round(speed, 6), background_family=family,
                        velocity=[round(vx, 6), vy], centers=_round_track(centers)))
    return writer.finish([])


def make_periodic_dataset(out: Path, n_train: int = 700, n_test: int = 400, num_frames: int = 16,
                          size: int = 64, seed: int = 0, speed_range=(1.0, 2.0), cycles_range=(1.0, 2.0),
                          fps: float = 10, clips_per_scene: int = 2, codec: str = "auto") -> BenchmarkDataset:
    """``oscillate`` (horizontal sinusoid) vs ``drift`` (constant velocity).

    Both classes have the same mean absolute speed m ~ U(speed_range): the
    drift moves at +-m, the oscillation completes ``cycles_range`` cycles over
    the clip with amplitude A = m / (4 f) (f = cycles per frame), random phase.
    """
    _check_common(num_frames, size)
    classes = ["drift", "oscillate"]
    lo, hi = _scaled(speed_range, size)
    params = dict(n_train=n_train, n_test=n_test, num_frames=num_frames, size=size,
                  speed_range_px_per_frame=[lo, hi], cycles_range=list(cycles_range), clips_per_scene=clips_per_scene)
    writer = _Writer(out, "periodic", seed, params, codec, fps, size, classes, hflip_safe=True, vflip_safe=True)
    for split, index in _split_plan(n_train, n_test):
        scene = index // clips_per_scene
        scene_rng = _rng(seed, "periodic", _SPLIT_CODES[split], 1, scene)
        family = TEXTURE_FAMILIES[int(scene_rng.integers(len(TEXTURE_FAMILIES)))]
        background = make_background(scene_rng, size, family)
        rng = _rng(seed, "periodic", _SPLIT_CODES[split], 0, index)
        label_idx = index % 2
        mean_speed = float(rng.uniform(lo, hi))
        cx, cy = rng.uniform(0, size, size=2)
        t = _centered_times(num_frames)
        if classes[label_idx] == "drift":
            sign = 1.0 if rng.random() < 0.5 else -1.0
            xs = cx + sign * mean_speed * t
            motion = dict(kind="drift", velocity=[round(sign * mean_speed, 6), 0.0])
        else:
            freq = float(rng.uniform(*cycles_range)) / num_frames
            amp = mean_speed / (4 * freq)
            phase = float(rng.uniform(0, 2 * np.pi))
            xs = cx + amp * np.sin(2 * np.pi * freq * t + phase)
            motion = dict(kind="oscillate", amplitude=round(amp, 6), cycles_per_frame=round(freq, 6),
                          phase=round(phase, 6))
        centers = np.stack([xs % size, np.full(num_frames, cy)], axis=1)
        frames, masks = _render_track(rng, background, centers, _radius(rng, size))
        vid = f"periodic_{split}_{index:05d}"
        group = f"{split}-scene{scene:05d}"
        writer.add(vid, frames, masks,
                   dict(labels=classes[label_idx], label_idx=label_idx, group=group, split=split,
                        background_family=family),
                   dict(split=split, group=group, label=classes[label_idx], label_idx=label_idx,
                        background_family=family, mean_speed=round(mean_speed, 6), motion=motion,
                        centers=_round_track(centers)))
    return writer.finish(classes)


def _direction_velocity(pattern: str, speed: float) -> tuple[float, float]:
    return {"right": (speed, 0.0), "left": (-speed, 0.0), "down": (0.0, speed), "up": (0.0, -speed)}[pattern]


def make_scene_bias_dataset(out: Path, n_train: int = 700, n_test: int = 400, n_val: int = 200,
                            num_frames: int = 12, size: int = 64, seed: int = 0, n_classes: int = 2,
                            bias: float = 0.9, speed_range=(1.0, 2.0), fps: float = 10,
                            codec: str = "auto") -> BenchmarkDataset:
    """Background texture family correlates with the class in train/val only.

    Class c moves in direction ``SCENE_BIAS_PATTERNS[c']`` and is paired with
    family ``TEXTURE_FAMILIES[c']``. In ``train`` and ``val`` (the biased
    distribution) a clip gets its class's family with probability ``bias``,
    otherwise one of the other families uniformly. In ``test`` every
    class x family combination occurs equally often (independent). Each clip
    has its own background instance, so ``group`` = clip.
    """
    _check_common(num_frames, size)
    if not 2 <= n_classes <= len(SCENE_BIAS_PATTERNS):
        raise ValueError(f"n_classes must be between 2 and {len(SCENE_BIAS_PATTERNS)}.")
    if not 0.0 <= bias <= 1.0:
        raise ValueError("bias must be in [0, 1].")
    patterns = list(SCENE_BIAS_PATTERNS[:n_classes])
    families = list(TEXTURE_FAMILIES[:n_classes])
    pair = dict(zip(patterns, families))
    classes = sorted(patterns)
    lo, hi = _scaled(speed_range, size)
    params = dict(n_train=n_train, n_val=n_val, n_test=n_test, num_frames=num_frames, size=size,
                  n_classes=n_classes, bias=bias, speed_range_px_per_frame=[lo, hi],
                  class_family={c: pair[c] for c in classes})
    writer = _Writer(out, "scene_bias", seed, params, codec, fps, size, classes, hflip_safe=False,
                     vflip_safe=n_classes <= 2)
    for split, index in _split_plan(n_train, n_test, n_val):
        rng = _rng(seed, "scene_bias", _SPLIT_CODES[split], 0, index)
        label_idx = index % n_classes
        label = classes[label_idx]
        if split == "test":
            family = families[(index // n_classes) % n_classes]
        elif rng.random() < bias:
            family = pair[label]
        else:
            others = [f for f in families if f != pair[label]]
            family = others[int(rng.integers(len(others)))]
        background = make_background(rng, size, family)
        speed = float(rng.uniform(lo, hi))
        vx, vy = _direction_velocity(label, speed)
        centers = _linear_track(rng, size, num_frames, vx, vy)
        frames, masks = _render_track(rng, background, centers, _radius(rng, size))
        vid = f"scene_bias_{split}_{index:05d}"
        writer.add(vid, frames, masks,
                   dict(labels=label, label_idx=label_idx, group=vid, split=split, background_family=family,
                        family_matches_class=family == pair[label]),
                   dict(split=split, group=vid, label=label, label_idx=label_idx, background_family=family,
                        family_matches_class=family == pair[label], velocity=[round(vx, 6), round(vy, 6)],
                        centers=_round_track(centers)))
    return writer.finish(classes)


# ---------------------------------------------------------------------------
# Untrimmed videos (weakly-supervised localisation oracle)
# ---------------------------------------------------------------------------

def _place_segments(rng, length, n_instances, seg_range, min_gap):
    """Non-overlapping [start, end) frame intervals, >= min_gap background
    frames before the first, between any two, and after the last."""
    lengths = [int(rng.integers(seg_range[0], seg_range[1] + 1)) for _ in range(n_instances)]
    while lengths and sum(lengths) + (len(lengths) + 1) * min_gap > length:
        lengths.pop()
    if not lengths:
        raise ValueError("Video too short for even one action instance; raise length_range or lower segment lengths.")
    slack = length - sum(lengths) - (len(lengths) + 1) * min_gap
    cuts = np.sort(rng.integers(0, slack + 1, size=len(lengths)))
    segments, cursor, used = [], 0, 0
    for seg_len, cut in zip(lengths, cuts):
        start = cursor + min_gap + (int(cut) - used)
        used = int(cut)
        segments.append((start, start + seg_len))
        cursor = start + seg_len
    return segments


def _pattern_track(rng, pattern, n, size, speed):
    x0, y0 = rng.uniform(0, size, size=2)
    t = np.arange(n, dtype=np.float64)
    if pattern in ("right", "left", "up", "down"):
        vx, vy = _direction_velocity(pattern, speed)
        xs, ys = x0 + vx * t, y0 + vy * t
        motion = dict(kind=pattern, velocity=[round(vx, 6), round(vy, 6)])
    elif pattern == "oscillate":
        period = float(rng.uniform(10, 20))
        amp = speed * period / 4
        phase = float(rng.uniform(0, 2 * np.pi))
        xs, ys = x0 + amp * np.sin(2 * np.pi * t / period + phase), np.full(n, y0)
        motion = dict(kind=pattern, amplitude=round(amp, 6), period_frames=round(period, 6))
    elif pattern == "circle":
        period = float(rng.uniform(12, 24))
        rad = speed * period / (2 * np.pi)
        turn = 1.0 if rng.random() < 0.5 else -1.0
        phase = float(rng.uniform(0, 2 * np.pi))
        ang = turn * 2 * np.pi * t / period + phase
        xs, ys = x0 + rad * np.cos(ang), y0 + rad * np.sin(ang)
        motion = dict(kind=pattern, radius=round(rad, 6), period_frames=round(period, 6), turn=turn)
    else:
        raise ValueError(f"Unknown motion pattern {pattern!r}.")
    return np.stack([xs % size, ys % size], axis=1), motion


def make_untrimmed_dataset(out: Path, n_videos: int = 40, length_range=(150, 300), n_classes: int = 3,
                           seed: int = 0, size: int = 48, fps: float = 10, max_instances: int = 3,
                           segment_len_range=(16, 40), min_gap: int = 8, background_motion: str = "mixed",
                           speed_range=(1.0, 2.0), test_fraction: float = 0.25,
                           codec: str = "auto") -> BenchmarkDataset:
    """Untrimmed videos with background-only stretches and 1..max_instances actions.

    Classes are motion patterns (``UNTRIMMED_PATTERNS[:n_classes]``). The blob
    is visible exactly inside action segments: frame f shows the blob iff
    ``start_frame <= f < end_frame`` for some segment (``t_start = start_frame /
    fps``, ``t_end = end_frame / fps``). ``background_motion``: ``static`` (no
    motion outside actions), ``distractor`` (a grey square bounces vertically
    for the whole video - motion that is not an action), or ``mixed`` (per video).
    """
    if not 1 <= n_classes <= len(UNTRIMMED_PATTERNS):
        raise ValueError(f"n_classes must be between 1 and {len(UNTRIMMED_PATTERNS)}.")
    if background_motion not in ("static", "distractor", "mixed"):
        raise ValueError("background_motion must be static, distractor or mixed.")
    if length_range[0] > length_range[1] or segment_len_range[0] > segment_len_range[1]:
        raise ValueError("Ranges must be (low, high) with low <= high.")
    _check_common(length_range[0], size)
    classes = sorted(UNTRIMMED_PATTERNS[:n_classes])
    lo, hi = _scaled(speed_range, size)
    n_test = int(round(n_videos * test_fraction))
    params = dict(n_videos=n_videos, n_test=n_test, length_range=list(length_range), n_classes=n_classes,
                  size=size, max_instances=max_instances, segment_len_range=list(segment_len_range),
                  min_gap=min_gap, background_motion=background_motion, speed_range_px_per_frame=[lo, hi])
    writer = _Writer(out, "untrimmed", seed, params, codec, fps, size, classes, hflip_safe=False,
                     vflip_safe=not ({"up", "down"} & set(classes)) and "circle" not in classes)
    segment_rows, anet = [], {}
    for index in range(n_videos):
        split = "test" if index >= n_videos - n_test else "train"
        rng = _rng(seed, "untrimmed", 0, 0, index)
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        family = TEXTURE_FAMILIES[int(rng.integers(len(TEXTURE_FAMILIES)))]
        background = make_background(rng, size, family)
        motion_kind = background_motion if background_motion != "mixed" else (
            "distractor" if rng.random() < 0.5 else "static")
        segments = _place_segments(rng, length, int(rng.integers(1, max_instances + 1)), segment_len_range, min_gap)

        frames = np.repeat(background[None], length, axis=0)
        masks = np.zeros((length, size, size), bool)
        centers: list = [None] * length
        distractor = None
        if motion_kind == "distractor":
            side = 0.15 * size
            x0 = float(rng.uniform(0, size - side))
            speed = float(rng.uniform(0.4, 0.8) * size / 32)
            span = size - side
            phase = float(rng.uniform(0, 2 * span))
            pos = (phase + speed * np.arange(length)) % (2 * span)
            ys = np.where(pos <= span, pos, 2 * span - pos)  # bounce (triangle wave)
            # A clearly visible grey: light on a dark background, dark on a light one.
            grey = float(rng.uniform(0.75, 0.9) if background.mean() < 0.5 else rng.uniform(0.1, 0.25))
            colour = np.full(3, grey, np.float32)
            for f in range(length):
                paint_square(frames[f], x0, float(ys[f]), side, colour)
            distractor = dict(x0=round(x0, 4), side=round(side, 4), colour=round(grey, 4),
                              y=np.round(ys, 4).tolist())

        seg_info = []
        for start, end in segments:
            label_idx = int(rng.integers(n_classes))
            label = classes[label_idx]
            radius = _radius(rng, size)
            track, motion = _pattern_track(rng, label, end - start, size, float(rng.uniform(lo, hi)))
            patch = make_blob_patch(rng, radius)
            for k, f in enumerate(range(start, end)):
                masks[f] = paint_blob(frames[f], patch, float(track[k, 0]), float(track[k, 1]), radius)
                centers[f] = [round(float(track[k, 0]), 4), round(float(track[k, 1]), 4)]
            seg_info.append(dict(label=label, label_idx=label_idx, start_frame=start, end_frame=end,
                                 t_start=round(start / fps, 6), t_end=round(end / fps, 6),
                                 radius=round(radius, 4), motion=motion))
        present = sorted({s["label"] for s in seg_info})
        multi_hot = [int(c in present) for c in classes]
        background_intervals, cursor = [], 0
        for s in seg_info:
            if s["start_frame"] > cursor:
                background_intervals.append([cursor, s["start_frame"]])
            cursor = s["end_frame"]
        if cursor < length:
            background_intervals.append([cursor, length])

        vid = f"untrimmed_{index:04d}"
        writer.add(vid, to_uint8(frames), masks,
                   dict(labels=";".join(present), group=vid, split=split,
                        **{f"label_{c}": h for c, h in zip(classes, multi_hot)},
                        num_frames=length, fps=fps, duration_sec=round(length / fps, 6),
                        background_family=family, background_motion=motion_kind),
                   dict(split=split, group=vid, multi_hot=multi_hot, labels=present, segments=seg_info,
                        background_intervals=background_intervals, background_family=family,
                        background_motion=motion_kind, distractor=distractor, centers=centers))
        for s in seg_info:
            segment_rows.append({"video-id": vid, "t-start": s["t_start"], "t-end": s["t_end"], "label": s["label"]})
        anet[vid] = {"subset": "validation" if split == "test" else "training",
                     "duration": round(length / fps, 6), "fps": fps, "num_frames": length,
                     "annotations": [{"segment": [s["t_start"], s["t_end"]], "label": s["label"]} for s in seg_info]}

    result = writer.finish(classes)
    gt_csv = result.root / "gt_segments.csv"
    pd.DataFrame(segment_rows, columns=["video-id", "t-start", "t-end", "label"]).to_csv(gt_csv, index=False)
    gt_json = result.root / "gt_activitynet.json"
    gt_json.write_text(json.dumps({"version": "deeptune-synthetic-untrimmed-1", "taxonomy": classes,
                                   "database": anet}, indent=1), encoding="utf-8")
    result.extra = {"gt_segments": gt_csv, "gt_activitynet": gt_json}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(argv=None):
    p = argparse.ArgumentParser(prog="python -m benchmarks.synthetic_motion",
                                description="Generate a seeded synthetic video benchmark (MP4 + manifest + ground truth).")
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output folder (created).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=None, help="Frame side in px (default 64; untrimmed 48).")
    p.add_argument("--num_frames", type=int, default=None, help="Frames per clip (default 12; periodic 16).")
    p.add_argument("--n_train", type=int, default=700)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_val", type=int, default=200, help="scene_bias only: biased held-out split.")
    p.add_argument("--bias", type=float, default=0.9, help="scene_bias only: P(family matches class) in train/val.")
    p.add_argument("--n_classes", type=int, default=None, help="untrimmed (default 3) / scene_bias (default 2).")
    p.add_argument("--n_videos", type=int, default=40, help="untrimmed only.")
    p.add_argument("--min_length", type=int, default=150, help="untrimmed only: shortest video in frames.")
    p.add_argument("--max_length", type=int, default=300, help="untrimmed only: longest video in frames.")
    p.add_argument("--max_instances", type=int, default=3, help="untrimmed only.")
    p.add_argument("--background_motion", choices=["static", "distractor", "mixed"], default="mixed")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--codec", default="auto", help="'auto' (H.264 if the OpenCV build can encode it, else mp4v) or a FourCC.")
    return p.parse_args(argv)


def generate(args) -> BenchmarkDataset:
    common = dict(seed=args.seed, fps=args.fps, codec=args.codec)
    if args.task == "untrimmed":
        return make_untrimmed_dataset(args.out, n_videos=args.n_videos, length_range=(args.min_length, args.max_length),
                                      n_classes=args.n_classes or 3, size=args.size or 48,
                                      max_instances=args.max_instances, background_motion=args.background_motion,
                                      **common)
    sized = dict(n_train=args.n_train, n_test=args.n_test, size=args.size or 64, **common)
    if args.task == "direction":
        return make_direction_dataset(args.out, num_frames=args.num_frames or 12, **sized)
    if args.task == "speed":
        return make_speed_dataset(args.out, num_frames=args.num_frames or 12, **sized)
    if args.task == "periodic":
        return make_periodic_dataset(args.out, num_frames=args.num_frames or 16, **sized)
    return make_scene_bias_dataset(args.out, n_val=args.n_val, num_frames=args.num_frames or 12,
                                   n_classes=args.n_classes or 2, bias=args.bias, **sized)


def main(argv=None) -> None:
    args = _parse(argv)
    result = generate(args)
    frame = pd.read_csv(result.manifest)
    print(json.dumps({"task": result.task, "root": str(result.root), "clips": len(frame),
                      "splits": {k: int((frame["split"] == k).sum()) for k in result.splits},
                      "codec": result.codec, "classes": result.classes,
                      "manifest": str(result.manifest), "metadata": str(result.metadata)}, indent=2))


if __name__ == "__main__":
    main()

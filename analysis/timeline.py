"""Sliding-window timeline: "what happens when" in a video of any length.

A clip classifier answers one question per clip. Run over overlapping windows
of a long video, it gives a probability per class over time; smoothing that
signal and merging runs of the same label turns it into **event segments**
``{start, end, label, confidence}``.

Pieces (all torch-free except the caller's ``score_windows``):

- :func:`make_windows` - window start/end times covering the whole video.
- :func:`median_smooth` - per-class running median over windows (removes
  one-window flicker), renormalised to sum to 1.
- :func:`owned_intervals` - overlapping windows -> a partition of the video:
  each window "owns" the stretch nearest its centre, so every second belongs to
  exactly one window and segments never overlap.
- :func:`segments_from_timeline` - argmax label per window, merge equal
  neighbours, absorb segments shorter than ``min_segment_sec`` into a
  neighbour, drop ``background`` labels from the segment list.
- :func:`run_timeline` - the whole pass for one video, given
  ``score_windows(windows) -> probabilities [W, C]`` (``infer.py --timeline``
  passes the trained run's own clip pipeline, so windows are sampled and
  preprocessed exactly as at training time). Windows are scored in batches and
  each decodes only its own frames, so memory does not grow with video length.

Output files: ``timeline.csv`` (one row per window) and ``segments.json``.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Window:
    index: int
    start: float
    end: float

    @property
    def centre(self) -> float:
        return (self.start + self.end) / 2.0


def make_windows(duration_sec: float, window_sec: float, stride_sec: float) -> list[Window]:
    """Windows ``[k * stride, k * stride + window]`` until the video ends; the last
    window is moved back to end exactly at the video's end, so the tail is covered.
    A video shorter than one window gets a single window ``[0, duration]``."""
    if not (window_sec > 0 and stride_sec > 0):
        raise ValueError('window_sec and stride_sec must be > 0')
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError(f'video duration must be > 0 seconds, got {duration_sec}')
    if duration_sec <= window_sec:
        return [Window(0, 0.0, float(duration_sec))]
    starts = []
    t = 0.0
    while t + window_sec < duration_sec - 1e-9:
        starts.append(t)
        t = round(t + stride_sec, 9)
    last = duration_sec - window_sec
    if not starts or last - starts[-1] > 1e-6:
        starts.append(last)
    return [Window(i, round(s, 6), round(s + window_sec, 6)) for i, s in enumerate(starts)]


def median_smooth(probs: np.ndarray, width: int) -> np.ndarray:
    """Running median of each class over ``width`` windows (odd; 1 = unchanged), renormalised.

    Edges use the windows available (the median of a shorter neighbourhood)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError('probs must be [windows, classes]')
    if width < 1 or width % 2 == 0:
        raise ValueError('the smoothing width must be an odd number >= 1')
    if width == 1 or len(probs) < 2:
        return probs.copy()
    half = width // 2
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = np.median(probs[max(0, i - half):i + half + 1], axis=0)
    total = out.sum(axis=1, keepdims=True)
    return np.where(total > 0, out / np.maximum(total, 1e-12), probs)


def owned_intervals(windows: Sequence[Window], duration_sec: float) -> list[tuple[float, float]]:
    """Each window owns the time closer to its centre than to its neighbours' centres."""
    if not windows:
        return []
    centres = [w.centre for w in windows]
    bounds = [0.0] + [(a + b) / 2.0 for a, b in zip(centres, centres[1:])] + [float(duration_sec)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(windows))]


@dataclass
class Segment:
    start: float
    end: float
    label: str
    confidence: float
    windows: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {'start_sec': round(self.start, 3), 'end_sec': round(self.end, 3), 'label': self.label,
                'confidence': round(float(self.confidence), 4), 'windows': self.windows}


def _merge_equal(segments: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segments:
        if out and out[-1].label == s.label:
            prev = out[-1]
            n = prev.windows + s.windows
            out[-1] = Segment(prev.start, s.end, s.label, (prev.confidence * prev.windows + s.confidence * s.windows) / n, n)
        else:
            out.append(Segment(s.start, s.end, s.label, s.confidence, s.windows))
    return out


def segments_from_timeline(intervals: Sequence[tuple[float, float]], labels: Sequence[str],
                           confidences: Sequence[float], min_segment_sec: float = 0.0,
                           background: Optional[Sequence[str]] = None) -> list[Segment]:
    """Turn per-window labels into segments.

    Equal neighbours merge (confidence = mean over their windows). Then, while a
    segment is shorter than ``min_segment_sec`` (and there is more than one), the
    shortest such segment is absorbed by the neighbour it is most like: the
    neighbour with the same label as the other side if both sides agree, else the
    longer neighbour. Labels in ``background`` are dropped from the result (after
    merging, so a short background gap still splits two events).
    """
    if not (len(intervals) == len(labels) == len(confidences)):
        raise ValueError('intervals, labels and confidences must have the same length')
    segs = _merge_equal([Segment(a, b, str(l), float(c), 1)
                         for (a, b), l, c in zip(intervals, labels, confidences)])
    while min_segment_sec > 0 and len(segs) > 1:
        short = [i for i, s in enumerate(segs) if s.duration < min_segment_sec - 1e-9]
        if not short:
            break
        i = min(short, key=lambda k: (segs[k].duration, k))
        left = segs[i - 1] if i > 0 else None
        right = segs[i + 1] if i + 1 < len(segs) else None
        if left is not None and right is not None:
            target = i - 1 if left.duration >= right.duration else i + 1
        else:
            target = i - 1 if left is not None else i + 1
        t = segs[target]
        s = segs[i]
        merged = Segment(min(t.start, s.start), max(t.end, s.end), t.label, t.confidence, t.windows + s.windows)
        lo, hi = min(i, target), max(i, target)
        segs = _merge_equal(segs[:lo] + [merged] + segs[hi + 1:])
    drop = set(background or ())
    return [s for s in segs if s.label not in drop]


@dataclass
class TimelineResult:
    windows: list[Window]
    probabilities: np.ndarray  # [W, C] as scored
    smoothed: np.ndarray       # [W, C] after median_smooth
    labels: list[str]          # predicted label per window (from smoothed)
    confidences: list[float]
    segments: list[Segment]
    duration: float
    scored: int = 0            # windows actually scored (the rest were skipped as static)


def static_windows(windows: Sequence[Window], motion: np.ndarray, fps: float, threshold: float) -> np.ndarray:
    """True for windows whose mean per-frame motion (``VideoReader.motion_curve``) is below ``threshold``.

    SCSampler-style scouting (Korbar et al., ICCV 2019): a window with no motion
    is skipped instead of scored. ``motion[t]`` is the change from frame t-1 to t,
    so a window covering frames [a, b) uses ``motion[a+1:b]``."""
    motion = np.asarray(motion, dtype=np.float64)
    if fps <= 0:
        raise ValueError('fps must be > 0')
    out = np.zeros(len(windows), dtype=bool)
    for k, w in enumerate(windows):
        a = int(round(w.start * fps))
        b = max(a + 2, int(round(w.end * fps)))
        part = motion[min(a + 1, len(motion)):min(b, len(motion))]
        out[k] = part.size > 0 and float(part.mean()) < threshold
    return out


def run_timeline(duration_sec: float, score_windows: Callable[[list[Window]], np.ndarray],
                 class_names: Sequence[str], window_sec: float = 2.0, stride_sec: float = 1.0,
                 smooth_windows: int = 3, min_segment_sec: float = 0.0,
                 background: Optional[Sequence[str]] = None, min_confidence: float = 0.0,
                 uncertain_label: str = 'uncertain', batch_windows: Optional[int] = None,
                 static: Optional[Callable[[list[Window]], np.ndarray]] = None,
                 static_label: str = 'static') -> TimelineResult:
    """Score every window, smooth, segment.

    ``score_windows`` is called on chunks of at most ``batch_windows`` windows
    (None: all at once; the clip pipeline still decodes one window at a time).
    Windows whose top smoothed probability is below ``min_confidence`` are
    labelled ``uncertain_label``. ``static(windows) -> bool mask`` marks windows
    to skip (``--skip_static``): they are not scored, are labelled
    ``static_label`` and do not take part in smoothing. Both labels count as
    background (left out of the segments)."""
    if batch_windows is not None and batch_windows < 1:
        raise ValueError('batch_windows must be >= 1')
    if min_confidence and not 0 <= min_confidence <= 1:
        raise ValueError('min_confidence must be between 0 and 1')
    windows = make_windows(duration_sec, window_sec, stride_sec)
    skip = np.zeros(len(windows), bool) if static is None else np.asarray(static(windows), dtype=bool)
    if skip.shape != (len(windows),):
        raise ValueError('static() must return one flag per window')
    todo = [w for w, s in zip(windows, skip) if not s]
    probs = np.zeros((len(windows), len(class_names)))
    smooth = np.zeros_like(probs)
    if todo:
        step = batch_windows or len(todo)
        parts = []
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            part = np.asarray(score_windows(chunk), dtype=np.float64)
            if part.shape != (len(chunk), len(class_names)):
                raise ValueError(f'score_windows returned {part.shape}, expected {(len(chunk), len(class_names))}')
            parts.append(part)
        scored = np.concatenate(parts)
        probs[~skip] = scored
        smooth[~skip] = median_smooth(scored, smooth_windows)
    labels, conf = [], []
    for k in range(len(windows)):
        if skip[k]:
            labels.append(static_label)
            conf.append(1.0)
            continue
        c = float(smooth[k].max())
        labels.append(uncertain_label if c < min_confidence else str(class_names[int(smooth[k].argmax())]))
        conf.append(c)
    bg = list(background or ()) + ([uncertain_label] if min_confidence > 0 else []) + ([static_label] if skip.any() else [])
    segs = segments_from_timeline(owned_intervals(windows, duration_sec), labels, conf, min_segment_sec, bg)
    return TimelineResult(windows, probs, smooth, labels, conf, segs, float(duration_sec), int((~skip).sum()))


def write_outputs(result: TimelineResult, out_dir, class_names: Sequence[str], video: str = '',
                  settings: Optional[dict] = None) -> dict[str, Path]:
    """``timeline.csv`` (one row per window) and ``segments.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = owned_intervals(result.windows, result.duration)
    tl = out_dir / 'timeline.csv'
    with tl.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['video', 'window', 'start_sec', 'end_sec', 'owned_start_sec', 'owned_end_sec', 'label',
                    'confidence'] + [f'probability_{n}' for n in class_names]
                   + [f'raw_probability_{n}' for n in class_names])
        for win, (a, b), lab, c, sm, raw in zip(result.windows, intervals, result.labels, result.confidences,
                                               result.smoothed, result.probabilities):
            w.writerow([video, win.index, f'{win.start:.3f}', f'{win.end:.3f}', f'{a:.3f}', f'{b:.3f}', lab,
                        f'{c:.4f}'] + [f'{v:.4f}' for v in sm] + [f'{v:.4f}' for v in raw])
    seg = out_dir / 'segments.json'
    seg.write_text(json.dumps({'video': video, 'duration_sec': round(result.duration, 3),
                               'classes': list(class_names), 'settings': settings or {},
                               'segments': [s.to_dict() for s in result.segments]}, indent=2), encoding='utf-8')
    return {'timeline': tl, 'segments': seg}


# ---------------------------------------------------------------------------
# Many videos (infer.py --timeline)
# ---------------------------------------------------------------------------

def video_info(source, backend: str = 'auto') -> tuple[float, float]:
    """(duration in seconds, fps) from the container, without decoding frames."""
    from datasets.video_io import VideoReader
    with VideoReader(source, backend=backend) as reader:
        meta = reader.metadata()
    duration = float(meta.duration) if meta.duration and meta.duration > 0 else (
        meta.frame_count / meta.fps if meta.fps else 0.0)
    return duration, float(meta.fps)


def video_motion(source, method: str = 'auto', backend: str = 'auto') -> np.ndarray:
    from datasets.video_io import VideoReader
    with VideoReader(source, backend=backend) as reader:
        return reader.motion_curve(method)


def _unique_ids(videos: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    ids = []
    for v in videos:
        stem = Path(str(v)).stem or 'video'
        n = seen.get(stem, 0)
        seen[stem] = n + 1
        ids.append(stem if n == 0 else f'{stem}_{n}')
    return ids


SEGMENT_TABLE_COLUMNS = ('video-id', 't-start', 't-end', 'label', 'score')


def run_video_timelines(videos: Sequence[str], out_dir, class_names: Sequence[str],
                        score_rows: Callable, settings: dict, backend: str = 'auto',
                        ground_truth=None, log=print) -> dict:
    """Timeline for every video; returns the run record (also written to ``timeline_run.json``).

    ``score_rows(df)`` gets a DataFrame with columns ``videos, start_sec, end_sec``
    (one row per window) and returns probabilities ``[rows, classes]``.
    ``settings``: window_sec, stride_sec, smooth_windows, min_segment_sec,
    background (list), min_confidence, skip_static (threshold or None),
    motion_source. Writes ``<out>/timeline/<video id>/{timeline.csv,segments.json}``,
    ``<out>/segments.csv`` (``video-id,t-start,t-end,label,score``: the
    localisation report's prediction format) and, with ``ground_truth`` (a CSV
    of ``video-id,t-start,t-end,label``), ``<out>/timeline_metrics.json``
    (detection mAP at tIoU 0.1-0.9 and boundary error at tIoU 0.5).
    """
    import time

    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = _unique_ids(videos)
    rows, record = [], {'settings': dict(settings), 'classes': list(class_names), 'videos': []}
    threshold = settings.get('skip_static')
    for video, vid in zip(videos, ids):
        t0 = time.perf_counter()
        duration, fps = video_info(video, backend)

        def score(windows, video=video):
            frame = pd.DataFrame({'videos': [video] * len(windows), 'start_sec': [w.start for w in windows],
                                  'end_sec': [w.end for w in windows]})
            return score_rows(frame)

        static = None
        if threshold is not None:
            curve = video_motion(video, settings.get('motion_source', 'auto'), backend)
            static = lambda ws, curve=curve, fps=fps: static_windows(ws, curve, fps, float(threshold))
        result = run_timeline(duration, score, class_names, settings['window_sec'], settings['stride_sec'],
                              settings.get('smooth_windows', 3), settings.get('min_segment_sec', 0.0),
                              settings.get('background') or None, settings.get('min_confidence', 0.0),
                              static=static)
        paths = write_outputs(result, out_dir / 'timeline' / vid, class_names, str(video), settings)
        seconds = time.perf_counter() - t0
        record['videos'].append({'video': str(video), 'video_id': vid, 'duration_sec': round(duration, 3),
                                 'windows': len(result.windows), 'scored_windows': result.scored,
                                 'segments': len(result.segments), 'seconds': round(seconds, 3),
                                 'folder': str(paths['segments'].parent)})
        for s in result.segments:
            rows.append({'video-id': vid, 't-start': round(s.start, 3), 't-end': round(s.end, 3),
                         'label': s.label, 'score': round(float(s.confidence), 4)})
        log(f'Timeline {vid}: {len(result.windows)} windows ({result.scored} scored), '
            f'{len(result.segments)} segment(s), {seconds:.1f} s')
    table = pd.DataFrame(rows, columns=list(SEGMENT_TABLE_COLUMNS))
    table.to_csv(out_dir / 'segments.csv', index=False)
    record['total_windows'] = sum(v['windows'] for v in record['videos'])
    record['scored_windows'] = sum(v['scored_windows'] for v in record['videos'])
    record['seconds'] = round(sum(v['seconds'] for v in record['videos']), 3)
    if ground_truth is not None:
        record['metrics'] = score_segments(ground_truth, table, ids)
        (out_dir / 'timeline_metrics.json').write_text(json.dumps(record['metrics'], indent=2), encoding='utf-8')
    (out_dir / 'timeline_run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def score_segments(ground_truth, prediction, video_ids: Optional[Sequence[str]] = None) -> dict:
    """Detection mAP (tIoU 0.1-0.9) and boundary error (tIoU 0.5) of a segment table.

    Ground truth is restricted to ``video_ids`` (the videos actually analysed)."""
    import pandas as pd

    from evaluators.video.localization_report import boundary_error, detection_map
    gt = ground_truth if isinstance(ground_truth, pd.DataFrame) else pd.read_csv(ground_truth)
    if video_ids is not None:
        gt = gt[gt['video-id'].astype(str).isin(set(map(str, video_ids)))]
    return {'detection_map': detection_map(gt, prediction), 'boundary_error': boundary_error(gt, prediction, 0.5),
            'n_videos': int(gt['video-id'].nunique())}

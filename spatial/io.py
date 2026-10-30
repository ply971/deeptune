"""Frame reading and the output files of zero-shot tracking.

Frame reading uses OpenCV directly. When A2's ``datasets.video_io.VideoReader``
lands, :func:`iter_frames` and :func:`video_info` are the only two functions to
switch over.

Output formats (one folder per video)
-------------------------------------
``detections.json``
    ``{"video", "fps", "width", "height", "num_frames", "every_n_frames",
    "prompts", "detector": {...}, "frames": [{"frame", "time_s",
    "detections": [{"box_xyxy": [x1, y1, x2, y2], "score", "label"}]}]}``.
    ``frame`` is the 0-based index in the video; only sampled frames appear.
``tracks.csv``
    MOTChallenge text, no header, one row per (frame, track):
    ``frame,id,x,y,w,h,score,class,-1,-1`` with 1-based ``frame`` (= video frame
    index + 1), pixel ``x, y`` = top-left corner, and 1-based ``class`` = position
    of the label in the prompt list. Caveat for S4: TrackEval's MOTChallenge loader
    is believed to keep only class 1 (pedestrian) rows, so rows of prompts 2..n may
    be dropped silently there (unverified); evaluate per prompt or remap the class
    column to 1.
``track_labels.csv``
    ``track_id,class_id,label`` (header), one row per track: the label column that
    MOT files lack.
``summary.json``
    per label: ``count`` (number of unique tracks), ``first_seen_s``,
    ``last_seen_s`` and per-track ``first_frame``, ``last_frame``,
    ``first_seen_s``, ``last_seen_s``, ``duration_s`` = (last - first + 1) / fps,
    ``num_observations`` (sampled frames where the track was output).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg', '.wmv')


@dataclass
class VideoInfo:
    path: str
    fps: float
    num_frames: int  # container estimate; the pipeline reports the frames actually read
    width: int
    height: int


def _open(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot open video: {path}')
    return cap


def video_info(path, default_fps: float = 30.0) -> VideoInfo:
    import cv2
    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = default_fps
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return VideoInfo(str(path), fps, max(n, 0), w, h)


def iter_frames(path, every_n: int = 1, max_frames: int | None = None
                ) -> Iterator[tuple[int, np.ndarray, bool]]:
    """Yield ``(frame_index, rgb_uint8 [H, W, 3], sampled)`` for every decoded frame.

    ``sampled`` is True on frames ``0, every_n, 2*every_n, ...``. All frames are
    yielded so callers can draw overlays on unsampled frames; ``max_frames`` caps
    the number of decoded frames.
    """
    import cv2
    if every_n < 1:
        raise ValueError('every_n must be >= 1')
    cap = _open(path)
    try:
        i = 0
        while max_frames is None or i < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), i % every_n == 0
            i += 1
    finally:
        cap.release()


def list_videos(directory) -> list[Path]:
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f'not a directory: {directory}')
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


# --------------------------------------------------------------------------- writers

def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    return path


def write_mot(path, rows: Sequence[dict], prompts: Sequence[str]) -> Path:
    """Write MOTChallenge rows; ``rows`` items have frame (0-based), track_id, box, score, label."""
    class_of = {p: i + 1 for i, p in enumerate(prompts)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        for r in sorted(rows, key=lambda r: (r['frame'], r['track_id'])):
            x1, y1, x2, y2 = r['box']
            w.writerow([r['frame'] + 1, r['track_id'], f'{x1:.2f}', f'{y1:.2f}', f'{x2 - x1:.2f}',
                        f'{y2 - y1:.2f}', f'{r["score"]:.4f}', class_of.get(r['label'], -1), -1, -1])
    return path


def read_mot(path) -> list[dict]:
    """Read a MOTChallenge file back into rows with 0-based ``frame`` and xyxy ``box``."""
    rows = []
    with Path(path).open(newline='', encoding='utf-8') as fh:
        for rec in csv.reader(fh):
            if not rec:
                continue
            f, tid, x, y, w, h, s, c = rec[:8]
            x, y, w, h = float(x), float(y), float(w), float(h)
            rows.append({'frame': int(float(f)) - 1, 'track_id': int(float(tid)),
                         'box': (x, y, x + w, y + h), 'score': float(s), 'class_id': int(float(c))})
    return rows


def write_track_labels(path, rows: Sequence[dict], prompts: Sequence[str]) -> Path:
    """One row per track with its final label (the last label emitted for it)."""
    class_of = {p: i + 1 for i, p in enumerate(prompts)}
    last: dict[int, str] = {}
    for r in sorted(rows, key=lambda r: r['frame']):
        last[r['track_id']] = r['label']
    path = Path(path)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['track_id', 'class_id', 'label'])
        for tid in sorted(last):
            w.writerow([tid, class_of.get(last[tid], -1), last[tid]])
    return path


def build_summary(rows: Sequence[dict], prompts: Sequence[str], fps: float) -> dict:
    """Object counts and per-track timelines from track rows (see module docstring)."""
    last_label: dict[int, str] = {}
    frames: dict[int, list[int]] = {}
    for r in sorted(rows, key=lambda r: r['frame']):
        last_label[r['track_id']] = r['label']
        frames.setdefault(r['track_id'], []).append(int(r['frame']))
    labels: dict[str, dict] = {p: {'count': 0, 'first_seen_s': None, 'last_seen_s': None, 'tracks': []}
                               for p in prompts}
    for tid in sorted(frames):
        f = frames[tid]
        lab = last_label[tid]
        entry = labels.setdefault(lab, {'count': 0, 'first_seen_s': None, 'last_seen_s': None, 'tracks': []})
        first, last = min(f), max(f)
        entry['tracks'].append({
            'track_id': tid, 'first_frame': first, 'last_frame': last,
            'first_seen_s': round(first / fps, 4), 'last_seen_s': round(last / fps, 4),
            'duration_s': round((last - first + 1) / fps, 4), 'num_observations': len(set(f)),
        })
    for entry in labels.values():
        tr = entry['tracks']
        entry['count'] = len(tr)
        if tr:
            entry['first_seen_s'] = min(t['first_seen_s'] for t in tr)
            entry['last_seen_s'] = max(t['last_seen_s'] for t in tr)
    return {'fps': fps, 'total_tracks': len(frames), 'labels': labels}

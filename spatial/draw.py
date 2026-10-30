"""Annotated MP4 overlay: track boxes with ``label #id`` captions."""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Sequence

import numpy as np


def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per track id (golden-ratio hue walk)."""
    h = (int(track_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame_rgb: np.ndarray, tracks: Sequence[dict]) -> np.ndarray:
    """Return a BGR copy of the frame with each track (``track_id, box, label``) drawn."""
    import cv2
    img = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = max(0.3, min(w, h) / 640)
    thick = max(1, int(round(2 * scale)))
    for t in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in t['box']]
        col = id_color(t['track_id'])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        text = f"{t['label']} #{t['track_id']}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        ty = max(th + base, y1)
        cv2.rectangle(img, (x1, ty - th - base), (x1 + tw, ty), col, -1)
        cv2.putText(img, text, (x1, ty - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return img


def write_overlay(video_path, out_path, rows: Sequence[dict], fps: float, max_frames: int | None = None,
                  hold_frames: int = 1) -> Path:
    """Re-read the video and draw tracks.

    Tracks exist only on sampled frames; each frame shows the tracks of the most
    recent sampled frame within ``hold_frames`` frames (``every_n_frames`` in the
    pipeline), so the overlay does not flicker when ``every_n_frames > 1``.
    """
    import cv2
    from spatial.io import iter_frames
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r['frame']), []).append(r)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    current: list[dict] = []
    current_at = -10 ** 9
    try:
        for idx, frame, _ in iter_frames(video_path, 1, max_frames):
            if idx in by_frame:
                current, current_at = by_frame[idx], idx
            shown = current if idx - current_at < hold_frames else []
            img = draw_tracks(frame, shown)
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f'cannot write video: {out_path}')
            writer.write(img)
    finally:
        if writer is not None:
            writer.release()
    return out_path

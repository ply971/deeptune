"""Synthetic moving-square scenes for the spatial (detection/tracking) tests."""
from __future__ import annotations

import numpy as np
import pytest

from spatial.detect import Detection


class Scene:
    """Squares moving on straight lines: ``objects`` = list of dicts with
    ``label``, ``start`` (x, y top-left at frame 0), ``vel`` (dx, dy per frame),
    ``size`` and ``color`` (BGR for OpenCV)."""

    def __init__(self, objects, n_frames=30, width=64, height=64):
        self.objects = objects
        self.n_frames = n_frames
        self.width = width
        self.height = height

    def box(self, k: int, f: int):
        o = self.objects[k]
        x = o['start'][0] + o['vel'][0] * f
        y = o['start'][1] + o['vel'][1] * f
        return (float(x), float(y), float(x + o['size']), float(y + o['size']))

    def truth(self):
        """frame -> [(object index, box)]"""
        return {f: [(k, self.box(k, f)) for k in range(len(self.objects))] for f in range(self.n_frames)}

    def detections(self, jitter=1.0, drop=None, low=None, score=0.9, low_score=0.2, seed=0):
        """Ground truth turned into noisy detections.

        ``drop``: {object index: set(frames)} with no detection; ``low``: {object
        index: set(frames)} where the score is ``low_score`` instead of ``score``.
        """
        rng = np.random.default_rng(seed)
        drop, low = drop or {}, low or {}
        table = {}
        for f in range(self.n_frames):
            dets = []
            for k, o in enumerate(self.objects):
                if f in drop.get(k, ()):
                    continue
                b = np.array(self.box(k, f)) + rng.uniform(-jitter, jitter, 4)
                s = low_score if f in low.get(k, ()) else score + rng.uniform(-0.05, 0.05)
                dets.append(Detection(tuple(b.tolist()), float(s), o['label']))
            table[f] = dets
        return table

    def write(self, path, fps=10.0):
        import cv2
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.width, self.height))
        assert w.isOpened()
        for f in range(self.n_frames):
            img = np.zeros((self.height, self.width, 3), np.uint8)
            for k, o in enumerate(self.objects):
                x1, y1, x2, y2 = [int(round(v)) for v in self.box(k, f)]
                cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), o['color'], -1)
            w.write(img)
        w.release()
        return path


def assign_ids(scene: Scene, outputs_per_frame, iou_min=0.5):
    """Map each ground-truth object to the track IDs that covered it.

    ``outputs_per_frame``: frame -> list of objects with track_id and box.
    Returns {object index: [track id per frame or None]}.
    """
    from spatial.track import iou_matrix
    seq = {k: [] for k in range(len(scene.objects))}
    for f in range(scene.n_frames):
        outs = outputs_per_frame.get(f, [])
        for k in range(len(scene.objects)):
            gt = scene.box(k, f)
            best, best_iou = None, iou_min
            for o in outs:
                v = iou_matrix([gt], [o.box])[0, 0]
                if v >= best_iou:
                    best, best_iou = o.track_id, v
            seq[k].append(best)
    return seq


@pytest.fixture
def two_squares():
    """Two lobsters moving right/down at different speeds, never overlapping."""
    return Scene([
        {'label': 'lobster', 'start': (2, 4), 'vel': (1.5, 0.0), 'size': 10, 'color': (0, 0, 255)},
        {'label': 'lobster', 'start': (4, 40), 'vel': (1.0, 0.3), 'size': 12, 'color': (0, 255, 0)},
    ])


@pytest.fixture
def make_scene():
    return Scene


@pytest.fixture
def track_ids_of():
    return assign_ids

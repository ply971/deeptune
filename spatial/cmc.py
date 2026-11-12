"""Camera-motion compensation (CMC) for tracking-by-detection.

A hand-held or panning camera moves every box between frames, even boxes of
objects that stand still. A constant-velocity Kalman filter then predicts the
box where it *was*, IoU with the new detection drops, and the track is lost or
swaps ID. BoT-SORT (Aharon, Orfaig, Bobrovsky, 2022, arXiv:2206.14651; reference
code MIT, github.com/NirAharon/BoT-SORT) fixes this by estimating the global
2-D motion of the background between consecutive frames and warping every
predicted track state with it before association.

This is a re-implementation of BoT-SORT's ``sparseOptFlow`` global motion
compensation, with one deliberate change: feature points inside the current
detection boxes are masked out, so the estimate follows the background and not
the tracked objects themselves (BoT-SORT's ORB variant masks a fixed central
region for the same reason).

Steps per frame (all OpenCV, no torch):

1. Grey image, downscaled so its longer side is at most ``max_side`` pixels.
2. ``goodFeaturesToTrack`` (Shi-Tomasi corners) on the previous frame.
3. ``calcOpticalFlowPyrLK`` to find them in the current frame.
4. ``estimateAffinePartial2D`` with RANSAC (rotation, uniform scale,
   translation) maps previous -> current; translation is rescaled to full
   resolution.

It falls back to the identity when there is no previous frame, too few points
or no RANSAC solution, so a textureless or synthetic video simply behaves like
plain ByteTrack. The frame size may not change between calls (a new size resets).
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class SparseFlowCMC:
    """Estimate the 2x3 affine that maps the previous frame onto the current one.

    Call ``cmc(frame_rgb, boxes)`` once per tracker step, in order. ``boxes`` are
    the current detections (xyxy, original pixels); their inside is ignored when
    picking feature points. ``uses_boxes`` tells :class:`spatial.track.ByteTracker`
    to pass them.
    """

    uses_boxes = True

    def __init__(self, max_side: int = 640, max_corners: int = 1000, quality: float = 0.01,
                 min_distance: int = 1, min_points: int = 10, box_margin: float = 0.05,
                 max_box_fraction: float = 0.8):
        if max_side < 32:
            raise ValueError('max_side must be >= 32')
        self.max_side = int(max_side)
        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.min_points = int(min_points)
        self.box_margin = float(box_margin)
        # If the boxes cover more of the frame than this, masking them would leave too
        # little background, so the whole frame is used instead.
        self.max_box_fraction = float(max_box_fraction)
        self._prev = None
        self._prev_points = None
        self._shape = None
        self.last_inliers = 0
        self.history: deque = deque(maxlen=256)  # recent affines (debugging); bounded for long videos

    def reset(self) -> None:
        self._prev = self._prev_points = self._shape = None

    def _grey(self, frame):
        import cv2
        frame = np.asarray(frame)
        if frame.ndim == 3:
            grey = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
        else:
            grey = frame.astype(np.uint8)
        h, w = grey.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            grey = cv2.resize(grey, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                              interpolation=cv2.INTER_AREA)
        return grey, scale

    def _mask(self, shape, boxes, scale):
        mask = np.full(shape, 255, np.uint8)
        if not boxes:
            return mask
        h, w = shape
        covered = np.zeros(shape, bool)
        for box in boxes:
            x1, y1, x2, y2 = [float(v) * scale for v in box]
            mx, my = (x2 - x1) * self.box_margin, (y2 - y1) * self.box_margin
            c1, r1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            c2, r2 = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
            if c2 > c1 and r2 > r1:
                covered[r1:r2, c1:c2] = True
        if covered.mean() > self.max_box_fraction:
            return mask
        mask[covered] = 0
        return mask

    def __call__(self, frame, boxes: Sequence | None = None) -> np.ndarray:
        import cv2
        grey, scale = self._grey(frame)
        mask = self._mask(grey.shape, list(boxes or []), scale)
        points = cv2.goodFeaturesToTrack(grey, maxCorners=self.max_corners, qualityLevel=self.quality,
                                         minDistance=self.min_distance, blockSize=3, mask=mask)
        affine = IDENTITY.copy()
        self.last_inliers = 0
        if self._prev is not None and self._shape == grey.shape and self._prev_points is not None \
                and len(self._prev_points) >= self.min_points:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev, grey, self._prev_points, None)
            if nxt is not None:
                ok = status.reshape(-1).astype(bool)
                src, dst = self._prev_points.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
                if len(src) >= self.min_points:
                    est, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                               ransacReprojThreshold=3.0)
                    if est is not None and np.isfinite(est).all():
                        affine = est.astype(np.float64)
                        affine[:, 2] /= scale  # translation back to full-resolution pixels
                        self.last_inliers = int(inliers.sum()) if inliers is not None else 0
        self._prev, self._prev_points, self._shape = grey, points, grey.shape
        self.history.append(affine)
        return affine


def compose(affines: Sequence[np.ndarray]) -> np.ndarray:
    """Chain 2x3 affines applied in order (first, then second, ...)."""
    total = np.eye(3)
    for a in affines:
        m = np.eye(3)
        m[:2] = np.asarray(a, dtype=np.float64).reshape(2, 3)
        total = m @ total
    return total[:2]

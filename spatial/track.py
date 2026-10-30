"""ByteTrack: multi-object tracking by associating every detection box.

Re-implementation (not vendored) of

    Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan,
    Ping Luo, Wenyu Liu, Xinggang Wang. "ByteTrack: Multi-Object Tracking by
    Associating Every Detection Box." ECCV 2022. (Reference code: MIT licence,
    github.com/ifzhang/ByteTrack.)

The algorithm per frame:

1. Split detections into **high** (``score >= high_thresh``) and **low**
   (``low_thresh <= score < high_thresh``).
2. Predict every tracked and lost track with a constant-velocity Kalman filter on
   ``(cx, cy, aspect, h)``.
3. First association: high detections vs tracked+lost tracks, cost = 1 - IoU
   (optionally fused with the detection score), Hungarian matching
   (``scipy.optimize.linear_sum_assignment``), gated at ``match_thresh``.
4. Second association: low detections vs the still-unmatched *tracked* tracks,
   gated at IoU distance 0.5. This is what recovers occluded/blurred objects whose
   detector score dropped. Tracks still unmatched become **lost**.
5. Unconfirmed tracks (born last frame) are matched to the remaining high
   detections (gate 0.7); unmatched unconfirmed tracks are removed.
6. Remaining high detections with ``score >= new_track_thresh`` start new tracks
   (confirmed on their second hit, or immediately on the first frame).
7. Lost tracks older than ``max_time_lost`` steps are removed; duplicate
   tracked/lost pairs (IoU > 0.85) keep the older one.

Differences from the reference code (all deliberate):

- IDs are per tracker instance (start at 1 for every video) instead of a
  process-wide counter.
- Detections carry a text label. By default association is **per label** (a
  "fish" detection can never continue a "lobster" track). With
  ``class_agnostic=True`` labels are ignored for matching and each track's label
  is the majority vote of its matched detections.
- Gating uses Hungarian matching on a cost matrix where pairs above the gate are
  made infeasible, then any such pair is dropped; the reference uses ``lap.lapjv``
  with ``cost_limit``. Both return only pairs under the gate.

Extension points (documented for S3 follow-ups):

- **BoT-SORT camera-motion compensation**: pass ``motion_compensator`` (a callable
  ``frame -> 2x3 affine`` mapping the previous frame onto the current one) and
  give :meth:`ByteTracker.update` the frame. The affine is applied to the
  predicted Kalman states (centre and velocity) before association
  (:func:`apply_affine_to_tracks`). A real ORB/ECC estimator is not built yet.
- **Re-ID / appearance**: :meth:`ByteTracker._distance` is the single place that
  builds the association cost; BoT-SORT adds an embedding distance there.
- **SAM 2 mask tracking**: every :class:`TrackOutput` carries ``mask=None``; a mask
  propagator can take each track's first box as a prompt (see tracks.csv) and fill
  masks per frame without changing the tracker.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- boxes

def xyxy_to_xyah(box) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(xyah) -> np.ndarray:
    cx, cy, a, h = [float(v) for v in xyah[:4]]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` xyxy boxes -> ``[N, M]``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def linear_assignment(cost: np.ndarray, thresh: float):
    """Minimum-cost matching keeping only pairs with ``cost <= thresh``.

    Returns ``(matches [K, 2], unmatched_rows, unmatched_cols)``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), list(range(n)), list(range(m))
    big = 1e6
    gated = np.where(cost > thresh, big, cost)
    rows, cols = linear_sum_assignment(gated)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return (np.array(matches, dtype=int).reshape(-1, 2),
            [r for r in range(n) if r not in mr], [c for c in range(m) if c not in mc])


# --------------------------------------------------------------------------- Kalman

class KalmanFilter:
    """Constant-velocity Kalman filter on ``(cx, cy, a, h)`` + velocities.

    Same noise model as ByteTrack / DeepSORT: process and measurement noise scale
    with the box height (``std_weight_position = 1/20``,
    ``std_weight_velocity = 1/160``); the aspect ratio gets small fixed noise.
    """

    ndim = 4

    def __init__(self, dt: float = 1.0):
        self._motion_mat = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64)
        mean = np.r_[measurement, np.zeros(self.ndim)]
        h = measurement[3]
        wp, wv = self._std_weight_position, self._std_weight_velocity
        std = [2 * wp * h, 2 * wp * h, 1e-2, 2 * wp * h,
               10 * wv * h, 10 * wv * h, 1e-5, 10 * wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        h = mean[3]
        wp, wv = self._std_weight_position, self._std_weight_velocity
        std = [wp * h, wp * h, 1e-2, wp * h, wv * h, wv * h, 1e-5, wv * h]
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        h = mean[3]
        wp = self._std_weight_position
        innovation_cov = np.diag(np.square([wp * h, wp * h, 1e-1, wp * h]))
        return self._update_mat @ mean, self._update_mat @ covariance @ self._update_mat.T + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol, lower), (covariance @ self._update_mat.T).T,
                                             check_finite=False).T
        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov


# --------------------------------------------------------------------------- tracks

NEW, TRACKED, LOST, REMOVED = 'new', 'tracked', 'lost', 'removed'


class Track:
    """One tracklet (ByteTrack's ``STrack``)."""

    def __init__(self, box_xyxy, score: float, label: str):
        self._det_box = np.asarray(box_xyxy, dtype=np.float64)
        self.score = float(score)
        self.label = label
        self.label_votes: Counter = Counter({label: 1})
        self.mean = None
        self.covariance = None
        self.track_id = 0
        self.state = NEW
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.kalman: KalmanFilter | None = None

    @property
    def box(self) -> np.ndarray:
        """Current xyxy box (Kalman state when initialised, else the detection)."""
        if self.mean is None:
            return self._det_box.copy()
        return xyah_to_xyxy(self.mean[:4])

    @property
    def end_frame(self) -> int:
        return self.frame_id

    def predict(self):
        mean = self.mean.copy()
        if self.state != TRACKED:
            mean[7] = 0  # freeze height velocity of lost tracks (reference behaviour)
        self.mean, self.covariance = self.kalman.predict(mean, self.covariance)

    def activate(self, kalman: KalmanFilter, frame_id: int, track_id: int):
        self.kalman = kalman
        self.track_id = track_id
        self.mean, self.covariance = kalman.initiate(xyxy_to_xyah(self._det_box))
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def _absorb(self, det: 'Track', frame_id: int, class_agnostic: bool):
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, xyxy_to_xyah(det._det_box))
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        self.label_votes[det.label] += 1
        if class_agnostic:
            self.label = self.label_votes.most_common(1)[0][0]

    def update(self, det: 'Track', frame_id: int, class_agnostic: bool = False):
        self.tracklet_len += 1
        self._absorb(det, frame_id, class_agnostic)

    def re_activate(self, det: 'Track', frame_id: int, class_agnostic: bool = False):
        self.tracklet_len = 0
        self._absorb(det, frame_id, class_agnostic)

    def __repr__(self):
        return f'Track(id={self.track_id}, {self.label}, {self.state}, box={np.round(self.box, 1).tolist()})'


@dataclass
class TrackOutput:
    """A confirmed track's state at one tracker step."""

    track_id: int
    box: tuple[float, float, float, float]  # xyxy pixels
    score: float
    label: str
    mask: object = None  # reserved for SAM 2 mask tracking (S2/S3)


def apply_affine_to_tracks(tracks: Sequence[Track], affine) -> None:
    """Warp predicted Kalman states with a 2x3 affine (BoT-SORT-style CMC).

    The centre is mapped with the full affine, the centre velocity with its
    linear part; height (and its velocity) scale with the affine's mean scale.
    Covariances of the centre/velocity blocks are rotated accordingly.
    """
    affine = np.asarray(affine, dtype=np.float64).reshape(2, 3)
    lin, t = affine[:, :2], affine[:, 2]
    scale = float(np.sqrt(abs(np.linalg.det(lin)))) or 1.0
    big = np.eye(8)
    big[0:2, 0:2] = lin
    big[4:6, 4:6] = lin
    big[3, 3] = big[7, 7] = scale
    for tr in tracks:
        if tr.mean is None:
            continue
        mean = big @ tr.mean
        mean[0:2] += t
        tr.mean = mean
        tr.covariance = big @ tr.covariance @ big.T


class ByteTracker:
    """ByteTrack over a stream of per-frame detections.

    Parameters
    ----------
    high_thresh: detections at or above it take part in the first association.
    low_thresh: detections in ``[low_thresh, high_thresh)`` take part in the second.
    new_track_thresh: minimum score to start a track (reference: high + 0.1).
    match_thresh: IoU-distance gate of the first association (reference 0.8).
    track_buffer: how many steps a lost track may be re-found (reference 30 at 30 fps).
    frame_rate: steps per second seen by the tracker; ``max_time_lost =
        round(frame_rate / 30 * track_buffer)`` as in the reference.
    fuse_score: multiply IoU by the detection score in the first association
        (reference behaviour for MOT17; off for MOT20).
    class_agnostic: ignore labels when matching (label = majority vote).
    motion_compensator: optional ``frame -> 2x3 affine`` (see module docstring).
    """

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float | None = None, match_thresh: float = 0.8,
                 track_buffer: int = 30, frame_rate: float = 30.0, fuse_score: bool = True,
                 class_agnostic: bool = False,
                 motion_compensator: Callable[[np.ndarray], np.ndarray] | None = None):
        if not 0 <= low_thresh <= high_thresh:
            raise ValueError('need 0 <= low_thresh <= high_thresh')
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(high_thresh + 0.1 if new_track_thresh is None else new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.max_time_lost = max(1, int(round(frame_rate / 30.0 * track_buffer)))
        self.fuse_score = fuse_score
        self.class_agnostic = class_agnostic
        self.motion_compensator = motion_compensator
        self.kalman = KalmanFilter()
        self.frame_id = 0
        self._next_id = 0
        self.tracked: list[Track] = []
        self.lost: list[Track] = []
        self.removed: list[Track] = []

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # The single place that builds association costs (BoT-SORT adds Re-ID here).
    def _distance(self, tracks: Sequence[Track], dets: Sequence[Track], fuse: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou = iou_matrix([t.box for t in tracks], [d.box for d in dets])
        if fuse:
            iou = iou * np.array([d.score for d in dets])[None, :]
        cost = 1.0 - iou
        if not self.class_agnostic:
            same = np.array([[t.label == d.label for d in dets] for t in tracks])
            cost = np.where(same, cost, np.inf)
        return cost

    def update(self, detections: Sequence, frame: np.ndarray | None = None) -> list[TrackOutput]:
        """Advance one step. ``detections``: objects with ``box`` (xyxy), ``score``, ``label``."""
        self.frame_id += 1
        fid = self.frame_id
        activated, refind, lost, removed = [], [], [], []

        dets = [Track(d.box, d.score, d.label) for d in detections]
        high = [d for d in dets if d.score >= self.high_thresh]
        low = [d for d in dets if self.low_thresh <= d.score < self.high_thresh]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # Step 2: first association, high-score detections.
        pool = _joint(tracked, self.lost)
        for t in pool:
            t.predict()
        if self.motion_compensator is not None and frame is not None:
            affine = self.motion_compensator(frame)
            apply_affine_to_tracks(pool, affine)
            apply_affine_to_tracks(unconfirmed, affine)
        cost = self._distance(pool, high, self.fuse_score)
        matches, u_track, u_det = linear_assignment(cost, self.match_thresh)
        for it, idet in matches:
            t, d = pool[it], high[idet]
            if t.state == TRACKED:
                t.update(d, fid, self.class_agnostic)
                activated.append(t)
            else:
                t.re_activate(d, fid, self.class_agnostic)
                refind.append(t)

        # Step 3: second association, low-score detections vs remaining tracked tracks.
        r_tracked = [pool[i] for i in u_track if pool[i].state == TRACKED]
        cost = self._distance(r_tracked, low, False)
        matches, u_track2, _ = linear_assignment(cost, 0.5)
        for it, idet in matches:
            t, d = r_tracked[it], low[idet]
            t.update(d, fid, self.class_agnostic)
            activated.append(t)
        for it in u_track2:
            t = r_tracked[it]
            if t.state != LOST:
                t.state = LOST
                lost.append(t)

        # Unconfirmed tracks (seen once) vs remaining high detections.
        rest = [high[i] for i in u_det]
        cost = self._distance(unconfirmed, rest, self.fuse_score)
        matches, u_unconf, u_det = linear_assignment(cost, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(rest[idet], fid, self.class_agnostic)
            activated.append(unconfirmed[it])
        for it in u_unconf:
            unconfirmed[it].state = REMOVED
            removed.append(unconfirmed[it])

        # Step 4: start new tracks.
        for idet in u_det:
            d = rest[idet]
            if d.score < self.new_track_thresh:
                continue
            d.activate(self.kalman, fid, self._new_id())
            activated.append(d)

        # Step 5: age out lost tracks.
        for t in self.lost:
            if fid - t.end_frame > self.max_time_lost:
                t.state = REMOVED
                removed.append(t)

        self.tracked = [t for t in self.tracked if t.state == TRACKED]
        self.tracked = _joint(self.tracked, activated)
        self.tracked = _joint(self.tracked, refind)
        self.lost = _sub(self.lost, self.tracked)
        self.lost.extend(lost)
        self.lost = _sub(self.lost, removed)
        self.removed.extend(removed)
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost)

        return [TrackOutput(t.track_id, tuple(float(v) for v in t.box), t.score, t.label)
                for t in self.tracked if t.is_activated]


def _joint(a: list[Track], b: list[Track]) -> list[Track]:
    seen = {t.track_id for t in a}
    return list(a) + [t for t in b if t.track_id not in seen and not seen.add(t.track_id)]


def _sub(a: list[Track], b: list[Track]) -> list[Track]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(tracked: list[Track], lost: list[Track]):
    if not tracked or not lost:
        return tracked, lost
    dist = 1.0 - iou_matrix([t.box for t in tracked], [t.box for t in lost])
    dup_t, dup_l = set(), set()
    for p, q in zip(*np.where(dist < 0.15)):
        if tracked[p].label != lost[q].label:
            continue
        if tracked[p].frame_id - tracked[p].start_frame > lost[q].frame_id - lost[q].start_frame:
            dup_l.add(q)
        else:
            dup_t.add(p)
    return ([t for i, t in enumerate(tracked) if i not in dup_t],
            [t for i, t in enumerate(lost) if i not in dup_l])

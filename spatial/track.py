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
- Matching uses ``scipy.optimize.linear_sum_assignment`` on the extended
  ``(n + m) x (n + m)`` matrix that ``lap.lapjv(extend_cost=True, cost_limit=t)``
  builds internally (each row/column may stay unmatched at ``t / 2``), so the
  matching rule is the reference's and ``lap`` is not needed. Ties (equal total
  cost) may break differently from lapjv.
- Score splits are inclusive: high is ``score >= high_thresh`` and low is
  ``low_thresh <= score < high_thresh``. The reference uses ``score > track_thresh``
  for high and ``0.1 < score < track_thresh`` for low, which drops a box scoring
  exactly ``track_thresh`` from both stages.
- ``max_time_lost = max(1, int(frame_rate / 30 * track_buffer))``, as in the reference
  except for the floor of 1 (so a stride never disables re-finding).
- Lost tracks removed for age this step leave the lost list immediately. The
  reference subtracts the *previous* removed list, so such a track stays in
  ``lost_stracks`` (and the next step's matching pool) for one more step and can
  still be re-activated: effectively a buffer one step longer than configured.
  Treated as a bug fix.
- Duplicate removal (tracked vs lost, IoU > 0.85) only compares tracks of the same
  label, unless ``class_agnostic``.

BoT-SORT additions (Aharon, Orfaig, Bobrovsky, "BoT-SORT: Robust Associations
Multi-Pedestrian Tracking", arXiv:2206.14651, 2022; reference code MIT,
github.com/NirAharon/BoT-SORT), re-implemented:

- **Camera-motion compensation**: pass ``motion_compensator`` (a callable
  ``frame -> 2x3 affine`` mapping the previous frame onto the current one; with
  ``uses_boxes = True`` it is called as ``(frame, detection_boxes)``, see
  :class:`spatial.cmc.SparseFlowCMC`) and give :meth:`ByteTracker.update` the frame.
  The affine warps the predicted Kalman states (centre and velocity) before
  association (:func:`apply_affine_to_tracks`).
- **Appearance (Re-ID)**: give :meth:`ByteTracker.update` one embedding per
  detection (``embeddings``, L2-normalised rows; see :mod:`spatial.reid`). Each
  track keeps an exponential moving average (``feat = 0.9 feat + 0.1 new``). In the
  first association and for unconfirmed tracks the cost is BoT-SORT's
  ``min(iou_dist, emb_dist)``, where ``emb_dist = (1 - cos) / 2`` counts only when it
  is at most ``appearance_thresh`` (0.25) *and* the boxes are close
  (``1 - IoU <= proximity_thresh``, 0.5); otherwise it is 1. The low-score second
  association stays IoU-only, as in BoT-SORT. Without embeddings the tracker is
  exactly ByteTrack.
- Kept from ByteTrack instead of BoT-SORT: the ``(cx, cy, a, h)`` Kalman state
  (BoT-SORT uses ``(cx, cy, w, h)``).

Every :class:`TrackOutput` also reports the step and box where the track was
born (``start_step``, ``birth_box``), so offline users can emit the first
observation that ByteTrack withholds until a track is confirmed. ``mask=None``
is reserved for SAM 2 mask tracking.
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
    """Gated minimum-cost matching with ``lap.lapjv(extend_cost=True, cost_limit=thresh)`` semantics.

    Every row and every column may stay unmatched at a cost of ``thresh / 2``, so a
    pair is matched only if that beats leaving both sides unmatched, and the total
    ``sum(matched costs) + thresh / 2 * (#unmatched rows + #unmatched cols)`` is
    minimised. This is the rule ByteTrack's reference ``matching.linear_assignment``
    uses. Plain Hungarian on a gated matrix would instead maximise the *number* of
    matches (two weak matches beating one strong one), which swaps IDs.

    Implemented as the standard ``(n + m) x (n + m)`` extension that ``lap`` builds
    internally: ``[[C, thresh/2], [thresh/2, 0]]``, solved with
    ``scipy.optimize.linear_sum_assignment``. Pairs above ``thresh`` (and ``inf``)
    are replaced by a large finite cost and can never be returned.

    Returns ``(matches [K, 2], unmatched_rows, unmatched_cols)``.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=int), list(range(n)), list(range(m))
    big = 1e6
    ext = np.full((n + m, n + m), thresh / 2.0)
    ext[n:, m:] = 0.0
    ext[:n, :m] = np.where(np.isfinite(cost) & (cost <= thresh), cost, big)
    rows, cols = linear_sum_assignment(ext)
    matches = [(r, c) for r, c in zip(rows, cols) if r < n and c < m and cost[r, c] <= thresh]
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
        self.birth_score = float(score)
        self.feat_alpha = 0.9
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
        self.feat: np.ndarray | None = None       # smoothed appearance (EMA), L2-normalised
        self.curr_feat: np.ndarray | None = None  # embedding of the last matched detection

    def set_feature(self, feat, alpha: float = 0.9) -> None:
        """Absorb one detection embedding into the EMA appearance (BoT-SORT's update)."""
        if feat is None:
            return
        feat = np.asarray(feat, dtype=np.float64).reshape(-1)
        feat = feat / max(float(np.linalg.norm(feat)), 1e-12)
        self.curr_feat = feat
        if self.feat is None:
            self.feat = feat
        else:
            smooth = alpha * self.feat + (1 - alpha) * feat
            self.feat = smooth / max(float(np.linalg.norm(smooth)), 1e-12)

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
        self.set_feature(det.curr_feat, self.feat_alpha)
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
    start_step: int = 0  # tracker step (1-based) where the track was born
    birth_box: tuple | None = None  # the detection box that started it
    birth_score: float | None = None
    feat: object = None  # embedding of the detection matched at this step (None without Re-ID)


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
        max(1, int(frame_rate / 30 * track_buffer))`` (reference: no floor).
    fuse_score: multiply IoU by the detection score in the first association and
        when confirming a new track (reference behaviour for MOT17; off for MOT20).
        Confirming then needs ``IoU * score > 0.3``, so a box scoring under 0.3 can
        never confirm: ``track.py`` turns it off for zero-shot detectors.
    class_agnostic: ignore labels when matching (label = majority vote).
    motion_compensator: optional ``frame -> 2x3 affine`` (see module docstring).
    appearance_thresh: largest ``(1 - cos) / 2`` appearance distance that may lower a
        cost (BoT-SORT 0.25). Only used when :meth:`update` gets embeddings.
    proximity_thresh: appearance only counts for pairs with ``1 - IoU`` at most this
        (BoT-SORT 0.5).
    feat_alpha: EMA momentum of each track's appearance (BoT-SORT 0.9).
    """

    def __init__(self, high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float | None = None, match_thresh: float = 0.8,
                 track_buffer: int = 30, frame_rate: float = 30.0, fuse_score: bool = True,
                 class_agnostic: bool = False,
                 motion_compensator: Callable[[np.ndarray], np.ndarray] | None = None,
                 appearance_thresh: float = 0.25, proximity_thresh: float = 0.5, feat_alpha: float = 0.9):
        if not 0 <= low_thresh <= high_thresh <= 1:
            raise ValueError(f'need 0 <= low_thresh ({low_thresh}) <= high_thresh ({high_thresh}) <= 1')
        if new_track_thresh is not None and not 0 <= new_track_thresh <= 1:
            raise ValueError(f'new_track_thresh must be in [0, 1], got {new_track_thresh}')
        if not 0 < match_thresh <= 1:
            raise ValueError(f'match_thresh (an IoU distance) must be in (0, 1], got {match_thresh}')
        if track_buffer < 0:
            raise ValueError(f'track_buffer must be >= 0, got {track_buffer}')
        if not frame_rate > 0:
            raise ValueError(f'frame_rate must be > 0, got {frame_rate}')
        if not 0 <= appearance_thresh <= 1:
            raise ValueError(f'appearance_thresh must be in [0, 1], got {appearance_thresh}')
        if not 0 < proximity_thresh <= 1:
            raise ValueError(f'proximity_thresh must be in (0, 1], got {proximity_thresh}')
        if not 0 <= feat_alpha < 1:
            raise ValueError(f'feat_alpha must be in [0, 1), got {feat_alpha}')
        self.appearance_thresh = float(appearance_thresh)
        self.proximity_thresh = float(proximity_thresh)
        self.feat_alpha = float(feat_alpha)
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(high_thresh + 0.1 if new_track_thresh is None else new_track_thresh)
        self.match_thresh = float(match_thresh)
        self.max_time_lost = max(1, int(frame_rate / 30.0 * track_buffer))
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

    # The single place that builds association costs.
    def _distance(self, tracks: Sequence[Track], dets: Sequence[Track], fuse: bool,
                  appearance: bool = False) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        iou = iou_matrix([t.box for t in tracks], [d.box for d in dets])
        far = (1.0 - iou) > self.proximity_thresh
        if fuse:
            iou = iou * np.array([d.score for d in dets])[None, :]
        cost = 1.0 - iou
        if appearance:
            cost = np.minimum(cost, self._appearance_distance(tracks, dets, far))
        if not self.class_agnostic:
            same = np.array([[t.label == d.label for d in dets] for t in tracks])
            cost = np.where(same, cost, np.inf)
        return cost

    def _appearance_distance(self, tracks, dets, far) -> np.ndarray:
        """BoT-SORT's gated embedding distance: ``(1 - cos) / 2``, or 1 when it is above
        ``appearance_thresh``, when the boxes are far apart, or when a side has no embedding."""
        emb = np.ones((len(tracks), len(dets)))
        ti = [i for i, t in enumerate(tracks) if t.feat is not None]
        di = [j for j, d in enumerate(dets) if d.curr_feat is not None]
        if ti and di:
            tf = np.stack([tracks[i].feat for i in ti])
            df = np.stack([dets[j].curr_feat for j in di])
            emb[np.ix_(ti, di)] = np.clip(1.0 - tf @ df.T, 0.0, 2.0) / 2.0
        emb[emb > self.appearance_thresh] = 1.0
        emb[far] = 1.0
        return emb

    def update(self, detections: Sequence, frame: np.ndarray | None = None,
               embeddings: Sequence | None = None) -> list[TrackOutput]:
        """Advance one step. ``detections``: objects with ``box`` (xyxy), ``score``, ``label``;
        ``embeddings``: optional appearance vector per detection (enables BoT-SORT's fused cost)."""
        self.frame_id += 1
        fid = self.frame_id
        activated, refind, lost, removed = [], [], [], []

        detections = list(detections)
        if embeddings is not None and len(embeddings) != len(detections):
            raise ValueError(f'got {len(embeddings)} embeddings for {len(detections)} detections')
        use_app = embeddings is not None and len(detections) > 0
        dets = [Track(d.box, d.score, d.label) for d in detections]
        for d in dets:
            d.feat_alpha = self.feat_alpha
        if use_app:
            for d, e in zip(dets, embeddings):
                d.set_feature(e)
        high = [d for d in dets if d.score >= self.high_thresh]
        low = [d for d in dets if self.low_thresh <= d.score < self.high_thresh]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked = [t for t in self.tracked if t.is_activated]

        # Step 2: first association, high-score detections.
        pool = _joint(tracked, self.lost)
        for t in pool:
            t.predict()
        if self.motion_compensator is not None and frame is not None:
            if getattr(self.motion_compensator, 'uses_boxes', False):
                affine = self.motion_compensator(frame, [d.box for d in detections])
            else:
                affine = self.motion_compensator(frame)
            apply_affine_to_tracks(pool, affine)
            apply_affine_to_tracks(unconfirmed, affine)
        cost = self._distance(pool, high, self.fuse_score, use_app)
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
        cost = self._distance(unconfirmed, rest, self.fuse_score, use_app)
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
        self.tracked, self.lost = _remove_duplicates(self.tracked, self.lost, self.class_agnostic)

        return [TrackOutput(t.track_id, tuple(float(v) for v in t.box), t.score, t.label,
                            start_step=t.start_frame, birth_box=tuple(float(v) for v in t._det_box),
                            birth_score=t.birth_score, feat=t.curr_feat)
                for t in self.tracked if t.is_activated]


def _joint(a: list[Track], b: list[Track]) -> list[Track]:
    seen = {t.track_id for t in a}
    return list(a) + [t for t in b if t.track_id not in seen and not seen.add(t.track_id)]


def _sub(a: list[Track], b: list[Track]) -> list[Track]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(tracked: list[Track], lost: list[Track], class_agnostic: bool = False):
    if not tracked or not lost:
        return tracked, lost
    dist = 1.0 - iou_matrix([t.box for t in tracked], [t.box for t in lost])
    dup_t, dup_l = set(), set()
    for p, q in zip(*np.where(dist < 0.15)):
        if not class_agnostic and tracked[p].label != lost[q].label:
            continue
        if tracked[p].frame_id - tracked[p].start_frame > lost[q].frame_id - lost[q].start_frame:
            dup_l.add(q)
        else:
            dup_t.add(p)
    return ([t for i, t in enumerate(tracked) if i not in dup_t],
            [t for i, t in enumerate(lost) if i not in dup_l])

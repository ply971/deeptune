"""Offline track post-processing: link fragments, fill gaps, drop noise.

An online tracker must decide at every frame. ``track.py`` processes a whole
video before it writes anything, so it can look back and repair the two most
common failures of tracking-by-detection:

1. **Fragmentation** - an object that is missed for longer than the tracker's
   buffer (occlusion, motion blur, leaving the frame briefly) comes back with a
   new ID, and is counted twice. :func:`link_tracklets` joins a tracklet that
   ends to one that starts later when the gap is short, the jump is plausible
   for the motion seen so far, the size is similar and (with Re-ID) the two
   look alike. A link is made only when it is **unambiguous**: the ending
   tracklet has exactly one candidate continuation inside the gates, and that
   candidate has exactly one candidate predecessor. A lone cat that walks behind
   a chair is joined; in a crowd, where a new track near a lost one may well be
   a different person, nothing is joined (on MOT17's development videos, the
   permissive version - a global assignment over all candidates - cut IDF1 by
   7 points through wrong merges; see docs/video-plan/reports/S3-botsort.md).
   This is the offline "tracklet association" idea of StrongSORT's AFLink
   (Du et al., IEEE TMM 2023), done here with hand-written motion/appearance
   gates instead of a learned link model.
2. **Holes** - frames inside a track where the object was not output (missed
   detections, the lost period before re-identification). :func:`interpolate`
   fills holes up to ``max_gap_s`` with linearly interpolated boxes, as
   ByteTrack's and StrongSORT's offline interpolation do. Filled rows are
   flagged (``interpolated=True``) and never count as observations.

:func:`drop_short_tracks` then removes tracks with fewer than ``min_len`` real
observations (one-off false positives that would inflate counts).

Rows are dicts with ``frame`` (0-based video frame), ``track_id``, ``box``
(xyxy), ``score``, ``label`` and optionally ``feat`` (appearance vector).
Nothing here imports torch.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from spatial.track import linear_assignment


@dataclass
class Tracklet:
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    feats: list = field(default_factory=list)

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    @property
    def label(self) -> str:
        return self.labels[-1]

    def end_feature(self, k: int, at_end: bool):
        feats = [f for f in (self.feats[-k:] if at_end else self.feats[:k]) if f is not None]
        if not feats:
            return None
        mean = np.mean(np.stack([np.asarray(f, dtype=np.float64) for f in feats]), axis=0)
        return mean / max(float(np.linalg.norm(mean)), 1e-12)


def tracklets_from_rows(rows: Sequence[dict]) -> dict[int, Tracklet]:
    out: dict[int, Tracklet] = {}
    for r in sorted(rows, key=lambda r: (r['track_id'], r['frame'])):
        t = out.setdefault(int(r['track_id']), Tracklet(int(r['track_id'])))
        if t.frames and t.frames[-1] == int(r['frame']):
            continue  # one row per frame and track
        t.frames.append(int(r['frame']))
        t.boxes.append(tuple(float(v) for v in r['box']))
        t.scores.append(float(r['score']))
        t.labels.append(str(r['label']))
        t.feats.append(r.get('feat'))
    return out


def _centre(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


def _size(box) -> float:
    x1, y1, x2, y2 = box
    return float(np.sqrt(max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)))


def _velocity(t: Tracklet, at_end: bool, k: int = 10) -> np.ndarray:
    """Centre velocity (pixels per frame) from a least-squares line over k observations."""
    idx = range(max(0, len(t.frames) - k), len(t.frames)) if at_end else range(min(k, len(t.frames)))
    f = np.array([t.frames[i] for i in idx], dtype=np.float64)
    if len(f) < 2 or np.ptp(f) == 0:
        return np.zeros(2)
    c = np.stack([_centre(t.boxes[i]) for i in idx])
    f = f - f.mean()
    return (f[:, None] * (c - c.mean(axis=0))).sum(axis=0) / (f ** 2).sum()


@dataclass
class LinkSettings:
    """Gates of :func:`link_tracklets` (see its docstring)."""

    max_gap_s: float = 3.0
    max_jump: float = 1.5           # allowed centre distance in box sizes at a zero gap ...
    jump_per_s: float = 1.5         # ... plus this many box sizes per second of gap
    max_extrapolate_s: float = 1.0  # velocity is trusted for at most this long
    size_ratio: float = 2.5         # box sizes may differ by at most this factor
    appearance_thresh: float = 0.3  # largest (1 - cos) / 2 between end and start appearance
    feature_window: int = 10        # observations averaged at each end for the appearance
    class_agnostic: bool = False
    unambiguous: bool = True        # link only one-candidate-each-way pairs (see module docstring)


def link_cost(a: Tracklet, b: Tracklet, fps: float, s: LinkSettings) -> float:
    """Cost in [0, 1] of "b continues a", or ``inf`` when a gate fails."""
    if b.first <= a.last:
        return np.inf
    if not s.class_agnostic and a.label != b.label:
        return np.inf
    gap_s = (b.first - a.last) / fps
    if gap_s > s.max_gap_s:
        return np.inf
    box_a, box_b = a.boxes[-1], b.boxes[0]
    size_a, size_b = _size(box_a), _size(box_b)
    if max(size_a, size_b) / min(size_a, size_b) > s.size_ratio:
        return np.inf
    scale = (size_a + size_b) / 2.0
    ca, cb = _centre(box_a), _centre(box_b)
    steps = min(b.first - a.last, s.max_extrapolate_s * fps)
    forward = ca + _velocity(a, True) * steps
    backward = cb - _velocity(b, False) * steps
    jump = min(np.linalg.norm(forward - cb), np.linalg.norm(backward - ca), np.linalg.norm(cb - ca)) / scale
    allowed = s.max_jump + s.jump_per_s * gap_s
    if jump > allowed:
        return np.inf
    motion = jump / allowed
    fa, fb = a.end_feature(s.feature_window, True), b.end_feature(s.feature_window, False)
    if fa is not None and fb is not None:
        app = float(np.clip(1.0 - fa @ fb, 0.0, 2.0)) / 2.0
        if app > s.appearance_thresh:
            return np.inf
        cost = 0.5 * motion + 0.5 * app / max(s.appearance_thresh, 1e-6)
    else:
        cost = motion
    # Shorter gaps first when everything else is equal.
    return float(min(0.999, 0.9 * cost + 0.1 * gap_s / max(s.max_gap_s, 1e-6)))


def link_tracklets(tracklets: dict[int, Tracklet], fps: float, settings: LinkSettings | None = None
                   ) -> dict[int, int]:
    """Map every track id to the id of the chain it joins (the chain's first id).

    ``b`` may continue ``a`` only if ``b`` starts after ``a`` ends and
    within ``max_gap_s``; the centre jump, measured from ``a``'s last box moved
    by its own velocity (and ``b``'s first box moved back by its velocity, and
    the plain distance; the smallest counts), is at most ``max_jump +
    jump_per_s * gap`` box sizes; the box sizes differ by at most
    ``size_ratio``; labels match (unless class-agnostic); and with appearance
    vectors the end and start appearance distance is at most
    ``appearance_thresh``. A chain also never overlaps itself in time, because
    each link goes forward in time from a tracklet's last frame.

    With ``unambiguous`` (default) a pair is linked only when ``a`` has no other
    candidate continuation and ``b`` no other candidate predecessor. Otherwise
    one global minimum-cost assignment between tracklet ends (rows) and starts
    (columns) picks the links, so every tracklet still gets at most one
    predecessor and one successor.
    """
    s = settings or LinkSettings()
    if fps <= 0:
        raise ValueError('fps must be > 0')
    ids = sorted(tracklets)
    mapping = {i: i for i in ids}
    if len(ids) < 2 or s.max_gap_s <= 0:
        return mapping
    cost = np.full((len(ids), len(ids)), np.inf)
    for r, ia in enumerate(ids):
        for c, ib in enumerate(ids):
            if ia != ib:
                cost[r, c] = link_cost(tracklets[ia], tracklets[ib], fps, s)
    if s.unambiguous:
        ok = np.isfinite(cost)
        single = (ok.sum(axis=1, keepdims=True) == 1) & (ok.sum(axis=0, keepdims=True) == 1)
        cost = np.where(ok & single, cost, np.inf)
    if not np.isfinite(cost).any():
        return mapping
    matches, _, _ = linear_assignment(cost, 1.0)
    successor = {ids[r]: ids[c] for r, c in matches.tolist()}
    has_pred = set(successor.values())
    for head in ids:
        if head in has_pred:
            continue
        node = head
        while node in successor:
            node = successor[node]
            mapping[node] = head
    return mapping


def apply_mapping(rows: Sequence[dict], mapping: dict[int, int]) -> list[dict]:
    """Relabel track ids. A merged chain takes its majority label (by observations)."""
    out = [{**r, 'track_id': mapping.get(int(r['track_id']), int(r['track_id']))} for r in rows]
    votes: dict[int, Counter] = {}
    for r in out:
        votes.setdefault(r['track_id'], Counter())[r['label']] += 1
    merged = {tid for tid, c in Counter(mapping.values()).items() if c > 1}
    for r in out:
        if r['track_id'] in merged:
            r['label'] = votes[r['track_id']].most_common(1)[0][0]
    return out


def interpolate(rows: Sequence[dict], fps: float, every_n_frames: int = 1, max_gap_s: float = 1.0
                ) -> list[dict]:
    """Fill holes inside each track on the sampled-frame grid (frames ``k * every_n_frames``).

    A hole between two observed frames ``f1 < f2`` of a track is filled when
    ``(f2 - f1) / fps <= max_gap_s``: boxes and scores are linearly
    interpolated; the new rows carry ``interpolated=True`` and no ``feat``.
    ``max_gap_s <= 0`` disables filling.
    """
    rows = [dict(r) for r in rows]
    if max_gap_s <= 0:
        return rows
    step = max(1, int(every_n_frames))
    by_track: dict[int, list[dict]] = {}
    for r in rows:
        by_track.setdefault(int(r['track_id']), []).append(r)
    filled: list[dict] = []
    for tid, trs in by_track.items():
        trs.sort(key=lambda r: r['frame'])
        for a, b in zip(trs, trs[1:]):
            f1, f2 = int(a['frame']), int(b['frame'])
            if f2 - f1 <= step or (f2 - f1) / fps > max_gap_s:
                continue
            start = (f1 // step + 1) * step
            ba, bb = np.asarray(a['box'], dtype=np.float64), np.asarray(b['box'], dtype=np.float64)
            for f in range(start, f2, step):
                w = (f - f1) / (f2 - f1)
                filled.append({'frame': f, 'track_id': tid, 'box': tuple((1 - w) * ba + w * bb),
                               'score': float((1 - w) * a['score'] + w * b['score']),
                               'label': a['label'], 'interpolated': True})
    return rows + filled


def drop_short_tracks(rows: Sequence[dict], min_len: int) -> list[dict]:
    """Remove tracks with fewer than ``min_len`` real (not interpolated) observations."""
    if min_len <= 1:
        return list(rows)
    real = Counter(int(r['track_id']) for r in rows if not r.get('interpolated'))
    return [r for r in rows if real[int(r['track_id'])] >= min_len]


def renumber(rows: Sequence[dict]) -> list[dict]:
    """Track ids 1, 2, 3, ... in order of first appearance (then old id)."""
    first: dict[int, int] = {}
    for r in rows:
        tid = int(r['track_id'])
        first[tid] = min(first.get(tid, r['frame']), r['frame'])
    order = {tid: i + 1 for i, tid in enumerate(sorted(first, key=lambda t: (first[t], t)))}
    return [{**r, 'track_id': order[int(r['track_id'])]} for r in rows]


@dataclass
class PostSettings:
    """Everything :func:`postprocess` does; the defaults are track.py's ``botsort`` preset."""

    link: LinkSettings = field(default_factory=LinkSettings)
    interpolate_max_gap_s: float = 1.0
    min_track_len: int = 3
    renumber: bool = True


def postprocess(rows: Sequence[dict], fps: float, every_n_frames: int = 1,
                settings: PostSettings | None = None) -> tuple[list[dict], dict]:
    """Drop short tracks, link, interpolate and renumber. Returns ``(rows, stats)``.

    Short (noise) tracks go first, so they can neither make a link look
    ambiguous nor be joined into a real track.
    """
    s = settings or PostSettings()
    rows = list(rows)
    raw_ids = {int(r['track_id']) for r in rows}
    rows = drop_short_tracks(rows, s.min_track_len)
    kept = {int(r['track_id']) for r in rows}
    mapping = link_tracklets(tracklets_from_rows(rows), fps, s.link)
    links = sum(1 for k, v in mapping.items() if k != v)
    rows = apply_mapping(rows, mapping)
    rows = interpolate(rows, fps, every_n_frames, s.interpolate_max_gap_s)
    final = {int(r['track_id']) for r in rows}
    if s.renumber:
        rows = renumber(rows)
    rows.sort(key=lambda r: (r['frame'], r['track_id']))
    stats = {'raw_tracks': len(raw_ids), 'links': links, 'dropped_short_tracks': len(raw_ids - kept),
             'interpolated_boxes': sum(1 for r in rows if r.get('interpolated')), 'final_tracks': len(final)}
    return rows, stats

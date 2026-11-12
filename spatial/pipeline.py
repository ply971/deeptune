"""Video -> detections -> ByteTrack / BoT-SORT tracks -> post-processing -> output files."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from spatial import io as sio
from spatial.detect import Detection, Detector, clean_prompts, cross_label_suppress
from spatial.track import ByteTracker

CMC_CHOICES = ('sparse', 'none')


def _clip_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box  # the Kalman state can drift past the frame edge
    return (min(max(x1, 0.0), frame_w), min(max(y1, 0.0), frame_h),
            min(max(x2, 0.0), frame_w), min(max(y2, 0.0), frame_h))


class OnlineTracking:
    """One video's tracker plus the rows it outputs (shared by :func:`track_video` and benchmarks).

    Call :meth:`step` once per sampled frame, in order, then :meth:`finish`.
    ``emit_birth`` adds each track's first detection, which ByteTrack outputs only
    from its second (confirming) step. Rows keep the matched detection's
    appearance vector under ``feat`` until :meth:`finish` (the offline linker uses it).
    """

    def __init__(self, steps_per_second: float, tracker_kwargs: dict | None = None, embedder=None,
                 cmc: str = 'none', emit_birth: bool = False):
        if cmc not in CMC_CHOICES:
            raise ValueError(f'cmc must be one of {CMC_CHOICES}, got {cmc!r}')
        self.compensator = None
        if cmc == 'sparse':
            from spatial.cmc import SparseFlowCMC
            self.compensator = SparseFlowCMC()
        self.cmc = cmc
        self.embedder = embedder
        self.emit_birth = emit_birth
        self.tracker = ByteTracker(frame_rate=steps_per_second, motion_compensator=self.compensator,
                                   **(tracker_kwargs or {}))
        self.rows: list[dict] = []
        self._step_frame: dict[int, int] = {}
        self._emitted: set[int] = set()

    def step(self, frame_index: int, frame, detections, embeddings=None, frame_size=None) -> None:
        """Track one sampled frame. ``frame`` may be None (no CMC, and embeddings must be
        given or the embedder unused); ``frame_size`` = (width, height) then clips boxes."""
        detections = list(detections)
        if embeddings is None and self.embedder is not None and frame is not None:
            embeddings = self.embedder.embed(frame, [d.box for d in detections])
        if frame_size is not None:
            frame_w, frame_h = frame_size
        elif frame is not None:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_w = frame_h = float('inf')
        outputs = self.tracker.update(detections, frame=frame, embeddings=embeddings)
        self._step_frame[self.tracker.frame_id] = frame_index
        for t in outputs:
            birth_frame = self._step_frame.get(t.start_step)
            if (self.emit_birth and t.track_id not in self._emitted and t.birth_box is not None
                    and birth_frame is not None and birth_frame != frame_index):
                self.rows.append({'frame': birth_frame, 'track_id': t.track_id,
                                  'box': _clip_box(t.birth_box, frame_w, frame_h),
                                  'score': float(t.birth_score), 'label': t.label})
            self._emitted.add(t.track_id)
            row = {'frame': frame_index, 'track_id': t.track_id, 'box': _clip_box(t.box, frame_w, frame_h),
                   'score': t.score, 'label': t.label}
            if t.feat is not None:
                row['feat'] = t.feat.astype('float32')
            self.rows.append(row)

    def finish(self, fps: float, every_n_frames: int = 1, post=None) -> tuple[list[dict], dict | None]:
        """Apply the offline post-processing (if any) and drop the appearance vectors."""
        rows, stats = self.rows, None
        if post is not None:
            from spatial.postprocess import postprocess
            rows, stats = postprocess(rows, fps, every_n_frames, post)
        for r in rows:
            r.pop('feat', None)
        return sorted(rows, key=lambda r: (r['frame'], r['track_id'])), stats

    def describe(self, post_stats=None) -> dict:
        botsort = self.embedder is not None or self.compensator is not None or post_stats is not None
        return {'tracker': 'botsort' if botsort else 'bytetrack',
                'reid': self.embedder.describe() if self.embedder is not None else None, 'cmc': self.cmc,
                'emit_birth': self.emit_birth, 'postprocess': post_stats}


def track_video(video_path, prompts: Sequence[str], detector: Detector, out_dir,
                every_n_frames: int = 1, max_frames: int | None = None,
                tracker_kwargs: dict | None = None, save_video: bool = False,
                cross_label_nms: float | None = None, log=print, *, embedder=None, cmc: str = 'none',
                emit_birth: bool = False, post=None) -> dict:
    """Detect + track one video and write its output folder.

    Frames are decoded once and streamed: sampled frames are detected in batches
    of ``detector.batch_size`` and fed to the tracker in order, so memory stays
    bounded by one batch. The tracker runs at ``fps / every_n_frames`` steps per
    second (``track_buffer`` is scaled accordingly, as in ByteTrack).
    ``cross_label_nms`` (IoU) keeps only the best-scoring label where boxes of
    different prompts overlap (see :func:`spatial.detect.cross_label_suppress`).

    BoT-SORT parts (all off by default, i.e. plain ByteTrack; ``track.py``'s
    ``botsort`` preset turns them on):

    - ``embedder``: appearance embedder (:mod:`spatial.reid`); every detection gets
      an embedding and the tracker uses the fused IoU/appearance cost.
    - ``cmc='sparse'``: camera-motion compensation (:class:`spatial.cmc.SparseFlowCMC`).
    - ``emit_birth``: also write each track's first detection, which ByteTrack only
      confirms (and outputs) one step later.
    - ``post``: a :class:`spatial.postprocess.PostSettings`; link fragments, fill
      holes and drop very short tracks before the files are written.

    Returns ``{"out_dir", "files", "summary"}``.
    """
    if every_n_frames < 1:
        raise ValueError('every_n_frames must be >= 1')
    prompts = clean_prompts(prompts)
    info = sio.video_info(video_path)  # fails before any output folder is created
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    online = OnlineTracking(info.fps / every_n_frames, tracker_kwargs, embedder, cmc, emit_birth)

    det_frames: list[dict] = []
    batch_idx: list[int] = []
    batch_frames: list = []
    n_read = 0
    width = height = 0
    t0 = time.time()

    def flush():
        if not batch_frames:
            return
        results = detector.detect(batch_frames, prompts, frame_indices=list(batch_idx))
        for fi, frame, dets in zip(batch_idx, batch_frames, results):
            dets = cross_label_suppress(dets, cross_label_nms)
            det_frames.append({'frame': fi, 'time_s': round(fi / info.fps, 4),
                               'detections': [d.to_dict() for d in dets]})
            online.step(fi, frame, dets)
        batch_idx.clear()
        batch_frames.clear()

    for idx, frame, sampled in sio.iter_frames(video_path, every_n_frames, max_frames):
        n_read = idx + 1
        height, width = frame.shape[:2]
        if sampled:
            batch_idx.append(idx)
            batch_frames.append(frame)
            if len(batch_frames) >= detector.batch_size:
                flush()
    flush()
    if n_read == 0:
        raise ValueError(f'no frames could be decoded from {video_path}')
    track_rows, post_stats = online.finish(info.fps, every_n_frames, post)

    files = {
        'detections': sio.write_json(out_dir / 'detections.json', {
            'video': str(video_path), 'fps': info.fps, 'width': width, 'height': height,
            'num_frames': n_read, 'every_n_frames': every_n_frames, 'prompts': prompts,
            'detector': detector.describe(), 'frames': det_frames}),
        'tracks': sio.write_mot(out_dir / 'tracks.csv', track_rows, prompts),
        'track_labels': sio.write_track_labels(out_dir / 'track_labels.csv', track_rows, prompts),
    }
    summary = sio.build_summary(track_rows, prompts, info.fps)
    summary = {'video': str(video_path), 'num_frames': n_read,
               'duration_s': round(n_read / info.fps, 4), 'every_n_frames': every_n_frames,
               'prompts': prompts, **summary}
    summary['tracking'] = online.describe(post_stats)
    files['summary'] = sio.write_json(out_dir / 'summary.json', summary)
    if save_video:
        from spatial.draw import write_overlay
        files['overlay'] = write_overlay(video_path, out_dir / 'overlay.mp4', track_rows, info.fps,
                                         max_frames=max_frames, hold_frames=every_n_frames)
    counts = ', '.join(f'{k}={v["count"]}' for k, v in summary['labels'].items())
    log(f'[track] {Path(video_path).name}: {n_read} frames in {time.time() - t0:.1f}s; tracks: {counts}')
    return {'out_dir': out_dir, 'files': {k: Path(v) for k, v in files.items()}, 'summary': summary}


def detections_from_json(path) -> dict[int, list[Detection]]:
    """Load ``detections.json`` back as ``frame -> [Detection]`` (e.g. to re-track
    with different settings through :class:`spatial.detect.FakeDetector`)."""
    import json
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {int(f['frame']): [Detection.from_dict(d) for d in f['detections']] for f in data['frames']}

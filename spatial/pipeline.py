"""Video -> zero-shot detections -> ByteTrack tracks -> output files."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from spatial import io as sio
from spatial.detect import Detection, Detector, clean_prompts, cross_label_suppress
from spatial.track import ByteTracker


def track_video(video_path, prompts: Sequence[str], detector: Detector, out_dir,
                every_n_frames: int = 1, max_frames: int | None = None,
                tracker_kwargs: dict | None = None, save_video: bool = False,
                cross_label_nms: float | None = None, log=print) -> dict:
    """Detect + track one video and write its output folder.

    Frames are decoded once and streamed: sampled frames are detected in batches
    of ``detector.batch_size`` and fed to the tracker in order, so memory stays
    bounded by one batch. The tracker runs at ``fps / every_n_frames`` steps per
    second (``track_buffer`` is scaled accordingly, as in ByteTrack).
    ``cross_label_nms`` (IoU) keeps only the best-scoring label where boxes of
    different prompts overlap (see :func:`spatial.detect.cross_label_suppress`).

    Returns ``{"out_dir", "files", "summary"}``.
    """
    if every_n_frames < 1:
        raise ValueError('every_n_frames must be >= 1')
    prompts = clean_prompts(prompts)
    info = sio.video_info(video_path)  # fails before any output folder is created
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = ByteTracker(frame_rate=info.fps / every_n_frames, **(tracker_kwargs or {}))

    det_frames: list[dict] = []
    track_rows: list[dict] = []
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
            for t in tracker.update(dets, frame=frame):
                track_rows.append({'frame': fi, 'track_id': t.track_id, 'box': t.box,
                                   'score': t.score, 'label': t.label})
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

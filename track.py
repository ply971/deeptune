"""Zero-shot detection + tracking of objects in video.

Example::

    python track.py --video lobsters.mp4 --prompt lobster --out runs/lobsters --save_video
    python track.py --video_dir clips/ --prompt lobster --prompt fish --detector owlv2 --out runs/clips

Writes, per video, ``<out>/<video stem>/`` with ``detections.json``,
``tracks.csv`` (MOTChallenge), ``track_labels.csv``, ``summary.json`` and
optionally ``overlay.mp4``; plus ``<out>/track_arguments.json`` and
``<out>/counts.csv`` (video, label, count). See ``spatial/io.py`` for formats.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from spatial.detect import (DEFAULT_MODELS, DETECTOR_DEFAULTS, LICENCES, Detector, build_detector,
                            check_grounding_dino_prompts, clean_prompts)
from spatial.track import ByteTracker
from spatial import io as sio

# Tracker thresholds per detector score scale (ByteTrack's 0.5/0.6 assume a
# trained closed-set detector; zero-shot scores are lower).
TRACK_DEFAULTS = {
    'grounding_dino': {'track_high_thresh': 0.35},
    'owlv2': {'track_high_thresh': 0.2},
}


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='track.py', description='Zero-shot (open-vocabulary) detection and ByteTrack tracking on video.')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=str, help='Path to one video file.')
    src.add_argument('--video_dir', type=str, help='Folder of videos (non-recursive: '
                     + ', '.join(sio.VIDEO_EXTENSIONS) + ').')
    p.add_argument('--prompt', action='append', required=True,
                   help='Object to find, in words (e.g. "lobster"). Repeat for several objects.')
    p.add_argument('--out', type=str, required=True, help='Output folder (one sub-folder per video).')
    p.add_argument('--detector', choices=sorted(DEFAULT_MODELS), default='grounding_dino',
                   help='Zero-shot detector backend (default: grounding_dino).')
    p.add_argument('--detector_model', type=str, default=None,
                   help='Hugging Face model id or local folder; default per detector: '
                   + ', '.join(f'{k}={v}' for k, v in DEFAULT_MODELS.items()) + '.')
    p.add_argument('--box_threshold', type=float, default=None,
                   help='Minimum box score kept from the detector (default: grounding_dino 0.2, owlv2 0.1).')
    p.add_argument('--text_threshold', type=float, default=None,
                   help='Grounding DINO only: minimum prompt-token score for a box to get that label '
                        '(default 0.25).')
    p.add_argument('--nms_iou', type=float, default=0.5,
                   help='Per-label NMS IoU applied to detector boxes; >= 1 disables (default 0.5).')
    p.add_argument('--every_n_frames', type=int, default=1,
                   help='Detect and track on every n-th frame only (default 1 = all frames).')
    p.add_argument('--max_frames', type=int, default=None, help='Stop after this many decoded frames.')
    p.add_argument('--batch_size', type=int, default=4, help='Frames per detector forward pass (default 4).')
    p.add_argument('--device', type=str, default='auto', help='auto (CUDA if available), cpu or cuda[:N].')
    p.add_argument('--track_high_thresh', type=float, default=None,
                   help='ByteTrack high-score threshold (first association). Default: grounding_dino 0.35, '
                        'owlv2 0.2.')
    p.add_argument('--track_low_thresh', type=float, default=None,
                   help='ByteTrack low-score threshold (second association). Default: the box threshold, '
                        'since lower boxes never reach the tracker.')
    p.add_argument('--new_track_thresh', type=float, default=None,
                   help='Minimum score to start a new track (default: track_high_thresh + 0.1).')
    p.add_argument('--match_thresh', type=float, default=0.8,
                   help='IoU-distance gate of the first association (default 0.8 = IoU >= 0.2).')
    p.add_argument('--track_buffer', type=int, default=30,
                   help='Steps a lost track can be re-found, at 30 steps/s (scaled by fps / every_n_frames; '
                        'default 30).')
    p.add_argument('--cross_label_nms', type=float, default=None,
                   help='IoU above which overlapping boxes of DIFFERENT prompts are merged into the '
                        'best-scoring one, so one object found by two prompts is counted once (default: off; '
                        'e.g. 0.7).')
    p.add_argument('--class_agnostic', action='store_true',
                   help='Match boxes across labels; each track takes the majority label.')
    p.add_argument('--save_video', action='store_true', help='Also write overlay.mp4 with boxes and track IDs.')
    return p


def resolve_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Validate the arguments and fill per-detector defaults (recorded in
    track_arguments.json). Every problem is reported through ``parser.error``
    (usage + message, exit code 2) before any video is read or weights load."""
    if any(not str(p).strip() for p in args.prompt):
        parser.error('--prompt must not be blank')
    args.prompt = clean_prompts(args.prompt)
    if args.detector == 'grounding_dino':
        try:
            check_grounding_dino_prompts(args.prompt)
        except ValueError as exc:
            parser.error(str(exc))
    d = DETECTOR_DEFAULTS[args.detector]
    if args.box_threshold is None:
        args.box_threshold = d['box_threshold']
    if args.text_threshold is None:
        args.text_threshold = d['text_threshold']
    if args.detector_model is None:
        args.detector_model = DEFAULT_MODELS[args.detector]
    if args.track_high_thresh is None:
        args.track_high_thresh = TRACK_DEFAULTS[args.detector]['track_high_thresh']
    if args.track_low_thresh is None:
        args.track_low_thresh = min(args.box_threshold, args.track_high_thresh)
    if args.new_track_thresh is None:
        args.new_track_thresh = round(args.track_high_thresh + 0.1, 4)
    for name in ('box_threshold', 'text_threshold'):
        v = getattr(args, name)
        if v is not None and not 0 <= v <= 1:
            parser.error(f'--{name} must be in [0, 1], got {v}')
    if args.nms_iou <= 0:
        parser.error('--nms_iou must be > 0 (use 1 to disable NMS)')
    if args.cross_label_nms is not None and args.cross_label_nms <= 0:
        parser.error('--cross_label_nms must be > 0')
    if args.every_n_frames < 1:
        parser.error('--every_n_frames must be >= 1')
    if args.batch_size < 1:
        parser.error('--batch_size must be >= 1')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max_frames must be >= 1')
    try:
        ByteTracker(**tracker_kwargs(args))
    except ValueError as exc:
        parser.error(f'invalid tracker settings: {exc}')
    return args


def tracker_kwargs(args: argparse.Namespace) -> dict:
    return {'high_thresh': args.track_high_thresh, 'low_thresh': args.track_low_thresh,
            'new_track_thresh': args.new_track_thresh, 'match_thresh': args.match_thresh,
            'track_buffer': args.track_buffer, 'class_agnostic': args.class_agnostic}


def save_track_args(args: argparse.Namespace, out: Path, detector: Detector) -> Path:
    record = {**vars(args), 'detector_resolved': detector.describe() if detector is not None else None,
              'detector_licence': LICENCES.get(args.detector_model, 'see the model card'),
              'created': datetime.now().isoformat(timespec='seconds'),
              'python': platform.python_version(), 'command': sys.argv}
    return sio.write_json(out / 'track_arguments.json', record)


def _video_out_dirs(videos: list[Path], out: Path) -> list[Path]:
    seen: dict[str, int] = {}
    dirs = []
    for v in videos:
        name = v.stem if [w.stem for w in videos].count(v.stem) == 1 else f'{v.stem}_{v.suffix.lstrip(".")}'
        seen[name] = seen.get(name, 0) + 1
        dirs.append(out / (name if seen[name] == 1 else f'{name}_{seen[name]}'))
    return dirs


def main(argv=None, detector: Detector | None = None) -> dict:
    """Run the CLI and return ``{"results", "errors"}`` (errors: video -> message).

    ``detector`` lets callers (tests, notebooks) pass a ready detector instead of
    loading ``--detector`` weights. Argument problems exit through ``parser.error``.
    Every video is opened first, so unreadable files are reported before any
    weights load; a folder run then continues past videos that fail. Use
    :func:`cli` for the process exit code.
    """
    from spatial.pipeline import track_video
    parser = get_parser()
    args = resolve_args(parser.parse_args(argv), parser)
    if args.video:
        videos = [Path(args.video)]
        if not videos[0].is_file():
            parser.error(f'video not found: {args.video}')
    else:
        try:
            videos = sio.list_videos(args.video_dir)
        except ValueError as exc:
            parser.error(str(exc))
        if not videos:
            parser.error(f'no videos in {args.video_dir}')
    errors: dict[str, str] = {}
    readable = []
    for v in videos:
        try:
            sio.video_info(v)
            readable.append(v)
        except ValueError as exc:
            errors[str(v)] = f'{type(exc).__name__}: {exc}'
    if args.video and errors:
        parser.error(errors[str(videos[0])])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if readable and detector is None:
        kw = {'box_threshold': args.box_threshold, 'batch_size': args.batch_size,
              'nms_iou': args.nms_iou, 'device': args.device}
        if args.detector == 'grounding_dino':
            kw['text_threshold'] = args.text_threshold
        detector = build_detector(args.detector, args.detector_model, **kw)
    save_track_args(args, out, detector)
    results = {}
    for video, vdir in zip(readable, _video_out_dirs(readable, out)):
        try:
            results[str(video)] = track_video(video, args.prompt, detector, vdir, args.every_n_frames,
                                              args.max_frames, tracker_kwargs(args), args.save_video,
                                              cross_label_nms=args.cross_label_nms)
        except Exception as exc:  # one bad file must not stop a folder run
            if args.video:
                raise
            errors[str(video)] = f'{type(exc).__name__}: {exc}'
    for video, msg in errors.items():
        print(f'[track] {Path(video).name}: FAILED ({msg})', file=sys.stderr)
    with (out / 'counts.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['video', 'label', 'count'])
        for video, res in results.items():
            for label, entry in res['summary']['labels'].items():
                w.writerow([video, label, entry['count']])
    if errors:
        (out / 'errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
        print(f'[track] {len(errors)} of {len(videos)} videos failed; see {out / "errors.json"}', file=sys.stderr)
    return {'results': results, 'errors': errors}


def cli(argv=None, detector: Detector | None = None) -> int:
    """Process exit code: 0 when every video succeeded, 1 when any failed
    (argument errors exit with 2 through argparse)."""
    return 1 if main(argv, detector)['errors'] else 0


if __name__ == '__main__':
    sys.exit(cli())

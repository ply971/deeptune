"""Detection + tracking of objects in video.

Example::

    python track.py --video lobsters.mp4 --prompt lobster --out runs/lobsters --save_video
    python track.py --video_dir clips/ --prompt lobster --prompt fish --detector owlv2 --out runs/clips
    python track.py --video cat.mp4 --prompt cat --tracker bytetrack --out runs/plain   # the old tracker

The default ``--tracker botsort`` is ByteTrack plus BoT-SORT's camera-motion
compensation and appearance (Re-ID) cost, plus offline post-processing: broken
tracks of one object are linked again, short holes are filled by interpolation
and one-off noise tracks are dropped (see ``spatial/postprocess.py``).

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

from spatial.detect import (DEFAULT_MODELS, DETECTOR_DEFAULTS, LICENCES, Detector, auto_detector, build_detector,
                            check_coco_prompts, check_grounding_dino_prompts, clean_prompts)
from spatial.track import ByteTracker
from spatial import io as sio
from spatial.pipeline import CMC_CHOICES
from spatial.reid import REID_CHOICES, resolve_reid

# Tracker thresholds per detector score scale (ByteTrack's 0.5/0.6 assume a
# trained closed-set detector; zero-shot scores are lower).
TRACK_DEFAULTS = {
    'pet_fasterrcnn': {'track_high_thresh': 0.4},
    'grounding_dino': {'track_high_thresh': 0.35},
    'owlv2': {'track_high_thresh': 0.2},
    'dfine': {'track_high_thresh': 0.5},  # a trained closed-set detector: ByteTrack's reference value
}
# Fusing multiplies IoU by the box score, and confirming a new track needs the result
# above 0.3. A zero-shot score under 0.3 could then never confirm, whatever the thresholds.
ZERO_SHOT_DETECTORS = frozenset(('grounding_dino', 'owlv2'))

# What each --tracker turns on; any of these flags given explicitly wins.
TRACKER_PRESETS = {
    'botsort': {'cmc': 'sparse', 'reid': 'auto', 'link_max_gap_s': 3.0, 'interpolate_max_gap_s': 1.0,
                'min_track_len': 3},
    'bytetrack': {'cmc': 'none', 'reid': 'none', 'link_max_gap_s': 0.0, 'interpolate_max_gap_s': 0.0,
                  'min_track_len': 1},
}


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='track.py', description='Pretrained or local fine-tuned detection and ByteTrack tracking on video.')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=str, help='Path to one video file.')
    src.add_argument('--video_dir', type=str, help='Folder of videos (non-recursive: '
                     + ', '.join(sio.VIDEO_EXTENSIONS) + ').')
    p.add_argument('--prompt', action='append', required=True,
                   help='Object to find, in words (e.g. "lobster"). Repeat for several objects.')
    p.add_argument('--out', type=str, required=True, help='Output folder (one sub-folder per video).')
    p.add_argument('--detector', choices=['auto'] + sorted(DEFAULT_MODELS), default='auto',
                   help='Detector backend. auto (default): dfine when every prompt is one of the 80 COCO classes '
                        '(person, cat, car, ...; most accurate), else grounding_dino (any words). '
                        'pet_fasterrcnn requires a local --detector_model checkpoint.')
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
    p.add_argument('--tracker', choices=sorted(TRACKER_PRESETS), default='botsort',
                   help='botsort (default): ByteTrack + camera-motion compensation + appearance Re-ID + offline '
                        'linking of broken tracks, gap filling and noise-track removal. bytetrack: plain ByteTrack '
                        '(the tracker before 2026-11). The flags below override single parts.')
    p.add_argument('--cmc', choices=CMC_CHOICES, default=None,
                   help='Camera-motion compensation: sparse (optical flow on the background) or none. '
                        'Default: from --tracker.')
    p.add_argument('--reid', choices=REID_CHOICES, default=None,
                   help='Appearance embedding per box: cnn (ResNet-18 ImageNet features, 45 MB download once), '
                        'color (HSV histogram, no download), none, or auto (cnn when its weights are already '
                        'cached, else color). Default: from --tracker.')
    p.add_argument('--appearance_thresh', type=float, default=0.25,
                   help='Largest appearance distance (1 - cos) / 2 that may join a box to a nearby track '
                        '(default 0.25, as BoT-SORT).')
    p.add_argument('--proximity_thresh', type=float, default=0.5,
                   help='Appearance only helps when the boxes overlap with IoU >= 1 - this (default 0.5).')
    p.add_argument('--link_max_gap_s', type=float, default=None,
                   help='Offline: join a track that ends to one that starts up to this many seconds later when '
                        'motion, size and appearance agree (0 = off). Default: from --tracker (botsort 3).')
    p.add_argument('--link_appearance_thresh', type=float, default=0.3,
                   help='Offline linking: largest appearance distance between the two ends (default 0.3).')
    p.add_argument('--interpolate_max_gap_s', type=float, default=None,
                   help='Offline: fill holes inside a track up to this many seconds with interpolated boxes '
                        '(0 = off). Default: from --tracker (botsort 1).')
    p.add_argument('--min_track_len', type=int, default=None,
                   help='Offline: drop tracks with fewer real observations than this (1 = keep all). '
                        'Default: from --tracker (botsort 3).')
    p.add_argument('--save_video', action='store_true', help='Also write overlay.mp4 with boxes and track IDs.')
    return p


def resolve_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Validate the arguments and fill per-detector defaults (recorded in
    track_arguments.json). Every problem is reported through ``parser.error``
    (usage + message, exit code 2) before any video is read or weights load."""
    if any(not str(p).strip() for p in args.prompt):
        parser.error('--prompt must not be blank')
    args.prompt = clean_prompts(args.prompt)
    args.detector_requested = args.detector
    if args.detector == 'auto':
        args.detector = auto_detector(args.prompt)
    if args.detector == 'dfine':
        try:
            check_coco_prompts(args.prompt)
        except ValueError as exc:
            parser.error(str(exc))
    if args.detector == 'pet_fasterrcnn':
        args.detector_model = args.detector_model or DEFAULT_MODELS[args.detector]
        if not args.detector_model or not Path(args.detector_model).is_file():
            parser.error('--detector_model must be a local trained cat checkpoint')
        if any(p.lower() != 'cat' for p in args.prompt):
            parser.error('The trained cat detector supports only --prompt cat')
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
    for name, value in TRACKER_PRESETS[args.tracker].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.reid_resolved = resolve_reid(args.reid)
    for name in ('link_max_gap_s', 'interpolate_max_gap_s'):
        if getattr(args, name) < 0:
            parser.error(f'--{name} must be >= 0 (0 turns it off)')
    if args.min_track_len < 1:
        parser.error('--min_track_len must be >= 1')
    if not 0 <= args.link_appearance_thresh <= 1:
        parser.error('--link_appearance_thresh must be in [0, 1]')
    try:
        ByteTracker(**tracker_kwargs(args))
    except ValueError as exc:
        parser.error(f'invalid tracker settings: {exc}')
    return args


def post_settings(args: argparse.Namespace):
    """Offline post-processing, or None when every part is off (plain ByteTrack output)."""
    from spatial.postprocess import LinkSettings, PostSettings
    if args.link_max_gap_s == 0 and args.interpolate_max_gap_s == 0 and args.min_track_len == 1:
        return None
    return PostSettings(link=LinkSettings(max_gap_s=args.link_max_gap_s,
                                          appearance_thresh=args.link_appearance_thresh,
                                          class_agnostic=args.class_agnostic),
                        interpolate_max_gap_s=args.interpolate_max_gap_s, min_track_len=args.min_track_len)


def tracker_kwargs(args: argparse.Namespace) -> dict:
    return {'high_thresh': args.track_high_thresh, 'low_thresh': args.track_low_thresh,
            'new_track_thresh': args.new_track_thresh, 'match_thresh': args.match_thresh,
            'track_buffer': args.track_buffer, 'class_agnostic': args.class_agnostic,
            'fuse_score': args.detector not in ZERO_SHOT_DETECTORS,
            'appearance_thresh': args.appearance_thresh, 'proximity_thresh': args.proximity_thresh}


def save_track_args(args: argparse.Namespace, out: Path, detector: Detector, embedder=None) -> Path:
    description = detector.describe() if detector is not None else None
    licence = LICENCES.get(args.detector_model, 'see the model card')
    if args.detector == 'pet_fasterrcnn' and description:
        licence = description.get('provenance', {}).get('weights_terms', 'see checkpoint provenance')
    record = {**vars(args), 'tracker_settings': tracker_kwargs(args),
              'detector_resolved': description,
              'detector_licence': licence,
              'reid_resolved_model': embedder.describe() if embedder is not None else None,
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


def main(argv=None, detector: Detector | None = None, embedder=None) -> dict:
    """Run the CLI and return ``{"results", "errors"}`` (errors: video -> message).

    ``detector`` lets callers (tests, notebooks) pass a ready detector instead of
    loading ``--detector`` weights; ``embedder`` likewise replaces the ``--reid`` model. Argument problems exit through ``parser.error``.
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
    if args.reid_resolved == 'none':
        embedder = None
    elif readable and embedder is None:
        from spatial.reid import build_embedder
        embedder = build_embedder(args.reid_resolved, device=args.device)
    save_track_args(args, out, detector, embedder)
    post = post_settings(args)
    results = {}
    for video, vdir in zip(readable, _video_out_dirs(readable, out)):
        try:
            results[str(video)] = track_video(video, args.prompt, detector, vdir, args.every_n_frames,
                                              args.max_frames, tracker_kwargs(args), args.save_video,
                                              cross_label_nms=args.cross_label_nms, embedder=embedder,
                                              cmc=args.cmc, emit_birth=args.tracker == 'botsort', post=post)
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

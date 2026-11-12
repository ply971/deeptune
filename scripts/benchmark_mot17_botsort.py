"""MOT17 ablation of DeepTune's BoT-SORT additions against plain ByteTrack.

Runs the same tracker code as ``track.py`` (``spatial.pipeline.OnlineTracking``
and ``spatial.postprocess``) on all 7 labelled MOT17 training videos for each
public detection stream (FRCNN, SDP, DPM), and scores every variant with the
official TrackEval (HOTA, MOTA, IDF1).

Variants (all with ByteTrack's reference thresholds, nothing tuned on MOT17):

- ``bytetrack``  plain ByteTrack (must reproduce the earlier ``mot17-run01`` numbers)
- ``cmc``        + sparse optical-flow camera-motion compensation
- ``reid``       + ResNet-18 appearance in the fused BoT-SORT cost
- ``post``       + offline linking (motion only), interpolation, birth boxes, min length 3
- ``botsort``    everything: track.py's default ``--tracker botsort``

Appearance vectors and camera-motion affines are computed once per (video,
detection stream) and cached, then replayed into every variant, so variants
differ only in the tracker. This measures association, not DeepTune's
detectors; it is the public training split, not the hidden test set. DPM
scores are raw margins and pass through the same declared sigmoid as
``benchmark_mot17.py``.

Example::

    python scripts/benchmark_mot17_botsort.py --data D:/DeepTuneBenchmarks/datasets/mot17 \
        --trackeval D:/DeepTuneBenchmarks/references/TrackEval-12c8791b303e0a0b50f753af204249e622d0281a \
        --out D:/DeepTuneBenchmarks/results/mot17-botsort01
"""
from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from spatial.pipeline import OnlineTracking
from spatial.postprocess import LinkSettings, PostSettings

SEQUENCES = ('02', '04', '05', '09', '10', '11', '13')
DETECTORS = ('FRCNN', 'SDP', 'DPM')
# Development videos (design decisions) and held-out videos (report only), declared
# before any post-processing setting was changed. See docs/video-plan/reports/S3-botsort.md.
DEV_SEQUENCES = ('02', '04', '05')
HELDOUT_SEQUENCES = ('09', '10', '11', '13')
_OFF = dict(cmc=False, reid=False, link=False, interp=False, minlen=False, birth=False)
_ONLINE = dict(_OFF, cmc=True, reid=True)
VARIANTS = {
    'bytetrack': dict(_OFF),
    'cmc': dict(_OFF, cmc=True),
    'reid': dict(_OFF, reid=True),
    'online': dict(_ONLINE),                     # BoT-SORT online part: CMC + Re-ID
    'interp': dict(_ONLINE, interp=True),
    'link': dict(_ONLINE, link=True),
    'birth': dict(_ONLINE, birth=True),
    'minlen': dict(_ONLINE, minlen=True),
    'post': dict(_OFF, link=True, interp=True, minlen=True, birth=True),
    'botsort': dict(_ONLINE, link=True, interp=True, minlen=True, birth=True),  # track.py's default
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def detection_rows(path, sequence_length, detector):
    """Same validation and DPM transform as scripts/benchmark_mot17.py."""
    raw = np.loadtxt(path, delimiter=',', ndmin=2)
    if raw.shape[1] < 7 or not np.isfinite(raw).all():
        raise ValueError(f'Invalid MOT detection table: {path}')
    frame = raw[:, 0]
    if (frame != frame.astype(int)).any() or (frame < 1).any() or (frame > sequence_length).any():
        raise ValueError('Detection frame numbers fall outside seqinfo.ini')
    scores = raw[:, 6]
    if detector == 'DPM':
        scores = 1 / (1 + np.exp(-scores))
    elif ((scores < 0) | (scores > 1)).any():
        raise ValueError(f'{detector} scores are outside [0,1]')
    frames: dict[int, list] = {}
    for row, score in zip(raw, scores):
        x, y, w, h = row[2:6]
        frames.setdefault(int(row[0]), []).append(SimpleNamespace(
            box=(float(x), float(y), float(x + w), float(y + h)), score=float(score), label='person'))
    return frames


class ReplayCMC:
    """Feeds cached affines back to the tracker in order (one per step)."""

    uses_boxes = True

    def __init__(self, affines):
        self.affines = list(affines)
        self.i = 0

    def __call__(self, frame, boxes=None):
        a = self.affines[self.i]
        self.i += 1
        return a


def image_folder(folder: Path) -> Path:
    """MOT17 ships the frames once, under the FRCNN copy of each video."""
    own = folder / 'img1'
    return own if own.is_dir() else folder.parent / (folder.name.rsplit('-', 1)[0] + '-FRCNN') / 'img1'


def precompute(folder: Path, frames: dict, length: int, cache: Path, embedder, need_cmc: bool, need_reid: bool):
    """Cached per-step CMC affines [T, 2, 3] and per-detection embeddings (list per frame)."""
    import cv2
    from spatial.cmc import SparseFlowCMC
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(cache, allow_pickle=False)) if cache.exists() else {}
    want_cmc = need_cmc and 'affines' not in data
    want_reid = need_reid and 'emb' not in data
    if want_cmc or want_reid:
        cmc = SparseFlowCMC()
        affines, embs = [], []
        for f in range(1, length + 1):
            bgr = cv2.imread(str(image_folder(folder) / f'{f:06d}.jpg'))
            if bgr is None:
                raise FileNotFoundError(image_folder(folder) / f'{f:06d}.jpg')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes = [d.box for d in frames.get(f, [])]
            if want_cmc:
                affines.append(cmc(rgb, boxes))
            if want_reid:
                embs.append(embedder.embed(rgb, boxes))
        if want_cmc:
            data['affines'] = np.stack(affines)
        if want_reid:
            data['emb'] = np.concatenate([e for e in embs if len(e)] or [np.zeros((0, 512), np.float32)])
            data['emb_counts'] = np.array([len(e) for e in embs])
        np.savez(cache, **data)
    emb_per_frame = None
    if need_reid:
        ends = np.cumsum(data['emb_counts'])
        starts = ends - data['emb_counts']
        emb_per_frame = [data['emb'][a:b] for a, b in zip(starts, ends)]
    return (data.get('affines') if need_cmc else None), emb_per_frame


def track_sequence(folder: Path, output: Path, variant: str, cache_dir: Path, embedder,
                   appearance_thresh: float = 0.25, link_appearance_thresh: float | None = None):
    info = configparser.ConfigParser()
    info.read(folder / 'seqinfo.ini')
    length = info.getint('Sequence', 'seqLength')
    fps = info.getfloat('Sequence', 'frameRate')
    width, height = info.getint('Sequence', 'imWidth'), info.getint('Sequence', 'imHeight')
    frames = detection_rows(folder / 'det/det.txt', length, folder.name.rsplit('-', 1)[1])
    v = VARIANTS[variant]
    affines, embs = precompute(folder, frames, length, cache_dir / f'{folder.name}.npz', embedder,
                               v['cmc'], v['reid'])
    start = time.perf_counter()
    online = OnlineTracking(fps, {'appearance_thresh': appearance_thresh}, emit_birth=v['birth'])
    if v['cmc']:
        online.tracker.motion_compensator = ReplayCMC(affines)
    dummy = np.zeros((1, 1, 3), np.uint8)  # CMC replays cached affines; it only needs a frame to be given
    for f in range(1, length + 1):
        online.step(f - 1, dummy if v['cmc'] else None, frames.get(f, []),
                    embeddings=embs[f - 1] if v['reid'] else None, frame_size=(width, height))
    post = None
    if v['link'] or v['interp'] or v['minlen']:
        defaults = PostSettings()
        link = LinkSettings() if link_appearance_thresh is None else LinkSettings(appearance_thresh=link_appearance_thresh)
        post = PostSettings(link=link if v['link'] else LinkSettings(max_gap_s=0),
                            interpolate_max_gap_s=defaults.interpolate_max_gap_s if v['interp'] else 0,
                            min_track_len=defaults.min_track_len if v['minlen'] else 1)
    rows, stats = online.finish(fps, 1, post)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        for r in rows:
            x1, y1, x2, y2 = r['box']
            writer.writerow([r['frame'] + 1, r['track_id'], x1, y1, x2 - x1, y2 - y1, r['score'], -1, -1, -1])
    return dict(sequence=folder.name, frames=length, observations=len(rows), postprocess=stats,
                tracker_seconds=time.perf_counter() - start, detections_sha256=sha256(folder / 'det/det.txt'),
                predictions_sha256=sha256(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='MOT17 root with train/<seq>/{img1,det,gt}')
    parser.add_argument('--trackeval', type=Path, required=True, help='Pinned official TrackEval source checkout')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--detectors', nargs='+', choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--sequences', choices=('all', 'dev', 'heldout'), default='all',
                        help='all 7 videos, the 3 development videos, or the 4 held-out videos')
    parser.add_argument('--cache', type=Path, default=None, help='Where cached affines/embeddings go (default OUT/cache)')
    parser.add_argument('--reid', choices=('cnn', 'color'), default='cnn',
                        help='Appearance for the Re-ID variants: ResNet-18 (cnn) or HSV colour histograms (color); '
                             'cached per model')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help="Tracker's fused-cost appearance gate (track.py --appearance_thresh)")
    parser.add_argument('--link_appearance_thresh', type=float, default=None,
                        help="Offline linker's appearance gate (track.py --link_appearance_thresh; default its 0.3)")
    args = parser.parse_args()
    sys.path.insert(0, str(args.trackeval.resolve()))
    for name, builtin in (('float', float), ('int', int)):  # TrackEval 12c8791 uses removed NumPy aliases
        if name not in np.__dict__:
            setattr(np, name, builtin)
    import trackeval
    args.out.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache or args.out / 'cache'
    embedder = None
    if any(VARIANTS[v]['reid'] for v in args.variants):
        from spatial.reid import build_embedder
        embedder = build_embedder(args.reid, device=args.device)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                       text=True).strip()
    report_path = args.out / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else dict(
        dataset='MOT17 labelled training split (7 videos x 3 public detection streams)',
        code_revision=revision, trackeval=args.trackeval.name, variants=VARIANTS,
        reid=embedder.describe() if embedder else None,
        protocol='ByteTrack reference thresholds; track.py botsort defaults for CMC/Re-ID/post-processing; '
                 'no setting tuned on MOT17 ground truth', dpm_score_transform='sigmoid(raw margin)',
        limitations=['Association-only, public detections', 'Public training split, not challenge test',
                     'Variants share videos and detections (paired comparison)'], experiments={})
    for detector in args.detectors:
        for variant in args.variants:
            key = f'{variant}_{detector}' + ('' if args.sequences == 'all' else f'_{args.sequences}')
            if VARIANTS[variant]['reid']:
                key += '' if args.reid == 'cnn' else f'_{args.reid}'
                key += '' if args.appearance_thresh == 0.25 else f'_a{args.appearance_thresh:g}'
            if VARIANTS[variant]['link'] and args.link_appearance_thresh is not None:
                key += f'_l{args.link_appearance_thresh:g}'
            if key in report['experiments']:
                print(f'skip {key} (done)', flush=True)
                continue
            numbers = {'all': SEQUENCES, 'dev': DEV_SEQUENCES, 'heldout': HELDOUT_SEQUENCES}[args.sequences]
            sequences = [f'MOT17-{n}-{detector}' for n in numbers]
            observations = []
            for seq in sequences:
                print(f'Track {key}: {seq}', flush=True)
                observations.append(track_sequence(args.data / 'train' / seq,
                                                   args.out / 'trackers' / key / 'data' / f'{seq}.txt', variant,
                                                   cache_root / (detector if args.reid == 'cnn' else f'{detector}-{args.reid}'),
                                                   embedder, args.appearance_thresh, args.link_appearance_thresh))
            evaluator = trackeval.Evaluator(dict(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_ONLY_COMBINED=True,
                                                 PLOT_CURVES=False, PRINT_RESULTS=False,
                                                 LOG_ON_ERROR=str(args.out / 'evaluation-errors.log')))
            dataset = trackeval.datasets.MotChallenge2DBox(dict(
                GT_FOLDER=str((args.data / 'train').resolve()), TRACKERS_FOLDER=str((args.out / 'trackers').resolve()),
                OUTPUT_FOLDER=str((args.out / 'metrics').resolve()), TRACKERS_TO_EVAL=[key],
                SEQ_INFO={seq: None for seq in sequences}, SKIP_SPLIT_FOL=True, BENCHMARK='MOT17',
                SPLIT_TO_EVAL='train', DO_PREPROC=True, PRINT_CONFIG=False))
            results, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                                                               trackeval.metrics.Identity()])
            if messages[dataset.get_name()][key] != 'Success':
                raise RuntimeError(messages)
            combined = results[dataset.get_name()][key]['COMBINED_SEQ']['pedestrian']
            metrics = dict(HOTA=float(np.mean(combined['HOTA']['HOTA'])) * 100,
                           DetA=float(np.mean(combined['HOTA']['DetA'])) * 100,
                           AssA=float(np.mean(combined['HOTA']['AssA'])) * 100,
                           MOTA=float(combined['CLEAR']['MOTA']) * 100, IDF1=float(combined['Identity']['IDF1']) * 100,
                           IDSW=int(combined['CLEAR']['IDSW']), FP=int(combined['CLEAR']['CLR_FP']),
                           FN=int(combined['CLEAR']['CLR_FN']))
            report['experiments'][key] = dict(variant=variant, detector=detector, sequences_set=args.sequences,
                                              reid=args.reid if VARIANTS[variant]['reid'] else None,
                                              appearance_thresh=args.appearance_thresh,
                                              link_appearance_thresh=args.link_appearance_thresh,
                                              metrics=metrics,
                                              sequences=observations)
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'RESULT {key}: ' + ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}'
                                                 for k, v in metrics.items()), flush=True)


if __name__ == '__main__':
    main()

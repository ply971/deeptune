"""End-to-end MOT17: DeepTune's own detectors + trackers on raw frames, scored by TrackEval.

Unlike ``benchmark_mot17_botsort.py`` (public detections, association only),
this runs ``track.py``'s detectors (prompt ``person``) on the MOT17 frames and
then its trackers, with exactly the settings ``track.py`` resolves for each
``--detector`` / ``--tracker`` pair (``track.resolve_args``). It compares the
old default (``grounding_dino`` + ``bytetrack``) with the new one
(``auto`` -> ``dfine`` + ``botsort``) and the combinations in between.

Detections, appearance vectors and camera-motion affines are computed once per
(detector, video) and cached; tracker variants replay them. Videos: the
held-out set of ``benchmark_mot17_botsort.py`` by default (09, 10, 11, 13).
MOT17 counts only pedestrians; TrackEval's preprocessing removes boxes matched
to distractor classes (sitting people, reflections...). Public training split,
not the challenge test set; nothing is tuned on it.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import track as track_cli
from benchmark_mot17_botsort import DEV_SEQUENCES, HELDOUT_SEQUENCES, SEQUENCES, ReplayCMC, image_folder
from spatial.detect import Detection, build_detector
from spatial.pipeline import OnlineTracking

DETECTORS = ('dfine', 'grounding_dino', 'owlv2')
TRACKERS = ('bytetrack', 'botsort')


def resolved_args(detector: str, tracker: str):
    parser = track_cli.get_parser()
    args = parser.parse_args(['--video', 'x', '--prompt', 'person', '--out', 'x', '--detector', detector,
                              '--tracker', tracker, '--reid', 'cnn' if tracker == 'botsort' else 'none'])
    return track_cli.resolve_args(args, parser)


def detect_sequence(folder: Path, detector, embedder, cache: Path, batch: int):
    """Cached detections / embeddings / CMC affines for one video."""
    import cv2
    from spatial.cmc import SparseFlowCMC
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        dets_json = json.loads(str(data['detections']))
        dets = {int(k): [Detection.from_dict(d) for d in v] for k, v in dets_json.items()}
        ends = np.cumsum(data['emb_counts'])
        embs = [data['emb'][a:b] for a, b in zip(ends - data['emb_counts'], ends)]
        return dets, embs, data['affines'], float(data['detect_seconds'])
    info = configparser.ConfigParser()
    info.read(folder / 'seqinfo.ini')
    length = info.getint('Sequence', 'seqLength')
    cmc = SparseFlowCMC()
    dets, embs, affines = {}, [], []
    seconds = 0.0
    frames, idx = [], []

    def flush():
        nonlocal seconds
        if not frames:
            return
        t = time.perf_counter()
        results = detector.detect(frames, ['person'])
        seconds += time.perf_counter() - t
        for f, frame, res in zip(idx, frames, results):
            dets[f] = res
            boxes = [d.box for d in res]
            embs.append(embedder.embed(frame, boxes))
            affines.append(cmc(frame, boxes))
        frames.clear()
        idx.clear()

    for f in range(1, length + 1):
        bgr = cv2.imread(str(image_folder(folder) / f'{f:06d}.jpg'))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        idx.append(f)
        if len(frames) >= batch:
            flush()
    flush()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, detections=json.dumps({str(k): [d.to_dict() for d in v] for k, v in dets.items()}),
             emb=np.concatenate([e for e in embs if len(e)] or [np.zeros((0, 512), np.float32)]),
             emb_counts=np.array([len(e) for e in embs]), affines=np.stack(affines), detect_seconds=seconds)
    return dets, embs, np.stack(affines), seconds


def track_sequence(folder: Path, dets, embs, affines, args, output: Path):
    info = configparser.ConfigParser()
    info.read(folder / 'seqinfo.ini')
    length = info.getint('Sequence', 'seqLength')
    fps = info.getfloat('Sequence', 'frameRate')
    size = (info.getint('Sequence', 'imWidth'), info.getint('Sequence', 'imHeight'))
    use_reid = args.reid_resolved != 'none'
    online = OnlineTracking(fps, track_cli.tracker_kwargs(args), emit_birth=args.tracker == 'botsort')
    if args.cmc == 'sparse':
        online.tracker.motion_compensator = ReplayCMC(affines)
    dummy = np.zeros((1, 1, 3), np.uint8)
    for f in range(1, length + 1):
        online.step(f - 1, dummy if args.cmc == 'sparse' else None, dets.get(f, []),
                    embeddings=embs[f - 1] if use_reid else None, frame_size=size)
    rows, stats = online.finish(fps, 1, track_cli.post_settings(args))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        for r in rows:
            x1, y1, x2, y2 = r['box']
            writer.writerow([r['frame'] + 1, r['track_id'], x1, y1, x2 - x1, y2 - y1, r['score'], -1, -1, -1])
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--trackeval', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--detectors', nargs='+', choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument('--trackers', nargs='+', choices=TRACKERS, default=list(TRACKERS))
    parser.add_argument('--sequences', choices=('all', 'dev', 'heldout'), default='heldout')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    sys.path.insert(0, str(args.trackeval.resolve()))
    for name, builtin in (('float', float), ('int', int)):
        if name not in np.__dict__:
            setattr(np, name, builtin)
    import trackeval
    from spatial.reid import CnnEmbedder
    embedder = CnnEmbedder(device=args.device)
    numbers = {'all': SEQUENCES, 'dev': DEV_SEQUENCES, 'heldout': HELDOUT_SEQUENCES}[args.sequences]
    seqs = [f'MOT17-{n}-FRCNN' for n in numbers]  # the GT is the same in every detector copy
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else dict(
        dataset=f'MOT17 labelled training split, {args.sequences} videos {list(numbers)}', prompt='person',
        protocol='track.py resolved settings per detector/tracker; nothing tuned on MOT17', experiments={})
    for name in args.detectors:
        detector = None
        cached = {}
        for seq in seqs:
            cache = args.out / 'cache' / name / f'{seq}.npz'
            if not cache.exists() and detector is None:
                base = resolved_args(name, 'botsort')
                kw = dict(box_threshold=base.box_threshold, batch_size=args.batch_size, nms_iou=base.nms_iou,
                          device=args.device)
                if name == 'grounding_dino':
                    kw['text_threshold'] = base.text_threshold
                detector = build_detector(name, base.detector_model, **kw)
            print(f'Detect {name}: {seq}', flush=True)
            cached[seq] = detect_sequence(args.data / 'train' / seq, detector, embedder, cache, args.batch_size)
        del detector
        for tracker in args.trackers:
            key = f'{name}_{tracker}'
            if key in report['experiments']:
                continue
            targs = resolved_args(name, tracker)
            stats = {}
            for seq in seqs:
                dets, embs, affines, _ = cached[seq]
                stats[seq] = track_sequence(args.data / 'train' / seq, dets, embs, affines, targs,
                                            args.out / 'trackers' / key / 'data' / f'{seq}.txt')
            evaluator = trackeval.Evaluator(dict(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_ONLY_COMBINED=True,
                                                 PLOT_CURVES=False, PRINT_RESULTS=False,
                                                 LOG_ON_ERROR=str(args.out / 'evaluation-errors.log')))
            dataset = trackeval.datasets.MotChallenge2DBox(dict(
                GT_FOLDER=str((args.data / 'train').resolve()), TRACKERS_FOLDER=str((args.out / 'trackers').resolve()),
                OUTPUT_FOLDER=str((args.out / 'metrics').resolve()), TRACKERS_TO_EVAL=[key],
                SEQ_INFO={s: None for s in seqs}, SKIP_SPLIT_FOL=True, BENCHMARK='MOT17', SPLIT_TO_EVAL='train',
                DO_PREPROC=True, PRINT_CONFIG=False))
            results, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                                                               trackeval.metrics.Identity()])
            if messages[dataset.get_name()][key] != 'Success':
                raise RuntimeError(messages)
            c = results[dataset.get_name()][key]['COMBINED_SEQ']['pedestrian']
            metrics = dict(HOTA=float(np.mean(c['HOTA']['HOTA'])) * 100, DetA=float(np.mean(c['HOTA']['DetA'])) * 100,
                           AssA=float(np.mean(c['HOTA']['AssA'])) * 100, MOTA=float(c['CLEAR']['MOTA']) * 100,
                           IDF1=float(c['Identity']['IDF1']) * 100, IDSW=int(c['CLEAR']['IDSW']),
                           FP=int(c['CLEAR']['CLR_FP']), FN=int(c['CLEAR']['CLR_FN']))
            frames = sum(len(cached[s][1]) for s in seqs)
            seconds = sum(cached[s][3] for s in seqs)
            report['experiments'][key] = dict(detector=name, tracker=tracker, metrics=metrics, postprocess=stats,
                                              detector_fps=frames / seconds if seconds else None,
                                              settings={k: v for k, v in vars(targs).items()
                                                        if k not in ('video', 'out', 'video_dir')})
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
            print(f'RESULT {key}: ' + ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}'
                                                 for k, v in metrics.items())
                  + f', detector_fps={frames / seconds:.1f}', flush=True)


if __name__ == '__main__':
    main()

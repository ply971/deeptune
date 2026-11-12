"""Phase 4 Part A on the Phase 2 untrimmed oracle: a trained clip model run as a timeline, scored by mAP@tIoU.

1. Generate the synthetic ``untrimmed`` task (benchmarks/synthetic_motion.py): long videos with background
   stretches and 1..k actions (blob motion patterns) of known start/end.
2. Training clips are cut from the *train* videos only: windows of ``--window_sec`` that lie wholly inside an
   action (label = its class) or wholly inside background (label ``background``); the window length is the same
   at training and at inference. Columns ``videos, labels, group, start_sec, end_sec`` - the manifest format
   deeptune.py already reads (segment-level labels, Part A item 4).
3. ``deeptune.py`` trains the model (train/val/test split grouped by video), evaluates and embeds as usual.
4. ``infer.py --timeline`` on the held-out *test* videos with ``--ground_truth``: detection mAP at tIoU 0.1-0.9
   and boundary error. Two references on the same videos and windows: a **random** control (each window gets a
   random class, same smoothing and segmentation) and a **pixel oracle** (the stub that sees the red blob: perfect
   "is there an action" but it must still guess the class from the majority label).
5. ``--skip_static``: the same timeline with still windows skipped; speed-up and mAP change.
6. ``analysis/unsupervised.py`` on the run's test-split embeddings: do clips of the same class cluster?

Writes ``<out>/report.json`` and ``<out>/report.md``. Not tuned on the test videos.

    python scripts/timeline_untrimmed.py --out .artifacts/timeline-untrimmed --model r3d_18 --epochs 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def training_windows(meta: dict, root: Path, window: float, stride: float, background: str) -> pd.DataFrame:
    rows = []
    for vid, info in meta['videos'].items():
        if info['split'] != 'train':
            continue
        fps = meta['fps']
        spans = [(s['start_frame'] / fps, s['end_frame'] / fps, s['label']) for s in info['segments']]
        spans += [(a / fps, b / fps, background) for a, b in info['background_intervals']]
        for a, b, label in spans:
            t = a
            while t + window <= b + 1e-9:
                rows.append({'videos': str(root / info['file']), 'labels': label, 'group': vid,
                             'start_sec': round(t, 4), 'end_sec': round(t + window, 4)})
                t += stride
    return pd.DataFrame(rows)


def random_control(test_videos, out, classes, settings, gt, seed):
    from analysis.timeline import run_video_timelines
    rng = np.random.default_rng(seed)

    def score_rows(frame):
        p = rng.random((len(frame), len(classes)))
        return p / p.sum(axis=1, keepdims=True)
    return run_video_timelines(test_videos, out, classes, score_rows, settings, ground_truth=gt, log=lambda *_: None)


def pixel_oracle(test_videos, out, classes, settings, gt, majority):
    """Sees the blob in the pixels (perfect action/background) but not the motion: always says the majority class."""
    from analysis.timeline import run_video_timelines
    from benchmarks.render import BLOB_REDNESS_THRESHOLD, redness
    from datasets.video_io import VideoReader

    def score_rows(frame):
        out_rows = []
        for row in frame.itertuples():
            with VideoReader(row.videos) as reader:
                a, b = reader.time_to_index(row.start_sec), reader.time_to_end_index(row.end_sec)
                frames = reader.get_frames(list(range(a, b)))
            share = float(((redness(frames) > BLOB_REDNESS_THRESHOLD).sum(axis=(1, 2)) > 2).mean())
            p = np.zeros(len(classes))
            p[classes.index('background')] = 1 - share
            p[classes.index(majority)] = share
            out_rows.append(p)
        return np.array(out_rows)
    return run_video_timelines(test_videos, out, classes, score_rows, settings, ground_truth=gt, log=lambda *_: None)


def brief(record):
    m = record['metrics']
    return {'map@0.3': m['detection_map']['map_per_threshold']['0.3'], 'map@0.5': m['detection_map']['map_per_threshold']['0.5'],
            'map@0.7': m['detection_map']['map_per_threshold']['0.7'], 'avg_map_0.1_0.7': m['detection_map']['averages']['avg_0.1_0.7'],
            'boundary_error_s': m['boundary_error']['mean_seconds'], 'matched@0.5': m['boundary_error']['n_matched'],
            'ground_truth': m['detection_map']['n_ground_truth'], 'predicted_segments': m['detection_map']['n_predictions'],
            'windows': record['total_windows'], 'scored_windows': record['scored_windows'], 'seconds': record['seconds']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='r3d_18', help='deeptune.py --model_version (default r3d_18).')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_frames', type=int, default=8)
    p.add_argument('--n_videos', type=int, default=40)
    p.add_argument('--n_classes', type=int, default=3)
    p.add_argument('--window_sec', type=float, default=1.6)
    p.add_argument('--stride_sec', type=float, default=0.4)
    p.add_argument('--train_stride_sec', type=float, default=0.4)
    p.add_argument('--smooth_windows', type=int, default=3)
    p.add_argument('--min_segment_sec', type=float, default=0.8)
    p.add_argument('--skip_static', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='More deeptune.py flags.')
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from benchmarks.synthetic_motion import make_untrimmed_dataset
    data_dir = args.out / 'data'
    if (data_dir / 'dataset.json').exists():
        meta = json.loads((data_dir / 'dataset.json').read_text(encoding='utf-8'))
    else:
        make_untrimmed_dataset(data_dir, n_videos=args.n_videos, n_classes=args.n_classes, seed=args.seed)
        meta = json.loads((data_dir / 'dataset.json').read_text(encoding='utf-8'))
    classes = ['background'] + list(meta['classes'])
    clips = training_windows(meta, data_dir, args.window_sec, args.train_stride_sec, 'background')
    manifest = args.out / 'train_windows.csv'
    clips.to_csv(manifest, index=False)
    print(f'{len(clips)} training windows from {clips["group"].nunique()} train videos: '
          + ', '.join(f'{k}={v}' for k, v in clips['labels'].value_counts().items()), flush=True)

    import deeptune
    runs = args.out / 'runs'
    run = next(runs.glob('deeptune-*'), None) if runs.exists() else None
    t0 = time.perf_counter()
    if run is None:
        deeptune.main(['--modality', 'video', '--df', str(manifest), '--raw-data', '--out', str(runs),
                       '--model_version', args.model, '--num_epochs', str(args.epochs), '--batch_size', str(args.batch_size),
                       '--num_frames', str(args.num_frames), '--grouper', 'group', '--fixed-seed', *args.extra])
        run = next(runs.glob('deeptune-*'))
    train_seconds = time.perf_counter() - t0

    test_videos = [str(data_dir / info['file']) for vid, info in meta['videos'].items() if info['split'] == 'test']
    test_list = args.out / 'test_videos.csv'
    pd.DataFrame({'videos': test_videos}).to_csv(test_list, index=False)
    gt = data_dir / 'gt_segments.csv'
    from desktop.config import build_infer_argv, infer_config_from_run, read_run_results
    import infer
    timeline = ['--timeline', '--window_sec', str(args.window_sec), '--stride_sec', str(args.stride_sec),
                '--smooth_windows', str(args.smooth_windows), '--min_segment_sec', str(args.min_segment_sec),
                '--background_label', 'background', '--ground_truth', str(gt)]
    results = {}
    for name, extra in (('model', []), ('model_skip_static', ['--skip_static', str(args.skip_static), '--timeline_motion', 'framediff'])):
        out = args.out / name
        argv = build_infer_argv(infer_config_from_run(read_run_results(run), test_list, out)) + timeline + extra
        assert infer.main(argv) == 0
        results[name] = brief(json.loads((out / 'timeline_run.json').read_text(encoding='utf-8')))
    settings = dict(window_sec=args.window_sec, stride_sec=args.stride_sec, smooth_windows=args.smooth_windows,
                    min_segment_sec=args.min_segment_sec, background=['background'])
    results['random_control'] = brief(random_control(test_videos, args.out / 'random', classes, settings, gt, args.seed))
    train_labels = clips.loc[clips['labels'] != 'background', 'labels']
    majority = train_labels.value_counts().idxmax()
    results['pixel_oracle_majority_class'] = brief(pixel_oracle(test_videos, args.out / 'oracle', classes, settings, gt, majority))

    clip_eval = json.loads(next(run.glob('eval_output_*/*.json')).read_text(encoding='utf-8')) if list(run.glob('eval_output_*/*.json')) else None
    unsup = None
    emb = next(run.glob('embed_output_*/*.parquet'), None)
    if emb is not None:
        from analysis import unsupervised
        unsupervised.main(['--embeddings', str(emb), '--out', str(args.out / 'unsupervised'), '--k_max', str(len(classes) + 3),
                           '--ae_epochs', '100'])
        u = json.loads((args.out / 'unsupervised' / 'summary.json').read_text(encoding='utf-8'))
        unsup = {'rows': u['rows'], 'suggested_k': u['kmeans']['suggested_k_silhouette'], 'elbow_k': u['kmeans']['elbow_k'],
                 'kmeans_ari': u['kmeans'].get('agreement', {}).get('ari'), 'kmeans_nmi': u['kmeans'].get('agreement', {}).get('nmi'),
                 'agglomerative_ari': u['agglomerative'].get('agreement', {}).get('ari'),
                 'pca_first_two': u['pca']['first_two'], 'lda_accuracy': u.get('lda', {}).get('training_accuracy')}
    report = {'task': 'synthetic untrimmed (Phase 2)', 'classes': classes, 'n_videos': args.n_videos,
              'test_videos': len(test_videos), 'model': args.model, 'epochs': args.epochs, 'run': str(run),
              'training_windows': int(len(clips)), 'train_seconds': round(train_seconds, 1), 'settings': settings,
              'skip_static_threshold': args.skip_static, 'results': results, 'clip_evaluation': clip_eval,
              'unsupervised_test_embeddings': unsup, 'pixel_oracle_class': majority}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    lines = [f'# Timeline on the synthetic untrimmed task ({args.model}, {args.epochs} epochs)', '',
             f'{len(test_videos)} held-out videos, {report["results"]["model"]["ground_truth"]} actions; window {args.window_sec} s, '
             f'stride {args.stride_sec} s, median over {args.smooth_windows}, min segment {args.min_segment_sec} s.', '',
             '| Method | mAP@0.3 | mAP@0.5 | mAP@0.7 | avg 0.1-0.7 | boundary error (s) | windows scored | seconds |',
             '|---|---|---|---|---|---|---|---|']
    for name, r in results.items():
        be = '-' if r['boundary_error_s'] is None else f'{r["boundary_error_s"]:.2f}'
        lines.append(f'| {name} | {r["map@0.3"]:.3f} | {r["map@0.5"]:.3f} | {r["map@0.7"]:.3f} | {r["avg_map_0.1_0.7"]:.3f} | '
                     f'{be} | {r["scored_windows"]}/{r["windows"]} | {r["seconds"]:.1f} |')
    if unsup:
        lines += ['', f'Unsupervised on the test-split clip embeddings ({unsup["rows"]} clips): K-Means ARI {unsup["kmeans_ari"]}, '
                  f'NMI {unsup["kmeans_nmi"]}, silhouette k = {unsup["suggested_k"]}, elbow k = {unsup["elbow_k"]}.']
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())

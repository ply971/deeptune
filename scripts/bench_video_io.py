"""Benchmark the video data layer (Phase 1): decode backends, old vs new
dataset, codec motion vectors vs Farneback, DataLoader worker start-up.

Runs on CPU by default (no GPU work at all). Example (from the repo root):

    python scripts/bench_video_io.py --clips 200
    python scripts/bench_video_io.py --clips 20 --workers 0 1 2 --json .artifacts/bench/io.json
    python scripts/bench_video_io.py --real "C:/data/ucf/*.avi" --nvdec

Generated clips are H.264 (x264 defaults, with B-frames) and MPEG-4 Part 2,
``--frames`` long at ``--size``x``--size``; ``--real`` adds files from a glob
(default: the repo's 1080p H.264 demo videos, which exercise the seek path).
Prints a Markdown table; ``--json`` also writes the raw numbers.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
import pandas as pd


def legacy_sample_frames(video_bytes: bytes, num_frames: int):
    """The pre-Phase-1 ParquetVideoDataset path: temp file + OpenCV CAP_PROP_POS_FRAMES seeks."""
    import cv2
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        name = tmp.name
    try:
        cap = cv2.VideoCapture(name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = []
        for idx in np.linspace(0, total - 1, num_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else out[-1])
        cap.release()
        return out
    finally:
        os.unlink(name)


def timed(fn, items):
    start = time.perf_counter()
    for item in items:
        fn(item)
    return time.perf_counter() - start


def rate(count, seconds):
    return count / seconds if seconds > 0 else float('inf')


def make_clips(folder: Path, count: int, frames: int, size: int):
    from datasets.video_synth import translating_clip
    h264, mpeg4 = [], []
    for i in range(count):
        dx, dy = (i % 5) - 2, (i % 3) - 1
        h264.append(translating_clip(folder / f'h264_{i:04}.mp4', frames, dx, dy, size=size, seed=i))
        if i < max(1, count // 4):
            mpeg4.append(translating_clip(folder / f'mpeg4_{i:04}.mp4', frames, dx, dy, size=size, seed=i, codec='mpeg4'))
    return h264, mpeg4


def bench_backends(paths, num_frames, rows, label):
    from datasets.video_io import VideoReader, available_backends
    found = available_backends()
    for backend in ('torchcodec', 'pyav', 'opencv'):
        if backend not in found:
            rows.append((f'decode {label}', backend, 'n/a (not installed)', ''))
            continue

        def read(path):
            with VideoReader(path, backend=backend) as reader:
                count = reader.metadata().frame_count
                reader.get_frames(np.linspace(0, count - 1, num_frames).astype(int))

        seconds = timed(read, paths)
        rows.append((f'decode {label}', backend, f'{rate(len(paths), seconds):.1f} clips/s',
                     f'{1000 * seconds / len(paths):.1f} ms/clip, {num_frames} frames uniform'))


def bench_nvdec(paths, num_frames, rows):
    try:
        import av
        from av.codec.hwaccel import HWAccel
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        accel = HWAccel(device_type='cuda', allow_software_fallback=False)

        def read(path):
            with av.open(str(path), hwaccel=accel) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i == 0:
                        frame.to_ndarray(format='rgb24')
        seconds = timed(read, paths)
        rows.append(('decode generated (all frames)', 'pyav + NVDEC', f'{rate(len(paths), seconds):.1f} clips/s',
                     f'{1000 * seconds / len(paths):.1f} ms/clip'))
    except Exception as exc:
        rows.append(('decode generated (all frames)', 'pyav + NVDEC', 'n/a', f'{type(exc).__name__}: {exc}'[:80]))


def bench_datasets(paths, num_frames, rows):
    from datasets.video_datasets import ParquetVideoDataset, VideoDataset
    blobs = [Path(p).read_bytes() for p in paths]
    labels = [i % 2 for i in range(len(paths))]
    seconds = timed(lambda b: legacy_sample_frames(b, num_frames), blobs)
    rows.append(('dataset __getitem__', 'old: bytes + temp file + OpenCV seek', f'{rate(len(blobs), seconds):.1f} clips/s',
                 f'{1000 * seconds / len(blobs):.1f} ms/clip'))
    start = time.perf_counter()
    in_memory = ParquetVideoDataset(pd.DataFrame({'videos': blobs, 'labels': labels}), num_frames=num_frames)
    built = time.perf_counter() - start
    seconds = timed(in_memory.__getitem__, range(len(in_memory)))
    pickled = len(pickle.dumps(in_memory))
    rows.append(('dataset __getitem__', 'new: bytes, PyAV in memory', f'{rate(len(blobs), seconds):.1f} clips/s',
                 f'{1000 * seconds / len(blobs):.1f} ms/clip; worker pickle {pickled / 1e6:.2f} MB before spill'))
    in_memory.prepare_for_workers()
    rows.append(('worker pickle size', 'new: bytes after prepare_for_workers', f'{len(pickle.dumps(in_memory)) / 1e3:.1f} kB',
                 'memory-mapped Arrow file'))
    start = time.perf_counter()
    by_path = VideoDataset(pd.DataFrame({'videos': [str(p) for p in paths], 'labels': labels}), num_frames=num_frames)
    built = time.perf_counter() - start
    seconds = timed(by_path.__getitem__, range(len(by_path)))
    rows.append(('dataset __getitem__', 'new: paths, PyAV', f'{rate(len(paths), seconds):.1f} clips/s',
                 f'{1000 * seconds / len(paths):.1f} ms/clip; construct {1000 * built:.1f} ms; '
                 f'pickle {len(pickle.dumps(by_path)) / 1e3:.1f} kB'))
    return by_path


def bench_motion(h264, mpeg4, num_frames, rows):
    from datasets.video_io import VideoReader
    for label, paths in (('H.264', h264), ('MPEG-4', mpeg4)):
        for method in ('codec', 'farneback'):
            def run(path):
                with VideoReader(path) as reader:
                    count = reader.metadata().frame_count
                    reader.get_motion_vectors(np.linspace(1, count - 1, num_frames).astype(int), method=method)
            seconds = timed(run, paths)
            rows.append((f'motion field ({num_frames} frames)', f'{method} / {label}',
                         f'{1000 * seconds / len(paths):.1f} ms/clip', f'{len(paths)} clips'))
        for method in ('codec', 'framediff'):
            seconds = timed(lambda p: VideoReader(p).motion_curve(method), paths)
            rows.append(('motion curve (whole clip, MGSampler)', f'{method} / {label}',
                         f'{1000 * seconds / len(paths):.1f} ms/clip', ''))


def bench_workers(dataset, workers, rows):
    from torch.utils.data import DataLoader
    from datasets.video_datasets import collate_video_samples
    for count in workers:
        loader = DataLoader(dataset, batch_size=4, num_workers=count, collate_fn=collate_video_samples,
                            persistent_workers=count > 0)
        start = time.perf_counter()
        iterator = iter(loader)
        next(iterator)
        first = time.perf_counter() - start
        for _ in iterator:
            pass
        total = time.perf_counter() - start
        rows.append(('DataLoader epoch (paths)', f'num_workers={count}', f'{total:.2f} s',
                     f'first batch after {first:.2f} s (worker start-up); {len(dataset)} clips'))
        del iterator, loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clips', type=int, default=200)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--size', type=int, default=112)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--real', default=str(ROOT / 'docs/source/_static/*.mp4'),
                        help='Glob of real videos (default: the repo demo videos, 1080p H.264). Empty to skip.')
    parser.add_argument('--real_limit', type=int, default=4)
    parser.add_argument('--workers', type=int, nargs='*', default=[0, 1, 2])
    parser.add_argument('--nvdec', action='store_true', help='Also try PyAV NVDEC decoding (needs CUDA).')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    if not args.nvdec:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

    import av
    import cv2
    import torch
    from datasets.video_io import available_backends
    rows = []
    env = {'python': sys.version.split()[0], 'torch': torch.__version__, 'av': av.__version__, 'opencv': cv2.__version__,
           'backends': available_backends(), 'cpu_count': os.cpu_count(), 'platform': sys.platform,
           'clips': args.clips, 'frames': args.frames, 'size': args.size}
    with tempfile.TemporaryDirectory(prefix='deeptune-bench-') as folder:
        start = time.perf_counter()
        h264, mpeg4 = make_clips(Path(folder), args.clips, args.frames, args.size)
        env['generate_seconds'] = round(time.perf_counter() - start, 2)
        bench_backends(h264, args.num_frames, rows, f'generated H.264 {args.size}px x{args.frames}')
        real = sorted(glob.glob(args.real))[:args.real_limit] if args.real else []
        if real:
            bench_backends(real, args.num_frames, rows, f'real files ({len(real)}, e.g. {Path(real[0]).name})')
        if args.nvdec:
            bench_nvdec(h264, args.num_frames, rows)
        else:
            rows.append(('decode generated', 'pyav + NVDEC', 'n/a', 'not run (pass --nvdec on a CUDA machine)'))
        dataset = bench_datasets(h264, args.num_frames, rows)
        bench_motion(h264[:max(1, len(h264) // 2)], mpeg4, args.num_frames, rows)
        if args.workers:
            bench_workers(dataset, args.workers, rows)
        rows.append(('epoch from cached snippet features', 'n/a', 'n/a until B1', 'feature cache is Phase 1 task 8'))
    print('\n' + ' | '.join(f'{k}={v}' for k, v in env.items()))
    print('\n| Measurement | Variant | Result | Notes |\n|---|---|---|---|')
    for row in rows:
        print('| ' + ' | '.join(str(c) for c in row) + ' |')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({'env': env, 'rows': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

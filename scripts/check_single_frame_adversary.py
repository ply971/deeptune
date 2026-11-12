"""Fine-tuned single_frame:resnet18 as the strongest single-frame adversary (Critic V-022, V-038).

Trains it on direction, speed and periodic (700/200/400, 64 px, 5 epochs, seed 0), on the default codec and on
truly lossless clips, so codec artefacts can be compared with none. Controls on the same settings:
  flow_probe            motion is learnable on direction and speed
  r3d_18                motion is learnable on periodic (flow_probe only sees mean flow, which cannot separate it)
  scene_bias            this adversary does learn when a single frame carries signal (the biased background)

Lossless clips are FFV1 in Matroska under a .mp4 name, because OpenCV silently writes H.264 for an FFV1 FourCC.
Every clip is decoded back through VideoReader and must equal its source frames exactly.

GPU job, one training at a time, about 20 minutes. A finished run is skipped, so it can be re-run after an interruption.
    python scripts/check_single_frame_adversary.py
"""
import argparse
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = ('direction', 'speed', 'periodic')
SIZES = dict(n_train=700, n_val=200, n_test=400, size=64, seed=0)
ADVERSARY = 'single_frame:resnet18'


def write_ffv1_verified(path, frames_rgb, fps, fourcc):
    import av
    import numpy as np
    from datasets.video_io import VideoReader
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_rgb = np.asarray(frames_rgb, dtype=np.uint8)
    with av.open(str(path), 'w', format='matroska') as container:
        stream = container.add_stream('ffv1', rate=int(fps))
        stream.width, stream.height = frames_rgb.shape[2], frames_rgb.shape[1]
        stream.pix_fmt = 'bgr0'
        stream.time_base = Fraction(1, int(fps))
        for i, frame in enumerate(frames_rgb):
            video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            video_frame.pts, video_frame.time_base = i, Fraction(1, int(fps))
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with VideoReader(path) as reader:
        back = reader.get_frames(list(range(len(frames_rgb))))
    if not np.array_equal(back, frames_rgb):
        raise RuntimeError(f'{path} is not lossless: max error {np.abs(back.astype(int) - frames_rgb.astype(int)).max()}')


def generate_lossless(task, out):
    import benchmarks.synthetic_motion as sm
    root = out / 'data_lossless' / task
    if (root / 'dataset.json').is_file():
        return root
    sm.write_mp4 = write_ffv1_verified
    make = {'direction': sm.make_direction_dataset, 'speed': sm.make_speed_dataset, 'periodic': sm.make_periodic_dataset}[task]
    start = time.time()
    make(root, num_frames={'periodic': 16}.get(task, 12), codec='FFV1', **SIZES)
    print(f'[gen] {task}: lossless clips verified frame by frame in {time.time() - start:.0f}s', flush=True)
    return root


def run(task, variant, data, models, out, name=None):
    run_out = out / variant / (name or task)
    if list(run_out.rglob('results.md')):
        print(f'[skip] {variant}/{name or task}: already done', flush=True)
        return 0
    run_out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / 'scripts' / 'run_motion_benchmark.py'), '--task', task, '--models', *models,
           '--n_train', '700', '--n_val', '200', '--n_test', '400', '--size', '64', '--epochs', '5', '--seed', '0',
           '--out', str(run_out)]
    if data is not None:
        cmd += ['--data', str(data)]
    start = time.time()
    with open(run_out / 'run.log', 'w', encoding='utf-8') as log:
        code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    print(f'[run] {variant}/{name or task}: exit {code} in {time.time() - start:.0f}s', flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / '.artifacts' / 'gpu-checks' / 'singleframe')
    out = parser.parse_args().out.resolve()
    failed = 0
    for task in TASKS:
        failed += run(task, 'default_codec', None, (ADVERSARY, 'flow_probe'), out) != 0
    for task in TASKS:
        failed += run(task, 'lossless', generate_lossless(task, out), (ADVERSARY, 'flow_probe'), out) != 0
    default_periodic = out / 'default_codec' / 'periodic' / 'data' / 'periodic_s64_n700-200-400_seed0'
    failed += run('periodic', 'default_codec', default_periodic, ('r3d_18',), out, 'periodic_control') != 0
    failed += run('periodic', 'lossless', generate_lossless('periodic', out), ('r3d_18',), out, 'periodic_control') != 0
    failed += run('scene_bias', 'default_codec', None, (ADVERSARY,), out, 'scene_bias_control') != 0
    print('ALL OK' if not failed else f'{failed} run(s) failed', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()

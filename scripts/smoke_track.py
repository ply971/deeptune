"""Zero-shot detection + tracking smoke run with REAL pretrained weights.

Downloads the default checkpoints on first use (Grounding DINO tiny ~0.7 GB,
OWLv2 base ~0.6 GB, both Apache-2.0) into the Hugging Face cache, so it is not
part of the unit tests. Uses the GPU when present.

Without ``--video`` it renders a synthetic clip (a red square and a blue circle
moving across a grey background) and prompts for "red square" / "blue circle".
It checks plumbing (every output file exists and parses, IDs are integers, boxes
lie inside the frame) and *reports* the counts; ``--expect_count N`` turns the
count into a pass/fail check. Real accuracy is measured later with S4 metrics.

Run from the repository root:
    python scripts/smoke_track.py                       # both detectors, synthetic clip
    python scripts/smoke_track.py --detector owlv2
    python scripts/smoke_track.py --video lobsters.mp4 --prompt lobster --save_video
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def synthetic_clip(path: Path, n_frames=48, size=(320, 240), fps=12.0) -> Path:
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in range(n_frames):
        img = np.full((h, w, 3), 170, np.uint8)
        x = int(20 + f * (w - 100) / n_frames)
        cv2.rectangle(img, (x, 40), (x + 60, 100), (0, 0, 220), -1)                        # red square
        cv2.circle(img, (int(w - 50 - f * (w - 100) / n_frames), 170), 32, (220, 60, 0), -1)  # blue circle
        out.write(img)
    out.release()
    return path


def check_outputs(vdir: Path, width: int, height: int) -> dict:
    from spatial.io import read_mot
    for name in ('detections.json', 'tracks.csv', 'track_labels.csv', 'summary.json'):
        if not (vdir / name).is_file():
            raise AssertionError(f'missing {name}')
    json.loads((vdir / 'detections.json').read_text())
    rows = read_mot(vdir / 'tracks.csv')
    for r in rows:
        x1, y1, x2, y2 = r['box']
        if not (-1 <= x1 < x2 <= width + 1 and -1 <= y1 < y2 <= height + 1):
            raise AssertionError(f'box outside frame: {r}')
    return json.loads((vdir / 'summary.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--detector', choices=['grounding_dino', 'owlv2', 'both'], default='both')
    p.add_argument('--video', type=Path, default=None)
    p.add_argument('--prompt', action='append', default=None)
    p.add_argument('--out', type=Path, default=Path('.artifacts/track-smoke'))
    p.add_argument('--every_n_frames', type=int, default=1)
    p.add_argument('--save_video', action='store_true')
    p.add_argument('--expect_count', type=int, default=None,
                   help='Fail unless every prompt has exactly this many tracks.')
    args = p.parse_args()

    import track
    detectors = ['grounding_dino', 'owlv2'] if args.detector == 'both' else [args.detector]
    failed = False
    with tempfile.TemporaryDirectory(prefix='deeptune-track-') as tmp:
        video = args.video or synthetic_clip(Path(tmp) / 'shapes.mp4')
        prompts = args.prompt or (['red square', 'blue circle'] if args.video is None else None)
        if not prompts:
            p.error('--prompt is required with --video')
        cap = cv2.VideoCapture(str(video))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        for name in detectors:
            out = args.out / name
            argv = ['--video', str(video), '--out', str(out), '--detector', name,
                    '--every_n_frames', str(args.every_n_frames)]
            for q in prompts:
                argv += ['--prompt', q]
            if args.save_video:
                argv.append('--save_video')
            t0 = time.time()
            try:
                track.main(argv)
                summary = check_outputs(out / Path(video).stem, width, height)
                counts = {k: v['count'] for k, v in summary['labels'].items()}
                status = 'ok'
                if args.expect_count is not None and any(c != args.expect_count for c in counts.values()):
                    status = f'COUNT MISMATCH (expected {args.expect_count})'
                    failed = True
                print(f'SMOKE {name}: {status} in {time.time() - t0:.1f}s, counts={counts}, out={out}')
            except Exception as exc:
                failed = True
                print(f'SMOKE {name}: FAIL {type(exc).__name__}: {exc}')
    print('SMOKE TRACK', 'FAIL' if failed else 'PASS')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()

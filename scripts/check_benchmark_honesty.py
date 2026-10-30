"""Generate the Phase 2 benchmarks at a chosen size and print every honesty-probe
number (the same probes the unit tests assert on, at CI size).

Example (the course-slide configuration, 700 train / 400 test, 12 frames):
    python scripts/check_benchmark_honesty.py --size 64 --n_train 700 --n_test 400 --out .artifacts/honesty
CPU only; flow probes cost about 15 ms per 12-frame 32x32 clip.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.probes import codec_mv_probe, direction_probes, periodic_probes, scene_bias_probes, speed_probes  # noqa: E402
from benchmarks.synthetic_motion import (  # noqa: E402
    make_direction_dataset,
    make_periodic_dataset,
    make_scene_bias_dataset,
    make_speed_dataset,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(".artifacts/honesty"))
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--mv_size", type=int, default=48, help="Frame size for the codec motion-vector probe.")
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--num_frames", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--tasks", nargs="+", default=["direction", "codec_mv", "speed", "periodic", "scene_bias"])
    args = p.parse_args()
    common = dict(n_train=args.n_train, n_test=args.n_test, seed=args.seed, codec=args.codec)
    report = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    for task in args.tasks:
        start = time.perf_counter()
        out = args.out / task
        if task == "direction":
            result = direction_probes(make_direction_dataset(out, num_frames=args.num_frames, size=args.size, **common),
                                      codec_mv=False)
        elif task == "codec_mv":
            result = codec_mv_probe(make_direction_dataset(out, num_frames=args.num_frames, size=args.mv_size, **common))
        elif task == "speed":
            result = speed_probes(make_speed_dataset(out, num_frames=args.num_frames, size=args.size, **common))
        elif task == "periodic":
            result = periodic_probes(make_periodic_dataset(out, num_frames=max(args.num_frames, 16), size=args.size,
                                                           **common))
        elif task == "scene_bias":
            result = scene_bias_probes(make_scene_bias_dataset(out, n_val=args.n_val, num_frames=args.num_frames,
                                                               size=args.size, **common))
        else:
            raise SystemExit(f"unknown task {task}")
        report[task] = result
        report[task + "_seconds"] = round(time.perf_counter() - start, 1)
        print(f"{task}: {json.dumps(result)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "honesty.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out / 'honesty.json'}")


if __name__ == "__main__":
    main()

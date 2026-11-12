"""Render the figures of docs/video-plan/benchmark.md from freshly generated,
seeded benchmark data (decoded from the MP4s, so the figures show what a model
actually reads).

    python scripts/make_benchmark_figures.py --out docs/video-plan/img
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from benchmarks.motion_features import read_frames  # noqa: E402
from benchmarks.render import BLOB_REDNESS_THRESHOLD, redness  # noqa: E402
from benchmarks.synthetic_motion import (  # noqa: E402
    make_direction_dataset,
    make_periodic_dataset,
    make_scene_bias_dataset,
    make_speed_dataset,
    make_untrimmed_dataset,
)

SCALE = 1  # native 64x64; the docs page scales the <img> up


def _gif(frames, path, fps=8):
    # One adaptive 48-colour palette, no dithering: small files, faithful colours.
    images = [Image.fromarray(cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST))
              .quantize(colors=48, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)


def class_gifs(work, out, seed):
    made = []
    for name, maker, kwargs in (
        ("direction", make_direction_dataset, dict(num_frames=12)),
        ("periodic", make_periodic_dataset, dict(num_frames=16)),
        ("speed", make_speed_dataset, dict(num_frames=12)),
    ):
        ds = maker(work / name, n_train=8, n_test=0, size=64, seed=seed, codec="mp4v", **kwargs)
        manifest = ds.load_manifest()
        if name == "speed":
            picks = {"slow": manifest["labels"].idxmin(), "fast": manifest["labels"].idxmax()}
        else:
            picks = {label: manifest.index[manifest["labels"] == label][0] for label in ds.classes}
        for label, row in picks.items():
            path = out / f"{name}_{label}.gif"
            _gif(read_frames(ds.root / manifest.loc[row, "videos"]), path)
            made.append(path)
    return made


def untrimmed_timeline(work, out, seed):
    ds = make_untrimmed_dataset(work / "untrimmed", n_videos=4, length_range=(200, 240), n_classes=3, seed=seed,
                                size=48, background_motion="distractor", codec="mp4v")
    meta = ds.load_metadata()
    vid = max(meta["videos"], key=lambda v: len(meta["videos"][v]["segments"]))
    info = meta["videos"][vid]
    frames = read_frames(ds.root / info["file"])
    fps, n = meta["fps"], len(frames)
    colours = dict(zip(meta["classes"], ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.2), gridspec_kw=dict(height_ratios=[1.6, 0.6, 1.2]))
    picks = np.linspace(0, n - 1, 12).astype(int)
    axes[0].imshow(np.concatenate([np.pad(frames[i], ((2, 2), (2, 2), (0, 0)), constant_values=255) for i in picks], 1))
    axes[0].set_xticks([(k + 0.5) * (frames.shape[2] + 4) for k in range(len(picks))])
    axes[0].set_xticklabels([f"{i / fps:.1f}s" for i in picks], fontsize=8)
    axes[0].set_yticks([])
    axes[0].set_title(f"{vid}: {n} frames @ {fps:g} fps, grey distractor square moves all the time; "
                      f"video-level label = {'; '.join(info['labels'])}", fontsize=9)
    ax = axes[1]
    for start, end in info["background_intervals"]:
        ax.axvspan(start / fps, end / fps, color="0.85")
    for s in info["segments"]:
        ax.axvspan(s["t_start"], s["t_end"], color=colours[s["label"]], alpha=0.85)
        ax.text((s["t_start"] + s["t_end"]) / 2, 0.5, s["label"], ha="center", va="center", fontsize=8, color="white")
    ax.set_xlim(0, n / fps)
    ax.set_yticks([])
    ax.set_title("ground truth (video-id, t-start, t-end, label); grey = background only", fontsize=9)
    red = (redness(frames) > BLOB_REDNESS_THRESHOLD).sum(axis=(1, 2))
    axes[2].plot(np.arange(n) / fps, red, color="tab:red", lw=1)
    axes[2].set_xlim(0, n / fps)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("blob pixels")
    axes[2].set_title("blob pixels detected in the decoded MP4 (non-zero exactly inside the segments)", fontsize=9)
    fig.tight_layout()
    path = out / "untrimmed_timeline.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def scene_bias_examples(work, out, seed):
    ds = make_scene_bias_dataset(work / "scene_bias", n_train=24, n_val=0, n_test=8, num_frames=12, size=64,
                                 seed=seed, bias=0.9, codec="mp4v")
    manifest = ds.load_manifest()
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.8))
    train = manifest[manifest.split == "train"]
    for col, label in enumerate(ds.classes):
        rows = train[(train.labels == label) & train.family_matches_class].head(2)
        for k, (_, row) in enumerate(rows.iterrows()):
            ax = axes[0, col * 2 + k]
            ax.imshow(read_frames(ds.root / row.videos)[6])
            ax.set_title(f"train: {label}\n{row.background_family}", fontsize=8)
    test = manifest[manifest.split == "test"]
    for k, (_, row) in enumerate(test.drop_duplicates(["labels", "background_family"]).iterrows()):
        ax = axes[1, k]
        ax.imshow(read_frames(ds.root / row.videos)[6])
        ax.set_title(f"test: {row.labels}\n{row.background_family}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("scene_bias: in train ~90% of 'left' clips have dots and 'right' clips stripes;\n"
                 "in test every class x family pair is equally common", fontsize=9)
    fig.tight_layout()
    path = out / "scene_bias_examples.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("docs/video-plan/img"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deeptune-figs-") as folder:
        work = Path(folder)
        made = class_gifs(work, args.out, args.seed) + untrimmed_timeline(work, args.out, args.seed) \
            + scene_bias_examples(work, args.out, args.seed)
    for path in made:
        print(f"{path}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()

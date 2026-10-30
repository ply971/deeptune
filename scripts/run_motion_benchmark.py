"""Train and evaluate a list of video models on a Phase 2 synthetic benchmark
and write a results table (accuracy and ROC AUC, or R^2 and MAE for the speed
regression task; time; device).

Models are given as specs:
  single_frame:<backbone>   Karpathy single-frame baseline (middle frame only)
  mean:<backbone>           frame pooling, mean over frames (late fusion)
  attention:<backbone>      frame pooling, learned attention over frames
  <native 3D name>          e.g. r3d_18, mc3_18, r2plus1d_18 (slow fusion)
  flow_logreg / flow_probe  reference probe: mean horizontal Farneback flow -> logistic regression
                            (speed: flow summary -> ridge regression)
<backbone> is any 2D model version DeepTune knows (resnet18, efficientnet_b0, ...).
"dummy" (an untrained resnet18) and "dummy_video3d" (an untrained r3d_18) exist
for quick CPU smoke runs only.

Every neural model is built with utils.build_video_model, fed through the
existing manifest loader + ParquetVideoDataset, and trained with the shared
vision Trainer. Every generated benchmark has its own seeded val split
(disjoint scenes): the Trainer validates on it and fits its per-prediction
error model (error_model.json) on it; the test split is touched only by the
final evaluation. Horizontal flip augmentation (--hflip) is refused on
direction tasks.

Example (GPU, the course-slide size):
  python scripts/run_motion_benchmark.py --task direction --models single_frame:resnet18 mean:resnet18 \
      attention:resnet18 r3d_18 flow_logreg --n_train 700 --n_test 400 --size 64 --epochs 5
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TASKS = ("direction", "scene_bias", "periodic", "speed")
EVAL_SPLITS = {"direction": ("test",), "periodic": ("test",), "speed": ("test",), "scene_bias": ("val", "test")}
REGRESSION_TASKS = {"speed"}  # mode=reg: label = speed in px/frame; metrics R^2 and MAE


def task_mode(task: str) -> str:
    return "reg" if task in REGRESSION_TASKS else "cls"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--models", nargs="+", required=True, help="Model specs (see above).")
    p.add_argument("--data", type=Path, default=None,
                   help="An already generated benchmark folder (with dataset.json). Default: generate one.")
    p.add_argument("--out", type=Path, default=Path(".artifacts/motion-benchmark"))
    p.add_argument("--n_train", type=int, default=700)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_val", type=int, default=200,
                   help="Validation split used by the Trainer (scene_bias: the biased held-out split).")
    p.add_argument("--size", type=int, default=64, help="Generated frame size.")
    p.add_argument("--num_frames", type=int, default=None, help="Generated clip length (default 12; periodic 16).")
    p.add_argument("--seed", type=int, default=0, help="Data and model seed.")
    p.add_argument("--codec", default="auto")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--added_layers", type=int, choices=[1, 2], default=1)
    p.add_argument("--embed_size", type=int, default=256)
    p.add_argument("--freeze_backbone", action="store_true", help="Train only the added head.")
    p.add_argument("--use_peft", action="store_true")
    p.add_argument("--resize", type=int, default=112, help="Frame side fed to 2D backbones.")
    p.add_argument("--hflip", action="store_true", help="Random clip-level horizontal flip (refused on direction tasks).")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (keep low on a laptop).")
    return p.parse_args(argv)


def generate_or_load(args):
    from benchmarks.synthetic_motion import (
        BenchmarkDataset, make_direction_dataset, make_periodic_dataset, make_scene_bias_dataset, make_speed_dataset)
    if args.data is not None:
        root = args.data.resolve()
        meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        if meta["task"] != args.task:
            raise SystemExit(f"{root} holds a {meta['task']!r} benchmark, not {args.task!r}.")
        splits = {s: root / f"{s}.csv" for s in ("train", "val", "test") if (root / f"{s}.csv").is_file()}
        if "val" not in splits:
            raise SystemExit(f"{root} has no val split. Regenerate it with --n_val > 0: the Trainer must not "
                             "validate (or fit its error model) on the test split.")
        return BenchmarkDataset(meta["task"], root, root / "manifest.csv", root / "dataset.json", root / "masks.npz",
                                splits, meta["codec"], meta["classes"])
    if args.n_val < 1:
        raise SystemExit("--n_val must be at least 1: the Trainer validates on the val split, never on test.")
    root = (args.out / "data"
            / f"{args.task}_s{args.size}_n{args.n_train}-{args.n_val}-{args.n_test}_seed{args.seed}")
    common = dict(n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, size=args.size, seed=args.seed,
                  codec=args.codec)
    if args.task == "direction":
        return make_direction_dataset(root, num_frames=args.num_frames or 12, **common)
    if args.task == "periodic":
        return make_periodic_dataset(root, num_frames=args.num_frames or 16, **common)
    if args.task == "speed":
        return make_speed_dataset(root, num_frames=args.num_frames or 12, **common)
    return make_scene_bias_dataset(root, num_frames=args.num_frames or 12, **common)


def parse_spec(spec: str):
    """'single_frame:resnet18' -> ('single_frame', 'resnet18'); 'r3d_18' -> ('native', 'r3d_18')."""
    if spec in ("flow_logreg", "flow_probe"):
        return "probe", spec
    if ":" in spec:
        pooling, version = spec.split(":", 1)
        if pooling not in ("single_frame", "mean", "attention"):
            raise SystemExit(f"Unknown temporal strategy {pooling!r} in {spec!r}.")
        return pooling, version
    return "native", spec


def build_model(kind, version, num_classes, args, mode="cls"):
    from src.video.frame_pool import VideoFramePoolClassifier
    from src.video.single_frame import VideoSingleFrameClassifier
    from utils import build_video_model, is_native_video_architecture

    if kind == "native" and not is_native_video_architecture(version):
        raise SystemExit(f"{version!r} is not a native video model; use e.g. mean:{version}.")
    if version == "dummy":  # untrained resnet18 for smoke runs; not resolvable by build_video_model
        from src.vision.resnet import adjustedResNet
        from src.vision.resnet_peft import adjustedPeftResNet
        backbone = adjustedPeftResNet if args.use_peft else adjustedResNet
        if kind == "single_frame":
            return VideoSingleFrameClassifier(backbone, "dummy", num_classes, args.added_layers, args.embed_size,
                                              args.freeze_backbone, task_type=mode)
        return VideoFramePoolClassifier(backbone, "dummy", num_classes, args.added_layers, args.embed_size,
                                        args.freeze_backbone, task_type=mode, pooling=kind)
    return build_video_model(model_version=version, num_classes=num_classes, added_layers=args.added_layers,
                             embed_size=args.embed_size, freeze_backbone=args.freeze_backbone, mode=mode,
                             use_peft=args.use_peft, pooling="mean" if kind == "native" else kind)


class ClipHFlip:
    """Dataset wrapper: flip the *whole clip* horizontally with probability p."""

    def __init__(self, dataset, p=0.5, seed=0):
        self.dataset, self.p, self.rng = dataset, p, random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clip, label, *rest = self.dataset[idx]
        if self.rng.random() < self.p:
            clip = clip.flip(-1)
        return (clip, label, *rest)


def make_loaders(ds, kind, version, num_frames, args):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets.video_datasets import ParquetVideoDataset, collate_video_samples, native_video_transform
    from handlers.video_manifest import load_video_manifest

    clip_transform = native_video_transform(version) if kind == "native" else None
    frame_transform = None if clip_transform is not None else transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loaders = {}
    for split, path in ds.splits.items():
        frame = load_video_manifest(path)
        if task_mode(args.task) == "reg":
            frame["labels"] = frame["labels"].astype(float)
        else:
            frame["labels"] = frame["label_idx"].astype(int)
        dataset = ParquetVideoDataset(frame[["videos", "labels"]], transform=frame_transform,
                                      num_frames=num_frames, clip_transform=clip_transform)
        if split == "train" and args.hflip:
            dataset = ClipHFlip(dataset, seed=args.seed)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                    collate_fn=collate_video_samples, num_workers=0)
    return loaders


def evaluate(model, loader, mode="cls"):
    """Class probabilities [N, C] (cls) or predictions [N] (reg), and the labels."""
    import torch
    from options import DEVICE

    model.eval()
    outputs, labels = [], []
    with torch.no_grad():
        for clips, y, *_ in loader:
            out = model(clips.to(DEVICE))
            outputs.append((torch.softmax(out, dim=1) if mode == "cls" else out.reshape(-1)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(labels)


def scores(outputs, labels, mode="cls"):
    """{"acc", "auc"} for classification, {"r2", "mae"} for regression."""
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    if mode == "reg":
        return {"r2": float(r2_score(labels, outputs)), "mae": float(mean_absolute_error(labels, outputs))}
    acc = float((outputs.argmax(1) == labels).mean())
    try:
        auc = float(roc_auc_score(labels, outputs[:, 1]) if outputs.shape[1] == 2
                    else roc_auc_score(labels, outputs, multi_class="ovr"))
    except ValueError:  # a split with a single class
        auc = float("nan")
    return {"acc": acc, "auc": auc}


def run_probe(ds, args):
    """Flow reference: mean horizontal Farneback flow -> logistic regression
    (classification), or the flow summary -> ridge regression (speed)."""
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from benchmarks.motion_features import flow_summary_features, mean_flow_features, read_frames

    mode = task_mode(args.task)
    featurize = flow_summary_features if mode == "reg" else mean_flow_features
    target = "labels" if mode == "reg" else "label_idx"
    start = time.perf_counter()
    feats = {}
    for split, path in ds.splits.items():
        frame = pd.read_csv(path)
        feats[split] = (np.stack([featurize(read_frames(ds.root / p)) for p in frame["videos"]]),
                        frame[target].values)
    estimator = RidgeCV(alphas=np.logspace(-3, 3, 13)) if mode == "reg" else LogisticRegression(max_iter=5000)
    model = make_pipeline(StandardScaler(), estimator).fit(*feats["train"])
    out = {"train_seconds": round(time.perf_counter() - start, 1)}
    for split in EVAL_SPLITS[args.task]:
        x, y = feats[split]
        out[split] = scores(model.predict(x) if mode == "reg" else model.predict_proba(x), y, mode)
    return out


def run_model(ds, kind, version, args, run_dir):
    import torch
    from options import DEVICE
    from trainers.vision.trainer import Trainer

    meta = ds.load_metadata()
    num_frames = next(iter(meta["videos"].values()))["num_frames"]
    torch.manual_seed(args.seed)
    mode = task_mode(args.task)
    model = build_model(kind, version, 1 if mode == "reg" else len(meta["classes"]), args, mode=mode)
    loaders = make_loaders(ds, kind, version, num_frames, args)
    val_loader = loaders["val"]  # never test: the Trainer logs on it and fits error_model.json on it
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    Trainer(model, loaders["train"], val_loader, learning_rate=args.lr, mode=mode, num_epochs=args.epochs,
            output_dir=run_dir).train()
    out = {"train_seconds": round(time.perf_counter() - start, 1), "device": str(DEVICE)}
    for split in EVAL_SPLITS[args.task]:
        start = time.perf_counter()
        out[split] = scores(*evaluate(model, loaders[split], mode), mode)
        out[f"{split}_eval_seconds"] = round(time.perf_counter() - start, 1)
    return out


def main(argv=None):
    args = parse_args(argv)
    import torch
    from benchmarks.guards import check_flip_augmentation
    from options import DEVICE

    torch.set_num_threads(max(1, args.threads))
    ds = generate_or_load(args)
    check_flip_augmentation(ds.root, hflip=args.hflip)
    meta = ds.load_metadata()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir, n = args.out / f"{args.task}_{stamp}", 1
    while out_dir.exists():
        n += 1
        out_dir = args.out / f"{args.task}_{stamp}_{n}"
    out_dir.mkdir(parents=True)
    rows = []
    for spec in args.models:
        kind, version = parse_spec(spec)
        print(f"=== {spec} ({kind}) on {args.task} ===", flush=True)
        if kind == "probe":
            result = run_probe(ds, args)
            result["device"] = "cpu (sklearn)"
        else:
            result = run_model(ds, kind, version, args, out_dir / spec.replace(":", "_"))
        row = {"model": spec, "family": kind, "device": result["device"], "train_s": result["train_seconds"]}
        for split in EVAL_SPLITS[args.task]:
            label = {"val": "biased_val", "test": "unbiased_test"}[split] if args.task == "scene_bias" else split
            for metric, value in result[split].items():
                row[f"{label}_{metric}"] = round(value, 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "results.csv", index=False)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(dataset=str(ds.root), codec=meta["codec"], classes=meta["classes"], params=meta["params"],
                  device=str(DEVICE), torch=torch.__version__)
    (out_dir / "results.json").write_text(json.dumps({"config": config, "results": rows}, indent=2))
    markdown = _markdown(table)
    (out_dir / "results.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"Results: {out_dir}")
    return out_dir


def _markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

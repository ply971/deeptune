"""Honesty probes: what information do the benchmark clips actually carry?

Each function takes a generated :class:`~benchmarks.synthetic_motion.BenchmarkDataset`,
fits simple scikit-learn models on the train split and scores them on held-out
splits. Used by ``tests/benchmarks/test_benchmark_honesty.py`` (with fixed
thresholds) and ``scripts/check_benchmark_honesty.py`` (to report numbers at
any size).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.motion_features import (
    appearance_features,
    codec_mv_features,
    flow_summary_features,
    mean_flow_features,
    read_frames,
    spectrum_features,
)
from benchmarks.synthetic_motion import BenchmarkDataset, load_blob_masks


def _classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def _load(ds: BenchmarkDataset):
    manifest = ds.load_manifest()
    clips = [read_frames(ds.root / p) for p in manifest["videos"]]
    return manifest, clips


def _cls_scores(features, labels, splits, eval_on=("test",)):
    x = np.asarray(features)
    model = _classifier().fit(x[splits["train"]], labels[splits["train"]])
    out = {"train": float(model.score(x[splits["train"]], labels[splits["train"]]))}
    for name in eval_on:
        if splits[name].any():
            out[name] = float(model.score(x[splits[name]], labels[splits[name]]))
    return out


def _splits(manifest):
    return {s: (manifest["split"] == s).values for s in ("train", "val", "test")}


FRAME_PICKERS = {
    "first": lambda c: c[0],
    "middle": lambda c: c[len(c) // 2],
    "last": lambda c: c[-1],
    "average": lambda c: c.mean(axis=0),
}


def direction_probes(ds: BenchmarkDataset, codec_mv: bool = True) -> dict:
    """Accuracy of: mean horizontal flow; single frames (first/middle/last);
    the frame average; codec motion vectors (``None`` if unavailable)."""
    manifest, clips = _load(ds)
    y, splits = manifest["label_idx"].values, _splits(manifest)
    out = {"flow": _cls_scores([mean_flow_features(c) for c in clips], y, splits)}
    for name, pick in FRAME_PICKERS.items():
        out[f"frame_{name}"] = _cls_scores([appearance_features(pick(c)) for c in clips], y, splits)
    out["codec_mv"] = codec_mv_probe(ds) if codec_mv else None
    return out


def codec_mv_probe(ds: BenchmarkDataset) -> dict | None:
    """Accuracy of the mean horizontal codec motion vector per inter frame, or
    ``None`` when PyAV is missing or a stream exports no motion vectors."""
    manifest = ds.load_manifest()
    feats = [codec_mv_features(ds.root / p) for p in manifest["videos"]]
    if any(f is None for f in feats):
        return None
    length = min(len(f) for f in feats)
    return _cls_scores([f[:length] for f in feats], manifest["label_idx"].values, _splits(manifest))


def speed_probes(ds: BenchmarkDataset) -> dict:
    """Held-out R^2 of ridge regression on flow summaries vs the middle frame."""
    manifest, clips = _load(ds)
    y, splits = manifest["labels"].values.astype(float), _splits(manifest)

    def r2(features):
        x = np.asarray(features)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        model.fit(x[splits["train"]], y[splits["train"]])
        return float(r2_score(y[splits["test"]], model.predict(x[splits["test"]])))

    return {"flow_r2": r2([flow_summary_features(c) for c in clips]),
            "frame_middle_r2": r2([appearance_features(c[len(c) // 2]) for c in clips])}


def periodic_probes(ds: BenchmarkDataset) -> dict:
    manifest, clips = _load(ds)
    y, splits = manifest["label_idx"].values, _splits(manifest)
    return {"flow": _cls_scores([flow_summary_features(c) for c in clips], y, splits),
            "frame_middle": _cls_scores([appearance_features(c[len(c) // 2]) for c in clips], y, splits)}


def scene_bias_probes(ds: BenchmarkDataset) -> dict:
    """Background-only (blob masked out, middle frame) vs motion (flow) probes,
    on the biased ``val`` split and the unbiased ``test`` split."""
    manifest, clips = _load(ds)
    masks = [load_blob_masks(ds.root, v) for v in manifest["video_id"]]
    y, splits = manifest["label_idx"].values, _splits(manifest)
    mid = len(clips[0]) // 2
    background = [spectrum_features(c[mid], k[mid]) for c, k in zip(clips, masks)]
    family = manifest["background_family"].values
    return {
        "background_class": _cls_scores(background, y, splits, ("val", "test")),
        "background_family": _cls_scores(background, family, splits, ("val", "test")),
        "flow_class": _cls_scores([mean_flow_features(c) for c in clips], y, splits, ("val", "test")),
        "family_match_rate": {s: float(manifest.loc[splits[s], "family_matches_class"].mean())
                              for s in ("train", "val", "test") if splits[s].any()},
    }

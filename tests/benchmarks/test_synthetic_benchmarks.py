"""Format, determinism and pipeline-compatibility tests for the Phase 2
synthetic benchmarks (benchmarks/synthetic_motion.py). The "is the data
honest" probes live in test_benchmark_honesty.py."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from benchmarks.guards import UnsafeAugmentationError, check_flip_augmentation
from benchmarks.motion_features import read_frames
from benchmarks.render import BLOB_REDNESS_THRESHOLD, redness, resolve_codec
from benchmarks.synthetic_motion import (
    load_blob_masks,
    main as synthetic_main,
    make_direction_dataset,
    make_periodic_dataset,
    make_scene_bias_dataset,
    make_speed_dataset,
    make_untrimmed_dataset,
)
from handlers.video_manifest import load_video_manifest, read_video_manifest

CODEC = "mp4v"  # explicit: skips the (slow, noisy) H.264 encoder probe


@pytest.fixture(scope="module")
def untrimmed(tmp_path_factory):
    return make_untrimmed_dataset(tmp_path_factory.mktemp("untrimmed"), n_videos=8, length_range=(120, 200),
                                  n_classes=4, seed=3, size=48, max_instances=3, codec=CODEC)


# ---------------------------------------------------------------------------
# Layout, manifest and pipeline compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("maker,kwargs", [
    (make_direction_dataset, dict(num_frames=8)),
    (make_direction_dataset, dict(num_frames=8, n_val=4)),
    (make_speed_dataset, dict(num_frames=8, n_val=4)),
    (make_periodic_dataset, dict(num_frames=12, n_val=4)),
    (make_scene_bias_dataset, dict(num_frames=8, n_val=4)),
])
def test_trimmed_layout_and_manifest_load_through_existing_loader(tmp_path, maker, kwargs):
    ds = maker(tmp_path / "data", n_train=6, n_test=4, size=32, seed=0, codec=CODEC, **kwargs)
    manifest = pd.read_csv(ds.manifest)
    assert {"videos", "labels", "group", "split", "video_id"} <= set(manifest.columns)
    expected = {"train": 6, "test": 4, **({"val": 4} if "n_val" in kwargs else {})}
    assert manifest["split"].value_counts().to_dict() == expected
    assert set(ds.splits) == set(expected)
    meta = ds.load_metadata()
    assert meta["task"] == ds.task and meta["codec"] == CODEC == ds.codec and meta["seed"] == 0
    assert set(meta["videos"]) == set(manifest["video_id"])
    # The existing video-list loader accepts every manifest as-is.
    for path in [ds.manifest, *ds.splits.values()]:
        frame = read_video_manifest(path)
        assert len(frame) == len(pd.read_csv(path))
    clip_frames = kwargs["num_frames"]
    for vid, row in manifest.set_index("video_id").iterrows():
        frames = read_frames(ds.root / row["videos"])
        assert frames.shape == (clip_frames, 32, 32, 3)
        info = meta["videos"][vid]
        assert info["num_frames"] == clip_frames and info["mask_shape"] == [clip_frames, 32, 32]
        assert len(info["centers"]) == clip_frames
        masks = load_blob_masks(ds.root, vid)
        assert masks.shape == (clip_frames, 32, 32) and masks.any(axis=(1, 2)).all()
    if ds.classes:
        # Label indices follow sorted class names, as DeepTune's LabelEncoder does.
        assert ds.classes == sorted(ds.classes)
        assert (manifest["labels"].map(ds.classes.index) == manifest["label_idx"]).all()
    else:
        assert pd.api.types.is_float_dtype(manifest["labels"]) and (manifest["labels"] > 0).all()


def test_manifest_feeds_parquet_video_dataset(tmp_path):
    from datasets.video_datasets import ParquetVideoDataset
    from handlers.raw_to_parquet_dataset import raw_to_parquet

    ds = make_direction_dataset(tmp_path / "d", n_train=4, n_test=2, num_frames=8, size=32, seed=1, codec=CODEC)
    frame = load_video_manifest(ds.splits["train"])
    frame["labels"] = frame["label_idx"]  # DeepTune's split step label-encodes names the same way
    dataset = ParquetVideoDataset(frame[["videos", "labels", "group"]], num_frames=8)
    clip, label, extras = dataset[0]
    assert tuple(clip.shape) == (8, 3, 32, 32) and int(label) in (0, 1) and extras["group"].startswith("train-")
    parquet = pd.read_parquet(raw_to_parquet(ds.manifest, tmp_path / "pq", "video"))
    assert len(parquet) == 6 and isinstance(parquet["videos"].iloc[0], bytes)
    assert set(parquet["labels"]) == {"left", "right"}


def test_groups_never_cross_splits_and_classes_are_balanced(tmp_path):
    ds = make_direction_dataset(tmp_path / "d", n_train=12, n_val=6, n_test=8, num_frames=6, size=32, seed=0, codec=CODEC)
    m = pd.read_csv(ds.manifest)
    assert m.groupby("group")["split"].nunique().max() == 1
    assert m.groupby("split")["label_idx"].mean().tolist() == [0.5, 0.5, 0.5]
    # Each scene (shared background) hosts one clip of each class.
    assert (m.groupby("group")["label_idx"].nunique() == 2).all()


def test_direction_ground_truth_matches_label_and_symmetric_trajectory(tmp_path):
    ds = make_direction_dataset(tmp_path / "d", n_train=10, n_test=0, num_frames=9, size=32, seed=5, codec=CODEC)
    for vid, info in ds.load_metadata()["videos"].items():
        vx, _ = info["velocity"]
        assert (vx > 0) == (info["label"] == "right") and abs(vx) > 0
        centers = np.array(info["centers"])
        steps = (np.diff(centers[:, 0]) + 16) % 32 - 16  # unwrap the torus
        assert np.allclose(steps, vx, atol=1e-3)


def test_generators_are_deterministic_and_seed_sensitive(tmp_path):
    a = make_direction_dataset(tmp_path / "a", n_train=4, n_test=2, num_frames=6, size=32, seed=7, codec=CODEC)
    b = make_direction_dataset(tmp_path / "b", n_train=4, n_test=2, num_frames=6, size=32, seed=7, codec=CODEC)
    c = make_direction_dataset(tmp_path / "c", n_train=4, n_test=2, num_frames=6, size=32, seed=8, codec=CODEC)
    meta_a, meta_b, meta_c = a.load_metadata(), b.load_metadata(), c.load_metadata()
    assert meta_a["videos"] == meta_b["videos"] and meta_a["videos"] != meta_c["videos"]
    for vid in meta_a["videos"]:
        fa = read_frames(a.root / meta_a["videos"][vid]["file"])
        fb = read_frames(b.root / meta_b["videos"][vid]["file"])
        assert np.array_equal(fa, fb)
        assert np.array_equal(load_blob_masks(a.root, vid), load_blob_masks(b.root, vid))
    # A clip does not depend on how many other clips were generated.
    d = make_direction_dataset(tmp_path / "d", n_train=2, n_test=1, num_frames=6, size=32, seed=7, codec=CODEC)
    assert d.load_metadata()["videos"]["direction_train_00001"] == meta_a["videos"]["direction_train_00001"]


def test_scene_bias_family_correlation(tmp_path):
    ds = make_scene_bias_dataset(tmp_path / "s", n_train=40, n_val=20, n_test=16, num_frames=4, size=32,
                                 seed=0, bias=1.0, codec=CODEC)
    m = pd.read_csv(ds.manifest)
    assert m.loc[m.split != "test", "family_matches_class"].all()
    test = m[m.split == "test"]
    # Unbiased split: every class x family combination equally often.
    counts = pd.crosstab(test["labels"], test["background_family"])
    assert counts.shape == (2, 2) and (counts.values == 4).all()


def test_scene_bias_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError, match="n_classes"):
        make_scene_bias_dataset(tmp_path / "x", n_train=2, n_test=2, n_classes=9, codec=CODEC)
    with pytest.raises(ValueError, match="bias"):
        make_scene_bias_dataset(tmp_path / "x", n_train=2, n_test=2, bias=1.5, codec=CODEC)
    with pytest.raises(ValueError, match="size"):
        make_direction_dataset(tmp_path / "x", n_train=2, n_test=2, size=33, codec=CODEC)


# ---------------------------------------------------------------------------
# Untrimmed videos: ground truth must match the rendered frames
# ---------------------------------------------------------------------------

def test_untrimmed_segments_are_valid_and_leave_background_stretches(untrimmed):
    meta = untrimmed.load_metadata()
    min_gap = meta["params"]["min_gap"]
    for vid, info in meta["videos"].items():
        n = info["num_frames"]
        segments = info["segments"]
        assert 1 <= len(segments) <= 3
        cursor = 0
        for s in segments:
            assert s["start_frame"] - cursor >= min_gap and s["end_frame"] > s["start_frame"]
            assert s["t_start"] == pytest.approx(s["start_frame"] / meta["fps"])
            assert s["t_end"] == pytest.approx(s["end_frame"] / meta["fps"])
            cursor = s["end_frame"]
        assert n - cursor >= min_gap
        # background_intervals is exactly the complement of the segments.
        covered = np.zeros(n, int)
        for s in segments:
            covered[s["start_frame"]:s["end_frame"]] += 1
        for start, end in info["background_intervals"]:
            covered[start:end] += 1
        assert (covered == 1).all()


def test_untrimmed_blob_present_iff_inside_a_segment(untrimmed):
    """Checked on the *decoded MP4 pixels*, independently of the generator:
    the blob is the only red-dominant thing in the scene."""
    meta = untrimmed.load_metadata()
    for vid, info in meta["videos"].items():
        frames = read_frames(untrimmed.root / info["file"])
        assert len(frames) == info["num_frames"]
        inside = np.zeros(len(frames), bool)
        smallest_blob = np.inf
        for s in info["segments"]:
            inside[s["start_frame"]:s["end_frame"]] = True
            smallest_blob = min(smallest_blob, np.pi * s["radius"] ** 2)
        red = redness(frames) > BLOB_REDNESS_THRESHOLD
        counts = red.sum(axis=(1, 2))
        assert (counts[inside] >= 0.25 * smallest_blob).all(), vid
        assert (counts[~inside] <= 2).all(), vid
        masks = load_blob_masks(untrimmed.root, vid)
        assert np.array_equal(masks.any(axis=(1, 2)), inside)
        for f in np.flatnonzero(inside):
            iou = (red[f] & masks[f]).sum() / (red[f] | masks[f]).sum()
            assert iou >= 0.6, (vid, f, iou)
        assert all((c is None) != inside[f] for f, c in enumerate(info["centers"]))


def test_untrimmed_video_labels_equal_segment_label_sets(untrimmed):
    meta = untrimmed.load_metadata()
    classes = meta["classes"]
    manifest = pd.read_csv(untrimmed.manifest).set_index("video_id")
    gt = pd.read_csv(untrimmed.extra["gt_segments"])
    assert list(gt.columns) == ["video-id", "t-start", "t-end", "label"]
    anet = json.loads(untrimmed.extra["gt_activitynet"].read_text())["database"]
    for vid, info in meta["videos"].items():
        seg_labels = {s["label"] for s in info["segments"]}
        assert set(info["labels"]) == seg_labels
        assert info["multi_hot"] == [int(c in seg_labels) for c in classes]
        row = manifest.loc[vid]
        assert set(row["labels"].split(";")) == seg_labels
        assert [int(row[f"label_{c}"]) for c in classes] == info["multi_hot"]
        rows = gt[gt["video-id"] == vid]
        assert rows[["t-start", "t-end"]].values.tolist() == [[s["t_start"], s["t_end"]] for s in info["segments"]]
        assert rows["label"].tolist() == [s["label"] for s in info["segments"]]
        assert [a["segment"] for a in anet[vid]["annotations"]] == [[s["t_start"], s["t_end"]] for s in info["segments"]]
        assert anet[vid]["duration"] == pytest.approx(info["num_frames"] / meta["fps"])
    read_video_manifest(untrimmed.manifest)  # loads through the existing loader


def test_untrimmed_distractor_moves_but_is_not_an_action(tmp_path):
    ds = make_untrimmed_dataset(tmp_path / "u", n_videos=2, length_range=(80, 80), n_classes=2, seed=0,
                                size=32, max_instances=1, segment_len_range=(16, 16),
                                background_motion="distractor", codec=CODEC)
    for vid, info in ds.load_metadata()["videos"].items():
        assert info["background_motion"] == "distractor"
        frames = read_frames(ds.root / info["file"]).astype(np.int16)
        start, end = info["background_intervals"][0]
        assert end - start >= 8
        # Background-only frames still change over time (the distractor moves) ...
        assert np.abs(frames[end - 1] - frames[start]).max() > 60
        # ... but contain no blob.
        assert not load_blob_masks(ds.root, vid)[start:end].any()
        assert len(set(np.round(info["distractor"]["y"], 3))) > 5


# ---------------------------------------------------------------------------
# Guards, codec, CLI
# ---------------------------------------------------------------------------

def test_flip_guard_blocks_direction_tasks_only(tmp_path):
    d = make_direction_dataset(tmp_path / "d", n_train=2, n_test=0, num_frames=4, size=32, codec=CODEC)
    p = make_periodic_dataset(tmp_path / "p", n_train=2, n_test=0, num_frames=4, size=32, codec=CODEC)
    with pytest.raises(UnsafeAugmentationError, match="direction"):
        check_flip_augmentation(d.root, hflip=True)
    with pytest.raises(UnsafeAugmentationError):
        check_flip_augmentation(d.splits["train"], hflip=True)  # a manifest inside the folder
    check_flip_augmentation(d.root, hflip=False, vflip=True)
    check_flip_augmentation(p.root, hflip=True, vflip=True)
    check_flip_augmentation(tmp_path / "not-a-benchmark", hflip=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_flip_augmentation(d.metadata, hflip=True, strict=False)
    assert any("unsafe" in str(w.message) for w in caught)


def test_resolve_codec():
    assert resolve_codec("mp4v") == "mp4v"
    with pytest.raises(ValueError):
        resolve_codec("h264-but-too-long")


def test_cli_generates_each_task(tmp_path, capsys):
    for task in ("direction", "speed", "periodic", "scene_bias"):
        synthetic_main(["--task", task, "--out", str(tmp_path / task), "--n_train", "2", "--n_test", "2",
                        "--n_val", "2", "--size", "32", "--num_frames", "4", "--codec", CODEC])
        report = json.loads(capsys.readouterr().out)
        assert report["task"] == task and report["codec"] == CODEC and report["clips"] == 6  # 2 train + 2 val + 2 test
        assert report["splits"] == {"train": 2, "val": 2, "test": 2}
    synthetic_main(["--task", "untrimmed", "--out", str(tmp_path / "u"), "--n_videos", "2", "--min_length", "60",
                    "--max_length", "70", "--size", "32", "--codec", CODEC])
    report = json.loads(capsys.readouterr().out)
    assert report["clips"] == 2 and (tmp_path / "u" / "gt_segments.csv").is_file()

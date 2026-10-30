"""Smoke tests for scripts/run_motion_benchmark.py (tiny clips, untrained models, CPU)."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.guards import UnsafeAugmentationError

_SPEC = importlib.util.spec_from_file_location(
    "run_motion_benchmark", Path(__file__).resolve().parents[2] / "scripts" / "run_motion_benchmark.py")
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

TINY = ["--n_train", "4", "--n_test", "4", "--n_val", "3", "--size", "32", "--num_frames", "4", "--epochs", "1",
        "--batch_size", "2", "--resize", "32", "--embed_size", "8", "--codec", "mp4v", "--threads", "2"]


def test_direction_table_has_every_model(tmp_path):
    out = bench.main(["--task", "direction", "--out", str(tmp_path),
                      "--models", "single_frame:dummy", "mean:dummy", "attention:dummy", "dummy_video3d", "flow_logreg",
                      *TINY])
    table = pd.read_csv(out / "results.csv")
    assert table["model"].tolist() == ["single_frame:dummy", "mean:dummy", "attention:dummy", "dummy_video3d",
                                       "flow_logreg"]
    assert {"test_acc", "test_auc", "train_s", "device"} <= set(table.columns)
    assert table["test_acc"].between(0, 1).all()
    report = json.loads((out / "results.json").read_text())
    assert report["config"]["codec"] == "mp4v" and report["config"]["classes"] == ["left", "right"]
    assert (out / "results.md").read_text().startswith("| model |")
    # Each neural model trained through the shared Trainer (its log is written).
    assert (out / "single_frame_dummy" / "training_log.csv").is_file()
    # The Trainer validated (and fitted its error model) on the 3-clip val split, never on the 4-clip test split.
    assert report["config"]["params"]["n_val"] == 3
    fit = json.loads((out / "single_frame_dummy" / "error_model_fit.json").read_text())
    assert fit["split"] == "validation" and fit["val_dataset_size"] == 3


def test_runner_refuses_data_without_a_val_split(tmp_path):
    from benchmarks.synthetic_motion import make_direction_dataset
    ds = make_direction_dataset(tmp_path / "old", n_train=4, n_test=4, num_frames=4, size=32, codec="mp4v")
    with pytest.raises(SystemExit, match="no val split"):
        bench.main(["--task", "direction", "--data", str(ds.root), "--out", str(tmp_path), "--models", "flow_logreg"])
    with pytest.raises(SystemExit, match="n_val"):
        bench.main(["--task", "direction", "--out", str(tmp_path), "--models", "flow_logreg", *TINY, "--n_val", "0"])


def test_scene_bias_reports_biased_and_unbiased_splits(tmp_path):
    out = bench.main(["--task", "scene_bias", "--out", str(tmp_path), "--models", "flow_logreg", *TINY])
    table = pd.read_csv(out / "results.csv")
    assert {"biased_val_acc", "unbiased_test_acc", "biased_val_auc", "unbiased_test_auc"} <= set(table.columns)


def test_reuses_generated_data_and_rejects_task_mismatch(tmp_path):
    first = bench.main(["--task", "periodic", "--out", str(tmp_path), "--models", "flow_logreg", *TINY])
    data = Path(json.loads((first / "results.json").read_text())["config"]["dataset"])
    again = bench.main(["--task", "periodic", "--out", str(tmp_path / "again"), "--data", str(data),
                        "--models", "single_frame:dummy", "--hflip", *TINY])
    assert len(pd.read_csv(again / "results.csv")) == 1
    with pytest.raises(SystemExit, match="periodic"):
        bench.main(["--task", "direction", "--data", str(data), "--out", str(tmp_path), "--models", "flow_logreg"])


def test_hflip_is_refused_on_direction(tmp_path):
    with pytest.raises(UnsafeAugmentationError):
        bench.main(["--task", "direction", "--out", str(tmp_path), "--models", "flow_logreg", "--hflip", *TINY])


def test_bad_specs_are_rejected():
    with pytest.raises(SystemExit):
        bench.parse_spec("max:resnet18")
    assert bench.parse_spec("r3d_18") == ("native", "r3d_18")
    assert bench.parse_spec("single_frame:resnet18") == ("single_frame", "resnet18")


def test_speed_regression_task(tmp_path):
    out = bench.main(["--task", "speed", "--out", str(tmp_path), "--models", "single_frame:dummy", "flow_probe", *TINY])
    table = pd.read_csv(out / "results.csv")
    assert table["model"].tolist() == ["single_frame:dummy", "flow_probe"]
    assert {"test_r2", "test_mae"} <= set(table.columns) and "test_acc" not in table.columns
    assert (table["test_mae"] >= 0).all()
    fit = json.loads((out / "single_frame_dummy" / "error_model_fit.json").read_text())
    assert fit["split"] == "validation" and fit["val_dataset_size"] == 3

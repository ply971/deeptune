"""deeptune.py refuses --video_flip on direction-labelled benchmark data (A2's
clip augmentation + benchmarks.guards), and allows it on ordinary data."""
import shutil

import pandas as pd
import pytest

import deeptune
from benchmarks.synthetic_motion import make_direction_dataset, make_periodic_dataset


class _ReachedDataLoading(Exception):
    """Raised by the stubbed raw_to_parquet: the guard let the run through."""


def _argv(df, out):
    return ["--modality", "video", "--df", str(df), "--out", str(out), "--raw-data", "--model_version", "resnet18",
            "--num_frames", "4", "--num_epochs", "1", "--video_augment", "--video_flip"]


@pytest.fixture
def stop_before_training(monkeypatch):
    def stub(*args, **kwargs):
        raise _ReachedDataLoading()
    monkeypatch.setattr(deeptune, "raw_to_parquet", stub)


def test_video_flip_is_refused_on_direction_benchmark(tmp_path, capsys, stop_before_training):
    ds = make_direction_dataset(tmp_path / "direction", n_train=6, n_test=4, num_frames=4, size=32, codec="mp4v")
    with pytest.raises(SystemExit) as exit_info:
        deeptune.main(_argv(ds.manifest, tmp_path / "out"))
    assert exit_info.value.code == 2
    assert "unsafe on the 'direction' benchmark" in capsys.readouterr().err


def test_video_flip_is_allowed_on_flip_safe_and_ordinary_data(tmp_path, stop_before_training):
    periodic = make_periodic_dataset(tmp_path / "periodic", n_train=6, n_test=4, num_frames=4, size=32, codec="mp4v")
    with pytest.raises(_ReachedDataLoading):
        deeptune.main(_argv(periodic.manifest, tmp_path / "out1"))
    # Ordinary user data: the same clips, but no benchmark dataset.json next to the list.
    plain = tmp_path / "plain"
    shutil.copytree(make_direction_dataset(tmp_path / "d", n_train=6, n_test=4, num_frames=4, size=32,
                                           codec="mp4v").root / "videos", plain / "videos")
    manifest = pd.read_csv(tmp_path / "d" / "manifest.csv")[["videos", "labels"]]
    manifest.to_csv(plain / "clips.csv", index=False)
    with pytest.raises(_ReachedDataLoading):
        deeptune.main(_argv(plain / "clips.csv", tmp_path / "out2"))

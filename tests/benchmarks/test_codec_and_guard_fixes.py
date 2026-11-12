"""A recorded codec is always true; a BOM cannot switch the flip guard off."""
import json

import cv2
import numpy as np
import pytest

from benchmarks import render
from benchmarks.guards import UnsafeAugmentationError, check_flip_augmentation
from benchmarks.synthetic_motion import make_direction_dataset


def _frames():
    return np.random.default_rng(0).integers(0, 255, (4, 32, 32, 3), dtype=np.uint8)


def test_write_mp4_accepts_a_codec_it_really_wrote(tmp_path):
    path = tmp_path / "a.mp4"
    render.write_mp4(path, _frames(), 10, "mp4v")
    assert render.codec_family(render.read_fourcc(path)) == "mpeg4"


def test_write_mp4_refuses_a_silently_replaced_codec(tmp_path, monkeypatch):
    """OpenCV really writes MPEG-4 here while the caller asked for FFV1."""
    real = cv2.VideoWriter_fourcc
    monkeypatch.setattr(render.cv2, "VideoWriter_fourcc", lambda *code: real(*"mp4v"))
    path = tmp_path / "a.mp4"
    with pytest.raises(RuntimeError, match="FFV1"):
        render.write_mp4(path, _frames(), 10, "FFV1")
    assert render.codec_family(render.read_fourcc(path)) == "mpeg4"  # the file exists: only the check caught it


def test_dataset_generation_stops_instead_of_recording_a_false_codec(tmp_path, monkeypatch):
    real = cv2.VideoWriter_fourcc
    monkeypatch.setattr(render.cv2, "VideoWriter_fourcc", lambda *code: real(*"mp4v"))
    root = tmp_path / "bench"
    with pytest.raises(RuntimeError, match="FFV1"):
        make_direction_dataset(root, n_train=2, n_test=2, num_frames=4, size=32, codec="FFV1")
    assert not (root / "dataset.json").exists()


@pytest.mark.parametrize("fourcc, family", [("mp4v", "mpeg4"), ("FMP4", "mpeg4"), ("avc1", "h264"), ("H264", "h264"),
                                            ("FFV1", "ffv1"), ("ffv1", "ffv1"), ("png ", "png")])
def test_codec_family(fourcc, family):
    assert render.codec_family(fourcc) == family


def test_unreadable_file_has_no_fourcc(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert render.read_fourcc(bad) == ""


def test_flip_guard_refuses_a_bom_saved_dataset_json(tmp_path):
    """Windows editors and PowerShell 5.1 write a BOM; the guard must still see the metadata."""
    meta = {"task": "direction", "hflip_safe": False, "vflip_safe": True}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(meta), encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with pytest.raises(UnsafeAugmentationError):
        check_flip_augmentation(tmp_path, hflip=True)
    with pytest.raises(UnsafeAugmentationError):
        check_flip_augmentation(path, hflip=True)

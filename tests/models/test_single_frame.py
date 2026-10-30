"""Single-frame baseline (Karpathy et al. 2014): a 2D backbone on the middle frame."""
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import utils
from src.video.single_frame import VideoSingleFrameClassifier
from src.vision.resnet import adjustedResNet
from src.vision.resnet_peft import adjustedPeftResNet


def _model(cls=adjustedResNet, **kwargs):
    defaults = dict(frame_model_cls=cls, frame_model_version="dummy", num_classes=3, added_layers=2,
                    embedding_layer_size=16)
    return VideoSingleFrameClassifier(**{**defaults, **kwargs}).eval()


@pytest.mark.parametrize("cls", [adjustedResNet, adjustedPeftResNet])
def test_single_frame_output_shapes(cls):
    torch.manual_seed(0)
    model = _model(cls)
    x = torch.randn(2, 5, 3, 32, 32)
    assert model(x).shape == (2, 3)
    assert model(x, extract_embed=True).shape == (2, 16)


def test_single_frame_regression_shape():
    model = _model(num_classes=1, task_type="reg")
    assert model(torch.randn(2, 4, 3, 32, 32)).shape == (2, 1)


@pytest.mark.parametrize("num_frames", [1, 4, 5])
def test_single_frame_sees_only_the_middle_frame(num_frames):
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(2, num_frames, 3, 32, 32)
    mid = VideoSingleFrameClassifier.middle_index(num_frames)
    assert mid == num_frames // 2
    with torch.no_grad():
        base = model(x)
        other = x.clone()
        for t in range(num_frames):
            if t != mid:
                other[:, t] = torch.randn_like(other[:, t])
        assert torch.allclose(model(other), base)
        changed = x.clone()
        changed[:, mid] = torch.randn_like(changed[:, mid])
        assert not torch.allclose(model(changed), base)
        assert torch.allclose(base, model.backbone(x[:, mid]))


def test_single_frame_rejects_non_clip_input():
    with pytest.raises(ValueError, match=r"\[B, T, C, H, W\]"):
        _model()(torch.randn(2, 3, 32, 32))


def test_build_video_model_registers_single_frame(monkeypatch):
    monkeypatch.setattr(utils, "get_video_model_cls", lambda version, use_peft=False: adjustedResNet)
    model = utils.build_video_model("dummy", num_classes=2, added_layers=1, embed_size=8, freeze_backbone=True,
                                    mode="cls", pooling="single_frame")
    assert isinstance(model, VideoSingleFrameClassifier)
    assert model.eval()(torch.randn(1, 3, 3, 32, 32)).shape == (1, 2)
    assert "single_frame" in utils.VIDEO_POOLING_CHOICES
    with pytest.raises(ValueError, match="native video architecture"):
        utils.build_video_model("dummy_video3d", num_classes=2, added_layers=1, embed_size=8,
                                freeze_backbone=False, mode="cls", pooling="single_frame")


def test_cli_and_infer_accept_single_frame_pooling(tmp_path):
    import cli
    import infer

    opts = cli.DeepTuneVisionOptions(utils.RunType.ONECALL, args=[
        "--modality", "video", "--df", str(tmp_path), "--out", str(tmp_path), "--pooling", "single_frame"])
    assert opts.pooling == "single_frame"
    parsed = infer.build_parser().parse_args([
        "--modality", "video", "--checkpoint", str(tmp_path), "--data", str(tmp_path), "--out", str(tmp_path),
        "--pooling", "single_frame"])
    assert parsed.pooling == "single_frame"


def test_single_frame_trains_with_existing_trainer_on_benchmark_clips(tmp_path):
    """End to end: Phase 2 direction clips -> existing manifest loader ->
    ParquetVideoDataset -> the shared vision Trainer."""
    from benchmarks.synthetic_motion import make_direction_dataset
    from datasets.video_datasets import ParquetVideoDataset
    from handlers.video_manifest import load_video_manifest
    from trainers.vision.trainer import Trainer

    ds = make_direction_dataset(tmp_path / "bench", n_train=4, n_test=2, num_frames=6, size=32, seed=0, codec="mp4v")
    loaders = {}
    for split in ("train", "test"):
        frame = load_video_manifest(ds.splits[split])
        frame["labels"] = frame["label_idx"]
        loaders[split] = DataLoader(ParquetVideoDataset(frame[["videos", "labels"]], num_frames=6), batch_size=2)
    torch.manual_seed(0)
    model = VideoSingleFrameClassifier(adjustedResNet, "dummy", num_classes=2, added_layers=1,
                                       embedding_layer_size=8)
    trainer = Trainer(model, loaders["train"], loaders["test"], learning_rate=1e-3, mode="cls", num_epochs=1,
                      output_dir=tmp_path / "run")
    trainer.train()
    log = pd.read_csv(tmp_path / "run" / "training_log.csv")
    assert len(log) == 1
    loss, acc = trainer.validate()
    assert loss > 0 and 0 <= acc <= 100


def test_single_frame_trains_on_tensor_batches(tmp_path):
    from trainers.vision.trainer import Trainer

    torch.manual_seed(0)
    data = TensorDataset(torch.randn(4, 3, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
    model = _model(adjustedPeftResNet, num_classes=2).train()
    Trainer(model, DataLoader(data, batch_size=2), DataLoader(data, batch_size=2), learning_rate=1e-3,
            mode="cls", num_epochs=1, output_dir=tmp_path).train()


def test_desktop_run_config_accepts_single_frame_pooling():
    from desktop.config import RunConfig, validate_config
    pooling_issues = lambda pooling: [i for i in validate_config(RunConfig(modality="video", pooling=pooling))
                                      if "pooling" in i]
    assert pooling_issues("single_frame") == [] and pooling_issues("mean") == []
    assert pooling_issues("max") != []

"""Trainable-parameter budget (--trainable_params) and the trainable/total report.

CPU only, dummy models only (weights=None): `dummy` 2D ResNet-18 and
`dummy_video3d` (r3d_18 without pretrained weights).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cli import DeepTuneVisionOptions
from src.video.frame_pool import VideoFramePoolClassifier
from src.video.video3d import adjustedVideo3D
from src.video.video3d_peft import adjustedPeftVideo3D
from src.vision.resnet import adjustedResNet
from src.vision.resnet_peft import adjustedPeftResNet
from trainers.param_budget import (
    ParamCountError,
    apply_top_down_budget,
    build_with_budget,
    choose_lora_rank,
    format_summary,
    lora_rank_line,
    parameter_group,
    parse_param_count,
    summarize_parameters,
)
from trainers.vision.trainer import Trainer
from utils import RunType, build_video_model


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def peft_video3d(rank, num_classes=3):
    return adjustedPeftVideo3D(num_classes, "dummy_video3d", 2, rank)


def peft_frame_pool(rank, num_classes=3):
    return VideoFramePoolClassifier(adjustedPeftResNet, "dummy", num_classes, 2, rank, pooling="attention")


# ---------------------------------------------------------------------------
# parse_param_count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("250000", 250_000), ("100k", 100_000), ("100K", 100_000), ("1.5M", 1_500_000),
    ("2m", 2_000_000), ("1_000_000", 1_000_000), ("1,000,000", 1_000_000), (" 50 k ", 50_000),
    ("1.5k", 1_500), ("0.5M", 500_000), (12345, 12345),
])
def test_parse_param_count_accepts(text, expected):
    assert parse_param_count(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "abc", "0", "0k", "-5", "-1k", "1.2345k", "1,5M", "10x", "1e5", "k", "1..5k", 0, -3, True])
def test_parse_param_count_rejects(text):
    with pytest.raises(ParamCountError):
        parse_param_count(text)


def test_parse_param_count_error_is_value_error_and_argparse_friendly(tmp_path, capsys):
    with pytest.raises(ValueError):
        parse_param_count("lots")
    with pytest.raises(SystemExit):
        DeepTuneVisionOptions(RunType.TRAIN, args=["--out", str(tmp_path), "--model_version", "resnet18", "--trainable_params", "lots"])
    err = capsys.readouterr().err
    assert "--trainable_params" in err and "invalid parameter count 'lots'" in err


def test_cli_records_trainable_params(tmp_path):
    args = DeepTuneVisionOptions(RunType.TRAIN, args=["--out", str(tmp_path), "--model_version", "resnet18", "--trainable_params", "100k"])
    assert args.trainable_params == 100_000
    args.save_args(tmp_path)
    saved = json.loads((tmp_path / "cli_arguments.json").read_text(encoding="utf-8"))
    assert saved["trainable_params"] == 100_000
    default = DeepTuneVisionOptions(RunType.TRAIN, args=["--out", str(tmp_path), "--model_version", "resnet18"])
    assert default.trainable_params is None
    onecall = DeepTuneVisionOptions(RunType.ONECALL, args=["--modality", "video", "--df", "x.parquet", "--out", str(tmp_path), "--trainable_params", "1.5M"])
    assert onecall.trainable_params == 1_500_000


# ---------------------------------------------------------------------------
# summarize_parameters / format_summary
# ---------------------------------------------------------------------------

def test_summary_matches_manual_numel_sum():
    model = adjustedResNet(3, "dummy", added_layers=2, embedding_layer_size=16, freeze_backbone=True)
    summary = summarize_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = trainable_count(model)
    head = sum(p.numel() for m in (model.fc1, model.fc2) for p in m.parameters())
    assert summary["total"] == total
    assert summary["trainable"] == trainable == head
    assert summary["frozen"] == total - trainable
    assert summary["trainable_pct"] == pytest.approx(100 * trainable / total)
    assert summary["breakdown"]["head"] == {"total": head, "trainable": head}
    assert summary["breakdown"]["backbone"] == {"total": total - head, "trainable": 0}
    assert summary["breakdown"]["lora"] == {"total": 0, "trainable": 0}


def test_summary_groups_lora_and_frame_pool_head():
    model = peft_frame_pool(4)
    summary = summarize_parameters(model)
    named = dict(model.named_parameters())
    lora = sum(p.numel() for n, p in named.items() if "lora_" in n)
    head = sum(p.numel() for n, p in named.items() if n.startswith(("backbone.fc1", "backbone.fc2", "attention")))
    assert lora > 0 and summary["breakdown"]["lora"]["total"] == lora
    assert summary["breakdown"]["head"]["total"] == head
    assert summary["total"] == sum(p.numel() for p in model.parameters())
    # peft freezes the original backbone; LoRA + head are what trains.
    assert summary["trainable"] == lora + head == trainable_count(model)
    assert parameter_group("backbone.model.conv1.base_layer.weight") == "backbone"
    assert parameter_group("attention.weight") == "head"


def test_format_summary_line():
    line = format_summary({"trainable": 98_304, "total": 33_231_000, "trainable_pct": 100 * 98_304 / 33_231_000})
    assert line == "trainable params: 98,304 || all params: 33,231,000 || trainable%: 0.2958"


# ---------------------------------------------------------------------------
# apply_top_down_budget (no PEFT)
# ---------------------------------------------------------------------------

def test_top_down_budget_trains_head_plus_contiguous_top_block():
    model = adjustedResNet(3, "dummy", added_layers=2, embedding_layer_size=16)
    head_size = sum(p.numel() for m in (model.fc1, model.fc2) for p in m.parameters())
    budget = head_size + 600_000
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    summary = apply_top_down_budget(model, budget)

    assert summary["trainable"] == trainable_count(model) <= budget
    assert all(p.requires_grad for m in (model.fc1, model.fc2) for p in m.parameters())
    backbone = [(n, p) for n, p in model.named_parameters() if n.startswith("model.")]
    flags = [p.requires_grad for _, p in backbone]
    k = sum(flags)
    assert 0 < k < len(backbone)
    assert flags == [False] * (len(backbone) - k) + [True] * k, "trainable backbone must be one top block"
    # The next tensor down would not have fitted.
    assert summary["trainable"] + backbone[len(backbone) - k - 1][1].numel() > budget
    assert summary["unfrozen_backbone_tensors"] == k
    # Shapes and values untouched.
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def test_top_down_budget_frozen_params_really_frozen_and_optimizer_gets_only_trainable(tmp_path):
    model = adjustedResNet(2, "dummy", added_layers=1)
    budget = 2 * 512 + 2 + 1_000_000
    apply_top_down_budget(model, budget)
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
    trainer = Trainer(model, loader, loader, 1e-3, "cls", 1, tmp_path)

    optimised = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert optimised == {id(p) for p in model.parameters() if p.requires_grad}

    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    trained = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert frozen and trained
    trainer.train()
    after = dict(model.named_parameters())
    # BatchNorm running stats are buffers, not parameters, so frozen weights must be bit-identical.
    assert all(torch.equal(frozen[n], after[n]) for n in frozen)
    assert any(not torch.equal(trained[n], after[n]) for n in trained)

    written = json.loads((tmp_path / "parameter_summary.json").read_text(encoding="utf-8"))
    assert written["trainable"] == trainable_count(model) <= budget
    assert written["total"] == sum(p.numel() for p in model.parameters())
    assert trainer.parameter_summary == written


def test_top_down_budget_head_too_big_raises():
    model = adjustedVideo3D(3, "dummy_video3d", added_layers=2, embedding_layer_size=64)
    head_size = 512 * 64 + 64 + 64 * 3 + 3
    with pytest.raises(ValueError, match=r"--added_layers 1") as info:
        apply_top_down_budget(model, head_size - 1)
    assert f"{head_size:,}" in str(info.value)


def test_top_down_budget_on_frame_pool_trains_attention_head():
    model = VideoFramePoolClassifier(adjustedResNet, "dummy", 3, 2, 8, pooling="attention")
    summary = apply_top_down_budget(model, 50_000)
    assert summary["trainable"] <= 50_000
    assert model.attention.weight.requires_grad and model.backbone.fc2.weight.requires_grad
    assert summary["breakdown"]["head"]["trainable"] == summary["breakdown"]["head"]["total"]


# ---------------------------------------------------------------------------
# choose_lora_rank (PEFT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [peft_video3d, peft_frame_pool], ids=["dummy_video3d", "frame_pool_resnet"])
def test_lora_rank_line_predicts_real_counts(build):
    slope, intercept = lora_rank_line(build(8), 8)
    assert slope > 0
    for rank in (2, 5, 11):
        assert slope * rank + intercept == trainable_count(build(rank))


@pytest.mark.parametrize("build", [peft_video3d, peft_frame_pool], ids=["dummy_video3d", "frame_pool_resnet"])
def test_choose_lora_rank_picks_largest_fitting_rank(build):
    slope, intercept = lora_rank_line(build(8), 8)
    budget = slope * 5 + intercept + slope - 1  # rank 5 fits, rank 6 is one parameter over
    calls = []

    def counted(rank):
        calls.append(rank)
        return build(rank)

    model, rank = choose_lora_rank(counted, budget, 8)
    assert rank == 5
    assert calls == [8, 5]
    assert trainable_count(model) <= budget
    assert trainable_count(build(6)) > budget
    assert model(torch.randn(1, 2, 3, 32, 32)).shape == (1, 3)


def test_choose_lora_rank_keeps_start_rank_when_it_is_the_answer():
    slope, intercept = lora_rank_line(peft_video3d(4), 4)
    calls = []
    model, rank = choose_lora_rank(lambda r: calls.append(r) or peft_video3d(r), slope * 4 + intercept, 4)
    assert rank == 4 and calls == [4]


def test_choose_lora_rank_survives_rank_equal_to_num_classes():
    # start_rank == num_classes, so fc2.bias (shape [num_classes]) looks rank-shaped;
    # the real rebuilt count is still checked against the budget.
    build = lambda r: peft_video3d(r, num_classes=3)
    true_slope, true_intercept = lora_rank_line(build(8), 8)
    budget = true_slope * 6 + true_intercept
    model, rank = choose_lora_rank(build, budget, 3)
    assert trainable_count(model) <= budget
    assert rank == 6


@pytest.mark.parametrize("build", [peft_video3d, peft_frame_pool], ids=["dummy_video3d", "frame_pool_resnet"])
def test_choose_lora_rank_too_small_budget_raises(build):
    slope, intercept = lora_rank_line(build(4), 4)
    with pytest.raises(ValueError, match="LoRA rank 1") as info:
        choose_lora_rank(build, slope + intercept - 1, 4)
    assert f"{slope + intercept:,}" in str(info.value)


def test_build_with_budget_reports_rank_and_overrides_freeze(capsys):
    slope, intercept = lora_rank_line(peft_video3d(4), 4)
    model, embed_size = build_with_budget(peft_video3d, slope * 3 + intercept, True, 4, freeze_backbone=True)
    out = capsys.readouterr().out
    assert embed_size == 3 and "overrides --freeze-backbone" in out and "trainable params:" in out
    model, embed_size = build_with_budget(lambda r: adjustedVideo3D(3, "dummy_video3d", 2, r), 100_000, False, 16)
    assert embed_size == 16 and trainable_count(model) <= 100_000


# ---------------------------------------------------------------------------
# End to end: trainers/video/train.py::train on tiny synthetic clips
# ---------------------------------------------------------------------------

def _write_clip(path: Path, frames=6, size=(32, 32), seed=0) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    rng = np.random.default_rng(seed)
    for _ in range(frames):
        writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def tiny_video_splits(tmp_path):
    clips = [_write_clip(tmp_path / f"clip{i}.mp4", seed=i) for i in range(4)]
    frame = pd.DataFrame({"videos": clips, "labels": [0, 1, 0, 1]})
    train_df, val_df = tmp_path / "train.parquet", tmp_path / "val.parquet"
    frame.to_parquet(train_df)
    frame.iloc[:2].to_parquet(val_df)
    return train_df, val_df


@pytest.mark.parametrize("use_peft", [True, False], ids=["peft", "top_down"])
def test_video_train_with_budget_saves_rebuildable_run(tiny_video_splits, tmp_path, monkeypatch, use_peft):
    import trainers.video.train as video_train
    monkeypatch.setattr(video_train, "transformations", None)  # keep the 32x32 frames: fast on CPU
    train_df, val_df = tiny_video_splits
    start_rank = 16
    if use_peft:
        slope, intercept = lora_rank_line(build_video_model("dummy_video3d", 2, 2, start_rank, False, "cls", use_peft=True), start_rank)
        budget = slope * 3 + intercept
    else:
        budget = 120_000
    argv = ["--out", str(tmp_path), "--model_version", "dummy_video3d", "--num_frames", "4",
            "--trainable_params", str(budget), "--embed_size", str(start_rank), "--added_layers", "2"]
    if use_peft:
        argv.append("--use-peft")
    args = DeepTuneVisionOptions(RunType.TRAIN, args=argv)

    weights = video_train.train(
        train_df=train_df, val_df=val_df, out=tmp_path / "out", freeze_backbone=False, use_peft=use_peft,
        fixed_seed=0, mode="cls", model_version="dummy_video3d", batch_size=2, num_epochs=1,
        learning_rate=1e-3, added_layers=2, num_classes=2, embed_size=start_rank, model_str="budget",
        args=args, num_frames=4, pooling="mean", trainable_params=args.trainable_params,
    )

    run_dir = Path(weights).parent
    saved = json.loads((run_dir / "cli_arguments.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "parameter_summary.json").read_text(encoding="utf-8"))
    assert saved["trainable_params"] == budget
    assert summary["trainable"] <= budget
    assert saved["embed_size"] == (3 if use_peft else start_rank)

    rebuilt = build_video_model("dummy_video3d", 2, saved["added_layers"], saved["embed_size"], False, "cls", use_peft=use_peft)
    rebuilt.load_state_dict(torch.load(weights, map_location="cpu"))
    assert summary["total"] == sum(p.numel() for p in rebuilt.parameters())


def test_video_train_budget_too_small_raises(tiny_video_splits, tmp_path, monkeypatch):
    import trainers.video.train as video_train
    monkeypatch.setattr(video_train, "transformations", None)
    train_df, val_df = tiny_video_splits
    args = DeepTuneVisionOptions(RunType.TRAIN, args=["--out", str(tmp_path), "--model_version", "dummy_video3d", "--trainable_params", "100"])
    with pytest.raises(ValueError, match="head alone"):
        video_train.train(
            train_df=train_df, val_df=val_df, out=tmp_path / "out", freeze_backbone=False, use_peft=False,
            fixed_seed=0, mode="cls", model_version="dummy_video3d", batch_size=2, num_epochs=1,
            learning_rate=1e-3, added_layers=2, num_classes=2, embed_size=16, model_str="budget",
            args=args, num_frames=4, trainable_params=args.trainable_params,
        )


def test_image_train_siglip_with_budget_raises(tmp_path):
    from unittest.mock import MagicMock
    from trainers.vision.train import train as train_images
    args = MagicMock(model_architecture="siglip")
    with pytest.raises(ValueError, match="SigLIP"):
        train_images(
            train_df=tmp_path / "t.parquet", val_df=tmp_path / "v.parquet", out=tmp_path, freeze_backbone=False,
            use_peft=False, fixed_seed=0, mode="cls", model_version="siglip", batch_size=2, num_epochs=1,
            learning_rate=1e-3, added_layers=1, num_classes=2, embed_size=8, model_str="siglip", args=args,
            trainable_params=1000,
        )


def test_image_train_with_budget_uses_top_down(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import trainers.vision.train as vision_train

    class FakeDataset:
        @staticmethod
        def from_parquet(parquet_file, transform):
            return TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))

    monkeypatch.setattr(vision_train, "ParquetImageDataset", FakeDataset)
    args = MagicMock(model_architecture="resnet")
    monkeypatch.setattr(vision_train, "get_model_cls", lambda model_architecture, use_peft: adjustedResNet)
    weights = vision_train.train(
        train_df=tmp_path / "t.parquet", val_df=tmp_path / "v.parquet", out=tmp_path, freeze_backbone=True,
        use_peft=False, fixed_seed=0, mode="cls", model_version="dummy", batch_size=2, num_epochs=1,
        learning_rate=1e-3, added_layers=1, num_classes=2, embed_size=8, model_str="img", args=args,
        trainable_params=300_000,
    )
    summary = json.loads((Path(weights).parent / "parameter_summary.json").read_text(encoding="utf-8"))
    assert 1026 < summary["trainable"] <= 300_000  # more than the 512*2+2 head: top backbone layers train too
    args.save_args.assert_called_once()


def test_onecall_rejects_budget_for_other_modalities(tmp_path, capsys):
    import deeptune
    data = tmp_path / "data.parquet"
    pd.DataFrame({"text": ["a", "b"], "labels": [0, 1]}).to_parquet(data)
    with pytest.raises(SystemExit):
        deeptune.main(["--modality", "text", "--df", str(data), "--out", str(tmp_path), "--model_version", "BERT",
                       "--trainable_params", "100k"])
    assert "only supported for the images and video" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Desktop GUI widget (Dear PyGui context, no viewport)
# ---------------------------------------------------------------------------

@pytest.fixture
def ui():
    from desktop import app
    app.dpg.create_context()
    app.build_ui()
    try:
        yield app
    finally:
        app.stop_video_playback(timeout=2)
        app.dpg.destroy_context()


def test_gui_budget_field_round_trips_and_is_shown_for_images_and_video(ui, tmp_path):
    from desktop.config import RunConfig
    cfg = RunConfig(modality="video", df=tmp_path, out=tmp_path / "out", model_version="resnet18",
                    trainable_params="100k")
    ui.apply_config(cfg)
    assert ui._build_run_config() == cfg
    assert "--trainable_params 100k" in ui.dpg.get_value("command_text")
    for modality in ui.gc.MODALITIES:
        ui.dpg.set_value("modality_radio", modality)
        ui.on_modality_changed()
        shown = ui.dpg.get_item_configuration("trainable_params_group")["show"]
        assert shown == (modality in ("images", "video"))

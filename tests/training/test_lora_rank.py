"""--lora_rank on native video PEFT models: the rank no longer sets the head's
width, runs recorded without it rebuild as before, and it reaches every rebuild
path (V-040). CPU, weights=None."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchvision

from cli import DEFAULT_LORA_RANK, DeepTuneVisionOptions
from desktop.config import InferenceConfig, RunResults, build_infer_argv, infer_config_from_run
from src.video.video3d_peft import adjustedPeftVideo3D
from trainers.param_budget import build_with_budget, summarize_parameters
from utils import RunType, build_video_model, resolve_lora_rank

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])


def _trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _rank_of(model):
    return model.lora_config.r


def _native(embed_size=8, lora_rank=None, added_layers=2, num_classes=3):
    return build_video_model('dummy_video3d', num_classes, added_layers, embed_size, False, 'cls', use_peft=True,
                             lora_rank=lora_rank)


@pytest.mark.parametrize('rank', [1, 4])
def test_head_width_does_not_depend_on_the_rank(rank):
    model = _native(embed_size=32, lora_rank=rank)
    assert model.fc1.out_features == 32 and model.fc2.in_features == 32
    assert _rank_of(model) == rank
    assert all(p.shape[0] == rank for n, p in model.named_parameters() if '.lora_A.' in n)


def test_without_a_rank_it_equals_embed_size_as_in_old_runs():
    model = _native(embed_size=5)
    assert _rank_of(model) == 5 and model.fc1.out_features == 5


def test_budget_searches_the_rank_only_with_many_classes():
    """The review's case: 101 classes and 1M used to give fc1 (10, 512) and fc2 (101, 10)."""
    built = []

    def build(rank):
        built.append(rank)
        return _native(embed_size=256, lora_rank=rank, num_classes=101)

    model, rank = build_with_budget(build, 1_000_000, True, 256, rank_is_head_width=False, log=lambda _: None)
    assert (model.fc1.in_features, model.fc1.out_features) == (512, 256)
    assert (model.fc2.in_features, model.fc2.out_features) == (256, 101)
    assert _rank_of(model) == rank == 8
    assert _trainable(model) <= 1_000_000
    assert summarize_parameters(_native(embed_size=256, lora_rank=rank + 1, num_classes=101))['trainable'] > 1_000_000
    assert built[:2] == [1, 2]


def test_a_budget_that_cannot_fit_is_an_error_not_a_narrow_head():
    head_only_budget = 100_000  # the 1000-wide head alone needs more than 500k
    with pytest.raises(ValueError, match='LoRA rank 1') as info:
        build_with_budget(lambda rank: _native(embed_size=1000, lora_rank=rank), head_only_budget, True, 1000,
                          rank_is_head_width=False, log=lambda _: None)
    assert '--embed_size' in str(info.value) and '--added_layers 1' in str(info.value)


def _train_args(tmp_path, *extra):
    return DeepTuneVisionOptions(RunType.TRAIN, args=['--out', str(tmp_path), *extra])


def test_cli_defaults_to_16_and_records_it_for_native_peft(tmp_path):
    args = _train_args(tmp_path, '--model_version', 'r3d_18', '--use-peft')
    assert args.lora_rank == DEFAULT_LORA_RANK == 16
    args.save_args(tmp_path)
    assert json.loads((tmp_path / 'cli_arguments.json').read_text(encoding='utf-8'))['lora_rank'] == 16


def test_cli_takes_an_explicit_rank(tmp_path):
    assert _train_args(tmp_path, '--model_version', 'swin3d_t', '--use-peft', '--lora_rank', '4').lora_rank == 4


@pytest.mark.parametrize('bad', ['0', '-1', 'x'])
def test_cli_rejects_a_bad_rank(tmp_path, bad):
    with pytest.raises(SystemExit):
        _train_args(tmp_path, '--model_version', 'r3d_18', '--use-peft', f'--lora_rank={bad}')


@pytest.mark.parametrize('flags, reason', [
    (['--model_version', 'resnet18', '--use-peft'], 'uses --embed_size as its LoRA rank'),
    (['--model_version', 'r3d_18'], 'does not use --use-peft'),
])
def test_cli_says_when_the_rank_is_ignored_and_records_none(tmp_path, capsys, flags, reason):
    args = _train_args(tmp_path, *flags, '--lora_rank', '4')
    out = capsys.readouterr().out
    assert '--lora_rank 4 is ignored' in out and reason in out
    assert args.lora_rank is None


def test_cli_is_silent_on_the_rank_when_it_was_not_given(tmp_path, capsys):
    args = _train_args(tmp_path, '--model_version', 'resnet18', '--use-peft')
    assert args.lora_rank is None and 'lora_rank' not in capsys.readouterr().out


@pytest.mark.parametrize('run_type', [RunType.EVAL, RunType.EMBED])
def test_eval_and_embed_leave_an_absent_rank_to_the_record(tmp_path, run_type):
    flags = ['--use_case', 'peft'] if run_type == RunType.EMBED else []
    args = DeepTuneVisionOptions(run_type, args=['--out', str(tmp_path), '--model_version', 'r3d_18', '--use-peft', *flags])
    assert args.lora_rank is None


def test_build_video_model_says_when_it_ignores_the_rank(capsys):
    from src.vision.resnet_peft import adjustedPeftResNet  # noqa: F401  (registered under 'resnet')
    model = build_video_model('dummy_video3d', 2, 1, 4, False, 'cls', use_peft=False, lora_rank=7)
    assert '--lora_rank 7 is ignored' in capsys.readouterr().out
    assert not any('lora_' in name for name, _ in model.named_parameters())


def test_resolve_lora_rank_prefers_the_flag_then_the_record(tmp_path):
    class Args:
        lora_rank = None

    checkpoint = tmp_path / 'model_weights.pth'
    assert resolve_lora_rank(Args(), checkpoint) is None
    (tmp_path / 'cli_arguments.json').write_text(json.dumps({'embed_size': 8}), encoding='utf-8')
    assert resolve_lora_rank(Args(), checkpoint) is None, 'an old record has no rank'
    (tmp_path / 'cli_arguments.json').write_text(json.dumps({'lora_rank': 3}), encoding='utf-8')
    assert resolve_lora_rank(Args(), checkpoint) == 3
    Args.lora_rank = 5
    assert resolve_lora_rank(Args(), checkpoint) == 5


def test_a_mocked_args_object_is_not_taken_for_a_rank():
    from unittest.mock import MagicMock
    assert resolve_lora_rank(MagicMock()) is None


# --- rebuilding from a run ----------------------------------------------------
def _clip(path: Path, level: int) -> bytes:
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (32, 32))
    for i in range(6):
        writer.write(np.full((32, 32, 3), (level + 7 * i) % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def clips(tmp_path):
    frame = pd.DataFrame({'videos': [_clip(tmp_path / f'c{i}.mp4', 40 + 60 * i) for i in range(4)], 'labels': [0, 1, 0, 1]})
    path = tmp_path / 'clips.parquet'
    frame.to_parquet(path)
    return path


class _Args:
    """No lora_rank: as an evaluation started without the flag."""

    def save_args(self, outdir):
        Path(outdir, 'cli_arguments.json').write_text('{}')


def _saved_run(tmp_path, record, **build):
    run = tmp_path / 'trainval'
    run.mkdir()
    torch.manual_seed(0)
    model = build_video_model('dummy_video3d', 2, 2, 8, False, 'cls', use_peft=True, **build)
    checkpoint = run / 'model_weights.pth'
    torch.save(model.state_dict(), checkpoint)
    (run / 'cli_arguments.json').write_text(json.dumps(record), encoding='utf-8')
    return model.eval(), checkpoint


@pytest.mark.parametrize('record, build, expected', [
    ({'embed_size': 8}, {}, 8),
    ({'embed_size': 8, 'lora_rank': 2}, {'lora_rank': 2}, 2),
], ids=['old_record', 'new_record'])
def test_a_run_rebuilds_for_eval_embed_and_infer_from_its_record(tmp_path, monkeypatch, clips, record, build, expected):
    import embed.video.embed as video_embed
    import evaluators.video.evaluate as video_eval
    import infer
    import inference.video as video_inference

    for module in (video_eval, video_embed, video_inference):
        monkeypatch.setattr(module, 'transformations', SMALL)
    model, checkpoint = _saved_run(tmp_path, record, **build)
    assert model.lora_config.r == expected

    seen = []
    real_build = video_eval.build_video_model

    def spying_build(**kwargs):
        built = real_build(**kwargs)
        seen.append((built.lora_config.r, built.fc1.out_features))
        return built

    for module in (video_eval, video_embed, video_inference):
        monkeypatch.setattr(module, 'build_video_model', spying_build)

    video_eval.evaluate(eval_df=clips, out=tmp_path, model_weights=checkpoint, num_classes=2,
                        model_version='dummy_video3d', mode='cls', added_layers=2, embed_size=8, batch_size=2,
                        freeze_backbone=False, args=_Args(), use_peft=True, model_str='eval', num_frames=2)

    args = _Args()
    args.video_backend = 'auto'
    _, shape = video_embed.embed(df_path=clips, out=tmp_path, model_weights=checkpoint, batch_size=2, use_case='peft',
                                 model_version='dummy_video3d', added_layers=2, embed_size=8, num_classes=2, args=args,
                                 mode='cls', grouper=None, model_str='embed', num_frames=2)
    assert shape[0] == 4

    data = tmp_path / 'infer_data'
    data.mkdir()
    (data / 'one.mp4').write_bytes((tmp_path / 'c0.mp4').read_bytes())
    argv = ['--modality', 'video', '--checkpoint', str(checkpoint), '--data', str(data), '--out', str(tmp_path / 'infer'),
            '--model_version', 'dummy_video3d', '--num_classes', '2', '--added_layers', '2', '--embed_size', '8',
            '--use-peft', '--num_frames', '2', '--batch_size', '1']
    assert infer.main(argv) == 0
    assert (tmp_path / 'infer' / 'predictions.csv').is_file()

    assert seen == [(expected, 8)] * 3


def test_infer_flag_wins_over_the_record(tmp_path):
    import infer
    (tmp_path / 'cli_arguments.json').write_text(json.dumps({'lora_rank': 3}), encoding='utf-8')
    base = ['--modality', 'video', '--checkpoint', str(tmp_path / 'w.pth'), '--data', '.', '--out', '.']
    args = infer.build_parser().parse_args(base)
    assert infer.apply_recorded_video_settings(args, base)['lora_rank'] == 3 and args.lora_rank == 3
    given = base + ['--lora_rank', '5']
    args = infer.build_parser().parse_args(given)
    infer.apply_recorded_video_settings(args, given)
    assert args.lora_rank == 5


def test_desktop_test_page_passes_the_recorded_rank(tmp_path):
    record = {'modality': 'video', 'model_version': 'r3d_18', 'use_peft': True, 'embed_size': 256, 'lora_rank': 4}
    results = RunResults(run_dir=tmp_path, checkpoint_path=tmp_path / 'w.pth', cli_arguments=record)
    config = infer_config_from_run(results, tmp_path / 'data', tmp_path / 'out')
    assert config.lora_rank == 4
    argv = build_infer_argv(config)
    assert argv[argv.index('--lora_rank') + 1] == '4' and argv[argv.index('--embed_size') + 1] == '256'


def test_desktop_test_page_passes_no_rank_for_an_old_run(tmp_path):
    record = {'modality': 'video', 'model_version': 'r3d_18', 'use_peft': True, 'embed_size': 16}
    results = RunResults(run_dir=tmp_path, checkpoint_path=tmp_path / 'w.pth', cli_arguments=record)
    config = infer_config_from_run(results, tmp_path / 'data', tmp_path / 'out')
    assert config.lora_rank is None and '--lora_rank' not in build_infer_argv(config)
    assert InferenceConfig().lora_rank is None


def test_a_saved_rank_that_does_not_match_gets_a_clear_message():
    state = adjustedPeftVideo3D(3, 'dummy_video3d', 2, 8, lora_rank=2).state_dict()
    with pytest.raises(RuntimeError, match='--lora_rank'):
        adjustedPeftVideo3D(3, 'dummy_video3d', 2, 8, lora_rank=4).load_state_dict(state)

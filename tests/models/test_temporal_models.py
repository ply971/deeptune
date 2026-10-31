"""Phase 3 temporal architectures: shapes, freezing, checkpoints, inflation, TSM, G5, old checkpoints, flags. CPU, tiny."""
import json
import subprocess
import sys

import pytest
import torch

from handlers.video_arch import format_pooling_spec, parse_pooling_spec, pooling_from_args
from src.video.frame_pool import VideoFramePoolClassifier
from src.video.temporal import VideoTSM, inflate_conv, temporal_shift
from src.vision.resnet import adjustedResNet
from utils import build_video_model

ARCHES = ['late_fusion', 'early_fusion:num_frames=4', 'tsm', 'cnn_rnn', 'cnn_rnn:rnn_type=gru,bidirectional=1,rnn_layers=2',
          'temporal_transformer:num_frames=4,tt_dim=16,tt_heads=2', 'two_stream:num_frames=4',
          'two_stream:num_frames=4,fusion=fc,motion_length=2']


def _clip(spec, batch=2, frames=4, size=32):
    channels = 5 if spec.startswith('two_stream') else 3
    return torch.randn(batch, frames, channels, size, size)


def _build(spec, mode='cls', added_layers=2, classes=3, freeze=False, version='resnet_tiny', peft=False, embed=8):
    torch.manual_seed(0)
    return build_video_model(version, classes, added_layers, embed, freeze, mode, use_peft=peft, pooling=spec)


@pytest.mark.parametrize('spec', ARCHES)
@pytest.mark.parametrize('mode,out', [('cls', 3), ('reg', 1)])
@pytest.mark.parametrize('added_layers', [1, 2])
def test_output_and_embedding_shapes(spec, mode, out, added_layers):
    model = _build(spec, mode, added_layers).eval()
    x = _clip(spec)
    assert model(x).shape == (2, out)
    embedding = model(x, extract_embed=True)
    assert embedding.shape == (2, model.embedding_dim)


@pytest.mark.parametrize('spec', ARCHES)
def test_frozen_backbone_stays_frozen_and_head_trains(spec):
    # The inflated first convolutions are the one exception (V-046): test_inflated_first_conv_trains_when_frozen.
    model = _build(spec, freeze=True)
    inflated = getattr(model, 'always_trainable', ())
    names = {n: p.requires_grad for n, p in model.named_parameters() if not n.startswith(inflated)}
    backbone = {n: g for n, g in names.items() if n.startswith('backbone.model.')}
    head = {n: g for n, g in names.items() if not n.startswith('backbone.')}
    assert backbone and not any(backbone.values())
    assert head and all(head.values())
    before = {n: p.clone() for n, p in model.named_parameters() if n.startswith('backbone.model.') and not n.startswith(inflated)}
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1).zero_grad()
    model(_clip(spec)).sum().backward()
    assert all(p.grad is None for n, p in model.named_parameters() if n in before)


@pytest.mark.parametrize('spec', ARCHES)
def test_checkpoint_round_trip_gives_the_same_output(spec, tmp_path):
    model = _build(spec).eval()
    path = tmp_path / 'm.pth'
    torch.save(model.state_dict(), path)
    torch.manual_seed(99)
    again = build_video_model('resnet_tiny', 3, 2, 8, False, 'cls', pooling=spec)
    again.load_state_dict(torch.load(path, weights_only=True))
    x = _clip(spec)
    assert torch.equal(model.eval()(x), again.eval()(x))


@pytest.mark.parametrize('spec', ARCHES)
def test_peft_backbones_train_few_parameters(spec):
    """'dummy' is an untrained resnet18. LoRA on the backbone, the rest frozen."""
    model = _build(spec, version='dummy', peft=True, embed=2)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable < 0.2 * total
    assert model.eval()(_clip(spec)).shape == (2, 3)


# --- inflation -------------------------------------------------------------------------------------------------------
def test_early_fusion_on_identical_frames_equals_the_2d_backbone():
    torch.manual_seed(0)
    plain = adjustedResNet(3, 'resnet_tiny', 1, 8, False).eval()
    model = _build('early_fusion:num_frames=4').eval()
    frame = torch.randn(2, 3, 32, 32)
    clip = frame.unsqueeze(1).repeat(1, 4, 1, 1, 1)
    with torch.no_grad():
        assert torch.allclose(model.backbone(clip.flatten(1, 2), extract_embed=True), plain.model(frame), atol=1e-5)


def test_inflating_rgb_to_a_non_multiple_is_refused():
    with pytest.raises(ValueError, match='multiple'):
        inflate_conv(torch.nn.Conv2d(3, 4, 3), 7)


@pytest.mark.parametrize('length', [3, 4])  # 2L = 6 is a multiple of 3 and must start the same way as 2L = 8
def test_motion_stream_start_is_the_mean_kernel_for_every_stack_length(length):
    conv = torch.nn.Conv2d(3, 4, 3)
    new = inflate_conv(conv, 2 * length, cross_modality=True)
    assert torch.allclose(new.weight.sum(dim=1), conv.weight.sum(dim=1), atol=1e-5)
    assert torch.allclose(new.weight[:, 0], new.weight[:, -1])


@pytest.mark.parametrize('spec', ['early_fusion:num_frames=4', 'tsm', 'two_stream:num_frames=4'])
def test_resnet_only_architectures_refuse_other_backbones_clearly(spec):
    with pytest.raises(ValueError, match='ResNet'):
        _build(spec, version='siglip_tiny')


def test_temporal_architectures_refuse_native_3d_models():
    with pytest.raises(ValueError, match='2D image backbone'):
        _build('tsm', version='dummy_video3d')


# --- TSM -------------------------------------------------------------------------------------------------------------
def test_shift_moves_channel_content_exactly_one_step():
    x = torch.arange(1.0, 5.0).reshape(1, 4, 1, 1, 1).repeat(1, 1, 8, 1, 1).reshape(4, 8, 1, 1)  # frame t holds t+1
    out = temporal_shift(x, 4, 8).reshape(4, 8)
    assert out[:, 0].tolist() == [2, 3, 4, 0]  # first fold reads the next frame
    assert out[:, 1].tolist() == [0, 1, 2, 3]  # second fold reads the previous frame
    assert torch.equal(out[:, 2:], x.reshape(4, 8)[:, 2:])  # the rest stays


def test_shift_on_identical_frames_changes_only_the_vacated_end_frames():
    """Vacated positions are zero, as in the paper, so identical frames are unchanged inside the clip only."""
    x = torch.randn(4, 8, 2, 2).mean(0, keepdim=True).repeat(4, 1, 1, 1)
    out = temporal_shift(x, 4, 8)
    assert torch.equal(out[1:3], x[1:3])
    assert torch.count_nonzero(out[0, 1]) == 0 and torch.count_nonzero(out[3, 0]) == 0


def test_tsm_that_shifts_nothing_equals_the_2d_backbone():
    torch.manual_seed(0)
    plain = adjustedResNet(3, 'resnet_tiny', 1, 8, False).eval()
    model = _build('tsm:tsm_fold_div=1000').eval()
    x = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        per_frame = plain.model(x.reshape(8, 3, 32, 32)).reshape(2, 4, -1).mean(1)
        assert torch.allclose(model.head.embed(per_frame), model(x, extract_embed=True), atol=1e-5)


def test_tsm_adds_no_parameters_and_keeps_the_2d_names():
    plain = adjustedResNet(3, 'resnet_tiny', 1, 8, False)
    model = _build('tsm')
    assert isinstance(model, VideoTSM)
    plain_names = {n: p.shape for n, p in plain.model.named_parameters()}
    model_names = {n[len('backbone.model.'):]: p.shape for n, p in model.named_parameters() if n.startswith('backbone.model.')}
    assert plain_names == model_names
    plain.model.load_state_dict({n[len('backbone.model.'):]: v for n, v in model.state_dict().items()
                                 if n.startswith('backbone.model.')})


def test_tsm_survives_deepcopy():
    """EMA and budget search copy models: the copy's hooks must follow the copy."""
    import copy
    model = _build('tsm').eval()
    twin = copy.deepcopy(model)
    x = _clip('tsm')
    assert torch.equal(model(x), twin(x))
    assert twin(x[:, :2]).shape == (2, 3)  # sets only the twin's frame count
    assert model._shift_state['num_frames'] == 4


# --- frame order -----------------------------------------------------------------------------------------------------
# Not early_fusion: its inflated kernels start identical across time, so it is order-blind until trained
# (the oracle test checks the trained model).
@pytest.mark.parametrize('spec', ['tsm', 'cnn_rnn', 'temporal_transformer:num_frames=4,tt_dim=16,tt_heads=2', 'late_fusion'])
def test_temporal_models_depend_on_frame_order(spec):
    model = _build(spec).eval()
    x = _clip(spec)
    assert not torch.allclose(model(x), model(x.flip(1)), atol=1e-4)


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_frame_pooling_ignores_frame_order(pooling):
    model = _build(pooling).eval()
    x = _clip(pooling)
    assert torch.allclose(model(x), model(x.flip(1)), atol=1e-5)


# --- two-stream ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('fusion', ['avg', 'fc'])
def test_two_stream_returns_each_stream_separately(fusion):
    model = _build(f'two_stream:num_frames=4,fusion={fusion}').eval()
    x = _clip('two_stream')
    separate = model(x, extract_embed='separate')
    dims = model.embedding_split
    assert separate['rgb'].shape == (2, dims['rgb'][1] - dims['rgb'][0])
    assert separate['motion'].shape == (2, dims['motion'][1] - dims['motion'][0])
    if fusion == 'avg':
        assert torch.equal(torch.cat([separate['rgb'], separate['motion']], dim=1), model(x, extract_embed=True))


def test_two_stream_streams_read_only_their_own_channels():
    model = _build('two_stream:num_frames=4').eval()
    x = _clip('two_stream')
    other_rgb, other_motion = x.clone(), x.clone()
    other_rgb[:, :, :3] = 0
    other_motion[:, :, 3:] = 0
    base = model(x, extract_embed='separate')
    assert torch.equal(model(other_motion, extract_embed='separate')['rgb'], base['rgb'])
    assert not torch.allclose(model(other_motion, extract_embed='separate')['motion'], base['motion'], atol=1e-4)
    assert torch.equal(model(other_rgb, extract_embed='separate')['motion'], base['motion'])


def test_two_stream_refuses_a_clip_without_motion_and_a_clip_shorter_than_the_stack():
    model = _build('two_stream:num_frames=4').eval()
    with pytest.raises(ValueError, match='clip_motion'):
        model(torch.randn(1, 4, 3, 32, 32))
    with pytest.raises(ValueError, match='stacks 4'):
        model(torch.randn(1, 2, 5, 32, 32))


# --- frame pooling on embeddings (G5) and old checkpoints ------------------------------------------------------------
def _pool(pooling, added_layers=2, mode='cls', classes=3):
    torch.manual_seed(0)
    return VideoFramePoolClassifier(adjustedResNet, 'resnet_tiny', classes, added_layers, 8, False, task_type=mode, pooling=pooling).eval()


@pytest.mark.parametrize('added_layers,size', [(2, 8), (1, 256)])
def test_attention_weighs_embeddings_and_the_head_classifies_the_pooled_embedding(added_layers, size):
    model = _pool('attention', added_layers)
    assert model.attention.in_features == size
    x = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        frames = model.backbone(x.reshape(8, 3, 32, 32), extract_embed=True).reshape(2, 4, -1)
        pooled = (frames * torch.softmax(model.attention(frames), dim=1)).sum(dim=1)
        assert torch.allclose(model(x, extract_embed=True), pooled)
        head = model.backbone.fc2(torch.relu(pooled)) if added_layers == 2 else model.backbone.fc1(pooled)
        assert torch.allclose(model(x), head)


def _old_state(model, pooling, classes=3):
    """The state dict the model of before G5 saved: no marker; attention over class logits."""
    state = {k: v for k, v in model.state_dict().items() if k != 'pool_format'}
    if pooling == 'attention':
        torch.manual_seed(5)
        old = torch.nn.Linear(classes, 1)
        state.update({'attention.weight': old.weight.detach(), 'attention.bias': old.bias.detach()})
    return state


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_old_checkpoints_load_and_predict_as_they_did(pooling):
    trained = _pool(pooling)
    state = _old_state(trained, pooling)
    torch.manual_seed(1)
    model = VideoFramePoolClassifier(adjustedResNet, 'resnet_tiny', 3, 2, 8, False, pooling=pooling)
    model.load_state_dict(state)
    model.eval()
    x = torch.randn(2, 4, 3, 32, 32)
    with torch.no_grad():
        logits = trained.backbone(x.reshape(8, 3, 32, 32)).reshape(2, 4, -1)
        if pooling == 'mean':
            expected = logits.mean(dim=1)
        else:
            weights = torch.softmax(logits @ state['attention.weight'].T + state['attention.bias'], dim=1)
            expected = (logits * weights).sum(dim=1)
        assert torch.allclose(model(x), expected, atol=1e-5)
    # Same keys as the file, so the run's error model fingerprint still matches.
    assert set(model.state_dict()) == set(state)


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_new_checkpoints_keep_embedding_pooling(pooling, tmp_path):
    model = _pool(pooling)
    torch.save(model.state_dict(), tmp_path / 'm.pth')
    again = VideoFramePoolClassifier(adjustedResNet, 'resnet_tiny', 3, 2, 8, False, pooling=pooling)
    again.load_state_dict(torch.load(tmp_path / 'm.pth', weights_only=True))
    assert not again.logit_pooling
    x = torch.randn(1, 4, 3, 32, 32)
    assert torch.equal(model(x), again.eval()(x))


# --- flags -----------------------------------------------------------------------------------------------------------
def test_pooling_spec_round_trip_and_refusals():
    spec = format_pooling_spec('cnn_rnn', rnn_type='gru', rnn_hidden=9, rnn_layers=2, bidirectional=True, tsm_fold_div=3)
    assert parse_pooling_spec(spec) == ('cnn_rnn', {'rnn_type': 'gru', 'rnn_hidden': 9, 'rnn_layers': 2, 'bidirectional': True})
    assert parse_pooling_spec('mean') == ('mean', {})
    for bad in ('nope', 'mean:x=1', 'tsm:rnn_hidden=3', 'tsm:tsm_fold_div=0', 'cnn_rnn:rnn_type=rnn', 'tsm:tsm_fold_div'):
        with pytest.raises(ValueError):
            parse_pooling_spec(bad)


def test_arch_flags_compose_the_spec_and_are_recorded(tmp_path):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    args = DeepTuneVisionOptions(RunType.ONECALL, args=[
        '--modality', 'video', '--df', 'x.parquet', '--out', str(tmp_path), '--model_version', 'resnet_tiny',
        '--num_frames', '6', '--video_arch', 'two_stream', '--fusion', 'fc', '--motion_length', '3', '--motion_source', 'mv'])
    assert parse_pooling_spec(args.pooling) == ('two_stream', {'num_frames': 6, 'fusion': 'fc', 'motion_length': 3})
    args.save_args(tmp_path)
    saved = json.loads((tmp_path / 'cli_arguments.json').read_text())
    assert saved['pooling'] == args.pooling and saved['video_arch'] == 'two_stream' and saved['motion_source'] == 'mv'
    assert saved['fusion'] == 'fc' and saved['motion_length'] == 3


def test_conflicting_pooling_and_arch_are_refused():
    with pytest.raises(ValueError, match='does not apply'):
        pooling_from_args({'video_arch': 'tsm', 'pooling': 'attention'})


def test_slow_fusion_selects_a_native_model_and_keeps_one_already_given(tmp_path):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    common = ['--modality', 'video', '--df', 'x.parquet', '--out', str(tmp_path), '--video_arch', 'slow_fusion']
    assert DeepTuneVisionOptions(RunType.ONECALL, args=[*common, '--model_version', 'resnet18']).model_version == 'r2plus1d_18'
    assert DeepTuneVisionOptions(RunType.ONECALL, args=[*common, '--model_version', 'r3d_18']).model_version == 'r3d_18'
    assert DeepTuneVisionOptions(RunType.ONECALL, args=[*common, '--model_version', 'dummy_video3d']).model_version == 'dummy_video3d'


def test_flag_modules_do_not_import_torch():
    """The desktop GUI imports these, and must never import torch."""
    code = 'import sys, handlers.video_arch, datasets.motion_stack; sys.exit(1 if "torch" in sys.modules else 0)'
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0


# --- SigLIP ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('spec', ['mean', 'attention', 'single_frame', 'late_fusion', 'cnn_rnn',
                                  'temporal_transformer:num_frames=4,tt_dim=16,tt_heads=2'])
@pytest.mark.parametrize('mode,out', [('cls', 3), ('reg', 1)])
def test_siglip_frame_backbone(spec, mode, out):
    model = _build(spec, mode, version='siglip_tiny').eval()
    x = torch.randn(2, 4, 3, 32, 32)
    assert model(x).shape == (2, out)
    assert model(x, extract_embed=True).ndim == 2


def test_siglip_lora_trains_only_the_adapters_and_the_head():
    model = _build('mean', version='siglip_tiny', peft=True, embed=2)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all('lora_' in n or 'fc' in n for n in trainable)
    assert any('lora_' in n for n in trainable)
    assert model.eval()(torch.randn(1, 4, 3, 32, 32)).shape == (1, 3)


def test_siglip_freezing_and_parameter_groups():
    from trainers.param_budget import summarize_parameters
    summary = summarize_parameters(_build('mean', version='siglip_tiny', freeze=True))
    assert summary['breakdown']['backbone']['total'] > 0 and summary['breakdown']['backbone']['trainable'] == 0
    assert summary['breakdown']['head']['trainable'] > 0


def test_siglip_preprocessing_is_the_models_own():
    from datasets.video_datasets import native_video_transform
    clip = torch.tensor([0, 255], dtype=torch.uint8).reshape(2, 1, 1, 1).repeat(1, 3, 64, 64)
    out = native_video_transform('siglip_tiny')(clip)
    assert out.shape == (2, 3, 32, 32)
    assert torch.allclose(out[0], torch.tensor(-1.0)) and torch.allclose(out[1], torch.tensor(1.0))
    assert native_video_transform('siglip').size == 384
    assert native_video_transform('resnet18') is None


# --- every 2D family ---------------------------------------------------------------------------------------------------
FAMILIES = ['resnet', 'densenet', 'swin', 'efficientnet', 'vgg', 'vit', 'convnext']


def _family_cls(family, peft=False):
    from utils import MODEL_CLS_MAP, PEFT_MODEL_CLS_MAP
    return (PEFT_MODEL_CLS_MAP if peft else MODEL_CLS_MAP)[family]


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('added_layers', [1, 2])
def test_embedding_pooling_works_on_every_family(family, added_layers):
    """G5 uses the backbone's extract_embed output and then its head, so these must agree in every family."""
    torch.manual_seed(0)
    model = VideoFramePoolClassifier(_family_cls(family), 'dummy', 3, added_layers, 4, False, pooling='attention').eval()
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        assert model(x).shape == (1, 3)
        assert model(x, extract_embed=True).shape == (1, model.embedding_dim)
        frame = x[:, 0]
        assert torch.allclose(model.backbone(frame), model._classify(model.backbone(frame, extract_embed=True)), atol=1e-4)


@pytest.mark.parametrize('peft', [False, True])
def test_densenet_with_one_added_layer_embeds_its_features_not_its_logits(peft):
    model = _family_cls('densenet', peft)(3, 'dummy', 1, 4, False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64), extract_embed=True).shape == (1, model.fc1.in_features)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('arch,options', [('late_fusion', {}), ('cnn_rnn', {'rnn_hidden': 8})])
def test_frame_based_architectures_work_on_every_family(family, arch, options):
    from src.video.temporal import build_temporal_model
    torch.manual_seed(0)
    model = build_temporal_model(arch, options, _family_cls(family), 'dummy', 3, 2, 4, False, 'cls').eval()
    with torch.no_grad():
        assert model(torch.randn(1, 2, 3, 224, 224)).shape == (1, 3)


# --- what each model reads ---------------------------------------------------------------------------------------------
def test_late_fusion_reads_only_the_first_and_last_frames():
    model = _build('late_fusion').eval()
    x = _clip('late_fusion')
    middle, last = x.clone(), x.clone()
    middle[:, 1:3] = 0
    last[:, -1] = 0
    assert torch.equal(model(x), model(middle))
    assert not torch.allclose(model(x), model(last), atol=1e-4)


@pytest.mark.parametrize('fusion', ['avg', 'fc'])
def test_two_stream_output_depends_on_both_streams(fusion):
    model = _build(f'two_stream:num_frames=4,fusion={fusion}').eval()
    x = _clip('two_stream')
    no_rgb, no_motion = x.clone(), x.clone()
    no_rgb[:, :, :3] = 0
    no_motion[:, :, 3:] = 0
    assert not torch.allclose(model(x), model(no_rgb), atol=1e-4)
    assert not torch.allclose(model(x), model(no_motion), atol=1e-4)


def test_two_stream_reads_the_middle_rgb_frame_and_the_motion_around_it():
    model = _build('two_stream:num_frames=4,motion_length=2').eval()
    x = _clip('two_stream')
    base = model(x, extract_embed='separate')

    def changed(frames, channels):
        other = x.clone()
        other[:, frames, channels] = 0
        return model(other, extract_embed='separate')
    assert torch.equal(changed([0, 1, 3], slice(0, 3))['rgb'], base['rgb'])  # T // 2 = 2 is the RGB frame
    assert not torch.allclose(changed([2], slice(0, 3))['rgb'], base['rgb'], atol=1e-4)
    assert torch.equal(changed([0, 3], slice(3, 5))['motion'], base['motion'])  # L = 2 around the middle: frames 1, 2
    for frame in (1, 2):
        assert not torch.allclose(changed([frame], slice(3, 5))['motion'], base['motion'], atol=1e-4)


def test_settings_that_cannot_work_are_refused_when_the_model_is_built():
    with pytest.raises(ValueError, match='motion_length'):
        _build('two_stream:num_frames=4,motion_length=5')
    with pytest.raises(ValueError, match='tsm_fold_div'):
        _build('tsm:tsm_fold_div=1')
    with pytest.raises(ValueError, match='tt_dim'):
        _build('temporal_transformer:num_frames=4,tt_dim=15,tt_heads=2')


# --- trainable-parameter budget ----------------------------------------------------------------------------------------
@pytest.mark.parametrize('spec', ARCHES)
def test_budget_without_peft_counts_the_new_layers_as_head(spec):
    from trainers.param_budget import build_with_budget, summarize_parameters
    head = summarize_parameters(_build(spec))['breakdown']['head']['total']
    budget = head + 2000
    model, _ = build_with_budget(lambda rank: _build(spec), budget, False, 8, log=lambda _: None)
    summary = summarize_parameters(model)
    assert head < summary['trainable'] <= budget  # the head, and part of the backbone from the output side
    assert summary['breakdown']['head']['trainable'] == head
    assert model.eval()(_clip(spec)).shape == (2, 3)


@pytest.mark.parametrize('spec', ARCHES)
def test_budget_with_peft_picks_a_rank_that_fits(spec):
    from trainers.param_budget import build_with_budget, summarize_parameters
    budget = summarize_parameters(_build(spec, version='dummy', peft=True, embed=1))['trainable'] + 50_000
    model, rank = build_with_budget(lambda r: _build(spec, version='dummy', peft=True, embed=r), budget, True, 8, log=lambda _: None)
    assert rank >= 1 and summarize_parameters(model)['trainable'] <= budget
    assert summarize_parameters(_build(spec, version='dummy', peft=True, embed=rank + 1))['trainable'] > budget
    assert model.eval()(_clip(spec)).shape == (2, 3)


# --- the standalone train, eval and embed commands -----------------------------------------------------------------------
@pytest.mark.parametrize('run_type,extra', [('TRAIN', []), ('EVAL', []), ('EMBED', ['--use_case', 'finetuned'])])
def test_standalone_commands_accept_the_architecture_flags(run_type, extra, tmp_path):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    common = ['--out', str(tmp_path), '--model_version', 'resnet_tiny', '--num_frames', '6', *extra]
    from_flag = DeepTuneVisionOptions(RunType[run_type], args=[*common, '--video_arch', 'tsm', '--tsm_fold_div', '4'])
    assert parse_pooling_spec(from_flag.pooling) == ('tsm', {'tsm_fold_div': 4})
    recorded = DeepTuneVisionOptions(RunType[run_type], args=[*common, '--pooling', 'early_fusion:num_frames=6'])
    assert parse_pooling_spec(recorded.pooling) == ('early_fusion', {'num_frames': 6})
    assert DeepTuneVisionOptions(RunType[run_type], args=common).pooling == 'mean'


# --- SigLIP: the path real weights take ----------------------------------------------------------------------------------
@pytest.fixture()
def tiny_siglip_weights(monkeypatch):
    """`--model_version siglip` loads a SiglipModel and uses its vision tower. Same classes, no download."""
    from transformers import SiglipConfig, SiglipModel
    import src.vision.siglip as siglip
    config = SiglipConfig(vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                             image_size=32, patch_size=8),
                          text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                           vocab_size=16, projection_size=32))
    monkeypatch.setattr(siglip, 'load_siglip_model_offline', lambda tiny=False: SiglipModel(config))


@pytest.mark.parametrize('peft', [False, True])
@pytest.mark.parametrize('spec', ['mean', 'late_fusion', 'cnn_rnn'])
def test_the_real_siglip_path_builds_and_runs(tiny_siglip_weights, peft, spec):
    from trainers.param_budget import summarize_parameters
    model = _build(spec, version='siglip', peft=peft, embed=2).eval()
    x = torch.randn(1, 4, 3, 32, 32)
    assert model(x).shape == (1, 3)
    assert model(x, extract_embed=True).ndim == 2
    summary = summarize_parameters(model)['breakdown']
    assert summary['backbone']['total'] > 0
    if peft:
        assert summary['lora']['trainable'] > 0 and summary['backbone']['trainable'] == 0


def test_siglip_preprocessing_matches_a_pil_bicubic_resize_when_shrinking():
    """On noise, a resize without antialiasing keeps most of the contrast; PIL, which SigLIP's processor uses, removes it."""
    import numpy as np
    from PIL import Image
    from datasets.video_datasets import native_video_transform
    noise = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    out = native_video_transform('siglip_tiny')(noise)[0]
    image = Image.fromarray(noise[0].permute(1, 2, 0).numpy()).resize((32, 32), Image.BICUBIC)
    expected = (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255 - 0.5) / 0.5
    assert (out - expected).abs().max() < 0.05


# --- frozen backbones must still see time (V-046) ------------------------------------------------------------------------
def _few_steps(model, spec):
    """A few updates on a clip whose frames differ, so the inflated kernels can stop being identical across the stack."""
    torch.manual_seed(3)
    x, y = _clip(spec, batch=4), torch.tensor([0, 1, 2, 0])
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.train()
    for _ in range(3):
        optimiser.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimiser.step()
    return model.eval()


@pytest.mark.parametrize('spec,conv', [('early_fusion:num_frames=4', 'backbone.model.conv1.'),
                                       ('two_stream:num_frames=4', 'backbone.model.motion.model.conv1.'),
                                       ('two_stream:num_frames=4,fusion=fc', 'backbone.model.motion.model.conv1.')])
def test_inflated_first_conv_trains_when_frozen(spec, conv, capsys):
    from trainers.param_budget import summarize_parameters
    model = _build(spec, freeze=True)
    assert 'stays trainable' in capsys.readouterr().out
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    assert [n for n, _ in trainable] == [conv + 'weight']
    summary = summarize_parameters(model)
    assert summary['breakdown']['backbone']['trainable'] == trainable[0][1].numel()  # and so in the reported count
    assert summary['trainable'] == summary['breakdown']['head']['total'] + trainable[0][1].numel()


def test_frozen_early_fusion_becomes_order_aware_by_training():
    model = _build('early_fusion:num_frames=4', freeze=True).eval()
    x = _clip('early_fusion')
    assert torch.allclose(model(x), model(x.flip(1)), atol=1e-4)  # the identical kernels: why the layer must train
    _few_steps(model, 'early_fusion')
    assert not torch.allclose(model(x), model(x.flip(1)), atol=1e-3)


@pytest.mark.parametrize('fusion', ['avg', 'fc'])
def test_frozen_two_stream_motion_stream_learns_order_and_axes(fusion):
    model = _build(f'two_stream:num_frames=4,fusion={fusion}', freeze=True).eval()
    x = _clip('two_stream')
    reordered, swapped = x.clone(), x.clone()
    reordered[:, :, 3:] = x[:, :, 3:].flip(1)
    swapped[:, :, 3:] = x[:, :, [4, 3]]
    motion = lambda clip: model(clip, extract_embed='separate')['motion']
    assert torch.allclose(motion(x), motion(reordered), atol=1e-4)
    assert torch.allclose(motion(x), motion(swapped), atol=1e-4)  # at the start: blind to order and to dx against dy
    _few_steps(model, 'two_stream')
    assert not torch.allclose(motion(x), motion(reordered), atol=1e-3)
    assert not torch.allclose(motion(x), motion(swapped), atol=1e-3)
    assert not torch.allclose(model(x), model(reordered), atol=1e-3)


@pytest.mark.parametrize('spec,conv', [('early_fusion:num_frames=4', 'backbone.model.conv1.weight'),
                                       ('two_stream:num_frames=4', 'backbone.model.motion.model.conv1.weight')])
def test_a_budget_keeps_the_inflated_conv_trainable(spec, conv):
    from trainers.param_budget import apply_top_down_budget, summarize_parameters
    model = _build(spec)
    params = dict(model.named_parameters())
    head = summarize_parameters(model)['breakdown']['head']['total']
    apply_top_down_budget(model, head + params[conv].numel())
    assert params[conv].requires_grad
    assert summarize_parameters(model)['trainable'] == head + params[conv].numel()
    with pytest.raises(ValueError, match='budget'):
        apply_top_down_budget(_build(spec), head + params[conv].numel() - 1)


# --- a recorded spec wins unless the option flag is given (V-046) ----------------------------------------------------------
def test_recorded_options_survive_and_explicit_flags_replace_them():
    recorded = {'video_arch': 'cnn_rnn', 'pooling': 'cnn_rnn:bidirectional=1,rnn_hidden=9,rnn_layers=2,rnn_type=gru'}
    assert parse_pooling_spec(pooling_from_args(recorded))[1] == {'bidirectional': True, 'rnn_hidden': 9, 'rnn_layers': 2, 'rnn_type': 'gru'}
    changed = parse_pooling_spec(pooling_from_args({**recorded, 'rnn_hidden': 16}))[1]
    assert changed == {'bidirectional': True, 'rnn_hidden': 16, 'rnn_layers': 2, 'rnn_type': 'gru'}
    assert parse_pooling_spec(pooling_from_args({'video_arch': 'tsm', 'pooling': 'mean'}))[1] == {'tsm_fold_div': 8}  # the default


def test_recorded_frame_count_wins_unless_given():
    args = {'video_arch': 'early_fusion', 'pooling': 'early_fusion:num_frames=4', 'num_frames': 8}
    assert parse_pooling_spec(pooling_from_args(args, num_frames_given=False))[1] == {'num_frames': 4}
    assert parse_pooling_spec(pooling_from_args(args, num_frames_given=True))[1] == {'num_frames': 8}


@pytest.mark.parametrize('given,fold', [([], 4), (['--tsm_fold_div', '8'], 8)])
def test_the_cli_keeps_a_recorded_spec(tmp_path, given, fold):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    options = DeepTuneVisionOptions(RunType.EVAL, args=['--out', str(tmp_path), '--model_version', 'resnet_tiny', '--video_arch', 'tsm',
                                                        '--pooling', 'tsm:tsm_fold_div=4', *given])
    assert parse_pooling_spec(options.pooling)[1] == {'tsm_fold_div': fold}


@pytest.mark.parametrize('given,frames', [([], 4), (['--num_frames', '8'], 8), (['--num_f', '8'], 8)])
def test_the_cli_builds_clips_for_the_recorded_frame_count(tmp_path, given, frames):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    options = DeepTuneVisionOptions(RunType.EVAL, args=['--out', str(tmp_path), '--model_version', 'resnet_tiny', '--video_arch', 'early_fusion',
                                                        '--pooling', 'early_fusion:num_frames=4', *given])
    assert options.num_frames == frames and parse_pooling_spec(options.pooling)[1] == {'num_frames': frames}


# --- where TSM shifts ------------------------------------------------------------------------------------------------------
def test_tsm_shifts_inside_the_residual_branch_only():
    """The block's output is the residual on the shifted features plus the input as it came in."""
    model = _build('tsm').eval()
    block = model.backbone.model.layer1  # no downsample here: the identity path is the input itself
    assert block.downsample is None
    seen = {}
    block.register_forward_pre_hook(lambda _, inputs: seen.update(raw=inputs[0].clone()), prepend=True)
    block.conv1.register_forward_pre_hook(lambda _, inputs: seen.update(branch=inputs[0].clone()))
    block.bn2.register_forward_hook(lambda _, inputs, output: seen.update(residual=output.clone()))
    block.register_forward_hook(lambda _, inputs, output: seen.update(out=output.clone()))
    with torch.no_grad():
        model(_clip('tsm'))
    assert torch.equal(seen['branch'], temporal_shift(seen['raw'], 4, 8))
    assert not torch.equal(seen['branch'], seen['raw'])
    assert torch.allclose(seen['out'], torch.relu(seen['residual'] + seen['raw']), atol=1e-6)


# --- DenseNet with one added layer on video, and the images guard (V-046) ----------------------------------------------------
def test_densenet_video_embedding_is_the_features_and_the_images_embedder_still_refuses():
    from embed.vision.embed import load_vision_model
    from src.vision.densenet import adjustedDenseNet
    model = VideoFramePoolClassifier(adjustedDenseNet, 'dummy', 3, 1, 4, False, pooling='mean').eval()
    with torch.no_grad():
        assert model(torch.randn(2, 2, 3, 64, 64), extract_embed=True).shape == (2, 1024)  # (2, 3), the logits, until V-046
    with pytest.raises(ValueError, match='intermediate embedding layer'):
        load_vision_model(model_architecture='densenet', model_version='dummy', use_case='finetuned', num_classes=3,
                          embed_size=4, added_layers=1)

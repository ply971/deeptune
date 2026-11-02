"""V-JEPA 2, VideoMAE, SigLIP 2 and DINOv3 through transformers. Tiny random configs and local files: nothing downloads."""
import json
import re
from dataclasses import replace

import numpy as np
import pytest
import torch

import handlers.video_backbones as registry
from handlers.video_backbones import (HUB_BACKBONES, WeightsLicenceNotAccepted, accept_weights_licences, licence_text,
                                      require_weights_choice, reset_accepted_licences)
from src.video.feature_backbones import SnippetBackbone, fixed_input, run_weight_licence
from src.video.feature_heads import AttentiveProbe, LinearProbe
from src.video.hub_backbones import (HubClipTransform, HubVideoModel, checked_from_pretrained, convert_videomae_state, encoder_token_grid,
                                     load_hub_model, probe_from_pooling)
from utils import build_video_model, get_model_architecture, get_video_model_cls, is_native_video_architecture

CLIP_MODELS = {'vjepa2_tiny': 4, 'videomae_tiny': 4}
FRAME_MODELS = ('siglip_tiny', 'siglip2_tiny', 'dinov3_tiny')


@pytest.fixture(autouse=True)
def clean_acceptance():
    reset_accepted_licences()
    yield
    reset_accepted_licences()


def _clip(version, frames=4, batch=2):
    size = HUB_BACKBONES[version].input_size
    return torch.randn(batch, frames, 3, size, size)


# --- the model contract ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('version', CLIP_MODELS)
@pytest.mark.parametrize('mode,expected', [('cls', 3), ('reg', 1)])
def test_clip_model_contract_through_build_video_model(version, mode, expected):
    model = build_video_model(version, 3, 2, 8, True, mode)
    assert model(_clip(version)).shape == (2, expected)
    assert model(_clip(version), extract_embed=True).shape == (2, model.embedding_dim)
    assert isinstance(model.head, AttentiveProbe), 'the default recipe is the attentive probe B1 built'


@pytest.mark.parametrize('version', FRAME_MODELS)
def test_frame_backbones_work_with_frame_pooling(version):
    model = build_video_model(version, 3, 2, 8, True, 'cls')
    assert model(_clip(version)).shape == (2, 3)


def test_linear_probe_is_b1s_too():
    model = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls', pooling='linear_probe')
    assert isinstance(model.head, LinearProbe)
    assert model(_clip('vjepa2_tiny'), extract_embed=True).shape == (2, 24)


def test_probe_spec_options_reach_the_head():
    model = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls', pooling='attentive_probe:probe_depth=2,probe_heads=2,probe_width=8')
    assert len(model.head.blocks) == 2 and model.head.query.shape[-1] == 8 and model.embedding_dim == 8


@pytest.mark.parametrize('pooling', ['attention', 'single_frame', 'tsm', 'late_fusion'])
def test_clip_models_refuse_frame_poolings(pooling):
    with pytest.raises(ValueError, match='attentive_probe'):
        probe_from_pooling(pooling, 'vjepa2_tiny')


def test_probe_spec_on_a_2d_backbone_points_to_the_feature_route():
    with pytest.raises(ValueError, match='train_feature_head'):
        build_video_model('resnet_tiny', 3, 2, 8, True, 'cls', pooling='attentive_probe')


def test_names_resolve_as_before_for_old_models():
    assert get_model_architecture('resnet50') == 'resnet'
    assert get_model_architecture('r3d_18') == 'r3d_18'
    assert not is_native_video_architecture('resnet18')
    assert is_native_video_architecture('vjepa2_vitl') and not is_native_video_architecture('siglip2_b16')
    assert get_video_model_cls('siglip2_b16') is get_video_model_cls('siglip')


# --- token grids ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('frames,depth', [(2, 1), (4, 2), (8, 4), (5, 2), (1, 1)])
def test_vjepa_grid_follows_the_frame_count(frames, depth):
    encoder = load_hub_model('vjepa2_tiny').eval()
    grid = encoder_token_grid(encoder, 'vjepa2_tiny', _clip('vjepa2_tiny', frames=frames, batch=1))
    assert grid.shape == (1, depth, 4, 4, 24)


def test_vjepa_tokens_land_in_the_cell_their_pixels_are_in():
    """With no attention layers each token sees only its own tubelet, so a pixel change shows where the token went."""
    from transformers import VJEPA2Config, VJEPA2Model
    encoder = VJEPA2Model(VJEPA2Config(patch_size=8, crop_size=32, frames_per_clip=4, hidden_size=24, num_attention_heads=2, num_hidden_layers=0,
                                       pred_hidden_size=12, pred_num_attention_heads=2, pred_num_hidden_layers=1, pred_num_mask_tokens=1)).encoder.eval()
    clip = torch.zeros(1, 4, 3, 32, 32)
    moved = clip.clone()
    moved[0, 2:4, :, 8:16, 24:32] = 1.0  # second tubelet, row 1, column 3
    with torch.no_grad():
        difference = (encoder_token_grid(encoder, 'vjepa2_tiny', moved) - encoder_token_grid(encoder, 'vjepa2_tiny', clip)).abs().sum(-1)[0]
    assert difference.nonzero().tolist() == [[1, 1, 3]]


def test_vjepa_refuses_a_size_that_would_scramble_its_positions():
    encoder = load_hub_model('vjepa2_tiny')
    with pytest.raises(ValueError, match='32x32'):
        encoder_token_grid(encoder, 'vjepa2_tiny', torch.zeros(1, 4, 3, 48, 48))


def test_videomae_refuses_another_frame_count_by_name():
    encoder = load_hub_model('videomae_tiny')
    with pytest.raises(ValueError, match='--num_frames 4'):
        encoder_token_grid(encoder, 'videomae_tiny', _clip('videomae_tiny', frames=8, batch=1))


def test_dinov3_grid_holds_patches_only():
    encoder = load_hub_model('dinov3_tiny').eval()
    with torch.no_grad():
        tokens = encoder(pixel_values=torch.zeros(1, 3, 32, 32)).last_hidden_state
        grid = encoder_token_grid(encoder, 'dinov3_tiny', torch.zeros(1, 2, 3, 32, 32))
    assert tokens.shape[1] == 1 + 2 + 16, 'class token, two registers, sixteen patches'
    assert grid.shape == (1, 2, 4, 4, 16)
    assert torch.equal(grid[0, 0].reshape(16, 16), tokens[0, 3:])


@pytest.mark.parametrize('version,frames,shape', [('vjepa2_tiny', 4, (2, 4, 4, 24)), ('videomae_tiny', 4, (2, 4, 4, 16)), ('siglip2_tiny', 4, (4, 4, 4, 32)),
                                                  ('siglip_tiny', 2, (2, 4, 4, 32)), ('dinov3_tiny', 2, (2, 4, 4, 16))])
def test_snippet_backbone_covers_every_family(version, frames, shape):
    backbone = SnippetBackbone(version)
    x = _clip(version, frames=frames, batch=1)
    assert backbone.token_grid(x).shape == (1, *shape)
    assert backbone(x).shape == (1, shape[-1])
    assert not any(p.requires_grad for p in backbone.parameters())
    assert backbone.weights_name is None and backbone.weights_licence.startswith('none')
    assert backbone.default_input_size == HUB_BACKBONES[version].input_size


def test_snippet_backbone_takes_motion_like_the_others():
    backbone = SnippetBackbone('vjepa2_tiny', in_channels=2)
    assert backbone.token_grid(torch.zeros(1, 4, 2, 32, 32)).shape == (1, 2, 4, 4, 24)


def test_snippet_backbone_refuses_to_pretend_real_weights_are_random():
    with pytest.raises(ValueError, match='_tiny'):
        SnippetBackbone('vjepa2_vitl', pretrained=False)


def test_tiny_extraction_is_reproducible():
    x = _clip('vjepa2_tiny', batch=1)
    assert torch.equal(SnippetBackbone('vjepa2_tiny').token_grid(x), SnippetBackbone('vjepa2_tiny').token_grid(x))


def test_fixed_input_is_the_registry():
    assert fixed_input('vjepa2_vitl', True) == (256, None)
    assert fixed_input('videomae_b16', True) == (224, 16)


# --- frozen and trainable -------------------------------------------------------------------------------------------
def test_frozen_encoder_keeps_no_graph_and_stays_in_eval():
    model = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls').train()
    assert not model.model.training and model.head.training
    assert not model.token_grid(_clip('vjepa2_tiny')).requires_grad
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    head_before = model.head.head.weight.clone()
    model(_clip('vjepa2_tiny')).sum().backward()
    optimizer.step()
    assert not torch.equal(head_before, model.head.head.weight)
    assert all(torch.equal(before[k], v) for k, v in model.model.state_dict().items())


def test_unfrozen_encoder_trains():
    model = build_video_model('vjepa2_tiny', 3, 2, 8, False, 'cls').train()
    assert model.model.training
    model(_clip('vjepa2_tiny')).sum().backward()
    assert all(p.grad is not None for p in model.model.parameters())


def test_budget_can_unfreeze_part_of_a_frozen_encoder():
    from trainers.param_budget import apply_top_down_budget, parameter_group
    model = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls')
    head_size = sum(p.numel() for n, p in model.named_parameters() if parameter_group(n) == 'head')
    apply_top_down_budget(model, head_size + 100)
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and parameter_group(n) == 'backbone')
    assert 0 < trainable <= 100
    model.train()
    assert model.model.training, 'part of the encoder trains, so it must not be stuck in eval or no_grad'
    assert model.token_grid(_clip('vjepa2_tiny')).requires_grad


@pytest.mark.parametrize('version', CLIP_MODELS)
def test_every_lora_adapter_gets_a_gradient_with_checkpointing_on(version):
    model = build_video_model(version, 3, 2, 8, True, 'cls', use_peft=True, lora_rank=2).train()
    adapters = {n: p for n, p in model.named_parameters() if 'lora_' in n}
    assert adapters and all(p.requires_grad for p in adapters.values())
    assert not [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n and not n.startswith('head.')]
    model(_clip(version)).sum().backward()
    dead = [n for n, p in adapters.items() if p.grad is None or (('lora_B' in n) and p.grad.abs().sum() == 0)]
    assert not dead, dead
    assert sum(1 for n in adapters if 'lora_A' in n) == 2, 'query and value of the one tiny layer'
    assert adapters[next(n for n in adapters if 'lora_A' in n)].shape[0] == 2, 'the rank'


@pytest.mark.parametrize('version', ['siglip2_tiny', 'dinov3_tiny'])
def test_frame_backbone_lora_adapters_get_a_gradient(version):
    model = build_video_model(version, 3, 2, 8, True, 'cls', use_peft=True).train()
    model(_clip(version)).sum().backward()
    adapters = {n: p for n, p in model.named_parameters() if 'lora_' in n}
    assert adapters and not [n for n, p in adapters.items() if p.grad is None]


def test_peft_model_round_trips_through_its_state_dict():
    model = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls', use_peft=True, lora_rank=2).eval()
    other = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls', use_peft=True, lora_rank=2).eval()
    other.load_state_dict(model.state_dict())
    x = _clip('vjepa2_tiny')
    with torch.no_grad():
        assert torch.equal(model(x), other(x))


# --- preprocessing --------------------------------------------------------------------------------------------------
def _noise_clip(height, width):
    rng = np.random.default_rng(0)
    coarse = rng.integers(0, 256, (2, 3, height // 8, width // 8), dtype=np.uint8)  # smooth enough that rounding stays small
    return torch.nn.functional.interpolate(torch.from_numpy(coarse).float(), size=(height, width), mode='bilinear').round().to(torch.uint8)


@pytest.mark.parametrize('height,width', [(320, 480), (480, 320), (300, 300), (360, 640), (1080, 1920), (1920, 1080)])
def test_vjepa_transform_matches_the_transformers_processor(height, width):
    from transformers import VJEPA2VideoProcessor
    clip = _noise_clip(height, width)
    theirs = VJEPA2VideoProcessor()(videos=[clip], return_tensors='pt')['pixel_values_videos'][0]
    ours = HubClipTransform('vjepa2_vitl')(clip)
    assert ours.shape == theirs.shape == (2, 3, 256, 256)
    assert (ours - theirs).abs().max() < 0.03, 'within the rounding of their uint8 resize'


@pytest.mark.parametrize('height,width', [(320, 480), (480, 320), (300, 533)])
def test_videomae_transform_matches_the_transformers_processor(height, width):
    from transformers import VideoMAEVideoProcessor
    clip = _noise_clip(height, width)
    theirs = VideoMAEVideoProcessor(image_mean=list(HUB_BACKBONES['videomae_b16'].mean), image_std=list(HUB_BACKBONES['videomae_b16'].std))(
        videos=[clip], return_tensors='pt')['pixel_values'][0]
    ours = HubClipTransform('videomae_b16')(clip)
    assert ours.shape == theirs.shape == (2, 3, 224, 224)
    assert (ours - theirs).abs().max() < 0.03


def test_the_crop_rounds_down_like_the_processor_when_it_falls_between_pixels():
    """16:9 at V-JEPA 2's sizes leaves 131.5 pixels on each side. A ramp's value is its position, so it shows the crop."""
    from extract_features import crop_window
    from transformers import VJEPA2VideoProcessor
    ramp = torch.linspace(0, 255, 640).round().to(torch.uint8).expand(1, 3, 360, 640)
    theirs = VJEPA2VideoProcessor()(videos=[ramp], return_tensors='pt')['pixel_values_videos'][0]
    ours = HubClipTransform('vjepa2_vitl')(ramp)
    wrong_side = HubClipTransform('vjepa2_vitl')(torch.cat([ramp[..., :1], ramp[..., :-1]], dim=-1))
    assert (ours - theirs).abs().mean() < (wrong_side - theirs).abs().mean()
    x0, x1, _, _ = crop_window('vjepa2_vitl', None, 640, 360)
    assert (x0, x1) == (pytest.approx(131 / 519), pytest.approx(387 / 519))


def test_dinov3_transform_matches_the_transformers_processor():
    from transformers import DINOv3ViTImageProcessor
    clip = _noise_clip(320, 480)
    theirs = DINOv3ViTImageProcessor()(images=[frame for frame in clip], return_tensors='pt')['pixel_values']
    ours = HubClipTransform('dinov3_vitb16')(clip)
    assert ours.shape == theirs.shape == (2, 3, 224, 224), 'the whole frame squashed, no centre crop'
    assert (ours - theirs).abs().max() < 0.03


def test_siglip2_resizes_bilinear_and_siglip_bicubic_as_their_config_files_say():
    from src.video.siglip_backbone import SiglipClipTransform
    clip = _noise_clip(320, 320)
    siglip2 = SiglipClipTransform('siglip2_b16')(clip)
    expected = torch.nn.functional.interpolate(clip.float() / 255.0, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
    assert torch.allclose(siglip2, (expected.clamp(0, 1) - 0.5) / 0.5)
    assert SiglipClipTransform('siglip').mode == 'bicubic'


@pytest.mark.parametrize('version', ['vjepa2_vitl', 'siglip2_b16', 'dinov3_vitb16', 'videomae_b16'])
def test_datasets_get_the_models_own_transform(version):
    from datasets.video_datasets import native_video_transform
    assert native_video_transform(version) is not None
    frames = native_video_transform(version)(_noise_clip(240, 320))
    assert frames.shape[-2:] == (HUB_BACKBONES[version].input_size,) * 2


def test_crop_window_is_what_the_transform_really_keeps():
    from extract_features import crop_window
    x0, x1, y0, y1 = crop_window('vjepa2_vitl', None, 480, 320)
    assert (y0, y1) == (pytest.approx(18 / 292), pytest.approx(274 / 292))
    info = HUB_BACKBONES['vjepa2_vitl']
    # A ramp's value is its position, so the first and last kept pixels say where the crop really fell.
    ramp = torch.linspace(0, 255, 480).round().to(torch.uint8).expand(1, 3, 320, 480)
    kept = HubClipTransform('vjepa2_vitl')(ramp)[0, 0] * info.std[0] + info.mean[0]
    assert kept[:, 0].mean().item() == pytest.approx(x0, abs=0.01)
    assert kept[:, -1].mean().item() == pytest.approx(x1, abs=0.01)
    assert crop_window('siglip2_b16', None, 480, 320) == (0.0, 1.0, 0.0, 1.0)
    assert crop_window('vjepa2_vitl', 256, 480, 320) == (0.0, 1.0, 0.0, 1.0)


# --- licences -------------------------------------------------------------------------------------------------------
def test_non_commercial_and_gated_weights_are_never_defaults():
    assert HUB_BACKBONES['vjepa2_vitl'].licence == 'MIT' and not HUB_BACKBONES['vjepa2_vitl'].opt_in
    assert HUB_BACKBONES['siglip2_b16'].licence == 'Apache-2.0' and not HUB_BACKBONES['siglip2_b16'].opt_in
    for version in ('videomae_b16', 'videomae_v2_b16'):
        assert HUB_BACKBONES[version].licence == 'CC-BY-NC-4.0' and HUB_BACKBONES[version].opt_in
    assert HUB_BACKBONES['dinov3_vitb16'].opt_in and HUB_BACKBONES['dinov3_vitb16'].gated


def test_no_vjepa21_is_registered():
    assert not [v for v in HUB_BACKBONES if '21' in v or '2.1' in v]


@pytest.mark.parametrize('version', ['videomae_b16', 'videomae_v2_b16', 'dinov3_vitb16'])
def test_opt_in_weights_are_refused_where_they_load(version, capsys):
    with pytest.raises(WeightsLicenceNotAccepted, match='--accept_weights_licence'):
        load_hub_model(version)
    assert HUB_BACKBONES[version].licence in capsys.readouterr().out, 'the licence is printed when chosen'


def test_accepting_another_licence_does_not_count():
    accept_weights_licences(['MIT'])
    with pytest.raises(WeightsLicenceNotAccepted):
        require_weights_choice('videomae_b16')
    accept_weights_licences(['cc-by-nc-4.0'])
    require_weights_choice('videomae_b16')


def test_licence_free_defaults_need_no_acceptance_but_are_printed(capsys):
    require_weights_choice('vjepa2_vitl')
    assert 'MIT' in capsys.readouterr().out


def test_run_records_name_the_terms_of_each_family():
    assert run_weight_licence('vjepa2_vitl') == licence_text('vjepa2_vitl')
    assert 'non-commercial' in run_weight_licence('videomae_b16')
    assert run_weight_licence('vjepa2_tiny').startswith('none')
    assert 'KINETICS400' in run_weight_licence('r3d_18')
    assert 'IMAGENET1K_V1' in run_weight_licence('resnet18'), 'the weights src/vision/resnet.py pins'
    assert 'non-commercial' in run_weight_licence('vit_h_14'), 'a DEFAULT with its own terms'
    assert run_weight_licence('resnet_tiny').startswith('none')


def test_torchvision_frame_weights_match_what_src_vision_builds():
    """run_weight_licences must describe the file a model really loads, read from the source of each family."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'src' / 'vision'
    assert 'ResNet18_Weights.IMAGENET1K_V1' in (root / 'resnet.py').read_text(encoding='utf-8')
    assert 'Swin_T_Weights.IMAGENET1K_V1' in (root / 'swin.py').read_text(encoding='utf-8')
    for family, call in (('vit', 'vit_b_16(weights="DEFAULT")'), ('efficientnet', 'efficientnet_b0(weights="DEFAULT")'), ('vgg', 'vgg11(weights="DEFAULT")')):
        assert call in (root / f'{family}.py').read_text(encoding='utf-8')


# --- loading ----------------------------------------------------------------------------------------------------------
def _with_hub_id(monkeypatch, version, path):
    monkeypatch.setitem(HUB_BACKBONES, version, replace(HUB_BACKBONES[version], hub_id=str(path)))


def test_vjepa_loads_from_a_saved_checkpoint_without_its_predictor(tmp_path, monkeypatch):
    from transformers import VJEPA2Config, VJEPA2Model
    source = load_hub_model('vjepa2_tiny')
    saved = VJEPA2Model(VJEPA2Config(patch_size=8, crop_size=32, frames_per_clip=4, hidden_size=24, num_attention_heads=2, num_hidden_layers=1,
                                     pred_hidden_size=12, pred_num_attention_heads=2, pred_num_hidden_layers=1, pred_num_mask_tokens=1))
    saved.save_pretrained(tmp_path)
    _with_hub_id(monkeypatch, 'vjepa2_vitl', tmp_path)
    monkeypatch.setitem(HUB_BACKBONES, 'vjepa2_vitl', replace(HUB_BACKBONES['vjepa2_vitl'], input_size=32, resize_short=36))
    encoder = load_hub_model('vjepa2_vitl')
    assert not any(k.startswith('predictor') for k in encoder.state_dict())
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, saved.encoder.state_dict()[key])
    assert next(encoder.parameters()).dtype == torch.float32
    assert source is not encoder


def test_partial_checkpoint_is_refused_not_left_random(tmp_path):
    from transformers import VideoMAEConfig, VideoMAEModel, VJEPA2Model
    VideoMAEModel(VideoMAEConfig(image_size=32, patch_size=8, num_frames=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=32)).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match='weights are missing'):
        checked_from_pretrained(VJEPA2Model, str(tmp_path))


def _tiny_videomae():
    """With non-zero query and value biases, which random init leaves at zero: a test on zeros cannot see a lost or swapped
    bias. The key layer's stays zero, since no old layout has one."""
    model = load_hub_model('videomae_tiny').eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('attention.attention.query.bias'):
                parameter.copy_(torch.linspace(0.5, 1.5, parameter.numel()))
            elif name.endswith('attention.attention.value.bias'):
                parameter.copy_(torch.linspace(-2.0, -1.0, parameter.numel()))
    return model


def _old_transformers_layout(state):
    """How older files store it: a videomae. prefix, q_bias and v_bias, no bias on the layers, a decoder beside it."""
    out = {'decoder.decoder_pred.weight': torch.zeros(1), 'encoder_to_decoder.weight': torch.zeros(1), 'mask_token': torch.zeros(1)}
    for key, value in state.items():
        if key.endswith(('query.bias', 'value.bias')):
            out['videomae.' + key.replace('query.bias', 'q_bias').replace('value.bias', 'v_bias')] = value
        elif not key.endswith('key.bias'):
            out['videomae.' + key] = value
    return out


def _upstream_v2_layout(state):
    """The names in OpenGVLab's modeling_videomaev2.py, read on its Hugging Face repository."""
    out = {'model.fc_norm.weight': torch.zeros(1), 'model.fc_norm.bias': torch.zeros(1)}
    renames = {'attention.output.dense': 'attn.proj', 'intermediate.dense': 'mlp.fc1', 'output.dense': 'mlp.fc2',
               'layernorm_before': 'norm1', 'layernorm_after': 'norm2'}
    for key, value in state.items():
        if key.startswith('embeddings.patch_embeddings.projection.'):
            out['model.patch_embed.proj.' + key.rsplit('.', 1)[1]] = value
            continue
        match = re.match(r'encoder\.layer\.(\d+)\.(.+)', key)
        block, rest = match.groups()
        if rest.startswith('attention.attention.'):
            layer, kind = rest[len('attention.attention.'):].split('.')
            if kind == 'bias' and layer != 'key':
                out[f'model.blocks.{block}.attn.{layer[0]}_bias'] = value
            continue
        for new, old in renames.items():
            if rest.startswith(new):
                rest = old + rest[len(new):]
                break
        out[f'model.blocks.{block}.{rest}'] = value
    for block in {k.split('.')[2] for k in state if k.startswith('encoder.layer.')}:
        out[f'model.blocks.{block}.attn.qkv.weight'] = torch.cat(
            [state[f'encoder.layer.{block}.attention.attention.{layer}.weight'] for layer in ('query', 'key', 'value')], dim=0)
    return out


@pytest.mark.parametrize('layout', [_old_transformers_layout, _upstream_v2_layout, dict])
def test_every_videomae_layout_gives_the_same_network(layout):
    source = _tiny_videomae()
    state = source.state_dict()
    target = load_hub_model('videomae_tiny').eval()
    target.load_state_dict(convert_videomae_state(layout(state)), strict=True)
    x = _clip('videomae_tiny', batch=1)
    with torch.no_grad():
        assert torch.equal(source(pixel_values=x).last_hidden_state, target(pixel_values=x).last_hidden_state)


def test_old_bias_names_are_not_lost():
    state = _tiny_videomae().state_dict()
    converted = convert_videomae_state(_old_transformers_layout(state))
    key = 'encoder.layer.0.attention.attention.query.bias'
    assert torch.equal(converted[key], state[key])


def test_unforeseen_videomae_layout_stops_at_the_strict_load():
    state = convert_videomae_state(_tiny_videomae().state_dict())
    state['encoder.layer.0.something_new.weight'] = torch.zeros(1)
    with pytest.raises(RuntimeError, match='something_new'):
        _tiny_videomae().load_state_dict(state, strict=True)


def test_videomae_real_path_reads_a_local_file(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from transformers import VideoMAEConfig
    import src.video.hub_backbones as hub
    source = _tiny_videomae()
    state = {k: v.clone() for k, v in source.state_dict().items()}
    save_file(_old_transformers_layout(state), str(tmp_path / 'model.safetensors'))
    VideoMAEConfig(image_size=32, patch_size=8, num_frames=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32).save_pretrained(tmp_path)
    _with_hub_id(monkeypatch, 'videomae_b16', tmp_path)
    accept_weights_licences(['CC-BY-NC-4.0'])
    loaded = hub.load_hub_model('videomae_b16')
    assert torch.equal(loaded.state_dict()['encoder.layer.0.attention.attention.value.bias'],
                       state['encoder.layer.0.attention.attention.value.bias'])
    assert state['encoder.layer.0.attention.attention.value.bias'].abs().min() > 0


def _save_videomae_folder(folder, state, filename='model.safetensors'):
    from safetensors.torch import save_file
    from transformers import VideoMAEConfig
    folder.mkdir(parents=True, exist_ok=True)
    if filename.endswith('.bin'):
        torch.save(state, folder / filename)
    else:
        save_file(state, str(folder / filename))
    VideoMAEConfig(image_size=32, patch_size=8, num_frames=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32).save_pretrained(folder)


def _weights_of(source):
    return {k: v.clone() for k, v in source.state_dict().items()}


def test_a_partial_videomae_file_is_refused_through_the_real_loader(tmp_path, monkeypatch):
    state = _old_transformers_layout(_weights_of(_tiny_videomae()))
    del state['videomae.encoder.layer.0.intermediate.dense.weight']
    _save_videomae_folder(tmp_path, state)
    _with_hub_id(monkeypatch, 'videomae_b16', tmp_path)
    accept_weights_licences(['CC-BY-NC-4.0'])
    with pytest.raises(RuntimeError, match='intermediate.dense.weight'):
        load_hub_model('videomae_b16')


def test_a_file_with_an_unforeseen_key_is_refused_through_the_real_loader(tmp_path, monkeypatch):
    state = _old_transformers_layout(_weights_of(_tiny_videomae()))
    state['videomae.encoder.layer.0.gamma_1'] = torch.zeros(16)
    _save_videomae_folder(tmp_path, state)
    _with_hub_id(monkeypatch, 'videomae_b16', tmp_path)
    accept_weights_licences(['CC-BY-NC-4.0'])
    with pytest.raises(RuntimeError, match='gamma_1'):
        load_hub_model('videomae_b16')


@pytest.mark.parametrize('key', ['model.pos_embed', 'videomae.embeddings.position_embeddings'])
def test_learned_positions_are_refused_rather_than_dropped(key):
    state = _weights_of(_tiny_videomae())
    state[key] = torch.ones(1, 32, 16)
    with pytest.raises(ValueError, match='position embeddings'):
        convert_videomae_state(state)


def test_the_bin_branch_reads_the_weights(tmp_path, monkeypatch):
    source = _tiny_videomae()
    state = _weights_of(source)
    _save_videomae_folder(tmp_path, _old_transformers_layout(state), 'pytorch_model.bin')
    _with_hub_id(monkeypatch, 'videomae_b16', tmp_path)
    accept_weights_licences(['CC-BY-NC-4.0'])
    loaded = load_hub_model('videomae_b16').eval()
    x = _clip('videomae_tiny', batch=1)
    with torch.no_grad():
        assert torch.equal(loaded(pixel_values=x).last_hidden_state, source(pixel_values=x).last_hidden_state)


class _NotWeights:
    """Anything pickle would rebuild. A weights file must not be able to do this."""


def test_a_bin_file_cannot_run_pickled_objects(tmp_path):
    from src.video.hub_backbones import _read_raw_weights
    torch.save({'x': _NotWeights()}, tmp_path / 'pytorch_model.bin')
    with pytest.raises(Exception, match='(?i)weights.?only|unsupported'):
        _read_raw_weights(str(tmp_path))


def test_a_folder_without_weights_is_a_clear_error(tmp_path):
    from src.video.hub_backbones import _read_raw_weights
    with pytest.raises(FileNotFoundError, match='model.safetensors'):
        _read_raw_weights(str(tmp_path))


def test_a_hub_repository_with_only_a_bin_file_falls_back_to_it(tmp_path, monkeypatch):
    """The download call is replaced, so nothing leaves the machine."""
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError
    from src.video.hub_backbones import _read_raw_weights
    torch.save({'x': torch.ones(2)}, tmp_path / 'pytorch_model.bin')
    asked = []

    def fake_download(repo, filename):
        asked.append(filename)
        if filename == 'model.safetensors':
            raise EntryNotFoundError('no such file')
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
    assert torch.equal(_read_raw_weights('some/repo')['x'], torch.ones(2))
    assert asked == ['model.safetensors', 'pytorch_model.bin']


# --- the CLI --------------------------------------------------------------------------------------------------------
def _options(*flags):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    return DeepTuneVisionOptions(RunType.ONECALL, args=['--modality', 'video', '--df', 'x.parquet', '--out', 'o', *flags])


def test_cli_default_recipe_is_frozen_with_an_attentive_probe_and_it_is_recorded():
    options = _options('--model_version', 'vjepa2_tiny')
    assert options.freeze_backbone is True
    assert options.pooling == 'attentive_probe:probe_depth=1,probe_heads=4,probe_width=256'
    assert options.to_dict()['pooling'] == options.pooling


def test_cli_can_train_the_encoder_when_asked():
    assert _options('--model_version', 'vjepa2_tiny', '--unfreeze_backbone').freeze_backbone is False
    assert _options('--model_version', 'vjepa2_tiny', '--use-peft').freeze_backbone is False


def test_cli_probe_flags_and_linear_probe():
    assert _options('--model_version', 'vjepa2_tiny', '--video_arch', 'attentive_probe', '--probe_depth', '2').pooling.startswith('attentive_probe:probe_depth=2')
    assert _options('--model_version', 'vjepa2_tiny', '--probe_width', '8', '--probe_heads', '2').pooling == 'attentive_probe:probe_depth=1,probe_heads=2,probe_width=8'
    assert _options('--model_version', 'vjepa2_tiny', '--video_arch', 'linear_probe').pooling == 'linear_probe'


def test_cli_leaves_other_models_alone():
    options = _options('--model_version', 'resnet18')
    assert options.pooling == 'mean' and options.freeze_backbone is False


def test_cli_says_when_it_replaces_the_default_pooling(capsys):
    options = _options('--model_version', 'vjepa2_tiny')
    assert options.pooling.startswith('attentive_probe')
    assert 'attentive_probe' in capsys.readouterr().out


@pytest.mark.parametrize('flags', [['--pooling', 'mean'], ['--pooling', 'attention'], ['--video_arch', 'frame_pool'],
                                   ['--video_arch', 'slow_fusion'], ['--video_arch', 'tsm']])
def test_cli_refuses_an_explicit_frame_pooling_on_a_clip_model(flags, capsys):
    """Otherwise an ablation of mean against the attentive probe compares two identical models."""
    with pytest.raises(SystemExit):
        _options('--model_version', 'vjepa2_tiny', *flags)
    assert '--video_arch linear_probe' in capsys.readouterr().err


@pytest.mark.parametrize('version,frames,expected', [('vjepa2_tiny', 8, 'attentive_probe'), ('videomae_tiny', 4, 'attentive_probe'), ('resnet18', 8, 'mean')])
def test_a_gui_built_run_with_the_default_pooling_parses(version, frames, expected, tmp_path):
    """Re-check of V-052: the GUI always sent --pooling mean, which the clip-model refusal then stopped."""
    from desktop.config import RunConfig, build_argv
    from cli import DeepTuneVisionOptions
    from utils import RunType
    argv = build_argv(RunConfig(modality='video', df=tmp_path / 'x.parquet', out=tmp_path / 'o', model_version=version, num_frames=frames))
    assert DeepTuneVisionOptions(RunType.ONECALL, args=argv).pooling.startswith(expected)


def test_a_gui_built_run_still_passes_a_pooling_the_user_chose(tmp_path):
    from desktop.config import RunConfig, build_argv
    argv = build_argv(RunConfig(modality='video', df=tmp_path / 'x.parquet', out=tmp_path / 'o', model_version='vjepa2_tiny', pooling='attention'))
    assert argv[argv.index('--pooling') + 1] == 'attention'  # and cli.py refuses it, as for any explicit frame pooling


def test_infer_refuses_an_explicit_mean_on_a_clip_model(tmp_path, capsys):
    """R1.md claims infer.py refuses this; nothing tested it (the re-check's surviving mutant). Refused before any weights load."""
    import infer
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'cli_arguments.json').write_text(json.dumps({'model_version': 'vjepa2_tiny', 'num_classes': 2, 'modality': 'video',
                                                        'pooling': 'attentive_probe:probe_depth=1,probe_heads=4,probe_width=256'}), encoding='utf-8')
    with pytest.raises(SystemExit):
        infer.main(['--modality', 'video', '--checkpoint', str(run / 'model_weights.pth'), '--data', str(tmp_path), '--out', str(tmp_path / 'o'),
                    '--model_version', 'vjepa2_tiny', '--pooling', 'mean'])
    assert '--pooling mean does not apply' in capsys.readouterr().err


def test_cli_keeps_an_explicit_mean_on_other_models():
    assert _options('--model_version', 'resnet18', '--pooling', 'mean').pooling == 'mean'
    assert _options('--model_version', 'vjepa2_tiny', '--pooling', 'linear_probe').pooling == 'linear_probe'


@pytest.mark.parametrize('flags', [['--model_version', 'resnet18'], ['--model_version', 'r3d_18'], ['--model_version', 'siglip_tiny'],
                                   ['--model_version', 'vjepa2_tiny', '--freeze-backbone'], ['--model_version', 'vjepa2_tiny', '--use-peft'],
                                   ['--model_version', 'siglip2_tiny', '--freeze-backbone']])
def test_cli_refuses_unfreeze_where_it_means_nothing_or_contradicts(flags, capsys):
    with pytest.raises(SystemExit):
        _options(*flags, '--unfreeze_backbone')
    assert '--unfreeze_backbone' in capsys.readouterr().err


@pytest.mark.parametrize('version,frozen', [('siglip2_tiny', True), ('dinov3_tiny', True), ('vjepa2_tiny', True), ('videomae_tiny', True),
                                            ('siglip_tiny', False), ('resnet18', False)])
def test_cli_frozen_defaults_follow_p101_and_leave_older_models_alone(version, frozen):
    assert _options('--model_version', version, '--num_frames', '4').freeze_backbone is frozen


def test_cli_can_unfreeze_the_appearance_stream():
    assert _options('--model_version', 'siglip2_tiny', '--unfreeze_backbone').freeze_backbone is False


def test_cli_refuses_opt_in_weights_without_the_licence_and_says_how(capsys):
    with pytest.raises(SystemExit):
        _options('--model_version', 'videomae_b16', '--num_frames', '16')
    error = capsys.readouterr().err
    assert '--accept_weights_licence CC-BY-NC-4.0' in error


def test_cli_records_the_acceptance():
    options = _options('--model_version', 'videomae_b16', '--num_frames', '16', '--accept_weights_licence', 'CC-BY-NC-4.0')
    assert options.to_dict()['accept_weights_licence'] == ['CC-BY-NC-4.0']


def test_cli_refuses_the_wrong_frame_count_up_front(capsys):
    with pytest.raises(SystemExit):
        _options('--model_version', 'videomae_tiny', '--num_frames', '8')
    assert '--num_frames 4' in capsys.readouterr().err


def test_desktop_config_accepts_the_command_line_models():
    from desktop.config import MODEL_VERSIONS_BY_MODALITY, VIDEO_CLI_ONLY_MODEL_VERSIONS
    assert set(HUB_BACKBONES) <= set(VIDEO_CLI_ONLY_MODEL_VERSIONS)
    assert not set(HUB_BACKBONES) & set(MODEL_VERSIONS_BY_MODALITY['video']), 'the GUI list is Phase 7'


def test_registry_is_torch_free():
    import subprocess
    import sys
    code = "import sys, handlers.video_backbones, handlers.video_arch; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, '-c', code]).returncode == 0
    assert registry.HUB_VIDEO_VERSIONS


# --- exact weights ----------------------------------------------------------------------------------------------------
def _fake_cache(root, repo, sha, filename='config.json'):
    folder = root / ('models--' + repo.replace('/', '--'))
    (folder / 'refs').mkdir(parents=True)
    (folder / 'refs' / 'main').write_text(sha)
    (folder / 'snapshots' / sha).mkdir(parents=True)
    (folder / 'snapshots' / sha / filename).write_text('{}')


@pytest.mark.parametrize('filename', ['config.json', 'model.safetensors'])
def test_the_snapshot_commit_is_read_from_the_local_cache(tmp_path, filename):
    from handlers.video_backbones import hub_snapshot_sha
    _fake_cache(tmp_path, 'facebook/some-model', 'a' * 40, filename)
    assert hub_snapshot_sha('facebook/some-model', cache_dir=tmp_path) == 'a' * 40
    assert hub_snapshot_sha('facebook/another-model', cache_dir=tmp_path) is None


def test_a_revision_that_cannot_be_read_says_so(tmp_path, monkeypatch):
    from handlers.video_backbones import weights_revision
    monkeypatch.setitem(HUB_BACKBONES, 'vjepa2_vitl', replace(HUB_BACKBONES['vjepa2_vitl'], hub_id=str(tmp_path)))
    assert weights_revision('vjepa2_vitl').startswith('unknown:')
    assert weights_revision('vjepa2_tiny').startswith('none')
    assert weights_revision('r3d_18').startswith('not applicable')


# --- budgets ----------------------------------------------------------------------------------------------------------
def test_budget_errors_name_the_flags_that_shape_this_head():
    from trainers.param_budget import apply_top_down_budget, build_with_budget
    model = build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls')
    with pytest.raises(ValueError, match='--probe_width') as error:
        apply_top_down_budget(model, 10)
    assert '--embed_size' not in str(error.value)
    with pytest.raises(ValueError, match='--probe_width') as error:
        build_with_budget(lambda rank: build_video_model('vjepa2_tiny', 3, 2, 8, True, 'cls', use_peft=True, lora_rank=rank), 10, True, 8, True, rank_is_head_width=False)
    assert '--embed_size' not in str(error.value)


def test_budget_errors_of_other_models_are_unchanged():
    from trainers.param_budget import apply_top_down_budget
    with pytest.raises(ValueError, match='--embed_size'):
        apply_top_down_budget(build_video_model('resnet_tiny', 3, 2, 8, True, 'cls'), 10)

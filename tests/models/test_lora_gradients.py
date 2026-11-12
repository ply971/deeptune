"""Every LoRA adapter must be able to learn, and coverage must count only those
(V-040). CPU, weights=None. Gradients are read in eval mode: stochastic depth in
Swin and MViT drops whole blocks at random in train mode, which would make a
live adapter look dead in one pass, and eval mode leaves the gradient path intact."""
import copy

import pytest
import torch
import torch.nn as nn
import torchvision
import torchvision.models.video as video_models

from src.video import video3d
from src.video.frame_pool import VideoFramePoolClassifier
from src.video.video3d_peft import adjustedPeftVideo3D, lora_target_modules, upgrade_legacy_lora_state
from src.vision.convnext_peft import adjustedPeftConvNext
from src.vision.densenet_peft import adjustedPeftDenseNet
from src.vision.efficientnet_peft import adjustedPeftEfficientNet
from src.vision.resnet_peft import adjustedPeftResNet
from src.vision.swin_peft import adjustedPeftSwin
from src.vision.vgg_peft import adjustedPeftVGGNet
from src.vision.vit_peft import adjustedViTPeft
from trainers.param_budget import ELIGIBLE_LORA_TYPES, format_lora_coverage, lora_coverage, summarize_parameters


def _lora_b(model):
    return {name: param for name, param in model.named_parameters() if '.lora_B.' in name}


def _dead_adapters(model, x):
    model.eval()
    model(x).sum().backward()
    return [name for name, param in _lora_b(model).items() if param.grad is None or param.grad.abs().sum() == 0]


def _eligible_in(backbone):
    return sum(isinstance(module, ELIGIBLE_LORA_TYPES) for module in backbone.modules())


def _swin_attention_layers(backbone):
    return [name for name, module in backbone.named_modules()
            if isinstance(module, nn.Linear) and name.endswith(('.attn.qkv', '.attn.proj'))]


@pytest.fixture
def no_downloads(monkeypatch):
    for version in ('mc3_18', 'r2plus1d_18', 'swin3d_t', 'mvit_v2_s'):
        monkeypatch.setitem(video3d._KINETICS400_WEIGHTS, version, None)


@pytest.mark.parametrize('version, clip', [
    ('mc3_18', (1, 4, 3, 32, 32)),
    ('r2plus1d_18', (1, 4, 3, 32, 32)),
    ('swin3d_t', (1, 4, 3, 32, 32)),
    ('mvit_v2_s', (1, 16, 3, 224, 224)),  # its position tables fix this input size
])
def test_every_native_video_adapter_has_a_gradient_and_coverage_is_honest(no_downloads, version, clip):
    torch.manual_seed(0)
    model = adjustedPeftVideo3D(3, version, 1, 8, lora_rank=2)
    assert _dead_adapters(model, torch.rand(*clip)) == []

    pristine = getattr(video_models, version)(weights=None)
    pristine.fc = pristine.head = nn.Identity()
    coverage = lora_coverage(model)
    assert coverage['lora_modules'] == len(_lora_b(model)) == len(lora_target_modules(pristine))
    assert coverage['eligible_modules'] == _eligible_in(pristine)


def test_swin3d_coverage_counts_attention_as_not_adapted(no_downloads, capsys):
    pristine = video_models.swin3d_t(weights=None)
    pristine.head = nn.Identity()
    attention = _swin_attention_layers(pristine)
    assert len(attention) == 24

    model = adjustedPeftVideo3D(3, 'swin3d_t', 1, 8, lora_rank=2)
    assert "does not adapt Swin's attention" in capsys.readouterr().out
    coverage = lora_coverage(model)
    assert coverage['eligible_modules'] == 52 and coverage['lora_modules'] == 52 - len(attention)
    assert not any('.attn.qkv.lora_' in name or '.attn.proj.lora_' in name for name, _ in model.named_parameters())
    assert format_lora_coverage(summarize_parameters(model)).startswith('LoRA adapters on 28 of 52')


class _OldSwin3D(adjustedPeftVideo3D):
    """Targets as saved by a714204 and by T1 before V-040: every Linear, attention included."""

    def applyPEFT(self, model):
        from peft import LoraConfig, get_peft_model
        targets = lora_target_modules(model) + _swin_attention_layers(model)
        return get_peft_model(model, LoraConfig(r=self.lora_rank, lora_alpha=16, lora_dropout=0.1, bias='none',
                                                target_modules=targets))


def _nudge_adapters(model):
    with torch.no_grad():
        for param in _lora_b(model).values():
            param.add_(0.01)  # non-zero, so a mistake in the conversion would show


def test_old_swin3d_attention_adapters_were_dead_and_loading_drops_them(no_downloads, capsys):
    torch.manual_seed(0)
    old = _OldSwin3D(3, 'swin3d_t', 1, 8, lora_rank=2)
    _nudge_adapters(old)
    x = torch.rand(1, 4, 3, 32, 32)
    dead = _dead_adapters(old, x)
    assert len(dead) == 24 and all('.attn.' in name for name in dead)
    with torch.no_grad():
        expected = old(x)

    new = adjustedPeftVideo3D(3, 'swin3d_t', 1, 8, lora_rank=2).eval()
    result = new.load_state_dict(old.state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert 'dropped' in capsys.readouterr().out
    with torch.no_grad():
        assert torch.equal(new(x), expected)
    assert lora_coverage(new)['lora_modules'] == 28


def test_upgrade_leaves_the_file_state_dict_alone(no_downloads):
    """The error model's fingerprint is taken from the file's state dict after loading."""
    old = _OldSwin3D(3, 'swin3d_t', 1, 8, lora_rank=2)
    state = old.state_dict()
    keys = list(state)
    new = adjustedPeftVideo3D(3, 'swin3d_t', 1, 8, lora_rank=2)
    upgraded, count = upgrade_legacy_lora_state(new, state)
    assert count == 24 and upgraded is not state and list(state) == keys


# --- 2D models ---------------------------------------------------------------
def _tiny_swin(weights=None):
    return torchvision.models.SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[1, 1], num_heads=[2, 2],
                                              window_size=[2, 2])


def _tiny_vit(weights=None):
    return torchvision.models.VisionTransformer(image_size=64, patch_size=16, num_layers=2, num_heads=2, hidden_dim=16,
                                                mlp_dim=32)


@pytest.mark.parametrize('cls', [adjustedPeftResNet, adjustedPeftDenseNet, adjustedPeftEfficientNet, adjustedPeftVGGNet,
                                 adjustedPeftConvNext, adjustedPeftSwin],
                         ids=['resnet', 'densenet', 'efficientnet', 'vgg', 'convnext', 'swin'])
def test_every_2d_adapter_has_a_gradient_on_the_dummy_version(cls):
    torch.manual_seed(0)
    model = cls(3, 'dummy', 1, 2)
    assert _lora_b(model), 'this model has no adapters at all'
    assert _dead_adapters(model, torch.rand(2, 3, 64, 64)) == []
    assert lora_coverage(model)['lora_modules'] == len(_lora_b(model))


def test_every_vit_adapter_has_a_gradient(monkeypatch):
    """Same layer types as the dummy version, vit_h_14, whose 632M parameters do not suit a CPU unit test."""
    monkeypatch.setattr(torchvision.models, 'vit_h_14', _tiny_vit)
    torch.manual_seed(0)
    model = adjustedViTPeft(3, 'dummy', 1, 2)
    assert _lora_b(model)
    assert _dead_adapters(model, torch.rand(2, 3, 64, 64)) == []
    assert lora_coverage(model)['lora_modules'] == len(_lora_b(model))


class _OldSwin(adjustedPeftSwin):
    """The targets before V-040: attention qkv and proj only."""

    def applyPEFT(self, model):
        from peft import LoraConfig, get_peft_model
        targets = [name for name, module in model.named_modules()
                   if any(layer in name for layer in ['qkv', 'proj', 'head']) and isinstance(module, nn.Linear)]
        return get_peft_model(model, LoraConfig(r=self.lora_attention_dimension, lora_alpha=16, lora_dropout=0.1,
                                                bias='none', target_modules=targets))


@pytest.fixture
def tiny_swin(monkeypatch):
    monkeypatch.setattr(torchvision.models, 'swin_b', _tiny_swin)


def _old_swin(num_classes=3, added_layers=1, rank=2):
    torch.manual_seed(0)
    old = _OldSwin(num_classes, 'dummy', added_layers, rank)
    _nudge_adapters(old)
    return old


def test_the_old_2d_swin_targets_never_trained(tiny_swin):
    """Guards the fix: on the same model all adapters of the old selection get no gradient."""
    old = _old_swin()
    assert len(_lora_b(old)) == 4
    assert len(_dead_adapters(old, torch.rand(2, 3, 32, 32))) == 4


def test_old_2d_swin_weights_load_and_predict_exactly_as_before(tiny_swin, capsys):
    old = _old_swin().eval()
    x = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = old(x)

    torch.manual_seed(99)
    new = adjustedPeftSwin(3, 'dummy', 1, 2).eval()
    result = new.load_state_dict(old.state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert 'before the Swin LoRA fix' in capsys.readouterr().out
    with torch.no_grad():
        assert torch.equal(new(x), expected)
    assert _dead_adapters(new, x) == []


def test_old_2d_swin_weights_load_through_the_frame_pool_wrapper(tiny_swin):
    """The wrapper's load_state_dict never calls the inner model's, so the upgrade must run as a hook."""
    torch.manual_seed(0)
    old = VideoFramePoolClassifier(_OldSwin, 'dummy', 3, 1, 2).eval()
    _nudge_adapters(old)
    x = torch.rand(1, 2, 3, 32, 32)
    with torch.no_grad():
        expected = old(x)

    torch.manual_seed(99)
    new = VideoFramePoolClassifier(adjustedPeftSwin, 'dummy', 3, 1, 2).eval()
    result = new.load_state_dict(old.state_dict())
    assert result.missing_keys == [] and result.unexpected_keys == []
    with torch.no_grad():
        assert torch.equal(new(x), expected)


def test_current_2d_swin_weights_pass_through_and_a_deep_copy_keeps_the_upgrade(tiny_swin, capsys):
    """ModelEma deep-copies the model, so the hook must act on the copy."""
    model = adjustedPeftSwin(3, 'dummy', 1, 2)
    capsys.readouterr()
    model.load_state_dict(model.state_dict())
    assert 'before the Swin LoRA fix' not in capsys.readouterr().out

    old = _old_swin().eval()
    duplicate = copy.deepcopy(model).eval()
    duplicate.load_state_dict(old.state_dict())
    assert torch.equal(duplicate.fc1.weight, old.fc1.weight)
    assert not torch.equal(model.fc1.weight, old.fc1.weight)


def test_wrong_rank_still_fails_on_old_2d_swin_weights(tiny_swin):
    with pytest.raises(RuntimeError, match='size mismatch'):
        adjustedPeftSwin(3, 'dummy', 2, 5).load_state_dict(_old_swin(added_layers=2, rank=2).state_dict())

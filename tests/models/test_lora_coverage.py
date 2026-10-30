"""LoRA target selection for native video models (V-007) and loading weights
saved before that fix. CPU, weights=None."""
import pytest
import torch
import torch.nn as nn
import torchvision.models.video as video_models

from src.video.video3d_peft import adjustedPeftVideo3D, lora_target_modules, upgrade_legacy_lora_state
from trainers.param_budget import lora_coverage, parameter_group


def _legacy_targets(model):
    """The selection from before the fix: exact types only."""
    available = [nn.Conv3d, nn.Conv2d, nn.Linear]
    return [n for n, m in model.named_modules() if type(m) in available and not (hasattr(m, 'groups') and m.groups != 1)]


def _peft(rank=3):
    return adjustedPeftVideo3D(3, 'dummy_video3d', 2, rank)


def test_every_r3d_18_conv_gets_lora_and_the_adapters_train():
    model = _peft()
    assert lora_coverage(model)['lora_modules'] == 20 and lora_coverage(model)['eligible_modules'] == 20
    x = torch.rand(2, 2, 3, 16, 16)
    model(x).sum().backward()
    # Every lora_A has a gradient path: lora_B starts at zero, so A's gradient
    # is zero at first, but it must exist; B's gradient must be real.
    lora_b = [p for n, p in model.named_parameters() if '.lora_B.' in n]
    assert len(lora_b) == 20
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_b)
    assert all(p.requires_grad for n, p in model.named_parameters() if 'lora_' in n)
    trained = [n for n, p in model.named_parameters() if parameter_group(n) == 'backbone' and p.requires_grad]
    assert trained == [], 'backbone weights stay frozen under LoRA'


def test_the_old_selection_was_the_bug():
    """Guards the test above: on the same model the exact-type check finds only 4 of 20."""
    backbone = video_models.r3d_18(weights=None)
    backbone.fc = nn.Identity()
    assert len(_legacy_targets(backbone)) == 4
    assert len(lora_target_modules(backbone)) == 20


def test_converting_conv_subclasses_changes_nothing_but_the_class():
    from src.video.video3d_peft import plain_conv_subclasses
    torch.manual_seed(0)
    backbone = video_models.r3d_18(weights=None).eval()
    x = torch.rand(1, 3, 2, 16, 16)
    state = {key: value.clone() for key, value in backbone.state_dict().items()}
    with torch.no_grad():
        before = backbone(x)
    plain_conv_subclasses(backbone)
    with torch.no_grad():
        assert torch.equal(backbone(x), before)
    assert set(state) == set(backbone.state_dict())
    assert all(type(m) is nn.Conv3d for m in backbone.modules() if isinstance(m, nn.Conv3d))


def test_a_conv_subclass_with_its_own_forward_is_left_alone():
    from src.video.video3d_peft import plain_conv_subclasses

    class Custom(nn.Conv3d):
        def forward(self, x):
            return super().forward(x) * 2

    net = nn.Sequential(Custom(1, 1, 1))
    assert lora_target_modules(net) == []
    plain_conv_subclasses(net)
    assert type(net[0]) is Custom


@pytest.mark.parametrize('builder', [video_models.mc3_18, video_models.r2plus1d_18])
def test_other_3d_cnns_are_fully_covered(builder):
    backbone = builder(weights=None)
    backbone.fc = nn.Identity()
    convs = [n for n, m in backbone.named_modules() if isinstance(m, nn.Conv3d)]
    assert lora_target_modules(backbone) == convs


def test_grouped_convs_and_attention_internals_are_skipped():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.plain = nn.Conv3d(4, 4, 1)
            self.depthwise = nn.Conv3d(4, 4, 1, groups=4)
            self.attention = nn.MultiheadAttention(4, 2)
            self.proj = nn.Linear(4, 4)

    assert lora_target_modules(Net()) == ['plain', 'proj']


def test_video_transformer_targets_are_unchanged_by_the_fix():
    backbone = video_models.swin3d_t(weights=None)
    backbone.head = nn.Identity()
    targets = lora_target_modules(backbone)
    assert set(targets) >= set(_legacy_targets(backbone))
    assert any('mlp' in name for name in targets)


def _legacy_state(rank=3, seed=0):
    """A state dict as an old run saved it, built with the old selection, with trained-looking LoRA."""
    from peft import LoraConfig, get_peft_model

    class Old(adjustedPeftVideo3D):
        def applyPEFT(self, model):
            config = LoraConfig(r=self.lora_attention_dimension, lora_alpha=16, lora_dropout=0.1, bias='none',
                                target_modules=_legacy_targets(model))
            return get_peft_model(model, config)

    torch.manual_seed(seed)
    old = Old(3, 'dummy_video3d', 2, rank)
    with torch.no_grad():
        for name, param in old.named_parameters():
            if 'lora_B' in name:
                param.add_(0.01)  # non-zero, so a mistake in the conversion would show
    return old, old.state_dict()


def test_old_weights_load_and_predict_exactly_as_before():
    old, state = _legacy_state()
    assert sum('.lora_A.' in key and key.startswith('peftmodel.') for key in state) == 4
    x = torch.rand(2, 2, 3, 16, 16)
    old.eval()
    with torch.no_grad():
        expected = old(x)

    torch.manual_seed(99)
    new = _peft().eval()
    assert new.load_state_dict(state).missing_keys == []
    with torch.no_grad():
        assert torch.equal(new(x), expected)
    assert lora_coverage(new)['lora_modules'] == 20


def test_old_weights_load_with_a_clear_log_line(capsys):
    _, state = _legacy_state()
    _peft().load_state_dict(state)
    assert 'before the LoRA coverage fix' in capsys.readouterr().out


def test_current_weights_pass_through_untouched():
    model = _peft()
    state = model.state_dict()
    same, upgraded = upgrade_legacy_lora_state(model, state)
    assert upgraded == 0 and same is state
    model.load_state_dict(state)


def test_weights_of_another_shape_fail_with_a_clear_message():
    _, state = _legacy_state(rank=3)
    with pytest.raises(RuntimeError, match='--embed_size'):
        _peft(rank=5).load_state_dict(state)
    with pytest.raises(RuntimeError, match='do not fit'):
        _peft().load_state_dict({'nonsense.weight': torch.zeros(1)})

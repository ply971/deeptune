"""Frozen backbones for snippet feature extraction (``extract_features.py``).

A :class:`SnippetBackbone` wraps a torchvision model and exposes its final
feature grid, whatever its family:

    ``token_grid(x)``: ``[B, T, C, H, W]`` -> ``[B, T', H', W', D]``
    ``forward(x, extract_embed=False)``: the grid's mean, ``[B, D]``

Frame backbones (ResNet, DenseNet, EfficientNet, VGG, ConvNeXt, Swin, ViT) run
on every frame, so ``T' = T``. Native video backbones (r3d_18, mc3_18,
r2plus1d_18, mvit_v2_s, swin3d_*) downsample time themselves: r3d_18 gives
``T' = T / 8``. ViT and MViT grids hold the patch tokens without the class
token, so a vector from those families is the patch mean, not the class token.

Motion input has 2 channels (dx, dy): ``in_channels=2`` replaces the first
convolution by one whose weights are the RGB filters averaged over channels
and repeated, the cross-modality initialisation of TSN (Wang et al., ECCV 2016).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from src.video.video3d import _KINETICS400_WEIGHTS, VIDEO3D_ARCHITECTURES

DUMMY_VIDEO_BACKBONE = 'dummy_video3d'  # r3d_18 without weights, for tests
_R3D_FAMILY = ('r3d_18', 'mc3_18', 'r2plus1d_18')
_FRAME_FAMILIES = ('resnet', 'densenet', 'efficientnet', 'vgg', 'convnext', 'swin', 'vit')
# Weight terms worth carrying with the features. Torchvision's code is BSD-3-Clause;
# the weights inherit the terms of the data they were trained on.
_LICENCE_NOTES = {
    'imagenet': 'torchvision weights (code BSD-3-Clause), trained on ImageNet-1K, whose terms are non-commercial research only; check before other use',
    'kinetics': 'torchvision weights (code BSD-3-Clause), trained on Kinetics-400 (YouTube videos with their own terms); check before commercial use',
    'none': 'none: random initialisation, no pretrained weights',
}


def _grid_resnet(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


def _grid_densenet(model, x):
    return F.relu(model.features(x))


def _grid_features(model, x):
    return model.features(x)


def _grid_swin(model, x):
    return model.norm(model.features(x)).permute(0, 3, 1, 2)  # NHWC -> NCHW


def _grid_vit(model, x):
    tokens = model._process_input(x)
    cls = model.class_token.expand(tokens.shape[0], -1, -1)
    tokens = model.encoder(torch.cat([cls, tokens], dim=1))[:, 1:]
    side = int(round(tokens.shape[1] ** 0.5))
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)


_FRAME_GRIDS = {'resnet': _grid_resnet, 'densenet': _grid_densenet, 'efficientnet': _grid_features, 'vgg': _grid_features,
                'convnext': _grid_features, 'swin': _grid_swin, 'vit': _grid_vit}


def _grid_r3d(model, x):  # [B, C, T, H, W] -> [B, T', H', W', D]
    x = model.layer4(model.layer3(model.layer2(model.layer1(model.stem(x)))))
    return x.permute(0, 2, 3, 4, 1)


def _grid_swin3d(model, x):
    return model.norm(model.features(model.pos_drop(model.patch_embed(x))))


def _grid_mvit(model, x):
    x = model.conv_proj(x).flatten(2).transpose(1, 2)
    x = model.pos_encoding(x)
    thw = (model.pos_encoding.temporal_size,) + tuple(model.pos_encoding.spatial_size)
    for block in model.blocks:
        x, thw = block(x, thw)
    x = model.norm(x)[:, 1:]  # drop the class token
    return x.reshape(x.shape[0], thw[0], thw[1], thw[2], x.shape[2])


def frame_family(version: str) -> str | None:
    for family in _FRAME_FAMILIES:
        if version.startswith(family):
            return family
    return None


def _inflate_first_conv(model: nn.Module, in_channels: int) -> None:
    """Replace the first 3-channel convolution by an `in_channels` one (see the module docstring)."""
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.Conv3d)) and child.in_channels == 3:
                replacement = type(child)(in_channels, child.out_channels, child.kernel_size, child.stride, child.padding,
                                          child.dilation, child.groups, child.bias is not None, child.padding_mode)
                with torch.no_grad():
                    mean = child.weight.mean(dim=1, keepdim=True)
                    replacement.weight.copy_(mean.repeat(1, in_channels, *[1] * (mean.ndim - 2)))
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                setattr(module, name, replacement)
                return
    raise ValueError(f'{type(model).__name__} has no 3-channel input convolution to inflate.')


class SnippetBackbone(nn.Module):
    """A frozen backbone in eval mode. `version`, `weights_name`, `weights_licence` and
    `default_input_size` describe it for the feature sidecars."""

    def __init__(self, version: str, pretrained: bool = True, in_channels: int = 3, seed: int = 0):
        super().__init__()
        self.version = version
        self.in_channels = in_channels
        family = frame_family(version)
        native = version in VIDEO3D_ARCHITECTURES or version == DUMMY_VIDEO_BACKBONE
        if not native and family is None:
            raise ValueError(f'Unknown feature backbone {version!r}. Use a native video architecture '
                             f'({", ".join(VIDEO3D_ARCHITECTURES)}) or a frame backbone such as resnet18.')
        pretrained = pretrained and version != DUMMY_VIDEO_BACKBONE
        builder_name = 'r3d_18' if version == DUMMY_VIDEO_BACKBONE else version
        if native:
            weights = _KINETICS400_WEIGHTS[builder_name] if pretrained else None
            self._grid_fn = (_grid_r3d if builder_name in _R3D_FAMILY
                             else _grid_mvit if builder_name == 'mvit_v2_s' else _grid_swin3d)
        else:
            try:
                available = tv_models.get_model_weights(version)  # also rejects unknown names
            except ValueError as exc:
                raise ValueError(f'Unknown feature backbone {version!r}: {exc}') from exc
            weights = available.DEFAULT if pretrained else None
            self._grid_fn = _FRAME_GRIDS[family]
        self.native = native
        # Random weights must be reproducible, or "extract twice" gives different arrays.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = tv_models.get_model(builder_name, weights=weights)
            if in_channels != 3:
                _inflate_first_conv(self.model, in_channels)
        self.weights_name = str(weights) if weights is not None else None
        self.weights_licence = _LICENCE_NOTES['none' if weights is None else 'kinetics' if native else 'imagenet']
        if builder_name in _R3D_FAMILY:
            self.default_input_size = 112
        else:
            self.default_input_size = int(weights.transforms().crop_size[0]) if weights is not None else 224
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):  # always frozen: BatchNorm and dropout stay in eval mode
        return super().train(False)

    def token_grid(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` -> ``[B, T', H', W', D]``."""
        B, T = x.shape[:2]
        if self.native:
            return self._grid_fn(self.model, x.permute(0, 2, 1, 3, 4))
        grid = self._grid_fn(self.model, x.reshape(B * T, *x.shape[2:]))  # [B*T, D, h, w]
        grid = grid.permute(0, 2, 3, 1)
        return grid.reshape(B, T, *grid.shape[1:])

    def forward(self, x: torch.Tensor, extract_embed: bool = False) -> torch.Tensor:
        return self.token_grid(x).mean(dim=(1, 2, 3))


def resolve_token_grid(requested: tuple[int, int, int] | None, available: tuple[int, int, int]) -> tuple[int, int, int] | None:
    """The grid to store: never more cells than the backbone produced. Upsampling would
    only repeat values while claiming a finer resolution, so each axis is capped."""
    if requested is None:
        return None
    return tuple(min(want, have) for want, have in zip(requested, available))


def pool_token_grid(grid: torch.Tensor, target: tuple[int, int, int] | None) -> torch.Tensor:
    """``[B, T', H', W', D]`` -> ``[B, D]`` (target None) or ``[B, t, h, w, D]`` (adaptive average pooling)."""
    if target is None:
        return grid.mean(dim=(1, 2, 3))
    if tuple(grid.shape[1:4]) == tuple(target):
        return grid
    return F.adaptive_avg_pool3d(grid.permute(0, 4, 1, 2, 3), target).permute(0, 2, 3, 4, 1)

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

Motion input has 2 channels (dx, dy): ``in_channels=2`` feeds dx, dy and their
magnitude to the unchanged RGB filters. This preserves direction information;
these are appearance weights, not weights trained on optical flow.

Backbones from ``transformers`` (V-JEPA 2, VideoMAE, SigLIP, DINOv3; ``handlers/video_backbones.py``) provide the same
two calls. Their grids hold patch tokens only, and each set of weights reports its own licence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from handlers.video_backbones import hub_backbone, licence_text, weights_revision
from src.video.video3d import _KINETICS400_WEIGHTS, VIDEO3D_ARCHITECTURES

DUMMY_VIDEO_BACKBONE = 'dummy_video3d'  # r3d_18 without weights, for tests
_R3D_FAMILY = ('r3d_18', 'mc3_18', 'r2plus1d_18')
_FRAME_FAMILIES = ('resnet', 'densenet', 'efficientnet', 'vgg', 'convnext', 'swin', 'vit')
# Licence notes are per set of weights, never per architecture: ViT-H's DEFAULT is a SWAG checkpoint with its
# own non-commercial licence, unlike the other ViT weights. Torchvision only links to a licence, and only for a few.
_LICENCE_TERMS = {'https://github.com/facebookresearch/SWAG/blob/main/LICENSE': 'CC BY-NC 4.0 (non-commercial)'}
_TRAINING_DATA = {'IMAGENET1K': 'ImageNet-1K (non-commercial research terms)',
                  'KINETICS400': 'Kinetics-400 (YouTube videos with their own terms)'}


def weight_licence(weights) -> str:
    """What is known about the terms of exactly these weights, from their torchvision metadata (needs no download)."""
    if weights is None:
        return 'none: random initialisation, no pretrained weights'
    url = weights.meta.get('license')
    if url:
        terms = f'{_LICENCE_TERMS[url]}, {url}' if url in _LICENCE_TERMS else f'licence at {url}'
    else:
        terms = (f'no weight licence in torchvision metadata, check the recipe {weights.meta.get("recipe", "(none given)")}; '
                 "torchvision's BSD-3-Clause covers its code, not these weights")
    data = next((note for key, note in _TRAINING_DATA.items() if key in weights.name), 'see the recipe')
    return f'{weights}: {terms}; training data: {data}'


def _frame_weights(version: str):
    """The torchvision weights src/vision builds for this frame backbone, or None when it has none: ResNet and 2D Swin
    pin IMAGENET1K_V1, the other families use DEFAULT."""
    try:
        available = tv_models.get_model_weights(version)
    except ValueError:  # resnet_tiny and other untrained test versions
        return None
    return available.IMAGENET1K_V1 if version.startswith(('resnet', 'swin_')) else available.DEFAULT


def run_weight_licence(version: str) -> str:
    """The terms of the weights a model of this version loads, as a run records them."""
    if hub_backbone(version) is not None:
        return licence_text(version)
    if version in VIDEO3D_ARCHITECTURES:
        return weight_licence(_KINETICS400_WEIGHTS[version])
    if frame_family(version) is not None:
        return weight_licence(_frame_weights(version))
    return weight_licence(None)


def record_weights(args, version: str) -> None:
    """Put the licence and exact revision of `version`'s weights on a run's arguments, so its record holds them."""
    args.weights_licence = run_weight_licence(version)
    args.weights_revision = weights_revision(version)


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


def fixed_input(version: str, pretrained: bool) -> tuple[int | None, int | None]:
    """(square size, snippet frames) this model accepts and nothing else; None where it is free. Read from the
    weights' metadata, so it needs no download. ViT and MViT have fixed position embeddings."""
    hub = hub_backbone(version)
    if hub is not None:
        return hub.input_size, hub.fixed_frames
    if version == 'mvit_v2_s':
        meta = _KINETICS400_WEIGHTS[version].meta
        return meta['min_size'][0], meta['min_temporal_size']
    if frame_family(version) == 'vit':
        if not pretrained:
            return 224, None  # torchvision's default image_size when no weights fix it
        try:
            return tv_models.get_model_weights(version).DEFAULT.meta['min_size'][0], None
        except ValueError:  # unknown name: SnippetBackbone reports it
            return None, None
    return None, None


class SnippetBackbone(nn.Module):
    """A frozen backbone in eval mode. `version`, `weights_name`, `weights_licence` and
    `default_input_size` describe it for the feature sidecars."""

    def __init__(self, version: str, pretrained: bool = True, in_channels: int = 3, seed: int = 0):
        super().__init__()
        self.version = version
        self.in_channels = in_channels
        if in_channels not in (2, 3):
            raise ValueError('Snippet backbones accept RGB (3 channels) or motion (dx, dy; 2 channels).')
        self.hub = hub_backbone(version)
        if self.hub is not None:
            self._init_hub(pretrained, seed)
            return
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
        self.weights_name = str(weights) if weights is not None else None
        self.weights_licence = weight_licence(weights)
        self.weights_revision = weights_revision(version)
        if builder_name in _R3D_FAMILY:
            self.default_input_size = 112
        else:
            self.default_input_size = int(weights.transforms().crop_size[0]) if weights is not None else 224
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _init_hub(self, pretrained: bool, seed: int) -> None:
        from src.video.hub_backbones import load_hub_model
        if not pretrained and not self.hub.is_untrained:
            raise ValueError(f'{self.version} has no random-weight mode: use its _tiny version for tests and timing.')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = load_hub_model(self.version)
        self.native = False
        self.weights_name = self.hub.hub_id
        self.weights_licence = licence_text(self.version)
        self.weights_revision = weights_revision(self.version)
        self.default_input_size = self.hub.input_size
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):  # always frozen: BatchNorm and dropout stay in eval mode
        return super().train(False)

    def token_grid(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` -> ``[B, T', H', W', D]``."""
        B, T = x.shape[:2]
        if self.in_channels == 2:
            magnitude = torch.linalg.vector_norm(x, dim=2, keepdim=True)
            x = torch.cat((x, magnitude), dim=2)
        if self.hub is not None:
            from src.video.hub_backbones import encoder_token_grid
            return encoder_token_grid(self.model, self.version, x)
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

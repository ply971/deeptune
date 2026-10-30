"""Temporal video architectures: the "four places to fuse time" (Karpathy et al., CVPR 2014), the two-stream
ConvNet (Simonyan & Zisserman, NeurIPS 2014), the Temporal Shift Module (Lin et al., ICCV 2019), and recurrent and
attention heads over per-frame embeddings. Built through ``utils.build_video_model`` (see handlers/video_arch.py).

Contract, shared with every DeepTune video model: ``forward(x, extract_embed=False)`` with x ``[B, T, C, H, W]``
returns ``[B, num_classes]`` (cls) or ``[B, 1]`` (reg), or with ``extract_embed=True`` the clip embedding the head
sees. The two-stream model takes C = 5: RGB + (dx, dy), attached by the dataset (``VideoDataOptions.clip_motion``).

Parameter names follow trainers/param_budget.py: the 2D network lives under ``backbone.model.`` and everything else
is the head, so ``--trainable_params`` and the parameter report work unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from handlers.video_arch import TEMPORAL_ARCHITECTURES


def _head_output_size(num_classes: int, task_type: str) -> int:
    if task_type not in ('cls', 'reg'):
        raise ValueError("task_type must be 'cls' or 'reg'.")
    return num_classes if task_type == 'cls' else 1


class ClipHead(nn.Module):
    """DeepTune's added layers, on clip features. ``added_layers=1``: the features are the embedding, then a linear
    classifier. ``added_layers=2``: fc1 gives the embedding, then ReLU and fc2 (the same as the adjusted*Net heads)."""

    def __init__(self, in_features: int, out_features: int, added_layers: int, embed_size: int):
        super().__init__()
        if added_layers not in (1, 2):
            raise ValueError(f'Temporal video models need added_layers 1 or 2, got {added_layers}.')
        self.added_layers = added_layers
        self.embedding_dim = embed_size if added_layers == 2 else in_features
        self.fc1 = nn.Linear(in_features, embed_size if added_layers == 2 else out_features)
        if added_layers == 2:
            self.fc2 = nn.Linear(embed_size, out_features)

    def embed(self, features):
        return self.fc1(features) if self.added_layers == 2 else features

    def classify(self, embedding):
        return self.fc2(torch.relu(embedding)) if self.added_layers == 2 else self.fc1(embedding)


def _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type):
    """An adjusted*Net (PEFT or not) reduced to a feature extractor. Returns (network, feature size)."""
    network = frame_model_cls(num_classes, version, 1, embed_size, freeze_backbone, task_type=task_type)
    size = network.fc1.in_features
    network.fc1 = nn.Identity()
    return network, size


def _frame_features(backbone, x):
    """Per-frame features ``[B, T, D]`` from a clip ``[B, T, C, H, W]``."""
    batch, frames = x.shape[:2]
    return backbone(x.reshape(batch * frames, *x.shape[2:]), extract_embed=True).reshape(batch, frames, -1)


def _check_clip(x, channels: int = 3):
    if x.dim() != 5 or x.shape[2] != channels:
        raise ValueError(f'Expected a clip batch shaped [B, T, {channels}, H, W], got {tuple(x.shape)}.')


def _resnet_core(network, arch: str, version: str):
    core = getattr(network, 'model', None)
    if core is None or not hasattr(core, 'conv1') or not hasattr(core, 'layer1'):
        raise ValueError(f'{arch} supports ResNet-family backbones (resnet18, resnet50, ...), not {version!r}.')
    return core


def inflate_conv(conv: nn.Conv2d, in_channels: int, cross_modality: bool = False) -> nn.Conv2d:
    """A copy of a first convolution taking `in_channels` channels, from the 2D weights (I3D-style inflation).

    RGB stacks (a multiple of the original channels) replicate the kernel and divide by the factor, so identical input
    frames give the 2D output. Motion stacks (`cross_modality`) replicate the kernel averaged over its channels, keeping
    the summed weight (cross-modality initialisation, Wang et al., TSN, ECCV 2016)."""
    if not cross_modality and in_channels % conv.in_channels:
        raise ValueError(f'An RGB stack needs a multiple of {conv.in_channels} channels, got {in_channels}.')
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation,
                    conv.groups, conv.bias is not None, conv.padding_mode).to(conv.weight.device, conv.weight.dtype)
    with torch.no_grad():
        old = conv.weight
        if not cross_modality:
            factor = in_channels // old.shape[1]
            new.weight.copy_(old.repeat(1, factor, 1, 1) / factor)
        else:
            new.weight.copy_(old.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * old.shape[1] / in_channels)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    new.weight.requires_grad = conv.weight.requires_grad
    if conv.bias is not None:
        new.bias.requires_grad = conv.bias.requires_grad
    return new


def inflate_first_conv(core, in_channels: int, cross_modality: bool = False) -> None:
    """Inflate ``core.conv1`` in place. Under LoRA the base layer and the adapter's input convolution both inflate."""
    conv = core.conv1
    if hasattr(conv, 'base_layer'):
        conv.base_layer = inflate_conv(conv.base_layer, in_channels, cross_modality)
        for name, adapter in conv.lora_A.items():
            conv.lora_A[name] = inflate_conv(adapter, in_channels, cross_modality)
        conv.in_features = in_channels
    else:
        core.conv1 = inflate_conv(conv, in_channels, cross_modality)


def temporal_shift(x: torch.Tensor, num_frames: int, fold_div: int) -> torch.Tensor:
    """TSM's shift on ``[B*T, C, H, W]``: the first C/fold_div channels move one frame back in time, the next
    C/fold_div one frame forward, the rest stay. Vacated positions are zero, as in the paper."""
    channels = x.shape[1]
    fold = channels // fold_div
    if fold == 0 or num_frames < 2:
        return x
    clip = x.reshape(-1, num_frames, *x.shape[1:])
    out = torch.zeros_like(clip)
    out[:, :-1, :fold] = clip[:, 1:, :fold]
    out[:, 1:, fold:2 * fold] = clip[:, :-1, fold:2 * fold]
    out[:, :, 2 * fold:] = clip[:, :, 2 * fold:]
    return out.reshape_as(x)


class _ShiftHook:
    """A forward pre-hook on a residual block's first convolution: the shift happens only inside the residual
    branch, the identity path keeps the unshifted features. Hooks add no parameters and keep the 2D names."""

    def __init__(self, state: dict, fold_div: int):
        self.state, self.fold_div = state, fold_div

    def __call__(self, module, inputs):
        return (temporal_shift(inputs[0], self.state['num_frames'], self.fold_div), *inputs[1:])


class VideoLateFusion(nn.Module):
    """Late fusion: a shared 2D backbone on the first and last sampled frames, embeddings concatenated and merged
    by FC layers. The merge always has a hidden layer, because a linear head on the concatenation cannot compare
    the two frames. The embedding is the concatenation."""

    def __init__(self, frame_model_cls, version, num_classes, added_layers=2, embed_size=1000, freeze_backbone=False,
                 task_type='cls'):
        super().__init__()
        self.backbone, size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        hidden = embed_size if added_layers == 2 else 2 * size
        self.fc1 = nn.Linear(2 * size, hidden)
        self.fc2 = nn.Linear(hidden, _head_output_size(num_classes, task_type))
        self.embedding_dim = 2 * size

    def forward(self, x, extract_embed=False):
        _check_clip(x)
        features = torch.cat([self.backbone(x[:, 0], extract_embed=True), self.backbone(x[:, -1], extract_embed=True)], dim=1)
        return features if extract_embed else self.fc2(torch.relu(self.fc1(features)))


class VideoEarlyFusion(nn.Module):
    """Early fusion: all T frames stacked on the channels of the first convolution, inflated to 3T inputs from the
    2D weights; the rest of the backbone is unchanged. T is fixed when the model is built."""

    def __init__(self, frame_model_cls, version, num_classes, added_layers=2, embed_size=1000, freeze_backbone=False,
                 task_type='cls', num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.backbone, size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        inflate_first_conv(_resnet_core(self.backbone, 'early_fusion', version), 3 * num_frames)
        self.head = ClipHead(size, _head_output_size(num_classes, task_type), added_layers, embed_size)
        self.embedding_dim = self.head.embedding_dim

    def forward(self, x, extract_embed=False):
        _check_clip(x)
        if x.shape[1] != self.num_frames:
            raise ValueError(f'This early_fusion model was built for {self.num_frames} frames, got {x.shape[1]}.')
        embedding = self.head.embed(self.backbone(x.flatten(1, 2), extract_embed=True))
        return embedding if extract_embed else self.head.classify(embedding)


class VideoTSM(nn.Module):
    """Temporal Shift Module: a 2D ResNet with a shift in each residual block ("residual shift"), per-frame features
    averaged over time. No extra parameters, and the state-dict names of the 2D backbone are unchanged."""

    def __init__(self, frame_model_cls, version, num_classes, added_layers=2, embed_size=1000, freeze_backbone=False,
                 task_type='cls', tsm_fold_div=8):
        super().__init__()
        if tsm_fold_div < 2:
            raise ValueError(f'--tsm_fold_div must be at least 2: 1/N of the channels shift each way, got {tsm_fold_div}.')
        self.tsm_fold_div = tsm_fold_div
        self.backbone, size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        core = _resnet_core(self.backbone, 'tsm', version)
        self._shift_state = {'num_frames': 1}
        for module in core.modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                module.conv1.register_forward_pre_hook(_ShiftHook(self._shift_state, tsm_fold_div))
        self.head = ClipHead(size, _head_output_size(num_classes, task_type), added_layers, embed_size)
        self.embedding_dim = self.head.embedding_dim

    def forward(self, x, extract_embed=False):
        _check_clip(x)
        self._shift_state['num_frames'] = x.shape[1]
        embedding = self.head.embed(_frame_features(self.backbone, x).mean(dim=1))
        return embedding if extract_embed else self.head.classify(embedding)


class VideoCNNRNN(nn.Module):
    """CNN + RNN: per-frame features into an LSTM or GRU; the last layer's final state is the clip feature."""

    def __init__(self, frame_model_cls, version, num_classes, added_layers=2, embed_size=1000, freeze_backbone=False,
                 task_type='cls', rnn_type='lstm', rnn_hidden=256, rnn_layers=1, bidirectional=False):
        super().__init__()
        self.backbone, size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        cell = {'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.rnn = cell(size, rnn_hidden, rnn_layers, batch_first=True, bidirectional=bidirectional)
        self.head = ClipHead(rnn_hidden * (2 if bidirectional else 1), _head_output_size(num_classes, task_type),
                             added_layers, embed_size)
        self.embedding_dim = self.head.embedding_dim

    def forward(self, x, extract_embed=False):
        _check_clip(x)
        _, state = self.rnn(_frame_features(self.backbone, x))
        hidden = state[0] if isinstance(state, tuple) else state  # LSTM: (h, c); GRU: h
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.rnn.bidirectional else hidden[-1]
        embedding = self.head.embed(hidden)
        return embedding if extract_embed else self.head.classify(embedding)


def _sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """The encoding of Vaswani et al. (2017), shaped ``[1, length, dim]``."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    rate = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * rate)
    table[:, 1::2] = torch.cos(position * rate[:dim // 2])
    return table.unsqueeze(0)


class VideoTemporalTransformer(nn.Module):
    """Per-frame features projected to normalised tokens, a learned positional encoding and CLS token, a small
    TransformerEncoder. The CLS token is the clip feature. T is at most the `num_frames` it was built for."""

    def __init__(self, frame_model_cls, version, num_classes, added_layers=2, embed_size=1000, freeze_backbone=False,
                 task_type='cls', num_frames=8, tt_layers=2, tt_heads=4, tt_dim=256):
        super().__init__()
        if tt_dim % tt_heads:
            raise ValueError(f'--tt_dim ({tt_dim}) must be a multiple of --tt_heads ({tt_heads}).')
        self.backbone, size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        self.project = nn.Sequential(nn.Linear(size, tt_dim), nn.LayerNorm(tt_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, tt_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Learned, but starting from the sinusoidal encoding at the scale of the normalised tokens. With unnormalised
        # tokens and the usual small random start the model memorises frames before it can tell their order
        # (direction, 1500 clips, held out: 0.545 against 1.000).
        self.positions = nn.Parameter(_sinusoidal_positions(num_frames + 1, tt_dim))
        layer = nn.TransformerEncoderLayer(tt_dim, tt_heads, dim_feedforward=4 * tt_dim, dropout=0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, tt_layers, norm=nn.LayerNorm(tt_dim), enable_nested_tensor=False)
        self.head = ClipHead(tt_dim, _head_output_size(num_classes, task_type), added_layers, embed_size)
        self.embedding_dim = self.head.embedding_dim

    def forward(self, x, extract_embed=False):
        _check_clip(x)
        if x.shape[1] + 1 > self.positions.shape[1]:
            raise ValueError(f'This temporal_transformer was built for at most {self.positions.shape[1] - 1} frames, got {x.shape[1]}.')
        tokens = self.project(_frame_features(self.backbone, x))
        tokens = torch.cat([self.cls_token.expand(len(x), -1, -1), tokens], dim=1) + self.positions[:, :tokens.shape[1] + 1]
        embedding = self.head.embed(self.encoder(tokens)[:, 0])
        return embedding if extract_embed else self.head.classify(embedding)


class _StreamPair(nn.Module):
    """Holds the two ResNets as ``backbone.model.rgb`` and ``backbone.model.motion``, which is where
    trainers/param_budget.py looks for pretrained parameters."""

    def __init__(self, rgb, motion):
        super().__init__()
        self.model = nn.ModuleDict({'rgb': rgb, 'motion': motion})


class VideoTwoStream(nn.Module):
    """Two-stream ConvNet: a spatial stream on the middle RGB frame, a temporal stream on L stacked motion fields
    (2L channels, first convolution inflated), fused late.

    ``fusion='avg'``: each stream has its own head and the logits are averaged. ``fusion='fc'``: one head on the
    concatenated features. Input is ``[B, T, 5, H, W]``: RGB, then (dx, dy) normalised by the dataset, see
    datasets/motion_stack.py. L fields are taken around the middle of the clip, so T must be at least L.

    ``extract_embed=True`` returns the embedding the head sees: for ``avg`` the two stream embeddings concatenated
    (split them with ``embedding_split``), for ``fc`` the fused head's embedding. ``extract_embed='separate'`` returns
    ``{'rgb': [B, Dr], 'motion': [B, Dm]}`` under either fusion, for long-video localisation.
    """

    def __init__(self, frame_model_cls, version, num_classes, added_layers=2, embed_size=1000, freeze_backbone=False,
                 task_type='cls', num_frames=8, fusion='avg', motion_length=None):
        super().__init__()
        if fusion not in ('avg', 'fc'):
            raise ValueError(f"fusion must be 'avg' or 'fc', got {fusion!r}.")
        self.fusion = fusion
        self.motion_length = motion_length or num_frames
        if self.motion_length > num_frames:
            raise ValueError(f'--motion_length ({self.motion_length}) cannot exceed the frames per clip ({num_frames}).')
        rgb, rgb_size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        motion, motion_size = _feature_backbone(frame_model_cls, version, num_classes, embed_size, freeze_backbone, task_type)
        _resnet_core(rgb, 'two_stream', version)
        inflate_first_conv(_resnet_core(motion, 'two_stream', version), 2 * self.motion_length, cross_modality=True)
        self.backbone = _StreamPair(rgb, motion)
        out = _head_output_size(num_classes, task_type)
        if fusion == 'avg':
            self.rgb_head = ClipHead(rgb_size, out, added_layers, embed_size)
            self.motion_head = ClipHead(motion_size, out, added_layers, embed_size)
            self.stream_dims = (self.rgb_head.embedding_dim, self.motion_head.embedding_dim)
            self.embedding_dim = sum(self.stream_dims)
        else:
            self.fusion_head = ClipHead(rgb_size + motion_size, out, added_layers, embed_size)
            self.stream_dims = (rgb_size, motion_size)
            self.embedding_dim = self.fusion_head.embedding_dim

    @property
    def embedding_split(self) -> dict:
        """Slices of ``extract_embed='separate'`` concatenated rgb-first, the order of B1's snippet features."""
        return {'rgb': (0, self.stream_dims[0]), 'motion': (self.stream_dims[0], sum(self.stream_dims))}

    def _motion_stack(self, x):
        frames = x.shape[1]
        if frames < self.motion_length:
            raise ValueError(f'This two_stream model stacks {self.motion_length} motion fields, but the clip has {frames} frames.')
        start = (frames - self.motion_length) // 2
        return x[:, start:start + self.motion_length, 3:5].flatten(1, 2)

    def _stream_embeddings(self, x):
        if x.dim() != 5 or x.shape[2] != 5:
            raise ValueError(f'two_stream expects [B, T, 5, H, W] (RGB + dx, dy), got {tuple(x.shape)}. Open the dataset with '
                             "VideoDataOptions(clip_motion='farneback'|'mv'); video_options_from_args does this for --video_arch two_stream.")
        rgb = self.backbone.model['rgb'](x[:, x.shape[1] // 2, :3], extract_embed=True)
        motion = self.backbone.model['motion'](self._motion_stack(x), extract_embed=True)
        if self.fusion == 'avg':
            return self.rgb_head.embed(rgb), self.motion_head.embed(motion)
        return rgb, motion

    def forward(self, x, extract_embed=False):
        rgb, motion = self._stream_embeddings(x)
        if extract_embed == 'separate':
            return {'rgb': rgb, 'motion': motion}
        if self.fusion == 'avg':
            if extract_embed:
                return torch.cat([rgb, motion], dim=1)
            return (self.rgb_head.classify(rgb) + self.motion_head.classify(motion)) / 2
        embedding = self.fusion_head.embed(torch.cat([rgb, motion], dim=1))
        return embedding if extract_embed else self.fusion_head.classify(embedding)


_MODELS = {'late_fusion': VideoLateFusion, 'early_fusion': VideoEarlyFusion, 'tsm': VideoTSM, 'cnn_rnn': VideoCNNRNN,
           'temporal_transformer': VideoTemporalTransformer, 'two_stream': VideoTwoStream}
assert set(_MODELS) == set(TEMPORAL_ARCHITECTURES)


def build_temporal_model(arch: str, options: dict, frame_model_cls, version: str, num_classes: int, added_layers: int,
                         embed_size: int, freeze_backbone: bool, task_type: str) -> nn.Module:
    """One of the architectures above; `options` come from handlers.video_arch.parse_pooling_spec."""
    return _MODELS[arch](frame_model_cls, version, num_classes, added_layers, embed_size, freeze_backbone,
                         task_type=task_type, **options)

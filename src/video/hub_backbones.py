"""Modern pretrained backbones through ``transformers``: V-JEPA 2 and VideoMAE (v1, v2) as clip models, SigLIP 2 and
DINOv3 as frame backbones (``src/video/siglip_backbone.py``). Names and licences are in ``handlers/video_backbones.py``.

* :func:`load_encoder` builds a model and loads its weights, refusing partial loads and opt-in weights nobody accepted.
* :func:`encoder_token_grid` gives any of them as ``[B, T', H', W', D]``, the grid ``SnippetBackbone`` caches.
* :class:`HubVideoModel` is the clip classifier ``utils.build_video_model`` returns for V-JEPA 2 and VideoMAE: frozen
  encoder plus B1's attentive probe by default (P-101, P-102), with an optional linear probe or LoRA.

V-JEPA 2: Assran et al., arXiv 2506.09985, MIT. VideoMAE: Tong et al., arXiv 2203.12602; VideoMAE V2: Wang et al.,
arXiv 2303.16727, both CC-BY-NC-4.0. SigLIP 2: Tschannen et al., arXiv 2502.14786, Apache-2.0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from handlers.video_arch import ARCH_OPTION_DEFAULTS, ARCH_OPTIONS, PROBE_ARCHITECTURES, parse_pooling_spec
from handlers.video_backbones import HUB_BACKBONES, hub_backbone, require_weights_choice
from src.video.feature_heads import build_feature_head

_LORA_TARGETS = ['query', 'value']  # both families call these layers, so an adapter gets a gradient


def _tiny_model(version: str) -> nn.Module:
    info = HUB_BACKBONES[version]
    size = info.input_size
    if info.family == 'vjepa2':
        from transformers import VJEPA2Config, VJEPA2Model
        return VJEPA2Model(VJEPA2Config(patch_size=8, crop_size=size, frames_per_clip=4, hidden_size=24, num_attention_heads=2,
                                        num_hidden_layers=1, pred_hidden_size=12, pred_num_attention_heads=2,
                                        pred_num_hidden_layers=1, pred_num_mask_tokens=1))
    if info.family == 'videomae':
        from transformers import VideoMAEConfig, VideoMAEModel
        return VideoMAEModel(VideoMAEConfig(image_size=size, patch_size=8, num_frames=4, hidden_size=16, num_hidden_layers=1,
                                            num_attention_heads=2, intermediate_size=32))
    if info.family == 'dinov3':
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        return DINOv3ViTModel(DINOv3ViTConfig(image_size=size, patch_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                              num_attention_heads=2, num_register_tokens=2))
    from transformers import SiglipVisionConfig, SiglipVisionModel
    return SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                                image_size=size, patch_size=8))


def checked_from_pretrained(model_cls, hub_id: str) -> nn.Module:
    """``from_pretrained`` that refuses a partial load. Transformers only warns about missing weights and leaves them
    at their random start, which would give plausible-looking features from an untrained layer."""
    model, loading = model_cls.from_pretrained(hub_id, dtype=torch.float32, output_loading_info=True)
    if loading['missing_keys'] or loading.get('mismatched_keys'):
        missing = sorted(loading['missing_keys'])
        raise ValueError(f'{hub_id} does not fit {model_cls.__name__}: {len(missing)} weights are missing '
                         f'(for example {missing[:3]}) and {len(loading.get("mismatched_keys") or ())} have another shape.')
    return model


def _read_raw_weights(hub_id: str) -> dict:
    """The weights file as it is on the Hub. No repository code runs, unlike AutoModel with trust_remote_code."""
    from pathlib import Path
    if Path(hub_id).is_dir():
        files = [p for p in (Path(hub_id) / 'model.safetensors', Path(hub_id) / 'pytorch_model.bin') if p.is_file()]
        if not files:
            raise FileNotFoundError(f'No model.safetensors or pytorch_model.bin in {hub_id}.')
        path = str(files[0])
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            path = hf_hub_download(hub_id, 'model.safetensors')
        except EntryNotFoundError:
            path = hf_hub_download(hub_id, 'pytorch_model.bin')
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location='cpu', weights_only=True)


_V2_RENAMES = (('attn.proj.', 'attention.output.dense.'), ('mlp.fc1.', 'intermediate.dense.'), ('mlp.fc2.', 'output.dense.'), ('norm1.', 'layernorm_before.'),
               ('norm2.', 'layernorm_after.'))
_NOT_ENCODER = ('decoder', 'encoder_to_decoder', 'mask_token', 'head', 'classifier', 'fc_norm', 'pos_embed', 'position_embeddings')


def convert_videomae_state(state: dict) -> dict:
    """Any VideoMAE weights file -> the keys of ``transformers.VideoMAEModel``.

    Three layouts exist. The current one. Older transformers files with ``q_bias`` and ``v_bias`` (5.16.1 expects the
    ``query`` and ``value`` layers to hold their own bias and has no conversion, so ``from_pretrained`` would leave
    those biases at zero with only a warning). And the upstream V2 layout (``model.blocks.N.attn.qkv``), whose ``qkv``
    is one matrix with no bias, split here row-wise as upstream's forward splits its output. The key layer has no bias
    in either old layout, which is what the zero here states."""
    out = {}
    for key, value in state.items():
        name = key
        for prefix in ('videomae.', 'model.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name.startswith(_NOT_ENCODER):
            continue
        name = name.replace('patch_embed.proj.', 'embeddings.patch_embeddings.projection.')
        if name.startswith('blocks.'):
            name = 'encoder.layer.' + name[len('blocks.'):]
            for old, new in _V2_RENAMES:
                name = name.replace(old, new)
        if name.endswith('.attn.qkv.weight'):
            base = name[:-len('attn.qkv.weight')] + 'attention.attention.'
            for part, weights in zip(('query', 'key', 'value'), value.chunk(3, dim=0)):
                out[f'{base}{part}.weight'] = weights
            continue
        for bias, layer in (('q_bias', 'query'), ('v_bias', 'value')):
            if name.endswith(bias):
                base = name[:-len(bias)].replace('.attn.', '.attention.attention.')
                out[f'{base}{layer}.bias'] = value
                out[f'{base}key.bias'] = torch.zeros_like(value)
                name = None
                break
        if name is not None:
            out[name] = value
    return out


def _load_videomae(info) -> nn.Module:
    from transformers import VideoMAEConfig, VideoMAEModel
    if info.version == 'videomae_v2_b16':  # its config names a remote-code class; these are the values in that file
        config = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16, tubelet_size=2, hidden_size=768, num_hidden_layers=12,
                                num_attention_heads=12, intermediate_size=3072, qkv_bias=True, use_mean_pooling=True)
    else:
        config = VideoMAEConfig.from_pretrained(info.hub_id)
    model = VideoMAEModel(config)
    state = convert_videomae_state(_read_raw_weights(info.hub_id))
    model.load_state_dict(state, strict=True)  # strict: a layout this did not foresee stops here, not at random weights
    return model.to(torch.float32)


def load_hub_model(version: str, gradient_checkpointing: bool = False) -> nn.Module:
    """The `version`'s network with its weights, or at random when it is a tiny test model. The V-JEPA 2 predictor,
    which only serves self-supervised training, is dropped. SigLIP's text tower goes too."""
    info = HUB_BACKBONES[version]
    require_weights_choice(version)
    if info.is_untrained:
        model = _tiny_model(version)
    elif info.family == 'vjepa2':
        from transformers import VJEPA2Model
        model = checked_from_pretrained(VJEPA2Model, info.hub_id)
    elif info.family == 'videomae':
        model = _load_videomae(info)
    elif info.family == 'dinov3':
        from transformers import DINOv3ViTModel
        model = checked_from_pretrained(DINOv3ViTModel, info.hub_id)
    elif version == 'siglip':  # the copy the images modality already uses
        from src.vision.siglip import load_siglip_model_offline
        model = load_siglip_model_offline().vision_model
    else:  # the fixed-size SigLIP 2 checkpoints have model_type siglip, not siglip2
        from transformers import SiglipModel
        model = checked_from_pretrained(SiglipModel, info.hub_id).vision_model
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model.encoder if info.family == 'vjepa2' else model


def encoder_token_grid(encoder: nn.Module, version: str, x: torch.Tensor) -> torch.Tensor:
    """``[B, T, C, H, W]`` -> ``[B, T', H', W', D]``. Refuses input sizes that would run but mean something else:
    V-JEPA 2 derives each token's position from its index and the configured grid, so another size would still give
    numbers while every position is wrong."""
    info = HUB_BACKBONES[version]
    batch, frames, _, height, width = x.shape
    config = encoder.config
    if info.fixed_frames is not None and frames != info.fixed_frames:
        raise ValueError(f'{version} only accepts clips of exactly {info.fixed_frames} frames, got {frames}. '
                         f'Use --num_frames {info.fixed_frames}.')
    if info.kind == 'frame':
        patch = config.patch_size
        flat = x.reshape(batch * frames, *x.shape[2:])
        tokens = encoder(pixel_values=flat).last_hidden_state
        if info.family == 'dinov3':
            tokens = tokens[:, 1 + config.num_register_tokens:]  # class token, then registers, then patches
        grid = (frames, height // patch, width // patch)
    else:
        if (height, width) != (info.input_size, info.input_size):
            raise ValueError(f'{version} only accepts {info.input_size}x{info.input_size} input, got {height}x{width}.')
        patch = config.patch_size
        if info.family == 'vjepa2':
            tokens = encoder(pixel_values_videos=x).last_hidden_state
        else:
            tokens = encoder(pixel_values=x).last_hidden_state
        # V-JEPA 2 repeats a clip shorter than one tubelet, and Conv3d drops a last odd frame.
        grid = (max(frames // config.tubelet_size, 1), height // patch, width // patch)
    return tokens.reshape(batch, *grid, tokens.shape[-1])


class HubClipTransform:
    """Each model's own preprocessing for a uint8 clip ``[T, C, H, W]``. With ``resize_short``: resize the short side,
    then centre-crop, like the V-JEPA 2 and VideoMAE video processors; otherwise squash to the input size, like
    DINOv3's. Checked against the transformers processors in tests/models/test_hub_backbones.py."""

    def __init__(self, version: str):
        self.info = HUB_BACKBONES[version]

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        info = self.info
        height, width = clip.shape[-2:]
        size = (info.input_size, info.input_size)
        if info.resize_short is not None:
            short = info.resize_short
            size = (short, int(short * width / height)) if height <= width else (int(short * height / width), short)
        frames = F.interpolate(clip.float() / 255.0, size=size, mode='bilinear', align_corners=False, antialias=True).clamp(0.0, 1.0)
        if info.resize_short is not None:
            top, left = int(round((size[0] - info.input_size) / 2.0)), int(round((size[1] - info.input_size) / 2.0))
            frames = frames[..., top:top + info.input_size, left:left + info.input_size]
        mean = torch.tensor(info.mean, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(info.std, dtype=frames.dtype).view(1, 3, 1, 1)
        return (frames - mean) / std


def probe_from_pooling(pooling: str, version: str) -> tuple[str, dict]:
    """(head name, head options) for the recorded spec. The default ``mean`` means the attentive probe (P-101)."""
    arch, options = parse_pooling_spec(pooling)
    if arch == 'mean':
        arch = 'attentive_probe'
    if arch not in PROBE_ARCHITECTURES:
        raise ValueError(f'{version} is a clip model with its own head: use --video_arch attentive_probe or linear_probe, not {pooling}.')
    values = {name: options.get(name, ARCH_OPTION_DEFAULTS[name]) for name in ARCH_OPTIONS[arch]}
    return arch, {name.removeprefix('probe_').replace('heads', 'num_heads'): value for name, value in values.items()}


class HubVideoModel(nn.Module):
    """A V-JEPA 2 or VideoMAE encoder and a probe. Clips are ``[B, T, C, H, W]``, as everywhere in DeepTune.

    An encoder with nothing to train runs in eval mode under no_grad, whatever ``train()`` is called with, so frozen
    training keeps no graph for it. ``--trainable_params`` may unfreeze part of it later, so this is read at each call.
    With LoRA, ``lora_rank`` is the rank, ``embedding_layer_size`` when None, and gradient checkpointing is on.
    The checkpoint holds the encoder's weights too, as every other DeepTune model's does."""

    def __init__(self, num_classes, version, head='attentive_probe', head_options=None, freeze_backbone=True, task_type='cls',
                 output_dim=1, use_peft=False, lora_rank=None, embedding_layer_size=16):
        super().__init__()
        info = hub_backbone(version)
        if info is None or info.kind != 'video':
            raise ValueError(f'{version} is not a clip model.')
        self.version = version
        self.model = load_hub_model(version, gradient_checkpointing=use_peft)
        width = self.model.config.hidden_size
        if freeze_backbone or use_peft:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        if use_peft:
            from peft import LoraConfig, inject_adapter_in_model
            self.lora_rank = embedding_layer_size if lora_rank is None else lora_rank
            config = LoraConfig(r=self.lora_rank, lora_alpha=16, lora_dropout=0.1, bias='none', target_modules=_LORA_TARGETS)
            self.model = inject_adapter_in_model(config, self.model)
        self.head = build_feature_head(head, width, num_classes, task_type, output_dim=output_dim, **(head_options or {}))
        self.embedding_dim = (head_options or {}).get('width', 256) if head == 'attentive_probe' else width

    def _encoder_is_fixed(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.model.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_is_fixed():
            self.model.eval()
        return self

    def token_grid(self, x: torch.Tensor) -> torch.Tensor:
        if self._encoder_is_fixed():
            with torch.no_grad():
                return encoder_token_grid(self.model, self.version, x)
        return encoder_token_grid(self.model, self.version, x)

    def forward(self, x, extract_embed=False):
        return self.head(self.token_grid(x), extract_embed=extract_embed)


def build_hub_video_model(version, num_classes, freeze_backbone, mode, use_peft, pooling, embed_size, lora_rank, added_layers=None):
    head, options = probe_from_pooling(pooling, version)
    return HubVideoModel(num_classes, version, head, options, freeze_backbone, mode, use_peft=use_peft, lora_rank=lora_rank,
                         embedding_layer_size=embed_size)

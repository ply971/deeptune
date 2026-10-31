"""Modern pretrained video backbones loaded through ``transformers``: names, licences, input rules. Torch-free, so the
GUI and the CLI can read it.

Licences come from the model cards (board F-002, read 2026-10-30, and the model cards named below). Weights with
restrictive terms are opt-in: naming the model is not enough, the user also has to name the licence with
``--accept_weights_licence``, and the loader refuses otherwise (``require_weights_choice``). The refusal sits where the
weights load, so no entry point can skip it. A run that recorded its acceptance rebuilds without being asked again.

Versions ending in ``_tiny`` are small, untrained models for tests and CPU smoke runs. They need no download.

Not registered: V-JEPA 2.1. ``transformers`` 5.16.1 has no loader for it, its upstream route runs repository code
through ``torch.hub``, and its README states no checkpoint licence (P-115, F-015).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class HubBackbone:
    version: str
    family: str  # vjepa2, videomae, siglip, dinov3
    kind: str  # 'video': one token grid per clip; 'frame': runs on each frame
    hub_id: Optional[str]  # None: untrained tiny model
    licence: Optional[str] = None  # the id --accept_weights_licence takes; None: no weights, so no licence
    licence_source: str = ''
    opt_in: bool = False
    gated: str = ''  # how access is restricted, when it is
    input_size: int = 224
    resize_short: Optional[int] = None  # None: squash the whole frame to the input size, with no centre crop
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    fixed_frames: Optional[int] = None  # None: any frame count
    memory_note: str = ''

    @property
    def is_untrained(self) -> bool:
        return self.hub_id is None


_NC = 'CC-BY-NC-4.0'
HUB_BACKBONES = {b.version: b for b in (
    HubBackbone('vjepa2_vitl', 'vjepa2', 'video', 'facebook/vjepa2-vitl-fpc64-256', 'MIT',
                'model card, board F-002', input_size=256, resize_short=292,
                memory_note='ViT-L, 300M parameters, about 1.2 GB of weights in float32. Start with frozen features at '
                            '--batch_size 1 or 2. LoRA on it is the last resort on 8 GB (P-103).'),
    HubBackbone('videomae_b16', 'videomae', 'video', 'MCG-NJU/videomae-base', _NC, 'model card, board F-002',
                opt_in=True, input_size=224, resize_short=224, fixed_frames=16,
                memory_note='ViT-B, about 87M parameters, 16 frames only.'),
    HubBackbone('videomae_v2_b16', 'videomae', 'video', 'OpenGVLab/VideoMAEv2-Base', _NC, 'model card, board F-002',
                opt_in=True, gated='the repository needs remote code, which DeepTune never runs: the weights are read '
                                   'into the transformers VideoMAE model. The distilled weights sit behind a request form',
                input_size=224, resize_short=224, fixed_frames=16,
                memory_note='ViT-B, about 87M parameters, 16 frames only.'),
    HubBackbone('siglip', 'siglip', 'frame', 'google/siglip-so400m-patch14-384', 'Apache-2.0', 'model card, read 2026-10-31',
                input_size=384, mean=HALF, std=HALF,
                memory_note='384 px frames and a large ViT: every frame costs far more than at 224 px.'),
    HubBackbone('siglip2_b16', 'siglip', 'frame', 'google/siglip2-base-patch16-224', 'Apache-2.0', 'model card, board F-002',
                input_size=224, mean=HALF, std=HALF,
                memory_note='ViT-B on each frame: about one image forward per frame.'),
    HubBackbone('dinov3_vitb16', 'dinov3', 'frame', 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'DINOv3-License',
                'model card, board F-002', opt_in=True, gated='Meta must approve your Hugging Face account first',
                input_size=224, memory_note='ViT-B on each frame: about one image forward per frame.'),
    HubBackbone('vjepa2_tiny', 'vjepa2', 'video', None, input_size=32, resize_short=36),
    HubBackbone('videomae_tiny', 'videomae', 'video', None, input_size=32, resize_short=36, fixed_frames=4),
    HubBackbone('siglip_tiny', 'siglip', 'frame', None, input_size=32, mean=HALF, std=HALF),
    HubBackbone('siglip2_tiny', 'siglip', 'frame', None, input_size=32, mean=HALF, std=HALF),
    HubBackbone('dinov3_tiny', 'dinov3', 'frame', None, input_size=32),
)}
HUB_VIDEO_VERSIONS = tuple(v for v, b in HUB_BACKBONES.items() if b.kind == 'video')

_accepted: set = set()


class WeightsLicenceNotAccepted(ValueError):
    """Opt-in weights were requested without naming their licence."""


def hub_backbone(version) -> Optional[HubBackbone]:
    return HUB_BACKBONES.get(str(version))


def is_hub_video_backbone(version) -> bool:
    info = hub_backbone(version)
    return info is not None and info.kind == 'video'


def is_hub_frame_backbone(version) -> bool:
    info = hub_backbone(version)
    return info is not None and info.kind == 'frame'


def accept_weights_licences(licences: Optional[Iterable[str]]) -> None:
    """Record, for this process, the licences the user named."""
    if isinstance(licences, str):
        licences = [licences]
    _accepted.update(str(name).casefold() for name in licences or ())


def reset_accepted_licences() -> None:
    _accepted.clear()


def licence_text(version) -> str:
    """One line for a run record, describing exactly these weights."""
    info = HUB_BACKBONES[str(version)]
    if info.is_untrained:
        return 'none: random initialisation, no pretrained weights'
    use = 'non-commercial only; ' if info.licence == _NC else ''
    gated = f'; access: {info.gated}' if info.gated else ''
    return f'{info.licence} ({use}{info.hub_id}; source: {info.licence_source}){gated}'


def require_weights_choice(version) -> None:
    """Print the licence of `version`'s weights. Opt-in weights also need the licence named."""
    info = hub_backbone(version)
    if info is None or info.is_untrained:
        return
    print(f'Weights of {version}: {licence_text(version)}')
    if info.opt_in and info.licence.casefold() not in _accepted:
        raise WeightsLicenceNotAccepted(
            f'{version} is opt-in: its weights are under {info.licence}'
            + (f', and {info.gated}' if info.gated else '')
            + f'. To use them, read the licence and add --accept_weights_licence {info.licence}.')


def check_frame_count(version, num_frames) -> None:
    info = hub_backbone(version)
    if info is not None and info.fixed_frames is not None and int(num_frames) != info.fixed_frames:
        raise ValueError(f'{version} only accepts clips of exactly {info.fixed_frames} frames, not {num_frames}. '
                         f'Use --num_frames {info.fixed_frames} (--snippet_frames for extract_features.py).')


def add_weights_licence_argument(group) -> None:
    group.add_argument('--accept_weights_licence', nargs='+', default=None, metavar='LICENCE',
                       help='Accept the licence of opt-in weights by name, as the refusal message spells it: '
                            f'{_NC} (VideoMAE, non-commercial), DINOv3-License (gated). Recorded with the run.')

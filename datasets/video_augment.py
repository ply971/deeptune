"""Clip-level augmentation (training only, ``--video_augment``).

Every spatial/colour parameter is drawn **once per clip** and applied to
every frame, so the clip stays temporally consistent (a per-frame random
crop would itself look like motion):

* random-resized crop (scale 0.6-1.0, aspect 3/4-4/3), resized back to the
  clip's own size so the downstream per-frame / Kinetics transforms see the
  same input size as without augmentation;
* mild colour jitter (brightness, contrast, saturation in [0.8, 1.2]);
* horizontal flip with p = 0.5, **off by default** (``--video_flip`` turns it
  on). A mirror image turns "moves left" into "moves right", so flipping
  silently corrupts any label that depends on horizontal direction -- the
  Phase 2 direction benchmark is exactly such a task. Only enable it when
  the classes are mirror-symmetric. When codec motion vectors or flow are
  used with a flipped clip, ``apply_to_motion`` mirrors the field and
  negates dx so motion stays consistent with the pixels.

Temporal jitter and speed perturbation change *which* frames are decoded,
so they live in ``datasets.video_sampling`` (``temporal_jitter``,
``speed_perturb``) and are switched on by the same flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ClipAugmentConfig:
    enabled: bool = False
    flip: bool = False                       # horizontal flip; off by default (see module docstring)
    crop_scale: tuple = (0.6, 1.0)
    crop_ratio: tuple = (3 / 4, 4 / 3)
    color_jitter: float = 0.2                # brightness/contrast/saturation factor range 1 +- x
    speed_perturb: float = 0.0               # handled by the sampler


@dataclass
class ClipAugmentParams:
    crop: tuple                              # (top, left, height, width) in pixels
    size: tuple                              # (H, W) of the clip
    flipped: bool
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    extra: dict = field(default_factory=dict)


class ClipAugment:
    """Apply one random crop/flip/colour draw to a whole uint8 clip ``[T, C, H, W]``."""

    def __init__(self, config: ClipAugmentConfig):
        self.config = config

    def sample_params(self, height: int, width: int, generator: Optional[torch.Generator] = None) -> ClipAugmentParams:
        cfg = self.config
        area = height * width
        crop = (0, 0, height, width)
        for _ in range(10):
            target = area * float(torch.empty(1).uniform_(*cfg.crop_scale, generator=generator))
            log_ratio = torch.log(torch.tensor(cfg.crop_ratio))
            ratio = float(torch.exp(torch.empty(1).uniform_(float(log_ratio[0]), float(log_ratio[1]), generator=generator)))
            w = int(round((target * ratio) ** 0.5))
            h = int(round((target / ratio) ** 0.5))
            if 0 < w <= width and 0 < h <= height:
                top = int(torch.randint(0, height - h + 1, (1,), generator=generator))
                left = int(torch.randint(0, width - w + 1, (1,), generator=generator))
                crop = (top, left, h, w)
                break
        flipped = bool(cfg.flip and float(torch.rand(1, generator=generator)) < 0.5)
        j = cfg.color_jitter

        def factor():
            return float(torch.empty(1).uniform_(1 - j, 1 + j, generator=generator)) if j > 0 else 1.0

        return ClipAugmentParams(crop=crop, size=(height, width), flipped=flipped,
                                 brightness=factor(), contrast=factor(), saturation=factor())

    def __call__(self, clip: torch.Tensor, params: Optional[ClipAugmentParams] = None):
        """Returns ``(augmented clip, params)``; `clip` is uint8 ``[T, C, H, W]``."""
        from torchvision.transforms.v2 import functional as F
        if not self.config.enabled:
            return clip, None
        height, width = clip.shape[-2:]
        params = params or self.sample_params(height, width)
        top, left, h, w = params.crop
        out = F.resized_crop(clip, top, left, h, w, [height, width], antialias=True)
        if params.flipped:
            out = out.flip(-1)
        if params.brightness != 1.0:
            out = F.adjust_brightness(out, params.brightness)
        if params.contrast != 1.0:
            out = F.adjust_contrast(out, params.contrast)
        if params.saturation != 1.0 and out.shape[-3] == 3:
            out = F.adjust_saturation(out, params.saturation)
        return out, params

    @staticmethod
    def apply_to_motion(field: np.ndarray, params: Optional[ClipAugmentParams]) -> np.ndarray:
        """Apply the clip's crop and flip to a motion field ``[T, 2, Hb, Wb]``.

        The crop is mapped to grid cells and resized back to the grid size;
        vectors are scaled by the zoom factor (a crop of half the width
        doubles apparent displacement). A flip mirrors the grid and negates dx.
        """
        from datasets.video_io import flip_motion_field
        if params is None:
            return field
        field = np.asarray(field, dtype=np.float32)
        height, width = params.size
        gh, gw = field.shape[-2:]
        top, left, h, w = params.crop
        if (h, w) != (height, width):
            import torch.nn.functional as tf
            y0, x0 = int(top * gh / height), int(left * gw / width)
            y1 = max(y0 + 1, int(round((top + h) * gh / height)))
            x1 = max(x0 + 1, int(round((left + w) * gw / width)))
            crop = torch.from_numpy(np.ascontiguousarray(field[..., y0:y1, x0:x1]))
            lead = crop.shape[:-3]
            crop = crop.reshape(-1, 2, crop.shape[-2], crop.shape[-1])
            crop = tf.interpolate(crop, size=(gh, gw), mode='bilinear', align_corners=False)
            crop[:, 0] *= width / w
            crop[:, 1] *= height / h
            field = crop.reshape(*lead, 2, gh, gw).numpy()
        return flip_motion_field(field) if params.flipped else field

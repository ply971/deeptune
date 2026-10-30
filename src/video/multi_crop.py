"""Multi-crop test-time averaging (``--test_crops K``), the spatial partner of
``--test_clips`` (src/video/multi_clip.py); together they give K x N views.

The frames have already been resized to the model's input size by this point,
so each view is a window covering 7/8 of each side, resized back to that size.
This differs from the 3-crop protocol of the Kinetics papers, which cut full
crops from a larger frame: here the views overlap more and are slightly
upsampled, so reported accuracy is not directly comparable to those numbers.

Classification averages the softmax probabilities and returns ``log(mean p)``,
as MultiClipModel does, so the consumers (TestTrainer, inference, error model)
see the averaged probabilities. Regression outputs and embeddings are averaged.
Crops run one after another, which keeps peak memory at one batch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

TEST_CROP_CHOICES = (1, 3, 5)
CROP_FRACTION = 0.875


def crop_boxes(height: int, width: int, crops: int, fraction: float = CROP_FRACTION) -> list[tuple[int, int, int, int]]:
    """Windows ``(top, left, h, w)``. 3 crops: start, centre and end of the longer
    side (width if square). 5 crops: the four corners and the centre."""
    if crops not in TEST_CROP_CHOICES:
        raise ValueError(f'test_crops must be one of {TEST_CROP_CHOICES}, not {crops}.')
    h, w = max(1, round(height * fraction)), max(1, round(width * fraction))
    bottom, right = height - h, width - w
    centre = (bottom // 2, right // 2, h, w)
    if crops == 1:
        return [(0, 0, height, width)]
    if crops == 3:
        if width >= height:
            return [(bottom // 2, 0, h, w), centre, (bottom // 2, right, h, w)]
        return [(0, right // 2, h, w), centre, (bottom, right // 2, h, w)]
    return [(0, 0, h, w), (0, right, h, w), (bottom, 0, h, w), (bottom, right, h, w), centre]


def crop_view(x: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    """One view of a batch of clips ``[B, T, C, H, W]``, same shape as the input.
    The window is the same on every frame, so motion inside the clip is kept."""
    top, left, h, w = box
    height, width = x.shape[-2:]
    window = x[..., top:top + h, left:left + w]
    if (h, w) == (height, width):
        return window
    batch, frames = x.shape[:2]
    flat = window.reshape(batch * frames, *window.shape[2:])
    resized = F.interpolate(flat, size=(height, width), mode='bilinear', align_corners=False)
    return resized.reshape(batch, frames, *resized.shape[1:])


class MultiCropModel(nn.Module):
    def __init__(self, model: nn.Module, crops: int, mode: str = 'cls'):
        super().__init__()
        crop_boxes(8, 8, crops)  # checks the count
        self.model = model
        self.crops = crops
        self.mode = mode

    def forward(self, x, extract_embed: bool = False):
        height, width = x.shape[-2:]
        total = None
        for box in crop_boxes(height, width, self.crops):
            view = crop_view(x, box)
            out = self.model(view, extract_embed=True) if extract_embed else self.model(view)
            if not extract_embed and self.mode == 'cls':
                out = torch.softmax(out.float(), dim=-1)
            total = out if total is None else total + out
        averaged = total / self.crops
        if extract_embed or self.mode != 'cls':
            return averaged
        return torch.log(averaged.clamp_min(1e-12))

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)


def wrap_for_test_crops(model: nn.Module, test_crops: int, mode: str) -> nn.Module:
    return MultiCropModel(model, int(test_crops), mode) if int(test_crops) > 1 else model

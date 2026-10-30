"""--test_crops: spatial test-time views (src/video/multi_crop.py)."""
import json

import pytest
import torch
import torch.nn as nn

from src.video.multi_clip import MultiClipModel, error_model_clips_mismatch, wrap_for_test_clips
from src.video.multi_crop import MultiCropModel, crop_boxes, crop_view, wrap_for_test_crops


class Probe(nn.Module):
    """Logits from the input mean, so each view gives a different, known output."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def forward(self, x, extract_embed=False):
        self.seen.append(x.detach().clone())
        pooled = x.mean(dim=(1, 2, 3, 4)) * self.scale
        return pooled[:, None].repeat(1, 3) if extract_embed else torch.stack([pooled, -pooled], dim=1)


def test_crop_boxes_are_inside_the_frame_and_differ():
    for height, width, crops in [(32, 32, 3), (32, 32, 5), (32, 64, 3), (64, 32, 3), (33, 47, 5)]:
        boxes = crop_boxes(height, width, crops)
        assert len(boxes) == len(set(boxes)) == crops
        for top, left, h, w in boxes:
            assert top >= 0 and left >= 0 and top + h <= height and left + w <= width
    assert crop_boxes(32, 64, 3)[0][0] == crop_boxes(32, 64, 3)[2][0]  # moves along the longer side
    assert crop_boxes(64, 32, 3)[0][1] == crop_boxes(64, 32, 3)[2][1]
    assert crop_boxes(32, 32, 1) == [(0, 0, 32, 32)]
    with pytest.raises(ValueError, match='test_crops'):
        crop_boxes(32, 32, 4)


def test_a_view_keeps_the_shape_and_uses_one_window_for_every_frame():
    clip = torch.zeros(2, 4, 3, 16, 16)
    for t in range(4):
        clip[:, t] = t
    clip[:, :, :, :, :8] += 100  # the left half is brighter, on every frame
    left = crop_view(clip, crop_boxes(16, 16, 3)[0])
    right = crop_view(clip, crop_boxes(16, 16, 3)[2])
    assert left.shape == clip.shape
    assert left.mean() > right.mean()
    # Frame differences survive the crop: the clip's motion signal is intact.
    assert torch.allclose(left[:, 1] - left[:, 0], torch.ones_like(left[:, 0]), atol=1e-4)


def test_classification_averages_probabilities_and_returns_their_log():
    torch.manual_seed(0)
    x = torch.rand(2, 2, 3, 16, 16)
    probe = Probe()
    expected = torch.stack([torch.softmax(probe(crop_view(x, box)), dim=1) for box in crop_boxes(16, 16, 3)]).mean(dim=0)
    probe.seen.clear()
    out = MultiCropModel(probe, 3, 'cls')(x)
    assert len(probe.seen) == 3
    assert torch.allclose(torch.softmax(out, dim=1), expected, atol=1e-5)


def test_regression_and_embeddings_average_plainly():
    x = torch.rand(2, 2, 3, 16, 16)
    probe = Probe()
    views = [crop_view(x, box) for box in crop_boxes(16, 16, 5)]
    assert torch.allclose(MultiCropModel(probe, 5, 'reg')(x), torch.stack([probe(v) for v in views]).mean(dim=0))
    assert torch.allclose(MultiCropModel(probe, 5, 'cls')(x, extract_embed=True),
                          torch.stack([probe(v, extract_embed=True) for v in views]).mean(dim=0))


def test_one_crop_is_no_wrapper_and_state_dict_passes_through():
    probe = Probe()
    assert wrap_for_test_crops(probe, 1, 'cls') is probe
    wrapped = wrap_for_test_crops(probe, 3, 'cls')
    assert set(wrapped.state_dict()) == set(probe.state_dict())
    wrapped.load_state_dict({'scale': torch.tensor(2.0)})
    assert probe.scale.item() == 2.0


def test_crops_multiply_with_clips():
    torch.manual_seed(1)
    probe = Probe()
    model = wrap_for_test_clips(probe, 2, 'cls', test_crops=3)
    assert isinstance(model, MultiClipModel) and isinstance(model.model, MultiCropModel)
    x = torch.rand(4, 2, 2, 3, 16, 16)  # [B, N, T, C, H, W]
    out = model(x)
    assert out.shape == (4, 2)
    assert len(probe.seen) == 3 and probe.seen[0].shape[0] == 8  # B * N clips, 3 views each
    assert torch.softmax(out, dim=1).sum(dim=1).allclose(torch.ones(4), atol=1e-5)
    assert isinstance(wrap_for_test_clips(probe, 1, 'cls', test_crops=3), MultiCropModel)
    assert wrap_for_test_clips(probe, 1, 'cls') is probe


def test_error_model_warns_on_another_crop_count(tmp_path):
    weights = tmp_path / 'model_weights.pth'
    weights.write_bytes(b'')
    (tmp_path / 'error_model.json').write_text(json.dumps({'fit_info': {'test_clips': 2, 'test_crops': 3}}))
    messages = []
    assert error_model_clips_mismatch(weights, 2, warn=messages.append, test_crops=1) is True
    assert '--test_crops 3' in messages[0]
    assert error_model_clips_mismatch(weights, 2, warn=messages.append, test_crops=3) is False

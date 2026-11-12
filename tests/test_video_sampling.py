"""Temporal sampling (datasets/video_sampling.py) and clip augmentation (datasets/video_augment.py)."""
import numpy as np
import pytest
import torch

from datasets.video_augment import ClipAugment, ClipAugmentConfig
from datasets.video_sampling import STRATEGIES, SamplingSpec, sample_clip_indices


def _sample(strategy, frames=100, n=8, train=False, seed=0, **kwargs):
    motion = kwargs.pop('motion', np.ones(frames) if strategy == 'motion' else None)
    spec = SamplingSpec(strategy=strategy, num_frames=n, **{k: v for k, v in kwargs.items() if k in
                        ('clip_stride', 'clip_duration_sec', 'temporal_jitter', 'speed_perturb')})
    rest = {k: v for k, v in kwargs.items() if k in ('num_clips', 'fps', 'start_frame', 'end_frame')}
    return sample_clip_indices(frames, spec, train=train, rng=np.random.default_rng(seed), motion=motion, **rest)


@pytest.mark.parametrize('strategy', STRATEGIES)
@pytest.mark.parametrize('train', [False, True])
def test_shapes_and_bounds(strategy, train):
    idx = _sample(strategy, train=train, num_clips=3, fps=25)
    assert idx.shape == (3, 8) and idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < 100
    assert np.all(np.diff(idx, axis=1) >= 0)


def test_uniform_eval_matches_the_original_linspace():
    for frames, n in [(10, 4), (97, 8), (5, 8), (1, 3)]:
        expected = np.linspace(0, frames - 1, n).astype(int)
        np.testing.assert_array_equal(_sample('uniform', frames, n)[0], expected)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_training_sampling_is_deterministic_with_a_seed(strategy):
    kw = dict(train=True, temporal_jitter=True, speed_perturb=0.2, fps=25)
    np.testing.assert_array_equal(_sample(strategy, seed=7, **kw), _sample(strategy, seed=7, **kw))


def test_training_sampling_varies_between_draws():
    draws = {tuple(_sample('segment', train=True, seed=s)[0]) for s in range(10)}
    assert len(draws) > 5


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_eval_sampling_is_deterministic(strategy):
    a = _sample(strategy, num_clips=2, fps=25)
    b = _sample(strategy, num_clips=2, fps=25, seed=99)
    np.testing.assert_array_equal(a, b)


def test_segment_sampling_covers_the_clip():
    frames, n = 64, 8
    for seed in range(20):
        idx = _sample('segment', frames, n, train=True, seed=seed)[0]
        segments = np.floor(idx * n / frames).astype(int)
        np.testing.assert_array_equal(segments, np.arange(n))  # one frame from every segment
    centre = _sample('segment', frames, n)[0]
    np.testing.assert_array_equal(centre, np.arange(n) * 8 + 4)


def test_dense_uses_the_stride_and_pads_short_clips():
    idx = _sample('dense', 100, 8, clip_stride=2)[0]
    assert np.all(np.diff(idx) == 2) and idx[0] == (100 - 15) // 2
    short = _sample('dense', 5, 8, clip_stride=1)[0]
    np.testing.assert_array_equal(short, [0, 1, 2, 3, 4, 4, 4, 4])  # last frame repeated


def test_dense_multi_clip_windows_are_spread_over_the_video():
    idx = _sample('dense', 100, 4, clip_stride=2, num_clips=3)
    assert idx[0, 0] == 0 and idx[-1, -1] == 99 and idx[0, 0] < idx[1, 0] < idx[2, 0]


def test_motion_sampling_puts_more_frames_where_the_motion_is():
    curve = np.r_[np.zeros(50), np.ones(50)]  # static first half
    for train in (False, True):
        idx = _sample('motion', 100, 16, train=train, motion=curve)[0]
        assert (idx >= 50).sum() >= 14, idx


def test_motion_sampling_with_flat_motion_is_close_to_uniform():
    idx = _sample('motion', 100, 10, motion=np.ones(100))[0]
    np.testing.assert_allclose(idx, np.arange(10) * 10 + 5, atol=1)


def test_motion_strategy_requires_a_curve():
    with pytest.raises(ValueError, match='motion curve'):
        sample_clip_indices(10, SamplingSpec('motion'), motion=None)


def test_clip_duration_limits_the_window():
    idx = _sample('uniform', 250, 8, clip_duration_sec=2.0, fps=25)[0]  # 50 frames
    assert idx[-1] - idx[0] <= 49 and idx[0] == 100
    train = _sample('uniform', 250, 8, clip_duration_sec=2.0, fps=25, train=True, seed=3)[0]
    assert train[-1] - train[0] <= 49
    with pytest.raises(ValueError, match='frame rate'):
        _sample('uniform', 250, 8, clip_duration_sec=2.0, fps=0)


def test_start_and_end_frames_restrict_sampling():
    for strategy in STRATEGIES:
        idx = _sample(strategy, 100, 6, start_frame=40, end_frame=60, fps=25)
        assert idx.min() >= 40 and idx.max() < 60


def test_multi_clip_uniform_clips_differ():
    idx = _sample('uniform', 100, 4, num_clips=3)
    assert len({tuple(row) for row in idx}) == 3


def test_temporal_jitter_only_shortens_the_span():
    spans = [np.ptp(_sample('uniform', 100, 8, train=True, temporal_jitter=True, seed=s)[0]) for s in range(30)]
    assert min(spans) >= 0.8 * 99 - 2 and max(spans) <= 99 and len(set(spans)) > 1


@pytest.mark.parametrize('bad', [dict(strategy='random'), dict(num_frames=0), dict(clip_stride=0),
                                 dict(clip_duration_sec=0), dict(speed_perturb=1.0)])
def test_invalid_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        SamplingSpec(**bad)


# ---------------------------------------------------------------------------
# Clip augmentation
# ---------------------------------------------------------------------------
def _ramp_clip(t=6, h=24, w=32):
    x = torch.arange(w, dtype=torch.uint8).view(1, 1, 1, w).expand(t, 3, h, w) * 7
    return x.clone()


def test_flip_is_off_by_default():
    assert ClipAugmentConfig().flip is False
    augment = ClipAugment(ClipAugmentConfig(enabled=True, crop_scale=(1.0, 1.0), crop_ratio=(4 / 3, 4 / 3),
                                            color_jitter=0.0))
    clip = _ramp_clip()
    for _ in range(20):
        out, params = augment(clip)
        assert not params.flipped
        assert out[0, 0, 0, 0] < out[0, 0, 0, -1]  # left stays darker than right


def test_same_crop_and_flip_on_every_frame():
    torch.manual_seed(0)
    augment = ClipAugment(ClipAugmentConfig(enabled=True, flip=True, color_jitter=0.2))
    clip = _ramp_clip(t=1).expand(5, 3, 24, 32).clone()  # identical frames
    flips = set()
    for _ in range(30):
        out, params = augment(clip)
        flips.add(params.flipped)
        assert out.shape == clip.shape and out.dtype == torch.uint8
        assert all(torch.equal(out[0], out[t]) for t in range(1, 5))  # one transform for the whole clip
    assert flips == {True, False}


def test_disabled_augment_returns_the_clip_unchanged():
    clip = _ramp_clip()
    out, params = ClipAugment(ClipAugmentConfig())(clip)
    assert out is clip and params is None


def test_flip_negates_motion_dx_for_the_same_clip():
    augment = ClipAugment(ClipAugmentConfig(enabled=True, flip=True, crop_scale=(1.0, 1.0),
                                            crop_ratio=(4 / 3, 4 / 3), color_jitter=0.0))
    clip = _ramp_clip()
    field = np.zeros((6, 2, 3, 4), np.float32)
    field[:, 0] = 2.0
    field[:, 1] = 1.0
    torch.manual_seed(1)
    while True:
        _, params = augment(clip)
        if params.flipped:
            break
    moved = ClipAugment.apply_to_motion(field, params)
    assert np.allclose(moved[:, 0], -2.0) and np.allclose(moved[:, 1], 1.0)


def test_crop_scales_motion_vectors_by_the_zoom():
    from datasets.video_augment import ClipAugmentParams
    field = np.ones((1, 2, 4, 4), np.float32)
    params = ClipAugmentParams(crop=(0, 0, 16, 16), size=(32, 32), flipped=False)
    moved = ClipAugment.apply_to_motion(field, params)
    assert moved.shape == field.shape and np.allclose(moved, 2.0)

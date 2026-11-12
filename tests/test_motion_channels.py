"""Motion fields for the two-stream model: sign and size against a known shift, the disk cache, rotation, normalisation,
and how the dataset attaches them to clips."""
import numpy as np
import pandas as pd
import pytest
import torch

from datasets.motion_stack import (MOTION_FULL_SCALE, clip_motion_field, load_motion_field, motion_cache_path,
                                   normalise_motion_field, rotate_motion_field)
from datasets.video_datasets import VideoDataOptions, open_video_dataset, video_options_from_args
from datasets.video_io import VideoReader
from datasets.video_synth import translating_clip

pytest.importorskip('av')

SIZE = 64


@pytest.mark.parametrize('dx,dy', [(3, 0), (-3, 0), (0, 2)])
@pytest.mark.parametrize('source', ['farneback', 'mv'])
def test_mean_motion_is_the_known_shift(tmp_path, source, dx, dy):
    path = translating_clip(tmp_path / 'clip.mp4', 12, dx, dy, size=SIZE, seed=1)
    with VideoReader(path) as reader:
        field = load_motion_field(reader, np.arange(1, 12), source)
    assert field.shape[1:] == (2, SIZE // 4, SIZE // 4)
    centre = field[..., 4:-4, 4:-4]  # new content enters at the edges
    mean = np.median(centre, axis=(0, 2, 3))
    assert abs(mean[0] - dx) <= 0.75 and abs(mean[1] - dy) <= 0.75
    if dx:
        assert np.sign(mean[0]) == np.sign(dx)


def test_cache_hit_returns_identical_arrays_without_computing(tmp_path, monkeypatch):
    path = translating_clip(tmp_path / 'clip.mp4', 8, 2, 0, size=SIZE)
    cache = tmp_path / 'cache'
    with VideoReader(path) as reader:
        first = load_motion_field(reader, [1, 2, 3], 'farneback', cache_dir=cache)
        fingerprint = reader.fingerprint()

        def fail(*args, **kwargs):
            raise AssertionError('recomputed instead of using the cache')
        monkeypatch.setattr(VideoReader, 'get_motion_vectors', fail)
        second = load_motion_field(reader, [1, 2, 3], 'farneback', cache_dir=cache)
        assert np.array_equal(first, second) and first.dtype == second.dtype
        with pytest.raises(AssertionError, match='recomputed'):  # other parameters are another entry
            load_motion_field(reader, [1, 2, 4], 'farneback', cache_dir=cache)
        with pytest.raises(AssertionError, match='recomputed'):
            load_motion_field(reader, [1, 2, 3], 'mv', cache_dir=cache)
    assert motion_cache_path(cache, fingerprint, [1, 2, 3], 'farneback', 4).is_file()


def test_damaged_cache_entry_is_recomputed(tmp_path):
    path = translating_clip(tmp_path / 'clip.mp4', 8, 2, 0, size=SIZE)
    cache = tmp_path / 'cache'
    with VideoReader(path) as reader:
        good = load_motion_field(reader, [1, 2], 'farneback', cache_dir=cache)
        entry = motion_cache_path(cache, reader.fingerprint(), [1, 2], 'farneback', 4)
        entry.write_bytes(b'')
        assert np.array_equal(load_motion_field(reader, [1, 2], 'farneback', cache_dir=cache), good)


def test_rotation_turns_the_grid_and_the_vectors():
    """A pattern moving right, turned 90 degrees counter-clockwise, moves up (dy < 0)."""
    field = np.zeros((1, 2, 2, 3), dtype=np.float32)
    field[0, 0] = 1.0
    turned = rotate_motion_field(field, 90)
    assert turned.shape == (1, 2, 3, 2)
    assert np.allclose(turned[0, 0], 0) and np.allclose(turned[0, 1], -1)
    assert np.allclose(rotate_motion_field(rotate_motion_field(field, 90), 270), field)
    assert rotate_motion_field(field, 0) is field


def test_normalisation_is_independent_of_resolution_and_saturates():
    moved = 0.1  # of the frame's width, per frame
    small = np.full((1, 2, 2, 2), moved * 32, dtype=np.float32)
    large = np.full((1, 2, 2, 2), moved * 320, dtype=np.float32)
    assert np.allclose(normalise_motion_field(small, 32, 32), normalise_motion_field(large, 320, 320))
    assert np.allclose(normalise_motion_field(small, 32, 32), moved / MOTION_FULL_SCALE)
    assert np.allclose(clip_motion_field(np.array([-3.0, 0.5, 3.0])), [-1.0, 0.5, 1.0])


def _dataset(tmp_path, dx=2, **options):
    path = translating_clip(tmp_path / 'clip.mp4', 12, dx, 0, size=SIZE, seed=2)
    frame = pd.DataFrame({'videos': [str(path)], 'labels': [1]})
    return path, frame, options


@pytest.mark.parametrize('source', ['farneback', 'mv'])
def test_clips_carry_motion_after_rgb_at_the_frame_size(tmp_path, source):
    _, frame, _ = _dataset(tmp_path)
    clip, _ = open_video_dataset(frame, num_frames=4, options=VideoDataOptions(clip_motion=source))[0]
    assert clip.shape == (4, 5, SIZE, SIZE)
    moving = clip[1:, 3][..., 16:-16, 16:-16]  # dx of the frames after the first, away from the edges
    assert moving.median() > 0.3  # +2 px/frame at 64 px is 2 / 3.2 = 0.625 before block averaging
    assert clip[:, 4].abs().median() < 0.2


def test_bytes_and_path_datasets_attach_the_same_motion(tmp_path):
    path, frame, _ = _dataset(tmp_path)
    options = VideoDataOptions(clip_motion='farneback')
    from_path = open_video_dataset(frame, num_frames=4, options=options)[0][0]
    from_bytes = open_video_dataset(frame.assign(videos=[path.read_bytes()]), num_frames=4, options=options)[0][0]
    assert torch.allclose(from_path[:, 3:], from_bytes[:, 3:], atol=1e-5)


def test_multiple_clips_keep_motion_per_clip(tmp_path):
    _, frame, _ = _dataset(tmp_path)
    clips = open_video_dataset(frame, num_frames=4, options=VideoDataOptions(clip_motion='farneback', temporal_sampling='segment'),
                                num_clips=2)[0][0]
    assert clips.shape == (2, 4, 5, SIZE, SIZE)


def test_a_flipped_training_clip_flips_and_negates_its_motion(tmp_path, monkeypatch):
    """The augmented motion must describe the augmented pixels, or flipping would corrupt the label."""
    import datasets.video_augment as augment
    _, frame, _ = _dataset(tmp_path)
    options = VideoDataOptions(clip_motion='farneback', video_augment=True, video_flip=True)

    def draw(flipped):
        def params(self, height, width, generator=None):
            return augment.ClipAugmentParams(crop=(0, 0, height, width), size=(height, width), flipped=flipped)
        monkeypatch.setattr(augment.ClipAugment, 'sample_params', params)
        torch.manual_seed(0)  # the same temporal jitter for both
        return open_video_dataset(frame, num_frames=4, train=True, options=options)[0][0]
    plain, flipped = draw(False), draw(True)
    assert torch.allclose(flipped[:, :3], plain[:, :3].flip(-1), atol=1e-4)
    assert torch.allclose(flipped[:, 3], -plain[:, 3].flip(-1), atol=1e-4)
    assert torch.allclose(flipped[:, 4], plain[:, 4].flip(-1), atol=1e-4)


def test_options_follow_the_architecture():
    args = {'pooling': 'two_stream:num_frames=8', 'motion_source': 'mv'}
    options = video_options_from_args(args)
    assert options.clip_motion == 'mv' and options.motion_source == 'codec'
    assert video_options_from_args({'pooling': 'two_stream', 'motion_source': 'auto'}).clip_motion == 'farneback'
    assert video_options_from_args({'pooling': 'two_stream', 'motion_source': 'codec'}).clip_motion == 'mv'
    plain = video_options_from_args({'pooling': 'mean', 'motion_source': 'farneback'})
    assert plain.clip_motion is None and plain.motion_source == 'framediff'
    with pytest.raises(ValueError, match='two_stream needs'):
        video_options_from_args({'pooling': 'two_stream', 'motion_source': 'framediff'})
    assert video_options_from_args({}).clip_motion is None
    assert video_options_from_args().clip_motion is None


def test_a_crop_scales_the_motion_by_its_zoom(tmp_path, monkeypatch):
    """A crop of half the width doubles the apparent motion, in the same units as the uncropped clip."""
    import datasets.video_augment as augment
    path = translating_clip(tmp_path / 'clip.mp4', 12, 1, 0, size=SIZE, seed=2)
    frame = pd.DataFrame({'videos': [str(path)], 'labels': [1]})

    def draw(crop):
        def params(self, height, width, generator=None):
            return augment.ClipAugmentParams(crop=crop, size=(height, width), flipped=False)
        monkeypatch.setattr(augment.ClipAugment, 'sample_params', params)
        torch.manual_seed(0)
        return open_video_dataset(frame, num_frames=4, train=True, options=VideoDataOptions(clip_motion='farneback', video_augment=True))[0][0]
    whole = draw((0, 0, SIZE, SIZE))[1:, 3][..., 16:-16, 16:-16].median()
    half = draw((0, 16, SIZE, SIZE // 2))[1:, 3][..., 16:-16, 16:-16].median()
    assert whole > 0.1
    assert abs(half / whole - 2) < 0.4


def test_a_file_without_motion_vectors_fails_clearly_instead_of_giving_zero_motion(tmp_path, monkeypatch):
    from datasets.video_io import MotionVectorsUnavailable
    _, frame, _ = _dataset(tmp_path)

    def unavailable(*args, **kwargs):
        raise MotionVectorsUnavailable('no motion vectors in this file')
    monkeypatch.setattr(VideoReader, 'get_motion_vectors', unavailable)
    with pytest.raises(ValueError, match='no motion vectors'):
        open_video_dataset(frame, num_frames=4, options=VideoDataOptions(clip_motion='mv'))[0]


def test_the_cache_keeps_two_videos_apart(tmp_path):
    """Same frames asked for, same parameters, one cache: each video must get its own motion."""
    left = translating_clip(tmp_path / 'left.mp4', 8, -2, 0, size=SIZE, seed=1)
    right = translating_clip(tmp_path / 'right.mp4', 8, 2, 0, size=SIZE, seed=1)
    cache = tmp_path / 'cache'
    fields = []
    for path in (left, right):
        with VideoReader(path) as reader:
            fields.append(load_motion_field(reader, [1, 2, 3], 'farneback', cache_dir=cache))
    assert np.median(fields[0][:, 0]) < -0.5 and np.median(fields[1][:, 0]) > 0.5
    assert len(list(cache.rglob('*.npy'))) == 2


def _non_square_clip(path, dx, width=96, height=48, frames=12):
    from datasets.video_synth import write_clip
    rng = np.random.default_rng(0)
    world = np.kron(rng.integers(0, 255, (height // 4, 2 * width // 4, 3), dtype=np.uint8), np.ones((4, 4, 1), dtype=np.uint8))
    return write_clip(path, [np.roll(world, dx * i, axis=1)[:height, :width] for i in range(frames)], codec='mpeg4',
                      options={'qscale': '2'})


def test_a_non_square_clip_scales_dx_by_its_width(tmp_path):
    """+2 px/frame at 96 px wide is 2 / (96 * 0.05) = 0.417; with width and height swapped it would be 0.833."""
    frame = pd.DataFrame({'videos': [str(_non_square_clip(tmp_path / 'wide.mp4', 2))], 'labels': [1]})
    clip = open_video_dataset(frame, num_frames=4, options=VideoDataOptions(clip_motion='farneback'))[0][0]
    assert clip.shape == (4, 5, 48, 96)
    assert abs(clip[1:, 3][..., 12:-12, 24:-24].median().item() - 2 / (96 * MOTION_FULL_SCALE)) < 0.15


def test_normalising_divides_each_axis_by_its_own_size():
    field = np.ones((1, 2, 2, 2), dtype=np.float32)
    out = normalise_motion_field(field, height=32, width=64)
    assert np.allclose(out[0, 0], 1 / (64 * MOTION_FULL_SCALE)) and np.allclose(out[0, 1], 1 / (32 * MOTION_FULL_SCALE))

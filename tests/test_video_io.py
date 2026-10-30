"""Decoder abstraction (datasets/video_io.py): exact frames, metadata, errors,
codec motion vectors and the Farneback fallback. Tiny generated clips only."""
import numpy as np
import pytest

av = pytest.importorskip('av')

from datasets import video_io
from datasets.video_io import (MotionVectorsUnavailable, VideoDecodeError, VideoReader, available_backends,
                               flip_motion_field, reencode_for_motion_vectors)
from datasets.video_synth import decode_frame_index, index_coded_clip, translating_clip, write_clip

BACKENDS = [name for name in ('torchcodec', 'pyav', 'opencv')]


def _backend(name):
    if name not in available_backends():
        pytest.skip(f'{name} is not installed')
    return name


@pytest.fixture(scope='module')
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp('clips')
    return {
        # libx264 defaults use B-frames, so decode order != presentation order.
        'h264': index_coded_clip(root / 'h264.mp4', 40),
        'mpeg4': index_coded_clip(root / 'mpeg4.mp4', 30, codec='mpeg4'),
        'long': index_coded_clip(root / 'long.mp4', 380, width=16, height=16, options={'g': '48'}),
    }


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('name', ['h264', 'mpeg4'])
def test_get_frames_returns_the_exact_requested_frames(clips, backend, name):
    reader = VideoReader(clips[name], backend=_backend(backend))
    count = reader.metadata().frame_count
    wanted = [0, 7, 3, 3, count - 1, 1, count // 2]
    frames = reader.get_frames(wanted)
    assert frames.dtype == np.uint8 and frames.shape == (len(wanted), 32, 32, 3)
    assert [decode_frame_index(f) for f in frames] == wanted


@pytest.mark.parametrize('backend', BACKENDS)
def test_sparse_reads_in_a_long_video_are_exact(clips, backend):
    # PyAV takes the keyframe-seek + timestamp path here (> SEQUENTIAL_READ_LIMIT).
    reader = VideoReader(clips['long'], backend=_backend(backend))
    wanted = [379, 5, 301, 150, 302, 377, 200]
    assert [decode_frame_index(f) for f in reader.get_frames(wanted)] == wanted


def test_sparse_reads_with_a_gop_longer_than_the_gap_between_targets(tmp_path):
    # Regression: keyframes every 250 frames, targets ~60 apart. A gap-based
    # re-seek rule looped forever here (each seek landed on the same keyframe).
    import time
    path = index_coded_clip(tmp_path / 'longgop.mp4', 390, width=16, height=16, options={'g': '250', 'keyint_min': '250'})
    reader = VideoReader(path, backend='pyav')
    wanted = [389, 320, 260, 200, 140, 80, 20]
    start = time.perf_counter()
    assert [decode_frame_index(f) for f in reader.get_frames(wanted)] == wanted
    assert time.perf_counter() - start < 20
    keys = reader._impl._keys.tolist()
    assert keys[0] == 0 and len(keys) >= 2 and 389 - keys[-1] > 96  # a target far past the last keyframe


def test_metadata_reports_fps_count_duration_size_codec(clips):
    meta = VideoReader(clips['h264'], backend='pyav').metadata()
    assert (meta.fps, meta.frame_count, meta.width, meta.height, meta.codec) == (25.0, 40, 32, 32, 'h264')
    assert meta.duration == pytest.approx(1.6, abs=1e-6)
    assert meta.frame_count_exact and not meta.variable_fps and meta.rotation == 0
    assert VideoReader(clips['h264'], backend='opencv').metadata().frame_count == 40


def test_bytes_input_decodes_in_memory_without_a_temp_file(clips, monkeypatch):
    data = clips['h264'].read_bytes()

    def no_temp_files(*args, **kwargs):
        raise AssertionError('PyAV must decode bytes without a temporary file')

    monkeypatch.setattr(video_io.tempfile, 'NamedTemporaryFile', no_temp_files)
    frames = VideoReader(data, backend='pyav').get_frames([2, 39])
    assert [decode_frame_index(f) for f in frames] == [2, 39]


def test_opencv_bytes_input_uses_and_removes_a_temp_file(clips):
    reader = VideoReader(clips['h264'].read_bytes(), backend='opencv')
    temp = reader._impl._temp
    assert temp and video_io.os.path.exists(temp)
    assert decode_frame_index(reader.get_frames([4])[0]) == 4
    reader.close()
    assert not video_io.os.path.exists(temp)


def test_out_of_range_indices_raise_unless_clamped(clips):
    reader = VideoReader(clips['mpeg4'], backend='pyav')
    with pytest.raises(IndexError, match='out of range'):
        reader.get_frames([0, 30])
    frames = reader.get_frames([28, 30, 31], clamp_to_end=True)
    assert [decode_frame_index(f) for f in frames] == [28, 29, 29]


@pytest.mark.parametrize('payload', [b'', b'not a video at all'])
def test_unreadable_or_empty_bytes_give_clear_errors(payload):
    with pytest.raises(VideoDecodeError):
        VideoReader(payload, backend='pyav').get_frames([0])


def test_missing_and_zero_byte_files_give_clear_errors(tmp_path):
    with pytest.raises(VideoDecodeError, match='not found'):
        VideoReader(tmp_path / 'missing.mp4')
    empty = tmp_path / 'empty.mp4'
    empty.write_bytes(b'')
    with pytest.raises(VideoDecodeError, match='empty'):
        VideoReader(empty)


def test_corrupt_file_reports_the_path(tmp_path):
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'\x00' * 2048)
    with pytest.raises(VideoDecodeError, match='bad.mp4'):
        VideoReader(bad, backend='pyav').metadata()


def test_portrait_video_keeps_its_shape(tmp_path):
    path = index_coded_clip(tmp_path / 'portrait.mp4', 6, width=32, height=48)
    for backend in ('pyav', 'opencv'):
        frames = VideoReader(path, backend=backend).get_frames([0, 5])
        assert frames.shape == (2, 48, 32, 3)


def test_display_rotation_is_applied_like_opencv(tmp_path):
    frames = []
    for _ in range(4):
        frame = np.zeros((32, 64, 3), np.uint8)
        frame[:, :16] = 255  # bright stripe on the left of the stored frame
        frames.append(frame)
    path = write_clip(tmp_path / 'rotated.mp4', frames, rotation=90)
    reader = VideoReader(path, backend='pyav')
    meta = reader.metadata()
    assert meta.rotation == 90 and (meta.width, meta.height) == (32, 64)
    ours = reader.get_frames([0])[0]
    theirs = VideoReader(path, backend='opencv').get_frames([0])[0]
    assert ours.shape == theirs.shape == (64, 32, 3)
    assert np.abs(ours.astype(int) - theirs.astype(int)).mean() < 6


def test_frame_times_and_time_to_index(clips):
    reader = VideoReader(clips['h264'], backend='pyav')
    times = reader.frame_times()
    assert len(times) == 40 and times[0] == 0 and np.allclose(np.diff(times), 0.04)
    assert reader.time_to_index(0.4) == 10
    assert reader.time_to_end_index(0.4) == 10
    assert reader.time_to_end_index(100) == 40


def test_unknown_backend_and_uninstalled_backend_are_explained(clips):
    with pytest.raises(ValueError, match='Unknown video backend'):
        VideoReader(clips['h264'], backend='gstreamer')
    if 'torchcodec' not in available_backends():
        with pytest.raises(RuntimeError, match='pip install torchcodec'):
            VideoReader(clips['h264'], backend='torchcodec')


# ---------------------------------------------------------------------------
# Motion vectors
# ---------------------------------------------------------------------------
def _moving_region_median(field):
    inner = field[:, :, 1:-1, 1:-1]  # skip border blocks (content enters/leaves there)
    return float(np.median(inner[:, 0])), float(np.median(inner[:, 1]))


@pytest.mark.parametrize('codec,options', [('libx264', {'refs': '1', 'bf': '0'}), ('mpeg4', {'bf': '0'})])
@pytest.mark.parametrize('dx,dy', [(3, -2), (-2, 1)])
def test_codec_motion_vectors_match_a_known_translation(tmp_path, codec, options, dx, dy):
    path = translating_clip(tmp_path / f'{codec}.mp4', 12, dx, dy, codec=codec, options=options)
    motion = VideoReader(path).get_motion_vectors(range(1, 12), block_size=8, method='codec')
    assert motion.source == 'codec' and motion.field.shape == (11, 2, 8, 8)
    assert not motion.is_intra.any()
    mx, my = _moving_region_median(motion.field)
    assert mx == pytest.approx(dx, abs=0.25) and my == pytest.approx(dy, abs=0.25)


def test_b_frames_keep_the_sign_convention(tmp_path):
    path = translating_clip(tmp_path / 'bframes.mp4', 16, 2, 0, options={'bf': '2', 'refs': '1', 'b-pyramid': 'none'})
    motion = VideoReader(path).get_motion_vectors(range(1, 16), method='codec')
    medians = [float(np.median(f[0, 1:-1, 1:-1])) for f in motion.field]
    assert all(m > 0 for m in medians), medians
    assert np.median(medians) == pytest.approx(2, abs=0.5)


def test_static_clip_has_near_zero_motion(tmp_path):
    path = translating_clip(tmp_path / 'static.mp4', 10, 0, 0)
    motion = VideoReader(path).get_motion_vectors(range(10))
    assert motion.is_intra[0]
    assert np.abs(motion.field).max() < 0.5


def test_first_frame_is_marked_intra_with_zero_field(tmp_path):
    path = translating_clip(tmp_path / 'moving.mp4', 8, 2, 0, options={'refs': '1', 'bf': '0'})
    motion = VideoReader(path).get_motion_vectors([0, 1])
    assert motion.is_intra.tolist() == [True, False]
    assert not motion.field[0].any()


def test_all_intra_codec_falls_back_to_farneback(tmp_path):
    path = translating_clip(tmp_path / 'intra.mkv', 8, 3, 0, codec='ffv1', pix_fmt='yuv420p')
    reader = VideoReader(path)
    with pytest.raises(MotionVectorsUnavailable):
        reader.get_motion_vectors([1, 2], method='codec')
    motion = reader.get_motion_vectors([1, 2, 3], method='auto')
    assert motion.source == 'farneback'
    mx, my = _moving_region_median(motion.field)
    assert mx == pytest.approx(3, abs=0.3) and abs(my) < 0.3


def test_farneback_and_codec_agree(tmp_path):
    path = translating_clip(tmp_path / 'agree.mp4', 6, -2, 2, codec='mpeg4')
    reader = VideoReader(path)
    codec = _moving_region_median(reader.get_motion_vectors([2, 3, 4], method='codec').field)
    flow = _moving_region_median(reader.get_motion_vectors([2, 3, 4], method='farneback').field)
    assert codec == pytest.approx(flow, abs=0.3)


def test_flip_negates_dx_and_mirrors_the_grid():
    field = np.zeros((1, 2, 2, 3), np.float32)
    field[0, 0, :, 0] = 5.0   # dx on the left column
    field[0, 1, :, 0] = -1.0  # dy on the left column
    flipped = flip_motion_field(field)
    assert np.all(flipped[0, 0, :, 2] == -5.0) and np.all(flipped[0, 1, :, 2] == -1.0)
    assert not flipped[0, :, :, 0].any()
    np.testing.assert_array_equal(flip_motion_field(flipped), field)


def test_reencode_gives_exact_vectors_for_multi_reference_h264(tmp_path):
    # x264's default (3 reference frames) makes some vectors span 2-3 frames.
    path = translating_clip(tmp_path / 'multiref.mp4', 14, 2, 0)
    clean = reencode_for_motion_vectors(path, tmp_path / 'clean.mp4')
    reader = VideoReader(clean)
    assert reader.metadata().codec == 'mpeg4' and reader.metadata().frame_count == 14
    motion = reader.get_motion_vectors(range(1, 14))
    assert motion.is_intra.sum() <= 1  # the GOP-12 keyframe at index 12
    medians = [float(np.median(f[0, 1:-1, 1:-1])) for f, intra in zip(motion.field, motion.is_intra) if not intra]
    assert medians == pytest.approx([2.0] * len(medians), abs=0.25) and len(medians) >= 12


def test_motion_curve_is_low_while_static_and_high_while_moving(tmp_path):
    path = translating_clip(tmp_path / 'half.mp4', 20, 3, 0, static_frames=10, options={'refs': '1', 'bf': '0'})
    reader = VideoReader(path)
    for method in ('codec', 'framediff'):
        curve = reader.motion_curve(method)
        assert len(curve) == 20 and curve[0] == 0
        assert curve[2:10].mean() < 0.2 * curve[12:].mean(), (method, curve)


def test_iter_frames_streams_every_frame_downscaled(clips):
    frames = list(VideoReader(clips['mpeg4']).iter_frames(width=16, gray=True))
    assert len(frames) == 30 and frames[0].shape == (16, 16)

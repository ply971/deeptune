"""Camera-motion compensation: the estimate itself, and what it does for the tracker."""
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from spatial.cmc import IDENTITY, SparseFlowCMC, compose
from spatial.detect import Detection
from spatial.track import ByteTracker


def _texture(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 4, w // 4, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_first_frame_and_textureless_frames_give_identity():
    cmc = SparseFlowCMC()
    blank = np.zeros((60, 80, 3), np.uint8)
    assert np.allclose(cmc(blank), IDENTITY)
    assert np.allclose(cmc(blank), IDENTITY)  # no corners -> identity, no crash
    cmc = SparseFlowCMC()
    assert np.allclose(cmc(_texture()), IDENTITY)  # nothing to compare with yet


@pytest.mark.parametrize('dx,dy', [(6, 0), (-4, 3), (0, -7)])
def test_recovers_a_known_camera_translation(dx, dy):
    base = _texture()
    cmc = SparseFlowCMC()
    cmc(base)
    a = cmc(_shift(base, dx, dy))
    assert a[:, :2] == pytest.approx(np.eye(2), abs=0.02)
    assert a[:, 2] == pytest.approx([dx, dy], abs=0.5)
    assert cmc.last_inliers >= 10


def test_translation_is_in_full_resolution_pixels_when_downscaled():
    base = cv2.resize(_texture(), (640, 480), interpolation=cv2.INTER_NEAREST)
    cmc = SparseFlowCMC(max_side=160)  # works on a 4x smaller image
    cmc(base)
    a = cmc(_shift(base, 12, -8))
    assert a[:, 2] == pytest.approx([12, -8], abs=2.0)


def test_detection_boxes_are_masked_so_the_background_wins():
    # Background moves +5 px; a large textured object moves -10 px. Masking the
    # object's box makes the estimate follow the background.
    bg = _texture(seed=1)
    obj = _texture(40, 50, seed=2)
    def scene(bg_dx, obj_x):
        img = _shift(bg, bg_dx, 0)
        img[40:80, obj_x:obj_x + 50] = obj
        return img
    cmc = SparseFlowCMC()
    cmc(scene(0, 60), boxes=[(60, 40, 110, 80)])
    a = cmc(scene(5, 50), boxes=[(50, 40, 100, 80)])
    assert a[0, 2] == pytest.approx(5, abs=0.75)


def test_compose_chains_affines():
    a = np.array([[1, 0, 2], [0, 1, 3.0]])
    b = np.array([[2, 0, 0], [0, 2, 0.0]])
    c = compose([a, b])  # first a, then b
    p = np.array([1.0, 1.0, 1.0])
    assert c @ p == pytest.approx(b[:, :2] @ (a @ p) + b[:, 2])


def test_cmc_keeps_one_id_through_sudden_camera_pans():
    """A still object filmed by a camera that jerks 15 px per frame from frame 10 on.

    In the image the 20 px box jumps 15 px at once (IoU 0.14): ByteTrack's
    constant-velocity prediction misses and starts a new ID. With CMC the
    predicted box moves with the camera, so the ID survives."""
    bg = cv2.resize(_texture(seed=3), (320, 240), interpolation=cv2.INTER_NEAREST)
    pans = [0] * 10 + [15] * 4 + [0] * 6 + [-15] * 4 + [0] * 6
    def run(compensate):
        tr = ByteTracker(high_thresh=0.5, motion_compensator=SparseFlowCMC() if compensate else None)
        cam, ids = 0, set()
        for p in pans:
            cam += p
            frame = _shift(bg, cam, 0)
            x = 140 + cam
            box = (float(x), 100.0, float(x + 20), 120.0)
            frame[100:120, x:x + 20] = 255
            for t in tr.update([Detection(box, 0.9, 'thing')], frame=frame):
                ids.add(t.track_id)
        return ids
    assert len(run(False)) > 1
    assert len(run(True)) == 1

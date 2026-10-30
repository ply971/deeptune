import numpy as np
import pytest

from spatial.track import KalmanFilter, xyah_to_xyxy, xyxy_to_xyah


def test_box_conversions_round_trip():
    box = np.array([10.0, 20.0, 40.0, 80.0])
    xyah = xyxy_to_xyah(box)
    assert xyah.tolist() == pytest.approx([25.0, 50.0, 0.5, 60.0])
    assert xyah_to_xyxy(xyah).tolist() == pytest.approx(box.tolist())


def test_initiate_starts_at_measurement_with_zero_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate([5.0, 6.0, 1.0, 20.0])
    assert mean.tolist() == pytest.approx([5, 6, 1, 20, 0, 0, 0, 0])
    assert cov.shape == (8, 8) and np.allclose(cov, cov.T) and np.all(np.diag(cov) > 0)


def test_predict_moves_by_velocity_and_grows_uncertainty():
    kf = KalmanFilter()
    mean, cov = kf.initiate([0.0, 0.0, 1.0, 10.0])
    mean[4:6] = [2.0, -1.0]
    new_mean, new_cov = kf.predict(mean, cov)
    assert new_mean[:4].tolist() == pytest.approx([2.0, -1.0, 1.0, 10.0])
    assert np.all(np.diag(new_cov) > np.diag(cov))


def test_constant_velocity_track_converges_to_true_velocity():
    kf = KalmanFilter()
    vx, vy = 3.0, -1.5
    def meas(t):
        return np.array([10 + vx * t, 50 + vy * t, 0.8, 20.0])
    mean, cov = kf.initiate(meas(0))
    for t in range(1, 25):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, meas(t))
    assert mean[4] == pytest.approx(vx, abs=0.05)
    assert mean[5] == pytest.approx(vy, abs=0.05)
    pred, _ = kf.predict(mean, cov)
    assert pred[:2].tolist() == pytest.approx(meas(25)[:2].tolist(), abs=0.1)


def test_update_with_predicted_measurement_keeps_mean_and_shrinks_covariance():
    kf = KalmanFilter()
    mean, cov = kf.initiate([30.0, 30.0, 1.0, 12.0])
    mean, cov = kf.predict(mean, cov)
    new_mean, new_cov = kf.update(mean, cov, mean[:4])
    assert new_mean.tolist() == pytest.approx(mean.tolist())
    assert np.all(np.diag(new_cov)[:4] < np.diag(cov)[:4])
    assert np.all(np.linalg.eigvalsh(new_cov) > 0)


def test_update_pulls_state_towards_measurement():
    kf = KalmanFilter()
    mean, cov = kf.initiate([0.0, 0.0, 1.0, 10.0])
    mean, cov = kf.predict(mean, cov)
    new_mean, _ = kf.update(mean, cov, [4.0, 0.0, 1.0, 10.0])
    assert 0.0 < new_mean[0] < 4.0
    assert new_mean[4] > 0  # positive x velocity inferred

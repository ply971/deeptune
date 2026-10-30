"""Is the benchmark data honest? (Phase 2 test oracle.)

Simple probes (benchmarks/probes.py) trained with scikit-learn on held-out
splits show *what information the clips carry*:
- direction: any single frame (and the frame average) is at chance, the mean
  horizontal Farneback flow solves it; so do codec motion vectors (PyAV, 48x48);
- speed / periodic: single frame useless, flow summary works;
- scene_bias: a background-only probe (blob masked out) is good on the biased
  distribution and at chance on the unbiased test split, while motion stays
  reliable on both.

Thresholds were fixed before looking at results and carry margin: chance-level
checks use <= 0.60 accuracy for two balanced classes on >= 200 held-out clips
(chance = 0.50, binomial s.d. ~ 0.035). Seeds are fixed, sizes tiny (32x32;
48x48 for the codec motion-vector probe).
"""
import pytest

from benchmarks.probes import (
    codec_mv_probe,
    direction_probes,
    periodic_probes,
    scene_bias_probes,
    speed_probes,
)
from benchmarks.synthetic_motion import (
    make_direction_dataset,
    make_periodic_dataset,
    make_scene_bias_dataset,
    make_speed_dataset,
)

CODEC = "mp4v"
CHANCE_MAX = 0.60


@pytest.fixture(scope="module")
def direction(tmp_path_factory):
    ds = make_direction_dataset(tmp_path_factory.mktemp("direction"), n_train=300, n_test=200, num_frames=12,
                                size=32, seed=0, codec=CODEC)
    return direction_probes(ds, codec_mv=False)


def test_direction_flow_probe_solves_the_task(direction):
    assert direction["flow"]["test"] >= 0.95, direction["flow"]


@pytest.mark.parametrize("which", ["first", "middle", "last", "average"])
def test_direction_single_frame_and_frame_average_are_at_chance(direction, which):
    scores = direction[f"frame_{which}"]
    assert scores["test"] <= CHANCE_MAX, f"{which}-frame appearance probe {scores}: a single frame leaks the class"
    # The probe is not simply broken: it can fit (memorise) the training scenes.
    assert scores["train"] > 0.9, scores


def test_direction_codec_motion_vector_probe(tmp_path_factory):
    """Mean horizontal codec motion vector per frame -> logistic regression.

    Run at 48x48, not 32x32: MPEG-4 predicts 16x16 macroblocks, and a 32x32
    frame has only four of them. There the encoder codes most of the small blob
    as intra/skip blocks (about 37% of frames carry a non-zero vector) and the
    same probe reaches only about 0.81 - a property of the codec, recorded in
    the Phase 2 report, not of the labels.
    """
    pytest.importorskip("av", reason="Codec motion vectors need PyAV (installed in Phase 1); OpenCV cannot export them.")
    ds = make_direction_dataset(tmp_path_factory.mktemp("direction48"), n_train=300, n_test=200, num_frames=12,
                                size=48, seed=0, codec=CODEC)
    scores = codec_mv_probe(ds)
    if scores is None:
        pytest.skip(f"The {ds.codec!r} encoder produced streams without motion vectors.")
    assert scores["test"] >= 0.95, scores


def test_speed_flow_predicts_speed_single_frame_does_not(tmp_path_factory):
    ds = make_speed_dataset(tmp_path_factory.mktemp("speed"), n_train=200, n_test=200, num_frames=12, size=32,
                            seed=0, codec=CODEC)
    scores = speed_probes(ds)
    assert scores["flow_r2"] >= 0.7, scores
    assert scores["frame_middle_r2"] <= 0.2, f"a single frame leaks the speed: {scores}"


def test_periodic_flow_separates_oscillation_from_drift(tmp_path_factory):
    ds = make_periodic_dataset(tmp_path_factory.mktemp("periodic"), n_train=200, n_test=200, num_frames=16,
                               size=32, seed=0, codec=CODEC)
    scores = periodic_probes(ds)
    assert scores["flow"]["test"] >= 0.9, scores
    assert scores["frame_middle"]["test"] <= CHANCE_MAX, scores


@pytest.fixture(scope="module")
def scene_bias(tmp_path_factory):
    ds = make_scene_bias_dataset(tmp_path_factory.mktemp("scene_bias"), n_train=300, n_val=200, n_test=200,
                                 num_frames=12, size=32, seed=0, bias=0.9, codec=CODEC)
    return scene_bias_probes(ds)


def test_scene_bias_background_only_probe_is_fooled(scene_bias):
    background = scene_bias["background_class"]
    assert background["val"] >= 0.80, f"background-only probe weak on the biased split: {background}"
    assert background["test"] <= CHANCE_MAX, f"class leaks through the background on the unbiased split: {background}"
    # Sanity: the same features do identify the family on the unbiased split.
    assert scene_bias["background_family"]["test"] >= 0.9, scene_bias["background_family"]


def test_scene_bias_motion_stays_reliable(scene_bias):
    flow = scene_bias["flow_class"]
    assert flow["val"] >= 0.95 and flow["test"] >= 0.95, flow


def test_scene_bias_split_statistics(scene_bias):
    rate = scene_bias["family_match_rate"]
    assert 0.8 <= rate["train"] <= 0.97 and 0.8 <= rate["val"] <= 0.97, rate
    assert rate["test"] == 0.5, rate

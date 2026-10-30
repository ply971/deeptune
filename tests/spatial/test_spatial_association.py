import numpy as np
import pytest

from spatial.detect import Detection, FakeDetector, clean_prompts, finalize_detections
from spatial.track import ByteTracker, Track, apply_affine_to_tracks, iou_matrix, linear_assignment, KalmanFilter


def test_iou_on_hand_made_boxes():
    a = [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [2, 2, 4, 4]]
    m = iou_matrix(a, b)
    assert m.shape == (3, 4)
    assert m[0].tolist() == pytest.approx([1.0, 50 / 150, 0.0, 4 / 100])
    assert iou_matrix([], b).shape == (0, 4)
    assert iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0.0  # degenerate, no division by zero


def test_hungarian_finds_global_optimum_not_greedy():
    # Greedy would take (0,0)=0.1 then be forced into (1,1)=0.9 (total 1.0).
    cost = np.array([[0.1, 0.2], [0.15, 0.9]])
    matches, ur, uc = linear_assignment(cost, thresh=0.95)
    assert sorted(map(tuple, matches.tolist())) == [(0, 1), (1, 0)]
    assert ur == [] and uc == []


def test_assignment_gate_leaves_expensive_pairs_unmatched():
    cost = np.array([[0.1, 0.99], [0.99, 0.95], [0.3, 0.2]])
    matches, ur, uc = linear_assignment(cost, thresh=0.5)
    assert sorted(map(tuple, matches.tolist())) == [(0, 0), (2, 1)]
    assert ur == [1] and uc == []


def test_assignment_never_matches_infeasible_pairs():
    cost = np.array([[np.inf, 0.2], [0.1, np.inf]])
    matches, _, _ = linear_assignment(cost, thresh=0.8)
    assert sorted(map(tuple, matches.tolist())) == [(0, 1), (1, 0)]
    matches, ur, uc = linear_assignment(np.array([[np.inf]]), thresh=0.8)
    assert len(matches) == 0 and ur == [0] and uc == [0]


def test_assignment_handles_empty_inputs():
    m, ur, uc = linear_assignment(np.zeros((0, 3)), 0.5)
    assert len(m) == 0 and ur == [] and uc == [0, 1, 2]
    m, ur, uc = linear_assignment(np.zeros((2, 0)), 0.5)
    assert len(m) == 0 and ur == [0, 1] and uc == []


def test_finalize_clips_drops_degenerate_and_runs_per_label_nms():
    boxes = [[-5, -5, 20, 20], [0, 0, 19, 19], [0, 0, 20, 20], [30, 30, 30.5, 40], [50, 50, 80, 80]]
    scores = [0.9, 0.8, 0.7, 0.95, 0.6]
    labels = ['lobster', 'lobster', 'fish', 'lobster', 'fish']
    dets = finalize_detections(boxes, scores, labels, width=64, height=64, nms_iou=0.5)
    # degenerate (0.5 px wide) dropped; overlapping lobster suppressed; fish at same place kept.
    assert [(d.label, round(d.score, 2)) for d in dets] == [('lobster', 0.9), ('fish', 0.7), ('fish', 0.6)]
    assert dets[0].box == (0.0, 0.0, 20.0, 20.0)
    assert dets[2].box == (50.0, 50.0, 64.0, 64.0)
    no_nms = finalize_detections(boxes, scores, labels, 64, 64, nms_iou=None)
    assert len(no_nms) == 4


def test_clean_prompts():
    assert clean_prompts([' lobster ', 'Lobster', '', 'fish']) == ['lobster', 'fish']
    with pytest.raises(ValueError):
        clean_prompts(['a. b'])
    with pytest.raises(ValueError):
        clean_prompts(['  '])


def test_fake_detector_lookup_counter_and_prompt_filter():
    table = {0: [Detection((1, 1, 5, 5), 0.9, 'lobster'), Detection((1, 1, 5, 5), 0.9, 'crab')],
             2: [Detection((2, 2, 6, 6), 0.8, 'Lobster')]}
    det = FakeDetector(table, batch_size=2)
    frames = np.zeros((3, 8, 8, 3), np.uint8)
    out = det.detect(frames, ['lobster'])
    assert [len(o) for o in out] == [1, 0, 1]
    assert out[2][0].label == 'lobster'  # normalised to the prompt's spelling
    assert det.calls == [[0, 1], [2]]
    out = det.detect(frames[:1], ['lobster'])  # counter continues at frame 3
    assert det.calls[-1] == [3] and out == [[]]
    det.detect(frames[:2], ['lobster'], frame_indices=[10, 20])
    assert det.calls[-1] == [10, 20]


def test_apply_affine_translation_moves_track_centres_only():
    t = Track((10, 10, 20, 30), 0.9, 'x')
    t.activate(KalmanFilter(), 1, 1)
    t.mean[4:6] = [1.0, 2.0]
    apply_affine_to_tracks([t], [[1, 0, 5], [0, 1, -3]])
    assert t.mean[:2].tolist() == pytest.approx([20.0, 17.0])
    assert t.mean[4:6].tolist() == pytest.approx([1.0, 2.0])
    assert t.mean[3] == pytest.approx(20.0)


def test_motion_compensator_hook_is_called_with_frame():
    seen = []
    def cmc(frame):
        seen.append(frame.shape)
        return np.array([[1.0, 0, 0], [0, 1.0, 0]])
    tr = ByteTracker(high_thresh=0.5, motion_compensator=cmc)
    frame = np.zeros((8, 8, 3), np.uint8)
    tr.update([Detection((1, 1, 5, 5), 0.9, 'a')], frame=frame)
    tr.update([Detection((1, 1, 5, 5), 0.9, 'a')], frame=frame)
    assert seen == [(8, 8, 3), (8, 8, 3)]

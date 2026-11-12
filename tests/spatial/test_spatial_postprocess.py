"""Offline post-processing: linking fragments, interpolation, short-track removal."""
import numpy as np
import pytest

from spatial.postprocess import (LinkSettings, PostSettings, Tracklet, apply_mapping, drop_short_tracks,
                                 interpolate, link_cost, link_tracklets, postprocess, renumber,
                                 tracklets_from_rows)

FPS = 10.0


def rows_for(tid, frames, x0=0.0, vx=1.0, y=0.0, size=10.0, label='cat', feat=None, score=0.9):
    out = []
    for f in frames:
        x = x0 + vx * f
        r = {'frame': f, 'track_id': tid, 'box': (x, y, x + size, y + size), 'score': score, 'label': label}
        if feat is not None:
            r['feat'] = np.asarray(feat, dtype=np.float32)
        out.append(r)
    return out


def ids(rows):
    return sorted({r['track_id'] for r in rows})


def test_two_fragments_of_one_moving_object_are_linked():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(40, 60))  # 2 s gap, same straight motion
    m = link_tracklets(tracklets_from_rows(rows), FPS)
    assert m == {1: 1, 2: 1}


def test_far_jump_is_not_linked():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(25, 45), x0=200)
    assert link_tracklets(tracklets_from_rows(rows), FPS) == {1: 1, 2: 2}


def test_gap_above_the_limit_is_not_linked():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(60, 80))  # 4 s
    assert link_tracklets(tracklets_from_rows(rows), FPS, LinkSettings(max_gap_s=3)) == {1: 1, 2: 2}
    assert link_tracklets(tracklets_from_rows(rows), FPS, LinkSettings(max_gap_s=5))[2] == 1


def test_overlapping_tracks_are_never_linked():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(15, 30), y=1)
    assert link_tracklets(tracklets_from_rows(rows), FPS) == {1: 1, 2: 2}


def test_labels_must_match_unless_class_agnostic():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(25, 45), label='dog')
    assert link_tracklets(tracklets_from_rows(rows), FPS)[2] == 2
    assert link_tracklets(tracklets_from_rows(rows), FPS, LinkSettings(class_agnostic=True))[2] == 1


def test_appearance_vetoes_a_lookalike_motion():
    a, b = [1, 0, 0], [0, 1, 0]
    same = rows_for(1, range(0, 20), feat=a) + rows_for(2, range(25, 45), feat=a)
    other = rows_for(1, range(0, 20), feat=a) + rows_for(2, range(25, 45), feat=b)
    assert link_tracklets(tracklets_from_rows(same), FPS)[2] == 1
    assert link_tracklets(tracklets_from_rows(other), FPS)[2] == 2


def test_very_different_sizes_are_not_linked():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(25, 45), size=40)
    assert link_tracklets(tracklets_from_rows(rows), FPS)[2] == 2


def test_chains_of_three_fragments_join_into_the_first_id():
    rows = rows_for(5, range(0, 10)) + rows_for(9, range(20, 30)) + rows_for(7, range(40, 50))
    m = link_tracklets(tracklets_from_rows(rows), FPS)
    assert m == {5: 5, 7: 5, 9: 5}


def test_ambiguous_continuations_are_not_linked_by_default():
    # One fragment ends; two candidates start after it (two objects). Which one
    # continues it is a guess, so by default neither is joined.
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(25, 45)) + rows_for(3, range(25, 45), y=8)
    assert link_tracklets(tracklets_from_rows(rows), FPS) == {1: 1, 2: 2, 3: 3}


def test_ambiguous_predecessors_are_not_linked_by_default():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(0, 20), y=8) + rows_for(3, range(25, 45), y=4)
    assert link_tracklets(tracklets_from_rows(rows), FPS) == {1: 1, 2: 2, 3: 3}


def test_permissive_linking_takes_the_best_of_several_candidates():
    rows = rows_for(1, range(0, 20)) + rows_for(2, range(25, 45)) + rows_for(3, range(25, 45), y=8)
    m = link_tracklets(tracklets_from_rows(rows), FPS, LinkSettings(unambiguous=False))
    assert m[2] == 1 and m[3] == 3


def test_link_cost_is_bounded_and_prefers_shorter_gaps():
    t = tracklets_from_rows(rows_for(1, range(0, 20)) + rows_for(2, range(22, 30)) + rows_for(3, range(35, 45)))
    near, far = link_cost(t[1], t[2], FPS, LinkSettings()), link_cost(t[1], t[3], FPS, LinkSettings())
    assert 0 <= near < far < 1
    assert link_cost(t[2], t[1], FPS, LinkSettings()) == np.inf  # backwards in time


def test_merged_chain_takes_the_majority_label():
    rows = rows_for(1, range(0, 30), label='cat') + rows_for(2, range(40, 45), label='kitten')
    out = apply_mapping(rows, {1: 1, 2: 1})
    assert {r['label'] for r in out} == {'cat'} and ids(out) == [1]


def test_interpolation_fills_holes_on_the_sampled_grid():
    rows = rows_for(1, [0, 3, 12]) + rows_for(2, [0, 30])
    out = interpolate(rows, FPS, every_n_frames=3, max_gap_s=1.0)
    filled = sorted(r['frame'] for r in out if r.get('interpolated') and r['track_id'] == 1)
    assert filled == [6, 9]                               # 3 -> 12 is 0.9 s: filled on the 3-frame grid
    assert not [r for r in out if r.get('interpolated') and r['track_id'] == 2]  # 3 s: too long
    r6 = [r for r in out if r['frame'] == 6 and r['track_id'] == 1][0]
    assert r6['box'] == pytest.approx((6, 0, 16, 10)) and 'feat' not in r6
    assert interpolate(rows, FPS, 3, 0) == rows


def test_short_tracks_are_dropped_counting_real_observations_only():
    rows = rows_for(1, range(10)) + rows_for(2, [0, 1]) + \
        [{'frame': f, 'track_id': 2, 'box': (0, 0, 1, 1), 'score': 0.5, 'label': 'cat', 'interpolated': True}
         for f in range(2, 9)]
    assert ids(drop_short_tracks(rows, 3)) == [1]
    assert ids(drop_short_tracks(rows, 1)) == [1, 2]


def test_renumber_orders_by_first_appearance():
    rows = rows_for(7, [5, 6]) + rows_for(3, [8, 9]) + rows_for(9, [0, 1])
    out = renumber(rows)
    assert {r['frame']: r['track_id'] for r in out}[0] == 1
    assert {r['frame']: r['track_id'] for r in out}[5] == 2
    assert {r['frame']: r['track_id'] for r in out}[8] == 3


def test_postprocess_end_to_end_stats():
    # The 2-frame ghost (id 6) is dropped first, so it cannot make the link look ambiguous.
    rows = rows_for(4, range(0, 20)) + rows_for(8, range(29, 49)) + rows_for(6, [22, 23], x0=25)  # 1.0 s gap
    out, stats = postprocess(rows, FPS, 1, PostSettings())
    assert ids(out) == [1]
    assert stats == {'raw_tracks': 3, 'links': 1, 'dropped_short_tracks': 1, 'interpolated_boxes': 9,
                     'final_tracks': 1}
    frames = [r['frame'] for r in out]
    assert frames == sorted(frames) and frames == list(range(49))


def test_tracklet_end_feature_ignores_missing_vectors():
    t = Tracklet(1, frames=[0, 1, 2], boxes=[(0, 0, 1, 1)] * 3, scores=[1] * 3, labels=['a'] * 3,
                 feats=[None, np.array([3.0, 4.0]), None])
    assert t.end_feature(10, True) == pytest.approx([0.6, 0.8])
    assert Tracklet(2, frames=[0], boxes=[(0, 0, 1, 1)], scores=[1], labels=['a'], feats=[None]).end_feature(5, False) is None

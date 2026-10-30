"""ByteTrack behaviour on synthetic moving squares with a scripted noisy detector."""
from spatial.detect import Detection
from spatial.track import ByteTracker


def run(table, n_frames, **kw):
    tracker = ByteTracker(**{'high_thresh': 0.5, 'low_thresh': 0.1, **kw})
    return {f: tracker.update(table.get(f, [])) for f in range(n_frames)}, tracker


def switches(seq):
    """Number of times the covering ID changes between two covered frames."""
    ids = [i for i in seq if i is not None]
    return sum(1 for a, b in zip(ids, ids[1:]) if a != b)


def test_ids_stable_through_jitter_drops_and_low_scores(two_squares, track_ids_of):
    scene = two_squares
    table = scene.detections(jitter=1.0, drop={0: {10, 11, 12, 13}}, low={1: {15, 16, 17, 18}})
    outs, _ = run(table, scene.n_frames)
    seq = track_ids_of(scene, outs)
    # Object 0: tracked, lost during the 4 dropped frames, re-found with the same ID.
    assert all(seq[0][f] is None for f in (10, 11, 12, 13))
    assert all(seq[0][f] is not None for f in range(scene.n_frames) if f not in (10, 11, 12, 13))
    # Object 1: kept through the low-score frames by the second association.
    assert all(i is not None for i in seq[1])
    assert switches(seq[0]) == 0 and switches(seq[1]) == 0
    ids = {i for s in seq.values() for i in s if i is not None}
    assert ids == {1, 2}  # stable integer IDs, one per object
    all_ids = {o.track_id for frame in outs.values() for o in frame}
    assert all_ids == {1, 2}  # no spurious extra tracks -> count is right


def test_second_association_is_what_keeps_low_score_objects(two_squares, track_ids_of):
    scene = two_squares
    table = scene.detections(jitter=1.0, low={1: {15, 16, 17, 18}})
    # Low threshold above the low scores: the second association never sees them.
    outs, _ = run(table, scene.n_frames, low_thresh=0.3)
    seq = track_ids_of(scene, outs)
    assert all(seq[1][f] is None for f in (15, 16, 17, 18))
    assert switches(seq[1]) == 0  # still re-found from the lost pool afterwards
    # Low-score detections alone never start a track.
    only_low = {f: [Detection(d.box, 0.2, d.label) for d in dets] for f, dets in table.items()}
    outs, _ = run(only_low, scene.n_frames)
    assert all(len(o) == 0 for o in outs.values())


def test_crossing_paths_keep_ids(make_scene, track_ids_of):
    # Same label, same row, opposite directions: the boxes fully overlap mid-way.
    scene = make_scene([
        {'label': 'lobster', 'start': (2, 26), 'vel': (2.0, 0.0), 'size': 10, 'color': (0, 0, 255)},
        {'label': 'lobster', 'start': (52, 27), 'vel': (-2.0, 0.0), 'size': 10, 'color': (0, 255, 0)},
    ], n_frames=26)
    outs, _ = run(scene.detections(jitter=0.5, seed=3), scene.n_frames)
    seq = track_ids_of(scene, outs, iou_min=0.3)
    first = [i for i in seq[0][:8]]
    last = [i for i in seq[0][-8:]]
    assert len(set(first)) == 1 and first[0] is not None
    assert set(last) == set(first), 'object 0 changed ID after crossing'
    assert seq[1][0] is not None and set(seq[1][-8:]) == {seq[1][0]}
    assert seq[0][0] != seq[1][0]


def test_crossing_with_occlusion_recovers_the_hidden_object_id(make_scene, track_ids_of):
    # The back object is not detected for 4 frames while the two overlap; the
    # Kalman filter coasts it through and it re-emerges with its own ID.
    scene = make_scene([
        {'label': 'lobster', 'start': (2, 26), 'vel': (2.0, 0.0), 'size': 10, 'color': (0, 0, 255)},
        {'label': 'lobster', 'start': (52, 27), 'vel': (-2.0, 0.0), 'size': 10, 'color': (0, 255, 0)},
    ], n_frames=27)
    outs, _ = run(scene.detections(jitter=0.5, drop={1: {11, 12, 13, 14}}, seed=3), scene.n_frames)
    seq = track_ids_of(scene, outs, iou_min=0.3)
    assert set(seq[0]) == {1}
    assert set(seq[1][:10]) == {2} and set(seq[1][16:]) == {2}
    assert {o.track_id for f in outs for o in outs[f]} == {1, 2}


def test_track_buffer_expiry_gives_new_id(two_squares, track_ids_of):
    scene = two_squares
    gap = set(range(8, 20))  # 12 frames missing
    table = scene.detections(jitter=0.5, drop={0: gap})
    outs, _ = run(table, scene.n_frames, track_buffer=5)  # max_time_lost = 5 steps at 30 fps
    seq = track_ids_of(scene, outs)
    before = {i for i in seq[0][:8]}
    after = {i for i in seq[0][21:]}
    assert len(before) == 1 and len(after) == 1 and before != after
    # With a long enough buffer the same ID comes back (Kalman coasts along the line).
    outs, _ = run(table, scene.n_frames, track_buffer=30)
    seq = track_ids_of(scene, outs, iou_min=0.3)
    assert switches(seq[0]) == 0


def test_labels_gate_association_by_default():
    box = (10.0, 10.0, 20.0, 20.0)
    table = {0: [Detection(box, 0.9, 'lobster')], 1: [Detection(box, 0.9, 'lobster')],
             2: [Detection(box, 0.9, 'fish')], 3: [Detection(box, 0.9, 'fish')]}
    outs, _ = run(table, 4)
    assert [(o.track_id, o.label) for o in outs[1]] == [(1, 'lobster')]
    assert [(o.track_id, o.label) for o in outs[3]] == [(2, 'fish')]


def test_class_agnostic_uses_majority_label():
    box = (10.0, 10.0, 20.0, 20.0)
    labels = ['lobster', 'lobster', 'fish', 'lobster', 'fish', 'fish', 'fish']
    table = {f: [Detection(box, 0.9, lab)] for f, lab in enumerate(labels)}
    outs, _ = run(table, len(labels), class_agnostic=True)
    assert {o.track_id for f in outs for o in outs[f]} == {1}
    assert outs[3][0].label == 'lobster'   # 3 lobster vs 1 fish so far
    assert outs[6][0].label == 'fish'      # 4 fish vs 3 lobster


def test_single_frame_false_positive_is_never_reported():
    table = {f: [Detection((5, 5, 15, 15), 0.9, 'lobster')] for f in range(10)}
    table[4] = table[4] + [Detection((40, 40, 50, 50), 0.95, 'lobster')]  # one-off blip
    outs, tracker = run(table, 10)
    assert {o.track_id for f in outs for o in outs[f]} == {1}
    assert any(t.track_id == 2 for t in tracker.removed)


def test_first_frame_tracks_are_confirmed_immediately_later_ones_after_two_hits():
    a, b = (5.0, 5.0, 15.0, 15.0), (40.0, 40.0, 50.0, 50.0)
    table = {0: [Detection(a, 0.9, 'x')], 1: [Detection(a, 0.9, 'x'), Detection(b, 0.9, 'x')],
             2: [Detection(a, 0.9, 'x'), Detection(b, 0.9, 'x')]}
    outs, _ = run(table, 3)
    assert [o.track_id for o in outs[0]] == [1]
    assert [o.track_id for o in outs[1]] == [1]
    assert sorted(o.track_id for o in outs[2]) == [1, 2]


def test_ids_restart_per_tracker_instance(two_squares):
    table = two_squares.detections()
    for _ in range(2):
        outs, _ = run(table, two_squares.n_frames)
        assert {o.track_id for f in outs for o in outs[f]} == {1, 2}

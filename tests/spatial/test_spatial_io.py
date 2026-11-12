import csv
import json

import numpy as np
import pytest

from spatial import io as sio
from spatial.detect import Detection
from spatial.pipeline import detections_from_json

PROMPTS = ['lobster', 'fish']
ROWS = [
    {'frame': 0, 'track_id': 1, 'box': (1.0, 2.0, 11.0, 22.0), 'score': 0.9, 'label': 'lobster'},
    {'frame': 0, 'track_id': 2, 'box': (30.5, 30.25, 40.0, 41.0), 'score': 0.8, 'label': 'fish'},
    {'frame': 1, 'track_id': 1, 'box': (2.0, 2.0, 12.0, 22.0), 'score': 0.85, 'label': 'lobster'},
    {'frame': 9, 'track_id': 3, 'box': (5.0, 5.0, 9.0, 9.0), 'score': 0.7, 'label': 'lobster'},
    {'frame': 19, 'track_id': 3, 'box': (6.0, 5.0, 10.0, 9.0), 'score': 0.75, 'label': 'lobster'},
]


def test_mot_file_layout(tmp_path):
    path = sio.write_mot(tmp_path / 'tracks.csv', ROWS, PROMPTS)
    lines = list(csv.reader(path.open()))
    assert len(lines) == len(ROWS)
    assert all(len(r) == 10 for r in lines)
    first = lines[0]
    assert first[:2] == ['1', '1']  # 1-based frame, track id
    assert [float(v) for v in first[2:6]] == [1.0, 2.0, 10.0, 20.0]  # x, y, w, h
    assert first[7:] == ['1', '-1', '-1']  # class 1 = lobster
    assert lines[1][7] == '2'  # fish
    assert [int(r[0]) for r in lines] == sorted(int(r[0]) for r in lines)


def test_mot_round_trip(tmp_path):
    path = sio.write_mot(tmp_path / 'tracks.csv', ROWS, PROMPTS)
    back = sio.read_mot(path)
    want = sorted(ROWS, key=lambda r: (r['frame'], r['track_id']))
    assert [(r['frame'], r['track_id']) for r in back] == [(r['frame'], r['track_id']) for r in want]
    for b, w in zip(back, want):
        assert b['box'] == pytest.approx(w['box'], abs=0.01)
        assert b['score'] == pytest.approx(w['score'], abs=1e-4)
        assert b['class_id'] == PROMPTS.index(w['label']) + 1


def test_track_labels_file(tmp_path):
    path = sio.write_track_labels(tmp_path / 'track_labels.csv', ROWS, PROMPTS)
    rows = list(csv.DictReader(path.open()))
    assert rows == [{'track_id': '1', 'class_id': '1', 'label': 'lobster'},
                    {'track_id': '2', 'class_id': '2', 'label': 'fish'},
                    {'track_id': '3', 'class_id': '1', 'label': 'lobster'}]


def test_summary_counts_and_times():
    s = sio.build_summary(ROWS, PROMPTS + ['crab'], fps=10.0)
    assert s['total_tracks'] == 3
    lob, fish, crab = s['labels']['lobster'], s['labels']['fish'], s['labels']['crab']
    assert lob['count'] == 2 and fish['count'] == 1 and crab['count'] == 0
    assert crab['first_seen_s'] is None and crab['tracks'] == []
    assert lob['first_seen_s'] == 0.0 and lob['last_seen_s'] == 1.9
    t3 = [t for t in lob['tracks'] if t['track_id'] == 3][0]
    assert t3 == {'track_id': 3, 'first_frame': 9, 'last_frame': 19, 'first_seen_s': 0.9,
                  'last_seen_s': 1.9, 'duration_s': 1.1, 'num_observations': 2}


def test_iter_frames_reads_rgb_and_marks_stride(tmp_path, two_squares):
    path = two_squares.write(tmp_path / 'v.mp4', fps=10)
    info = sio.video_info(path)
    assert info.fps == pytest.approx(10.0) and (info.width, info.height) == (64, 64)
    frames = list(sio.iter_frames(path, every_n=3))
    assert len(frames) == two_squares.n_frames
    assert [i for i, _, s in frames if s] == list(range(0, two_squares.n_frames, 3))
    _, f0, _ = frames[0]
    assert f0.shape == (64, 64, 3) and f0.dtype == np.uint8
    x1, y1, x2, y2 = [int(v) for v in two_squares.box(0, 0)]
    patch = f0[y1 + 2:y2 - 2, x1 + 2:x2 - 2].reshape(-1, 3).mean(0)
    assert patch[0] > 150 and patch[2] < 80  # red square (BGR (0,0,255)) is red in RGB
    assert len(list(sio.iter_frames(path, max_frames=5))) == 5


def test_bad_video_and_dir_errors(tmp_path):
    with pytest.raises(ValueError):
        sio.video_info(tmp_path / 'missing.mp4')
    with pytest.raises(ValueError):
        sio.list_videos(tmp_path / 'nope')
    (tmp_path / 'b.MP4').write_bytes(b'')
    (tmp_path / 'a.avi').write_bytes(b'')
    (tmp_path / 'notes.txt').write_text('x')
    assert [p.name for p in sio.list_videos(tmp_path)] == ['a.avi', 'b.MP4']


def test_detections_json_round_trip(tmp_path):
    payload = {'frames': [{'frame': 4, 'time_s': 0.4,
                           'detections': [Detection((1, 2, 3, 4), 0.5, 'fish').to_dict()]},
                          {'frame': 5, 'time_s': 0.5, 'detections': []}]}
    path = tmp_path / 'detections.json'
    path.write_text(json.dumps(payload))
    back = detections_from_json(path)
    assert back == {4: [Detection((1.0, 2.0, 3.0, 4.0), 0.5, 'fish')], 5: []}

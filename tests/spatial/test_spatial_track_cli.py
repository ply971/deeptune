"""track.py end-to-end with a scripted detector injected through ``main(detector=...)``."""
import csv
import json

import pytest

import track
from spatial import io as sio
from spatial.detect import FakeDetector


@pytest.fixture
def lobster_video(tmp_path, two_squares):
    path = two_squares.write(tmp_path / 'lobsters.mp4', fps=10)
    table = two_squares.detections(jitter=1.0, drop={0: {10, 11, 12}}, low={1: {15, 16, 17}})
    return path, table


def test_cli_writes_all_outputs(tmp_path, lobster_video, two_squares, track_ids_of):
    video, table = lobster_video
    out = tmp_path / 'run'
    res = track.main(['--video', str(video), '--prompt', 'lobster', '--prompt', 'fish', '--out', str(out),
                      '--save_video'], detector=FakeDetector(table))
    vdir = out / 'lobsters'
    for name in ('detections.json', 'tracks.csv', 'track_labels.csv', 'summary.json', 'overlay.mp4'):
        assert (vdir / name).is_file(), name
    assert not res['errors']

    dets = json.loads((vdir / 'detections.json').read_text())
    assert dets['num_frames'] == two_squares.n_frames and dets['fps'] == pytest.approx(10.0)
    assert [f['frame'] for f in dets['frames']] == list(range(two_squares.n_frames))
    assert dets['detector']['name'] == 'fake' and dets['prompts'] == ['lobster', 'fish']
    assert len(dets['frames'][10]['detections']) == 1  # object 0 dropped there

    summary = json.loads((vdir / 'summary.json').read_text())
    assert summary['labels']['lobster']['count'] == 2
    assert summary['labels']['fish']['count'] == 0
    assert summary['labels']['lobster']['first_seen_s'] == 0.0
    assert summary['duration_s'] == pytest.approx(3.0)

    rows = sio.read_mot(vdir / 'tracks.csv')
    assert {r['track_id'] for r in rows} == {1, 2} and {r['class_id'] for r in rows} == {1}
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r['frame'], []).append(type('T', (), {'track_id': r['track_id'], 'box': r['box']}))
    seq = track_ids_of(two_squares, by_frame)
    assert len({i for i in seq[0] if i}) == 1 and len({i for i in seq[1] if i}) == 1

    labels = list(csv.DictReader((vdir / 'track_labels.csv').open()))
    assert [(r['track_id'], r['label']) for r in labels] == [('1', 'lobster'), ('2', 'lobster')]

    import cv2
    cap = cv2.VideoCapture(str(vdir / 'overlay.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == two_squares.n_frames
    cap.release()

    saved = json.loads((out / 'track_arguments.json').read_text())
    assert saved['prompt'] == ['lobster', 'fish'] and saved['detector'] == 'grounding_dino'
    assert saved['box_threshold'] == 0.2 and saved['track_high_thresh'] == 0.35
    assert saved['track_low_thresh'] == 0.2 and saved['new_track_thresh'] == pytest.approx(0.45)
    assert saved['detector_resolved']['name'] == 'fake'
    counts = list(csv.DictReader((out / 'counts.csv').open()))
    assert [(r['label'], r['count']) for r in counts] == [('lobster', '2'), ('fish', '0')]


def test_cli_frame_stride_only_detects_sampled_frames(tmp_path, lobster_video):
    video, table = lobster_video
    fake = FakeDetector(table, batch_size=4)
    track.main(['--video', str(video), '--prompt', 'lobster', '--out', str(tmp_path / 'o'),
                '--every_n_frames', '3', '--track_low_thresh', '0.1'], detector=fake)
    seen = [i for call in fake.calls for i in call]
    assert seen == list(range(0, 30, 3))
    assert all(len(c) <= 4 for c in fake.calls)
    rows = sio.read_mot(tmp_path / 'o' / 'lobsters' / 'tracks.csv')
    assert {r['frame'] % 3 for r in rows} == {0}
    assert {r['track_id'] for r in rows} == {1, 2}


def test_cli_video_dir_and_max_frames(tmp_path, two_squares):
    vids = tmp_path / 'vids'
    vids.mkdir()
    two_squares.write(vids / 'a.mp4')
    two_squares.write(vids / 'b.mp4')
    (vids / 'broken.mp4').write_bytes(b'not a video')
    table = two_squares.detections()
    res = track.main(['--video_dir', str(vids), '--prompt', 'lobster', '--out', str(tmp_path / 'o'),
                      '--max_frames', '12'], detector=FakeDetector(lambda f, frame, p: table[f]))
    assert sorted(p.name for p in (tmp_path / 'o').iterdir() if p.is_dir()) == ['a', 'b']
    assert list(res['errors']) == [str(vids / 'broken.mp4')]
    assert (tmp_path / 'o' / 'errors.json').is_file()
    for name in ('a', 'b'):
        s = json.loads((tmp_path / 'o' / name / 'summary.json').read_text())
        assert s['num_frames'] == 12 and s['labels']['lobster']['count'] == 2
    counts = list(csv.DictReader((tmp_path / 'o' / 'counts.csv').open()))
    assert len(counts) == 2 and {r['count'] for r in counts} == {'2'}


def test_cli_owlv2_defaults_and_argument_errors(tmp_path, lobster_video):
    video, table = lobster_video
    track.main(['--video', str(video), '--prompt', 'lobster', '--detector', 'owlv2', '--out', str(tmp_path / 'o')],
               detector=FakeDetector(table))
    saved = json.loads((tmp_path / 'o' / 'track_arguments.json').read_text())
    assert saved['box_threshold'] == 0.1 and saved['text_threshold'] is None
    assert saved['detector_model'] == 'google/owlv2-base-patch16-ensemble'
    with pytest.raises(SystemExit):
        track.main(['--video', str(video), '--out', str(tmp_path / 'x')])  # no --prompt
    with pytest.raises(SystemExit):
        track.main(['--video', str(video), '--video_dir', str(tmp_path), '--prompt', 'a', '--out', 'x'])
    with pytest.raises(SystemExit):
        track.main(['--video', str(tmp_path / 'missing.mp4'), '--prompt', 'a', '--out', str(tmp_path / 'x')],
                   detector=FakeDetector({}))
    with pytest.raises(SystemExit):
        track.main(['--video', str(video), '--prompt', 'a', '--out', str(tmp_path / 'x'), '--every_n_frames', '0'],
                   detector=FakeDetector({}))


def test_cli_help_lists_every_option(capsys):
    with pytest.raises(SystemExit):
        track.main(['--help'])
    text = capsys.readouterr().out
    for flag in ('--video', '--video_dir', '--prompt', '--detector', '--box_threshold', '--text_threshold',
                 '--every_n_frames', '--batch_size', '--track_buffer', '--save_video', '--class_agnostic'):
        assert flag in text

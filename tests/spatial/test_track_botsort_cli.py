"""track.py with the BoT-SORT preset (default) versus plain ByteTrack."""
import json

import pytest

import track
from spatial import io as sio
from spatial.detect import FakeDetector


@pytest.fixture
def long_gap_video(tmp_path, make_scene):
    """One lobster, missed for 1.5 s (15 frames at 10 fps): longer than ByteTrack's buffer
    (30 steps at 30/s = 1 s), so plain ByteTrack gives it a second ID."""
    scene = make_scene([{'label': 'lobster', 'start': (2, 20), 'vel': (0.8, 0.0), 'size': 12,
                         'color': (0, 0, 255)}], n_frames=60, width=64, height=64)
    path = scene.write(tmp_path / 'one.mp4', fps=10)
    table = scene.detections(jitter=0.5, drop={0: set(range(20, 35))})
    return path, table


def _summary(out, name='one'):
    return json.loads((out / name / 'summary.json').read_text())


def _run(video, table, out, *extra):
    # --reid color keeps BoT-SORT runs machine-independent (auto picks cnn only where its weights are cached);
    # a bytetrack run keeps its preset's reid=none.
    reid = [] if 'bytetrack' in extra else ['--reid', 'color']
    return track.main(['--video', str(video), '--prompt', 'lobster', '--out', str(out), *reid, *extra],
                      detector=FakeDetector(table))


def test_bytetrack_preset_splits_a_long_miss_and_botsort_links_it(tmp_path, long_gap_video):
    video, table = long_gap_video
    _run(video, table, tmp_path / 'byte', '--tracker', 'bytetrack')
    _run(video, table, tmp_path / 'bot')
    byte, bot = _summary(tmp_path / 'byte'), _summary(tmp_path / 'bot')
    assert byte['labels']['lobster']['count'] == 2
    assert byte['tracking']['tracker'] == 'bytetrack' and byte['tracking']['postprocess'] is None
    assert bot['labels']['lobster']['count'] == 1
    assert bot['tracking']['postprocess']['links'] == 1
    assert bot['tracking']['reid']['name'] == 'color' and bot['tracking']['cmc'] == 'sparse'


def test_linking_off_keeps_the_split_and_interpolation_fills_short_holes(tmp_path, long_gap_video):
    video, table = long_gap_video
    _run(video, table, tmp_path / 'nolink', '--link_max_gap_s', '0')
    assert _summary(tmp_path / 'nolink')['labels']['lobster']['count'] == 2
    _run(video, table, tmp_path / 'fill', '--interpolate_max_gap_s', '2')
    rows = sio.read_mot(tmp_path / 'fill' / 'one' / 'tracks.csv')
    assert sorted(r['frame'] for r in rows) == list(range(60))  # the 15 missed frames are filled
    _run(video, table, tmp_path / 'nofill', '--interpolate_max_gap_s', '0')
    rows = sio.read_mot(tmp_path / 'nofill' / 'one' / 'tracks.csv')
    assert not set(range(21, 35)) & {r['frame'] for r in rows}


def test_botsort_writes_the_birth_frame_bytetrack_does_not(tmp_path, make_scene):
    scene = make_scene([{'label': 'lobster', 'start': (2, 4), 'vel': (1.0, 0.0), 'size': 10, 'color': (0, 0, 255)},
                        {'label': 'lobster', 'start': (2, 40), 'vel': (1.0, 0.0), 'size': 10, 'color': (0, 255, 0)}],
                       n_frames=20)
    video = scene.write(tmp_path / 'late.mp4', fps=10)
    table = scene.detections(jitter=0.3, drop={1: set(range(0, 5))})  # second object appears at frame 5
    for preset in ('bytetrack', 'botsort'):
        _run(video, table, tmp_path / preset, '--tracker', preset)
    first = {p: min(t['first_frame'] for t in _summary(tmp_path / p, 'late')['labels']['lobster']['tracks']
                    if t['first_frame'] > 0) for p in ('bytetrack', 'botsort')}
    assert first == {'bytetrack': 6, 'botsort': 5}


def test_one_off_false_positives_are_dropped(tmp_path, long_gap_video):
    from spatial.detect import Detection
    video, table = long_gap_video
    table = {f: list(d) for f, d in table.items()}
    for f in (40, 41):  # a 2-frame ghost far from the lobster
        table[f].append(Detection((40.0, 45.0, 52.0, 57.0), 0.95, 'lobster'))
    _run(video, table, tmp_path / 'keep', '--tracker', 'bytetrack')
    _run(video, table, tmp_path / 'drop')
    assert _summary(tmp_path / 'keep')['labels']['lobster']['count'] == 3
    assert _summary(tmp_path / 'drop')['labels']['lobster']['count'] == 1
    assert _summary(tmp_path / 'drop')['tracking']['postprocess']['dropped_short_tracks'] == 1


def test_presets_and_overrides_are_recorded(tmp_path, long_gap_video):
    video, table = long_gap_video
    _run(video, table, tmp_path / 'a', '--cmc', 'none', '--min_track_len', '5')
    saved = json.loads((tmp_path / 'a' / 'track_arguments.json').read_text())
    assert saved['tracker'] == 'botsort' and saved['cmc'] == 'none' and saved['min_track_len'] == 5
    assert saved['reid'] == 'color' and saved['reid_resolved'] == 'color'
    assert saved['link_max_gap_s'] == 3.0 and saved['interpolate_max_gap_s'] == 1.0
    assert saved['tracker_settings']['appearance_thresh'] == 0.25
    assert saved['reid_resolved_model']['name'] == 'color'
    track.main(['--video', str(video), '--prompt', 'lobster', '--out', str(tmp_path / 'b'), '--tracker', 'bytetrack'],
               detector=FakeDetector(table))
    saved = json.loads((tmp_path / 'b' / 'track_arguments.json').read_text())
    assert (saved['cmc'], saved['reid_resolved'], saved['link_max_gap_s'], saved['min_track_len']) == ('none', 'none', 0.0, 1)
    assert saved['reid_resolved_model'] is None


def test_injected_embedder_is_used(tmp_path, long_gap_video):
    from spatial.reid import ColorHistEmbedder

    class Counting(ColorHistEmbedder):
        calls = 0

        def embed(self, frame, boxes):
            Counting.calls += 1
            return super().embed(frame, boxes)
    video, table = long_gap_video
    track.main(['--video', str(video), '--prompt', 'lobster', '--out', str(tmp_path / 'o'), '--reid', 'cnn'],
               detector=FakeDetector(table), embedder=Counting())
    assert Counting.calls == 60


@pytest.mark.parametrize('extra', [
    ['--link_max_gap_s', '-1'], ['--interpolate_max_gap_s', '-0.5'], ['--min_track_len', '0'],
    ['--link_appearance_thresh', '1.5'], ['--appearance_thresh', '2'], ['--proximity_thresh', '0'],
    ['--tracker', 'sort'], ['--reid', 'resnet50'], ['--cmc', 'orb'],
], ids=lambda a: ' '.join(a))
def test_bad_tracker_arguments_are_usage_errors(tmp_path, long_gap_video, extra, capsys):
    video, _ = long_gap_video
    fake = FakeDetector({})
    with pytest.raises(SystemExit) as exc:
        track.main(['--video', str(video), '--prompt', 'x', '--out', str(tmp_path / 'o')] + extra, detector=fake)
    assert exc.value.code == 2 and fake.calls == [] and not (tmp_path / 'o').exists()

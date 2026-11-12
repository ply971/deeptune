"""The Analyse Video page: logic, subprocess handling and real widget callbacks.

CPU only, no network, no weights. Subprocess tests launch small fake scripts that
write the files track.py and infer.py write. Their contents come from what
spatial/io.py and inference/video.py produce, and test_fake_track_output_matches
_the_real_pipeline runs the real pipeline with a scripted detector to keep that honest.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from desktop import analyse, analyse_page, app, runner
from desktop import config as gc

REPO = Path(__file__).resolve().parents[1]


def write_clip(path: Path, frames: int = 12, size=(64, 48)) -> Path:
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
    assert writer.isOpened()
    for value in range(frames):
        writer.write(np.full((size[1], size[0], 3), value * 10 % 256, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path):
    return write_clip(tmp_path / 'street.mp4')


def summary_for(labels: dict) -> dict:
    return {'video': 'street.mp4', 'num_frames': 12, 'duration_s': 1.2, 'every_n_frames': 2,
            'prompts': list(labels), 'fps': 10.0, 'total_tracks': sum(e['count'] for e in labels.values()),
            'labels': labels}


CAR = {'count': 3, 'first_seen_s': 0.2, 'last_seen_s': 1.1, 'tracks': []}
NOTHING = {'count': 0, 'first_seen_s': None, 'last_seen_s': None, 'tracks': []}


def write_track_output(out: Path, video: Path, labels: dict, overlay_from: Path | None = None, args: dict | None = None):
    folder = out / video.stem
    folder.mkdir(parents=True)
    (folder / 'summary.json').write_text(json.dumps(summary_for(labels)), encoding='utf-8')
    (out / 'track_arguments.json').write_text(json.dumps(args or {
        'detector_model': 'IDEA-Research/grounding-dino-tiny', 'detector_licence': 'Apache-2.0',
        'new_track_thresh': 0.45, 'detector_resolved': {'device': 'cuda'}}), encoding='utf-8')
    if overlay_from is not None:
        (folder / 'overlay.mp4').write_bytes(overlay_from.read_bytes())
    return folder


def video_run(tmp_path: Path, *, error_model: bool = False, mode: str = 'cls', name='deeptune-1',
              model_version: str = 'r3d_18') -> Path:
    run = tmp_path / 'out' / name
    train = run / 'train_output'
    train.mkdir(parents=True)
    (train / 'model_weights.pth').write_bytes(b'not read by these tests')
    (train / 'cli_arguments.json').write_text(json.dumps(
        {'modality': 'video', 'model_version': model_version, 'mode': mode, 'num_classes': 2, 'pooling': 'mean',
         'num_frames': 8}),
        encoding='utf-8')
    splits = run / 'data_splits_1'
    splits.mkdir()
    (splits / 'label_mapping.json').write_text(json.dumps({'walk': 0, 'run': 1}), encoding='utf-8')
    if error_model:
        (train / 'error_model.json').write_text('{}', encoding='utf-8')
    return run


# ---------------------------------------------------------------------------
# GUI stays torch-free
# ---------------------------------------------------------------------------

def test_importing_the_app_and_page_does_not_import_torch():
    code = ("import sys, desktop.app, desktop.analyse_page, desktop.analyse, desktop.runner\n"
            "bad = [m for m in ('torch', 'transformers', 'torchvision') if m in sys.modules]\n"
            "assert not bad, bad")
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_frozen_track_worker_entry_runs_track_cli_without_the_gui():
    code = ("import runpy, sys\n"
            "sys.argv = ['app.py', '--track-worker', '--help']\n"
            "try:\n"
            "    runpy.run_path(r'desktop/app.py', run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    assert exc.code == 0, exc.code\n"
            "assert 'dearpygui' not in sys.modules and 'torch' not in sys.modules\n")
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert '--save_video' in result.stdout


# ---------------------------------------------------------------------------
# Building the command line
# ---------------------------------------------------------------------------

def parse_with_real_track_cli(argv):
    import track
    parser = track.get_parser()
    return track.resolve_args(parser.parse_args(argv), parser)


def test_track_argv_is_accepted_by_track_cli(tmp_path):
    cfg = analyse.TrackConfig(video=Path('clip one.mp4'), prompts=['person', 'red car'], out=tmp_path,
                              detector='owlv2', every_n_frames=4)
    args = parse_with_real_track_cli(analyse.build_track_argv(cfg))
    assert args.video == 'clip one.mp4' and args.out == str(tmp_path)
    assert args.prompt == ['person', 'red car']
    assert args.detector == 'owlv2' and args.every_n_frames == 4 and args.save_video


def test_prompt_starting_with_dash_is_not_taken_for_a_flag(tmp_path):
    cfg = analyse.TrackConfig(video=Path('a.mp4'), prompts=['-bird'], out=tmp_path)
    assert parse_with_real_track_cli(analyse.build_track_argv(cfg)).prompt == ['-bird']


@pytest.mark.parametrize('missing', ['video', 'prompts', 'out'])
def test_track_argv_requires_everything(tmp_path, missing):
    cfg = analyse.TrackConfig(video=Path('a.mp4'), prompts=['a'], out=tmp_path)
    setattr(cfg, missing, [] if missing == 'prompts' else None)
    with pytest.raises(ValueError):
        analyse.build_track_argv(cfg)


@pytest.mark.parametrize('detector', ['grounding_dino', 'owlv2'])
def test_default_score_matches_track_cli_and_adds_no_flags(tmp_path, detector):
    resolved = parse_with_real_track_cli(['--video', 'a.mp4', '--prompt', 'x', '--out', str(tmp_path),
                                          '--detector', detector])
    assert analyse.default_track_score(detector) == resolved.new_track_thresh
    assert analyse.threshold_flags(detector, None) == []
    assert analyse.threshold_flags(detector, resolved.new_track_thresh) == []


@pytest.mark.parametrize('detector,score', [('grounding_dino', 0.1), ('grounding_dino', 0.3), ('grounding_dino', 0.6),
                                            ('owlv2', 0.05), ('owlv2', 0.15), ('owlv2', 0.5)])
def test_a_box_at_the_chosen_score_can_reach_a_track(tmp_path, detector, score):
    """The setting must open every gate on the way to a new track, or lowering it does nothing."""
    cfg = analyse.TrackConfig(video=Path('a.mp4'), prompts=['x'], out=tmp_path, detector=detector, track_score=score)
    args = parse_with_real_track_cli(analyse.build_track_argv(cfg))
    assert args.new_track_thresh == score
    assert args.track_high_thresh <= score
    assert args.box_threshold <= score
    if detector == 'grounding_dino':
        assert args.text_threshold <= score


def test_real_tracker_starts_a_track_only_after_the_threshold_is_lowered(tmp_path):
    from spatial.detect import Detection
    from spatial.track import ByteTracker

    def tracks_with(score):
        cfg = analyse.TrackConfig(video=Path('a.mp4'), prompts=['x'], out=tmp_path, track_score=score)
        args = parse_with_real_track_cli(analyse.build_track_argv(cfg))
        import track
        tracker = ByteTracker(frame_rate=10, **track.tracker_kwargs(args))
        box = (10.0, 10.0, 30.0, 30.0)
        seen = []
        for _ in range(4):
            seen += tracker.update([Detection(box, 0.3, 'x')])
        return seen

    assert tracks_with(None) == []
    assert tracks_with(0.25)


def test_parse_prompts():
    assert analyse.parse_prompts(' person, red car ;bird\ndog ') == ['person', 'red car', 'bird', 'dog']
    assert analyse.parse_prompts('car, CAR') == ['car']
    assert analyse.parse_prompts('  , ;\n') == []
    assert analyse.parse_prompts(None) == []


def test_validate_track_config_reports_each_problem(tmp_path, clip):
    good = dict(video=clip, prompts=['x'], out=tmp_path)
    assert analyse.validate_track_config(analyse.TrackConfig(**good)) == []
    assert any('Type at least' in i for i in analyse.validate_track_config(analyse.TrackConfig(**{**good, 'prompts': []})))
    assert any('not found' in i for i in analyse.validate_track_config(
        analyse.TrackConfig(**{**good, 'video': tmp_path / 'gone.mp4'})))
    assert any('frames step' in i for i in analyse.validate_track_config(analyse.TrackConfig(**good, every_n_frames=0)))
    assert any('threshold' in i for i in analyse.validate_track_config(analyse.TrackConfig(**good, track_score=0)))
    assert any('identical' in i for i in analyse.validate_track_config(
        analyse.TrackConfig(**{**good, 'prompts': ['St. Bernard', 'st bernard']})))
    # OWLv2 has no phrase splitting, so those two prompts are fine for it.
    assert analyse.validate_track_config(
        analyse.TrackConfig(**{**good, 'prompts': ['St. Bernard', 'st bernard']}, detector='owlv2')) == []


# ---------------------------------------------------------------------------
# Checking the video
# ---------------------------------------------------------------------------

def test_check_video_accepts_a_real_clip_and_describes_it(clip):
    check = analyse.check_video(f'"{clip}"')
    assert check.ok and check.path == clip
    assert '64x48' in check.description and '1.2 s' in check.description


def test_check_video_error_cases(tmp_path):
    assert 'Choose a video' in analyse.check_video('').error
    assert 'Choose a video' in analyse.check_video(None).error
    assert 'not found' in analyse.check_video(tmp_path / 'gone.mp4').error
    assert 'folder' in analyse.check_video(tmp_path).error
    (tmp_path / 'notes.txt').write_text('x')
    assert 'not a supported' in analyse.check_video(tmp_path / 'notes.txt').error
    (tmp_path / 'broken.mp4').write_bytes(b'this is not a video' * 50)
    assert 'Could not read' in analyse.check_video(tmp_path / 'broken.mp4').error
    (tmp_path / 'empty.mp4').write_bytes(b'')
    assert 'Could not read' in analyse.check_video(tmp_path / 'empty.mp4').error


# ---------------------------------------------------------------------------
# Runs and predictions
# ---------------------------------------------------------------------------

def test_list_video_runs_hides_other_modalities(tmp_path):
    video_run(tmp_path, name='deeptune-video')
    images = tmp_path / 'out' / 'deeptune-images' / 'train_output'
    images.mkdir(parents=True)
    (images / 'cli_arguments.json').write_text('{"modality": "images"}', encoding='utf-8')
    broken = tmp_path / 'out' / 'deeptune-broken' / 'train_output'
    broken.mkdir(parents=True)
    (broken / 'cli_arguments.json').write_text('{not json', encoding='utf-8')
    choices = analyse.list_video_runs(tmp_path / 'out')
    assert [p.name for p in choices.runs.values()] == ['deeptune-video']
    assert choices.hidden == 2 and '2 run(s)' in choices.message


def test_list_video_runs_explains_an_empty_list(tmp_path):
    assert 'No trained runs' in analyse.list_video_runs(tmp_path / 'missing').message
    images = tmp_path / 'deeptune-images' / 'train_output'
    images.mkdir(parents=True)
    (images / 'cli_arguments.json').write_text('{"modality": "images"}', encoding='utf-8')
    assert 'None of the 1' in analyse.list_video_runs(tmp_path).message


def prediction(columns: dict) -> gc.InferenceResults:
    return gc.InferenceResults(out_dir=Path('.'), predictions=pd.DataFrame(columns))


def test_classifier_prediction_with_error_model(tmp_path):
    run = gc.read_run_results(video_run(tmp_path, error_model=True))
    view = analyse.summarise_prediction(prediction({'predicted_class': [1], 'predicted_label': ['run'],
                                                    'confidence': [0.873], 'expected_error': [0.12]}), run, Path('a.mp4'))
    assert not view.error
    assert view.answer == 'Predicted class: run'
    assert view.confidence == 'Confidence: 87.3%'
    assert 'about 12.0% were wrong' in view.expected_error and 'chance' not in view.expected_error
    assert 'deeptune-1' in view.model and 'r3d_18' in view.model and view.video == 'a.mp4'


def test_regression_prediction_has_no_confidence_and_states_its_error_in_units(tmp_path):
    run = gc.read_run_results(video_run(tmp_path, mode='reg', error_model=True))
    view = analyse.summarise_prediction(prediction({'predicted_value': [2.5], 'expected_error': [0.4]}), run, Path('a.mp4'))
    assert view.answer == 'Predicted value: 2.5'
    assert 'not available' in view.confidence
    assert 'about 0.4' in view.expected_error and 'absolute error' in view.expected_error


def test_run_without_error_model_says_so_plainly(tmp_path):
    run = gc.read_run_results(video_run(tmp_path))
    view = analyse.summarise_prediction(prediction({'predicted_label': ['run'], 'confidence': [0.9]}), run, Path('a.mp4'))
    assert not view.error
    assert 'This run has no error model' in view.expected_error


def test_error_model_that_was_refused_is_not_described_as_missing(tmp_path):
    run = gc.read_run_results(video_run(tmp_path, error_model=True))
    view = analyse.summarise_prediction(prediction({'predicted_label': ['run'], 'confidence': [0.9]}), run, Path('a.mp4'))
    assert 'could not be used' in view.expected_error


def test_nan_expected_error_is_not_shown_as_a_number(tmp_path):
    run = gc.read_run_results(video_run(tmp_path))
    view = analyse.summarise_prediction(prediction({'predicted_label': ['run'], 'confidence': [0.9],
                                                    'expected_error': [float('nan')]}), run, Path('a.mp4'))
    assert 'nan' not in view.expected_error.lower().replace('not available', '')
    assert 'not available' in view.expected_error


@pytest.mark.parametrize('results', [
    gc.InferenceResults(out_dir=Path('.')),
    gc.InferenceResults(out_dir=Path('.'), predictions=pd.DataFrame()),
    gc.InferenceResults(out_dir=Path('.'), predictions=pd.DataFrame({'sample_id': [0]})),
    gc.InferenceResults(out_dir=Path('.'), predictions=pd.DataFrame({'predicted_value': ['oops']})),
])
def test_unusable_predictions_are_errors_not_crashes(tmp_path, results):
    run = gc.read_run_results(video_run(tmp_path))
    assert analyse.summarise_prediction(results, run, Path('a.mp4')).error


def test_class_names_note(tmp_path):
    run = gc.read_run_results(video_run(tmp_path))
    assert analyse.class_names_note(run) == 'It can only answer with: run, walk.'
    run.label_mapping = {f'c{i:02d}': i for i in range(20)}
    assert analyse.class_names_note(run).endswith(', ....')
    run.label_mapping = None
    assert analyse.class_names_note(run) == ''


# ---------------------------------------------------------------------------
# Reading tracking results
# ---------------------------------------------------------------------------

def test_track_results_are_read_with_counts_times_and_model(tmp_path, clip):
    write_track_output(tmp_path, clip, {'car': CAR, 'bus': NOTHING}, overlay_from=clip)
    results = analyse.read_track_results(tmp_path, clip)
    assert not results.error and results.total_tracks == 3
    assert [label.row for label in results.labels] == [('car', '3', '0.2 s', '1.1 s'), ('bus', '0', '-', '-')]
    assert results.overlay == tmp_path / clip.stem / 'overlay.mp4'
    assert 'grounding-dino-tiny' in results.model and 'cuda' in results.model and 'Apache-2.0' in results.model
    assert analyse.frames_note(results) == 'Looked at 6 of 12 frames (every 2).'


def test_fake_track_output_matches_the_real_pipeline(tmp_path, clip):
    """The real pipeline and a scripted detector, so these tests follow track.py's real files."""
    import track
    from spatial.detect import Detection, FakeDetector
    box = (5.0, 5.0, 25.0, 25.0)
    detector = FakeDetector({i: [Detection(box, 0.9, 'car')] for i in range(12)})
    out = tmp_path / 'real'
    code = track.cli(['--video', str(clip), '--prompt', 'car', '--prompt', 'bus', '--out', str(out), '--save_video'],
                     detector=detector)
    assert code == 0
    results = analyse.read_track_results(out, clip, 'grounding_dino')
    assert not results.error
    assert [label.name for label in results.labels] == ['car', 'bus']
    assert results.labels[0].count == 1 and results.labels[1].count == 0
    assert results.labels[0].first_seen_s is not None
    assert results.overlay is not None and analyse.check_video(results.overlay).ok
    assert results.num_frames == 12


def test_empty_results_are_no_tracks_not_a_failure(tmp_path, clip):
    folder = write_track_output(tmp_path, clip, {'car': NOTHING})
    (folder / 'detections.json').write_text(json.dumps({'frames': [{'frame': 0, 'detections': []}]}), encoding='utf-8')
    results = analyse.read_track_results(tmp_path, clip)
    assert not results.error and results.total_tracks == 0
    assert 'No tracks found' in results.no_tracks_message
    assert 'no boxes' in results.no_tracks_message and 'lower score threshold' in results.no_tracks_message


def test_no_tracks_message_tells_apart_weak_boxes(tmp_path, clip):
    folder = write_track_output(tmp_path, clip, {'car': NOTHING})
    (folder / 'detections.json').write_text(json.dumps({'frames': [
        {'frame': 0, 'detections': [{'box_xyxy': [0, 0, 1, 1], 'score': 0.31, 'label': 'car'}]},
        {'frame': 1, 'detections': [{'box_xyxy': [0, 0, 1, 1], 'score': 0.22, 'label': 'car'}]},
    ]}), encoding='utf-8')
    message = analyse.read_track_results(tmp_path, clip).no_tracks_message
    assert '2 box(es)' in message and 'best score 0.31' in message and 'at least 0.45' in message


def test_empty_label_table_counts_as_no_tracks(tmp_path, clip):
    write_track_output(tmp_path, clip, {})
    results = analyse.read_track_results(tmp_path, clip)
    assert not results.error and 'No tracks found' in results.no_tracks_message


@pytest.mark.parametrize('content,expected', [
    ('{not json', 'could not be read'),
    ('[1, 2]', 'not a JSON object'),
    ('{"fps": 10}', 'no labels'),
    ('{"labels": ["car"]}', 'no labels'),
    ('{"labels": {"car": 3}}', 'not an object'),
    ('{"labels": {"car": {"count": "three"}}}', 'not a number of tracks'),
    ('{"labels": {"car": {"count": -1}}}', 'not a number of tracks'),
    ('{"labels": {"car": {"count": true}}}', 'not a number of tracks'),
    ('{"labels": {"car": {"count": 1, "first_seen_s": "soon"}}}', 'not a time'),
])
def test_malformed_summary_is_an_error_not_a_crash(tmp_path, clip, content, expected):
    folder = tmp_path / clip.stem
    folder.mkdir()
    (folder / 'summary.json').write_text(content, encoding='utf-8')
    results = analyse.read_track_results(tmp_path, clip, 'owlv2')
    assert expected in results.error
    assert 'owlv2' in results.model.lower()


def test_missing_summary_is_an_error(tmp_path, clip):
    assert 'summary.json is missing' in analyse.read_track_results(tmp_path, clip).error


def test_bad_track_arguments_still_name_the_requested_detector(tmp_path, clip):
    write_track_output(tmp_path, clip, {'car': CAR})
    (tmp_path / 'track_arguments.json').write_text('{broken', encoding='utf-8')
    results = analyse.read_track_results(tmp_path, clip, 'owlv2')
    assert not results.error and 'google/owlv2' in results.model


def test_empty_overlay_file_is_not_offered(tmp_path, clip):
    folder = write_track_output(tmp_path, clip, {'car': CAR})
    (folder / 'overlay.mp4').write_bytes(b'')
    assert analyse.read_track_results(tmp_path, clip).overlay is None


def test_last_log_lines_keeps_the_end():
    assert analyse.last_log_lines('a\n\nb\nc  \n', count=2) == 'b\nc'
    assert analyse.last_log_lines('') == ''


def test_detector_note_states_download_size_and_notices_the_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('HF_HOME', raising=False)
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path))
    assert '0.7 GB' in analyse.detector_note('grounding_dino')
    assert '0.6 GB' in analyse.detector_note('owlv2')
    assert 'already in the local' not in analyse.detector_note('owlv2')
    snapshot = tmp_path / 'models--google--owlv2-base-patch16-ensemble' / 'snapshots' / 'abc'
    snapshot.mkdir(parents=True)
    assert 'already in the local' not in analyse.detector_note('owlv2')  # a stopped download leaves this
    assert '0.6 GB' in analyse.track_button_label('owlv2') and '1.3 GB' in analyse.detector_note('owlv2')
    (snapshot / 'model.safetensors').write_bytes(b'x')
    assert 'already in the local' in analyse.detector_note('owlv2')
    assert 'GB' not in analyse.track_button_label('owlv2')
    assert 'already in the local' not in analyse.detector_note('grounding_dino')


# ---------------------------------------------------------------------------
# start_track_run, with a fake script instead of track.py
# ---------------------------------------------------------------------------

FAKE_TRACK = r"""
import json, shutil, sys, time
from pathlib import Path
args = sys.argv[1:]
video = Path(args[args.index('--video') + 1])
out = Path(args[args.index('--out') + 1])
behaviour = %r
print('fake detector loading', flush=True)
if behaviour == 'crash':
    print('CUDA out of memory: tried to allocate 2 GB', flush=True)
    sys.exit(3)
if behaviour == 'hang':
    time.sleep(60)
if behaviour == 'no_summary':
    sys.exit(0)
folder = out / video.stem
folder.mkdir(parents=True)
labels = {a.split('=', 1)[1]: {'count': 0 if behaviour == 'empty' else 2, 'first_seen_s': None if behaviour == 'empty' else 0.1,
                               'last_seen_s': None if behaviour == 'empty' else 0.9, 'tracks': []}
          for a in args if a.startswith('--prompt=')}
(folder / 'summary.json').write_text(json.dumps({'num_frames': 12, 'every_n_frames': 1, 'labels': labels}))
(folder / 'detections.json').write_text(json.dumps({'frames': []}))
(out / 'track_arguments.json').write_text(json.dumps({'detector_model': 'fake/detector', 'new_track_thresh': 0.45}))
if '--save_video' in args:
    shutil.copy(video, folder / 'overlay.mp4')
print('[track] finished', flush=True)
"""

FAKE_INFER = r"""
import sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index('--out') + 1])
out.mkdir(parents=True, exist_ok=True)
(out / 'predictions.csv').write_text('sample_id,predicted_class,predicted_label,confidence\n0,1,run,0.75\n')
print('Inference complete.', flush=True)
"""


def fake_track(monkeypatch, tmp_path, behaviour='ok'):
    script = tmp_path / 'fake_track.py'
    script.write_text(FAKE_TRACK % behaviour, encoding='utf-8')
    monkeypatch.setattr(runner, 'TRACK_SCRIPT', script)


def fake_infer(monkeypatch, tmp_path):
    script = tmp_path / 'fake_infer.py'
    script.write_text(FAKE_INFER, encoding='utf-8')
    monkeypatch.setattr(runner, 'INFER_SCRIPT', script)


def drain(handle, timeout=60):
    handle.thread.join(timeout=timeout)
    assert not handle.thread.is_alive()
    chunks = []
    while not handle.log_queue.empty():
        chunks.append(handle.log_queue.get_nowait())
    return handle.done_queue.get_nowait(), ''.join(chunks)


def test_start_track_run_refuses_a_bad_config_before_launching(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='not found'):
        runner.start_track_run(analyse.TrackConfig(video=tmp_path / 'gone.mp4', prompts=['x'], out=tmp_path / 'o'))
    assert not (tmp_path / 'o').exists()


def test_start_track_run_streams_the_log_and_writes_results(tmp_path, monkeypatch, clip):
    monkeypatch.delenv('HF_HOME', raising=False)
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path / 'empty-cache'))
    fake_track(monkeypatch, tmp_path)
    out = tmp_path / 'o'
    done, log = drain(runner.start_track_run(analyse.TrackConfig(video=clip, prompts=['car'], out=out, detector='owlv2')))
    assert done == ('success', '')
    assert 'Loading OWLv2' in log and '0.6 GB' in log and 'fake detector loading' in log
    assert analyse.read_track_results(out, clip).labels[0].count == 2


def test_start_track_run_reports_a_non_zero_exit(tmp_path, monkeypatch, clip):
    fake_track(monkeypatch, tmp_path, 'crash')
    done, log = drain(runner.start_track_run(analyse.TrackConfig(video=clip, prompts=['car'], out=tmp_path / 'o')))
    assert done[0] == 'error' and 'Tracking exited with code 3' in done[1]
    assert 'out of memory' in log


def test_start_track_run_stop_ends_the_process_cleanly(tmp_path, monkeypatch, clip):
    fake_track(monkeypatch, tmp_path, 'hang')
    handle = runner.start_track_run(analyse.TrackConfig(video=clip, prompts=['car'], out=tmp_path / 'o'))
    deadline = time.monotonic() + 30
    while handle._process_state.process is None and time.monotonic() < deadline:
        time.sleep(.02)
    process = handle._process_state.process
    assert process is not None
    assert handle.cancel()
    done, _ = drain(handle, timeout=30)
    assert done[0] == 'cancelled'
    assert process.poll() is not None


def test_infer_run_still_uses_infer_script_and_wording(tmp_path, monkeypatch):
    script = tmp_path / 'bad_infer.py'
    script.write_text('import sys; print("boom", flush=True); sys.exit(4)', encoding='utf-8')
    monkeypatch.setattr(runner, 'INFER_SCRIPT', script)
    cfg = gc.InferenceConfig(run_dir=tmp_path, modality='video', checkpoint_path=tmp_path / 'w.pth', data=tmp_path,
                             out=tmp_path / 'o')
    done, log = drain(runner.start_infer_run(cfg))
    assert done[0] == 'error' and 'Inference exited with code 4' in done[1]
    assert 'Starting inference' in log and 'boom' in log


# ---------------------------------------------------------------------------
# The page, driven through the real widget callbacks
# ---------------------------------------------------------------------------

def dispatch(ui, item, data=None):
    callback = ui.dpg.get_item_callback(item)
    assert callback is not None
    ui.dpg.run_callbacks([[callback, item, data, ui.dpg.get_item_user_data(item)]])


@pytest.fixture
def ui():
    app.dpg.create_context()
    app.build_ui()
    try:
        yield app
    finally:
        analyse_page.shutdown()
        app.stop_video_playback(timeout=2)
        app.dpg.destroy_context()


def text_under(ui, item) -> str:
    """Every text value below an item, as the user would read it."""
    parts = []
    children = ui.dpg.get_item_children(item) or {}
    for slot in children.values():
        for child in slot:
            if ui.dpg.get_item_type(child) == 'mvAppItemType::mvText':
                parts.append(str(ui.dpg.get_value(child)))
            parts.append(text_under(ui, child))
    return '\n'.join(p for p in parts if p)


def wait_for_action(ui, timeout=60):
    deadline = time.monotonic() + timeout
    while analyse_page._active is not None and time.monotonic() < deadline:
        analyse_page.poll()
        time.sleep(.02)
    assert analyse_page._active is None, 'the action did not finish'


def open_page(ui, tmp_path, *, video=None, run=False, **run_kwargs):
    ui.dpg.set_value('outdir_input', str(tmp_path / 'out'))
    if run:
        video_run(tmp_path, **run_kwargs)
    ui.on_nav_selected(user_data='analyse')
    if video is not None:
        ui.dpg.set_value(analyse_page.VIDEO_INPUT, str(video))
        dispatch(ui, 'analyse_load_button')


def test_page_is_reachable_and_states_its_limit(ui):
    ui.on_nav_selected(user_data='analyse')
    assert ui.dpg.is_item_shown('page_analyse')
    assert not ui.dpg.is_item_shown('page_test')
    assert 'ANALYSE VIDEO' in ui.dpg.get_value('workspace_location')
    page = text_under(ui, 'page_analyse')
    assert 'answers only with its own classes' in page and 'not your model' in page
    assert '0.7 GB' in page


@pytest.mark.parametrize('action', ['segment', 'ask'])
def test_reserved_actions_say_coming_later_and_cannot_be_used(ui, action):
    button = f'analyse_{action}_button'
    assert 'coming later' in ui.dpg.get_item_label(button)
    assert ui.dpg.get_item_configuration(button)['enabled'] is False
    assert ui.dpg.get_item_callback(button) is None


def test_every_app_page_still_switches(ui):
    for key, _ in ui.NAV_ITEMS:
        ui.on_nav_selected(user_data=key)
        assert ui.dpg.is_item_shown(f'page_{key}')
        for other, _ in ui.NAV_ITEMS:
            if other != key:
                assert not ui.dpg.is_item_shown(f'page_{other}')


def test_loading_a_video_previews_it(ui, tmp_path, clip):
    open_page(ui, tmp_path, video=clip)
    assert '64x48' in ui.dpg.get_value('analyse_video_status')
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not ui.dpg.does_item_exist(analyse_page.TEXTURE):
        analyse_page.poll()
        time.sleep(.01)
    assert ui.dpg.does_item_exist(analyse_page.IMAGE)
    assert ui.dpg.is_item_shown('analyse_player_controls')
    # Leaving the page pauses it, and the Dataset page's own player is not involved.
    ui.on_nav_selected(user_data='overview')
    assert not analyse_page._player.state.playing
    assert ui._video_player is None


@pytest.mark.parametrize('name,bytes_,expected', [
    ('gone.mp4', None, 'not found'), ('broken.mp4', b'junk' * 100, 'Could not read'), ('a.txt', b'x', 'not a supported'),
])
def test_bad_video_is_explained_and_clears_the_preview(ui, tmp_path, clip, name, bytes_, expected):
    open_page(ui, tmp_path, video=clip)
    assert analyse_page._player is not None
    bad = tmp_path / name
    if bytes_ is not None:
        bad.write_bytes(bytes_)
    ui.dpg.set_value(analyse_page.VIDEO_INPUT, str(bad))
    dispatch(ui, 'analyse_load_button')
    assert expected in ui.dpg.get_value('analyse_video_status')
    assert analyse_page._player is None


def test_predict_without_a_run_asks_for_one(ui, tmp_path, clip, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: pytest.fail('must not launch'))
    open_page(ui, tmp_path, video=clip)
    dispatch(ui, 'analyse_predict_button')
    assert 'Choose a trained run' in ui.dpg.get_value('analyse_predict_message')


def test_predict_without_a_video_asks_for_one(ui, tmp_path, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: pytest.fail('must not launch'))
    open_page(ui, tmp_path, run=True)
    dispatch(ui, 'analyse_predict_button')
    assert 'Choose a video' in ui.dpg.get_value('analyse_predict_message')


def test_run_without_a_checkpoint_is_explained(ui, tmp_path, clip, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: pytest.fail('must not launch'))
    run = video_run(tmp_path)
    (run / 'train_output' / 'model_weights.pth').unlink()
    open_page(ui, tmp_path, video=clip)
    assert 'no saved checkpoint' in ui.dpg.get_value('analyse_run_info')
    dispatch(ui, 'analyse_predict_button')
    assert 'no saved checkpoint' in ui.dpg.get_value('analyse_predict_message')


def test_predict_reuses_the_test_page_path_and_shows_the_answer(ui, tmp_path, clip, monkeypatch):
    fake_infer(monkeypatch, tmp_path)
    open_page(ui, tmp_path, video=clip, run=True)
    assert 'r3d_18' in ui.dpg.get_value('analyse_run_info')
    assert 'run, walk' in ui.dpg.get_value('analyse_run_info')
    seen = []
    real = runner.start_infer_run
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: seen.append(cfg) or real(cfg))
    dispatch(ui, 'analyse_predict_button')
    assert ui.dpg.get_item_configuration('analyse_track_button')['enabled'] is False
    wait_for_action(ui)
    cfg = seen[0]
    assert cfg.modality == 'video' and cfg.data == clip and cfg.model_version == 'r3d_18'
    assert cfg.label_mapping_path is not None
    assert cfg.out.parent.name == 'desktop-inference'
    results = text_under(ui, 'analyse_predict_results')
    assert 'Predicted class: run' in results and 'Confidence: 75.0%' in results
    assert 'This run has no error model' in results
    assert 'Produced by your trained model: deeptune-1 (r3d_18' in results
    assert ui.dpg.get_item_configuration('analyse_track_button')['enabled'] is True


def test_prediction_clears_when_another_run_or_video_is_chosen(ui, tmp_path, clip, monkeypatch):
    fake_infer(monkeypatch, tmp_path)
    open_page(ui, tmp_path, video=clip, run=True)
    dispatch(ui, 'analyse_predict_button')
    wait_for_action(ui)
    assert text_under(ui, 'analyse_predict_results')
    dispatch(ui, 'analyse_run_combo', None)
    assert not text_under(ui, 'analyse_predict_results')


def test_empty_prompt_is_refused_without_launching(ui, tmp_path, clip, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_track_run', lambda cfg: pytest.fail('must not launch'))
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', ' , ')
    dispatch(ui, 'analyse_track_button')
    assert 'Type at least one thing' in ui.dpg.get_value('analyse_track_message')


def test_track_without_a_video_is_refused(ui, tmp_path, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_track_run', lambda cfg: pytest.fail('must not launch'))
    open_page(ui, tmp_path)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    assert 'Choose a video' in ui.dpg.get_value('analyse_track_message')


def test_track_without_an_output_folder_is_refused(ui, tmp_path, clip, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_track_run', lambda cfg: pytest.fail('must not launch'))
    ui.on_nav_selected(user_data='analyse')
    ui.dpg.set_value('outdir_input', '')
    ui.dpg.set_value(analyse_page.VIDEO_INPUT, str(clip))
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    assert 'output folder' in ui.dpg.get_value('analyse_track_message')


def test_detector_choice_moves_the_threshold_and_the_download_note(ui):
    dpg = ui.dpg
    assert dpg.get_value('analyse_score_input') == pytest.approx(analyse.default_track_score('grounding_dino'))
    dpg.set_value('analyse_score_input', 0.9)
    dpg.set_value('analyse_detector_combo', 'OWLv2')
    dispatch(ui, 'analyse_detector_combo', 'OWLv2')
    assert dpg.get_value('analyse_score_input') == pytest.approx(analyse.default_track_score('owlv2'))
    assert '0.6 GB' in dpg.get_value('analyse_detector_note')


def test_track_shows_counts_model_estimate_note_and_plays_the_overlay(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path)
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', 'car, bus')
    ui.dpg.set_value('analyse_every_n_input', 3)
    seen = []
    real = runner.start_track_run
    monkeypatch.setattr(analyse_page, 'start_track_run', lambda cfg: seen.append(cfg) or real(cfg))
    dispatch(ui, 'analyse_track_button')
    assert ui.dpg.is_item_shown('analyse_active_indicator')
    assert ui.dpg.get_item_configuration('analyse_track_stop_button')['enabled'] is True
    assert ui.dpg.get_item_configuration('analyse_predict_stop_button')['enabled'] is False
    wait_for_action(ui)
    cfg = seen[0]
    assert cfg.prompts == ['car', 'bus'] and cfg.every_n_frames == 3 and cfg.detector == 'grounding_dino'
    assert analyse.threshold_flags(cfg.detector, cfg.track_score) == []
    assert cfg.out.parent == tmp_path / 'out' / 'analyse-video'
    results = text_under(ui, 'analyse_track_results')
    for expected in ('car', 'bus', '0.1 s', '0.9 s', 'fake/detector', 'not your trained model', 'tracker estimates',
                     'not measured accuracy'):
        assert expected in results
    assert 'No tracks found' not in results
    assert analyse_page._player.path == cfg.out / clip.stem / 'overlay.mp4'
    assert 'overlay' in ui.dpg.get_value('analyse_showing_text')
    dispatch(ui, 'analyse_show_original_button')
    assert analyse_page._player.path == clip
    assert 'your video' in ui.dpg.get_value('analyse_showing_text')
    dispatch(ui, 'analyse_show_overlay_button')
    assert analyse_page._player.path.name == 'overlay.mp4'
    assert not ui.gc.list_runs(tmp_path / 'out')


def test_track_with_nothing_found_is_not_shown_as_a_failure(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path, 'empty')
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', 'unicorn')
    dispatch(ui, 'analyse_track_button')
    wait_for_action(ui)
    message_color = ui.dpg.get_item_configuration('analyse_track_message')['color']
    assert 'no tracks found' in ui.dpg.get_value('analyse_track_message')
    assert [round(c * 255) if c <= 1 else round(c) for c in message_color[:3]] != list(ui.ERROR)
    assert ui.dpg.get_value('status_text') == 'Tracking complete: no tracks found'
    results = text_under(ui, 'analyse_track_results')
    assert 'No tracks found' in results and 'lower score threshold' in results and 'different prompt' in results
    assert 'unicorn' in results


def test_track_failure_shows_the_last_log_lines(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path, 'crash')
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    wait_for_action(ui)
    message = ui.dpg.get_value('analyse_track_message')
    assert 'exited with code 3' in message and 'out of memory' in message
    assert not text_under(ui, 'analyse_track_results')
    assert analyse_page._player.path == clip  # the preview is untouched


def test_track_finishing_without_a_summary_is_an_error(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path, 'no_summary')
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    wait_for_action(ui)
    assert 'summary.json is missing' in ui.dpg.get_value('analyse_track_message')


def test_stop_cancels_and_shows_no_result(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path, 'hang')
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    deadline = time.monotonic() + 30
    state = analyse_page._active.handle._process_state
    while state.process is None and time.monotonic() < deadline:
        analyse_page.poll()
        time.sleep(.02)
    dispatch(ui, 'analyse_track_stop_button')
    wait_for_action(ui)
    assert 'Stopped' in ui.dpg.get_value('analyse_track_message')
    assert not text_under(ui, 'analyse_track_results')
    assert ui.dpg.get_item_configuration('analyse_track_button')['enabled'] is True


def test_second_action_cannot_start_while_one_runs(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path, 'hang')
    open_page(ui, tmp_path, video=clip, run=True)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    active = analyse_page._active
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: pytest.fail('must not launch'))
    analyse_page.on_predict_clicked()
    assert analyse_page._active is active


def test_compact_layout_handles_the_page(ui):
    ui.on_nav_selected(user_data='analyse')
    ui.update_layout(900, 640)
    assert ui.dpg.get_item_configuration(analyse_page.VIDEO_INPUT)['width'] == -1
    ui.update_layout(1320, 920)
    assert ui.dpg.get_item_configuration(analyse_page.VIDEO_INPUT)['width'] == -280


# ---------------------------------------------------------------------------
# Runs whose recorded options differ, and what infer.py warned about
# ---------------------------------------------------------------------------

def add_eval_args(run: Path, **changes) -> None:
    train = json.loads((run / 'train_output' / 'cli_arguments.json').read_text(encoding='utf-8'))
    folder = run / 'eval_output'
    folder.mkdir()
    (folder / 'cli_arguments.json').write_text(json.dumps({**train, **changes}), encoding='utf-8')


def test_matching_recorded_settings_raise_nothing(tmp_path):
    run_dir = video_run(tmp_path)
    add_eval_args(run_dir)
    run = gc.read_run_results(run_dir)
    assert analyse.run_problems(run) == [] and analyse.run_warnings(run) == []


def test_model_settings_that_disagree_stop_predict(tmp_path):
    run_dir = video_run(tmp_path, model_version='resnet18')
    add_eval_args(run_dir, pooling='attention')
    problems = analyse.run_problems(gc.read_run_results(run_dir))
    assert len(problems) == 1 and "pooling: 'mean' in training, 'attention' in evaluation" in problems[0]


def test_clip_settings_that_disagree_only_warn(tmp_path):
    run_dir = video_run(tmp_path)
    add_eval_args(run_dir, num_frames=16)
    train = run_dir / 'train_output' / 'cli_arguments.json'
    train.write_text(json.dumps({**json.loads(train.read_text(encoding='utf-8')), 'num_frames': 8}), encoding='utf-8')
    run = gc.read_run_results(run_dir)
    assert analyse.run_problems(run) == []
    assert any('num_frames: 8 in training, 16 in evaluation' in w for w in analyse.run_warnings(run))


@pytest.mark.parametrize('field', ['model_version', 'num_classes'])
def test_run_missing_a_model_setting_is_not_rebuilt_from_defaults(tmp_path, field):
    run_dir = video_run(tmp_path)
    args = run_dir / 'train_output' / 'cli_arguments.json'
    saved = json.loads(args.read_text(encoding='utf-8'))
    del saved[field]
    args.write_text(json.dumps(saved), encoding='utf-8')
    assert any(field in p for p in analyse.run_problems(gc.read_run_results(run_dir)))


def test_regression_run_needs_no_class_count_or_names(tmp_path):
    run_dir = video_run(tmp_path, mode='reg')
    args = run_dir / 'train_output' / 'cli_arguments.json'
    saved = json.loads(args.read_text(encoding='utf-8'))
    del saved['num_classes']
    args.write_text(json.dumps(saved), encoding='utf-8')
    run = gc.read_run_results(run_dir)
    run.label_mapping = None
    assert analyse.run_problems(run) == [] and analyse.run_warnings(run) == []


def test_class_names_that_do_not_fit_the_class_count_are_flagged(tmp_path):
    run = gc.read_run_results(video_run(tmp_path))
    run.label_mapping = {'a': 0, 'b': 1, 'c': 2}
    assert any('3 class names for 2' in w for w in analyse.run_warnings(run))
    run.label_mapping = None
    assert any('no class names' in w for w in analyse.run_warnings(run))


def test_native_model_is_not_described_with_a_pooling_it_ignores(tmp_path):
    assert 'pooling' not in analyse.describe_run(gc.read_run_results(video_run(tmp_path)))
    frames = video_run(tmp_path, name='deeptune-2', model_version='resnet18')
    assert 'mean pooling' in analyse.describe_run(gc.read_run_results(frames))


def test_infer_warnings_are_read_from_what_the_real_code_prints(tmp_path):
    """Real warning functions, so a reworded message fails here and not silently on the page."""
    from evaluators.error_model import load_error_model_near
    from src.video.multi_clip import error_model_clips_mismatch
    lines = []
    (tmp_path / 'error_model.json').write_text('{}', encoding='utf-8')
    checkpoint = tmp_path / 'model_weights.pth'
    assert error_model_clips_mismatch(checkpoint, 3, warn=lines.append)
    assert load_error_model_near(checkpoint, 'cls', warn=lines.append) is None
    assert len(lines) == 2
    log = 'Loaded 1 sample(s)\n' + '\n'.join(lines) + '\nInference complete.\n'
    assert analyse.prediction_log_warnings(log) == lines
    assert analyse.prediction_log_warnings('Loaded 1 sample(s)\nInference complete.\n') == []


@pytest.mark.parametrize('log,expected', [
    ('RuntimeError: CUDA out of memory. Tried to allocate 2 GB', 'ran out of memory'),
    ("OSError: We couldn't connect to 'https://huggingface.co'", 'network connection'),
    ('huggingface_hub.errors.LocalEntryNotFoundError: cannot find', 'network connection'),
    ('OSError: [Errno 28] No space left on device', 'disk is full'),
    ('RuntimeError: Error(s) in loading state_dict: size mismatch for fc.weight', 'does not fit'),
    ('ValueError: something else', ''),
])
def test_failure_hint(log, expected):
    assert (expected in analyse.failure_hint(log)) and bool(analyse.failure_hint(log)) == bool(expected)


def test_predict_is_stopped_when_recorded_model_settings_disagree(ui, tmp_path, clip, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: pytest.fail('must not launch'))
    add_eval_args(video_run(tmp_path), model_version='resnet18')
    open_page(ui, tmp_path, video=clip)
    assert 'disagree' in ui.dpg.get_value('analyse_run_info')
    dispatch(ui, 'analyse_predict_button')
    assert 'model_version' in ui.dpg.get_value('analyse_predict_message')


def test_checkpoint_deleted_after_the_run_was_chosen_stops_predict(ui, tmp_path, clip, monkeypatch):
    monkeypatch.setattr(analyse_page, 'start_infer_run', lambda cfg: pytest.fail('must not launch'))
    open_page(ui, tmp_path, video=clip, run=True)
    (tmp_path / 'out' / 'deeptune-1' / 'train_output' / 'model_weights.pth').unlink()
    dispatch(ui, 'analyse_predict_button')
    assert 'checkpoint is missing' in ui.dpg.get_value('analyse_predict_message')


def test_prediction_and_run_survive_the_label_changing_when_the_page_reopens(ui, tmp_path, clip, monkeypatch):
    fake_infer(monkeypatch, tmp_path)
    first = video_run(tmp_path, name='deeptune-1')
    second = video_run(tmp_path, name='deeptune-2')
    os.utime(second, (1_000_000_000, 1_000_000_000))
    open_page(ui, tmp_path, video=clip)
    assert 'deeptune-1' in ui.dpg.get_value('analyse_run_combo')
    dispatch(ui, 'analyse_predict_button')
    wait_for_action(ui)
    os.utime(first, (1_100_000_000, 1_100_000_000))  # the same folder, so a different label
    ui.on_nav_selected(user_data='overview')
    ui.on_nav_selected(user_data='analyse')
    assert 'deeptune-1' in ui.dpg.get_value('analyse_run_combo')
    assert analyse_page._selected_run.run_dir == first
    assert 'Predicted class: run' in text_under(ui, 'analyse_predict_results')


FAKE_INFER_WARNS = FAKE_INFER.replace(
    "print('Inference complete.', flush=True)",
    "print('Warning: the error model (x) was fitted on 1 clip(s) per video, but this run averages 3; '\n"
    "      'expected_error may be miscalibrated.', flush=True)\nprint('Inference complete.', flush=True)")


def test_infer_warning_reaches_the_page_beside_the_result(ui, tmp_path, clip, monkeypatch):
    script = tmp_path / 'fake_infer.py'
    script.write_text(FAKE_INFER_WARNS, encoding='utf-8')
    monkeypatch.setattr(runner, 'INFER_SCRIPT', script)
    open_page(ui, tmp_path, video=clip, run=True)
    dispatch(ui, 'analyse_predict_button')
    wait_for_action(ui)
    assert 'may be miscalibrated' in text_under(ui, 'analyse_predict_results')


def test_failed_predict_says_why_in_plain_words(ui, tmp_path, clip, monkeypatch):
    script = tmp_path / 'bad_infer.py'
    script.write_text('import sys; print("RuntimeError: size mismatch for fc.weight", flush=True); sys.exit(1)',
                      encoding='utf-8')
    monkeypatch.setattr(runner, 'INFER_SCRIPT', script)
    open_page(ui, tmp_path, video=clip, run=True)
    dispatch(ui, 'analyse_predict_button')
    wait_for_action(ui)
    message = ui.dpg.get_value('analyse_predict_message')
    assert 'does not fit the model' in message and 'exited with code 1' in message and 'size mismatch' in message
    assert not text_under(ui, 'analyse_predict_results')


def test_a_new_track_run_drops_the_old_overlay(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path)
    open_page(ui, tmp_path, video=clip)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    wait_for_action(ui)
    assert analyse_page._player.path.name == 'overlay.mp4'
    fake_track(monkeypatch, tmp_path, 'crash')
    dispatch(ui, 'analyse_track_button')
    wait_for_action(ui)
    assert analyse_page._player.path == clip
    assert ui.dpg.get_item_configuration('analyse_show_overlay_button')['enabled'] is False


def test_track_button_discloses_a_download_before_it_starts(ui, tmp_path, monkeypatch):
    monkeypatch.delenv('HF_HOME', raising=False)
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path))
    ui.on_nav_selected(user_data='analyse')
    assert '0.7 GB' in ui.dpg.get_item_label('analyse_track_button')
    assert str(tmp_path) in ui.dpg.get_value('analyse_detector_note')
    ui.dpg.set_value('analyse_detector_combo', 'OWLv2')
    dispatch(ui, 'analyse_detector_combo', 'OWLv2')
    assert '0.6 GB' in ui.dpg.get_item_label('analyse_track_button')
    weights = tmp_path / 'models--google--owlv2-base-patch16-ensemble' / 'snapshots' / 'abc'
    weights.mkdir(parents=True)
    (weights / 'model.safetensors').write_bytes(b'x')
    ui.on_nav_selected(user_data='analyse')
    assert ui.dpg.get_item_label('analyse_track_button') == 'Find, track and count'
    assert 'no download is expected' in ui.dpg.get_value('analyse_detector_note')


# ---------------------------------------------------------------------------
# A real tiny run: deeptune.py's files, infer.py's files and what the page reads
# ---------------------------------------------------------------------------

def test_real_run_layout_and_real_infer_reach_the_page_logic(tmp_path, monkeypatch, capsys):
    """Nothing faked but the model size: a dummy 3D model trained for one epoch on CPU."""
    pytest.importorskip('av')
    import torchvision

    import deeptune
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import infer
    import inference.video as infer_mod
    import trainers.video.train as train_mod
    from datasets.video_synth import translating_clip

    small = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
    for module in (train_mod, eval_mod, embed_mod, infer_mod):
        monkeypatch.setattr(module, 'transformations', small)
    versions = dict(gc.MODEL_VERSIONS_BY_MODALITY)
    versions['video'] = [*versions['video'], 'dummy_video3d']
    monkeypatch.setattr(gc, 'MODEL_VERSIONS_BY_MODALITY', versions)

    rows = [{'videos': str(translating_clip(tmp_path / f'{name}_{i}.mp4', 10, dx, 0, size=32, seed=i)), 'labels': name}
            for name, dx in (('left', -2), ('right', 2)) for i in range(6)]
    pd.DataFrame(rows).to_csv(tmp_path / 'clips.csv', index=False)
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'dummy_video3d', '--num_epochs', '1',
                   '--batch_size', '4', '--num_frames', '2', '--added_layers', '1', '--embed_size', '16',
                   '--fixed-seed', '--df', str(tmp_path / 'clips.csv'), '--raw-data'])

    choices = analyse.list_video_runs(out)
    assert list(p.parent.name for p in choices.runs.values()) == ['out'] and choices.hidden == 0
    run = gc.read_run_results(next(iter(choices.runs.values())))
    assert analyse.run_problems(run) == [] and analyse.run_warnings(run) == []
    clip = Path(rows[0]['videos'])

    def predict(name):
        target = analyse.timestamped_folder(tmp_path / name)
        capsys.readouterr()
        assert infer.main(gc.build_infer_argv(gc.infer_config_from_run(run, clip, target))) == 0
        log = capsys.readouterr().out
        return analyse.summarise_prediction(gc.read_inference_results(target), run, clip), log

    view, log = predict('fitted')
    assert not view.error
    assert view.answer in ('Predicted class: left', 'Predicted class: right')
    assert view.confidence.startswith('Confidence: ')
    # 12 clips are too few to fit on, so the real file says constant, and the page must say what that means.
    assert 'run-wide validation rate, the same for every video' in view.expected_error
    assert 'chance' not in view.expected_error
    assert analyse.prediction_log_warnings(log) == []

    # An error model fitted on other weights: refused, said on the page, and no number.
    error_model = Path(run.checkpoint_path).parent / 'error_model.json'
    saved = json.loads(error_model.read_text(encoding='utf-8'))
    saved['fit_info']['weights_sha256'] = 'f' * 64
    error_model.write_text(json.dumps(saved), encoding='utf-8')
    view, log = predict('refused')
    assert 'could not be used' in view.expected_error and '%' not in view.expected_error
    assert any('different weights' in line for line in analyse.prediction_log_warnings(log))

    # The file gone: the plain statement, and still no number.
    error_model.unlink()
    view, _ = predict('none')
    assert not view.error and 'This run has no error model' in view.expected_error
    assert view.answer.startswith('Predicted class: ')


# ---------------------------------------------------------------------------
# Review V-041
# ---------------------------------------------------------------------------

def moving_box(frame: int, speed: float = 12.0, score: float = 0.8):
    from spatial.detect import Detection
    return [Detection((5.0 + speed * frame, 5.0, 45.0 + speed * frame, 45.0), score, 'x')] if frame >= 1 else []


def count_through_real_cli(tmp_path, clip, source, name, track_score=None, every_n=1):
    """The real pipeline and tracker with the page's own flags. Only the detector is scripted."""
    import track
    from spatial.detect import FakeDetector
    out = tmp_path / name
    cfg = analyse.TrackConfig(video=clip, prompts=['x'], out=out, every_n_frames=every_n, track_score=track_score)
    assert track.cli(analyse.build_track_argv(cfg), detector=FakeDetector(source)) == 0
    return analyse.read_track_results(out, clip, 'grounding_dino')


def test_page_default_frames_step_counts_a_moving_object_that_enters_late(ui, tmp_path):
    clip = write_clip(tmp_path / 'long.mp4', frames=40, size=(640, 48))
    source = {i: moving_box(i) for i in range(40)}
    step = ui.dpg.get_value('analyse_every_n_input')
    assert step == 1
    assert count_through_real_cli(tmp_path, clip, source, 'page', every_n=step).labels[0].count == 1
    # What the old default did to the same video.
    assert count_through_real_cli(tmp_path, clip, source, 'five', every_n=5).labels[0].count == 0


def test_a_step_above_one_is_disclosed_before_and_after_the_run(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path)
    open_page(ui, tmp_path, video=clip)
    assert analyse.FRAMES_STEP_NOTE in text_under(ui, 'page_analyse')
    assert 'missed' in analyse.FRAMES_STEP_NOTE and 'split' in analyse.FRAMES_STEP_NOTE
    results = analyse.TrackResults(video_dir=tmp_path, video='a.mp4', labels=[], every_n_frames=5, num_frames=10)
    analyse_page._render_tracking(results)
    assert analyse.FRAMES_STEP_NOTE in text_under(ui, 'analyse_track_results')
    analyse_page._render_tracking(analyse.TrackResults(video_dir=tmp_path, video='a.mp4', every_n_frames=1, num_frames=10))
    assert analyse.FRAMES_STEP_NOTE not in text_under(ui, 'analyse_track_results')


@pytest.mark.parametrize('detector', ['grounding_dino', 'owlv2'])
@pytest.mark.parametrize('score', [0.05, 0.1, 0.15, 0.25, 0.3, 0.45, 0.6, 0.9])
def test_box_from_frame_2_is_tracked_at_a_threshold_equal_to_its_score(tmp_path, detector, score):
    import track
    from spatial.detect import Detection
    from spatial.track import ByteTracker
    cfg = analyse.TrackConfig(video=Path('a.mp4'), prompts=['x'], out=tmp_path, detector=detector, track_score=score)
    tracker = ByteTracker(frame_rate=30, **track.tracker_kwargs(parse_with_real_track_cli(analyse.build_track_argv(cfg))))
    box = Detection((10.0, 10.0, 50.0, 50.0), score, 'x')
    seen = set()
    for frame in range(50):
        seen |= {o.track_id for o in tracker.update([box] if frame >= 1 else [])}
    assert len(seen) == 1


def test_fusing_stays_the_default_for_other_callers_and_is_recorded(tmp_path, clip):
    import track
    from spatial.detect import FakeDetector
    from spatial.track import ByteTracker
    assert ByteTracker().fuse_score is True
    assert track.cli(['--video', str(clip), '--prompt', 'x', '--out', str(tmp_path / 'o')], detector=FakeDetector({})) == 0
    record = json.loads((tmp_path / 'o' / 'track_arguments.json').read_text(encoding='utf-8'))
    assert record['tracker_settings']['fuse_score'] is False


def test_advice_for_a_weak_box_is_true_when_followed(tmp_path):
    clip = write_clip(tmp_path / 'still.mp4', frames=30)
    source = {i: moving_box(i, speed=0.0, score=0.306) for i in range(30)}
    first = count_through_real_cli(tmp_path, clip, source, 'default')
    assert first.total_tracks == 0
    assert 'best score 0.31' in first.no_tracks_message and 'at least 0.45' in first.no_tracks_message
    assert 'Set the score threshold to 0.30 or lower' in first.no_tracks_message
    assert count_through_real_cli(tmp_path, clip, source, 'followed', track_score=0.30).total_tracks == 1


def test_message_does_not_advise_a_lower_threshold_that_cannot_help(tmp_path):
    """A strong box seen once, and a box too fast to match: the threshold is not the problem."""
    clip = write_clip(tmp_path / 'once.mp4', frames=30)
    for name, source in (('once', {3: moving_box(3, speed=0.0)}), ('fast', {i: moving_box(i, speed=40.0) for i in range(30)})):
        results = count_through_real_cli(tmp_path, clip, source, name)
        assert results.total_tracks == 0
        assert 'A lower threshold will not help' in results.no_tracks_message
        assert 'needs a score of at least' not in results.no_tracks_message
        assert 'at 0.45' in results.no_tracks_message
        assert count_through_real_cli(tmp_path, clip, source, f'{name}-low', track_score=0.1).total_tracks == 0


def test_message_for_no_boxes_names_the_lowest_score_kept(tmp_path):
    clip = write_clip(tmp_path / 'empty.mp4', frames=6)
    assert 'below 0.20' in count_through_real_cli(tmp_path, clip, {}, 'none').no_tracks_message


def fitted_error_model(run_dir: Path, **fit) -> None:
    (run_dir / 'train_output' / 'error_model.json').write_text(json.dumps({'method': fit.pop('method'), 'fit_info': fit}),
                                                               encoding='utf-8')


def stated(run_dir: Path, **columns):
    frame = {'predicted_label': ['run'], 'confidence': [0.9], 'expected_error': [0.12], **columns}
    return analyse.summarise_prediction(prediction(frame), gc.read_run_results(run_dir), Path('a.mp4'))


def test_isotonic_estimate_is_worded_as_what_validation_showed(tmp_path):
    run_dir = video_run(tmp_path)
    fitted_error_model(run_dir, method='isotonic', n_samples=240, fallback_reason=None)
    view = stated(run_dir)
    assert view.expected_error == 'Expected error: on validation clips with similar confidence, about 12.0% were wrong (240 clips).'
    assert any('never trained on' in note for note in view.notes)


def test_constant_estimate_gives_the_real_count_and_not_the_smoothed_value_as_observed(tmp_path):
    run_dir = video_run(tmp_path)
    fitted_error_model(run_dir, method='constant', n_samples=12, n_errors=0, fallback_reason='only 12 fit samples (< 30)')
    view = stated(run_dir, expected_error=[0.0714])
    assert '0 of 12 validation clips were wrong' in view.expected_error and '7.1%' in view.expected_error
    assert '7.1% were wrong' not in view.expected_error
    assert 'the same for every video' in view.expected_error


def test_constant_estimate_without_a_count_does_not_claim_an_observed_share(tmp_path):
    run_dir = video_run(tmp_path)
    fitted_error_model(run_dir, method='constant', n_samples=12, fallback_reason='only 12 fit samples (< 30)')
    view = stated(run_dir)
    assert 'were wrong' not in view.expected_error


def test_constant_estimate_says_it_is_the_same_for_every_video(tmp_path):
    run_dir = video_run(tmp_path)
    fitted_error_model(run_dir, method='constant', n_samples=12, fallback_reason='only 12 fit samples (< 30)')
    view = stated(run_dir)
    assert 'run-wide validation rate, the same for every video' in view.expected_error
    assert '12 clips' in view.expected_error and 'only 12 fit samples' in view.expected_error
    assert 'similar confidence' not in view.expected_error


def test_estimate_is_withheld_when_clip_settings_do_not_match(tmp_path):
    run_dir = video_run(tmp_path)
    fitted_error_model(run_dir, method='isotonic', n_samples=240, test_clips=1)
    args = run_dir / 'train_output' / 'cli_arguments.json'
    args.write_text(json.dumps({**json.loads(args.read_text(encoding='utf-8')), 'test_clips': 3}), encoding='utf-8')
    text = stated(run_dir).expected_error
    assert 'withheld' in text and 'fitted on 1 clip(s)' in text and 'averages 3' in text
    assert '12' not in text


def test_estimate_is_shown_when_clip_counts_match(tmp_path):
    run_dir = video_run(tmp_path)
    fitted_error_model(run_dir, method='isotonic', n_samples=240, test_clips=3)
    args = run_dir / 'train_output' / 'cli_arguments.json'
    args.write_text(json.dumps({**json.loads(args.read_text(encoding='utf-8')), 'test_clips': 3}), encoding='utf-8')
    assert 'about 12.0%' in stated(run_dir).expected_error


def test_changing_the_video_during_a_run_cannot_mislabel_its_result(ui, tmp_path, clip, monkeypatch):
    fake_track(monkeypatch, tmp_path)
    other = write_clip(tmp_path / 'video_B.mp4')
    open_page(ui, tmp_path, video=clip, run=True)
    ui.dpg.set_value('analyse_prompt_input', 'car')
    dispatch(ui, 'analyse_track_button')
    for item in (analyse_page.VIDEO_INPUT, 'analyse_browse_button', 'analyse_load_button', 'analyse_run_combo',
                 'analyse_refresh_button'):
        assert ui.dpg.get_item_configuration(item)['enabled'] is False, item
    # The callbacks refuse too: a click can be queued, and a file dialog can already be open.
    ui.dpg.set_value(analyse_page.VIDEO_INPUT, str(other))
    dispatch(ui, 'analyse_load_button')
    analyse_page.on_video_file_selected(0, {'file_path_name': str(other), 'selections': {other.name: str(other)}})
    run_before = analyse_page._selected_run
    dispatch(ui, 'analyse_run_combo')
    analyse_page.on_refresh_runs()
    assert analyse_page._selected_run is run_before
    assert analyse_page._original == clip
    wait_for_action(ui)
    assert f'Video: {clip.name}' in text_under(ui, 'analyse_track_results')
    assert analyse_page._player.path.parent.name == clip.stem
    for item in (analyse_page.VIDEO_INPUT, 'analyse_load_button', 'analyse_run_combo', 'analyse_refresh_button'):
        assert ui.dpg.get_item_configuration(item)['enabled'] is True, item


def test_train_settings_rebuild_the_model_when_the_eval_record_lacks_them(tmp_path):
    run_dir = video_run(tmp_path, model_version='resnet18')
    args = run_dir / 'train_output' / 'cli_arguments.json'
    saved = {**json.loads(args.read_text(encoding='utf-8')), 'pooling': 'single_frame', 'num_frames': 16}
    args.write_text(json.dumps(saved), encoding='utf-8')
    evaluated = {k: v for k, v in saved.items() if k not in ('pooling', 'num_frames')}
    (run_dir / 'eval_output').mkdir()
    (run_dir / 'eval_output' / 'cli_arguments.json').write_text(json.dumps(evaluated), encoding='utf-8')
    run = gc.read_run_results(run_dir)
    assert 'pooling' not in run.cli_arguments  # the eval record, which is the bug's source
    assert analyse.run_problems(run) == []
    argv = gc.build_infer_argv(gc.infer_config_from_run(run, Path('a.mp4'), tmp_path / 'o'))
    assert argv[argv.index('--num_frames') + 1] == '16' and argv[argv.index('--pooling') + 1] == 'single_frame'
    assert 'single_frame pooling' in analyse.describe_run(run)


def test_settings_that_do_not_change_the_weights_are_not_refused(tmp_path):
    native = video_run(tmp_path, name='deeptune-native')
    add_eval_args(native, pooling='attention')
    assert analyse.run_problems(gc.read_run_results(native)) == []
    plain = video_run(tmp_path, name='deeptune-plain')
    add_eval_args(plain, freeze_backbone=True)
    assert analyse.run_problems(gc.read_run_results(plain)) == []


def test_freeze_backbone_is_still_refused_for_a_peft_run(tmp_path):
    run_dir = video_run(tmp_path)
    args = run_dir / 'train_output' / 'cli_arguments.json'
    args.write_text(json.dumps({**json.loads(args.read_text(encoding='utf-8')), 'use_peft': True, 'freeze_backbone': False}),
                    encoding='utf-8')
    add_eval_args(run_dir, freeze_backbone=True)
    assert any('freeze_backbone' in p for p in analyse.run_problems(gc.read_run_results(run_dir)))


def test_frame_model_with_no_recorded_pooling_is_refused(tmp_path):
    run_dir = video_run(tmp_path, model_version='resnet18')
    args = run_dir / 'train_output' / 'cli_arguments.json'
    saved = json.loads(args.read_text(encoding='utf-8'))
    del saved['pooling']
    args.write_text(json.dumps(saved), encoding='utf-8')
    assert any('pooling' in p for p in analyse.run_problems(gc.read_run_results(run_dir)))


def test_run_that_records_no_frame_count_says_what_is_used(tmp_path):
    run_dir = video_run(tmp_path)
    args = run_dir / 'train_output' / 'cli_arguments.json'
    saved = json.loads(args.read_text(encoding='utf-8'))
    del saved['num_frames']
    args.write_text(json.dumps(saved), encoding='utf-8')
    assert any('so 8 are used' in w for w in analyse.run_warnings(gc.read_run_results(run_dir)))


def test_argv_of_a_run_whose_records_agree_is_unchanged(tmp_path):
    run_dir = video_run(tmp_path, model_version='resnet18')
    add_eval_args(run_dir)
    run = gc.read_run_results(run_dir)
    assert gc.checkpoint_arguments(run) == run.cli_arguments


def test_other_modalities_keep_their_eval_record(tmp_path):
    checkpoint = tmp_path / 'train_output' / 'weights.pth'
    checkpoint.parent.mkdir()
    (checkpoint.parent / 'cli_arguments.json').write_text('{"modality": "images", "model_version": "a"}', encoding='utf-8')
    run = gc.RunResults(run_dir=tmp_path, checkpoint_path=checkpoint, cli_arguments={'modality': 'images', 'model_version': 'b'})
    assert gc.checkpoint_arguments(run)['model_version'] == 'b'


@pytest.mark.parametrize('label', ['007', '1', 'NA', 'walk'])
def test_class_name_is_shown_exactly_as_written(tmp_path, label):
    out = tmp_path / 'infer'
    out.mkdir()
    pd.DataFrame({'sample_id': [0], 'predicted_class': [0], 'predicted_label': [label], 'confidence': [0.5]}).to_csv(
        out / 'predictions.csv', index=False)
    run = gc.read_run_results(video_run(tmp_path))
    view = analyse.summarise_prediction(gc.read_inference_results(out), run, Path('a.mp4'))
    assert view.answer == f'Predicted class: {label}'
    assert view.confidence == 'Confidence: 50.0%'


def test_empty_cells_stay_missing_in_numeric_columns(tmp_path):
    (tmp_path / 'predictions.csv').write_text('sample_id,predicted_value,confidence\n0,2.5,\n', encoding='utf-8')
    frame = gc.read_inference_results(tmp_path).predictions
    assert frame['predicted_value'].iloc[0] == 2.5 and pd.isna(frame['confidence'].iloc[0])


def test_empty_label_falls_back_to_the_class_number(tmp_path):
    (tmp_path / 'predictions.csv').write_text('sample_id,predicted_class,predicted_label,confidence\n0,1,,0.5\n', encoding='utf-8')
    run = gc.read_run_results(video_run(tmp_path))
    assert analyse.summarise_prediction(gc.read_inference_results(tmp_path), run, Path('a.mp4')).answer == 'Predicted class: 1'


def test_loading_message_mentions_a_download_only_when_one_will_happen(tmp_path, monkeypatch):
    monkeypatch.delenv('HF_HOME', raising=False)
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path))
    assert '0.6 GB' in analyse.loading_message('owlv2')
    weights = tmp_path / 'models--google--owlv2-base-patch16-ensemble' / 'snapshots' / 'abc'
    weights.mkdir(parents=True)
    (weights / 'model.safetensors').write_bytes(b'x')
    message = analyse.loading_message('owlv2')
    assert 'GB' not in message and 'Loading OWLv2' in message


def test_refresh_reads_the_run_record_again(ui, tmp_path, clip):
    open_page(ui, tmp_path, video=clip, run=True)
    assert 'r3d_18' in ui.dpg.get_value('analyse_run_info')
    args = tmp_path / 'out' / 'deeptune-1' / 'train_output' / 'cli_arguments.json'
    args.write_text(json.dumps({**json.loads(args.read_text(encoding='utf-8')), 'model_version': 'mc3_18'}), encoding='utf-8')
    dispatch(ui, 'analyse_refresh_button')
    assert 'mc3_18' in ui.dpg.get_value('analyse_run_info')
    assert analyse_page._selected_run.cli_arguments['model_version'] == 'mc3_18'

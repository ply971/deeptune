"""Test page timeline controls: checks before launch, argv, and reading a timeline run's outputs (no torch)."""
import ast
import json
from pathlib import Path

import pandas as pd

from desktop import cli_options as co
from desktop import config as gc
from desktop import tools


def _cfg(tmp_path, **options):
    data = tmp_path / 'clips'
    data.mkdir(exist_ok=True)
    return gc.InferenceConfig(modality='video', run_dir=tmp_path, checkpoint_path=tmp_path / 'w.pth', data=data,
                              out=tmp_path / 'out', options=options)


def test_timeline_settings_without_the_timeline_are_refused(tmp_path):
    issues = gc.validate_infer_config(_cfg(tmp_path, window_sec=3.0))
    assert any('only apply with "Analyse each video as a timeline"' in i for i in issues)
    assert not any('timeline' in i.lower() for i in gc.validate_infer_config(_cfg(tmp_path)))


def test_timeline_settings_are_checked_like_infer_does(tmp_path):
    ok = _cfg(tmp_path, timeline=True, window_sec=2.0, stride_sec=0.5, smooth_windows=5)
    assert gc.validate_infer_config(ok) == []
    for bad, words in (({'stride_sec': 3.0}, 'longer than a window'), ({'smooth_windows': 4}, 'odd number'),
                       ({'min_confidence': 2.0}, 'between 0 and 1'), ({'ground_truth': str(tmp_path / 'no.csv')}, 'not found'),
                       ({'mode': 'reg'}, 'classification run'), ({'skip_static': -1.0}, 'motion threshold')):
        issues = gc.validate_infer_config(_cfg(tmp_path, timeline=True, **bad))
        assert any(words in i for i in issues), (bad, issues)


def test_timeline_argv_sends_the_flags_set(tmp_path):
    values = {'timeline': True, 'window_sec': 1.5, 'background_label': ['background', 'idle'], 'skip_static': 0.5}
    argv = co.options_argv(co.INFER_OPTIONS, values)
    assert '--timeline' in argv
    assert argv[argv.index('--window_sec') + 1] == '1.5'
    i = argv.index('--background_label')
    assert argv[i + 1:i + 3] == ['background', 'idle']
    assert '--stride_sec' not in argv  # defaults are not sent
    import infer
    parsed = infer.build_parser().parse_args(['--modality', 'video', '--checkpoint', 'w', '--data', 'd', '--out', 'o', *argv])
    assert parsed.timeline and parsed.background_label == ['background', 'idle'] and parsed.skip_static == 0.5


def _timeline_run(out: Path, with_metrics=True):
    out.mkdir(parents=True)
    pd.DataFrame({'video-id': ['007', '007'], 't-start': [0.0, 5.0], 't-end': [3.0, 9.0], 'label': ['jump', 'NA'],
                  'score': [0.9, 0.8]}).to_csv(out / 'segments.csv', index=False)
    (out / 'timeline_run.json').write_text(json.dumps({
        'settings': {'window_sec': 2.0, 'stride_sec': 1.0, 'smooth_windows': 3, 'background': ['none']},
        'videos': [{'video_id': '007', 'duration_sec': 10.0, 'windows': 9, 'scored_windows': 9, 'segments': 2, 'seconds': 1.2}],
        'total_windows': 9, 'scored_windows': 9, 'seconds': 1.2}), encoding='utf-8')
    if with_metrics:
        (out / 'timeline_metrics.json').write_text(json.dumps({
            'detection_map': {'map_per_threshold': {'0.1': 1.0, '0.3': 0.9, '0.5': 0.8, '0.7': 0.5},
                              'averages': {'avg_0.1_0.7': 0.77}, 'n_ground_truth': 2, 'n_predictions': 2},
            'boundary_error': {'mean_seconds': 0.3}}), encoding='utf-8')


def test_results_reader_shows_segments_and_localisation_metrics(tmp_path):
    _timeline_run(tmp_path / 'run')
    results = gc.read_inference_results(tmp_path / 'run')
    assert results.predictions['video-id'].tolist() == ['007', '007']  # names stay text
    assert results.predictions['label'].tolist() == ['jump', 'NA']
    assert results.metrics['mAP@tIoU 0.5'] == 0.8 and results.metrics['boundary error (s, tIoU 0.5)'] == 0.3
    assert results.warnings == []
    _timeline_run(tmp_path / 'plain', with_metrics=False)
    assert gc.read_inference_results(tmp_path / 'plain').metrics is None


def test_reports_list_the_timeline_settings_and_videos(tmp_path):
    _timeline_run(tmp_path / 'run')
    sections = tools.inference_reports(tmp_path / 'run')
    timeline = next(s for s in sections if s.title == 'Timeline')
    rows = dict(timeline.rows)
    assert rows['window_sec'] == '2.0' and rows['windows scored'] == '9 of 9'
    assert timeline.table['video_id'].tolist() == ['007']
    (tmp_path / 'bad').mkdir()
    (tmp_path / 'bad' / 'timeline_run.json').write_text('{"settings": 3, "videos": 5}', encoding='utf-8')
    titles = [s.title for s in tools.inference_reports(tmp_path / 'bad')]
    assert any('timeline' in t.lower() for t in titles)  # a malformed record is shown as unreadable, never raises


def test_desktop_modules_still_do_not_import_torch():
    for module in (gc, co, tools):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        names = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert 'torch' not in names, module.__name__

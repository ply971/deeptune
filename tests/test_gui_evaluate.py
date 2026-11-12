"""G2: the Evaluate & Benchmarks page. The video evaluators run as modules from the repository root, the scripts/
benchmarks only from a source checkout, report.py's compare word goes first, and the results view reads what each
tool wrote. Every flag reaching the argv is tests/test_gui_coverage.py's job; this file checks the rest."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from desktop import cli_options as co
from desktop import runner, tools

REPO = Path(__file__).resolve().parents[1]


def _cfg(tool, **values):
    return tools.ToolConfig(tool, values)


# --- how each tool is launched ------------------------------------------------------------------------------------------
def test_evaluators_run_as_modules_from_the_root_and_scripts_by_path(tmp_path):
    cmd, cwd = runner._worker_command(REPO / 'evaluators/video/report.py', '--tool-worker', ['--eval_dir', 'x'],
                                      module='evaluators.video.report', cwd=REPO)
    assert cmd == [sys.executable, '-u', '-m', 'evaluators.video.report', '--eval_dir', 'x'] and cwd == REPO
    cmd, cwd = runner._worker_command(REPO / 'scripts/timeline_untrimmed.py', '--tool-worker', ['--out', 'o'], cwd=REPO)
    assert cmd == [sys.executable, '-u', str(REPO / 'scripts/timeline_untrimmed.py'), '--out', 'o'] and cwd == REPO
    cmd, cwd = runner._worker_command(REPO / 'deeptune.py', '--deeptune-worker', ['--df', 'd'])  # unchanged for the old pages
    assert cmd == [sys.executable, '-u', str(REPO / 'deeptune.py'), '--df', 'd'] and cwd == REPO
    for tool in tools.EVALUATE_TOOLS:
        spec = tools.TOOLS[tool]
        assert (REPO / spec.script).is_file(), spec.script
        assert bool(spec.module) != spec.checkout_only, tool  # evaluators by module; scripts/ only from a checkout
        if spec.module:
            assert Path(*spec.module.split('.')).with_suffix('.py').as_posix() == spec.script


def test_start_tool_run_hands_the_worker_the_module_and_the_root(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, '_worker_process', lambda *args: seen.append(args) or args[6].put(('success', '')))
    (tmp_path / 'gt.csv').write_text('x\n', encoding='utf-8')
    cfg = _cfg('localization_report', ground_truth=str(tmp_path / 'gt.csv'), predictions=str(tmp_path / 'gt.csv'),
               out=str(tmp_path / 'loc'))
    handle = runner.start_tool_run(cfg)
    handle.thread.join(timeout=10)
    script, sentinel, _, _, argv, _, _, _, tool, out, module, cwd = seen[0]
    assert (script, sentinel, tool, module, cwd) == (REPO / 'evaluators/video/localization_report.py', '--tool-worker',
                                                     'evaluators.video.localization_report', 'evaluators.video.localization_report', REPO)
    assert out == tmp_path / 'loc' and out.is_dir()
    assert argv == ['--ground_truth', str(tmp_path / 'gt.csv'), '--predictions', str(tmp_path / 'gt.csv'), '--out', str(tmp_path / 'loc')]


def test_a_packaged_build_refuses_the_checkout_only_benchmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    with pytest.raises(ValueError, match='source checkout'):
        runner.start_tool_run(_cfg('timeline_untrimmed', out=str(tmp_path / 't')))
    assert not (tmp_path / 't').exists()


def test_the_frozen_worker_and_the_packaging_know_every_packaged_tool():
    from desktop import app
    packaged = {spec.module or spec.name for spec in tools.TOOLS.values() if not spec.checkout_only}
    assert set(app._TOOL_WORKERS) == packaged
    spec_text = (REPO / 'packaging' / 'deeptune.spec').read_text(encoding='utf-8')
    assert all(f'"{name}"' in spec_text for name in packaged)


def test_frozen_tool_worker_runs_an_evaluator_by_module_name():
    code = ("import runpy, sys\n"
            "sys.argv = ['app.py', '--tool-worker', 'evaluators.video.localization_report', '--help']\n"
            "try:\n"
            "    runpy.run_path(r'desktop/app.py', run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    assert exc.code in (0, None), exc.code\n"
            "assert 'dearpygui' not in sys.modules\n")
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert '--boundary_tiou' in result.stdout


def test_compare_goes_first_and_writes_no_folder(tmp_path):
    from evaluators.video import report
    for name in ('a', 'b'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'predictions.csv').write_text('x\n', encoding='utf-8')
    cfg = _cfg('eval_report', a=str(tmp_path / 'a'), b=str(tmp_path / 'b'), n_boot=500)
    argv = runner.tool_argv(cfg, leading=['compare'])
    assert argv[0] == 'compare'
    parsed = report.build_parser().parse_args(argv[1:])
    assert (parsed.a, parsed.b, parsed.n_boot) == (tmp_path / 'a', tmp_path / 'b', 500)
    assert tools.validate_tool(cfg, ['compare']) == []
    assert tools.output_folder(cfg, ['compare']) is None
    # The report mode needs its own folder; the compare folders alone are not enough.
    assert tools.validate_tool(cfg) == ['Choose the evaluation folder (a folder with predictions.csv).']


def test_remainder_flags_stay_last_after_extra_and_leading_words():
    cfg = _cfg('timeline_untrimmed', out='o', extra=['--num_workers', '0'], epochs=2)
    assert runner.tool_argv(cfg, ['--x']) == ['--out', 'o', '--epochs', '2', '--x', '--extra', '--num_workers', '0']


def test_a_repeated_flag_is_sent_once_per_value_and_parses_back():
    import importlib.util
    spec = importlib.util.spec_from_file_location('g2_train_cat_detector', REPO / 'scripts' / 'train_cat_detector.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = tools.build_tool_argv(_cfg('train_cat_detector', data='d', out='o', exclude_image=['Abyssinian_1', 'Bengal_2']))
    assert argv[-4:] == ['--exclude-image', 'Abyssinian_1', '--exclude-image', 'Bengal_2']
    assert module.build_parser().parse_args(argv).exclude_image == ['Abyssinian_1', 'Bengal_2']


def test_output_folders(tmp_path):
    weights = tmp_path / 'run' / 'trainval_output_x' / 'model_weights.pth'
    assert tools.output_folder(_cfg('abstention_fit', checkpoint=str(weights))) == weights.parent
    assert tools.output_folder(_cfg('eval_report', eval_dir=str(tmp_path))) == tmp_path
    assert tools.output_folder(_cfg('run_motion_benchmark', out='rel/out')) == REPO / 'rel' / 'out'  # tools run from the root
    assert tools.output_folder(_cfg('localization_report')) is None


# --- checks before a process starts ----------------------------------------------------------------------------------
@pytest.mark.parametrize('tool, values, message', [
    ('run_motion_benchmark', {'task': 'direction', 'hflip': True}, 'flip is refused on the direction task'),
    ('run_motion_benchmark', {'task': 'speed', 'added_layers': 3}, 'Added layers must be 1 or 2'),
    ('run_motion_benchmark', {'task': 'speed', 'n_val': 0}, 'Validation clips must be at least 1'),
    ('abstention_fit', {'split_name': 'Test', 'target_error': 0.1}, 'may not be validation, test or train'),
    ('abstention_fit', {'target_error': 1.5}, 'target error must be between 0 and 1'),
    ('eval_report', {'fit_split': 'test'}, 'may not be fitted on test rows'),
    ('benchmark_mot17_botsort', {'variants': []}, 'Pick at least one: Tracker variants'),
    ('benchmark_mot17_e2e', {'trackers': []}, 'Pick at least one: Trackers'),
    ('timeline_untrimmed', {'window_sec': 0.0}, 'Window (s) must be above 0'),
    ('train_cat_detector', {'epochs': 0}, 'Epochs must be at least 1'),
    ('localization_report', {'boundary_tiou': 0.0}, 'tIoU must be above 0'),
])
def test_refused_settings_are_explained(tool, values, message):
    assert any(message in issue for issue in tools.validate_tool(_cfg(tool, **values))), tools.validate_tool(_cfg(tool, **values))


def test_abstention_needs_a_deeptune_run_and_a_new_name(tmp_path):
    weights = tmp_path / 'model_weights.pth'
    weights.write_bytes(b'w')
    (tmp_path / 'calib.csv').write_text('x\n', encoding='utf-8')
    cfg = _cfg('abstention_fit', checkpoint=str(weights), calib_df=str(tmp_path / 'calib.csv'), target_error=0.1)
    assert any('cli_arguments.json' in issue for issue in tools.validate_tool(cfg))
    (tmp_path / 'cli_arguments.json').write_text('{}', encoding='utf-8')
    assert tools.validate_tool(cfg) == []


def test_a_cat_download_may_name_a_folder_that_does_not_exist_yet(tmp_path):
    missing = str(tmp_path / 'pets')
    assert tools.validate_tool(_cfg('train_cat_detector', data=missing, out='o')) != []
    assert tools.validate_tool(_cfg('train_cat_detector', data=missing, out='o', download=True)) == []


# --- results ----------------------------------------------------------------------------------------------------------
def test_a_benchmark_report_becomes_one_row_per_experiment(tmp_path):
    report = {'dataset': 'MOT17', 'protocol': {'tracker': 'botsort'}, 'experiments': {
        'botsort_FRCNN': {'variant': 'botsort', 'detector': 'FRCNN', 'metrics': {'HOTA': 49.4, 'IDF1': 57.9}, 'sequences': [{'x': 1}]},
        'bytetrack_FRCNN': {'variant': 'bytetrack', 'detector': 'FRCNN', 'metrics': {'HOTA': 46.0, 'IDF1': 52.1}}}}
    (tmp_path / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    sections = tools.tool_sections('benchmark_mot17_botsort', tmp_path)
    assert [s.title for s in sections] == ['report.json', 'report.json: experiments']
    assert ('dataset', 'MOT17') in sections[0].rows and ('protocol / tracker', 'botsort') in sections[0].rows
    table = sections[1].table
    assert list(table['experiment']) == ['botsort_FRCNN', 'bytetrack_FRCNN'] and list(table['HOTA']) == [49.4, 46.0]
    assert 'sequences' not in table.columns


def test_the_motion_and_cat_results_show_their_tables(tmp_path):
    pd.DataFrame({'model': ['r3d_18', 'flow_logreg'], 'accuracy': [0.9, 0.6]}).to_csv(tmp_path / 'results.csv', index=False)
    (tmp_path / 'results.md').write_text('| model | accuracy |\n', encoding='utf-8')
    titles = [s.title for s in tools.tool_sections('run_motion_benchmark', tmp_path)]
    assert titles == ['results.csv', 'results.md']
    (tmp_path / 'report.json').write_text(json.dumps({'best_epoch': 3, 'epochs': [{'epoch': 1, 'ap50': 0.5}, {'epoch': 2, 'ap50': 0.6}]}))
    sections = tools.tool_sections('train_cat_detector', tmp_path)
    assert sections[0].rows == [('best_epoch', '3')]
    assert list(sections[1].table['ap50']) == [0.5, 0.6]


@pytest.mark.parametrize('payload', ['not json', '[1, 2]', '{"experiments": 5}', '{"a": {"b": [1, {"c": 2}]}}'])
def test_the_results_view_never_crashes_on_odd_files(tmp_path, payload):
    (tmp_path / 'report.json').write_text(payload, encoding='utf-8')
    (tmp_path / 'results.csv').write_bytes(b'\xff\xfe\x00bad')
    for tool in ('benchmark_mot17_e2e', 'run_motion_benchmark'):
        assert isinstance(tools.tool_sections(tool, tmp_path), list)


def test_json_rows_flatten_and_stop_at_the_limit():
    rows = tools.json_rows({'a': {'b': 1, 'c': [1, 2]}, 'long': list(range(20)), 'n': None})
    assert rows == [('a / b', '1'), ('a / c', '1, 2'), ('long', '20 values'), ('n', 'Unavailable')]
    assert len(tools.json_rows({str(i): i for i in range(200)}, limit=50)) == 50


# --- end to end: a real localisation report through the real runner --------------------------------------------------
def test_scoring_segments_end_to_end(tmp_path):
    truth = pd.DataFrame({'video-id': ['v1', 'v1', 'v2'], 't-start': [0.0, 5.0, 1.0], 't-end': [2.0, 7.0, 3.0],
                          'label': ['run', 'jump', 'run']})
    prediction = truth.assign(score=[0.9, 0.8, 0.7])
    truth.to_csv(tmp_path / 'gt.csv', index=False)
    prediction.to_csv(tmp_path / 'pred.csv', index=False)
    cfg = _cfg('localization_report', ground_truth=str(tmp_path / 'gt.csv'), predictions=str(tmp_path / 'pred.csv'),
               out=str(tmp_path / 'loc'))
    handle = runner.start_tool_run(cfg)
    handle.thread.join(timeout=180)
    log = ''
    while True:
        try:
            log += handle.log_queue.get_nowait()
        except queue.Empty:
            break
    assert handle.done_queue.get_nowait() == ('success', ''), log
    sections = tools.tool_sections('localization_report', tmp_path / 'loc')
    rows = dict(sections[0].rows)
    assert float(rows['detection / map_per_threshold / 0.5']) == pytest.approx(1.0), rows  # predictions = the truth
    assert [s.title for s in sections][-1] == 'per_class_ap.csv'
    record = json.loads((tmp_path / 'loc' / 'desktop-run.json').read_text(encoding='utf-8'))
    assert record['status'] == 'success'


# --- the page -----------------------------------------------------------------------------------------------------------
@pytest.fixture
def gui():
    from desktop import app, tool_pages
    app.dpg.create_context()
    app.build_ui()
    try:
        yield app
    finally:
        tool_pages.shutdown()
        app.dpg.destroy_context()


def test_the_page_lists_every_tool_and_its_command(gui):
    from desktop import tool_pages
    assert ('evaluate', '11   Evaluate & Benchmarks') in gui.NAV_ITEMS
    gui.on_nav_selected(user_data='evaluate')
    assert gui.dpg.get_item_configuration('page_evaluate')['show']
    listed = [tool for _, _, names in tool_pages.EVALUATE_GROUPS for tool in names]
    assert sorted(listed) == sorted(tools.EVALUATE_TOOLS) and len(listed) == len(set(listed))
    for tool in tools.EVALUATE_TOOLS:
        assert gui.dpg.does_item_exist(f'{tool}_start_button')
        command = gui.dpg.get_value(tool_pages.COMMAND_TAG[tool])
        spec = tools.TOOLS[tool]
        assert (f'-m {spec.module}' in command) if spec.module else (Path(spec.script).name in command), command


def test_a_refused_setting_is_shown_and_nothing_starts(gui, monkeypatch):
    from desktop import option_widgets as ow
    from desktop import tool_pages
    started = []
    monkeypatch.setattr(tool_pages, 'start_tool_run', lambda *a, **k: started.append(a))
    specs = co.by_key(co.MOTION_BENCHMARK_OPTIONS)
    gui.dpg.set_value(ow.tag('motion', specs['task']), 'direction')
    gui.dpg.set_value(ow.tag('motion', specs['models']), 'dummy')
    gui.dpg.set_value(ow.tag('motion', specs['out']), 'out')
    gui.dpg.set_value(ow.tag('motion', specs['hflip']), True)
    assert tool_pages.start('run_motion_benchmark') is False and not started
    assert 'flip is refused' in gui.dpg.get_value('evaluate_status')


def test_the_results_view_reads_the_chosen_tools_folder(gui, tmp_path):
    from desktop import option_widgets as ow
    from desktop import tool_pages
    (tmp_path / 'report.json').write_text(json.dumps({'experiments': {'dfine_botsort': {'metrics': {'HOTA': 52.6}}}}))
    gui.dpg.set_value(ow.tag('mote2e', co.by_key(co.MOT17_E2E_OPTIONS)['out']), str(tmp_path))
    gui.dpg.set_value('evaluate_results_tool', tools.TOOLS['benchmark_mot17_e2e'].title)
    tool_pages.on_refresh(user_data=tool_pages._results_tool())
    texts = [gui.dpg.get_value(item) for item in gui.dpg.get_all_items()
             if gui.dpg.get_item_type(item) == 'mvAppItemType::mvText' and isinstance(gui.dpg.get_value(item), str)]
    assert 'report.json: experiments' in texts and 'dfine_botsort' in texts and '52.6' in texts

"""G1: the GUI catch-up. Each new control reaches argv, defaults add nothing, GUI-built runs parse with the real CLIs,
and the new pages and result views work headless (no viewport, no torch in the GUI code under test)."""
from __future__ import annotations

import contextlib
import io
import json
import queue
import runpy
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from desktop import analyse
from desktop import cli_options as co
from desktop import config as gc
from desktop import tools

REPO = Path(__file__).resolve().parents[1]
CLASSIC_FLAGS = {'--modality', '--df', '--out', '--model_version', '--mode', '--batch_size', '--added_layers', '--embed_size',
                 '--num_epochs', '--learning_rate', '--target', '--num_classes', '--raw-data', '--fixed-seed', '--num_frames',
                 '--pooling'}


def _options(argv):
    from cli import DeepTuneVisionOptions
    from utils import RunType
    with contextlib.redirect_stdout(io.StringIO()):
        return DeepTuneVisionOptions(RunType.ONECALL, args=argv)


def _flags(argv):
    return {word.split('=', 1)[0] for word in argv if word.startswith('--')}


def video_cfg(tmp_path, **changes) -> gc.RunConfig:
    return replace(gc.RunConfig(modality='video', df=tmp_path / 'clips.parquet', raw_data=False, out=tmp_path / 'out'), **changes)


# --- defaults add nothing ---------------------------------------------------------------------------------------------
def test_default_settings_send_exactly_the_flags_they_sent_before_g1(tmp_path):
    for modality in gc.MODALITIES:
        cfg = gc.RunConfig(modality=modality, df=tmp_path, out=tmp_path / 'o', model_version=gc.MODEL_VERSIONS_BY_MODALITY[modality][0])
        extra = _flags(gc.build_argv(cfg)) - CLASSIC_FLAGS - {'--type', '--time_idx_column'}
        assert not extra, (modality, extra)


def test_default_tool_and_track_settings_send_only_what_is_required():
    assert tools.build_tool_argv(tools.ToolConfig('select_model', {'df': 'd.parquet', 'out': 's'})) == ['--df', 'd.parquet', '--out', 's']
    assert tools.build_tool_argv(tools.ToolConfig('extract_features', {'data': 'c.csv', 'out': 'f'})) == ['--data', 'c.csv', '--out', 'f']
    argv = analyse.build_track_argv(analyse.TrackConfig(video=Path('v.mp4'), prompts=['cat'], out=Path('o')))
    assert argv == ['--video', 'v.mp4', '--out', 'o', '--detector', 'grounding_dino', '--every_n_frames', '1', '--save_video',
                    '--prompt=cat']


def test_default_test_settings_send_the_recorded_argv_unchanged(tmp_path):
    cfg = gc.InferenceConfig(run_dir=tmp_path, modality='video', checkpoint_path=tmp_path / 'w.pth', data=tmp_path, out=tmp_path / 'p',
                             video_options={'test_clips': 5})
    argv = gc.build_infer_argv(cfg)
    assert argv[argv.index('--test_clips') + 1] == '5' and argv.count('--test_clips') == 1
    assert '--no_abstain' not in argv and '--ensemble' not in argv


# --- GUI-built runs parse with the real CLI ---------------------------------------------------------------------------
@pytest.mark.parametrize('version', gc.GUI_MODEL_VERSIONS['video'])
def test_every_gui_video_model_with_default_settings_parses(version, tmp_path):
    """The R1 lesson: a default the GUI sends must not be one the CLI refuses for some models."""
    hub = gc.hub_backbone(version)
    frames = 16 if version == 'mvit_v2_s' else (hub.fixed_frames if hub and hub.fixed_frames else 8)
    options = {'accept_weights_licence': [hub.licence]} if hub is not None and hub.opt_in else {}
    cfg = video_cfg(tmp_path, model_version=version, num_frames=frames, options=options)
    assert not [issue for issue in gc.options_issues(cfg)], gc.options_issues(cfg)
    parsed = _options(gc.build_argv(cfg))
    if hub is not None and hub.kind == 'video':
        assert parsed.pooling.startswith('attentive_probe')


def test_a_clip_model_with_a_probe_choice_parses(tmp_path):
    cfg = video_cfg(tmp_path, model_version='vjepa2_tiny', options={'video_arch': 'linear_probe'})
    assert _options(gc.build_argv(cfg)).pooling == 'linear_probe'
    cfg = video_cfg(tmp_path, model_version='vjepa2_tiny', options={'video_arch': 'attentive_probe', 'probe_depth': 2})
    assert 'probe_depth=2' in _options(gc.build_argv(cfg)).pooling


def test_temporal_architecture_options_compose_the_pooling_spec(tmp_path):
    cfg = video_cfg(tmp_path, options={'video_arch': 'tsm', 'tsm_fold_div': 4})
    assert _options(gc.build_argv(cfg)).pooling == 'tsm:tsm_fold_div=4'
    cfg = video_cfg(tmp_path, options={'video_arch': 'cnn_rnn', 'rnn_type': 'gru', 'bidirectional': True})
    assert _options(gc.build_argv(cfg)).pooling == 'cnn_rnn:bidirectional=1,rnn_hidden=256,rnn_layers=1,rnn_type=gru'


def test_a_full_video_recipe_parses_and_its_training_options_are_valid(tmp_path):
    from handlers.training_args import TrainingOptions
    options = {'video_augment': True, 'video_flip': True, 'video_speed_perturb': 0.1, 'mixup_alpha': 0.2, 'cutmix_alpha': 1.0,
               'mix_prob': 0.5, 'label_smoothing': 0.1, 'precision': 'bf16', 'seed': 7, 'select_best': True,
               'early_stopping_patience': 3, 'lr_scheduler': 'cosine', 'warmup_epochs': 0.5, 'grad_accum_steps': 2,
               'test_clips': 3, 'test_crops': 3, 'validation': 'kfold', 'k': 3, 'report_boot': 200, 'target_error': 0.1,
               'abstain_guarantee': 'silent_error', 'export_dfanalyze': True, 'dfanalyze_extras': ['tta', 'ids'],
               'temporal_sampling': 'segment', 'num_workers': '0'}
    cfg = video_cfg(tmp_path, options=options)
    assert gc.options_issues(cfg) == []
    parsed = _options(gc.build_argv(cfg))
    assert (parsed.video_augment, parsed.video_flip, parsed.test_crops, parsed.validation, parsed.target_error) == (True, True, 3, 'kfold', 0.1)
    assert parsed.dfanalyze_extras == ['tta', 'ids'] and parsed.num_workers == 0
    assert TrainingOptions.from_args(parsed).validate('cls').precision == 'bf16'


def test_video_only_options_stay_off_image_and_table_runs(tmp_path):
    options = {'video_augment': True, 'validation': 'kfold', 'precision': 'bf16', 'target_error': 0.1}
    image = gc.build_argv(replace(video_cfg(tmp_path, options=options), modality='images'))
    assert '--precision' in image and '--video_augment' not in image and '--validation' not in image
    table = gc.build_argv(replace(video_cfg(tmp_path, options=options), modality='tabular', model_version='tabpfn'))
    assert not {'--precision', '--video_augment', '--validation', '--target_error'} & _flags(table)


# --- GUI-side checks --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('options,model,message', [
    ({}, 'videomae_b16', 'I accept the licence'),
    ({'video_flip': True}, 'resnet18', 'Augment training clips'),
    ({'cv_only': True}, 'resnet18', 'kfold or bootstrap632'),
    ({'unfreeze_backbone': True}, 'resnet18', 'frozen by default'),
    ({'video_arch': 'tsm'}, 'vjepa2_vitl', 'attentive_probe or linear_probe'),
    ({'num_workers': 'many'}, 'resnet18', 'DataLoader workers'),
    ({'target_error': 1.5}, 'resnet18', 'Target error rate'),
])
def test_validate_run_explains_refused_combinations(options, model, message, tmp_path):
    frames = 16 if model == 'videomae_b16' else 8
    issues = gc.validate_run(video_cfg(tmp_path, model_version=model, num_frames=frames, options=options))
    assert any(message in issue for issue in issues), issues


def test_the_cli_path_is_unchanged_by_gui_only_checks(tmp_path):
    """deeptune.py runs validate_config on its own command line, which holds no GUI options: the licence check stays
    the CLI's (handlers/video_backbones), and validate_config adds nothing new."""
    cfg = video_cfg(tmp_path, model_version='videomae_b16', num_frames=16)
    assert not any('licence' in issue for issue in gc.validate_config(cfg))
    assert any('licence' in issue for issue in gc.validate_run(cfg))


def test_settings_with_options_save_and_load(tmp_path):
    cfg = video_cfg(tmp_path, model_version='videomae_b16', num_frames=16,
                    options={'accept_weights_licence': ['CC-BY-NC-4.0'], 'test_crops': 3, 'seed': 4, 'mix_prob': 0.5})
    gc.save_config(cfg, tmp_path / 's.json')
    loaded = gc.load_config(tmp_path / 's.json')
    # A licence is accepted by a deliberate tick each session: it is never saved or restored.
    assert loaded.options == {key: value for key, value in cfg.options.items() if key != 'accept_weights_licence'}
    assert loaded.model_version == 'videomae_b16' and 'accept_weights_licence' not in loaded.options
    payload = json.loads((tmp_path / 's.json').read_text(encoding='utf-8'))
    payload['config']['options'] = {'not_a_flag': 1}
    (tmp_path / 'bad.json').write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown setting'):
        gc.load_config(tmp_path / 'bad.json')


# --- Test page (infer.py) ---------------------------------------------------------------------------------------------
def _infer_parse(argv):
    import infer
    return infer.build_parser().parse_args(argv)


def test_an_override_replaces_the_recorded_value_once(tmp_path):
    cfg = gc.InferenceConfig(run_dir=tmp_path, modality='video', checkpoint_path=tmp_path / 'w.pth', data=tmp_path, out=tmp_path / 'p',
                             video_options={'test_clips': 5}, options={'test_clips': 2, 'no_abstain': True})
    argv = gc.build_infer_argv(cfg)
    assert argv.count('--test_clips') == 1 and _infer_parse(argv).test_clips == 2 and _infer_parse(argv).no_abstain


def test_an_ensemble_sends_only_the_flags_infer_accepts_with_it(tmp_path):
    import infer
    cfg = gc.InferenceConfig(run_dir=tmp_path, modality='video', checkpoint_path=tmp_path / 'w.pth', data=tmp_path, out=tmp_path / 'p',
                             label_mapping_path=tmp_path / 'labels.json', video_options={'test_clips': 5},
                             options={'ensemble': ['a', 'b'], 'allow_unverified_labels': True})
    argv = gc.build_infer_argv(cfg)
    assert {flag[2:] for flag in _flags(argv)} <= infer.ENSEMBLE_FLAGS
    assert _infer_parse(argv).ensemble == [Path('a'), Path('b')]
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    ok = replace(cfg, options={'ensemble': [str(tmp_path / 'a'), str(tmp_path / 'b')]})
    assert gc.validate_infer_config(ok) == []
    assert any('recorded' in i for i in gc.validate_infer_config(replace(ok, options={**ok.options, 'test_clips': 3})))
    assert any('only applies to an ensemble' in i for i in gc.validate_infer_config(replace(ok, options={'allow_unverified_labels': True})))


# --- tools ------------------------------------------------------------------------------------------------------------
def test_select_model_checks_mirror_the_cli_refusals():
    base = {'df': 'missing.parquet', 'out': 'o', 'backbones': ['resnet18']}
    issues = tools.validate_tool(tools.ToolConfig('select_model', {**base, 'search': 'random', 'top_families': 2}))
    assert any('only for the smart strategy' in i for i in issues)
    issues = tools.validate_tool(tools.ToolConfig('select_model', {**base, 'augmentations': ['aug_flip']}))
    assert any('do not depend on direction' in i for i in issues)
    assert any('metric' in i for i in tools.validate_tool(tools.ToolConfig('select_model', {**base, 'metric': 'mae'})))


def test_select_model_argv_keeps_deeptune_args_last_with_the_dry_run(tmp_path):
    import select_model
    from desktop.runner import tool_argv
    cfg = tools.ToolConfig('select_model', {'df': 'd.parquet', 'out': 'o', 'backbones': ['resnet18', 'r3d_18'],
                                            'deeptune_args': ['--precision', 'bf16'], 'search': 'exhaustive'})
    argv = tool_argv(cfg, ['--dry_run'])
    assert argv[-3:] == ['--deeptune_args', '--precision', 'bf16']
    parsed = select_model.build_parser().parse_args(argv)
    assert parsed.dry_run and parsed.deeptune_args == ['--precision', 'bf16'] and parsed.backbones == ['resnet18', 'r3d_18']


def test_ensemble_needs_one_kind_of_member(tmp_path):
    assert 'not both' in tools.validate_tool(tools.ToolConfig('ensemble', {'out': 'o', 'runs': ['a'], 'predictions': ['b']}))[0]
    (tmp_path / 'a.csv').write_text('x')
    (tmp_path / 'b.csv').write_text('x')
    ok = tools.ToolConfig('ensemble', {'out': str(tmp_path / 'o'), 'predictions': [str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv')]})
    assert tools.validate_tool(ok) == []
    import ensemble
    assert ensemble.build_parser().parse_args(tools.build_tool_argv(ok)).predictions == [tmp_path / 'a.csv', tmp_path / 'b.csv']


def test_feature_and_probe_argv_parse_with_their_scripts():
    import extract_features
    import train_feature_head
    argv = tools.build_tool_argv(tools.ToolConfig('extract_features', {'data': 'c.csv', 'out': 'f', 'streams': ['rgb', 'motion'],
                                                                       'model_version': 'vjepa2_vitl', 'cache_tokens': '8x4x4'}))
    parsed = extract_features.build_parser().parse_args(argv)
    assert parsed.streams == ['rgb', 'motion'] and parsed.cache_tokens == '8x4x4'
    argv = tools.build_tool_argv(tools.ToolConfig('train_feature_head', {'features': 'f', 'train_df': 't', 'val_df': 'v', 'out': 'p',
                                                                         'head': 'linear_probe', 'fixed_seed': True}))
    parsed = train_feature_head.build_parser().parse_args(argv)
    assert parsed.head == 'linear_probe' and parsed.fixed_seed


def test_track_options_replace_what_the_sensitivity_chose_and_a_folder_replaces_the_video(tmp_path):
    import track
    cfg = analyse.TrackConfig(video=Path('v.mp4'), prompts=['cat'], out=Path('o'), track_score=0.1,
                              options={'box_threshold': 0.05, 'class_agnostic': True, 'device': 'cpu'})
    argv = analyse.build_track_argv(cfg)
    assert argv.count('--box_threshold') == 1
    parsed = track.get_parser().parse_args(argv)
    assert parsed.box_threshold == 0.05 and parsed.class_agnostic and parsed.device == 'cpu' and parsed.new_track_thresh == 0.1
    folder = analyse.build_track_argv(replace(cfg, options={'video_dir': str(tmp_path)}))
    parsed = track.get_parser().parse_args(folder)
    assert parsed.video is None and parsed.video_dir == str(tmp_path)
    assert analyse.validate_track_config(replace(cfg, video=None, options={'video_dir': str(tmp_path / 'none')}))[0].startswith('Folder not found')


# --- device -----------------------------------------------------------------------------------------------------------
def test_device_choice_sets_the_environment_of_every_run(monkeypatch):
    from desktop import runner
    assert co.device_env('auto') == {} and co.device_env('cpu') == {'CUDA_VISIBLE_DEVICES': '-1'}
    assert co.device_env('cuda:1') == {'CUDA_VISIBLE_DEVICES': '1'}
    with pytest.raises(ValueError):
        runner.set_device('tpu')
    seen = {}

    class FakeProcess:
        stdout = None
        pid = 1

        def wait(self):
            return 0

        def poll(self):
            return 0

    def fake_popen(cmd, **kwargs):
        seen['cmd'], seen['env'] = cmd, kwargs['env']
        return FakeProcess()
    monkeypatch.setattr(runner.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(runner, '_device', 'auto')
    runner.set_device('cpu')
    log_q, done_q = queue.Queue(), queue.Queue()
    runner._worker_process(runner.ROOT / 'ensemble.py', '--tool-worker', 'Ensemble', '', ['--help'], log_q, done_q,
                           runner._ProcessState(), 'ensemble')
    assert seen['env']['CUDA_VISIBLE_DEVICES'] == '-1' and done_q.get_nowait()[0] == 'success'
    assert seen['cmd'][-2:] == [str(runner.ROOT / 'ensemble.py'), '--help']  # a checkout runs the script itself


def test_frozen_tool_worker_runs_the_script_without_the_gui():
    code = ("import runpy, sys\n"
            "sys.argv = ['app.py', '--tool-worker', 'ensemble', '--help']\n"
            "try:\n"
            "    runpy.run_path(r'desktop/app.py', run_name='__main__')\n"
            "except SystemExit as exc:\n"
            "    assert exc.code in (0, None), exc.code\n"
            "assert 'dearpygui' not in sys.modules and 'torch' not in sys.modules\n")
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert '--predictions' in result.stdout


# --- result readers ---------------------------------------------------------------------------------------------------
def _fake_run(tmp_path) -> Path:
    run = tmp_path / 'deeptune-1'
    (run / 'eval_output_x').mkdir(parents=True)
    (run / 'validation_kfold').mkdir()
    (run / 'dfanalyze_export_x').mkdir()
    (run / 'eval_output_x' / 'evaluation_report.json').write_text(json.dumps({
        'n': 40, 'split': 'test', 'metrics': {'accuracy': 0.8}, 'intervals': {'accuracy': {'low': 0.7, 'high': 0.9, 'level': 0.95}},
        'warnings': ['few groups']}), encoding='utf-8')
    (run / 'validation_kfold' / 'cv_summary.json').write_text(json.dumps({
        'scheme': 'kfold', 'k': 5, 'n_rows': 100, 'grouped': True, 'out_of_fold': {'metrics': {'accuracy': 0.75}, 'intervals': {}},
        'across_folds': {'accuracy': {'mean': 0.74, 'std': 0.05}}, 'caveats': ['folds share data']}), encoding='utf-8')
    (run / 'eval_output_x' / 'abstention_report.json').write_text(json.dumps({
        'target_error': 0.05, 'guarantee_kind': 'selective_risk', 'status': 'certified', 'guarantee_holds': True,
        'headline': 'Answers 60% of clips.', 'at_fitted_threshold': {'coverage': 0.6, 'error': 0.03, 'not_sure_clips': 16}}), encoding='utf-8')
    (run / 'dfanalyze_export_x' / 'dfanalyze_export.json').write_text(json.dumps({'command': 'df-analyze --df x.csv', 'rows': 'test'}),
                                                                       encoding='utf-8')
    return run


def test_run_reports_show_intervals_validation_abstention_and_export(tmp_path):
    found = tools.run_reports(_fake_run(tmp_path))
    sections = {name: next(s for s in found if s.title.startswith(name)) for name in
                ('Test report', 'Validation', 'Not sure (abstention)', 'df-analyze export')}
    assert ('accuracy', '0.8  95% [0.7, 0.9]') in sections['Test report'].rows
    assert ('out-of-fold accuracy', '0.75') in sections['Validation'].rows
    abstention = sections['Not sure (abstention)']
    assert ('coverage (answered share)', '0.6') in abstention.rows and 'Answers 60% of clips.' in abstention.notes
    assert any('df-analyze --df' in note for note in sections['df-analyze export'].notes)


def test_selection_results_lead_with_the_sealed_test_score(tmp_path):
    (tmp_path / 'selection.json').write_text(json.dumps({
        'winner': {'label': 'r3d_18 frozen', 'cv_accuracy_mean': 0.81}, 'runner_up': {'label': 'resnet18 full'},
        'reason': 'Highest-ranked.', 'notes': ["The winner's CV score is optimistic."], 'sealed_test_uses': 1,
        'test': {'metrics': {'accuracy': 0.72}, 'intervals': {'accuracy': {'low': 0.6, 'high': 0.8, 'level': 0.95}}}}), encoding='utf-8')
    pd.DataFrame({'rank': [1, 2], 'candidate': ['r3d_18 frozen', 'resnet18 full'], 'cv_accuracy_mean': [0.81, 0.79]}).to_csv(
        tmp_path / 'leaderboard.csv', index=False)
    sections = tools.selection_sections(tmp_path)
    assert sections[0].title == 'Sealed test score (the honest number)'
    assert ('accuracy', '0.72  95% [0.6, 0.8]') in sections[0].rows
    assert ('winner', 'r3d_18 frozen') in sections[1].rows
    assert list(sections[2].table['candidate']) == ['r3d_18 frozen', 'resnet18 full']


def test_an_unfinished_search_says_it_will_resume(tmp_path):
    (tmp_path / 'ledger.jsonl').write_text('{}\n')
    assert tools.is_resumable(tmp_path)
    assert 'resume' in tools.selection_sections(tmp_path)[0].notes[0]


def test_inference_reports_show_abstention_and_ensemble(tmp_path):
    (tmp_path / 'abstention_summary.json').write_text(json.dumps({'answered': 30, 'not_sure': 10, 'answer_rate': 0.75}))
    (tmp_path / 'ensemble.json').write_text(json.dumps({'members': ['a', 'b'], 'labels_verified': True}))
    titles = [s.title for s in tools.inference_reports(tmp_path)]
    assert titles[:2] == ['Not sure (abstention)', 'Ensemble']


# --- headless widgets -------------------------------------------------------------------------------------------------
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


def ready_video_cfg(tmp_path, **changes):
    """A video run the Run page accepts: a raw clip folder (no clips are read to validate it)."""
    return video_cfg(tmp_path, df=tmp_path, raw_data=True, **changes)


def test_model_page_advanced_controls_reach_the_command(gui, tmp_path):
    from desktop import option_widgets as ow
    gui.apply_config(ready_video_cfg(tmp_path))
    assert gui.dpg.get_item_configuration('run_button')['enabled'], gui.dpg.get_value('validation_text')
    spec = co.by_key(co.DEEPTUNE_OPTIONS)
    gui.dpg.set_value(ow.tag('dt', spec['video_augment']), True)
    gui.dpg.set_value(ow.tag('dt', spec['validation']), 'kfold')
    gui.dpg.set_value(ow.tag('dt', spec['target_error']), '0.05')
    gui.on_config_changed()
    cfg = gui._build_run_config()
    assert cfg.options == {'video_augment': True, 'validation': 'kfold', 'target_error': 0.05}
    command = gui.dpg.get_value('command_text')
    assert '--video_augment' in command and '--validation kfold' in command and '--target_error 0.05' in command
    gui.dpg.set_value(ow.tag('dt', spec['target_error']), 'often')
    gui.on_config_changed()
    assert 'Target error rate must be a number' in gui.dpg.get_value('validation_text')
    assert not gui.dpg.get_item_configuration('run_button')['enabled']


def test_settings_restore_the_advanced_controls(gui, tmp_path):
    cfg = video_cfg(tmp_path, options={'precision': 'bf16', 'dfanalyze_extras': ['motion'], 'seed': 3, 'mix_prob': 0.5})
    gui.apply_config(cfg)
    assert gui._build_run_config().options == cfg.options


def test_video_sections_hide_for_images_and_hub_models_set_their_rules(gui, tmp_path):
    from desktop import option_widgets as ow
    spec = co.by_key(co.DEEPTUNE_OPTIONS)
    gui.dpg.set_value('modality_radio', 'images')
    gui.on_modality_changed()
    assert not gui.dpg.get_item_configuration(ow.row_tag('dt', spec['video_augment']))['show']
    assert gui.dpg.get_item_configuration(ow.row_tag('dt', spec['precision']))['show']
    assert not gui.dpg.get_item_configuration(ow.section_tag('dt', co.S_ABSTAIN))['show']
    gui.apply_config(ready_video_cfg(tmp_path))
    assert 'videomae_b16' in gui.dpg.get_item_configuration('model_version_combo')['items']
    gui.dpg.set_value('model_version_combo', 'videomae_b16')
    gui.on_model_version_changed()
    assert gui.dpg.get_value('num_frames_input') == 16
    assert not gui.dpg.get_item_configuration('pooling_radio')['enabled']
    assert 'CC-BY-NC-4.0' in gui.dpg.get_value('hub_backbone_note')
    licence = f"{ow.tag('dt', spec['accept_weights_licence'])}__CC-BY-NC-4.0"
    assert gui.dpg.get_value(licence) is False  # never pre-ticked
    gui.on_config_changed()
    assert not gui.dpg.get_item_configuration('run_button')['enabled']
    gui.dpg.set_value(licence, True)
    gui.on_config_changed()
    assert '--accept_weights_licence CC-BY-NC-4.0' in gui.dpg.get_value('command_text')


def test_test_page_options_go_into_the_inference_config(gui, tmp_path):
    from desktop import option_widgets as ow
    spec = co.by_key(co.INFER_OPTIONS)
    gui.dpg.set_value(ow.tag('infer', spec['no_abstain']), True)
    gui.dpg.set_value(ow.tag('infer', spec['test_crops']), '3')
    gui.dpg.set_value('test_batch_size_input', 4)
    results = gc.RunResults(run_dir=tmp_path, checkpoint_path=tmp_path / 'w.pth', cli_arguments={'modality': 'video', 'model_version': 'resnet18'})
    cfg = gui._test_config(results, tmp_path, tmp_path / 'p')
    argv = gc.build_infer_argv(cfg)
    parsed = _infer_parse(argv)
    assert parsed.no_abstain and parsed.test_crops == 3 and parsed.batch_size == 4


def test_choosing_another_test_run_clears_the_overrides_of_the_last_one(gui, tmp_path):
    from desktop import option_widgets as ow
    spec = co.by_key(co.INFER_OPTIONS)
    runs = []
    for name in ('deeptune-a', 'deeptune-b'):
        train = tmp_path / name / 'trainval_output_x'
        train.mkdir(parents=True)
        (train / 'model_weights.pth').write_bytes(b'w')
        runs.append(tmp_path / name)
    gui._test_run_history = {'a': runs[0], 'b': runs[1]}
    gui.on_test_run_selected(user_data='a')
    gui.dpg.set_value(ow.tag('infer', spec['model_version']), 'r3d_18')
    gui.dpg.set_value(ow.tag('infer', spec['no_abstain']), True)
    gui.on_test_run_selected(user_data='a')
    assert gui.dpg.get_value(ow.tag('infer', spec['model_version'])) == 'r3d_18'
    gui.on_test_run_selected(user_data='b')
    assert gui.dpg.get_value(ow.tag('infer', spec['model_version'])) == ''
    assert gui.dpg.get_value(ow.tag('infer', spec['no_abstain']))  # a test option, not an override, stays


class _FakeHandle:
    def __init__(self):
        self.log_queue, self.done_queue = queue.Queue(), queue.Queue()
        self.thread = threading.Thread(target=lambda: None)
        self.thread.start()

    def cancel(self):
        return False

    def close(self, timeout=0):
        pass


def test_model_selection_starts_only_after_a_dry_run_of_the_same_settings(gui, tmp_path, monkeypatch):
    from desktop import option_widgets as ow
    from desktop import tool_pages
    launched = []

    def fake_start(cfg, extra=None):
        launched.append(tool_pages.tool_argv(cfg, extra))
        return _FakeHandle()
    monkeypatch.setattr(tool_pages, 'start_tool_run', fake_start)
    data = tmp_path / 'dev.parquet'
    data.write_bytes(b'x')
    spec = co.by_key(co.SELECT_OPTIONS)
    for key, value in (('df', str(data)), ('out', str(tmp_path / 'search')), ('backbones', 'resnet18 r3d_18')):
        gui.dpg.set_value(ow.tag('sel', spec[key]), value)
    tool_pages.on_changed()
    assert not gui.dpg.get_item_configuration('select_start_button')['enabled']
    tool_pages.on_start_search()
    assert launched == []
    tool_pages.on_check_grid()
    assert '--dry_run' in launched[-1]
    handle = tool_pages._running['select'].handle
    handle.log_queue.put('120 candidates, about 3.5 hours\n')
    handle.done_queue.put(('success', ''))
    tool_pages.poll()
    assert '120 candidates' in gui.dpg.get_value('select_log_box')
    assert gui.dpg.get_item_configuration('select_start_button')['enabled']
    tool_pages.on_start_search()
    assert '--dry_run' not in launched[-1] and launched[-1][:2] == ['--df', str(data)]
    tool_pages._running.clear()
    gui.dpg.set_value(ow.tag('sel', spec['k']), 3)  # a changed setting needs a new check
    tool_pages.on_changed()
    assert not gui.dpg.get_item_configuration('select_start_button')['enabled']


def test_new_pages_render_results_from_disk(gui, tmp_path):
    from desktop import option_widgets as ow
    from desktop import tool_pages
    (tmp_path / 'selection.json').write_text(json.dumps({'winner': {'label': 'w'}, 'test': {'metrics': {'accuracy': 0.7}},
                                                         'sealed_test_uses': 1}))
    pd.DataFrame({'rank': [1], 'candidate': ['w']}).to_csv(tmp_path / 'leaderboard.csv', index=False)
    gui.dpg.set_value(ow.tag('sel', co.by_key(co.SELECT_OPTIONS)['out']), str(tmp_path))
    tool_pages.on_refresh(user_data='select_model')
    texts = [gui.dpg.get_value(item) for item in gui.dpg.get_all_items()
             if gui.dpg.get_item_type(item) == 'mvAppItemType::mvText' and isinstance(gui.dpg.get_value(item), str)]
    assert 'Sealed test score (the honest number)' in texts and '0.7' in texts
    gui.render_results(gc.read_run_results(_fake_run(tmp_path)))
    assert gui.dpg.get_item_children('results_reports_container', 1)


def test_ensemble_page_fills_runs_from_the_output_folder(gui, tmp_path):
    from desktop import option_widgets as ow
    from desktop import tool_pages
    for name in ('deeptune-a', 'deeptune-b'):
        train = tmp_path / name / 'trainval_output_x'
        train.mkdir(parents=True)
        (train / 'model_weights.pth').write_bytes(b'w')
        (train / 'cli_arguments.json').write_text(json.dumps({'modality': 'video'}))
    gui.dpg.set_value('outdir_input', str(tmp_path))
    tool_pages.on_fill_runs()
    lines = gui.dpg.get_value(ow.tag('ens', co.by_key(co.ENSEMBLE_OPTIONS)['runs'])).splitlines()
    assert len(lines) == 2 and all(line.endswith('model_weights.pth') for line in lines)


def test_device_combo_sets_the_runner_device(gui, monkeypatch):
    from desktop import runner
    monkeypatch.setattr(runner, '_device', 'auto')
    gui.dpg.set_value('device_combo', 'cpu')
    gui.on_device_changed()
    assert runner.device() == 'cpu'


# --- G1 fix round (V-058) -----------------------------------------------------------------------------------------------
def _selection(tmp_path, **payload):
    (tmp_path / 'selection.json').write_text(json.dumps({'winner': {'label': 'w'}, 'reason': 'r', 'notes': [], **payload}))
    return tools.selection_sections(tmp_path)[0]


def test_the_sealed_score_is_the_honest_number_only_when_used_once(tmp_path):
    once = _selection(tmp_path, sealed_test_uses=1, test={'metrics': {'accuracy': 0.7}})
    assert once.title == 'Sealed test score (the honest number)' and 'scored the final winner once' in once.notes[0]
    reused = _selection(tmp_path, sealed_test_uses=2, test={'metrics': {'accuracy': 0.9}})
    assert 'honest' not in reused.title and 'reused 2 times' in reused.title and 'not an unbiased estimate' in reused.title
    assert 'no longer an unbiased estimate' in reused.notes[0] and not any('scored the final winner once' in n for n in reused.notes)
    unknown = _selection(tmp_path, test={'metrics': {'accuracy': 0.9}})
    assert 'honest' not in unknown.title


def test_a_sealed_test_that_did_not_score_says_so_without_the_scored_once_note(tmp_path):
    section = _selection(tmp_path, sealed_test_uses=1, test=None,
                         final={'status': 'not_scored', 'reason': 'The sealed test set already scored an earlier winner (A).'})
    assert section.title == 'Sealed test score: not scored'
    assert 'earlier winner' in section.notes[0] and not any('scored the final winner once' in n for n in section.notes)
    assert section.rows == []


@pytest.mark.parametrize('name,payload', [
    ('evaluation_report.json', {'metrics': [0.8], 'warnings': []}),
    ('evaluation_report.json', {'metrics': {'accuracy': 0.8}, 'warnings': None}),
    ('cv_summary.json', {'scheme': 'kfold', 'caveats': None, 'out_of_fold': [1]}),
    ('abstention_report.json', {'at_fitted_threshold': [1], 'warnings': 'one'}),
])
def test_readers_never_crash_on_unexpected_json(tmp_path, name, payload):
    folder = tmp_path / ('validation_kfold' if name == 'cv_summary.json' else 'eval_output_x')
    folder.mkdir()
    (folder / name).write_text(json.dumps(payload))
    tools.run_reports(tmp_path)  # no exception, whatever the shape
    (folder / name).write_text('{"scheme": "kf')  # a half-written file
    assert [s.title for s in tools.run_reports(tmp_path)] == ['Could not read']


def test_selection_readers_survive_odd_types(tmp_path):
    (tmp_path / 'selection.json').write_text(json.dumps({'winner': ['w'], 'test': 'x', 'notes': None, 'runner_up': 3}))
    titles = [s.title for s in tools.selection_sections(tmp_path)]
    assert titles[0] == 'Sealed test score: not scored'


# --- recipe files instead of --config -----------------------------------------------------------------------------------
@pytest.mark.parametrize('name,text', [
    ('recipe.json', json.dumps({'learning_rate': 3e-4, 'precision': 'bf16', 'num_epochs': 30})),
    ('recipe.yaml', 'learning_rate: 3e-4\nprecision: bf16\nnum_epochs: 30\n'),
])
def test_a_recipe_fills_the_page_and_the_run_sends_what_the_page_shows(gui, tmp_path, monkeypatch, name, text):
    import deeptune
    from desktop import option_widgets as ow
    from desktop import runner
    seen = []
    monkeypatch.setattr(runner, '_run_process', lambda args, log_q, done_q, *rest: (seen.append(args), done_q.put(('success', ''))))
    monkeypatch.setattr(gui, 'SESSION_PATH', tmp_path / 'session.json')
    gui.apply_config(ready_video_cfg(tmp_path))
    (tmp_path / name).write_text(text, encoding='utf-8')
    gui.on_load_recipe(app_data={'file_path_name': str(tmp_path / name)})
    assert gui.dpg.get_value('learning_rate_input') == pytest.approx(3e-4)
    assert gui.dpg.get_value('num_epochs_input') == 30
    assert gui.dpg.get_value(ow.tag('dt', co.by_key(co.DEEPTUNE_OPTIONS)['precision'])) == 'bf16'
    gui._current_run = None
    gui.on_run_clicked()
    gui._current_run.thread.join(timeout=10)
    gui._current_run = None
    argv = seen[-1]
    assert '--config' not in argv
    with contextlib.redirect_stdout(io.StringIO()):
        args, _, training = deeptune.parse_args(argv)
    assert (args.learning_rate, args.num_epochs, training.precision) == (3e-4, 30, 'bf16')


def test_a_recipe_with_an_unknown_key_is_refused_whole(gui, tmp_path):
    gui.apply_config(ready_video_cfg(tmp_path))
    (tmp_path / 'bad.json').write_text(json.dumps({'learning_rate': 3e-4, 'learnin_rate': 1e-3, 'config': 'x.json'}))
    gui.on_load_recipe(app_data={'file_path_name': str(tmp_path / 'bad.json')})
    status = gui.dpg.get_value('recipe_status')
    assert 'was not loaded' in status and 'learnin_rate' in status and 'config' in status
    assert gui.dpg.get_value('learning_rate_input') == pytest.approx(1e-4)  # nothing applied


# --- Test-page overrides --------------------------------------------------------------------------------------------------
def _recorded_run(tmp_path, **record):
    run = tmp_path / 'deeptune-1'
    (run / 'trainval_output_x').mkdir(parents=True)
    (run / 'data_splits_x').mkdir()
    (run / 'trainval_output_x' / 'model_weights.pth').write_bytes(b'w')
    (run / 'trainval_output_x' / 'cli_arguments.json').write_text(json.dumps({'modality': 'video', 'model_version': 'resnet18',
                                                                             'num_classes': 3, **record}))
    (run / 'data_splits_x' / 'label_mapping.json').write_text(json.dumps({'a': 0, 'b': 1, 'c': 2}))
    return run


@pytest.mark.parametrize('record,override', [({'pooling': 'single_frame'}, {'pooling': 'mean'}),
                                             ({'pooling': 'tsm:tsm_fold_div=8'}, {'tsm_fold_div': 2}),
                                             ({}, {'label_mapping': 'other.json'})])
def test_model_defining_overrides_need_a_confirmation(tmp_path, record, override):
    """override_load.py (V-058): such overrides still load the weights strictly and change the logits silently."""
    results = gc.read_run_results(_recorded_run(tmp_path, **record))
    cfg = gc.infer_config_from_run(results, data=tmp_path, out=tmp_path / 'p')
    cfg.options = override
    assert any('change the model rebuilt' in issue for issue in gc.validate_infer_config(cfg))
    cfg.confirm_model_overrides = True
    assert not any('change the model rebuilt' in issue for issue in gc.validate_infer_config(cfg))
    cfg.options = {'test_clips': 3, 'temporal_sampling': 'segment'}  # test-time overrides need no confirmation
    cfg.confirm_model_overrides = False
    assert gc.validate_infer_config(cfg) == []


def test_a_confirmed_override_writes_overrides_json_and_warns_in_the_log(tmp_path, monkeypatch):
    from desktop import runner
    monkeypatch.setattr(runner, '_infer_process', lambda args, log_q, done_q, *rest: done_q.put(('success', '')))
    results = gc.read_run_results(_recorded_run(tmp_path, pooling='single_frame'))
    cfg = gc.infer_config_from_run(results, data=tmp_path, out=tmp_path / 'pred')
    cfg.options, cfg.confirm_model_overrides = {'pooling': 'mean', 'test_clips': 2}, True
    handle = runner.start_infer_run(cfg)
    handle.thread.join(timeout=10)
    record = json.loads((tmp_path / 'pred' / 'overrides.json').read_text(encoding='utf-8'))
    assert record['model_overrides'] == {'pooling': 'mean'} and record['test_time_overrides'] == {'test_clips': 2}
    assert record['model_overrides_confirmed'] and '--pooling' in record['argv']
    assert 'WARNING' in handle.log_queue.get_nowait()
    assert tools.inference_reports(tmp_path / 'pred')[0].title == 'Overrides of the recorded settings'
    cfg.options = {}
    assert gc.write_overrides(cfg, []) is None


def test_a_three_way_override_can_turn_a_recorded_flag_off(tmp_path):
    results = gc.read_run_results(_recorded_run(tmp_path, freeze_backbone=True))
    cfg = gc.infer_config_from_run(results, data=tmp_path, out=tmp_path / 'p')
    assert _infer_parse(gc.build_infer_argv(cfg)).freeze_backbone
    cfg.options = {'freeze_backbone': False, 'use_peft': True}
    parsed = _infer_parse(gc.build_infer_argv(cfg))
    assert parsed.freeze_backbone is False and parsed.use_peft is True


def test_the_test_page_blocks_model_overrides_until_confirmed(gui, tmp_path):
    from desktop import option_widgets as ow
    run = _recorded_run(tmp_path, pooling='single_frame')
    gui._test_run_history = {'r': run}
    gui._selected_test_run = None
    gui._current_infer_run = None
    gui.on_test_run_selected(user_data='r')
    assert gui.dpg.get_item_configuration('test_run_button')['enabled']
    gui.dpg.set_value(ow.tag('infer', co.by_key(co.INFER_OPTIONS)['pooling']), 'mean')
    gui.on_test_options_changed()
    assert not gui.dpg.get_item_configuration('test_run_button')['enabled']
    assert gui.dpg.get_item_configuration('infer_confirm_group')['show']
    gui.dpg.set_value('infer_confirm_model_overrides', True)
    gui.on_test_options_changed()
    assert gui.dpg.get_item_configuration('test_run_button')['enabled']


# --- ensemble membership --------------------------------------------------------------------------------------------------
def test_the_selected_runs_labels_go_with_an_ensemble_only_when_it_is_a_member(tmp_path):
    run = _recorded_run(tmp_path)
    other = tmp_path / 'other_run'
    other.mkdir()
    results = gc.read_run_results(run)
    cfg = gc.infer_config_from_run(results, data=tmp_path, out=tmp_path / 'p')
    cfg.options = {'ensemble': [str(other), str(tmp_path / 'third')]}
    assert '--label_mapping' not in gc.build_infer_argv(cfg)
    cfg.options = {'ensemble': [str(run / 'trainval_output_x' / 'model_weights.pth'), str(other)]}
    argv = gc.build_infer_argv(cfg)
    assert _infer_parse(argv).label_mapping == cfg.label_mapping_path
    spec = co.by_key(co.INFER_OPTIONS)['ensemble']
    assert 'exactly' in spec.label and 'besides' not in spec.help and 'only if it is listed' in spec.help


# --- model selection in the frozen build ---------------------------------------------------------------------------------
def test_model_selection_runs_its_scripts_through_the_frozen_workers(tmp_path, monkeypatch):
    from handlers.model_selection import probe
    from handlers.model_selection import runner as ms_runner
    seen = []

    class Done:
        def wait(self):
            return 0

    monkeypatch.setattr(ms_runner.subprocess, 'Popen', lambda cmd, **kw: seen.append((cmd, kw['cwd'])) or Done())
    ms_runner.SubprocessRunner(python='py')(['--modality', 'video'], tmp_path / 'a', 'cv')
    with open(tmp_path / 'log.txt', 'w', encoding='utf-8') as log:
        probe.FeatureProbeRunner(python='py')._call('extract_features.py', ['--data', 'x'], log)
    assert seen[0] == (['py', '-u', str(ms_runner.REPO / 'deeptune.py'), '--modality', 'video'], str(ms_runner.REPO))
    assert seen[1][0] == ['py', '-u', str(ms_runner.REPO / 'extract_features.py'), '--data', 'x']
    exe = tmp_path / 'DeepTune.exe'
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(exe))
    ms_runner.SubprocessRunner()(['--modality', 'video'], tmp_path / 'b', 'cv')
    with open(tmp_path / 'log.txt', 'w', encoding='utf-8') as log:
        probe.FeatureProbeRunner()._call('train_feature_head.py', ['--out', 'o'], log)
    assert seen[2] == ([str(exe), '--deeptune-worker', '--modality', 'video'], str(tmp_path))
    assert seen[3][0] == [str(exe), '--tool-worker', 'train_feature_head', '--out', 'o']


# --- device, licences, select flags, resume ------------------------------------------------------------------------------
def test_run_records_say_which_device_the_run_got(tmp_path, monkeypatch):
    from desktop import runner

    class Done:
        stdout = None
        pid = 1

        def wait(self):
            return 0

        def poll(self):
            return 0
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda cmd, **kw: Done())
    monkeypatch.setattr(runner, '_device', 'auto')
    runner.set_device('cuda:1')
    log_q, done_q = queue.Queue(), queue.Queue()
    runner._worker_process(runner.ROOT / 'ensemble.py', '--tool-worker', 'Ensemble', '', ['--help'], log_q, done_q,
                           runner._ProcessState(), 'ensemble', tmp_path)
    record = json.loads((tmp_path / 'desktop-run.json').read_text(encoding='utf-8'))
    assert (record['device'], record['CUDA_VISIBLE_DEVICES'], record['status']) == ('cuda:1', '1', 'success')
    first = log_q.get_nowait()
    assert first.startswith('Device: cuda:1 (CUDA_VISIBLE_DEVICES=1)') and 'falls back to the CPU' in first


def test_changing_the_device_invalidates_the_dry_run(gui, tmp_path, monkeypatch):
    from desktop import runner, tool_pages
    monkeypatch.setattr(runner, '_device', 'auto')
    monkeypatch.setattr(tool_pages, '_checked_select_argv', tool_pages.tool_argv(tool_pages.config('select_model')))
    monkeypatch.setattr(tool_pages, '_checked_select_device', 'auto')
    assert tool_pages._select_checked()
    gui.dpg.set_value('device_combo', 'cpu')
    gui.on_device_changed()
    assert not tool_pages._select_checked()


def test_licences_are_not_restored_from_the_session(gui, tmp_path):
    from desktop import option_widgets as ow
    cfg = ready_video_cfg(tmp_path, model_version='videomae_b16', num_frames=16, options={'accept_weights_licence': ['CC-BY-NC-4.0']})
    gc.save_config(cfg, tmp_path / 'session.json')
    gui.apply_config(gc.load_config(tmp_path / 'session.json'))
    box = f"{ow.tag('dt', co.by_key(co.DEEPTUNE_OPTIONS)['accept_weights_licence'])}__CC-BY-NC-4.0"
    assert gui.dpg.get_value(box) is False
    assert not gui.dpg.get_item_configuration('run_button')['enabled']


def test_include_is_refused_outside_the_smart_search_as_the_cli_does():
    import select_model
    base = {'df': 'd.parquet', 'out': 'o', 'backbones': ['resnet18'], 'include': ['backbone=resnet18']}
    assert any('--include' in i for i in tools.validate_tool(tools.ToolConfig('select_model', {**base, 'search': 'random'})))
    assert not any('--include' in i for i in tools.validate_tool(tools.ToolConfig('select_model', base)))
    args = select_model.build_parser().parse_args(['--df', 'd.parquet', '--out', 'o', '--search', 'random', '--include', 'backbone=x',
                                                   '--backbones', 'resnet18'])
    with pytest.raises(ValueError, match='--include'):
        select_model.options_from_args(args)


def test_resume_with_a_validation_scheme_is_refused_before_a_process_starts(tmp_path):
    (tmp_path / 'last.pt').write_bytes(b'x')
    issues = gc.validate_run(video_cfg(tmp_path, options={'resume': str(tmp_path / 'last.pt'), 'validation': 'kfold'}))
    assert '--resume cannot be combined with --validation: every fold must train a fresh model.' in issues


def test_the_seed_tooltip_says_the_gui_fixes_the_seed_by_default():
    assert '--fixed-seed' in co.by_key(co.DEEPTUNE_OPTIONS)['seed'].help

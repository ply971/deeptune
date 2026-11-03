"""G1: every flag of every entry point is reachable from the desktop GUI, through what its Start button sends.

For each argparse flag of deeptune.py, infer.py, select_model.py, ensemble.py, extract_features.py,
train_feature_head.py and track.py, the test builds the real UI headless, sets the flag's control (a widget, or for
the oldest controls the widgets apply_config sets) to a value that differs from the CLI default, presses the page's
real Start handler (app.on_run_clicked, app.on_run_test_clicked, analyse_page.on_track_clicked, tool_pages.start and
tool_pages.on_check_grid), and captures the argv the runner would launch: the subprocess functions are replaced, nothing
else. The flag must be in that argv and parse back, with the entry point's real parser, to the value that was set.
deeptune.py command lines must also pass deeptune.parse_args, the CLI's own checks. Every deeptune.py flag that applies to
video (all but the five table and time-series ones) must pass in a video run.

The flags come from the parsers, not from the GUI, so a new flag without a control fails here. The only exceptions are
ALLOW_LIST below, each with its reason.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from desktop import cli_options as co
from desktop import config as gc

REPO = Path(__file__).resolve().parents[1]

# (entry point, flag) -> why the GUI does not offer it. Keep this short; every entry needs a reason.
ALLOW_LIST: dict = {
    ('deeptune.py', '--config'): 'The Model page reads a recipe file into its own controls ("Load recipe file", '
                                 'desktop/config.recipe_config), so the run sends exactly what the page shows; passing '
                                 '--config as well would let values the page does not show override it.',
}


def _parsers():
    import deeptune
    import ensemble
    import extract_features
    import infer
    import select_model
    import track
    import train_feature_head
    return {'deeptune.py': deeptune.build_parser(), 'infer.py': infer.build_parser(), 'select_model.py': select_model.build_parser(),
            'ensemble.py': ensemble.build_parser(), 'extract_features.py': extract_features.build_parser(),
            'train_feature_head.py': train_feature_head.build_parser(), 'track.py': track.get_parser()}


PARSERS = _parsers()
SPECS = {'deeptune.py': co.DEEPTUNE_OPTIONS, 'infer.py': co.INFER_OPTIONS, 'select_model.py': co.SELECT_OPTIONS,
         'ensemble.py': co.ENSEMBLE_OPTIONS, 'extract_features.py': co.EXTRACT_OPTIONS,
         'train_feature_head.py': co.PROBE_OPTIONS, 'track.py': co.TRACK_OPTIONS}
PAGE_PREFIXES = {'deeptune.py': 'dt', 'infer.py': 'infer', 'select_model.py': 'sel', 'ensemble.py': 'ens',
                 'extract_features.py': 'feat', 'train_feature_head.py': 'probe', 'track.py': 'track'}
TOOL = {'select_model.py': 'select_model', 'ensemble.py': 'ensemble', 'extract_features.py': 'extract_features',
        'train_feature_head.py': 'train_feature_head'}
# deeptune.py flags that only apply to tables and time series; every other one must work in a video run.
NOT_VIDEO = {'--type', '--continuous_cols', '--categorical_cols', '--finetuning-mode', '--time_idx_column', '--modality'}


def flags_of(parser: argparse.ArgumentParser) -> list:
    return [action for action in parser._actions if action.option_strings and not isinstance(action, argparse._HelpAction)]


def _spec(entry, action):
    return next((opt for opt in SPECS[entry] if opt.flag in action.option_strings), None)


# --- a non-default value for a control, valid for the real parser --------------------------------------------------
TEXT_CANDIDATES = ('custom_value', '2', 'attention', '8x4x4')


def _accepts(action, text) -> bool:
    try:
        value = action.type(text) if action.type is not None else text
    except (TypeError, ValueError, argparse.ArgumentTypeError):
        return False
    return action.choices is None or value in action.choices


def _convert(action, text):
    return action.type(text) if action.type is not None else text


def gui_value(opt: co.Opt, action) -> tuple:
    """(value for the control, value the parser must return). Never the control's default, nor the CLI default (they
    differ for the Test page's override controls, which start blank)."""
    if opt.kind in (co.BOOL, co.TRI):
        return True, action.const
    if opt.kind == co.CHOICE:
        choice = next(c for c in opt.choices if c != opt.default and _accepts(action, str(c))
                      and _convert(action, str(c)) != action.default)
        return choice, _convert(action, str(choice))
    if opt.kind == co.MULTI:
        picked = [next(c for c in reversed(opt.choices) if c not in (opt.default or ()))]
        return picked, [_convert(action, str(c)) for c in picked]
    if opt.kind in (co.INT, co.OPT_INT):
        value = (opt.default + 1) if isinstance(opt.default, int) else 3
        value += value == action.default
        assert _accepts(action, str(value)), (opt.flag, value)
        return value, _convert(action, str(value))
    if opt.kind in (co.FLOAT, co.OPT_FLOAT):
        value = opt.default / 2 if opt.default else 0.25
        assert _accepts(action, repr(value)), (opt.flag, value)
        return value, _convert(action, repr(value))
    if opt.kind in (co.WORDS, co.LINES):
        for words in (['path_a', 'path_b'], ['1', '2'], ['0.5', '0.25']):
            if all(_accepts(action, w) for w in words):
                return words, [_convert(action, w) for w in words]
        raise AssertionError(f'no list value for {opt.flag}')
    text = next(t for t in TEXT_CANDIDATES if t != opt.default and _accepts(action, t) and _convert(action, t) != action.default)
    return text, _convert(action, text)


def set_widget(dpg, prefix, opt, value):
    from desktop import option_widgets as ow
    tag = ow.tag(prefix, opt)
    if opt.kind == co.MULTI:
        for choice in opt.choices:
            dpg.set_value(f'{tag}__{choice}', choice in value)
    elif opt.kind == co.WORDS:
        dpg.set_value(tag, ' '.join(map(str, value)))
    elif opt.kind == co.LINES:
        dpg.set_value(tag, '\n'.join(map(str, value)))
    elif opt.kind == co.TRI:
        dpg.set_value(tag, 'on' if value else 'off')
    elif opt.kind in (co.TEXT, co.OPT_INT, co.OPT_FLOAT):
        dpg.set_value(tag, co.format_value(value))
    elif opt.kind == co.CHOICE:
        dpg.set_value(tag, str(value))
    else:
        dpg.set_value(tag, value)


def _quiet_parse(parser, argv):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        try:
            return parser.parse_args(argv)
        except SystemExit:
            return None


def _flag_in(argv, action) -> bool:
    return any(word in action.option_strings or word.split('=', 1)[0] in action.option_strings for word in argv)


# --- the studio: the real UI, headless, with the launches captured ---------------------------------------------------
class Studio:
    """Builds the UI once, writes small real files the pages check for, and replaces the runner's subprocess functions
    with ones that only record the argv they were given."""

    def __init__(self, tmp: Path, patch):
        from desktop import analyse, analyse_page, app, runner, tool_pages
        self.app, self.analyse, self.analyse_page, self.runner, self.tool_pages = app, analyse, analyse_page, runner, tool_pages
        self.dpg = app.dpg
        self.tmp = tmp
        self.captured: list = []

        def fake(position):
            def run(*args, **kwargs):
                self.captured.append(list(args[position]))
                done_q = args[position + 2]
                done_q.put(('success', ''))
            return run
        patch.setattr(runner, '_run_process', fake(0))
        patch.setattr(runner, '_infer_process', fake(0))
        patch.setattr(runner, '_worker_process', fake(4))
        patch.setattr(app, 'SESSION_PATH', tmp / 'session.json')
        self._files()
        self.dpg.create_context()
        app.build_ui()

    def close(self):
        self.analyse_page.shutdown()
        self.tool_pages.shutdown()
        self.app.stop_video_playback(timeout=2)
        self.dpg.destroy_context()

    def _files(self):
        t = self.tmp
        for folder in ('clips', 'images', 'features', 'videos', 'member_a', 'member_b', 'new_clips'):
            (t / folder).mkdir()
        frame = pd.DataFrame({'videos': [f'v{i}.mp4' for i in range(40)], 'labels': [0, 1] * 20})
        for name in ('clips.parquet', 'other.parquet', 'test.parquet', 'dev.parquet'):
            frame.to_parquet(t / name)
        pd.DataFrame({'x': range(40), 'time': range(40), 'labels': [0, 1] * 20}).to_csv(t / 'table.csv', index=False)
        for name in ('rows.csv', 'pred_a.csv', 'pred_b.csv'):
            pd.DataFrame({'videos': ['a.mp4'], 'labels': [0]}).to_csv(t / name, index=False)
        (t / 'grid.json').write_text(json.dumps({'backbones': ['resnet18']}), encoding='utf-8')
        (t / 'labels.json').write_text(json.dumps({'a': 0, 'b': 1}), encoding='utf-8')
        (t / 'last_checkpoint.pt').write_bytes(b'checkpoint')
        import cv2
        import numpy as np
        writer = cv2.VideoWriter(str(t / 'videos' / 'clip.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        for value in range(12):
            writer.write(np.full((48, 64, 3), value * 10, dtype=np.uint8))
        writer.release()
        self.clip = t / 'videos' / 'clip.mp4'
        run = t / 'runs' / 'deeptune-1'
        (run / 'trainval_output_x').mkdir(parents=True)
        (run / 'data_splits_x').mkdir()
        (run / 'trainval_output_x' / 'model_weights.pth').write_bytes(b'w')
        (run / 'trainval_output_x' / 'cli_arguments.json').write_text(json.dumps(
            {'modality': 'video', 'model_version': 'resnet18', 'num_classes': 2, 'mode': 'cls', 'pooling': 'mean'}), encoding='utf-8')
        (run / 'data_splits_x' / 'label_mapping.json').write_text(json.dumps({'a': 0, 'b': 1}), encoding='utf-8')
        self.run = run

    # Values for controls whose page checks that a file exists, or that need a recognised word.
    def value_override(self, entry, key):
        t = self.tmp
        paths = {
            ('deeptune.py', 'resume'): str(t / 'last_checkpoint.pt'), ('deeptune.py', 'test_df'): str(t / 'test.parquet'),
            ('infer.py', 'ensemble'): [str(t / 'member_a'), str(t / 'member_b')],
            ('select_model.py', 'df'): str(t / 'other.parquet'), ('select_model.py', 'grid'): str(t / 'grid.json'),
            ('select_model.py', 'gpu_memory_gb'): '6',
            ('ensemble.py', 'runs'): [str(t / 'member_a'), str(t / 'member_b')],
            ('ensemble.py', 'predictions'): [str(t / 'pred_a.csv'), str(t / 'pred_b.csv')],
            ('ensemble.py', 'data'): str(t / 'new_clips'), ('ensemble.py', 'label_mapping'): str(t / 'labels.json'),
            ('extract_features.py', 'data'): str(t / 'rows.csv'),
            ('train_feature_head.py', 'features'): str(t / 'images'), ('train_feature_head.py', 'train_df'): str(t / 'rows.csv'),
            ('train_feature_head.py', 'val_df'): str(t / 'rows.csv'), ('train_feature_head.py', 'test_df'): str(t / 'rows.csv'),
            ('track.py', 'video_dir'): str(t / 'videos'),
        }
        if key == 'out' and entry in TOOL:  # an output folder is created when the run starts: keep it under tmp
            return str(t / f'{TOOL[entry]}_out')
        return paths.get((entry, key))

    # --- deeptune.py: the Model page, Start training ------------------------------------------------------------------
    def deeptune_contexts(self):
        t, video = self.tmp, dict(modality='video', model_version='resnet18', num_classes=2)
        return [
            ('video', gc.RunConfig(df=t / 'clips.parquet', raw_data=False, out=t / 'out', **video)),
            ('video', gc.RunConfig(df=t / 'clips', raw_data=True, out=t / 'out', **video)),
            ('images', gc.RunConfig(modality='images', model_version='resnet18', df=t / 'images', raw_data=True, out=t / 'out')),
            ('tabular', gc.RunConfig(modality='tabular', model_version='gandalf', df=t / 'table.csv', out=t / 'out', continuous_cols=['x'])),
            ('tabular', gc.RunConfig(modality='tabular', model_version='tabpfn', df=t / 'table.csv', out=t / 'out')),
            ('timeseries', gc.RunConfig(modality='timeseries', model_version='deepAR', df=t / 'table.csv', out=t / 'out',
                                        time_idx_column='time')),
        ]

    # A flag the CLI only accepts beside another setting is tried with that setting too.
    DEEPTUNE_PREREQUISITES = {'video_flip': {'video_augment': True}, 'cv_only': {'validation': 'kfold'}}
    DEEPTUNE_MODEL_PREREQUISITES = {'unfreeze_backbone': 'vjepa2_vitl'}

    def press_run(self, cfg, set_control=None):
        app = self.app
        app._current_run = None
        app.apply_config(cfg)
        if set_control is not None:
            set_control()
        app.on_config_changed()
        self.captured.clear()
        app.on_run_clicked()
        handle, app._current_run = app._current_run, None
        if handle is not None:
            handle.thread.join(timeout=10)
        if not self.captured:
            return None, self.dpg.get_value('validation_text')
        import deeptune
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                deeptune.parse_args(self.captured[-1])
            except SystemExit:
                return None, f'deeptune.parse_args refused: {err.getvalue().strip().splitlines()[-1:]}'
        return self.captured[-1], ''

    # --- infer.py: the Test page, Test model ---------------------------------------------------------------------------
    def press_test(self, ensemble: bool, set_control=None):
        from desktop import option_widgets as ow
        app = self.app
        app._current_infer_run = None
        app._test_run_history = {'run': self.run}
        app._selected_test_run = None
        app.on_test_run_selected(user_data='run')
        ow.apply(co.INFER_OPTIONS, 'infer', {'ensemble': self.value_override('infer.py', 'ensemble')} if ensemble else {})
        self.dpg.set_value('infer_confirm_model_overrides', True)
        self.dpg.set_value('test_batch_size_input', 16)
        self.dpg.set_value('test_data_path_input', str(self.tmp / 'new_clips'))
        if set_control is not None:
            set_control()
        self.captured.clear()
        app.on_run_test_clicked()
        handle, app._current_infer_run = app._current_infer_run, None
        if handle is not None:
            handle.thread.join(timeout=10)
        return (self.captured[-1], '') if self.captured else (None, self.dpg.get_value('test_stage_text'))

    # --- track.py: the Analyse page, Find, track and count -----------------------------------------------------------
    def press_track(self, set_control=None):
        from desktop import option_widgets as ow
        page = self.analyse_page
        page._active = None
        self.dpg.set_value('outdir_input', str(self.tmp / 'out'))
        self.dpg.set_value(page.VIDEO_INPUT, str(self.clip))
        self.dpg.set_value('analyse_prompt_input', 'person')
        self.dpg.set_value('analyse_detector_combo', self.analyse.DETECTORS[0].label)
        self.dpg.set_value('analyse_every_n_input', 1)
        ow.apply(co.TRACK_OPTIONS, page.TRACK_PREFIX, {})
        if set_control is not None:
            set_control()
        self.captured.clear()
        page.on_track_clicked()
        active, page._active = page._active, None
        if active is not None:
            active.handle.thread.join(timeout=10)
        return (self.captured[-1], '') if self.captured else (None, self.dpg.get_value('analyse_track_message'))

    # --- the tool pages ---------------------------------------------------------------------------------------------
    def tool_contexts(self, entry):
        t = self.tmp
        return {
            'select_model.py': [{'df': str(t / 'dev.parquet'), 'out': str(t / 'search'), 'backbones': ['resnet18']}],
            'ensemble.py': [{'out': str(t / 'ens'), 'runs': [str(t / 'member_a'), str(t / 'member_b')], 'data': str(t / 'new_clips')},
                            {'out': str(t / 'ens'), 'predictions': [str(t / 'pred_a.csv'), str(t / 'pred_b.csv')]}],
            'extract_features.py': [{'data': str(t / 'rows.csv'), 'out': str(t / 'features')}],
            'train_feature_head.py': [{'features': str(t / 'features'), 'train_df': str(t / 'rows.csv'), 'val_df': str(t / 'rows.csv'),
                                       'out': str(t / 'probe')}],
        }[entry]

    SELECT_PREREQUISITES = {'n_trials': {'search': 'random'}, 'bayes_startup': {'search': 'bayes'},
                            'augmentations': {'allow_flip': True}}

    def press_tool(self, entry, values, set_control=None, check_grid=False):
        from desktop import option_widgets as ow
        tool = TOOL[entry]
        pages = self.tool_pages
        pages._running.clear()
        ow.apply(SPECS[entry], PAGE_PREFIXES[entry], values)
        if set_control is not None:
            set_control()
        self.captured.clear()
        if check_grid:
            pages.on_check_grid()
        else:
            pages.start(tool)
        page = pages.PAGE[tool]
        run = pages._running.pop(page, None)
        if run is not None:
            run.handle.thread.join(timeout=10)
        return (self.captured[-1], '') if self.captured else (None, self.dpg.get_value(f'{page}_status'))


@pytest.fixture(scope='module')
def studio(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        built = Studio(tmp_path_factory.mktemp('studio'), patch)
        try:
            yield built
        finally:
            built.close()


# --- per-flag check ------------------------------------------------------------------------------------------------
def _field_bindings(studio):
    """Controls that predate G1: (set the control, value the parser must return, or a check of it)."""
    t, app, dpg, page = studio.tmp, studio.app, studio.dpg, studio.analyse_page
    owlv2 = next(d.label for d in studio.analyse.DETECTORS if d.key == 'owlv2')
    return {
        'deeptune.py': {
            '--modality': ('modality', 'images', 'images'), '--df': ('df', t / 'other.parquet', (t / 'other.parquet').resolve()),
            '--raw-data': ('raw_data', True, True), '--out': ('out', t / 'elsewhere', (t / 'elsewhere').resolve()),
            '--target': ('target', 'label_col', 'label_col'), '--grouper': ('grouper', 'subject', 'subject'),
            '--model_version': ('model_version', 'resnet50', 'resnet50'), '--num_classes': ('num_classes', 5, 5),
            '--mode': ('mode', 'reg', 'reg'), '--added_layers': ('added_layers', 1, 1), '--embed_size': ('embed_size', 64, 64),
            '--freeze-backbone': ('freeze_backbone', True, True), '--use-peft': ('use_peft', True, True),
            '--fixed-seed': ('fixed_seed', True, True), '--batch_size': ('batch_size', 7, 7), '--num_epochs': ('num_epochs', 3, 3),
            '--learning_rate': ('learning_rate', 0.003, 0.003), '--trainable_params': ('trainable_params', '100k', 100_000),
            '--num_frames': ('num_frames', 12, 12), '--pooling': ('pooling', 'attention', 'attention'),
            '--type': ('gandalf_type', 'regression', 'regression'), '--continuous_cols': ('continuous_cols', ['a', 'b'], ['a', 'b']),
            '--categorical_cols': ('categorical_cols', ['c'], ['c']), '--finetuning-mode': ('finetuning_mode', True, True),
            '--time_idx_column': ('time_idx_column', 'tcol', 'tcol'),
        },
        'infer.py': {
            '--modality': (None, 'video'), '--checkpoint': (None, studio.run / 'trainval_output_x' / 'model_weights.pth'),
            '--data': (lambda: dpg.set_value('test_data_path_input', str(t / 'images')), t / 'images'),
            '--out': (None, lambda out: out.parent == studio.run / 'desktop-inference'),
            '--batch_size': (lambda: dpg.set_value('test_batch_size_input', 7), 7),
        },
        'track.py': {
            '--video': (None, str(studio.clip)),
            '--prompt': (lambda: dpg.set_value('analyse_prompt_input', 'cat, dog'), ['cat', 'dog']),
            '--out': (None, lambda out: Path(out).parent == t / 'out' / 'analyse-video'),
            '--detector': (lambda: dpg.set_value('analyse_detector_combo', owlv2), 'owlv2'),
            '--every_n_frames': (lambda: dpg.set_value('analyse_every_n_input', 3), 3),
            '--save_video': (None, True),  # the Analyse page always asks for the overlay video
        },
    }


def _matches(got, expected) -> bool:
    return expected(got) if callable(expected) else got == expected


def _check(entry, action, argv, expected) -> str:
    if not _flag_in(argv, action):
        return 'not in argv'
    parsed = _quiet_parse(PARSERS[entry], argv)
    if parsed is None:
        return f'real parser refused {argv}'
    got = getattr(parsed, action.dest)
    return '' if _matches(got, expected) else f'parsed {got!r}, expected {expected!r}'


def check_flag(studio, entry, action) -> str:
    """'' if pressing the page's Start button sends `action` with the value its control was set to, else why not."""
    flag = action.option_strings[0]
    spec = _spec(entry, action)
    bindings = _field_bindings(studio).get(entry, {})
    binding = next((bindings[f] for f in action.option_strings if f in bindings), None)
    if binding is None and spec is None:
        return 'no GUI control'
    if spec is not None:
        value, expected = gui_value(spec, action)
        override = studio.value_override(entry, spec.key)
        if override is not None:
            value = override
            expected = [_convert(action, str(v)) for v in value] if isinstance(value, list) else _convert(action, str(value))
        prefix = PAGE_PREFIXES[entry]
        set_control = lambda: set_widget(studio.dpg, prefix, spec, value)  # noqa: E731
    tried = []
    if entry == 'deeptune.py':
        for modality, context in studio.deeptune_contexts():
            if flag not in NOT_VIDEO and modality != 'video':
                continue  # a flag that applies to video must work in a video run
            if binding is not None:
                field, value, expected = binding
                argv, why = studio.press_run(replace(context, **{field: value}))
            else:
                context = replace(context, options=dict(studio.DEEPTUNE_PREREQUISITES.get(spec.key, {})))
                if spec.key in studio.DEEPTUNE_MODEL_PREREQUISITES:
                    context = replace(context, model_version=studio.DEEPTUNE_MODEL_PREREQUISITES[spec.key])
                argv, why = studio.press_run(context, set_control)
            if argv is None:
                tried.append(f'{modality}: not launched: {why[:160]}')
                continue
            problem = _check(entry, action, argv, expected)
            if not problem:
                return ''
            tried.append(f'{modality}: {problem}')
        return '; '.join(tried) or 'no video context tried'
    if entry == 'infer.py':
        for ensemble in (False, True):
            if binding is not None:
                control, expected = binding
                argv, why = studio.press_test(ensemble, control)
            else:
                argv, why = studio.press_test(ensemble, set_control)
            if argv is None:
                tried.append(f'not launched: {why[:160]}')
                continue
            problem = _check(entry, action, argv, expected)
            if not problem:
                return ''
            tried.append(problem)
        return '; '.join(tried)
    if entry == 'track.py':
        control, expected = binding if binding is not None else (set_control, expected)
        argv, why = studio.press_track(control)
        return f'not launched: {why[:160]}' if argv is None else _check(entry, action, argv, expected)
    for context in studio.tool_contexts(entry):
        values = dict(context)
        if entry == 'select_model.py':
            values.update(studio.SELECT_PREREQUISITES.get(spec.key, {}))
        argv, why = studio.press_tool(entry, values, None if spec.hidden else set_control, check_grid=spec.hidden)
        if argv is None:
            tried.append(f'not launched: {why[:160]}')
            continue
        problem = _check(entry, action, argv, expected)
        if not problem:
            return ''
        tried.append(problem)
    return '; '.join(tried)


@pytest.mark.parametrize('entry', list(PARSERS))
def test_every_flag_reaches_the_argv_its_start_button_sends(studio, entry):
    failures = {action.option_strings[0]: problem for action in flags_of(PARSERS[entry])
                if (entry, action.option_strings[0]) not in ALLOW_LIST and (problem := check_flag(studio, entry, action))}
    assert not failures, f'{entry}: {len(failures)} flag(s) not reachable from the GUI: {failures}'


def test_the_allow_list_is_short_documented_and_current():
    assert len(ALLOW_LIST) <= 10
    flags = {(entry, action.option_strings[0]) for entry, parser in PARSERS.items() for action in flags_of(parser)}
    for key, reason in ALLOW_LIST.items():
        assert key in flags, f'{key} is allow-listed but no longer a flag'
        assert isinstance(reason, str) and len(reason) > 20, key
    assert not any(opt.flag == '--config' for opt in co.DEEPTUNE_OPTIONS), 'allow-listed, so no control may send it'


def test_the_check_is_not_vacuous(studio):
    """A page that drops a flag, or sends the wrong value, must be caught."""
    action = next(a for a in flags_of(PARSERS['deeptune.py']) if '--video_augment' in a.option_strings)
    assert check_flag(studio, 'deeptune.py', action) == ''
    original = co.DEEPTUNE_OPTIONS
    try:
        co.DEEPTUNE_OPTIONS = tuple(opt for opt in original if opt.key != 'video_augment')
        assert 'not in argv' in check_flag(studio, 'deeptune.py', action)
    finally:
        co.DEEPTUNE_OPTIONS = original


def test_spec_defaults_and_choices_match_the_cli():
    """'Keep the defaults identical to the CLI defaults': a control starts at the parser's default, and offers its
    choices. Override controls (Test page, blank = what the run recorded) start blank on purpose."""
    empty = (None, False, (), [])
    mismatches = []
    for entry, specs in SPECS.items():
        actions = {flag: action for action in flags_of(PARSERS[entry]) for flag in action.option_strings}
        for opt in specs:
            action = actions[opt.flag]
            if opt.section in co.OVERRIDE_SECTIONS or opt.required:
                continue
            default = action.default
            same = (opt.default in empty and default in empty) or opt.default == default or (
                opt.kind in co.LIST_KINDS and list(opt.default or ()) == list(default or ()))
            if not same and opt.kind == co.BOOL and action.const is False:
                same = opt.default is False  # store_false: the box says "skip", unticked = the CLI default
            if not same:
                mismatches.append(f'{entry} {opt.flag}: GUI {opt.default!r}, CLI {default!r}')
            if action.choices is not None and opt.kind in (co.CHOICE, co.MULTI):
                if set(map(str, opt.choices)) != set(map(str, action.choices)):
                    mismatches.append(f'{entry} {opt.flag}: GUI choices {opt.choices}, CLI {list(action.choices)}')
    assert not mismatches, mismatches


def test_gui_constants_match_their_sources():
    import select_model
    from desktop import tools
    from handlers.model_selection.grid import AUGMENTATIONS
    from handlers.model_selection.ranking import METRICS
    from handlers.model_selection.search import STRATEGIES
    from handlers.video_arch import VIDEO_ARCH_CHOICES
    from handlers.video_backbones import HUB_BACKBONES
    assert co.STRATEGIES == tuple(sorted(STRATEGIES))
    assert co.AUGMENTATIONS == tuple(AUGMENTATIONS)
    assert co.SELECTION_METRICS == (*METRICS['cls'], *METRICS['reg'])
    assert co.VIDEO_ARCH_CHOICES == tuple(VIDEO_ARCH_CHOICES)
    assert set(co.HUB_LICENCES) == {b.licence for b in HUB_BACKBONES.values() if b.opt_in}
    assert tools.SELECT_STRATEGY_FLAGS == select_model.STRATEGY_FLAGS


def test_every_control_is_a_real_widget_that_reads_back_its_value(studio):
    from desktop import option_widgets as ow
    for entry, specs in SPECS.items():
        actions = {flag: action for action in flags_of(PARSERS[entry]) for flag in action.option_strings}
        prefix = PAGE_PREFIXES[entry]
        expected = {}
        for opt in specs:
            if opt.hidden:
                continue
            value, _ = gui_value(opt, actions[opt.flag])
            set_widget(studio.dpg, prefix, opt, value)
            expected[opt.key] = value
        values, problems = ow.read(specs, prefix)
        assert not problems, (entry, problems)
        assert values == expected, entry
        ow.apply(specs, prefix, {})
    hidden = [opt.flag for specs in SPECS.values() for opt in specs if opt.hidden]
    assert hidden == ['--dry_run']  # set by the Check grid button


# --- the GUI process stays torch-free ----------------------------------------------------------------------------------
def test_building_the_whole_gui_does_not_import_torch():
    code = ("import sys\n"
            "from desktop import app\n"
            "app.dpg.create_context()\n"
            "app.build_ui()\n"
            "for key, _ in app.NAV_ITEMS:\n"
            "    app.on_nav_selected(user_data=key)\n"
            "for modality in app.gc.MODALITIES:\n"
            "    app.dpg.set_value('modality_radio', modality)\n"
            "    app.on_modality_changed()\n"
            "app.on_config_changed()\n"
            "from desktop import tool_pages\n"
            "tool_pages.on_changed()\n"
            "app.dpg.destroy_context()\n"
            "bad = [m for m in ('torch', 'torchvision', 'transformers') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "print('ok')\n")
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('ok')

"""G1: every flag of every entry point is reachable from the desktop GUI.

For each argparse flag of deeptune.py, infer.py, select_model.py, ensemble.py, extract_features.py,
train_feature_head.py and track.py, the test finds the GUI control that sets it, sets that control to a value that
differs from the CLI default, runs the argv builder the GUI itself calls (desktop/config.build_argv,
build_infer_argv, desktop/tools.build_tool_argv, desktop/analyse.build_track_argv), and parses the result with the
entry point's real parser. The flag must be in the argv, and must parse back to the value that was set.

The list of flags comes from the parsers, not from the GUI, so a new flag without a control fails here. The only
exceptions are ALLOW_LIST below, each with its reason. A second test drives the same values through the real widgets
(headless Dear PyGui), so a control that exists in a spec but not on screen fails too.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from desktop import analyse
from desktop import cli_options as co
from desktop import config as gc
from desktop import tools

REPO = Path(__file__).resolve().parents[1]

# (entry point, flag) -> why the GUI does not offer it. Keep this short; every entry needs a reason.
ALLOW_LIST: dict = {}


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


def flags_of(parser: argparse.ArgumentParser) -> list:
    return [action for action in parser._actions if action.option_strings and not isinstance(action, argparse._HelpAction)]


# --- the GUI's argv builders, in the settings ("contexts") where a flag can apply --------------------------------------
BASE_RUN = dict(df=Path('data/clips.parquet'), raw_data=False, out=Path('runs'), num_classes=3)
DEEPTUNE_CONTEXTS = [
    gc.RunConfig(modality='video', model_version='resnet18', **BASE_RUN),
    gc.RunConfig(modality='images', model_version='resnet18', **BASE_RUN),
    gc.RunConfig(modality='tabular', model_version='gandalf', continuous_cols=['x'], **BASE_RUN),
    gc.RunConfig(modality='tabular', model_version='tabpfn', **BASE_RUN),
    gc.RunConfig(modality='timeseries', model_version='deepAR', **BASE_RUN),
]
INFER_CONTEXTS = [
    gc.InferenceConfig(run_dir=Path('run'), modality='video', checkpoint_path=Path('run/train/model_weights.pth'),
                       data=Path('clips'), out=Path('pred'), model_version='resnet18'),
    gc.InferenceConfig(run_dir=Path('run'), modality='video', checkpoint_path=Path('run/train/model_weights.pth'),
                       data=Path('clips'), out=Path('pred'), options={'ensemble': ['run_a', 'run_b']}),
]
TOOL_CONTEXTS = {
    'select_model.py': [tools.ToolConfig('select_model', {'df': 'dev.parquet', 'out': 'search'})],
    'ensemble.py': [tools.ToolConfig('ensemble', {'out': 'ens', 'runs': ['run_a', 'run_b'], 'data': 'clips'}),
                    tools.ToolConfig('ensemble', {'out': 'ens', 'predictions': ['a.csv', 'b.csv']})],
    'extract_features.py': [tools.ToolConfig('extract_features', {'data': 'clips.csv', 'out': 'features'})],
    'train_feature_head.py': [tools.ToolConfig('train_feature_head', {'features': 'f', 'train_df': 't.csv', 'val_df': 'v.csv',
                                                                       'out': 'probe'})],
}
TRACK_CONTEXTS = [analyse.TrackConfig(video=Path('clip.mp4'), prompts=['person'], out=Path('tracked'))]
SPECS = {'deeptune.py': co.DEEPTUNE_OPTIONS, 'infer.py': co.INFER_OPTIONS, 'select_model.py': co.SELECT_OPTIONS,
         'ensemble.py': co.ENSEMBLE_OPTIONS, 'extract_features.py': co.EXTRACT_OPTIONS,
         'train_feature_head.py': co.PROBE_OPTIONS, 'track.py': co.TRACK_OPTIONS}


def _with_option(cfg, key, value):
    if isinstance(cfg, tools.ToolConfig):
        return tools.ToolConfig(cfg.tool, {**cfg.values, key: value})
    return replace(cfg, options={**(cfg.options or {}), key: value})


def _argv(entry: str, cfg) -> list:
    if entry == 'deeptune.py':
        return gc.build_argv(cfg)
    if entry == 'infer.py':
        return gc.build_infer_argv(cfg)
    if entry == 'track.py':
        return analyse.build_track_argv(cfg)
    return tools.build_tool_argv(cfg)


def _contexts(entry: str) -> list:
    return {'deeptune.py': DEEPTUNE_CONTEXTS, 'infer.py': INFER_CONTEXTS, 'track.py': TRACK_CONTEXTS}.get(entry) or TOOL_CONTEXTS[entry]


# Controls that existed before G1 are fields of the config dataclasses rather than specs: (field, value set, value parsed).
FIELD_BINDINGS = {
    'deeptune.py': {
        '--modality': ('modality', 'images', 'images'), '--df': ('df', Path('other.parquet'), Path('other.parquet')),
        '--raw-data': ('raw_data', True, True), '--out': ('out', Path('elsewhere'), Path('elsewhere')),
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
        '--modality': ('modality', 'video', 'video'), '--checkpoint': ('checkpoint_path', Path('w.pth'), Path('w.pth')),
        '--data': ('data', Path('new'), Path('new')), '--out': ('out', Path('p'), Path('p')), '--batch_size': ('batch_size', 7, 7),
    },
    'track.py': {
        '--video': ('video', Path('other.mp4'), 'other.mp4'), '--prompt': ('prompts', ['cat', 'dog'], ['cat', 'dog']),
        '--out': ('out', Path('o2'), 'o2'), '--detector': ('detector', 'owlv2', 'owlv2'),
        '--every_n_frames': ('every_n_frames', 3, 3),
        '--save_video': (None, None, True),  # the Analyse page always asks for the overlay video
    },
}
# Where the GUI shows each field-bound control (checked against the real widgets below).
FIELD_WIDGETS = {
    'deeptune.py': 'desktop.app.CONFIG_TAGS',
    'infer.py': {'modality': 'test_run_combo', 'checkpoint_path': 'test_run_combo', 'data': 'test_data_path_input',
                 'out': 'test_run_button', 'batch_size': 'test_batch_size_input'},
    'track.py': {'video': 'analyse_video_input', 'prompts': 'analyse_prompt_input', 'out': 'analyse_track_button',
                 'detector': 'analyse_detector_combo', 'every_n_frames': 'analyse_every_n_input', None: 'analyse_track_button'},
}


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
    if opt.kind == co.BOOL:
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


def _parse(parser, argv):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            return parser.parse_args(argv)
        except SystemExit:
            return None


def _flag_in(argv, action) -> bool:
    return any(word in action.option_strings or word.split('=', 1)[0] in action.option_strings for word in argv)


def check_flag(entry: str, action) -> str:
    """'' if the GUI reaches `action` with a non-default value that parses back, else what went wrong."""
    parser = PARSERS[entry]
    binding = next((f for f in action.option_strings if f in FIELD_BINDINGS.get(entry, {})), None)
    spec = next((opt for opt in SPECS[entry] if opt.flag in action.option_strings), None)
    if binding is None and spec is None:
        return 'no GUI control'
    tried = []
    for context in _contexts(entry):
        if binding is not None:
            field, value, expected = FIELD_BINDINGS[entry][binding]
            cfg = replace(context, **{field: value}) if field else context
        else:
            value, expected = gui_value(spec, action)
            cfg = _with_option(context, spec.key, value)
        try:
            argv = _argv(entry, cfg)
        except ValueError as exc:
            tried.append(f'builder refused: {exc}')
            continue
        if not _flag_in(argv, action):
            tried.append('not in argv')
            continue
        parsed = _parse(parser, argv)
        if parsed is None:
            tried.append(f'real parser refused {argv}')
            continue
        got = getattr(parsed, action.dest)
        if got != expected:
            tried.append(f'parsed {got!r}, expected {expected!r}')
            continue
        return ''
    return '; '.join(tried)


def _all_flags():
    return [(entry, action) for entry, parser in PARSERS.items() for action in flags_of(parser)]


@pytest.mark.parametrize('entry', list(PARSERS))
def test_every_flag_of_every_entry_point_is_reachable_from_the_gui(entry):
    failures = {action.option_strings[0]: problem for action in flags_of(PARSERS[entry])
                if (entry, action.option_strings[0]) not in ALLOW_LIST and (problem := check_flag(entry, action))}
    assert not failures, f'{entry}: {len(failures)} flag(s) not reachable from the GUI: {failures}'


def test_the_allow_list_is_short_documented_and_current():
    assert len(ALLOW_LIST) <= 10
    flags = {(entry, action.option_strings[0]) for entry, action in _all_flags()}
    for key, reason in ALLOW_LIST.items():
        assert key in flags, f'{key} is allow-listed but no longer a flag'
        assert isinstance(reason, str) and len(reason) > 20, key


def test_the_check_is_not_vacuous():
    """A builder that drops a flag, or sends the wrong value, must be caught."""
    action = next(a for a in flags_of(PARSERS['deeptune.py']) if '--video_augment' in a.option_strings)
    assert check_flag('deeptune.py', action) == ''
    original = co.DEEPTUNE_OPTIONS
    try:
        co.DEEPTUNE_OPTIONS = tuple(opt for opt in original if opt.key != 'video_augment')
        assert check_flag('deeptune.py', action) != ''  # the builder no longer emits it
    finally:
        co.DEEPTUNE_OPTIONS = original
    lr = next(a for a in flags_of(PARSERS['deeptune.py']) if '--learning_rate' in a.option_strings)
    FIELD_BINDINGS['deeptune.py']['--learning_rate'] = ('learning_rate', 0.003, 0.004)
    try:
        assert 'expected' in check_flag('deeptune.py', lr)
    finally:
        FIELD_BINDINGS['deeptune.py']['--learning_rate'] = ('learning_rate', 0.003, 0.003)


def test_spec_defaults_and_choices_match_the_cli():
    """'Keep the defaults identical to the CLI defaults': a control starts at the parser's default, and offers its
    choices. Override controls (Test page, blank = what the run recorded) start blank on purpose."""
    empty = (None, False, (), [])
    mismatches = []
    for entry, specs in SPECS.items():
        actions = {flag: action for action in flags_of(PARSERS[entry]) for flag in action.option_strings}
        for opt in specs:
            action = actions[opt.flag]
            if opt.section == co.S_OVERRIDE or opt.required:
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


# --- the same values through the real widgets ------------------------------------------------------------------------
PAGE_PREFIXES = {'deeptune.py': 'dt', 'infer.py': 'infer', 'select_model.py': 'sel', 'ensemble.py': 'ens',
                 'extract_features.py': 'feat', 'train_feature_head.py': 'probe', 'track.py': 'track'}


def _set_widget(dpg, prefix, opt, value):
    from desktop import option_widgets as ow
    tag = ow.tag(prefix, opt)
    if opt.kind == co.MULTI:
        for choice in opt.choices:
            dpg.set_value(f'{tag}__{choice}', choice in value)
    elif opt.kind in (co.WORDS,):
        dpg.set_value(tag, ' '.join(value))
    elif opt.kind == co.LINES:
        dpg.set_value(tag, '\n'.join(value))
    elif opt.kind in (co.TEXT, co.OPT_INT, co.OPT_FLOAT):
        dpg.set_value(tag, co.format_value(value))
    elif opt.kind == co.CHOICE:
        dpg.set_value(tag, str(value))
    else:
        dpg.set_value(tag, value)


@pytest.fixture
def gui():
    from desktop import app
    app.dpg.create_context()
    app.build_ui()
    try:
        yield app
    finally:
        from desktop import tool_pages
        tool_pages.shutdown()
        app.dpg.destroy_context()


def test_every_control_is_a_real_widget_that_reads_back_its_value(gui):
    from desktop import option_widgets as ow
    dpg = gui.dpg
    for entry, specs in SPECS.items():
        actions = {flag: action for action in flags_of(PARSERS[entry]) for flag in action.option_strings}
        prefix = PAGE_PREFIXES[entry]
        expected = {}
        for opt in specs:
            if opt.hidden:
                continue
            value, _ = gui_value(opt, actions[opt.flag])
            _set_widget(dpg, prefix, opt, value)
            expected[opt.key] = value
        values, problems = ow.read(specs, prefix)
        assert not problems, (entry, problems)
        assert values == expected, entry
    hidden = [opt.flag for specs in SPECS.values() for opt in specs if opt.hidden]
    assert hidden == ['--dry_run']  # set by the Check grid button: tests/test_gui_pages.py


def test_field_bound_controls_are_widgets(gui):
    for entry, widgets in FIELD_WIDGETS.items():
        if isinstance(widgets, str):
            widgets = {field: tag for field, tag in gui.CONFIG_TAGS.items()}
            fields = {binding[0] for binding in FIELD_BINDINGS[entry].values()}
            assert fields <= set(widgets), fields - set(widgets)
        for tag in widgets.values():
            assert gui.dpg.does_item_exist(tag), (entry, tag)


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

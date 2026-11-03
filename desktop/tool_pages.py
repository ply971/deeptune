"""G1's new pages: Model Selection (select_model.py), Ensemble (ensemble.py) and Features & Probes
(extract_features.py, train_feature_head.py).

Every control comes from a desktop/cli_options spec, so each flag of these scripts has a widget (checked by
tests/test_gui_coverage.py). Each launch runs the script as a subprocess through desktop/runner.start_tool_run; this
process never imports torch. Results are read back from the output folder by desktop/tools.py.

Like analyse_page.py, the page borrows app.py's building blocks through a Shell instead of importing app.py.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import dearpygui.dearpygui as dpg
import pandas as pd

from desktop import cli_options as co
from desktop import config as gc
from desktop import dpg_context as ui
from desktop import option_widgets as ow
from desktop import tools
from desktop.log_stream import LineBuffer
from desktop import runner
from desktop.runner import RunHandle, start_tool_run, tool_argv

PREFIX = {'select_model': 'sel', 'ensemble': 'ens', 'extract_features': 'feat', 'train_feature_head': 'probe'}
PAGE = {'select_model': 'select', 'ensemble': 'ensemble', 'extract_features': 'features', 'train_feature_head': 'features'}
LOG = {'select': 'select_log', 'ensemble': 'ensemble_log', 'features': 'features_log'}
RESULTS = {'select': 'select_results_container', 'ensemble': 'ensemble_results_container', 'features': 'features_results_container'}


@dataclass
class _Run:
    tool: str
    handle: RunHandle
    started: float
    out: Optional[Path]
    dry_run: bool = False
    argv: list = field(default_factory=list)
    device: str = 'auto'


_shell = None
_running: dict = {}  # page -> _Run
_buffers: dict = {page: LineBuffer() for page in LOG}
# select_model.py: the argv of the last dry run that passed. Start search is enabled only for those exact settings,
# so the counts and time estimate are always seen before a search trains anything. The device it ran on counts too: the
# estimate and the memory checks depend on it.
_checked_select_argv: Optional[list] = None
_checked_select_device: Optional[str] = None


def _colors():
    return _shell.palette


def config(tool: str) -> tools.ToolConfig:
    values, _ = ow.read(tools.TOOLS[tool].specs, PREFIX[tool])
    return tools.ToolConfig(tool, values)


def problems(tool: str) -> list[str]:
    values, found = ow.read(tools.TOOLS[tool].specs, PREFIX[tool])
    return found + tools.validate_tool(tools.ToolConfig(tool, values))


def command_line(tool: str, extra: Optional[list] = None) -> str:
    cfg = config(tool)
    script = Path(gc.__file__).resolve().parents[1] / cfg.spec.script
    return subprocess.list2cmdline([sys.executable, str(script), *tool_argv(cfg, extra)])


# --- launching --------------------------------------------------------------------------------------------------------
def start(tool: str, extra: Optional[list] = None) -> bool:
    page = PAGE[tool]
    if page in _running:
        return False
    found = problems(tool)
    status = f'{page}_status'
    if found:
        _say(status, found[0], _colors().warning)
        return False
    cfg = config(tool)
    try:
        handle = start_tool_run(cfg, extra)
    except Exception as exc:  # noqa: BLE001 - shown to the user, as on the other pages
        _say(status, str(exc), _colors().error)
        return False
    _buffers[page].clear()
    dpg.set_value(f'{LOG[page]}_box', '')
    _running[page] = _Run(tool, handle, time.monotonic(), tools.output_folder(cfg), dry_run='--dry_run' in (extra or []),
                          argv=tool_argv(cfg), device=runner.device())
    _say(status, f'Running {cfg.spec.script}...', _colors().analysis)
    _shell.set_status(f'{cfg.spec.title} in progress', _colors().analysis)
    _update_buttons()
    return True


def stop(page: str) -> None:
    run = _running.get(page)
    if run is not None and run.handle.cancel():
        _say(f'{page}_status', 'Stopping...', _colors().warning)


def poll() -> None:
    for page, run in list(_running.items()):
        changed = False
        for _ in range(300):
            try:
                _buffers[page].feed(run.handle.log_queue.get_nowait())
            except queue.Empty:
                break
            changed = True
        if changed:
            dpg.set_value(f'{LOG[page]}_box', _buffers[page].render())
            if dpg.get_value(f'{LOG[page]}_follow'):
                dpg.set_y_scroll(f'{LOG[page]}_window', dpg.get_y_scroll_max(f'{LOG[page]}_window'))
        seconds = int(time.monotonic() - run.started)
        dpg.set_value(f'{page}_elapsed', f'Elapsed  {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}')
        try:
            status, message = run.handle.done_queue.get_nowait()
        except queue.Empty:
            continue
        while True:
            try:
                _buffers[page].feed(run.handle.log_queue.get_nowait())
            except queue.Empty:
                break
        dpg.set_value(f'{LOG[page]}_box', _buffers[page].render())
        del _running[page]
        finished(page, run, status, message)


def finished(page: str, run: _Run, status: str, message: str) -> None:
    global _checked_select_argv, _checked_select_device
    colors = _colors()
    if status == 'success':
        if run.dry_run:
            _checked_select_argv, _checked_select_device = run.argv, run.device
            _say(f'{page}_status', 'Dry run passed: the counts and the time estimate are in the log above. Start search runs '
                                   'exactly these settings; change any setting and check again.', colors.success)
        else:
            _say(f'{page}_status', 'Complete.', colors.success)
        _shell.set_status(f'{tools.TOOLS[run.tool].title} complete', colors.success)
    elif status == 'cancelled':
        _say(f'{page}_status', 'Stopped. Partial output may remain in the output folder.', colors.warning)
        _shell.set_status('Stopped', colors.warning)
    else:
        _say(f'{page}_status', (message.splitlines() or ['Failed'])[0] + ' See the log.', colors.error)
        _shell.set_status(f'{tools.TOOLS[run.tool].title} failed - check the log', colors.error)
    if run.out is not None:
        show_results(run.tool, run.out)
    _update_buttons()


def shutdown() -> None:
    for run in list(_running.values()):
        run.handle.close()
    _running.clear()


# --- results ----------------------------------------------------------------------------------------------------------
def show_results(tool: str, out: Path) -> None:
    reader = {'select_model': tools.selection_sections, 'ensemble': tools.ensemble_sections,
              'extract_features': tools.feature_sections, 'train_feature_head': tools.probe_sections}[tool]
    try:
        sections = reader(out)
    except OSError as exc:
        sections = [tools.Section('Could not read', notes=[str(exc)])]
    render_sections(RESULTS[PAGE[tool]], sections, _colors(), f'Nothing to show yet in {out}.')


def render_sections(container: str, sections: list, colors, empty: str = 'Nothing to show yet.') -> None:
    """Draw desktop/tools.Section blocks: a title, label/value rows, notes and an optional table."""
    dpg.delete_item(container, children_only=True)
    if not sections:
        dpg.add_text(empty, parent=container, color=colors.muted, wrap=850)
        return
    for section in sections:
        dpg.add_text(section.title, parent=container, color=colors.analysis)
        if section.rows:
            with dpg.table(parent=container, header_row=False, row_background=True, borders_innerH=True, resizable=True):
                dpg.add_table_column()
                dpg.add_table_column()
                for label, value in section.rows:
                    with dpg.table_row():
                        dpg.add_text(str(label))
                        dpg.add_text(str(value))
        if section.table is not None and not section.table.empty:
            frame = section.table.head(200)
            with dpg.table(parent=container, header_row=True, resizable=True, borders_innerV=True, scrollY=True,
                           scrollX=True, height=min(320, 34 + 30 * len(frame)), policy=dpg.mvTable_SizingFixedFit):
                for column in frame.columns:
                    dpg.add_table_column(label=str(column))
                for _, row in frame.iterrows():
                    with dpg.table_row():
                        for column in frame.columns:
                            value = row[column]
                            dpg.add_text('' if pd.isna(value) else tools.number(value) if isinstance(value, float) else str(value))
        for note in section.notes:
            if note:
                dpg.add_text(str(note), parent=container, color=colors.muted, wrap=850)
        if section.path is not None:
            dpg.add_text(f'From {section.path}', parent=container, color=colors.muted, wrap=850)
        dpg.add_spacer(height=6, parent=container)


# --- small helpers ----------------------------------------------------------------------------------------------------
def _say(tag: str, message: str, color) -> None:
    dpg.set_value(tag, message)
    dpg.configure_item(tag, color=color)


def _update_buttons() -> None:
    select_running = 'select' in _running
    dpg.configure_item('select_check_button', enabled=not select_running)
    dpg.configure_item('select_start_button', enabled=not select_running and _select_checked())
    dpg.configure_item('select_stop_button', enabled=select_running)
    dpg.configure_item('ensemble_start_button', enabled='ensemble' not in _running)
    dpg.configure_item('ensemble_stop_button', enabled='ensemble' in _running)
    for tool in ('extract_features', 'train_feature_head'):
        dpg.configure_item(f'{tool}_start_button', enabled='features' not in _running)
    dpg.configure_item('features_stop_button', enabled='features' in _running)


def _select_checked() -> bool:
    try:
        return (_checked_select_argv is not None and tool_argv(config('select_model')) == _checked_select_argv
                and _checked_select_device == runner.device())
    except ValueError:
        return False


def on_changed(sender: Any = 0, app_data: Any = None, user_data: Any = None) -> None:
    """Any control changed: refresh the command line, the readiness text and the buttons."""
    for tool, page, tag in (('select_model', 'select', 'select_command'), ('ensemble', 'ensemble', 'ensemble_command'),
                            ('extract_features', 'features', 'extract_features_command'),
                            ('train_feature_head', 'features', 'train_feature_head_command')):
        try:
            dpg.set_value(tag, command_line(tool))
        except ValueError as exc:
            dpg.set_value(tag, str(exc))
    found = problems('select_model')
    resumable = tools.is_resumable(config('select_model').get('out'))
    if found:
        message, color = found[0], _colors().warning
    elif _select_checked():
        message, color = 'Checked. Start search runs these settings.', _colors().success
    else:
        message, color = ('Check the grid first: the dry run prints the candidate count and the time estimate.'
                          + (' This folder already holds a search, which will resume.' if resumable else '')), _colors().analysis
    if 'select' not in _running:
        _say('select_status', message, color)
    _update_buttons()


def _open_folder(path) -> None:
    try:
        if path is None or not Path(path).expanduser().is_dir():
            raise ValueError('Run it first, or choose an existing output folder.')
        target = str(Path(path).expanduser().resolve())
        if hasattr(os, 'startfile'):
            os.startfile(target)
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', target])
    except (OSError, ValueError) as exc:
        _shell.set_status(str(exc), _colors().error)


def on_check_grid(sender: Any = 0, app_data: Any = None) -> None:
    start('select_model', ['--dry_run'])


def on_start_search(sender: Any = 0, app_data: Any = None) -> None:
    if not _select_checked():
        _say('select_status', 'Check the grid first (dry run) with these exact settings.', _colors().warning)
        _update_buttons()
        return
    start('select_model')


def on_refresh(sender: Any = 0, app_data: Any = None, user_data: str = 'select_model') -> None:
    out = tools.output_folder(config(user_data))
    if out is None:
        _shell.set_status('Choose the output folder first.', _colors().warning)
        return
    show_results(user_data, out)


def on_fill_runs(sender: Any = 0, app_data: Any = None) -> None:
    """Fill the ensemble's runs with every video run under the Run page's output folder that has weights."""
    base = (_shell.output_folder() or '').strip()
    if not base:
        _shell.set_status('Choose an output folder on the Run page first.', _colors().warning)
        return
    members = []
    for run_dir in gc.list_runs(Path(base).expanduser()):
        results = gc.read_run_results(run_dir)
        if results.checkpoint_path is not None and (results.cli_arguments or {}).get('modality') == 'video':
            members.append(str(results.checkpoint_path))
    spec = co.by_key(co.ENSEMBLE_OPTIONS)['runs']
    dpg.set_value(ow.tag('ens', spec), '\n'.join(members))
    _shell.set_status(f'{len(members)} video run(s) with weights found. Remove the ones you do not want.', _colors().success)
    on_changed()


# --- building ---------------------------------------------------------------------------------------------------------
def _launch_row(page: str, buttons: list) -> None:
    with ui.group(horizontal=True):
        for tag, label, callback, user_data, primary in buttons:
            button = dpg.add_button(tag=tag, label=label, callback=callback, user_data=user_data, height=44)
            if primary:
                dpg.bind_item_theme(button, _shell.themes['run_button'])
        dpg.add_text('Elapsed  00:00:00', tag=f'{page}_elapsed', color=_colors().muted)


def _log_card(page: str, out_getter) -> None:
    with _shell.card('LIVE OUTPUT'):
        with ui.group(horizontal=True):
            dpg.add_checkbox(tag=f'{LOG[page]}_follow', label='Follow output', default_value=True)
            dpg.add_button(label='Open output folder', callback=lambda: _open_folder(out_getter()))
        with ui.child_window(tag=f'{LOG[page]}_window', height=220, horizontal_scrollbar=True):
            dpg.add_text('', tag=f'{LOG[page]}_box', wrap=0)


def _command(tag: str) -> None:
    with ui.collapsing_header(label='Reproducible command'):
        dpg.add_input_text(tag=tag, readonly=True, multiline=True, width=-1, height=70)


def build(shell) -> None:
    global _shell, _checked_select_argv
    _shell = shell
    _running.clear()
    _checked_select_argv = None
    colors = shell.palette
    with ui.group(tag='page_select', show=False):
        shell.heading('Model selection', 'Pick the best video model of a grid by grouped cross-validation. A sealed test set '
                                         'is set aside first, and only the final winner is scored on it, once.')
        with shell.card('SEARCH'):
            ow.render(co.SELECT_OPTIONS, 'sel', on_changed, open_sections=(co.G_DATA, co.G_GRID), sections=co.SELECT_SECTIONS)
        dpg.add_spacer(height=10)
        with shell.card('CHECK, THEN SEARCH'):
            dpg.add_text('1. Check the grid: a dry run prints the candidate count and a rough time estimate, and trains '
                         'nothing.  2. Start the search with the same settings. Rerunning into the same folder resumes.',
                         color=colors.muted, wrap=850)
            _launch_row('select', [('select_check_button', 'Check grid and estimate (dry run)', on_check_grid, None, False),
                                   ('select_start_button', 'Start search', on_start_search, None, True),
                                   ('select_stop_button', 'Stop', lambda: stop('select'), None, False)])
            dpg.add_text('', tag='select_status', color=colors.analysis, wrap=850)
            _command('select_command')
        dpg.add_spacer(height=10)
        _log_card('select', lambda: tools.output_folder(config('select_model')))
        dpg.add_spacer(height=10)
        with shell.card('RESULTS'):
            dpg.add_button(label='Refresh', callback=on_refresh, user_data='select_model')
            with ui.child_window(tag=RESULTS['select'], height=420, horizontal_scrollbar=True):
                dpg.add_text('The sealed-test score, the winner and the leaderboard appear here.', color=colors.muted)

    with ui.group(tag='page_ensemble', show=False):
        shell.heading('Ensemble', 'Average class probabilities across trained video runs, or across finished predictions '
                                  'tables. Class indices are verified unless you allow otherwise.')
        with shell.card('MEMBERS AND DATA'):
            dpg.add_button(label='Fill runs from the output folder', callback=on_fill_runs)
            ow.render(co.ENSEMBLE_OPTIONS, 'ens', on_changed, headers=False)
        dpg.add_spacer(height=10)
        with shell.card('LAUNCH'):
            _launch_row('ensemble', [('ensemble_start_button', 'Run ensemble', lambda: start('ensemble'), None, True),
                                     ('ensemble_stop_button', 'Stop', lambda: stop('ensemble'), None, False)])
            dpg.add_text('', tag='ensemble_status', color=colors.analysis, wrap=850)
            _command('ensemble_command')
        dpg.add_spacer(height=10)
        _log_card('ensemble', lambda: tools.output_folder(config('ensemble')))
        dpg.add_spacer(height=10)
        with shell.card('RESULTS'):
            dpg.add_button(label='Refresh', callback=on_refresh, user_data='ensemble')
            with ui.child_window(tag=RESULTS['ensemble'], height=360, horizontal_scrollbar=True):
                dpg.add_text('ensemble.json, the metrics and the predictions appear here.', color=colors.muted)

    with ui.group(tag='page_features', show=False):
        shell.heading('Features & probes', 'Decode once, extract features once: cache snippet features with a frozen '
                                           'backbone, then train a light probe on them in seconds.')
        with shell.card('1. EXTRACT FEATURES'):
            ow.render(co.EXTRACT_OPTIONS, 'feat', on_changed, headers=False)
            _launch_row('features_extract', [('extract_features_start_button', 'Extract features',
                                              lambda: start('extract_features'), None, True)])
            _command('extract_features_command')
        dpg.add_spacer(height=10)
        with shell.card('2. TRAIN A PROBE'):
            ow.render(co.PROBE_OPTIONS, 'probe', on_changed, headers=False)
            _launch_row('features_probe', [('train_feature_head_start_button', 'Train probe',
                                            lambda: start('train_feature_head'), None, True)])
            _command('train_feature_head_command')
        dpg.add_spacer(height=10)
        with shell.card('STATUS'):
            with ui.group(horizontal=True):
                dpg.add_button(tag='features_stop_button', label='Stop', callback=lambda: stop('features'), enabled=False)
                dpg.add_text('Elapsed  00:00:00', tag='features_elapsed', color=colors.muted)
            dpg.add_text('', tag='features_status', color=colors.analysis, wrap=850)
        dpg.add_spacer(height=10)
        _log_card('features', lambda: (_running['features'].out if 'features' in _running else
                                       tools.output_folder(config('train_feature_head')) or tools.output_folder(config('extract_features'))))
        dpg.add_spacer(height=10)
        with shell.card('RESULTS'):
            with ui.group(horizontal=True):
                dpg.add_button(label='Show feature folder', callback=on_refresh, user_data='extract_features')
                dpg.add_button(label='Show probe results', callback=on_refresh, user_data='train_feature_head')
            with ui.child_window(tag=RESULTS['features'], height=320, horizontal_scrollbar=True):
                dpg.add_text('The feature folder and the probe metrics appear here.', color=colors.muted)
    on_changed()

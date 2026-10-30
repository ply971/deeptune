"""The Analyse Video page: give a video to a model you trained, or to a detector.

Each action runs as a subprocess (see runner.py), so this process never imports
torch. The logic that needs no window is in desktop/analyse.py.

To add an action: give it a card in build(), start it with _begin(kind, ...), and
add a branch for it to _FINISHERS. Only one action runs at a time on this page,
because each may load a model into the same 8 GB GPU.

app.py hands over its shared building blocks in a Shell, instead of this module
importing app.py. `python desktop/app.py` runs that file as __main__, so importing
it again here would create a second copy of its state.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Optional

import dearpygui.dearpygui as dpg

from desktop import analyse
from desktop import config as gc
from desktop import dpg_context as ui
from desktop.log_stream import LineBuffer
from desktop.runner import RunHandle, start_infer_run, start_track_run
from desktop.video_player import VideoPlayer
from handlers.video_manifest import VIDEO_EXTENSIONS

VIDEO_INPUT = 'analyse_video_input'
PREVIEW = 'analyse_preview_container'
TEXTURE = 'analyse_preview_texture'
IMAGE = 'analyse_preview_image'
VIDEO_DIALOG = 'file_dialog_analyse_video'
PREDICT, TRACK = 'predict', 'track'


@dataclass(frozen=True)
class Palette:
    accent: tuple[int, int, int]
    analysis: tuple[int, int, int]
    success: tuple[int, int, int]
    error: tuple[int, int, int]
    warning: tuple[int, int, int]
    primary: tuple[int, int, int]
    muted: tuple[int, int, int]


@dataclass(frozen=True)
class Shell:
    """What the page borrows from app.py, so it matches the rest of the studio."""
    card: Callable[..., ContextManager[Any]]
    heading: Callable[[str, str], None]
    set_status: Callable[[str, tuple[int, int, int]], None]
    dialog_path: Callable[[Any], Optional[str]]
    show_dialog: Callable[..., None]
    themes: dict
    output_folder: Callable[[], str]
    palette: Palette


@dataclass
class _Running:
    kind: str
    handle: RunHandle
    out_dir: Path
    video: Path
    started: float
    detector: str = ''
    run: Optional[gc.RunResults] = None


_shell: Optional[Shell] = None
_choices = analyse.RunChoices()
_selected_run: Optional[gc.RunResults] = None
_active: Optional[_Running] = None
_buffer = LineBuffer()
_last_out_dir: Optional[Path] = None
_prediction: Optional[analyse.PredictionView] = None
_tracking: Optional[analyse.TrackResults] = None

_player: Optional[VideoPlayer] = None
_texture_size: Optional[tuple[int, int]] = None
_original: Optional[Path] = None
_overlay: Optional[Path] = None
_showing_overlay = False


def _colors() -> Palette:
    assert _shell is not None
    return _shell.palette


def _message(kind: str, text: str, color: tuple[int, int, int]) -> None:
    dpg.set_value(f'analyse_{kind}_message', text)
    dpg.configure_item(f'analyse_{kind}_message', color=color)


def _problem(kind: str, text: str) -> None:
    """A problem with what the user entered: said next to the button they pressed."""
    _message(kind, text, _colors().warning)
    _shell.set_status(text.splitlines()[0][:120], _colors().warning)


# ---------------------------------------------------------------------------
# The video and its preview
# ---------------------------------------------------------------------------

def on_video_file_selected(sender: Any, app_data: Any) -> None:
    path = _shell.dialog_path(app_data)
    if path and _active is None:
        dpg.set_value(VIDEO_INPUT, path)
        on_load_video()


def on_load_video(sender: Any = 0, app_data: Any = None) -> None:
    """Check the typed path and preview it. Old results belong to the old video, so they go."""
    global _original
    if _active is not None:  # the running action's results belong to the current video
        return
    check = analyse.check_video(dpg.get_value(VIDEO_INPUT))
    _clear_prediction()
    _clear_tracking()
    _set_overlay_buttons()
    if not check.ok:
        _original = None
        _stop_player()
        dpg.set_value('analyse_video_status', check.error)
        dpg.configure_item('analyse_video_status', color=_colors().error)
        return
    _original = check.path
    dpg.set_value('analyse_video_status', check.description)
    dpg.configure_item('analyse_video_status', color=_colors().analysis)
    _play(check.path, overlay=False)


def _video_for_action(kind: str) -> Optional[Path]:
    """The video an action should use: what the box says now, even if never previewed."""
    check = analyse.check_video(dpg.get_value(VIDEO_INPUT))
    if not check.ok:
        _problem(kind, check.error)
        return None
    if check.path != _original:
        on_load_video()
    return check.path


def _page_is_open() -> bool:
    return bool(dpg.does_item_exist('nav_analyse') and dpg.get_value('nav_analyse'))


def _play(path: Path, overlay: bool) -> None:
    global _player, _showing_overlay
    _stop_player()
    _player = VideoPlayer(path, autoplay=_page_is_open(), loop=True)
    _showing_overlay = overlay
    dpg.configure_item('analyse_player_controls', show=True)
    dpg.set_value('analyse_showing_text', 'Showing: tracker overlay' if overlay else 'Showing: your video')
    _set_overlay_buttons()


def _stop_player() -> None:
    global _player, _texture_size
    if _player is not None:
        _player.close(timeout=0)
        _player = None
    _texture_size = None
    if dpg.does_item_exist(IMAGE):
        dpg.delete_item(IMAGE)
    if dpg.does_item_exist(TEXTURE):
        dpg.delete_item(TEXTURE)
    if dpg.does_item_exist('analyse_player_controls'):
        dpg.configure_item('analyse_player_controls', show=False)


def _set_overlay_buttons() -> None:
    # Hidden too: the theme draws a disabled button like an enabled one.
    has_overlay = _overlay is not None
    dpg.configure_item('analyse_show_original_button', show=has_overlay,
                       enabled=_showing_overlay and _original is not None)
    dpg.configure_item('analyse_show_overlay_button', show=has_overlay, enabled=has_overlay and not _showing_overlay)


def on_show_original(sender: Any = 0, app_data: Any = None) -> None:
    if _original is not None:
        _play(_original, overlay=False)


def on_show_overlay(sender: Any = 0, app_data: Any = None) -> None:
    if _overlay is not None:
        _play(_overlay, overlay=True)


def on_play_pause(sender: Any = 0, app_data: Any = None) -> None:
    if _player is not None:
        _player.pause() if _player.state.playing else _player.play()


def on_restart(sender: Any = 0, app_data: Any = None) -> None:
    if _player is not None:
        _player.seek(0)
        _player.play()


def on_seek(sender: Any = 0, app_data: Any = None) -> None:
    if _player is not None:
        _player.seek(float(app_data))


def fit_preview() -> None:
    """Fit the display only; the decoded frames stay untouched."""
    if not _texture_size or not dpg.does_item_exist(IMAGE):
        return
    width, height = _texture_size
    available = max(100, dpg.get_item_rect_size(PREVIEW)[0] - 44)
    scale = min(available / width, 360 / height, 1.0)
    dpg.configure_item(IMAGE, width=round(width * scale), height=round(height * scale))
    dpg.configure_item(PREVIEW, height=max(160, round(height * scale) + 44))


def _show_frame(width: int, height: int, pixels: Any) -> None:
    global _texture_size
    if _texture_size != (width, height) or not dpg.does_item_exist(TEXTURE):
        if dpg.does_item_exist(IMAGE):
            dpg.delete_item(IMAGE)
        if dpg.does_item_exist(TEXTURE):
            dpg.delete_item(TEXTURE)
        dpg.add_dynamic_texture(width, height, pixels, tag=TEXTURE, parent='preview_textures')
        _texture_size = (width, height)
    else:
        dpg.set_value(TEXTURE, pixels)
    if not dpg.does_item_exist(IMAGE):
        dpg.add_image(TEXTURE, tag=IMAGE, parent=PREVIEW)
    fit_preview()


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f'{minutes}:{seconds:02d}'


def _poll_player() -> None:
    if _player is None:
        return
    state, frame = _player.poll()
    if frame is not None:
        _show_frame(frame.width, frame.height, frame.pixels)
    ready = state.ready and not state.error
    dpg.configure_item('analyse_play_button', label='Pause' if state.playing else 'Play', enabled=ready)
    dpg.configure_item('analyse_restart_button', enabled=ready)
    dpg.configure_item('analyse_seek_slider', enabled=ready and state.duration > 0, max_value=max(state.duration, 0.01))
    if not dpg.is_item_active('analyse_seek_slider'):
        dpg.set_value('analyse_seek_slider', state.position)
    dpg.set_value('analyse_time_text',
                  f"{_clock(state.position)} / {_clock(state.duration) if state.duration else '--:--'}")
    if state.error:
        dpg.set_value('analyse_video_status', f'Could not play this video: {state.error}')
        dpg.configure_item('analyse_video_status', color=_colors().error)


# ---------------------------------------------------------------------------
# Trained runs
# ---------------------------------------------------------------------------

def on_refresh_runs(sender: Any = 0, app_data: Any = None) -> None:
    """List the video runs, keeping the current choice when it is still there."""
    global _choices
    if _active is not None:  # the running action's results belong to the current run
        return
    out_text = (_shell.output_folder() or '').strip()
    if not out_text:
        _choices = analyse.RunChoices()
        _show_run_list([], 'Choose an output folder on the Run page first.')
        return
    try:
        _choices = analyse.list_video_runs(Path(out_text).expanduser())
    except OSError as exc:
        _choices = analyse.RunChoices()
        _show_run_list([], f'Cannot read the output folder: {exc}')
        return
    _show_run_list(list(_choices.runs), _choices.message)


def _show_run_list(labels: list[str], note: str) -> None:
    global _selected_run
    dpg.configure_item('analyse_run_combo', items=labels)
    dpg.set_value('analyse_run_note', note)
    if not labels:
        _selected_run = None
        _clear_prediction()
        dpg.set_value('analyse_run_combo', '')
        dpg.set_value('analyse_run_info', 'No run chosen.')
        dpg.set_value('analyse_run_warning', '')
        return
    # By folder, not label: a label holds the folder's time, which a prediction changes.
    selected_dir = _selected_run.run_dir if _selected_run is not None else None
    kept = next((label for label, run_dir in _choices.runs.items() if run_dir == selected_dir), None)
    if kept is not None:
        dpg.set_value('analyse_run_combo', kept)
        _load_run(selected_dir)  # the record may have changed since it was read
        return
    dpg.set_value('analyse_run_combo', labels[0])
    on_run_selected()


def on_run_selected(sender: Any = 0, app_data: Any = None, user_data: Any = None) -> None:
    run_dir = _choices.runs.get(dpg.get_value('analyse_run_combo'))
    if run_dir is None or _active is not None:
        return
    _clear_prediction()
    _load_run(run_dir)


def _load_run(run_dir: Path) -> None:
    global _selected_run
    dpg.set_value('analyse_run_warning', '')
    try:
        _selected_run = gc.read_run_results(run_dir)
    except (OSError, ValueError) as exc:
        _selected_run = None
        dpg.set_value('analyse_run_info', f'Cannot read this run: {exc}')
        return
    problems = analyse.run_problems(_selected_run)
    if problems:
        dpg.set_value('analyse_run_info', '\n'.join(problems))
        return
    dpg.set_value('analyse_run_warning', '\n'.join(analyse.run_warnings(_selected_run)))
    info = analyse.describe_run(_selected_run)
    classes = analyse.class_names_note(_selected_run)
    dpg.set_value('analyse_run_info', f'{info}. {classes}' if classes else info)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _begin(kind: str, handle: RunHandle, out_dir: Path, video: Path, detector: str = '',
           run: Optional[gc.RunResults] = None) -> None:
    global _active, _last_out_dir
    _active = _Running(kind, handle, out_dir, video, time.monotonic(), detector, run)
    _last_out_dir = out_dir
    _message(kind, f'Starting on {video.name}...', _colors().analysis)
    _shell.set_status('Analysing video', _colors().analysis)
    _set_running(True)


def _set_running(running: bool) -> None:
    kind = _active.kind if _active is not None else None
    for name in (PREDICT, TRACK):
        dpg.configure_item(f'analyse_{name}_button', enabled=not running)
        dpg.configure_item(f'analyse_{name}_stop_button', enabled=running and name == kind)
    # A result belongs to the video and run it started with, so neither can change under it.
    for item in (VIDEO_INPUT, 'analyse_browse_button', 'analyse_load_button', 'analyse_run_combo',
                 'analyse_refresh_button'):
        dpg.configure_item(item, enabled=not running)
    dpg.configure_item('analyse_active_indicator', show=running)


def on_predict_clicked(sender: Any = 0, app_data: Any = None) -> None:
    if _active is not None:
        return
    _clear_prediction()
    if _selected_run is None:
        _problem(PREDICT, 'Choose a trained run first. Runs are listed from the Run page\'s output folder.')
        return
    problems = analyse.run_problems(_selected_run)
    if problems:
        _problem(PREDICT, '\n'.join(problems))
        return
    video = _video_for_action(PREDICT)
    if video is None:
        return
    _buffer.clear()
    dpg.set_value('analyse_log_box', '')
    try:
        out_dir = analyse.timestamped_folder(_selected_run.run_dir / 'desktop-inference')
        cfg = gc.infer_config_from_run(_selected_run, data=video, out=out_dir)
        handle = start_infer_run(cfg)
    except Exception as exc:
        _message(PREDICT, str(exc), _colors().error)
        _shell.set_status('Could not start the prediction', _colors().error)
        return
    _begin(PREDICT, handle, out_dir, video, run=_selected_run)


def on_track_clicked(sender: Any = 0, app_data: Any = None) -> None:
    if _active is not None:
        return
    _clear_tracking()
    if _showing_overlay:  # the overlay it was showing belongs to the results just cleared
        on_show_original()
    _set_overlay_buttons()
    prompts = analyse.parse_prompts(dpg.get_value('analyse_prompt_input'))
    if not prompts:
        _problem(TRACK, 'Type at least one thing to find, such as: person, car')
        return
    video = _video_for_action(TRACK)
    if video is None:
        return
    base = (_shell.output_folder() or '').strip()
    if not base:
        _problem(TRACK, 'Choose an output folder on the Run page first. Results are saved there.')
        return
    detector = analyse.detector_by_label(dpg.get_value('analyse_detector_combo')).key
    cfg = analyse.TrackConfig(video=video, prompts=prompts, out=analyse.new_output_folder(Path(base).expanduser()),
                              detector=detector, every_n_frames=int(dpg.get_value('analyse_every_n_input')),
                              track_score=round(float(dpg.get_value('analyse_score_input')), 4))
    issues = analyse.validate_track_config(cfg)
    if issues:
        _problem(TRACK, issues[0])
        return
    _buffer.clear()
    dpg.set_value('analyse_log_box', '')
    try:
        handle = start_track_run(cfg)
    except Exception as exc:
        _message(TRACK, str(exc), _colors().error)
        _shell.set_status('Could not start tracking', _colors().error)
        return
    _begin(TRACK, handle, cfg.out, video, detector=detector)


def on_stop_clicked(sender: Any = 0, app_data: Any = None) -> None:
    if _active is not None and _active.handle.cancel():
        _shell.set_status('Stopping...', _colors().warning)
        for name in (PREDICT, TRACK):
            dpg.configure_item(f'analyse_{name}_stop_button', enabled=False)


def on_detector_changed(sender: Any = 0, app_data: Any = None) -> None:
    """Each detector scores on its own scale, so its threshold starts at its own default."""
    key = analyse.detector_by_label(dpg.get_value('analyse_detector_combo')).key
    dpg.set_value('analyse_score_input', analyse.default_track_score(key))
    _refresh_detector_notice()


def _refresh_detector_notice() -> None:
    """Whether a run would download anything can change after any run, so ask the cache again."""
    key = analyse.detector_by_label(dpg.get_value('analyse_detector_combo')).key
    dpg.set_value('analyse_detector_note', analyse.detector_note(key))
    dpg.configure_item('analyse_track_button', label=analyse.track_button_label(key))


def on_open_output(sender: Any = 0, app_data: Any = None) -> None:
    try:
        if _last_out_dir is None or not _last_out_dir.is_dir():
            raise ValueError('Run an action first.')
        path = str(_last_out_dir.resolve())
        if hasattr(os, 'startfile'):
            os.startfile(path)
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', path])
    except (OSError, ValueError) as exc:
        _shell.set_status(str(exc), _colors().warning)


def _finish_prediction(active: _Running) -> None:
    global _prediction
    results = gc.read_inference_results(active.out_dir)
    view = analyse.summarise_prediction(results, active.run, active.video)
    view.warnings.extend(analyse.run_warnings(active.run))
    view.warnings.extend(analyse.prediction_log_warnings(_buffer.render()))
    if view.error:
        _message(PREDICT, view.error, _colors().error)
        _shell.set_status('Prediction failed - check the log', _colors().error)
        return
    _prediction = view
    _message(PREDICT, 'Complete.', _colors().success)
    _shell.set_status('Prediction complete', _colors().success)
    _render_prediction(view)


def _finish_tracking(active: _Running) -> None:
    global _tracking, _overlay
    results = analyse.read_track_results(active.out_dir, active.video, active.detector)
    if results.error:
        _message(TRACK, results.error, _colors().error)
        _shell.set_status('Tracking failed - check the log', _colors().error)
        return
    _tracking = results
    if results.total_tracks:
        _message(TRACK, 'Complete.', _colors().success)
        _shell.set_status('Tracking complete', _colors().success)
    else:
        _message(TRACK, 'Complete: no tracks found. See below.', _colors().analysis)
        _shell.set_status('Tracking complete: no tracks found', _colors().analysis)
    _render_tracking(results)
    _overlay = results.overlay
    if _overlay is not None:
        _play(_overlay, overlay=True)
    else:
        _set_overlay_buttons()


_FINISHERS: dict[str, Callable[[_Running], None]] = {PREDICT: _finish_prediction, TRACK: _finish_tracking}


def _poll_action() -> None:
    global _active
    if _active is None:
        return
    seconds = int(time.monotonic() - _active.started)
    dpg.set_value('analyse_elapsed_text', f'Elapsed  {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}')
    changed = False
    for _ in range(300):
        try:
            _buffer.feed(_active.handle.log_queue.get_nowait())
        except queue.Empty:
            break
        changed = True
    if changed:
        content = _buffer.render()
        dpg.set_value('analyse_log_box', content)
        if dpg.get_value('analyse_follow_log_checkbox'):
            dpg.set_y_scroll('analyse_log_window', dpg.get_y_scroll_max('analyse_log_window'))
        line = next((line.strip()[:200] for line in reversed(content.splitlines()) if line.strip()), '')
        if line:
            _message(_active.kind, line, _colors().analysis)
    try:
        status, message = _active.handle.done_queue.get_nowait()
    except queue.Empty:
        return
    while True:
        try:
            _buffer.feed(_active.handle.log_queue.get_nowait())
        except queue.Empty:
            break
    finished, _active = _active, None
    content = _buffer.render()
    dpg.set_value('analyse_log_box', content)
    _set_running(False)
    _refresh_detector_notice()
    if status == 'success':
        _FINISHERS[finished.kind](finished)
    elif status == 'cancelled':
        _message(finished.kind, 'Stopped. No result is shown; partial files may remain in the output folder.',
                 _colors().warning)
        _shell.set_status('Stopped', _colors().warning)
    else:
        summary = message.splitlines()[0] if message else 'Unknown failure'
        hint = analyse.failure_hint(content)
        summary = f'{hint}\n{summary}' if hint else summary
        tail = analyse.last_log_lines(content)
        _message(finished.kind, f'{summary}\nLast log lines:\n{tail}' if tail else summary, _colors().error)
        _shell.set_status('Failed - see the message on the page', _colors().error)


def poll() -> None:
    """Once per rendered frame."""
    _poll_action()
    _poll_player()


def on_page_changed(page: str) -> None:
    if page == 'analyse':
        on_refresh_runs()
        _refresh_detector_notice()
    elif _player is not None:
        _player.pause()


def shutdown() -> None:
    """Stop whatever is running. Called when the window closes."""
    global _active
    if _active is not None:
        _active.handle.close()
        _active = None
    if _player is not None:
        _player.close(timeout=2)
    _forget_state()


def _forget_state() -> None:
    global _choices, _selected_run, _prediction, _tracking, _last_out_dir
    global _player, _texture_size, _original, _overlay, _showing_overlay
    _choices = analyse.RunChoices()
    _selected_run = _prediction = _tracking = _last_out_dir = None
    _player = _texture_size = _original = _overlay = None
    _showing_overlay = False
    _buffer.clear()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _clear_prediction() -> None:
    global _prediction
    _prediction = None
    if dpg.does_item_exist('analyse_predict_results'):
        dpg.delete_item('analyse_predict_results', children_only=True)
    _message(PREDICT, '', _colors().muted)


def _clear_tracking() -> None:
    global _tracking, _overlay
    _tracking = None
    _overlay = None
    if dpg.does_item_exist('analyse_track_results'):
        dpg.delete_item('analyse_track_results', children_only=True)
    _message(TRACK, '', _colors().muted)


def _render_prediction(view: analyse.PredictionView) -> None:
    parent = 'analyse_predict_results'
    dpg.delete_item(parent, children_only=True)
    dpg.add_text(view.answer, parent=parent, color=_colors().primary, wrap=0)
    dpg.add_text(view.confidence, parent=parent, color=_colors().primary, wrap=0)
    dpg.add_text(view.expected_error, parent=parent, color=_colors().primary, wrap=0)
    dpg.add_text(f'Video: {view.video}', parent=parent, color=_colors().muted, wrap=0)
    dpg.add_text(f'Produced by your trained model: {view.model}', parent=parent, color=_colors().analysis, wrap=0)
    for warning in view.warnings:
        dpg.add_text(warning, parent=parent, color=_colors().warning, wrap=0)
    for note in view.notes:
        dpg.add_text(note, parent=parent, color=_colors().muted, wrap=0)


def _render_tracking(results: analyse.TrackResults) -> None:
    parent = 'analyse_track_results'
    dpg.delete_item(parent, children_only=True)
    if results.no_tracks_message:
        dpg.add_text(results.no_tracks_message, parent=parent, color=_colors().analysis, wrap=0)
    if results.labels:
        with ui.table(parent=parent, header_row=True, row_background=True, borders_innerH=True, resizable=True,
                      tag='analyse_counts_table'):
            for title in ('Prompt', 'Count', 'First seen', 'Last seen'):
                dpg.add_table_column(label=title)
            for label in results.labels:
                with ui.table_row():
                    for cell in label.row:
                        dpg.add_text(cell)
    dpg.add_text(f'Video: {results.video}', parent=parent, color=_colors().muted, wrap=0)
    frames = analyse.frames_note(results)
    if frames:
        dpg.add_text(frames, parent=parent, color=_colors().muted, wrap=0)
    if (results.every_n_frames or 1) > 1:
        dpg.add_text(analyse.FRAMES_STEP_NOTE, parent=parent, color=_colors().warning, wrap=0)
    if results.overlay is None:
        dpg.add_text('No overlay video was written for this run.', parent=parent, color=_colors().warning, wrap=0)
    dpg.add_text(f'Produced by the detector, not your trained model: {results.model}', parent=parent,
                 color=_colors().analysis, wrap=0)
    dpg.add_text(analyse.TRACKER_NOTE, parent=parent, color=_colors().muted, wrap=0)


# ---------------------------------------------------------------------------
# Building the page
# ---------------------------------------------------------------------------

def build_dialogs() -> None:
    with ui.file_dialog(directory_selector=False, show=False, callback=on_video_file_selected,
                        tag=VIDEO_DIALOG, width=800, height=480):
        for ext in ('.*', *sorted(VIDEO_EXTENSIONS)):
            dpg.add_file_extension(ext)


def build(shell: Shell) -> None:
    global _shell, _active
    _shell = shell
    _active = None
    _forget_state()
    colors = shell.palette
    with ui.group(tag='page_analyse', show=False):
        shell.heading('Analyse Video', 'Give one video to a model you trained, or describe what to find in it.')
        dpg.add_text(analyse.LIMIT_NOTE, color=colors.analysis, wrap=850)
        dpg.add_spacer(height=8)
        _build_video_card(shell)
        dpg.add_spacer(height=10)
        _build_model_card(shell)
        dpg.add_spacer(height=10)
        _build_track_card(shell)
        dpg.add_spacer(height=10)
        _build_activity_card(shell)
        dpg.add_spacer(height=10)
        _build_reserved_card(shell)
    dpg.bind_item_theme(PREVIEW, shell.themes['media'])
    dpg.bind_item_theme('analyse_play_button', shell.themes['run_button'])
    _set_overlay_buttons()


def _build_video_card(shell: Shell) -> None:
    colors = shell.palette
    with shell.card('VIDEO'):
        with ui.group(horizontal=True):
            dpg.add_input_text(tag=VIDEO_INPUT, hint='Paste a video path or browse...', width=-280, on_enter=True,
                               callback=on_load_video)
            dpg.add_button(label='Browse file', tag='analyse_browse_button', callback=shell.show_dialog,
                           user_data=VIDEO_DIALOG)
            dpg.add_button(label='Load / preview', tag='analyse_load_button', callback=on_load_video)
        dpg.add_text('Choose a video to begin.', tag='analyse_video_status', color=colors.muted, wrap=850)
        with ui.group(tag='analyse_player_controls', show=False):
            with ui.group(horizontal=True):
                dpg.add_button(label='Play', tag='analyse_play_button', callback=on_play_pause, enabled=False)
                dpg.add_button(label='Restart', tag='analyse_restart_button', callback=on_restart, enabled=False)
                dpg.add_button(label='Show my video', tag='analyse_show_original_button', callback=on_show_original)
                dpg.add_button(label='Show tracker overlay', tag='analyse_show_overlay_button', callback=on_show_overlay)
                dpg.add_text('', tag='analyse_showing_text', color=colors.muted)
            with ui.group(horizontal=True):
                dpg.add_slider_float(tag='analyse_seek_slider', width=-150, min_value=0, max_value=1, format='%.1f s',
                                     callback=on_seek, clamped=True, no_input=True, enabled=False)
                dpg.add_text('0:00 / 0:00', tag='analyse_time_text')
        with ui.child_window(tag=PREVIEW, height=160, horizontal_scrollbar=True):
            pass


def _build_model_card(shell: Shell) -> None:
    colors = shell.palette
    with shell.card('USE MY TRAINED MODEL'):
        with ui.group(horizontal=True):
            dpg.add_button(label='Refresh', tag='analyse_refresh_button', callback=on_refresh_runs)
            dpg.add_text('Video runs from the Run page\'s output folder.', color=colors.muted)
        dpg.add_combo(tag='analyse_run_combo', items=[], width=-1, callback=on_run_selected)
        dpg.add_text('No run chosen.', tag='analyse_run_info', color=colors.analysis, wrap=850)
        dpg.add_text('', tag='analyse_run_warning', color=colors.warning, wrap=850)
        dpg.add_text('', tag='analyse_run_note', color=colors.muted, wrap=850)
        with ui.group(horizontal=True):
            button = dpg.add_button(tag='analyse_predict_button', label='Predict on this video', height=44,
                                    callback=on_predict_clicked)
            dpg.bind_item_theme(button, shell.themes['run_button'])
            dpg.add_button(tag='analyse_predict_stop_button', label='Stop', height=44, enabled=False,
                           callback=on_stop_clicked)
        dpg.add_text('', tag='analyse_predict_message', color=colors.muted, wrap=850)
        with ui.group(tag='analyse_predict_results'):
            pass


def _build_track_card(shell: Shell) -> None:
    colors = shell.palette
    first = analyse.DETECTORS[0]
    with shell.card('FIND, TRACK AND COUNT'):
        dpg.add_text('Type what to find. Separate several things with commas.', color=colors.muted, wrap=850)
        dpg.add_input_text(tag='analyse_prompt_input', hint='person, car', width=-1)
        with ui.group(horizontal=True):
            dpg.add_combo(tag='analyse_detector_combo', items=[d.label for d in analyse.DETECTORS],
                          default_value=first.label, width=200, label='Detector', callback=on_detector_changed)
            dpg.add_input_int(tag='analyse_every_n_input', label='Frames step', default_value=1, width=130,
                              min_value=1, min_clamped=True)
            with ui.tooltip(dpg.last_item()):
                dpg.add_text(f'Detect on every n-th frame. 1 looks at every frame and is the slowest. '
                             f'{analyse.FRAMES_STEP_NOTE}', wrap=380)
            dpg.add_input_double(tag='analyse_score_input', label='Score threshold', width=150, step=0.05,
                                 format='%.2f', min_value=0.01, max_value=1.0, min_clamped=True, max_clamped=True,
                                 default_value=analyse.default_track_score(first.key))
            with ui.tooltip(dpg.last_item()):
                dpg.add_text('The score a box needs to start a track. Lower finds more, and finds more wrong things.',
                             wrap=380)
        dpg.add_text(analyse.FRAMES_STEP_NOTE, color=colors.muted, wrap=850)
        dpg.add_text(analyse.detector_note(first.key), tag='analyse_detector_note', color=colors.muted, wrap=850)
        with ui.group(horizontal=True):
            button = dpg.add_button(tag='analyse_track_button', label=analyse.track_button_label(first.key), height=44,
                                    callback=on_track_clicked)
            dpg.bind_item_theme(button, shell.themes['run_button'])
            dpg.add_button(tag='analyse_track_stop_button', label='Stop', height=44, enabled=False,
                           callback=on_stop_clicked)
        dpg.add_text('', tag='analyse_track_message', color=colors.muted, wrap=850)
        with ui.group(tag='analyse_track_results'):
            pass


def _build_activity_card(shell: Shell) -> None:
    colors = shell.palette
    with shell.card('LIVE OUTPUT'):
        with ui.group(horizontal=True):
            dpg.add_checkbox(tag='analyse_follow_log_checkbox', label='Follow output', default_value=True)
            dpg.add_button(label='Open output folder', tag='analyse_open_output_button', callback=on_open_output)
            dpg.add_text('Elapsed  00:00:00', tag='analyse_elapsed_text', color=colors.muted)
            dpg.add_text('ACTIVE', tag='analyse_active_indicator', show=False, color=colors.accent)
        with ui.child_window(tag='analyse_log_window', height=160, horizontal_scrollbar=True):
            dpg.add_text('', tag='analyse_log_box', wrap=0)


def _build_reserved_card(shell: Shell) -> None:
    with shell.card('MORE ACTIONS'):
        for action in analyse.RESERVED_ACTIONS:
            with ui.group(horizontal=True):
                # Disabled and without a callback: pressing it does nothing at all.
                dpg.add_button(label=action.label, tag=f'analyse_{action.key}_button', enabled=False, width=220)
                dpg.add_text(action.description, color=shell.palette.muted, wrap=560)

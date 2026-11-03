"""Dear PyGui widgets for desktop/cli_options specs (G1).

`render` builds one widget per spec, grouped into collapsible sections; `read` turns the widgets back into
{key: value} (only values that differ from the CLI default); `apply` sets the widgets from such a dict. Widget tags are
f'{prefix}_{opt.key}', so tests and pages can reach any control by its flag name.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import dearpygui.dearpygui as dpg

from desktop import cli_options as co
from desktop import dpg_context as ui

FILE_DIALOG, FOLDER_DIALOG = 'file_dialog_option_file', 'file_dialog_option_folder'
_browse: dict = {}  # the widget a Browse button is filling: {'tag', 'lines', 'callback'}


def tag(prefix: str, opt: co.Opt) -> str:
    return f'{prefix}_{opt.key}'


def row_tag(prefix: str, opt: co.Opt) -> str:
    return f'{tag(prefix, opt)}_row'


def section_tag(prefix: str, section: str) -> str:
    return f'{prefix}_section_' + ''.join(ch if ch.isalnum() else '_' for ch in section.lower())


def _tooltip(text: str) -> None:
    if text:
        with ui.tooltip(dpg.last_item()):
            dpg.add_text(text, wrap=380)


def _widget(prefix: str, opt: co.Opt, callback: Optional[Callable]) -> None:
    t = tag(prefix, opt)
    with ui.group(tag=row_tag(prefix, opt), horizontal=opt.kind in (co.MULTI,) or bool(opt.browse)):
        if opt.kind == co.BOOL:
            dpg.add_checkbox(tag=t, label=opt.label, default_value=bool(opt.default), callback=callback)
        elif opt.kind == co.INT:
            dpg.add_input_int(tag=t, label=opt.label, default_value=int(opt.default), width=150, callback=callback)
        elif opt.kind == co.FLOAT:
            dpg.add_input_double(tag=t, label=opt.label, default_value=float(opt.default), format='%.6g', width=150,
                                 step=0, callback=callback)
        elif opt.kind == co.CHOICE:
            items = [str(c) for c in opt.choices]
            if opt.default is None:
                items = [co.DEFAULT_LABEL, *items]
            dpg.add_combo(tag=t, label=opt.label, items=items, width=220, callback=callback,
                          default_value=co.DEFAULT_LABEL if opt.default is None else str(opt.default))
        elif opt.kind == co.MULTI:
            dpg.add_text(opt.label)
            _tooltip(opt.help)
            for choice in opt.choices:
                dpg.add_checkbox(tag=f'{t}__{choice}', label=str(choice), callback=callback,
                                 default_value=choice in (opt.default or ()))
            return
        elif opt.kind == co.LINES:
            dpg.add_input_text(tag=t, label=opt.label, multiline=True, height=64, width=-260 if opt.browse else -200,
                               default_value='\n'.join(map(str, opt.default or ())), callback=callback)
        else:
            default = ' '.join(map(str, opt.default or ())) if opt.kind == co.WORDS else ('' if opt.default is None else str(opt.default))
            hint = 'blank = default' if opt.default in (None, (), []) and not opt.required else ''
            dpg.add_input_text(tag=t, label=opt.label, default_value=default, hint=hint, callback=callback,
                               width=-260 if opt.browse else (420 if opt.kind in (co.TEXT, co.WORDS) else 150))
        _tooltip(opt.help)
        if opt.browse:
            dpg.add_button(label='Browse', tag=f'{t}_browse', callback=_on_browse,
                           user_data={'tag': t, 'lines': opt.kind == co.LINES, 'callback': callback, 'kind': opt.browse})


def render(specs: Iterable[co.Opt], prefix: str, callback: Optional[Callable] = None, open_sections: Iterable[str] = (),
           sections: Optional[Iterable[str]] = None, headers: bool = True) -> None:
    """Widgets for every spec that has one, under one collapsing header per section (in `sections` order)."""
    specs = [opt for opt in specs if not opt.hidden]
    order = list(sections) if sections is not None else list(dict.fromkeys(opt.section for opt in specs))
    open_sections = set(open_sections)
    for section in order:
        members = [opt for opt in specs if opt.section == section]
        if not members:
            continue
        if headers:
            with ui.collapsing_header(label=section, tag=section_tag(prefix, section), default_open=section in open_sections):
                for opt in members:
                    _widget(prefix, opt, callback)
        else:
            with ui.group(tag=section_tag(prefix, section)):
                for opt in members:
                    _widget(prefix, opt, callback)


def raw_value(prefix: str, opt: co.Opt) -> Any:
    t = tag(prefix, opt)
    if opt.kind == co.MULTI:
        return [c for c in opt.choices if dpg.get_value(f'{t}__{c}')]
    return dpg.get_value(t)


def read(specs: Iterable[co.Opt], prefix: str, values: Optional[dict] = None) -> tuple[dict, list[str]]:
    """{key: value} of every widget whose value differs from the CLI default, and the problems found. Hidden specs keep
    what `values` holds for them."""
    result, problems = {}, []
    for opt in specs:
        if opt.hidden:
            if values and opt.key in values:
                result[opt.key] = values[opt.key]
            continue
        value, problem = co.parse_text(opt, raw_value(prefix, opt))
        if problem:
            problems.append(problem)
        elif not co.is_default(opt, value):
            result[opt.key] = value
    return result, problems


def apply(specs: Iterable[co.Opt], prefix: str, values: dict) -> None:
    for opt in specs:
        if opt.hidden:
            continue
        value = (values or {}).get(opt.key, opt.empty)
        t = tag(prefix, opt)
        if opt.kind == co.MULTI:
            for choice in opt.choices:
                dpg.set_value(f'{t}__{choice}', choice in (value or ()))
        elif opt.kind == co.BOOL:
            dpg.set_value(t, bool(value))
        elif opt.kind == co.INT:
            dpg.set_value(t, int(value))
        elif opt.kind == co.FLOAT:
            dpg.set_value(t, float(value))
        elif opt.kind == co.CHOICE:
            dpg.set_value(t, co.DEFAULT_LABEL if value is None else str(value))
        elif opt.kind == co.LINES:
            dpg.set_value(t, '\n'.join(map(str, value or ())))
        elif opt.kind == co.WORDS:
            dpg.set_value(t, ' '.join(map(str, value or ())))
        else:
            dpg.set_value(t, '' if value is None else co.format_value(value))


def show_rows(specs: Iterable[co.Opt], prefix: str, visible: Callable[[co.Opt], bool]) -> None:
    """Show the widgets `visible` accepts, hide the others, and hide a section with nothing to show."""
    shown_sections: dict = {}
    for opt in specs:
        if opt.hidden:
            continue
        shown = bool(visible(opt))
        dpg.configure_item(row_tag(prefix, opt), show=shown)
        shown_sections[opt.section] = shown_sections.get(opt.section, False) or shown
    for section, shown in shown_sections.items():
        if dpg.does_item_exist(section_tag(prefix, section)):
            dpg.configure_item(section_tag(prefix, section), show=shown)


# --- Browse buttons ---------------------------------------------------------------------------------------------------
def build_dialogs() -> None:
    with ui.file_dialog(directory_selector=False, show=False, callback=on_browse_selected, tag=FILE_DIALOG, width=800, height=480):
        for ext in ('.*', '.parquet', '.csv', '.xlsx', '.json', '.yaml', '.yml', '.pt', '.pth'):
            dpg.add_file_extension(ext)
    with ui.file_dialog(directory_selector=True, show=False, callback=on_browse_selected, tag=FOLDER_DIALOG, width=800, height=480):
        pass


def _on_browse(sender: Any = 0, app_data: Any = None, user_data: Optional[dict] = None) -> None:
    _browse.clear()
    _browse.update(user_data or {})
    dpg.show_item(FOLDER_DIALOG if _browse.get('kind') == 'folder' else FILE_DIALOG)


def on_browse_selected(sender: Any = 0, app_data: Any = None) -> None:
    if not isinstance(app_data, dict) or not _browse.get('tag'):
        return
    selections = app_data.get('selections') or {}
    path = next(iter(selections.values()), None) or app_data.get('file_path_name')
    if not path:
        return
    target = _browse['tag']
    if _browse.get('lines'):
        current = [line for line in str(dpg.get_value(target) or '').splitlines() if line.strip()]
        dpg.set_value(target, '\n'.join([*current, str(path)]))
    else:
        dpg.set_value(target, str(path))
    if _browse.get('callback') is not None:
        _browse['callback']()

from pathlib import Path


def test_launch_cli_forwards_video_without_running_training(monkeypatch):
    from desktop import app
    calls = []
    monkeypatch.setattr(app, 'run_gui', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(app.sys, 'argv', ['app.py', '--page', 'analyse', '--video', 'cat.mov'])
    assert app.main() == 0
    assert calls == [{'initial_page': 'analyse', 'initial_video': Path('cat.mov')}]


def test_installed_cat_is_offered_not_selected_in_page07(tmp_path, monkeypatch):
    """User decision 2026-11-02 (replaces test_installed_cat_is_selected_in_page07): page 07 always starts on the
    general detector, so its behaviour does not depend on what is installed; an installed cat detector gets a hint,
    and choosing it fills in the prompt and the checkpoint path."""
    from desktop import app, analyse, analyse_page
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'not loaded in GUI')
    monkeypatch.setattr(analyse, 'installed_cat_checkpoint', lambda: checkpoint)
    app.dpg.create_context()
    try:
        app.build_ui()
        assert app.dpg.get_value('analyse_detector_combo') == analyse.DETECTORS[0].label
        assert app.dpg.get_value('analyse_prompt_input') == ''
        assert not app.dpg.get_item_configuration('analyse_checkpoint_input')['show']
        assert app.dpg.get_item_configuration('analyse_cat_hint')['show']
        assert str(checkpoint) in app.dpg.get_value('analyse_cat_hint')

        app.dpg.set_value('analyse_detector_combo', 'Trained cat detector')
        analyse_page.on_detector_changed()
        assert app.dpg.get_value('analyse_prompt_input') == 'cat'
        assert app.dpg.get_value('analyse_checkpoint_input') == str(checkpoint)
        assert app.dpg.get_item_configuration('analyse_checkpoint_input')['show']
        assert not app.dpg.get_item_configuration('analyse_cat_hint')['show']
    finally:
        analyse_page.shutdown()
        app.dpg.destroy_context()


def test_no_cat_hint_without_an_installed_detector(monkeypatch):
    from desktop import app, analyse, analyse_page
    monkeypatch.setattr(analyse, 'installed_cat_checkpoint', lambda: None)
    app.dpg.create_context()
    try:
        app.build_ui()
        assert app.dpg.get_value('analyse_detector_combo') == analyse.DETECTORS[0].label
        assert not app.dpg.get_item_configuration('analyse_cat_hint')['show']
    finally:
        analyse_page.shutdown()
        app.dpg.destroy_context()

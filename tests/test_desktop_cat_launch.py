from pathlib import Path


def test_launch_cli_forwards_video_without_running_training(monkeypatch):
    from desktop import app
    calls = []
    monkeypatch.setattr(app, 'run_gui', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(app.sys, 'argv', ['app.py', '--page', 'analyse', '--video', 'cat.mov'])
    assert app.main() == 0
    assert calls == [{'initial_page': 'analyse', 'initial_video': Path('cat.mov')}]


def test_installed_cat_is_selected_in_page07(tmp_path, monkeypatch):
    from desktop import app, analyse, analyse_page
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'not loaded in GUI')
    monkeypatch.setattr(analyse, 'installed_cat_checkpoint', lambda: checkpoint)
    app.dpg.create_context()
    try:
        app.build_ui()
        assert app.dpg.get_value('analyse_detector_combo') == 'Trained cat detector'
        assert app.dpg.get_value('analyse_prompt_input') == 'cat'
        assert app.dpg.get_value('analyse_checkpoint_input') == str(checkpoint)
        assert app.dpg.get_item_configuration('analyse_checkpoint_input')['show']
    finally:
        analyse_page.shutdown()
        app.dpg.destroy_context()

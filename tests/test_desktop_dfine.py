"""Page 07's D-FINE choice: note, prompt check, argv and the better-detector tip (no torch import)."""
import json
from pathlib import Path

from desktop import analyse


def test_dfine_is_offered_after_the_open_vocabulary_detectors_and_default_is_unchanged():
    keys = [d.key for d in analyse.DETECTORS]
    assert keys[0] == 'grounding_dino' and 'dfine' in keys
    assert analyse.detector_by_key('dfine').model_id == 'ustc-community/dfine-large-obj2coco-e25'


def test_dfine_note_says_what_it_can_find(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path))
    note = analyse.detector_note('dfine')
    assert '80 COCO' in note and 'Apache-2.0' in note and '0.13 GB' in note
    assert 'both open-vocabulary detectors together are about 1.3 GB' in analyse.detector_note('owlv2')


def test_dfine_prompts_are_checked_before_anything_runs(tmp_path):
    clip = tmp_path / 'v.mp4'
    clip.write_bytes(b'x')
    cfg = analyse.TrackConfig(video=clip, prompts=['lobster'], out=tmp_path / 'o', detector='dfine')
    issues = analyse.validate_track_config(cfg)
    assert any('COCO' in i and 'Choose Grounding DINO or OWLv2' in i for i in issues)
    cfg.prompts = ['people', 'cars']
    assert not any('COCO' in i for i in analyse.validate_track_config(cfg))
    argv = analyse.build_track_argv(cfg)
    assert argv[argv.index('--detector') + 1] == 'dfine'


def test_better_detector_tip_only_for_everyday_objects_on_open_vocabulary_detectors():
    assert 'D-FINE' in analyse.better_detector_hint('grounding_dino', ['cat'])
    assert 'D-FINE' in analyse.better_detector_hint('owlv2', ['person', 'car'])
    assert analyse.better_detector_hint('grounding_dino', ['lobster']) == ''
    assert analyse.better_detector_hint('dfine', ['cat']) == ''
    assert analyse.better_detector_hint('pet_fasterrcnn', ['cat']) == ''
    assert analyse.better_detector_hint('grounding_dino', []) == ''


def test_results_reader_reports_the_detector_used(tmp_path):
    out = tmp_path / 'run'
    (out / 'v').mkdir(parents=True)
    (out / 'track_arguments.json').write_text(json.dumps({'detector': 'dfine', 'detector_model': 'm'}))
    (out / 'v' / 'summary.json').write_text(json.dumps({
        'video': 'v.mp4', 'num_frames': 3, 'every_n_frames': 1, 'total_tracks': 1,
        'labels': {'cat': {'count': 1, 'first_seen_s': 0.0, 'last_seen_s': 0.1, 'tracks': []}}}))
    results = analyse.read_track_results(out, Path('v.mp4'), 'dfine')
    assert results.detector == 'dfine'


def test_analyse_module_still_does_not_import_torch():
    import ast
    tree = ast.parse(Path(analyse.__file__).read_text(encoding='utf-8'))
    names = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
             for a in getattr(n, 'names', [])}
    mods = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert 'torch' not in names | mods and 'transformers' not in names | mods

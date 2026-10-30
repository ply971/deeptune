"""deeptune.py -> train -> eval -> embed with the training-basics flags, on tiny
clips: a run stopped after one epoch and resumed through the CLI must end with
the same weights as one run of two epochs. CPU, dummy_video3d, 32x32 frames."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchvision

pytest.importorskip('av')

from datasets.video_synth import translating_clip
from evaluators.error_model import weights_fingerprint

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
BASE = ['--modality', 'video', '--model_version', 'dummy_video3d', '--batch_size', '4', '--num_frames', '2',
        '--added_layers', '1', '--embed_size', '16', '--raw-data', '--no_error_model']


@pytest.fixture()
def small_pipeline(monkeypatch):
    import desktop.config as config
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import inference.video as infer_mod
    import trainers.video.train as train_mod
    for module in (train_mod, eval_mod, embed_mod, infer_mod):
        monkeypatch.setattr(module, 'transformations', SMALL)
    versions = dict(config.MODEL_VERSIONS_BY_MODALITY)
    versions['video'] = [*versions['video'], 'dummy_video3d']
    monkeypatch.setattr(config, 'MODEL_VERSIONS_BY_MODALITY', versions)


@pytest.fixture()
def manifest(tmp_path):
    rows = []
    (tmp_path / 'clips').mkdir()
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(6):
            path = translating_clip(tmp_path / 'clips' / f'{label}_{i}.mp4', 10, dx, 0, size=32, seed=i)
            rows.append({'videos': str(path), 'labels': label, 'group': f'{label}-src{i // 2}'})
    path = tmp_path / 'clips.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(out: Path, manifest: Path, *extra):
    import deeptune
    deeptune.main([*BASE, '--df', str(manifest), '--out', str(out), *extra])
    return next(out.glob('deeptune-*/trainval_output_*'))


def test_cli_resume_matches_the_uninterrupted_run_and_config_flags_arrive(tmp_path, manifest, small_pipeline):
    config = tmp_path / 'run.yaml'
    pytest.importorskip('yaml')
    config.write_text('checkpoint_every: 1\nselect_best: true\nseed: 7\ngrad_accum_steps: 2\nlabel_smoothing: 0.1\n',
                      encoding='utf-8')

    whole = _run(tmp_path / 'whole', manifest, '--config', str(config), '--num_epochs', '2')
    first = _run(tmp_path / 'first', manifest, '--config', str(config), '--num_epochs', '1')
    second = _run(tmp_path / 'second', manifest, '--config', str(config), '--num_epochs', '2',
                  '--resume', str(first / 'last_checkpoint.pt'))

    final_whole = torch.load(whole / 'model_weights.pth', weights_only=True)
    final_second = torch.load(second / 'model_weights.pth', weights_only=True)
    assert weights_fingerprint(final_whole) == weights_fingerprint(final_second)

    log = pd.read_csv(second / 'training_log.csv')
    assert log['epoch'].tolist() == [1, 2]
    pd.testing.assert_frame_equal(pd.read_csv(whole / 'training_log.csv'), log)

    saved = json.loads((second / 'cli_arguments.json').read_text(encoding='utf-8'))
    assert saved['seed'] == 7 and saved['select_best'] is True and saved['grad_accum_steps'] == 2
    assert saved['resume'] == str(first / 'last_checkpoint.pt') and saved['config'] == str(config)

    record = json.loads((second / 'training_run.json').read_text(encoding='utf-8'))
    assert record['seed_requested'] == 7 and record['seed_effective'] == 7
    assert record['selection']['validation_used_for_model_selection'] is True
    assert record['data_identity']['train_file'] and record['data_identity']['val_file']
    assert (second / 'best_model_weights.pth').is_file()


def test_cli_refuses_a_bad_combination(tmp_path, manifest, small_pipeline, capsys):
    import deeptune
    with pytest.raises(SystemExit):
        deeptune.main([*BASE, '--df', str(manifest), '--out', str(tmp_path / 'o'), '--mode', 'reg', '--label_smoothing', '0.1'])
    assert 'label_smoothing' in capsys.readouterr().err


def test_test_crops_reach_evaluation_and_the_saved_arguments(tmp_path, manifest, small_pipeline):
    run = _run(tmp_path / 'o', manifest, '--num_epochs', '1', '--fixed-seed', '--test_crops', '3')
    saved = json.loads((run / 'cli_arguments.json').read_text(encoding='utf-8'))
    assert saved['test_crops'] == 3
    assert pd.read_csv(next(run.parent.glob('eval_output_*/predictions.csv'))).shape[0] > 0


def test_test_crops_are_in_the_error_model_fit(tmp_path, manifest, small_pipeline):
    import deeptune
    argv = [a for a in BASE if a != '--no_error_model']
    deeptune.main([*argv, '--df', str(manifest), '--out', str(tmp_path / 'o'), '--num_epochs', '1', '--fixed-seed', '--test_crops', '3'])
    fit = json.loads(next((tmp_path / 'o').glob('deeptune-*/trainval_output_*/error_model_fit.json')).read_text(encoding='utf-8'))
    assert fit['test_crops'] == 3 and fit['test_clips'] == 1

    # The Test page path picks the recorded setting up, and the error model still fits.
    from desktop.config import build_infer_argv, infer_config_from_run, read_run_results
    import infer
    run = next((tmp_path / 'o').glob('deeptune-*'))
    frame = pd.read_parquet(next(run.glob('data_splits_*/test_split.parquet')))
    data = tmp_path / 'new'
    for row in frame.itertuples():
        folder = data / ('left' if 'left' in Path(row.videos).name else 'right')
        folder.mkdir(parents=True, exist_ok=True)
        (folder / Path(row.videos).name).write_bytes(Path(row.videos).read_bytes())
    cfg = infer_config_from_run(read_run_results(run), data, tmp_path / 'infer')
    argv = build_infer_argv(cfg)
    assert argv[argv.index('--test_crops') + 1] == '3'
    assert infer.main(argv) == 0
    inferred = pd.read_csv(tmp_path / 'infer' / 'predictions.csv')
    assert len(inferred) == len(frame) and 'expected_error' in inferred.columns


def test_training_args_module_stays_torch_free():
    import subprocess
    import sys
    code = "import sys; import handlers.training_args; sys.exit(1 if 'torch' in sys.modules else 0)"
    assert subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1]).returncode == 0

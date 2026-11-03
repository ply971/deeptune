"""select_model.py end to end: real deeptune.py subprocesses on tiny synthetic clips. CPU, 2 candidates, k=2, 1 epoch.

Two untrained tiny models (resnet_tiny, a frame model, and vjepa2_tiny, a clip model with a probe), 4 frames of 32x32
clips. Checks the files, and that the sealed test rows reached no cross-validation run.
"""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('av')

from datasets.video_synth import translating_clip


def _dataset(root: Path) -> Path:
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(10):
            path = translating_clip(root / f'{label}_{i}.mp4', 6, dx, 0, size=32, seed=i)
            rows.append({'videos': str(path.resolve()), 'labels': label, 'group': f'{label}-src{i // 2}'})
    path = root / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.mark.slow
def test_select_model_end_to_end_through_deeptune_subprocesses(tmp_path):
    import select_model
    from handlers.model_selection.runner import SubprocessRunner
    data, out = _dataset(tmp_path), tmp_path / 'search'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    code = select_model.main(['--search', 'exhaustive', '--df', str(data), '--out', str(out), '--backbones', 'resnet_tiny', 'vjepa2_tiny', '--num_frames', '4',
                              '--k', '2', '--epochs', '1', '--batch_size', '4', '--boot', '50', '--no_pilot', '--gpu_memory_gb', 'none',
                              '--seed', '7'],
                             runner=SubprocessRunner(env=env))
    assert code == 0, (out / 'ledger.jsonl').read_text() if (out / 'ledger.jsonl').exists() else 'no ledger'
    ledger = [json.loads(line) for line in (out / 'ledger.jsonl').read_text().splitlines()]
    assert sorted(entry['config']['backbone'] for entry in ledger) == ['resnet_tiny', 'vjepa2_tiny']
    assert {entry['status'] for entry in ledger} == {'done'} and all(len(entry['folds']) == 2 for entry in ledger)
    board = pd.read_csv(out / 'leaderboard.csv')
    assert list(board['rank']) == [1, 2] and board['cv_accuracy_mean'].between(0, 1).all()
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['winner']['key'] == board.loc[0, 'key'] and selection['paired_difference']['unit'] == 'group'
    assert selection['test']['n'] == selection['split']['sealed_test']['rows'] > 0
    assert 'accuracy' in selection['test']['intervals']

    sealed = pd.read_parquet(out / 'data' / 'sealed_test.parquet')
    for entry in ledger:  # every cross-validated row is a development row
        run = out / entry['run_dir']
        assert '--cv_only' in (run / 'deeptune.log').read_text(encoding='utf-8').splitlines()[1]
        assert not list(run.glob('*/data_splits_*/test_split.parquet')) and not list(run.glob('*/eval_output_*'))
        oof = pd.read_csv(next(run.glob('*/validation_kfold/oof_predictions.csv')))
        assert not set(oof['video_id']) & set(sealed['videos'].map(lambda p: Path(p).name))
        assert len(oof) == selection['split']['dev']['rows']
    final = out / selection['final']['run_dir']
    split = json.loads(next(final.glob('*/data_splits_*/split_summary.json')).read_text())
    assert split['test_source'] == '--test_df' and split['sample_counts']['test'] == len(sealed)
    assert split['validation_seed'] == 7 and selection['seeds']['refit_validation'] == 7  # --seed reaches the refit's validation split
    assert sorted(pd.read_parquet(next(final.glob('*/data_splits_*/test_split.parquet')))['videos']) == sorted(sealed['videos'])


@pytest.mark.slow
def test_smart_search_end_to_end_with_a_real_feature_probe_screen(tmp_path):
    """--search smart through real subprocesses: the family screen by cached features + attentive probe (extract_features.py,
    train_feature_head.py), tuning and the final step through deeptune.py. Probe and deeptune.py runs share the folds."""
    import select_model
    from handlers.model_selection.runner import SubprocessRunner
    data, out = _dataset(tmp_path), tmp_path / 'search'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    code = select_model.main(['--df', str(data), '--out', str(out), '--backbones', 'siglip_tiny', 'vjepa2_tiny', '--train_modes', 'frozen', 'full',
                              '--num_frames', '4', '--augmentations', 'off', 'aug', '--top_families', '1', '--recheck_models', '0',
                              '--screen_views', 'frozen',  # the frozen view only: the probe path, at R3b's first cost
                              '--k', '2', '--epochs', '1', '--batch_size', '4', '--boot', '50', '--no_pilot', '--gpu_memory_gb', 'none',
                              '--profile_clips', '4', '--seed', '7'],
                             runner=SubprocessRunner(env=env))
    ledger = [json.loads(line) for line in (out / 'ledger.jsonl').read_text().splitlines()]
    assert code == 0, (out / 'ledger.jsonl').read_text()
    probes = [entry for entry in ledger if entry.get('protocol') == 'probe']
    cv = [entry for entry in ledger if entry.get('protocol') == 'cv']
    assert sorted(entry['config']['backbone'] for entry in probes) == ['siglip_tiny', 'vjepa2_tiny']
    assert {entry['status'] for entry in probes + cv} == {'done'}
    assert len({entry['fold_signature'] for entry in probes + cv}) == 1  # the probe ran deeptune.py's own folds
    assert len({entry['config']['backbone'] for entry in cv}) == 1  # --top_families 1: one family tuned
    for entry in cv:
        command = (out / entry['run_dir'] / 'deeptune.log').read_text(encoding='utf-8').splitlines()[1]
        frozen = entry['config']['train_mode'] == 'frozen'
        assert '--cv_only' in command and ('--video_augment' in command) == (entry['config'].get('augmentation') == 'aug')
        assert not (frozen and '--video_augment' in command)
    stages = json.loads((out / 'stages.json').read_text())
    assert stages[0]['stage'] == 'screen protocol (frozen view)' and 'probe' in stages[0]['decision']
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['strategy'] == 'smart' and selection['test']['n'] == selection['split']['sealed_test']['rows'] > 0
    assert (out / 'profile.json').is_file() and json.loads((out / 'profile.json').read_text())['facts']['clip_probe']['opened'] == 4
    sealed = pd.read_parquet(out / 'data' / 'sealed_test.parquet')
    sealed_ids = set(sealed['videos'].map(lambda p: Path(p).name))
    for entry in probes + cv:  # every cross-validated row, probe or deeptune.py, is a development row
        oof = pd.read_csv(out / entry['oof_path'])
        assert not set(oof['video_id']) & sealed_ids and len(oof) == selection['split']['dev']['rows']
    for log in (out / 'runs').rglob('probe.log'):
        assert 'sealed_test' not in log.read_text(encoding='utf-8')
    final = out / selection['final']['run_dir']
    assert sorted(pd.read_parquet(next(final.glob('*/data_splits_*/test_split.parquet')))['videos']) == sorted(sealed['videos'])

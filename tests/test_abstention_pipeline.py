"""--target_error through deeptune.py and infer.py on tiny clips. CPU, dummy_video3d, 32x32, one epoch."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torchvision

pytest.importorskip('av')

from datasets.video_synth import translating_clip
from evaluators.abstention import NOT_SURE, AbstentionRule

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
GROUPS_PER_CLASS = 8


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


def _manifest(root: Path, groups=True) -> Path:
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(GROUPS_PER_CLASS * 2):
            path = translating_clip(root / f'{label}_{i}.mp4', 6, dx, 0, size=32, seed=i)
            row = {'videos': str(path), 'labels': label}
            if groups:
                row['group'] = f'{label}-src{i // 2}'
            rows.append(row)
    path = root / 'clips.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(root, *extra, groups=True):
    import deeptune
    out = root / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'dummy_video3d', '--num_epochs', '1', '--batch_size', '4', '--num_frames', '2', '--added_layers', '1',
                   '--embed_size', '16', '--fixed-seed', '--num_workers', '0', '--df', str(_manifest(root, groups)), '--raw-data', '--report_boot', '20', *extra])
    return next(out.glob('deeptune-*'))


def _split(run, name):
    return pd.read_parquet(next(run.glob(f'data_splits_*/{name}_split.parquet')))


def _infer(run, out, *extra):
    import infer
    recorded = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    weights = next(run.glob('trainval_output_*/model_weights.pth'))
    test = _split(run, 'test')[['videos', 'labels']]
    data = out.parent / f'{out.name}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(data, index=False)
    code = infer.main(['--modality', 'video', '--checkpoint', str(weights), '--data', str(data), '--out', str(out), '--model_version', 'dummy_video3d', '--num_classes', str(recorded['num_classes']),
                       '--added_layers', '1', '--embed_size', '16', '--batch_size', '4', '--num_workers', '0', *extra])
    assert code == 0
    return pd.read_csv(out / 'predictions.csv')


def test_off_by_default_changes_nothing(tmp_path, small_pipeline):
    run = _run(tmp_path)
    assert not list(run.glob('data_splits_*/calibration_split.parquet')) and not list(run.glob('trainval_output_*/abstention.json')) and not list(run.glob('eval_output_*/abstention_report.json'))
    recorded = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    assert recorded['target_error'] is None and recorded['calib_fraction'] == 0.1
    predictions = _infer(run, tmp_path / 'infer')
    assert 'decision' not in predictions.columns and not (tmp_path / 'infer' / 'abstention_summary.json').exists()


def test_full_run_splits_fits_saves_reports_and_says_not_sure(tmp_path, small_pipeline):
    run = _run(tmp_path, '--target_error', '0.45', '--abstain_delta', '0.5', '--test_clips', '2')
    parts = {name: _split(run, name) for name in ('train', 'val', 'test', 'calibration')}
    names = list(parts)
    for i, a in enumerate(names):  # group-disjoint everywhere, and no clip lost
        for b in names[i + 1:]:
            assert not set(parts[a]['group']) & set(parts[b]['group'])
    assert sum(map(len, parts.values())) == 4 * GROUPS_PER_CLASS
    assert len(parts['calibration']) >= 2

    recorded = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    assert recorded['target_error'] == 0.45 and recorded['abstain_delta'] == 0.5 and recorded['abstain_guarantee'] == 'selective_risk'

    rule = AbstentionRule.load(next(run.glob('trainval_output_*/abstention.json')))
    info = rule.info
    assert info['split'] == 'calibration' and info['n_clips'] == len(parts['calibration']) and info['unit'] == 'group' and info['n_units'] == parts['calibration']['group'].nunique()
    assert info['pipeline']['test_clips'] == 2 and info['temperature_applied'] is False and info['weights_sha256'] and info['void_reasons'] == []
    assert info['status'] in ('met', 'target_not_met') and info['headline']
    assert info['selection'] is not None and 'validation_used_for_model_selection' in info['selection']
    assert next(run.glob('trainval_output_*/abstention_fit_table.csv')).is_file()

    eval_dir = next(run.glob('eval_output_*'))
    report = json.loads((eval_dir / 'abstention_report.json').read_text())
    assert report['split'] == 'test' and report['n_clips'] == len(parts['test']) and report['unit'] == 'group' and report['target_error'] == 0.45
    assert (eval_dir / 'selective_risk_curve.csv').is_file()
    decisions = pd.read_csv(eval_dir / 'predictions_abstention.csv', keep_default_na=False)
    assert len(decisions) == len(parts['test']) and set(decisions['decision']) <= {NOT_SURE, '0', '1'}

    # Inference reloads the rule and takes --test_clips 2 from the run, so nothing mismatches.
    predictions = _infer(run, tmp_path / 'infer', )
    summary = json.loads((tmp_path / 'infer' / 'abstention_summary.json').read_text())
    assert summary['pipeline_mismatches'] == [] and summary['n_clips'] == len(parts['test']) and summary['guarantee_holds'] == info['guarantee_holds']
    answered = rule.answers(predictions['confidence'].to_numpy())
    assert predictions['answered'].tolist() == answered.tolist()
    assert (predictions['decision'] == NOT_SURE).tolist() == (~answered).tolist()
    assert predictions.loc[answered, 'decision'].tolist() == predictions.loc[answered, 'predicted_label'].astype(str).tolist()
    metrics = json.loads((tmp_path / 'infer' / 'inference_metrics.json').read_text())
    assert metrics['abstention']['not_sure'] == int((~answered).sum())

    # Other settings: warned, and the guarantee is no longer claimed.
    _infer(run, tmp_path / 'other', '--test_clips', '1')
    other = json.loads((tmp_path / 'other' / 'abstention_summary.json').read_text())
    assert other['guarantee_holds'] is False and any('test_clips' in issue for issue in other['pipeline_mismatches']) and other['headline'].startswith('No guarantee')

    # Opting out gives the plain predictions.
    plain = _infer(run, tmp_path / 'plain', '--no_abstain')
    assert 'decision' not in plain.columns and not (tmp_path / 'plain' / 'abstention_summary.json').exists()


def test_target_out_of_reach_ends_with_every_clip_not_sure(tmp_path, small_pipeline):
    run = _run(tmp_path, '--target_error', '0.001', '--report_boot', '0')
    info = json.loads(next(run.glob('trainval_output_*/abstention.json')).read_text())
    assert info['status'] == 'target_not_met' and info['threshold'] is None and not info['guarantee_holds'] and 'cannot be met' in info['headline']
    predictions = _infer(run, tmp_path / 'infer')
    assert (predictions['decision'] == NOT_SURE).all()


def test_expected_error_can_rank_clips(tmp_path, small_pipeline):
    run = _run(tmp_path, '--target_error', '0.45', '--abstain_delta', '0.5', '--abstain_score', 'expected_error', '--report_boot', '0')
    rule = AbstentionRule.load(next(run.glob('trainval_output_*/abstention.json')))
    assert rule.score_name == 'expected_error'
    predictions = _infer(run, tmp_path / 'infer')
    assert predictions['answered'].tolist() == rule.answers(1 - predictions['expected_error'].to_numpy()).tolist()


def test_rule_for_other_weights_is_refused_at_inference(tmp_path, small_pipeline, capsys):
    run = _run(tmp_path, '--target_error', '0.45', '--report_boot', '0')
    path = next(run.glob('trainval_output_*/abstention.json'))
    data = json.loads(path.read_text())
    data['weights_sha256'] = 'something else'
    path.write_text(json.dumps(data))
    predictions = _infer(run, tmp_path / 'infer')
    assert 'decision' not in predictions.columns and 'different weights' in capsys.readouterr().out


def test_no_group_column_says_clip_level_in_every_output(tmp_path, small_pipeline, capsys):
    run = _run(tmp_path, '--target_error', '0.45', '--abstain_delta', '0.5', '--report_boot', '0', groups=False)
    output = capsys.readouterr().out
    assert 'clip level and assumes independent clips' in output
    info = json.loads(next(run.glob('trainval_output_*/abstention.json')).read_text())
    assert info['unit'] == 'row' and info['guarantee_level'] == 'clip' and info['assumes_independent_clips'] is True
    report = json.loads(next(run.glob('eval_output_*/abstention_report.json')).read_text())
    assert report['assumes_independent_clips'] is True
    _infer(run, tmp_path / 'infer')
    summary = json.loads((tmp_path / 'infer' / 'abstention_summary.json').read_text())
    assert summary['assumes_independent_clips'] is True and summary['guarantee_level'] == 'clip'
    if info['status'] == 'met':
        assert 'independent clips' in summary['headline']


def test_standalone_fit_voids_the_guarantee_when_groups_overlap(tmp_path, small_pipeline, capsys):
    from evaluators.video import abstention_fit
    run = _run(tmp_path, '--report_boot', '0')
    weights = next(run.glob('trainval_output_*/model_weights.pth'))
    val = next(run.glob('data_splits_*/val_split.parquet'))
    train = next(run.glob('data_splits_*/train_split.parquet'))
    assert abstention_fit.main(['--checkpoint', str(weights), '--calib_df', str(train), '--target_error', '0.45', '--abstain_delta', '0.5', '--split_name', 'held_out', '--grouper', 'group',
                                '--other_df', str(train), str(val)]) == 0
    info = json.loads((weights.parent / 'abstention.json').read_text())
    assert info['guarantee_holds'] is False and info['void_reasons'] and info['headline'].startswith(('No guarantee', 'Target error'))
    with pytest.raises(ValueError, match='Refusing'):
        abstention_fit.main(['--checkpoint', str(weights), '--calib_df', str(val), '--target_error', '0.45', '--split_name', 'validation'])


@pytest.mark.parametrize('extra, needle', [(['--modality', 'images'], 'video modality'), (['--mode', 'reg'], 'classification')])
def test_unsupported_runs_are_refused(tmp_path, small_pipeline, extra, needle, capsys):
    import deeptune
    base = {'--modality': 'video', '--mode': 'cls'}
    for i in range(0, len(extra), 2):
        base[extra[i]] = extra[i + 1]
    arguments = [item for pair in base.items() for item in pair]
    with pytest.raises(SystemExit):
        deeptune.main([*arguments, '--out', str(tmp_path / 'out'), '--model_version', 'dummy_video3d', '--df', str(_manifest(tmp_path)), '--raw-data', '--target_error', '0.1'])
    assert needle in capsys.readouterr().err


def test_flags_have_help_and_are_off_by_default():
    from cli import DeepTuneVisionOptions
    from utils import RunType
    args = DeepTuneVisionOptions(RunType.ONECALL, args=['--modality', 'video', '--df', 'x', '--out', 'y'])
    assert args.target_error is None and args.abstain_guarantee == 'selective_risk' and args.calib_fraction == 0.1
    text = args.parser.format_help()
    for flag in ('--target_error', '--abstain_delta', '--abstain_guarantee', '--abstain_score', '--calib_fraction'):
        assert flag in text
    assert 'promises no accuracy' in text
    with pytest.raises(SystemExit):
        DeepTuneVisionOptions(RunType.ONECALL, args=['--modality', 'video', '--df', 'x', '--out', 'y', '--target_error', '1.5'])

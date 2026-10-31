"""deeptune.py with --validation, the final report and --export_dfanalyze, on tiny clips. CPU, dummy_video3d, 32x32, one epoch."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torchvision

pytest.importorskip('av')

from datasets.video_synth import translating_clip

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])


@pytest.fixture()
def small_pipeline(monkeypatch):
    import desktop.config as config
    import embed.video.dfanalyze as export_mod
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import trainers.video.train as train_mod
    for module in (train_mod, eval_mod, embed_mod, export_mod):
        monkeypatch.setattr(module, 'transformations', SMALL)
    versions = dict(config.MODEL_VERSIONS_BY_MODALITY)
    versions['video'] = [*versions['video'], 'dummy_video3d']
    monkeypatch.setattr(config, 'MODEL_VERSIONS_BY_MODALITY', versions)


def _manifest(root: Path, per_class=10, groups=True) -> Path:
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(per_class):
            path = translating_clip(root / f'{label}_{i}.mp4', 6, dx, 0, size=32, seed=i)
            row = {'videos': str(path), 'labels': label}
            if groups:
                row['group'] = f'{label}-src{i // 2}'
            rows.append(row)
    path = root / 'clips.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _run(root, out, *extra):
    import deeptune
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'dummy_video3d', '--num_epochs', '1', '--batch_size', '4',
                   '--num_frames', '2', '--added_layers', '1', '--embed_size', '16', '--fixed-seed', '--num_workers', '0',
                   '--df', str(_manifest(root)), '--raw-data', '--report_boot', '50', *extra])
    return next(Path(out).glob('deeptune-*'))


def _split(run, name):
    return pd.read_parquet(next(run.glob(f'data_splits_*/{name}_split.parquet')))


def test_kfold_end_to_end_keeps_the_test_rows_out_and_groups_whole(tmp_path, small_pipeline):
    run = _run(tmp_path, tmp_path / 'out', '--validation', 'kfold', '--k', '2', '--keep_fold_models')
    root = run / 'validation_kfold'
    test_videos = set(_split(run, 'test')['videos'])
    pool = pd.concat([_split(run, 'train'), _split(run, 'val')])
    seen = set()
    for fold in sorted(p for p in root.glob('fold_*') if p.is_dir()):
        parts = {role: pd.read_parquet(fold / f'{role}.parquet') for role in ('train', 'val', 'eval')}
        for frame in parts.values():
            assert not set(frame['videos']) & test_videos  # the sealed set reached no fold
        for a, b in (('train', 'val'), ('train', 'eval'), ('val', 'eval')):
            assert not set(parts[a]['group']) & set(parts[b]['group'])
        seen |= set(parts['eval']['videos'])
    assert seen == set(pool['videos'])  # every pool row is scored out of fold
    summary = json.loads((root / 'cv_summary.json').read_text())
    assert summary['grouped'] is True and summary['n_scored_rows'] == len(pool)
    assert summary['out_of_fold']['intervals']['accuracy']['unit'] == 'group'
    assert summary['calibration']['eval_rows'] == 'out-of-fold'
    oof = pd.read_csv(root / 'oof_predictions.csv')
    assert sorted(oof['row_id']) == list(range(len(pool))) and oof['expected_error'].notna().all()
    assert len(pd.read_csv(root / 'fold_metrics.csv')) == 2

    # The final model and its one test score are the usual ones, with the report beside them.
    predictions = pd.read_csv(next(run.glob('eval_output_*/predictions.csv')))
    assert len(predictions) == len(test_videos)
    report = json.loads(next(run.glob('eval_output_*/evaluation_report.json')).read_text())
    assert report['split'] == 'test' and report['n'] == len(test_videos) and report['resampling_unit'] == 'group'
    calibration = report['calibration']
    assert calibration['fit_split'] == 'validation' and calibration['n_fit'] == len(_split(run, 'val'))
    assert calibration['after'] is not None and calibration['temperature'] > 0
    assert next(run.glob('eval_output_*/reliability_diagram.png')).is_file()
    recorded = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    assert recorded['validation'] == 'kfold' and recorded['k'] == 2 and recorded['report_boot'] == 50


def test_fold_files_are_removed_unless_asked_to_keep_them(tmp_path, small_pipeline):
    run = _run(tmp_path, tmp_path / 'out', '--validation', 'kfold', '--k', '2', '--no_error_model')
    fold = run / 'validation_kfold' / 'fold_0'
    assert not list(fold.glob('*.parquet')) and not list(fold.rglob('*.pth'))
    assert next(fold.glob('eval_output_*/predictions.csv')).is_file()
    assert pd.read_csv(run / 'validation_kfold' / 'oof_predictions.csv')['expected_error'].isna().all()


def test_bootstrap632_end_to_end_draws_whole_groups(tmp_path, small_pipeline):
    run = _run(tmp_path, tmp_path / 'out', '--validation', 'bootstrap632', '--n_boot', '2', '--keep_fold_models')
    root = run / 'validation_bootstrap632'
    summary = json.loads((root / 'cv_summary.json').read_text())['bootstrap632']
    assert summary['resampled_units'] == 'groups' and summary['error_632_plus'] is not None
    test_videos = set(_split(run, 'test')['videos'])
    fold = pd.read_parquet(root / 'resample_0' / 'eval.parquet')
    fit = pd.concat([pd.read_parquet(root / 'resample_0' / f'{role}.parquet') for role in ('train', 'val')])
    assert not set(fold['group']) & set(fit['group']) and not (set(fold['videos']) | set(fit['videos'])) & test_videos
    apparent = pd.read_parquet(root / 'apparent' / 'eval.parquet')
    assert not set(apparent['videos']) & test_videos


def test_export_with_tta_motion_pca_and_ids(tmp_path, small_pipeline):
    run = _run(tmp_path, tmp_path / 'out', '--export_dfanalyze', '--dfanalyze_extras', 'tta', 'motion', 'ids', '--dfanalyze_clips', '2',
               '--dfanalyze_dims', '3', '--report_boot', '0')
    folder = next(run.glob('dfanalyze_export_*'))
    manifest = json.loads((folder / 'dfanalyze_export.json').read_text())
    test = _split(run, 'test')
    assert manifest['pca']['n_fit_rows'] == len(_split(run, 'train')) and manifest['pca']['fit_split'] == 'train'
    assert manifest['clips_per_video'] == 2 and manifest['motion_method'] == 'framediff'
    table = pd.read_csv(folder / 'dfanalyze.csv')
    assert len(table) == len(test) and table['group'].tolist() == test['group'].tolist()
    assert table['target'].tolist() == test['labels'].tolist()
    assert {'pca_0', 'pca_std_0', 'motion_mean', 'split', 'video_id'} <= set(table.columns)
    assert (table['motion_mean'] > 0).all()  # the clips move
    assert not list(run.glob('calibration_fit'))  # --report_boot 0
    assert json.loads((folder / 'cli_arguments.json').read_text())['dfanalyze_dims'] == 3


def test_every_split_is_embedded_with_test_time_sampling(tmp_path, small_pipeline, monkeypatch):
    import embed.video.dfanalyze as export_mod
    seen = []
    original = export_mod.open_video_dataset

    def spy(path, **kw):
        seen.append(kw['train'])
        return original(path, **kw)
    monkeypatch.setattr(export_mod, 'open_video_dataset', spy)
    _run(tmp_path, tmp_path / 'out', '--export_dfanalyze', '--video_augment', '--report_boot', '0')
    assert seen == [False, False, False]


def test_std_is_zero_for_identical_clips_and_mean_matches_the_model(tmp_path, small_pipeline):
    """Two clips of a static video are the same, so the per-clip spread is 0: the std really is across clips."""
    import torch

    import embed.video.dfanalyze as export_mod
    from datasets.video_datasets import VideoDataOptions
    from src.video.feature_backbones import SnippetBackbone
    path = translating_clip(tmp_path / 'still.mp4', 6, 0, 0, size=32, seed=1)
    table = tmp_path / 'one.csv'
    pd.DataFrame({'videos': [str(path)], 'labels': [1]}).to_csv(table, index=False)
    model = SnippetBackbone('dummy_video3d', pretrained=False)
    part = export_mod.embed_split(model, table, 'dummy_video3d', 2, VideoDataOptions(num_workers=0), 2, 1, want_std=True, device=torch.device('cpu'))
    assert part.embedding.shape == (1, 512) and part.embedding_std.shape == (1, 512)
    assert np.abs(part.embedding_std).max() < 1e-3 and part.target.tolist() == [1]


def test_flags_are_refused_where_they_do_not_apply(tmp_path, capsys):
    import deeptune
    frame = tmp_path / 'x.parquet'
    pd.DataFrame({'a': [1.0], 'labels': [0]}).to_parquet(frame)
    base = ['--out', str(tmp_path), '--df', str(frame)]
    with pytest.raises(SystemExit):
        deeptune.main(['--modality', 'images', *base, '--validation', 'kfold'])
    assert 'only supported for the video modality' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        deeptune.main(['--modality', 'images', *base, '--export_dfanalyze'])
    assert 'only supported for the video modality' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        deeptune.main(['--modality', 'video', *base, '--validation', 'kfold', '--k', '1'])
    assert '--k' in capsys.readouterr().err


def test_an_old_run_without_the_report_still_reads(tmp_path):
    """Results reads what it finds: a run from before Phase 5 has no report, calibration or validation files."""
    from desktop.config import read_run_results
    run = tmp_path / 'deeptune-old'
    (run / 'eval_output_x').mkdir(parents=True)
    (run / 'eval_output_x' / 'full_metrics.json').write_text(json.dumps({'loss': 1.0, 'accuracy': .5}))
    results = read_run_results(run)
    assert results.metrics['accuracy'] == .5 and not results.warnings


def test_a_new_run_does_not_confuse_results_with_fold_folders(tmp_path, small_pipeline):
    from desktop.config import read_run_results
    run = _run(tmp_path, tmp_path / 'out', '--validation', 'kfold', '--k', '2', '--no_error_model')
    results = read_run_results(run)
    assert results.metrics is not None and 'validation_kfold' not in str(results.checkpoint_path) and not results.warnings
    assert 'calibration_fit' not in str(next(iter(run.glob('eval*/full_metrics.json'))))

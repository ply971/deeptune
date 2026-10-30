"""deeptune.py -> train / eval / embed -> infer.py end to end on tiny clips, for
the new path manifest (groups, segment sampling, augmentation, multi-clip TTA)
and the original bytes-in-parquet format. CPU, dummy_video3d, 32x32 frames."""
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
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import inference.video as infer_mod
    import trainers.video.train as train_mod
    for module in (train_mod, eval_mod, embed_mod, infer_mod):
        monkeypatch.setattr(module, 'transformations', SMALL)
    versions = dict(config.MODEL_VERSIONS_BY_MODALITY)
    versions['video'] = [*versions['video'], 'dummy_video3d']
    monkeypatch.setattr(config, 'MODEL_VERSIONS_BY_MODALITY', versions)


def _clips(root: Path, per_class: int = 6):
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(per_class):
            path = translating_clip(root / f'{label}_{i}.mp4', 10, dx, 0, size=32, seed=i)
            rows.append({'videos': str(path), 'labels': label, 'group': f'{label}-src{i // 2}'})
    return pd.DataFrame(rows)


def _run(tmp_path, out, extra):
    import deeptune
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'dummy_video3d', '--num_epochs', '1',
                   '--batch_size', '4', '--num_frames', '2', '--added_layers', '1', '--embed_size', '16',
                   '--fixed-seed', *extra])
    run = next(out.glob('deeptune-*'))
    return run


def test_path_manifest_with_groups_sampling_augment_and_test_clips(tmp_path, small_pipeline):
    manifest = tmp_path / 'clips.csv'
    _clips(tmp_path).to_csv(manifest, index=False)
    out = tmp_path / 'out'
    run = _run(tmp_path, out, ['--df', str(manifest), '--raw-data', '--temporal_sampling', 'segment',
                               '--video_augment', '--test_clips', '2'])
    splits = run.glob('data_splits_*/*_split.parquet')
    frames = {p.stem: pd.read_parquet(p) for p in splits}
    assert set(frames) == {'train_split', 'val_split', 'test_split'}
    for frame in frames.values():
        assert frame['videos'].map(type).eq(str).all()  # paths, not bytes
    groups = [set(f['group']) for f in frames.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    dataset_parquet = next(run.glob('video_dataset_*.parquet'))
    assert dataset_parquet.stat().st_size < 20_000  # no clip bytes copied

    train_args = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    assert train_args['temporal_sampling'] == 'segment' and train_args['test_clips'] == 2
    assert train_args['video_augment'] is True and train_args['grouper'] == 'group'
    predictions = pd.read_csv(next(run.glob('eval_output_*/predictions.csv')))
    assert len(predictions) == len(frames['test_split'])
    embeddings = pd.read_parquet(next(run.glob('embed_output_*/*.parquet')))
    assert len(embeddings) == len(frames['test_split']) and 'group' in embeddings.columns

    # The Test page path: rebuild the pipeline from cli_arguments.json and run infer.py.
    from desktop.config import build_infer_argv, infer_config_from_run, read_run_results
    import infer
    data = tmp_path / 'new'
    for row in frames['test_split'].itertuples():
        (data / ('left' if 'left' in Path(row.videos).name else 'right')).mkdir(parents=True, exist_ok=True)
        (data / ('left' if 'left' in Path(row.videos).name else 'right') / Path(row.videos).name).write_bytes(Path(row.videos).read_bytes())
    cfg = infer_config_from_run(read_run_results(run), data, tmp_path / 'infer')
    argv = build_infer_argv(cfg)
    assert argv[argv.index('--test_clips') + 1] == '2'
    assert infer.main(argv) == 0
    inferred = pd.read_csv(tmp_path / 'infer' / 'predictions.csv')
    assert len(inferred) == len(frames['test_split'])
    assert inferred['clip_disagreement'].between(0, 1).all()
    assert 'expected_error' in inferred.columns


def test_original_bytes_parquet_still_trains_evaluates_and_infers(tmp_path, small_pipeline):
    table = _clips(tmp_path).drop(columns=['group'])
    table['videos'] = table['videos'].map(lambda p: Path(p).read_bytes())
    table['labels'] = table['labels'].map({'left': 0, 'right': 1})
    source = tmp_path / 'old.parquet'
    table.to_parquet(source)
    out = tmp_path / 'out'
    run = _run(tmp_path, out, ['--df', str(source)])
    split = pd.read_parquet(next(run.glob('data_splits_*/test_split.parquet')))
    assert split['videos'].map(type).eq(bytes).all()
    metrics = json.loads(next(run.glob('eval_output_*/full_metrics.json')).read_text())
    assert 'accuracy' in metrics
    saved = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    # A saved run from before Phase 1 lacks the new keys: inference must still work.
    for key in ('temporal_sampling', 'test_clips', 'video_backend', 'num_workers', 'video_augment'):
        saved.pop(key)
    from desktop.config import RunResults, build_infer_argv, infer_config_from_run, read_run_results
    import infer
    results = read_run_results(run)
    results.cli_arguments = saved
    clip = tmp_path / 'one.mp4'
    clip.write_bytes(split['videos'].iloc[0])
    argv = build_infer_argv(infer_config_from_run(results, clip, tmp_path / 'infer'))
    assert '--test_clips' not in argv
    assert infer.main(argv) == 0
    assert len(pd.read_csv(tmp_path / 'infer' / 'predictions.csv')) == 1

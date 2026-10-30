"""Every Phase 3 option through deeptune.py (train, eval, embed), then a new infer.py that rebuilds the model from the
run's cli_arguments.json. CPU, tiny clips and models."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torchvision

pytest.importorskip('av')

from datasets.video_synth import translating_clip

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
OPTIONS = {
    'frame_pool_attention': ('resnet_tiny', ['--pooling', 'attention'], 16),
    'single_frame': ('resnet_tiny', ['--video_arch', 'single_frame'], 16),
    'late_fusion': ('resnet_tiny', ['--video_arch', 'late_fusion'], None),
    'early_fusion': ('resnet_tiny', ['--video_arch', 'early_fusion'], 16),
    'slow_fusion': ('dummy_video3d', ['--video_arch', 'slow_fusion'], None),
    'tsm': ('resnet_tiny', ['--video_arch', 'tsm', '--tsm_fold_div', '4'], 16),
    'lstm': ('resnet_tiny', ['--video_arch', 'cnn_rnn', '--rnn_hidden', '8'], 16),
    'gru': ('resnet_tiny', ['--video_arch', 'cnn_rnn', '--rnn_type', 'gru', '--rnn_hidden', '8', '--bidirectional', '--rnn_layers', '2'], 16),
    'transformer': ('resnet_tiny', ['--video_arch', 'temporal_transformer', '--tt_dim', '16', '--tt_heads', '2', '--tt_layers', '1'], 16),
    'two_stream_farneback': ('resnet_tiny', ['--video_arch', 'two_stream', '--motion_source', 'farneback'], 32),
    'two_stream_mv_fc': ('resnet_tiny', ['--video_arch', 'two_stream', '--motion_source', 'mv', '--fusion', 'fc', '--motion_length', '2'], 16),
    'siglip_mean': ('siglip_tiny', [], 16),
    'siglip_cnn_rnn': ('siglip_tiny', ['--video_arch', 'cnn_rnn', '--rnn_hidden', '8'], 16),
}


@pytest.fixture()
def small_pipeline(monkeypatch):
    import embed.video.embed as embed_mod
    import evaluators.video.evaluate as eval_mod
    import inference.video as infer_mod
    import trainers.video.train as train_mod
    for module in (train_mod, eval_mod, embed_mod, infer_mod):
        monkeypatch.setattr(module, 'transformations', SMALL)


def _clips(root: Path, per_class: int = 5):
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(per_class):
            path = translating_clip(root / f'{label}_{i}.mp4', 10, dx, 0, size=48, seed=i)
            rows.append({'videos': str(path), 'labels': label})
    return pd.DataFrame(rows)


@pytest.mark.parametrize('name', OPTIONS)
def test_train_eval_embed_and_infer(name, tmp_path, small_pipeline):
    import deeptune
    import infer
    version, flags, embedding_dim = OPTIONS[name]
    manifest = tmp_path / 'clips.csv'
    _clips(tmp_path).to_csv(manifest, index=False)
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', version, '--num_epochs', '1', '--batch_size', '2',
                   '--num_frames', '4', '--added_layers', '2', '--embed_size', '16', '--fixed-seed', '--df', str(manifest), '--raw-data', *flags])
    run = next(out.glob('deeptune-*'))

    saved = json.loads(next(run.glob('trainval_output_*/cli_arguments.json')).read_text())
    test_split = pd.read_parquet(next(run.glob('data_splits_*/test_split.parquet')))
    metrics = json.loads(next(run.glob('eval_output_*/full_metrics.json')).read_text())
    assert 'accuracy' in metrics
    assert len(pd.read_csv(next(run.glob('eval_output_*/predictions.csv')))) == len(test_split)
    embeddings = pd.read_parquet(next(run.glob('embed_output_*/*.parquet')))
    assert len(embeddings) == len(test_split)
    if embedding_dim is not None:
        assert embeddings.shape[1] == embedding_dim + 1  # the features and the label column

    # A new process knows only the checkpoint: no architecture flag here, it comes from cli_arguments.json.
    checkpoint = next(run.glob('trainval_output_*/model_weights.pth'))
    assert infer.main(['--modality', 'video', '--checkpoint', str(checkpoint), '--data', str(tmp_path / 'left_0.mp4'),
                       '--out', str(tmp_path / 'infer'), '--model_version', saved['model_version'], '--num_classes', '2',
                       '--added_layers', '2', '--embed_size', '16', '--batch_size', '1']) == 0
    predictions = pd.read_csv(tmp_path / 'infer' / 'predictions.csv')
    assert len(predictions) == 1 and str(predictions['predicted_label'].iloc[0]) in ('0', '1', 'left', 'right')


def test_two_stream_with_augmentation_test_clips_and_cache(tmp_path, small_pipeline):
    import deeptune
    import infer
    manifest = tmp_path / 'clips.csv'
    _clips(tmp_path).to_csv(manifest, index=False)
    cache = tmp_path / 'motion'
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', 'resnet_tiny', '--num_epochs', '1', '--batch_size', '2',
                   '--num_frames', '4', '--added_layers', '1', '--embed_size', '16', '--fixed-seed', '--df', str(manifest), '--raw-data',
                   '--video_arch', 'two_stream', '--motion_cache', str(cache), '--video_augment', '--test_clips', '2', '--temporal_sampling', 'segment'])
    assert list(cache.rglob('*.npy'))
    run = next(out.glob('deeptune-*'))
    checkpoint = next(run.glob('trainval_output_*/model_weights.pth'))
    assert infer.main(['--modality', 'video', '--checkpoint', str(checkpoint), '--data', str(tmp_path / 'right_1.mp4'),
                       '--out', str(tmp_path / 'infer'), '--model_version', 'resnet_tiny', '--num_classes', '2',
                       '--added_layers', '1', '--embed_size', '16', '--batch_size', '1']) == 0
    assert 'clip_disagreement' in pd.read_csv(tmp_path / 'infer' / 'predictions.csv').columns


def test_a_two_stream_run_works_from_the_desktop_test_page_settings(tmp_path, small_pipeline):
    """The GUI's Test page rebuilds infer.py's flags from the run; it must carry the architecture and the motion source."""
    from desktop.config import InferenceConfig, build_infer_argv
    argv = build_infer_argv(InferenceConfig(checkpoint_path=tmp_path / 'm.pth', data=tmp_path, out=tmp_path / 'i', modality='video',
                                            model_version='resnet18', num_classes=2, pooling='two_stream:num_frames=8',
                                            video_options={'motion_source': 'mv'}))
    assert argv[argv.index('--pooling') + 1] == 'two_stream:num_frames=8'
    assert argv[argv.index('--motion_source') + 1] == 'mv'
    import infer
    parsed = infer.build_parser().parse_args(argv)
    assert parsed.pooling == 'two_stream:num_frames=8' and parsed.motion_source == 'mv'


@pytest.mark.parametrize('recorded,expected,motion', [
    ({'pooling': 'two_stream:fusion=fc,num_frames=8', 'motion_source': 'mv'}, 'two_stream:fusion=fc,num_frames=8', 'mv'),
    ({'pooling': 'tsm:tsm_fold_div=4'}, 'tsm:tsm_fold_div=4', None),
    ({'pooling': 'attention'}, 'attention', None),  # a run from before Phase 3
    ({}, 'mean', None),
])
def test_the_desktop_test_page_rebuilds_a_recorded_run(tmp_path, recorded, expected, motion):
    from desktop.config import RunResults, build_infer_argv, infer_config_from_run
    import infer
    run = RunResults(tmp_path, cli_arguments={'modality': 'video', 'model_version': 'resnet18', 'num_classes': 2, **recorded},
                     checkpoint_path=tmp_path / 'm.pth')
    argv = build_infer_argv(infer_config_from_run(run, tmp_path / 'clip.mp4', tmp_path / 'i'))
    parsed = infer.build_parser().parse_args(argv)
    assert parsed.pooling == expected
    assert ('--motion_source' in argv) == (motion is not None)
    if motion:
        assert parsed.motion_source == motion

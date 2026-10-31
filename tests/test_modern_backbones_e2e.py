"""Tiny modern backbones through the real entry points: deeptune.py train/eval/embed then infer.py, and the feature route
extract_features.py then train_feature_head.py, then an ensemble of two probes. CPU, tiny clips, no downloads."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torchvision

pytest.importorskip('av')

from datasets.snippet_features import FeatureStore
from datasets.video_synth import translating_clip
from handlers.video_backbones import reset_accepted_licences

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])


@pytest.fixture(autouse=True)
def clean_acceptance():
    reset_accepted_licences()
    yield
    reset_accepted_licences()


def _manifest(root: Path, per_class: int = 5) -> Path:
    rows = []
    for label, dx in (('left', -2), ('right', 2)):
        for i in range(per_class):
            path = translating_clip(root / f'{label}_{i}.mp4', 10, dx, 0, size=48, seed=i)
            rows.append({'videos': str(path), 'labels': label, 'label_idx': int(dx > 0), 'video_id': f'{label}_{i}'})
    manifest = root / 'clips.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _weights(out: Path) -> Path:
    return next(out.rglob('trainval_output_*/model_weights.pth'))


@pytest.mark.parametrize('version,frames,flags,head', [
    ('vjepa2_tiny', '4', [], 'attentive_probe:probe_depth=1,probe_heads=4,probe_width=256'),
    ('vjepa2_tiny', '6', ['--video_arch', 'linear_probe'], 'linear_probe'),
    ('vjepa2_tiny', '4', ['--use-peft', '--lora_rank', '2'], 'attentive_probe:probe_depth=1,probe_heads=4,probe_width=256'),
    ('videomae_tiny', '4', [], 'attentive_probe:probe_depth=1,probe_heads=4,probe_width=256'),
    ('siglip2_tiny', '4', ['--freeze-backbone'], 'mean'),
    ('dinov3_tiny', '4', ['--freeze-backbone'], 'mean'),
])
def test_deeptune_then_a_new_infer_rebuilds_from_the_record(version, frames, flags, head, tmp_path):
    import deeptune
    import infer
    manifest = _manifest(tmp_path)
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--out', str(out), '--model_version', version, '--num_epochs', '1', '--batch_size', '2', '--num_frames', frames,
                   '--added_layers', '2', '--embed_size', '8', '--fixed-seed', '--df', str(manifest), '--raw-data', '--report_boot', '0', *flags])
    checkpoint = _weights(out)
    recorded = json.loads((checkpoint.parent / 'cli_arguments.json').read_text(encoding='utf-8-sig'))
    assert recorded['pooling'] == head and recorded['num_frames'] == int(frames)
    assert recorded['weights_licence'][version].startswith('none')
    assert list(out.rglob('embed*/*')) or list(out.rglob('*embed*')), 'the embedding step ran'
    predictions = tmp_path / 'infer'
    code = infer.main(['--modality', 'video', '--checkpoint', str(checkpoint), '--data', str(tmp_path / 'left_0.mp4'), '--out', str(predictions),
                       '--model_version', version, '--num_classes', str(recorded['num_classes']), '--added_layers', '2', '--embed_size', '8',
                       *(['--use-peft'] if '--use-peft' in flags else [])])
    assert code == 0
    assert len(pd.read_csv(predictions / 'predictions.csv')) == 1


def _extract(manifest, out, version, snippet_frames='4'):
    import extract_features
    assert extract_features.main(['--data', str(manifest), '--out', str(out), '--model_version', version, '--snippet_frames', snippet_frames,
                                  '--cache_tokens', '2x2x2', '--batch_size', '2']) == 0


def _split(manifest: Path, root: Path) -> tuple:
    rows = pd.read_csv(manifest)
    picks = {'train': rows.iloc[[0, 1, 2, 5, 6, 7]], 'val': rows.iloc[[3, 8]], 'test': rows.iloc[[4, 9]]}
    paths = {}
    for name, part in picks.items():
        paths[name] = root / f'{name}.csv'
        part.to_csv(paths[name], index=False)
    return paths['train'], paths['val'], paths['test']


def test_the_default_recipe_on_cached_features_and_an_ensemble_of_two_streams(tmp_path):
    """P-101 end to end: V-JEPA 2 features and SigLIP 2 features, an attentive probe on each, the two averaged."""
    import ensemble
    import train_feature_head
    manifest = _manifest(tmp_path)
    train, val, test = _split(manifest, tmp_path)
    probes = {}
    for version in ('vjepa2_tiny', 'siglip2_tiny'):
        features = tmp_path / f'features_{version}'
        _extract(manifest, features, version)
        sidecar = FeatureStore(features).read_sidecar('rgb', 'left_0')
        assert sidecar['weights_licence'].startswith('none') and sidecar['token_grid'] == [2, 2, 2]
        probes[version] = tmp_path / f'probe_{version}'
        train_feature_head.main(['--features', str(features), '--train_df', str(train), '--val_df', str(val), '--test_df', str(test), '--out', str(probes[version]),
                                 '--target', 'label_idx', '--num_snippets', '2', '--num_epochs', '1', '--batch_size', '2', '--probe_width', '8', '--probe_heads', '2'])
        recorded = json.loads((probes[version] / 'cli_arguments.json').read_text())
        assert recorded['head'] == 'attentive_probe'
        assert recorded['weights_licence']['rgb'].startswith('none')
    out = tmp_path / 'both'
    assert ensemble.main(['--predictions', *[str(p / 'predictions.csv') for p in probes.values()], '--out', str(out)]) == 0
    tables = [pd.read_csv(p / 'predictions.csv').set_index('video_id') for p in probes.values()]
    result = pd.read_csv(out / 'predictions.csv').set_index('video_id')
    assert result['probability_1'].to_dict() == pytest.approx(((tables[0]['probability_1'] + tables[1]['probability_1']) / 2).to_dict())
    assert [next(iter(m['weights_licence'].values())) for m in json.loads((out / 'ensemble.json').read_text())['members']][0].startswith('none')


def test_extraction_refuses_opt_in_weights_before_writing_anything(tmp_path, capsys):
    import extract_features
    manifest = _manifest(tmp_path, per_class=1)
    out = tmp_path / 'features'
    with pytest.raises(SystemExit):
        extract_features.main(['--data', str(manifest), '--out', str(out), '--model_version', 'videomae_b16'])
    assert '--accept_weights_licence CC-BY-NC-4.0' in capsys.readouterr().err
    assert not out.exists()


def test_extraction_refuses_a_snippet_length_the_model_cannot_take(tmp_path):
    from datasets.snippet_features import FeatureConfigError
    from extract_features import extract_features
    manifest = _manifest(tmp_path, per_class=1)
    with pytest.raises(FeatureConfigError, match='--snippet_frames 4'):
        extract_features(manifest, tmp_path / 'features', model_version='videomae_tiny', snippet_frames=8)
    with pytest.raises(FeatureConfigError, match='--feature_input_size 32'):
        extract_features(manifest, tmp_path / 'features', model_version='vjepa2_tiny', snippet_frames=4, feature_input_size=48)


def test_pretrained_embeddings_use_the_bare_tiny_backbone():
    from argparse import Namespace

    from embed.video.embed import load_embedding_model
    args = Namespace()
    backbone = load_embedding_model(None, 'pretrained', 'vjepa2_tiny', 2, 8, 2, args, 'cls')
    assert args.weights_licence.startswith('none')
    assert backbone.default_input_size == 32

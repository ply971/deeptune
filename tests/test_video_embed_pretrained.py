"""`--use_case pretrained` for video embeddings gives bare backbone features, never the output of a random head."""
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision

pytest.importorskip('av')

import embed.video.embed as embed_mod
from datasets.video_synth import translating_clip
from src.video.feature_backbones import SnippetBackbone

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])


class _Args(types.SimpleNamespace):
    def save_args(self, out):
        self.saved_to = Path(out)


@pytest.fixture()
def clips(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, 'transformations', SMALL)
    rows = [{'videos': str(translating_clip(tmp_path / f'c{i}.mp4', 6, i - 2, 0, size=32, seed=i)), 'labels': i % 2} for i in range(4)]
    path = tmp_path / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(path)
    return path


def _embed(df, out, version='resnet18', added_layers=2, embed_size=16, pooling='mean'):
    return embed_mod.embed(df, out, None, 2, 'pretrained', version, added_layers, embed_size, 2, _Args(), 'cls', None, 'test', num_frames=2, pooling=pooling)


@pytest.fixture()
def no_download(monkeypatch):
    """Pretrained weights are not fetched in a unit test; `asked` records that they were requested."""
    asked = []

    def build(version, pretrained=True, **kw):
        asked.append(pretrained)
        return SnippetBackbone(version, pretrained=False, **kw)
    monkeypatch.setattr(embed_mod, 'SnippetBackbone', build)
    return asked


def _table(out):
    return pd.read_parquet(next(Path(out).glob('embed_output_*/*_embeddings.parquet')))


def test_pretrained_asks_for_weights_and_gives_backbone_width_not_the_head(clips, tmp_path, no_download):
    _, shape = _embed(clips, tmp_path / 'a', added_layers=2, embed_size=16)
    assert no_download == [True]
    assert shape == (4, 512 + 1)  # resnet18's 512 features + label; added_layers=2 would have given 16 from a random fc1


def test_pretrained_ignores_the_head_settings(clips, tmp_path, no_download):
    _embed(clips, tmp_path / 'a', added_layers=2, embed_size=16)
    _embed(clips, tmp_path / 'b', added_layers=1, embed_size=64)
    a, b = _table(tmp_path / 'a'), _table(tmp_path / 'b')
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_features_equal_a_direct_backbone_forward(clips, tmp_path, no_download):
    _embed(clips, tmp_path / 'a', version='dummy_video3d')
    table = _table(tmp_path / 'a')
    assert table.shape == (4, 512 + 1)
    from datasets.video_datasets import native_video_transform, open_video_dataset
    dataset = open_video_dataset(clips, transform=SMALL, num_frames=2, clip_transform=native_video_transform('dummy_video3d'))
    clip = dataset[1][0].unsqueeze(0)
    with torch.no_grad():
        direct = SnippetBackbone('dummy_video3d', pretrained=False)(clip).numpy()[0]
    np.testing.assert_allclose(table.drop(columns='label').to_numpy(dtype=np.float32)[1], direct, rtol=1e-5, atol=1e-5)


def test_a_two_stream_spec_still_gives_rgb_features(clips, tmp_path, no_download):
    """Two-stream clips carry motion channels, which a bare RGB backbone cannot take (V-048)."""
    from handlers.video_arch import clip_motion_source
    assert clip_motion_source('two_stream:resnet18', 'farneback') is not None
    args = _Args(pooling='two_stream:resnet18', motion_source='farneback')
    embed_mod.embed(clips, tmp_path / 'a', None, 2, 'pretrained', 'resnet18', 2, 16, 2, args, 'cls', None, 'test', num_frames=2, pooling='two_stream:resnet18')
    _embed(clips, tmp_path / 'b')
    np.testing.assert_array_equal(_table(tmp_path / 'a').to_numpy(), _table(tmp_path / 'b').to_numpy())


def test_the_record_says_where_embeddings_came_from(tmp_path, no_download):
    args = _Args()
    embed_mod.load_embedding_model(None, 'pretrained', 'resnet18', 2, 16, 2, args, 'cls', 'attention')
    assert args.embedding_source.startswith('bare backbone resnet18')
    assert args.weights_licence == SnippetBackbone('resnet18', pretrained=False).weights_licence


def test_finetuned_still_loads_its_checkpoint_and_uses_the_head(tmp_path):
    from utils import build_video_model
    model = build_video_model('dummy_video3d', 2, 2, 16, False, 'cls')
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01)
    path = tmp_path / 'model_weights.pth'
    torch.save(model.state_dict(), path)
    args = _Args()
    got = embed_mod.load_embedding_model(path, 'finetuned', 'dummy_video3d', 2, 16, 2, args, 'cls')
    for key, value in model.state_dict().items():
        assert torch.equal(got.state_dict()[key], value)
    x = torch.zeros(1, 2, 3, 32, 32)
    assert got.eval()(x, extract_embed=True).shape[-1] == 16
    assert str(path) in args.embedding_source

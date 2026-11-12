"""The feature-cache rows of scripts/bench_video_io.py run and report real measurements."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('av')

from datasets.video_synth import translating_clip

BENCH = Path(__file__).resolve().parents[1] / 'scripts' / 'bench_video_io.py'


def _bench_module():
    spec = importlib.util.spec_from_file_location('bench_video_io_under_test', BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_rows_cover_pixels_frame_cache_features_and_token_grids(tmp_path):
    clips = [translating_clip(tmp_path / f'c{i}.mp4', 16, 2, 0, size=32, seed=i) for i in range(3)]
    rows = []
    _bench_module().bench_feature_cache(clips, 8, rows, tmp_path, input_size=32, grids=('none', '8', '99x99x99'),
                                        batch_size=2, snippets_per_video=2)
    variants = [row[1] for row in rows]
    assert 'from pixels (decode every epoch)' in variants
    assert '--frame_cache, 2nd epoch' in variants
    for grid in ('none', '8', '99x99x99'):
        assert any(f'from cached features, tokens={grid}' in v for v in variants)
    storage = {row[0]: float(row[2].split()[0]) for row in rows if row[0].startswith('snippet features')}
    assert len(storage) == 3
    assert storage['snippet features, tokens=none'] < storage['snippet features, tokens=8']
    assert not any('failed' in str(row[2]) or 'until' in str(row[2]) for row in rows)
    assert 'Timing only' in next(row[3] for row in rows if 'from cached features' in row[1])

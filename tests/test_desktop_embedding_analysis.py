"""Features & probes page, card 3: analyse_embeddings.py checks, argv and results view (the GUI side stays torch-free)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from desktop import tools


def _table(path: Path, n=60, outliers=6):
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 5, size=(2, 6))
    X = np.vstack([centres[i % 2] + rng.normal(0, 0.5, size=6) for i in range(n)] + [rng.normal(0, 9, size=6) for _ in range(outliers)])
    frame = pd.DataFrame(X, columns=[str(i) for i in range(6)])
    frame['label'] = [f'c{i % 2}' for i in range(n)] + ['odd'] * outliers
    frame.to_csv(path, index=False)
    return path


def test_checks_before_launch(tmp_path):
    cfg = lambda **v: tools.ToolConfig('analyse_embeddings', {'embeddings': str(tmp_path / 'e.csv'), 'out': str(tmp_path / 'o'), **v})  # noqa: E731
    assert any('not found' in i for i in tools.validate_tool(cfg()))
    _table(tmp_path / 'e.csv')
    assert tools.validate_tool(cfg()) == []
    for bad, words in (({'k_min': 5, 'k_max': 3}, 'Cluster range'), ({'target_fpr': 1.0}, 'between 0 and 1'),
                       ({'ae_epochs': 0}, 'at least 1'), ({'outlier_label': ['odd'], 'label_column': ''}, 'label column')):
        assert any(words in i for i in tools.validate_tool(cfg(**bad))), bad
    (tmp_path / 'e.txt').write_text('x')
    assert any('.parquet or .csv' in i for i in tools.validate_tool(cfg(embeddings=str(tmp_path / 'e.txt'))))


def test_argv_runs_and_results_are_read_back(tmp_path):
    table = _table(tmp_path / 'e.csv')
    cfg = tools.ToolConfig('analyse_embeddings', {'embeddings': str(table), 'out': str(tmp_path / 'o'),
                                                  'outlier_label': ['odd'], 'ae_epochs': 40, 'k_max': 4, 'no_plots': True})
    argv = tools.build_tool_argv(cfg)
    assert argv[:4] == ['--embeddings', str(table), '--out', str(tmp_path / 'o')]
    repo = Path(tools.__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, str(repo / 'analyse_embeddings.py'), *argv], cwd=repo, capture_output=True,
                          text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    sections = tools.embedding_analysis_sections(tmp_path / 'o')
    overview = dict(sections[0].rows)
    assert overview['clips'] == '66'
    assert 'Anomaly AUC (known outliers)' in overview
    assert sections[1].title == 'Clips, most unusual first'
    top = sections[1].table.head(6)
    assert (top['label'] == 'odd').sum() >= 4  # the injected clips lead the list

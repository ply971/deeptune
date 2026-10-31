"""handlers/get_intersection.py: the helper for df-analyze's holdout, for every modality's embedding file."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from handlers import get_intersection as gi

ROOT = Path(__file__).resolve().parents[1]


def _embeddings(n=20, seed=0):
    values = np.random.default_rng(seed).normal(size=(n, 4)).astype(np.float32)
    frame = pd.DataFrame(values)  # integer column names, as every embed script writes them
    frame['label'] = np.arange(n) % 2
    return frame


def test_finds_the_holdout_rows_with_integer_column_names(tmp_path):
    frame = _embeddings()
    frame.to_parquet(tmp_path / 'e.parquet')
    holdout = frame.sample(8, random_state=0)
    holdout.to_csv(tmp_path / 'h.csv', index=False)
    path = gi.main(['--df_parquet_path', str(tmp_path / 'e.parquet'), '--df_csv_path', str(tmp_path / 'h.csv'), '--out', str(tmp_path / 'new' / 'dir')])
    out = pd.read_parquet(path)
    assert len(out) == 8 and path.name.startswith('intersection_')
    expected = frame.loc[sorted(holdout.index)].reset_index(drop=True).rename(columns=str)
    pd.testing.assert_frame_equal(out, expected)


def test_integer_and_text_column_names_are_the_same_column():
    left = pd.DataFrame({0: [.5, .25], 1: [1., 2.]})
    right = pd.DataFrame({'0': [.25], '1': [2.]})
    out = gi.intersect(left, right)
    assert out.columns.tolist() == ['0', '1'] and out['0'].tolist() == [.25]


def test_rows_sharing_only_the_label_do_not_match():
    frame = _embeddings()
    other = _embeddings(seed=1)
    assert gi.intersect(frame, other).empty


def test_keeps_text_columns_and_columns_only_the_csv_has():
    left = pd.DataFrame({'a': [.1, .2, .3], 'text': ['x', 'y', 'z'], 'row': [0, 1, 2]})
    right = pd.DataFrame({'a': [.3, .1], 'text': ['z', 'x'], 'pred': [7, 8]})
    out = gi.intersect(left, right)
    assert out['row'].tolist() == [0, 2] and out['pred'].tolist() == [8, 7] and out['a'].tolist() == [.1, .3]


@pytest.mark.parametrize('shared', [['target'], ['label', 'video_id'], ['group', 'split', 'row_id']])
def test_sharing_only_non_features_is_refused(shared):
    """A merge on the target alone once returned all 10 Parquet rows from a 2-row CSV."""
    left = pd.DataFrame({'emb_0': np.arange(10.), **{c: [0, 1] * 5 for c in shared}})
    right = pd.DataFrame({'f0': [0., 1.], **{c: [0, 1] for c in shared}})
    with pytest.raises(ValueError, match='no features'):
        gi.intersect(left, right)


def test_identical_csv_rows_do_not_multiply_a_row():
    left = pd.DataFrame({'a': [.5], 'b': [1]})
    assert len(gi.intersect(left, pd.concat([left, left]))) == 1


def test_no_shared_column_is_an_error():
    with pytest.raises(ValueError, match='share no column'):
        gi.intersect(pd.DataFrame({'a': [1]}), pd.DataFrame({'b': [1]}))


def test_runs_as_a_module_without_torch(tmp_path):
    frame = _embeddings()
    frame.to_parquet(tmp_path / 'e.parquet')
    frame.head(3).to_csv(tmp_path / 'h.csv', index=False)
    code = ('import sys; from handlers.get_intersection import main; '
            f"main(['--df_parquet_path', r'{tmp_path / 'e.parquet'}', '--df_csv_path', r'{tmp_path / 'h.csv'}', '--out', r'{tmp_path}']); "
            "assert 'torch' not in sys.modules")
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert 'There are 3 records' in done.stdout

from pathlib import Path

import pandas as pd
import pytest

from handlers.video_manifest import read_video_manifest
from handlers.raw_to_parquet_dataset import raw_to_parquet


def test_mp4_manifest_resolves_paths_and_preserves_bytes_labels_and_groups(tmp_path):
    for name in ('first.MP4', 'second.mp4'):
        (tmp_path / name).write_bytes(name.encode())
    path = tmp_path / 'videos.csv'
    pd.DataFrame({'videos': ['first.MP4', 'second.mp4'], 'labels': ['walk', 'run'], 'patient': ['a', 'b']}).to_csv(path, index=False)
    frame = read_video_manifest(path)
    assert all(Path(name).is_absolute() for name in frame.videos)
    output = pd.read_parquet(raw_to_parquet(path, tmp_path / 'out', 'video'))
    assert output.videos.tolist() == [b'first.MP4', b'second.mp4']
    assert output.labels.tolist() == ['walk', 'run']
    assert output.patient.tolist() == ['a', 'b']


def test_missing_mp4_produces_an_actionable_row_error(tmp_path):
    path = tmp_path / 'videos.csv'
    pd.DataFrame({'videos': ['missing.mp4'], 'labels': ['walk']}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='row 1: file not found'):
        read_video_manifest(path)


def test_duplicate_videos_cannot_leak_across_dataset_splits(tmp_path):
    (tmp_path / 'clip.mp4').write_bytes(b'clip')
    path = tmp_path / 'videos.csv'
    pd.DataFrame({'videos': ['clip.mp4', './clip.mp4'], 'labels': ['a', 'a']}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='more than once'):
        read_video_manifest(path)


def test_one_video_may_give_several_clips_as_time_ranges_of_one_group(tmp_path):
    (tmp_path / 'long.mp4').write_bytes(b'clip')
    path = tmp_path / 'videos.csv'
    rows = {'videos': ['long.mp4', 'long.mp4', './long.mp4'], 'labels': ['jump', 'none', 'run'],
            'start_sec': [0.0, 2.0, 4.0], 'end_sec': [2.0, 4.0, 6.0], 'group': ['long'] * 3}
    pd.DataFrame(rows).to_csv(path, index=False)
    assert len(read_video_manifest(path)) == 3


@pytest.mark.parametrize('change,message', [
    ({'start_sec': [0.0, 0.0], 'end_sec': [2.0, 2.0]}, 'same start_sec/end_sec'),
    ({'group': None}, 'need a group column'),
    ({'group': ['a', 'b']}, 'share one group'),
    ({'start_sec': None, 'end_sec': None}, 'more than once'),
    ({'start_sec': [float('nan')] * 2, 'end_sec': [float('nan')] * 2}, 'needs a start_sec and end_sec'),  # blank = whole video
    ({'start_sec': [0.0, float('nan')], 'end_sec': [2.0, float('nan')]}, 'needs a start_sec and end_sec'),
    ({'start_sec': [0.0, 4.0], 'end_sec': [2.0, 3.0]}, 'start before end'),
])
def test_repeated_videos_must_be_distinct_ranges_of_one_group(tmp_path, change, message):
    (tmp_path / 'long.mp4').write_bytes(b'clip')
    rows = {'videos': ['long.mp4', 'long.mp4'], 'labels': ['a', 'b'], 'start_sec': [0.0, 2.0], 'end_sec': [2.0, 4.0],
            'group': ['g', 'g']}
    for key, value in change.items():
        if value is None:
            rows.pop(key)
        else:
            rows[key] = value
    path = tmp_path / 'videos.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        read_video_manifest(path)

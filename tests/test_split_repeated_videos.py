"""Clips cut from one video file (time ranges) must never land in two splits, whatever column the split groups by."""
import pandas as pd
import pytest

from handlers.split_dataset import check_group_split, prepare_whole_dataset, split_dataset


def _ranges_table(tmp_path, n_videos=20, per_video=3):
    """n_videos files x per_video time ranges; 'group' is the video, 'actor' varies inside a video."""
    rows = []
    for v in range(n_videos):
        for r in range(per_video):
            rows.append({'videos': str(tmp_path / f'v{v:02d}.mp4'), 'labels': ['a', 'b'][(v + r) % 2],
                         'start_sec': 2.0 * r, 'end_sec': 2.0 * r + 2, 'group': f'v{v:02d}', 'actor': f'p{(v + r) % 7}'})
    return pd.DataFrame(rows)


def _split(tmp_path, frame, grouper=None):
    path = tmp_path / 'clips.parquet'
    frame.to_parquet(path, index=False)
    return split_dataset(0.6, 0.2, 0.2, path, tmp_path / 'out', True, False, 'labels', 'video', grouper=grouper)


def test_a_grouper_that_splits_a_video_is_refused(tmp_path):
    # Review V-061: grouping by 'actor' put 8 of 20 videos in both train and test, silently.
    with pytest.raises(ValueError, match="different 'actor' values"):
        _split(tmp_path, _ranges_table(tmp_path), grouper='actor')


def test_repeated_videos_without_a_grouper_are_split_by_row_as_before(tmp_path):
    # No grouper: the run warns instead (tests/test_evaluation_pipeline_e2e.py checks the warning), so the split
    # itself goes through.
    paths = _split(tmp_path, _ranges_table(tmp_path).drop(columns=['group', 'actor']))
    assert sum(len(pd.read_parquet(p)) for p in paths) == 60


def test_grouping_by_video_keeps_every_video_in_one_split(tmp_path):
    paths = _split(tmp_path, _ranges_table(tmp_path))
    owner = {}
    for name, path in zip(('train', 'val', 'test'), paths):
        for video in pd.read_parquet(path)['videos']:
            assert owner.setdefault(video, name) == name
    assert len(owner) == 20


def test_a_consistent_coarser_grouper_is_accepted(tmp_path):
    frame = _ranges_table(tmp_path)
    frame['site'] = frame['group'].str[-1]          # 10 sites, each holding whole videos
    paths = _split(tmp_path, frame, grouper='site')
    videos = [set(pd.read_parquet(p)['videos']) for p in paths]
    assert not (videos[0] & videos[1]) and not (videos[0] & videos[2]) and not (videos[1] & videos[2])


def test_cross_validation_pool_refuses_a_grouper_that_splits_a_video(tmp_path):
    path = tmp_path / 'clips.parquet'
    _ranges_table(tmp_path).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="different 'actor' values"):
        prepare_whole_dataset(path, tmp_path / 'out', False, 'labels', 'video', grouper='actor')


def test_group_check_also_catches_one_video_in_two_splits():
    a = pd.DataFrame({'videos': ['x.mp4', 'y.mp4'], 'g': [1, 2], 'labels': [0, 1]})
    b = pd.DataFrame({'videos': ['x.mp4'], 'g': [3], 'labels': [0]})
    with pytest.raises(AssertionError, match='appears in both'):
        check_group_split({'train': a, 'test': b}, 'g')

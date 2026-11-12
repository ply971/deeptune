"""Read a labeled list of local video files without loading the ML stack."""
from pathlib import Path
import os

import pandas as pd

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv', '.mpeg', '.mpg'}


def read_video_manifest(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)
    if 'videos' not in frame or 'labels' not in frame:
        raise ValueError("A video list needs 'videos' (file paths) and 'labels' columns.")
    if frame.empty or frame['labels'].isna().any():
        raise ValueError('Add at least one clip and give every clip a label.')
    paths = []
    for index, value in enumerate(frame['videos']):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Video list row {index + 1} has no file path.')
        clip = Path(value).expanduser()
        clip = (path.parent / clip).resolve() if not clip.is_absolute() else clip.resolve()
        if not clip.is_file():
            raise ValueError(f'Video list row {index + 1}: file not found: {clip}')
        if clip.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f'Video list row {index + 1}: unsupported video extension: {clip.suffix}')
        paths.append(str(clip))
    keys = [os.path.normcase(value) for value in paths]
    if len(set(keys)) != len(keys):
        _check_repeated_videos(frame, keys)
    frame['videos'] = paths
    return frame


def _check_repeated_videos(frame: pd.DataFrame, keys: list) -> None:
    """A file may appear on several rows only as distinct time ranges (start_sec/end_sec: segment-level labels) of
    one group, so all clips cut from one video stay in one split and nothing leaks from train to test."""
    if 'start_sec' not in frame or 'end_sec' not in frame:
        raise ValueError('The same video appears more than once. Remove duplicates before training, or give each row '
                         'its own start_sec/end_sec range (and a group column) to cut several clips from one video.')
    start = pd.to_numeric(frame['start_sec'], errors='coerce')
    end = pd.to_numeric(frame['end_sec'], errors='coerce')
    repeated = pd.Series(keys).duplicated(keep=False).to_numpy()
    no_range = repeated & ~(start.notna() & end.notna() & (start < end)).to_numpy()
    if no_range.any():  # a blank range is the whole video, and NaN never equals NaN, so the set below would miss it
        raise ValueError(f'Video list row {int(no_range.argmax()) + 1}: a video that appears on several rows needs a '
                         'start_sec and end_sec on every row (start before end).')
    ranges = list(zip(keys, start, end))
    if len(set(ranges)) != len(ranges):
        raise ValueError('The same video appears more than once with the same start_sec/end_sec. Remove duplicates before training.')
    if 'group' not in frame or frame['group'].isna().any():
        raise ValueError('Clips cut from the same video need a group column (for example the video name), so they '
                         'all land in the same split and the test split never sees a training video.')
    groups = pd.DataFrame({'key': keys, 'group': frame['group'].astype(str)}).groupby('key')['group'].nunique()
    if (groups > 1).any():
        raise ValueError('Clips cut from the same video must share one group, or one video could be in two splits: '
                         + ', '.join(groups[groups > 1].index[:3]))


def load_video_manifest(path: Path, read_bytes: bool = True) -> pd.DataFrame:
    """The validated manifest; `read_bytes=False` keeps the resolved absolute
    paths (the scalable default for --raw-data training, see --video_storage)
    instead of embedding every clip's bytes."""
    frame = read_video_manifest(path)
    if read_bytes:
        frame['videos'] = frame['videos'].map(lambda name: Path(name).read_bytes())
    return frame

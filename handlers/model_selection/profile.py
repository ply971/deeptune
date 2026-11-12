"""Stage 0 of a search: a profile of the development rows, with suggestions. It never changes the data.

It reads `dev.parquet` only (never the sealed test rows) and, for clips stored as file paths, opens a seeded sample of
up to `clips` of them for their length and a quick motion probe (Farneback flow from benchmarks/motion_features.py on 8
small grey frames). Every number is printed with what it is; the thresholds behind the suggestions are rough rules of
thumb, and the profile says so.

Cheap label checks: one video file listed with different labels, the same file listed twice, label spellings that differ
only in case or spaces, classes with fewer groups than folds, and groups whose clips carry several labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

LOW_MOTION, HIGH_MOTION = 0.002, 0.02  # mean flow per frame as a share of the frame width; rough rules of thumb
SMALL_DATA = 200  # development clips
PROBE_FRAMES, PROBE_WIDTH = 8, 64


def _clip_facts(path: str) -> Optional[dict]:
    """Frame count, fps, size and a motion number of one clip, or None if it cannot be opened."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        wanted = set(np.linspace(0, max(frames - 1, 0), PROBE_FRAMES).round().astype(int).tolist()) if frames else set(range(PROBE_FRAMES))
        grabbed, index = [], 0
        while len(grabbed) < len(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                scale = PROBE_WIDTH / frame.shape[1]
                small = cv2.resize(frame, (PROBE_WIDTH, max(8, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
                grabbed.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()
    motion = None
    if len(grabbed) >= 2:
        from benchmarks.motion_features import farneback_flow
        flow = farneback_flow(np.stack(grabbed))
        step = max(1.0, (frames - 1) / (len(grabbed) - 1)) if frames else 1.0  # frames between the sampled ones
        motion = float(np.linalg.norm(flow, axis=-1).mean() / PROBE_WIDTH / step)
    return {'frames': frames or index, 'fps': fps, 'width': width, 'height': height, 'motion': motion}


def _label_issues(frame: pd.DataFrame, target: str, group_col: Optional[str], k: int, mode: str) -> list:
    issues = []
    if 'videos' in frame and frame['videos'].map(lambda v: isinstance(v, str)).all():
        by_file = frame.groupby('videos')[target].nunique()
        conflicting = by_file[by_file > 1]
        if len(conflicting):
            issues.append(f'{len(conflicting)} video file(s) are listed with different labels, e.g. {Path(conflicting.index[0]).name}.')
        repeated = frame.duplicated(['videos', target]).sum()
        if repeated and not {'start_sec', 'end_sec'} & set(frame.columns):
            issues.append(f'{int(repeated)} row(s) repeat a video file with the same label (and no trim columns): duplicates count twice.')
    if mode == 'cls':
        labels = frame[target].astype(str)
        spellings = labels.groupby(labels.str.strip().str.casefold()).unique()
        mixed = [list(values) for values in spellings if len(values) > 1]
        if mixed:
            issues.append(f'labels that differ only in case or spaces: {mixed[:3]}. They count as different classes.')
        if group_col:
            per_class = frame.groupby(target)[group_col].nunique()
            short = per_class[per_class < k]
            if len(short):
                issues.append(f'class(es) with fewer groups than k={k} folds: {dict(short.astype(int))}; some folds will hold out none of '
                              'their clips, and the fold scores of those classes are not comparable.')
            mixed_groups = frame.groupby(group_col)[target].nunique()
            if (mixed_groups > 1).any():
                issues.append(f'{int((mixed_groups > 1).sum())} group(s) hold clips of several labels (fine if expected, e.g. one '
                              'recording with several actions).')
    return issues


def profile_development(dev: Path, split: dict, settings, grid, mode: str, clips: int = 16) -> dict:
    """Facts and suggestions about the development rows. Reads dev.parquet and opens up to `clips` sampled clip files."""
    from handlers.split_dataset import resolve_grouper
    frame = pd.read_parquet(dev)
    target = settings.target
    group_col = resolve_grouper(list(frame.columns), settings.grouper, 'video')
    facts = {'rows': int(len(frame)), 'unit': split.get('unit'), 'clips_treated_as_independent': str(split.get('unit', '')).startswith('row')}
    if group_col and group_col in frame:
        sizes = frame.groupby(group_col).size()
        facts['groups'] = {'column': group_col, 'count': int(len(sizes)), 'clips_per_group_min': int(sizes.min()),
                           'clips_per_group_median': float(sizes.median()), 'clips_per_group_max': int(sizes.max())}
    if mode == 'cls':
        counts = frame[target].astype(str).value_counts()
        facts['classes'] = {str(k): int(v) for k, v in counts.items()}
        facts['imbalance_ratio'] = round(float(counts.max() / counts.min()), 3) if len(counts) > 1 else None
    else:
        values = pd.to_numeric(frame[target], errors='coerce')
        facts['target'] = {'mean': float(values.mean()), 'sd': float(values.std()), 'min': float(values.min()), 'max': float(values.max())}
    facts['label_issues'] = _label_issues(frame, target, group_col if group_col in frame else None, settings.k, mode)

    probe = {'asked': int(clips), 'opened': 0}
    paths = frame['videos'] if 'videos' in frame else pd.Series(dtype=object)
    if clips and len(paths) and paths.map(lambda v: isinstance(v, str)).all():
        rng = np.random.default_rng(settings.seed)
        chosen = rng.choice(len(paths), size=min(int(clips), len(paths)), replace=False)
        found = [facts_ for facts_ in (_clip_facts(paths.iloc[i]) for i in sorted(chosen)) if facts_ is not None]
        probe['opened'] = len(found)
        if found:
            frames = np.array([f['frames'] for f in found], dtype=float)
            probe.update(frames_median=float(np.median(frames)), frames_min=int(frames.min()), frames_max=int(frames.max()),
                         fps_median=float(np.median([f['fps'] for f in found])),
                         size_median=f'{int(np.median([f["width"] for f in found]))}x{int(np.median([f["height"] for f in found]))}')
            motion = [f['motion'] for f in found if f['motion'] is not None]
            if motion:
                probe['motion_median'] = float(np.median(motion))
                probe['motion_unit'] = 'mean Farneback flow per frame, as a share of the frame width'
    elif clips and len(paths):
        probe['note'] = 'clips are stored as bytes in the Parquet file; length and motion were not probed'
    facts['clip_probe'] = probe
    return {'what': 'development rows only (dev.parquet); the sealed test rows were not read. Nothing was changed.',
            'facts': facts, 'suggestions': _suggestions(facts, settings, grid, mode)}


def _suggestions(facts: dict, settings, grid, mode: str) -> list:
    out = []
    if facts['clips_treated_as_independent']:
        out.append('Clips are treated as independent. If several come from one person, scene or recording, add a group column '
                   '(or --grouper): otherwise every score reads too high.')
    if facts['rows'] < SMALL_DATA:
        out.append(f'Only {facts["rows"]} development clips (under {SMALL_DATA}): prefer frozen backbones, probes or LoRA over full '
                   'fine-tuning, and expect wide intervals.')
    # Class imbalance is warned about by the engine itself, with the metric in view (search.py).
    out += [f'Label check: {issue}' for issue in facts['label_issues']]
    probe = facts['clip_probe']
    asked = [int(v) for v in grid.num_frames if v != 'default'] + ([8] if 'default' in grid.num_frames else [])
    if probe.get('frames_median') and asked and max(asked) > probe['frames_median']:
        out.append(f'Clips have about {probe["frames_median"]:.0f} frames (median of {probe["opened"]} opened), fewer than the '
                   f'{max(asked)} frames the grid asks for: frames are then repeated, which adds cost and no information.')
    motion = probe.get('motion_median')
    if motion is not None:
        if motion < LOW_MOTION:
            out.append(f'Little motion (median {motion:.4f} of the width per frame): frame models with pooling may match 3D models '
                       'here, at a lower cost. (A rough rule of thumb.)')
        elif motion > HIGH_MOTION:
            out.append(f'Strong motion (median {motion:.4f} of the width per frame): temporal models (3D CNNs, tsm, temporal '
                       'attention) are worth keeping in the grid. (A rough rule of thumb.)')
    if probe.get('asked') and not probe.get('opened') and not probe.get('note'):
        out.append('No clip file could be opened for the length and motion probe; check the paths before training.')
    return out


def print_profile(profile: dict, say: Callable[[str], None]) -> None:
    facts = profile['facts']
    say('Data profile (development rows only; nothing was changed):')
    groups = facts.get('groups')
    say(f'  {facts["rows"]} clips' + (f' in {groups["count"]} groups ({groups["column"]}; {groups["clips_per_group_min"]} to '
                                      f'{groups["clips_per_group_max"]} clips each)' if groups else '') + f'; split unit: {facts["unit"]}.')
    if 'classes' in facts:
        say(f'  classes: {len(facts["classes"])}, clips per class {min(facts["classes"].values())} to {max(facts["classes"].values())}.')
    probe = facts['clip_probe']
    if probe.get('opened'):
        say(f'  clip probe ({probe["opened"]} of {probe["asked"]} sampled clips opened): about {probe["frames_median"]:.0f} frames '
            f'({probe["frames_min"]} to {probe["frames_max"]}), {probe["fps_median"]:.1f} fps, {probe["size_median"]}'
            + (f'; motion {probe["motion_median"]:.4f} ({probe["motion_unit"]})' if 'motion_median' in probe else '') + '.')
    for suggestion in profile['suggestions']:
        say(f'  Suggestion: {suggestion}')

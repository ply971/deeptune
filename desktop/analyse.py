"""Logic behind the Analyse Video page: no Dear PyGui and no torch.

Builds the track.py command line, checks what the user entered, and turns the
files infer.py and track.py leave behind into text the page can show. Every
function here works on plain values and paths, so it is tested without a window
or a model. The page itself is desktop/analyse_page.py.

The modules imported below only load numpy, scipy and pandas at import time.
torch and transformers stay in the subprocess; tests/test_desktop_analyse.py
checks that.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from evaluators.error_model import find_error_model
from handlers.video_manifest import VIDEO_EXTENSIONS
from spatial.detect import (DEFAULT_MODELS, DETECTOR_DEFAULTS, LICENCES, check_grounding_dino_prompts,
                            clean_prompts)
from track import TRACK_DEFAULTS

from . import config as gc

# ---------------------------------------------------------------------------
# What the page offers. A new action adds an entry here when it is reserved,
# and its own card in analyse_page.py when it works.
# ---------------------------------------------------------------------------

LIMIT_NOTE = ('A trained clip classifier answers only with its own classes. Tracking and counting use the '
              'detector named below, not your model.')
TRACKER_NOTE = ('Counts are tracker estimates (unique track IDs), not measured accuracy. A track that is lost '
                'and found again can be counted twice, so check the overlay before you trust a count.')
CONFIDENCE_NOTE = "Confidence is the model's own score, not a measured accuracy on this video."


@dataclass(frozen=True)
class ReservedAction:
    key: str
    title: str
    description: str

    @property
    def label(self) -> str:
        return f'{self.title} (coming later)'


RESERVED_ACTIONS = (
    ReservedAction('segment', 'Segment', 'Masks that follow an object through the video.'),
    ReservedAction('ask', 'Ask', 'Free-form questions about the video, with the sampled frames as evidence.'),
)


@dataclass(frozen=True)
class DetectorChoice:
    key: str
    label: str
    download_gb: float

    @property
    def model_id(self) -> str:
        return DEFAULT_MODELS[self.key]


DETECTORS = (
    DetectorChoice('grounding_dino', 'Grounding DINO', 0.7),
    DetectorChoice('owlv2', 'OWLv2', 0.6),
)


def detector_by_label(label: str) -> DetectorChoice:
    for choice in DETECTORS:
        if choice.label == label:
            return choice
    raise ValueError(f'Unknown detector: {label}')


def detector_by_key(key: str) -> DetectorChoice:
    for choice in DETECTORS:
        if choice.key == key:
            return choice
    raise ValueError(f'Unknown detector: {key}')


def hf_hub_cache() -> Path:
    """Where huggingface_hub keeps downloads, by its documented variables."""
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE'])
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']) / 'hub'
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def detector_is_cached(model_id: str) -> bool:
    """Are the weights there? A snapshot folder alone can be left by a stopped download."""
    snapshots = hf_hub_cache() / f"models--{model_id.replace('/', '--')}" / 'snapshots'
    return any(any(snapshots.glob(f'*/*{suffix}')) for suffix in ('.safetensors', '.bin'))


def detector_note(key: str) -> str:
    """The notice shown before a detector is used."""
    choice = detector_by_key(key)
    total = sum(d.download_gb for d in DETECTORS)
    note = (f'{choice.label} ({choice.model_id}, {LICENCES.get(choice.model_id, "see the model card")}). '
            f'The model is about {choice.download_gb} GB; both detectors together are about {total:.1f} GB.')
    if detector_is_cached(choice.model_id):
        return note + ' It is already in the local Hugging Face cache, so no download is expected.'
    return note + f' It is not in the local cache yet, so the first run downloads it to {hf_hub_cache()}.'


def track_button_label(key: str) -> str:
    """The button says so when pressing it starts a download."""
    choice = detector_by_key(key)
    if detector_is_cached(choice.model_id):
        return 'Find, track and count'
    return f'Download about {choice.download_gb} GB, then track'


def loading_message(key: str) -> str:
    """First line of the live log. track.py prints nothing until a video is done."""
    choice = detector_by_key(key)
    return (f'Loading {choice.label} and tracking. The first use downloads about {choice.download_gb} GB. '
            'track.py reports no progress per frame, so this line can stay here until it finishes.\n')


# ---------------------------------------------------------------------------
# Checking the video
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoCheck:
    path: Optional[Path] = None
    error: str = ''
    description: str = ''

    @property
    def ok(self) -> bool:
        return not self.error


def check_video(text: object) -> VideoCheck:
    """Is this a video OpenCV can decode? Uses the decoder the preview and track.py use."""
    cleaned = str(text or '').strip().strip('"')
    if not cleaned:
        return VideoCheck(error='Choose a video first.')
    path = Path(cleaned).expanduser()
    if not path.exists():
        return VideoCheck(error=f'Video not found: {path}')
    if path.is_dir():
        return VideoCheck(error=f'This is a folder, not a video file: {path}')
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return VideoCheck(error=f'{path.name} is not a supported video type. Use one of: '
                                + ', '.join(sorted(VIDEO_EXTENSIONS)))
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        if not ok or frame is None:
            return VideoCheck(error=f'Could not read {path.name}. Its codec may be unsupported or the file may be '
                                    'corrupt.')
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    height, width = frame.shape[:2]
    parts = [f'{width}x{height}']
    if math.isfinite(fps) and fps > 0:
        if count > 0:
            parts.append(f'{count / fps:.1f} s')
        parts.append(f'{fps:.3g} fps')
    return VideoCheck(path=path, description=f'{path.name}  |  ' + ', '.join(parts))


# ---------------------------------------------------------------------------
# Trained runs
# ---------------------------------------------------------------------------

def run_modality(run_dir: Path) -> Optional[str]:
    """The modality a run recorded, read the way gc.read_run_results finds the file."""
    path = gc._first_glob(run_dir, 'eval*/cli_arguments.json') or gc._first_glob(run_dir, 'train*/cli_arguments.json')
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None
    return data.get('modality') if isinstance(data, dict) else None


@dataclass
class RunChoices:
    runs: dict[str, Path] = field(default_factory=dict)  # label -> run folder, newest first
    hidden: int = 0  # runs that are not video models

    @property
    def message(self) -> str:
        if self.runs:
            return f'{self.hidden} run(s) that are not video models are hidden.' if self.hidden else ''
        if self.hidden:
            return (f'None of the {self.hidden} run(s) in this output folder is a video model. '
                    'Train one first, or choose another output folder on the Run page.')
        return 'No trained runs in this output folder yet.'


def list_video_runs(out_dir: Path) -> RunChoices:
    """Runs under `out_dir` that the page can use. Only a video model can take a clip."""
    choices = RunChoices()
    for run_dir in gc.list_runs(out_dir):
        if run_modality(run_dir) != 'video':
            choices.hidden += 1
            continue
        stamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        choices.runs[f'{stamp}  |  {run_dir.name}'] = run_dir
    return choices


def describe_run(run: gc.RunResults) -> str:
    """Which model this is, in the form every prediction is labelled with."""
    args = run.cli_arguments or {}
    parts = [str(args.get('model_version') or 'unknown model')]
    # Native 3D models ignore --pooling, whatever the run recorded.
    if args.get('pooling') and args.get('model_version') not in gc.VIDEO_NATIVE_MODEL_VERSIONS:
        parts.append(f"{args['pooling']} pooling")
    parts.append('regression' if args.get('mode') == 'reg' else 'classification')
    text = f"{Path(run.run_dir).name} ({', '.join(parts)})"
    if run.checkpoint_path is not None:
        text += f', checkpoint {Path(run.checkpoint_path).name}'
    return text


def class_names_note(run: gc.RunResults, limit: int = 12) -> str:
    """The classes this model can answer with, when the run recorded them."""
    if (run.cli_arguments or {}).get('mode') == 'reg' or not run.label_mapping:
        return ''
    names = sorted(str(name) for name in run.label_mapping)
    shown = ', '.join(names[:limit])
    return f"It can only answer with: {shown}{', ...' if len(names) > limit else ''}."


# Settings that decide the network's shape, and so whether the checkpoint fits.
BUILD_SETTINGS = ('model_version', 'num_classes', 'mode', 'pooling', 'added_layers', 'embed_size', 'use_peft',
                  'freeze_backbone')
# Settings that only change how test clips are built.
CLIP_SETTINGS = ('num_frames', 'video_backend', 'temporal_sampling', 'clip_stride', 'clip_duration_sec',
                 'motion_source', 'test_clips')


def _recorded_arguments(run_dir: Path, pattern: str) -> dict:
    path = gc._first_glob(Path(run_dir), pattern)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def recorded_differences(run_dir: Path, names: tuple[str, ...]) -> list[str]:
    """Settings the run's training and evaluation folders record differently.
    The page prefers the evaluation record and infer.py the training one, so a difference matters."""
    trained = _recorded_arguments(run_dir, 'train*/cli_arguments.json')
    evaluated = _recorded_arguments(run_dir, 'eval*/cli_arguments.json')
    return [f'{name}: {trained[name]!r} in training, {evaluated[name]!r} in evaluation' for name in names
            if trained.get(name) is not None and evaluated.get(name) is not None and trained[name] != evaluated[name]]


def run_problems(run: gc.RunResults) -> list[str]:
    """Why this run's model cannot be rebuilt. Each one stops Predict before it launches."""
    if run.checkpoint_path is None:
        return ['This run has no saved checkpoint yet (it may still be training).']
    problems = []
    if not Path(run.checkpoint_path).is_file():
        problems.append(f'The checkpoint is missing: {run.checkpoint_path}')
    args = run.cli_arguments or {}
    # infer_config_from_run would otherwise fall back to resnet18 with 2 classes.
    required = ['model_version'] + ([] if args.get('mode') == 'reg' else ['num_classes'])
    missing = [name for name in required if not args.get(name)]
    if missing:
        problems.append(f"This run did not record {', '.join(missing)}, so its model cannot be rebuilt.")
    differences = recorded_differences(run.run_dir, BUILD_SETTINGS)
    if differences:
        problems.append("This run's recorded settings disagree, so the model that fits its checkpoint is unknown. "
                        + '; '.join(differences) + '.')
    return problems


def run_warnings(run: gc.RunResults) -> list[str]:
    """What the user should know about a usable run's result."""
    warnings = []
    differences = recorded_differences(run.run_dir, CLIP_SETTINGS)
    if differences:
        warnings.append("Clip settings were recorded differently, so this prediction may not match the run's "
                        'evaluation. ' + '; '.join(differences) + '. The log lists the values used.')
    args = run.cli_arguments or {}
    if args.get('mode') != 'reg':
        if not run.label_mapping:
            warnings.append('This run recorded no class names, so a prediction shows only a class number.')
        elif args.get('num_classes') and len(run.label_mapping) != int(args['num_classes']):
            warnings.append(f"This run recorded {len(run.label_mapping)} class names for {args['num_classes']} "
                            'classes, so a predicted name may be wrong.')
    return warnings


# ---------------------------------------------------------------------------
# Reading a prediction
# ---------------------------------------------------------------------------

@dataclass
class PredictionView:
    model: str
    video: str
    answer: str = ''
    confidence: str = ''
    expected_error: str = ''
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ''


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _present(value: object) -> bool:
    if value is None:
        return False
    return not (isinstance(value, float) and math.isnan(value))


def summarise_prediction(results: gc.InferenceResults, run: gc.RunResults, video: Path) -> PredictionView:
    """What one video's predictions.csv says, with the model that said it."""
    view = PredictionView(model=describe_run(run), video=Path(video).name)
    view.notes.extend(results.warnings)
    frame = results.predictions
    if frame is None or frame.empty:
        view.error = 'The run finished but wrote no prediction.'
        for warning in results.warnings:
            view.error += f' {warning}'
        return view
    row = frame.iloc[0]
    is_classifier = 'predicted_label' in frame.columns or 'predicted_class' in frame.columns
    if is_classifier:
        label = row.get('predicted_label')
        view.answer = f"Predicted class: {label if _present(label) else row.get('predicted_class')}"
        confidence = _finite(row.get('confidence'))
        view.confidence = f'Confidence: {confidence:.1%}' if confidence is not None else 'Confidence: not available'
        view.notes.append(CONFIDENCE_NOTE)
    elif 'predicted_value' in frame.columns:
        value = _finite(row.get('predicted_value'))
        if value is None:
            view.error = f"The run wrote an unusable predicted value: {row.get('predicted_value')!r}"
            return view
        view.answer = f'Predicted value: {value:.4g}'
        view.confidence = 'Confidence: not available for a regression model'
    else:
        view.error = 'predictions.csv has no predicted class or value. Columns: ' + ', '.join(map(str, frame.columns))
        return view

    estimate = _finite(row.get('expected_error')) if 'expected_error' in frame.columns else None
    if estimate is not None:
        stated = (f'{estimate:.1%} chance that this prediction is wrong' if is_classifier
                  else f'about {estimate:.4g} (expected absolute error, in target units)')
        view.expected_error = f"Expected error: {stated}. Estimated by the run's error model, not measured on this video."
    elif find_error_model(run.checkpoint_path) is not None:
        view.expected_error = ("Expected error: not available. This run's error model could not be used for this "
                               'prediction; the log gives the reason.')
    else:
        view.expected_error = 'Expected error: not available. This run has no error model (error_model.json).'
    return view


# What infer.py prints when it cannot honour a run's recorded options or error model.
_INFER_WARNING_STARTS = ('Ignoring error model', 'Ignoring unreadable error model', 'Warning: the error model',
                         'Ignoring the recorded')


def prediction_log_warnings(log: str) -> list[str]:
    """infer.py's warnings that change what a result means, so they reach the page and not only the log."""
    return [line.strip() for line in (log or '').splitlines() if line.strip().startswith(_INFER_WARNING_STARTS)]


_FAILURE_HINTS = (
    (r'out of memory', 'The GPU ran out of memory. Close other GPU jobs, including any training run, and try again.'),
    (r'no space left|not enough space|errno 28', 'The disk is full. Free some space and try again.'),
    (r"couldn't connect|connectionerror|connection error|max retries|localentrynotfound|offline mode|name resolution",
     'The model weights could not be downloaded or found. The first use needs a network connection.'),
    (r'size mismatch|missing key\(s\)|unexpected key\(s\)',
     "The checkpoint does not fit the model rebuilt from this run's recorded settings."),
)


def failure_hint(log: str) -> str:
    """A plain reason for the failures a user can do something about, or ''."""
    text = (log or '').lower()
    return next((hint for pattern, hint in _FAILURE_HINTS if re.search(pattern, text)), '')


# ---------------------------------------------------------------------------
# The track.py command line
# ---------------------------------------------------------------------------

def parse_prompts(text: object) -> list[str]:
    """Prompts typed with commas, semicolons or new lines between them."""
    parts = str(text or '').replace(';', ',').replace('\n', ',').split(',')
    try:
        return clean_prompts(parts)
    except ValueError:
        return []


def default_track_score(detector: str) -> float:
    """The score a box needs to start a track when track.py gets no threshold flags."""
    return round(TRACK_DEFAULTS[detector]['track_high_thresh'] + 0.1, 4)


@dataclass
class TrackConfig:
    video: Optional[Path] = None
    prompts: list[str] = field(default_factory=list)
    out: Optional[Path] = None
    detector: str = 'grounding_dino'
    every_n_frames: int = 1
    # Score a box needs to start a track. None = track.py's own defaults.
    track_score: Optional[float] = None


def threshold_flags(detector: str, score: Optional[float]) -> list[str]:
    """One sensitivity setting -> the track.py flags that make it take effect.

    A box only starts a track if it survives every gate below it, so lowering
    --new_track_thresh alone would change nothing: track.py keeps the first gates
    at their defaults. Those gates move down with the setting and are never raised.
    """
    if score is None or round(float(score), 4) == default_track_score(detector):
        return []
    score = round(float(score), 4)
    defaults = DETECTOR_DEFAULTS[detector]
    flags = ['--new_track_thresh', str(score)]
    if score < TRACK_DEFAULTS[detector]['track_high_thresh']:
        flags += ['--track_high_thresh', str(score)]
    if score < defaults['box_threshold']:
        flags += ['--box_threshold', str(score)]
    if defaults['text_threshold'] is not None and score < defaults['text_threshold']:
        flags += ['--text_threshold', str(score)]
    return flags


def build_track_argv(cfg: TrackConfig) -> list[str]:
    """Translate a TrackConfig into track.py's arguments. --save_video is always on."""
    if cfg.video is None or cfg.out is None or not cfg.prompts:
        raise ValueError('A video, at least one prompt and an output folder are all required.')
    argv = ['--video', str(cfg.video), '--out', str(cfg.out), '--detector', cfg.detector,
            '--every_n_frames', str(int(cfg.every_n_frames)), '--save_video']
    # --prompt=value, so a prompt that starts with '-' is not taken for a flag.
    argv += [f'--prompt={prompt}' for prompt in cfg.prompts]
    return argv + threshold_flags(cfg.detector, cfg.track_score)


def validate_track_config(cfg: TrackConfig) -> list[str]:
    """The checks that keep a bad setting from loading a detector first."""
    issues: list[str] = []
    check = check_video(cfg.video)
    if not check.ok:
        issues.append(check.error)
    if not cfg.prompts:
        issues.append('Type at least one thing to find, such as: person, car')
    elif cfg.detector == 'grounding_dino':
        try:
            check_grounding_dino_prompts(cfg.prompts)
        except ValueError as exc:
            issues.append(str(exc))
    if cfg.detector not in DEFAULT_MODELS:
        issues.append(f'Unknown detector: {cfg.detector}')
    if int(cfg.every_n_frames) < 1:
        issues.append('The frames step must be 1 or more.')
    if cfg.track_score is not None and not 0 < float(cfg.track_score) <= 1:
        issues.append('The score threshold must be above 0 and at most 1.')
    if cfg.out is None or not str(cfg.out).strip():
        issues.append('Choose an output folder.')
    return issues


def timestamped_folder(base: Path) -> Path:
    return Path(base) / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def new_output_folder(base: Path) -> Path:
    """Under the Run page's output folder. gc.list_runs never mistakes it for a run."""
    return timestamped_folder(Path(base) / 'analyse-video')


# ---------------------------------------------------------------------------
# Reading a tracking result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackedLabel:
    name: str
    count: int
    first_seen_s: Optional[float]
    last_seen_s: Optional[float]

    @property
    def row(self) -> tuple[str, str, str, str]:
        return self.name, str(self.count), format_seconds(self.first_seen_s), format_seconds(self.last_seen_s)


@dataclass
class TrackResults:
    video_dir: Path
    model: str = ''
    labels: list[TrackedLabel] = field(default_factory=list)
    num_frames: Optional[int] = None
    every_n_frames: Optional[int] = None
    overlay: Optional[Path] = None
    no_tracks_message: str = ''
    error: str = ''

    @property
    def total_tracks(self) -> int:
        return sum(label.count for label in self.labels)


def format_seconds(seconds: Optional[float]) -> str:
    return '-' if seconds is None else f'{seconds:.1f} s'


def _read_json_object(path: Path) -> tuple[Optional[dict], str]:
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return None, f'{path.name} is missing.'
    except (OSError, ValueError) as exc:
        return None, f'{path.name} could not be read: {exc}'
    if not isinstance(value, dict):
        return None, f'{path.name} is not a JSON object.'
    return value, ''


def _parse_label(name: str, entry: object) -> TrackedLabel:
    if not isinstance(entry, dict):
        raise ValueError(f'the entry for {name!r} is not an object')
    count = entry.get('count')
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f'the count for {name!r} is not a number of tracks: {count!r}')
    times = []
    for key in ('first_seen_s', 'last_seen_s'):
        value = entry.get(key)
        if value is not None and _finite(value) is None:
            raise ValueError(f'{key} for {name!r} is not a time: {value!r}')
        times.append(None if value is None else float(value))
    return TrackedLabel(str(name), count, times[0], times[1])


def _detector_used(out: Path, requested: Optional[str]) -> tuple[str, Optional[float]]:
    """(model that ran, score that starts a track), from track_arguments.json."""
    record, _ = _read_json_object(out / 'track_arguments.json')
    record = record or {}
    resolved = record.get('detector_resolved') if isinstance(record.get('detector_resolved'), dict) else {}
    model_id = record.get('detector_model') or resolved.get('model_id')
    if not model_id:
        model_id = detector_by_key(requested).model_id if requested in DEFAULT_MODELS else 'the chosen detector'
    text = f'{model_id} + ByteTrack'
    extras = [str(v) for v in (record.get('detector_licence'), resolved.get('device')) if v]
    if extras:
        text += f" ({', '.join(extras)})"
    return text, _finite(record.get('new_track_thresh'))


def _detected_boxes(video_dir: Path, size_limit: int = 64_000_000) -> tuple[Optional[int], Optional[float]]:
    """(boxes, best score) in detections.json, or (None, None) if unavailable or very large."""
    path = video_dir / 'detections.json'
    try:
        if path.stat().st_size > size_limit:
            return None, None
        frames = json.loads(path.read_text(encoding='utf-8-sig'))['frames']
        scores = [float(d['score']) for frame in frames for d in frame['detections']]
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    return len(scores), max(scores) if scores else None


def no_tracks_message(video_dir: Path, track_score: Optional[float]) -> str:
    """Why nothing was tracked, and what to change. An empty result is not a failure."""
    boxes, best = _detected_boxes(video_dir)
    advice = 'Try a lower score threshold or a different prompt.'
    if boxes:
        needed = f' Starting a track needs a score of at least {track_score:.2f}.' if track_score is not None else ''
        return (f'No tracks found. The detector returned {boxes} box(es), best score {best:.2f}, but none of them '
                f'became a track.{needed} {advice}')
    if boxes == 0:
        return f'No tracks found. The detector returned no boxes in the frames it looked at. {advice}'
    return f'No tracks found. {advice}'


def read_track_results(out: Path, video: Path, requested_detector: Optional[str] = None) -> TrackResults:
    """Read what one successful track.py run wrote for one video. Never raises on bad files."""
    out = Path(out)
    video_dir = out / Path(video).stem  # track.py: <out>/<video stem>/ for a single --video
    results = TrackResults(video_dir=video_dir)
    results.model, track_score = _detector_used(out, requested_detector)
    summary, problem = _read_json_object(video_dir / 'summary.json')
    if summary is None:
        results.error = f'Tracking finished but its result could not be used: {problem}'
        return results
    labels = summary.get('labels')
    if not isinstance(labels, dict):
        results.error = 'Tracking finished but its result could not be used: summary.json has no labels.'
        return results
    try:
        results.labels = [_parse_label(name, entry) for name, entry in labels.items()]
    except ValueError as exc:
        results.error = f'Tracking finished but its result could not be used: in summary.json, {exc}.'
        return results
    for name in ('num_frames', 'every_n_frames'):
        value = summary.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            setattr(results, name, value)
    overlay = video_dir / 'overlay.mp4'
    results.overlay = overlay if overlay.is_file() and overlay.stat().st_size > 0 else None
    if results.total_tracks == 0:
        results.no_tracks_message = no_tracks_message(video_dir, track_score)
    return results


def frames_note(results: TrackResults) -> str:
    if results.num_frames is None:
        return ''
    step = results.every_n_frames or 1
    looked_at = math.ceil(results.num_frames / step)
    return f'Looked at {looked_at} of {results.num_frames} frames (every {step}).'


def last_log_lines(text: str, count: int = 8) -> str:
    """The end of a log, for a run that failed."""
    lines = [line.rstrip() for line in (text or '').splitlines() if line.strip()]
    return '\n'.join(lines[-count:])


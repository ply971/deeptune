"""Declarative GUI controls for command-line flags (G1, the GUI catch-up).

Each `Opt` describes one flag of an entry point (deeptune.py, infer.py, select_model.py, ensemble.py,
extract_features.py, train_feature_head.py, track.py): its widget kind, label, help, section and the CLI default.
The same spec drives three things, so they cannot drift apart:

* the widgets (desktop/option_widgets.py renders a spec list into collapsible sections),
* the values a page holds ({key: value}, saved with the settings),
* the argv (`options_argv`): a flag is emitted only when its value differs from the CLI default, so the common path
  sends exactly what it sent before G1, and a default that the CLI refuses for some models is never sent.

tests/test_gui_coverage.py checks every flag of every entry point against these specs, through the real parsers.

Pure Python: no torch and no Dear PyGui, like desktop/config.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional

BOOL, INT, FLOAT, TEXT, CHOICE, MULTI, WORDS, LINES = 'bool', 'int', 'float', 'text', 'choice', 'multi', 'words', 'lines'
# A store_true flag as an override of a recorded value: None keeps the record, True sends the flag, False removes it.
TRI = 'tristate'
LIST_KINDS = (MULTI, WORDS, LINES)
DEFAULT_LABEL = '(default)'  # the combo item of a choice whose CLI default is "not given"
TRI_ITEMS = ('(recorded)', 'on', 'off')


@dataclass(frozen=True)
class Opt:
    flag: str  # the CLI spelling, e.g. '--video_augment'
    kind: str
    label: str
    help: str = ''
    default: Any = None  # the CLI default; the widget starts here and nothing is emitted at it
    choices: tuple = ()
    section: str = ''
    modalities: tuple = ()  # deeptune.py only: modalities where the flag applies (empty: all)
    required: bool = False  # always emitted
    remainder: bool = False  # argparse.REMAINDER: emitted last, words as given
    hidden: bool = False  # no widget of its own: a button sets it (e.g. select_model.py --dry_run)
    browse: str = ''  # 'file' or 'folder': the widget gets a Browse button

    @property
    def key(self) -> str:
        return self.flag.lstrip('-').replace('-', '_')

    @property
    def empty(self):
        """The value of an untouched widget."""
        if self.kind in LIST_KINDS:
            return list(self.default or [])
        if self.kind == BOOL:
            return bool(self.default)
        return self.default


def as_override(opt: Opt, section: str) -> Opt:
    """The same flag as a "blank = keep what was recorded" control: nothing is emitted until the user sets it. A
    store_true flag becomes a three-way choice, so an override can also turn a recorded flag off."""
    kind = {INT: NUMBER_TEXT['int'], FLOAT: NUMBER_TEXT['float'], BOOL: TRI}.get(opt.kind, opt.kind)
    return replace(opt, kind=kind, default=[] if opt.kind in LIST_KINDS else None, section=section, required=False)


# Number controls whose default is "not given" are text boxes (blank = not given); these kinds parse them.
OPT_INT, OPT_FLOAT = 'opt_int', 'opt_float'
NUMBER_TEXT = {'int': OPT_INT, 'float': OPT_FLOAT}


def is_default(opt: Opt, value) -> bool:
    if opt.kind in LIST_KINDS:
        return list(value or []) == list(opt.default or [])
    if opt.kind == BOOL:
        return bool(value) == bool(opt.default)
    if opt.kind == TRI:
        return value is None
    if value is None or value == '':
        return opt.default is None
    if isinstance(value, float) and isinstance(opt.default, (int, float)) and not isinstance(opt.default, bool):
        return math.isclose(value, float(opt.default), rel_tol=1e-12, abs_tol=0.0)
    return value == opt.default


def format_value(value) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def option_argv(opt: Opt, value) -> list[str]:
    """The flag and its value(s), or [] at the CLI default."""
    if is_default(opt, value) and not opt.required:
        return []
    if opt.kind in (BOOL, TRI):
        return [opt.flag] if value else []  # TRI False removes the flag: see removed_flags
    if opt.kind in LIST_KINDS:
        words = [format_value(v) for v in value or []]
        return [opt.flag, *words] if words or opt.remainder else []
    text = format_value(value)
    # A value that starts with '-' would be read as a flag: --flag=value keeps it a value.
    return [f'{opt.flag}={text}'] if text.startswith('-') else [opt.flag, text]


def options_argv(specs: Iterable[Opt], values: dict, applies=None) -> list[str]:
    """argv for every spec whose value differs from the CLI default. `applies(opt)` can drop flags that do not apply
    to the current settings (e.g. video flags on an image run). Remainder flags come last."""
    argv, tail = [], []
    for opt in specs:
        if applies is not None and not applies(opt):
            continue
        value = values.get(opt.key, opt.empty)
        (tail if opt.remainder else argv).extend(option_argv(opt, value))
    return argv + tail


def removed_flags(specs: Iterable[Opt], values: dict) -> list[str]:
    """Flags that three-way overrides set to off: the recorded argv must drop them."""
    return [opt.flag for opt in specs if opt.kind == TRI and values.get(opt.key) is False]


def parse_text(opt: Opt, raw) -> tuple[Any, Optional[str]]:
    """A widget's raw value -> (value, problem)."""
    if opt.kind == BOOL:
        return bool(raw), None
    if opt.kind == TRI:
        if raw in (None, '', TRI_ITEMS[0]):
            return None, None
        if raw in ('on', 'off'):
            return raw == 'on', None
        if isinstance(raw, bool):
            return raw, None
        return None, f'{opt.label}: choose {", ".join(TRI_ITEMS)}.'
    if opt.kind == MULTI:
        return [c for c in opt.choices if c in (raw or [])], None
    if opt.kind in (WORDS, LINES):
        if isinstance(raw, (list, tuple)):
            return [str(v) for v in raw if str(v).strip()], None
        text = str(raw or '')
        parts = text.splitlines() if opt.kind == LINES else text.replace(',', ' ').split()
        return [p.strip().strip('"') for p in parts if p.strip()], None
    if opt.kind == CHOICE:
        if raw in (None, '', DEFAULT_LABEL):
            return opt.default, None
        for choice in opt.choices:
            if str(choice) == str(raw):
                return choice, None
        return opt.default, f'{opt.label}: choose one of {", ".join(map(str, opt.choices))}.'
    if opt.kind in (INT, FLOAT):
        if isinstance(raw, bool) or raw is None or raw == '':
            return opt.default, None
        try:
            value = int(raw) if opt.kind == INT else float(raw)
        except (TypeError, ValueError):
            return opt.default, f'{opt.label} must be a number.'
        return value, None
    if opt.kind in (OPT_INT, OPT_FLOAT):
        text = str(raw if raw is not None else '').strip()
        if not text:
            return None, None
        try:
            value = int(text) if opt.kind == OPT_INT else float(text)
        except ValueError:
            return None, f'{opt.label} must be a {"whole number" if opt.kind == OPT_INT else "number"}, or blank.'
        if opt.kind == OPT_FLOAT and not math.isfinite(value):
            return None, f'{opt.label} must be a finite number.'
        return value, None
    text = str(raw if raw is not None else '').strip().strip('"')
    if not text:
        return (opt.default if opt.default not in (None, '') else None), None
    return text, None


def coerce_values(specs: Iterable[Opt], values: dict) -> tuple[dict, list[str]]:
    """Check stored values (e.g. loaded from a settings file) against their specs: (clean values, problems)."""
    by_key = {opt.key: opt for opt in specs}
    clean, problems = {}, []
    for key, raw in (values or {}).items():
        opt = by_key.get(key)
        if opt is None:
            problems.append(f'Unknown setting {key!r}.')
            continue
        if opt.kind == CHOICE and raw is not None and raw not in opt.choices:
            problems.append(f'{opt.label}: {raw!r} is not one of the choices.')
            continue
        if opt.kind in LIST_KINDS and not (isinstance(raw, list) and all(isinstance(v, (str, int, float)) for v in raw)):
            problems.append(f'{opt.label}: expected a list.')
            continue
        if opt.kind == BOOL and not isinstance(raw, bool):
            problems.append(f'{opt.label}: expected true or false.')
            continue
        if opt.kind == TRI and raw is not None and not isinstance(raw, bool):
            problems.append(f'{opt.label}: expected true, false or null.')
            continue
        if opt.kind in (INT, OPT_INT) and raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
            problems.append(f'{opt.label}: expected a whole number.')
            continue
        if opt.kind in (FLOAT, OPT_FLOAT) and raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
            problems.append(f'{opt.label}: expected a number.')
            continue
        if opt.kind == TEXT and raw is not None and not isinstance(raw, str):
            problems.append(f'{opt.label}: expected text.')
            continue
        clean[key] = float(raw) if opt.kind in (FLOAT, OPT_FLOAT) and raw is not None else raw
    return clean, problems


def non_default(specs: Iterable[Opt], values: dict) -> list[Opt]:
    return [opt for opt in specs if not is_default(opt, values.get(opt.key, opt.empty))]


def by_key(specs: Iterable[Opt]) -> dict:
    return {opt.key: opt for opt in specs}


def merge_flags(argv: list[str], extra: list[str]) -> list[str]:
    """`argv` with every flag of `extra` replaced by extra's value(s), so each flag appears once. Values never start with
    '--' (option_argv writes such values as --flag=value)."""
    def groups(words):
        out = []
        for word in words:
            if word.startswith('--'):
                out.append([word])
            elif out:
                out[-1].append(word)
            else:
                out.append([word])
        return out
    names = {group[0].split('=', 1)[0] for group in groups(extra)}
    kept = [group for group in groups(argv) if group[0].split('=', 1)[0] not in names]
    return [word for group in kept + groups(extra) for word in group]


# ---------------------------------------------------------------------------------------------------------------------
# deeptune.py (Model & Training page). The flags the page has always had are RunConfig fields (desktop/config.py);
# everything below lives in RunConfig.options.
# ---------------------------------------------------------------------------------------------------------------------
VIDEO, VISION = ('video',), ('images', 'video')
HUB_LICENCES = ('CC-BY-NC-4.0', 'DINOv3-License')  # handlers/video_backbones.py: the opt-in licences, never pre-ticked
VIDEO_ARCH_CHOICES = ('frame_pool', 'single_frame', 'slow_fusion', 'late_fusion', 'early_fusion', 'tsm', 'cnn_rnn',
                      'temporal_transformer', 'two_stream', 'attentive_probe', 'linear_probe')

S_ARCH, S_BACKBONE, S_DATA, S_AUG = 'Temporal architecture', 'Pretrained backbone and LoRA', 'Video data and decoding', 'Augmentation'
S_RECIPE, S_EVAL, S_ABSTAIN, S_DFA = 'Training recipe', 'Evaluation and validation', 'Not sure (abstention)', 'df-analyze export'

ARCH_OPTIONS = (
    Opt('--video_arch', CHOICE, 'Temporal architecture', 'How the model fuses time, with the model as backbone. (default): '
        'frame pooling as chosen above. attentive_probe and linear_probe are for V-JEPA 2 and VideoMAE.',
        None, VIDEO_ARCH_CHOICES, S_ARCH, VIDEO),
    Opt('--tsm_fold_div', OPT_INT, 'TSM fold divisor', 'tsm: shift 1/N of the channels each way (default 8).', None, (), S_ARCH, VIDEO),
    Opt('--rnn_type', CHOICE, 'RNN cell', 'cnn_rnn: recurrent cell (default lstm).', None, ('lstm', 'gru'), S_ARCH, VIDEO),
    Opt('--rnn_hidden', OPT_INT, 'RNN hidden size', 'cnn_rnn (default 256).', None, (), S_ARCH, VIDEO),
    Opt('--rnn_layers', OPT_INT, 'RNN layers', 'cnn_rnn (default 1).', None, (), S_ARCH, VIDEO),
    Opt('--bidirectional', BOOL, 'Bidirectional RNN', 'cnn_rnn: read the clip in both directions.', False, (), S_ARCH, VIDEO),
    Opt('--tt_layers', OPT_INT, 'Transformer layers', 'temporal_transformer (default 2).', None, (), S_ARCH, VIDEO),
    Opt('--tt_heads', OPT_INT, 'Transformer heads', 'temporal_transformer (default 4).', None, (), S_ARCH, VIDEO),
    Opt('--tt_dim', OPT_INT, 'Transformer width', 'temporal_transformer: a multiple of the heads (default 256).', None, (), S_ARCH, VIDEO),
    Opt('--fusion', CHOICE, 'Two-stream fusion', 'two_stream: average logits (avg) or an FC head (fc); default avg.',
        None, ('avg', 'fc'), S_ARCH, VIDEO),
    Opt('--motion_length', OPT_INT, 'Motion fields per clip', 'two_stream (default: frames per clip).', None, (), S_ARCH, VIDEO),
    Opt('--motion_cache', TEXT, 'Motion cache folder', 'two_stream: keep motion fields in this folder (default off).',
        None, (), S_ARCH, VIDEO, browse='folder'),
    Opt('--probe_width', OPT_INT, 'Probe width', 'attentive_probe: a multiple of the heads (default 256).', None, (), S_ARCH, VIDEO),
    Opt('--probe_depth', OPT_INT, 'Probe depth', 'attentive_probe: cross-attention blocks, 1 to 4 (default 1).', None, (), S_ARCH, VIDEO),
    Opt('--probe_heads', OPT_INT, 'Probe heads', 'attentive_probe (default 4).', None, (), S_ARCH, VIDEO),
)
VIDEO_DATA_OPTIONS = (
    Opt('--video_backend', CHOICE, 'Decoder', 'auto = torchcodec if installed, else PyAV, else OpenCV.', 'auto',
        ('auto', 'torchcodec', 'pyav', 'opencv'), S_DATA, VIDEO),
    Opt('--temporal_sampling', CHOICE, 'Frame sampling', 'uniform: evenly spaced. segment: TSN. dense: consecutive frames '
        'a clip stride apart. motion: more frames where things move.', 'uniform', ('uniform', 'segment', 'dense', 'motion'), S_DATA, VIDEO),
    Opt('--clip_stride', OPT_INT, 'Clip stride', 'dense sampling: frame step inside a clip (default 1).', None, (), S_DATA, VIDEO),
    Opt('--clip_duration_sec', OPT_FLOAT, 'Clip window (s)', 'Sample from a window this many seconds long (default: whole clip).',
        None, (), S_DATA, VIDEO),
    Opt('--motion_source', CHOICE, 'Motion source', 'Motion curve for motion sampling, or the motion stream of two_stream.',
        'auto', ('auto', 'codec', 'framediff', 'farneback', 'mv'), S_DATA, VIDEO),
    Opt('--test_clips', INT, 'Test clips per video', 'Clips per video at test time; predictions average them (default 1).',
        1, (), S_EVAL, VIDEO),
    Opt('--test_crops', CHOICE, 'Test crops per clip', '1, 3 (along the long side) or 5 (corners and centre); multiplies '
        'with test clips, and so does the run time.', 1, (1, 3, 5), S_EVAL, VIDEO),
    Opt('--num_workers', TEXT, 'DataLoader workers', "auto (default) or a whole number. Each worker re-imports torch on "
        "Windows (15-40 s).", 'auto', (), S_DATA, VIDEO),
    Opt('--frame_cache', TEXT, 'Frame cache folder', 'Keep sampled, resized frames of deterministic clips here (default off).',
        None, (), S_DATA, VIDEO, browse='folder'),
    Opt('--frame_cache_size', INT, 'Frame cache short side', 'With a frame cache: shrink frames to this short side (default 256; '
        '0 keeps the decoded size).', 256, (), S_DATA, VIDEO),
    Opt('--frame_resize', OPT_INT, 'Pre-model resize', 'Short-side resize before the model, independent of the cache; 0 disables '
        '(default: the cache size when caching, else 0).', None, (), S_DATA, VIDEO),
)
DEEPTUNE_OPTIONS = (
    *ARCH_OPTIONS,
    Opt('--unfreeze_backbone', BOOL, 'Train the whole encoder (V-JEPA 2, VideoMAE)', 'They are frozen by default. Needs a lot of '
        'GPU memory; see the README.', False, (), S_BACKBONE, VIDEO),
    Opt('--accept_weights_licence', MULTI, 'I accept the licence of these weights', 'Opt-in weights: CC-BY-NC-4.0 (VideoMAE, '
        'non-commercial only) and DINOv3-License (gated by Meta). Tick only after reading the licence. Recorded with the run.',
        (), HUB_LICENCES, S_BACKBONE, VIDEO),
    Opt('--lora_rank', OPT_INT, 'LoRA rank', 'With PEFT / LoRA on native video models (default 16). Other models use the '
        'embedding size as rank.', None, (), S_BACKBONE, VISION),
    *VIDEO_DATA_OPTIONS,
    Opt('--video_storage', CHOICE, 'Clip storage', 'With raw data: keep file paths (default) or embed clips as bytes.',
        'paths', ('paths', 'bytes'), S_DATA, VIDEO),
    Opt('--video_augment', BOOL, 'Augment training clips', 'Temporal jitter, one random-resized crop and colour jitter per clip.',
        False, (), S_AUG, VIDEO),
    Opt('--video_flip', BOOL, 'Also flip horizontally', 'With augmentation. A mirror turns left motion into right motion, so do '
        'not use it when labels depend on direction.', False, (), S_AUG, VIDEO),
    Opt('--video_speed_perturb', FLOAT, 'Speed perturbation', 'With augmentation: playback speed in [1-x, 1+x], x in [0, 1) '
        '(default 0 = off).', 0.0, (), S_AUG, VIDEO),
    Opt('--mixup_alpha', FLOAT, 'Mixup alpha', 'Classification only (default 0 = off).', 0.0, (), S_AUG, VISION),
    Opt('--cutmix_alpha', FLOAT, 'CutMix alpha', 'Classification only (default 0 = off). With mixup too, each batch picks one.',
        0.0, (), S_AUG, VISION),
    Opt('--mix_prob', FLOAT, 'Mix probability', 'Share of batches that get mixup or CutMix (default 1).', 1.0, (), S_AUG, VISION),
    Opt('--label_smoothing', FLOAT, 'Label smoothing', 'Classification only (default 0).', 0.0, (), S_AUG, VISION),
    Opt('--no_error_model', BOOL, 'Skip the per-prediction error model', 'Without it, outputs have no expected_error column.',
        False, (), S_EVAL, VISION),
    Opt('--precision', CHOICE, 'Mixed precision', 'bf16 or fp16 on CUDA (default no). Does nothing on CPU.', 'no',
        ('no', 'bf16', 'fp16'), S_RECIPE, VISION),
    Opt('--seed', OPT_INT, 'Seed', 'Seeds everything, and in deeptune.py also the data split. Blank: 42, because the GUI ticks '
        '"Use a reproducible seed (42)" (--fixed-seed) by default for reproducibility; untick it there for a random, recorded '
        'seed. The command line has no fixed seed unless asked.', None, (), S_RECIPE, VISION),
    Opt('--checkpoint_every', INT, 'Checkpoint every N epochs', 'Write last_checkpoint.pt (default 0 = off).', 0, (), S_RECIPE, VISION),
    Opt('--resume', TEXT, 'Resume from checkpoint', 'A last_checkpoint.pt; use the same settings as the first run.', None, (),
        S_RECIPE, VISION, browse='file'),
    Opt('--resume_allow_data_change', BOOL, 'Resume even if the data changed', 'A warm start, not a resume.', False, (), S_RECIPE, VISION),
    Opt('--select_best', BOOL, 'Keep the best epoch', 'Reload the weights with the best validation score after training.',
        False, (), S_RECIPE, VISION),
    Opt('--early_stopping_patience', INT, 'Early stopping patience', 'Stop after this many epochs without improvement '
        '(default 0 = off).', 0, (), S_RECIPE, VISION),
    Opt('--early_stopping_min_delta', FLOAT, 'Early stopping min delta', 'Smallest change that counts (default 0).', 0.0, (),
        S_RECIPE, VISION),
    Opt('--selection_metric', CHOICE, 'Selection metric', 'What best-epoch selection compares (default val_loss).', 'val_loss',
        ('val_loss', 'val_accuracy'), S_RECIPE, VISION),
    Opt('--lr_scheduler', CHOICE, 'LR schedule', 'cosine to the minimum LR, or step decay (default none).', 'none',
        ('none', 'cosine', 'step'), S_RECIPE, VISION),
    Opt('--warmup_epochs', FLOAT, 'Warmup epochs', 'Linear warmup; fractions allowed (default 0).', 0.0, (), S_RECIPE, VISION),
    Opt('--lr_step_size', OPT_INT, 'Step size (epochs)', 'Step schedule (default: a third of the epochs).', None, (), S_RECIPE, VISION),
    Opt('--lr_gamma', FLOAT, 'Step gamma', 'Step schedule decay factor (default 0.1).', 0.1, (), S_RECIPE, VISION),
    Opt('--min_lr', FLOAT, 'Minimum LR', 'Final LR of the cosine schedule (default 0).', 0.0, (), S_RECIPE, VISION),
    Opt('--grad_accum_steps', INT, 'Gradient accumulation', 'Update every N batches (default 1).', 1, (), S_RECIPE, VISION),
    Opt('--freeze_bn', CHOICE, 'Freeze BatchNorm', 'off (default), on, or auto (on under batch size 16 with pretrained BatchNorm).',
        'off', ('off', 'on', 'auto'), S_RECIPE, VISION),
    Opt('--layer_decay', FLOAT, 'Layer-wise LR decay', 'Each block further from the output gets lr * this (default 1 = off).',
        1.0, (), S_RECIPE, VISION),
    Opt('--ema_decay', FLOAT, 'Weight EMA decay', 'e.g. 0.999 (default 0 = off).', 0.0, (), S_RECIPE, VISION),
    Opt('--experiment_logger', CHOICE, 'Experiment logger', 'Also log to MLflow or W&B if installed (default none).', 'none',
        ('none', 'mlflow', 'wandb'), S_RECIPE, VISION),
    Opt('--experiment_name', TEXT, 'Experiment name', 'Run name for the experiment logger.', None, (), S_RECIPE, VISION),
    Opt('--validation', CHOICE, 'Validation scheme', 'holdout: only the test score. kfold: k fresh models. bootstrap632: '
        "Efron's .632+. Training time is multiplied by the number of models.", 'holdout', ('holdout', 'kfold', 'bootstrap632'),
        S_EVAL, VIDEO),
    Opt('--k', INT, 'Folds (k)', 'For kfold (default 5, at least 2).', 5, (), S_EVAL, VIDEO),
    Opt('--n_boot', INT, 'Bootstrap models', 'For bootstrap632 (default 50).', 50, (), S_EVAL, VIDEO),
    Opt('--keep_fold_models', BOOL, 'Keep every fold model', 'By default only scores and metrics stay.', False, (), S_EVAL, VIDEO),
    Opt('--cv_only', BOOL, 'Cross-validate only (no test score)', 'Run the scheme on every row and stop; nothing here is a '
        'test score.', False, (), S_EVAL, VIDEO),
    Opt('--test_df', TEXT, 'Test split file (.parquet)', 'Use this file as the test split instead of carving 20%.', None, (),
        S_EVAL, VIDEO, browse='file'),
    Opt('--report_boot', INT, 'Report bootstrap resamples', 'For the intervals of evaluation_report.json (default 2000; 0: no '
        'report).', 2000, (), S_EVAL, VIDEO),
    Opt('--target_error', OPT_FLOAT, 'Target error rate', 'Say "not sure" rather than exceed this error among answered clips, '
        'e.g. 0.05. Blank = off. Takes the calibration share out of training.', None, (), S_ABSTAIN, VIDEO),
    Opt('--abstain_delta', FLOAT, 'Delta', 'Chance the calibration draw is unlucky (default 0.05).', 0.05, (), S_ABSTAIN, VIDEO),
    Opt('--abstain_guarantee', CHOICE, 'Guarantee', 'selective_risk (default): error among answered clips. silent_error: the chance '
        'a new group has any answered wrong clip.', 'selective_risk', ('selective_risk', 'silent_error'), S_ABSTAIN, VIDEO),
    Opt('--abstain_score', CHOICE, 'Ranking score', 'max_prob (default) or expected_error (needs the error model).', 'max_prob',
        ('max_prob', 'expected_error'), S_ABSTAIN, VIDEO),
    Opt('--calib_fraction', FLOAT, 'Calibration share', 'Share of the data held out for calibration, whole groups (default 0.1).',
        0.1, (), S_ABSTAIN, VIDEO),
    Opt('--export_dfanalyze', BOOL, 'Export embeddings for df-analyze', 'CSV files in dfanalyze_export_*/ with the command to run.',
        False, (), S_DFA, VIDEO),
    Opt('--dfanalyze_extras', MULTI, 'Extras', 'tta: mean/std over several clips. motion: motion statistics. ids: split and '
        'video_id columns (dropped by the printed command).', (), ('tta', 'motion', 'ids'), S_DFA, VIDEO),
    Opt('--dfanalyze_clips', INT, 'Clips for tta', 'At least 2 (default 5).', 5, (), S_DFA, VIDEO),
    Opt('--dfanalyze_dims', OPT_INT, 'PCA components', 'Blank keeps every dimension.', None, (), S_DFA, VIDEO),
    Opt('--dfanalyze_rows', CHOICE, 'Rows of dfanalyze.csv', 'Default: test for a trained model.', None, ('test', 'all'), S_DFA, VIDEO),
)
DEEPTUNE_SECTIONS = (S_ARCH, S_BACKBONE, S_DATA, S_AUG, S_RECIPE, S_EVAL, S_ABSTAIN, S_DFA)


def deeptune_applies(modality: str):
    return lambda opt: not opt.modalities or modality in opt.modalities


# ---------------------------------------------------------------------------------------------------------------------
# infer.py (Test page). The run's record supplies the model settings (desktop/config.infer_config_from_run); these
# controls run the test, and the "override" ones replace a recorded value only when the user fills them in.
# ---------------------------------------------------------------------------------------------------------------------
S_TEST = 'Test options'
# Two kinds of override, blank = what the run recorded. Test-time ones change how clips are read for this test. Model-
# defining ones change the model rebuilt from the trained weights, or its labels: the weights may still load (a pooling or
# a TSM fold is not in the state dict) and give confident, wrong predictions, so they need an explicit confirmation.
S_OVERRIDE = 'Advanced: test-time settings (blank = recorded)'
S_MODEL_OVERRIDE = 'Advanced: change the rebuilt model or its labels (blank = recorded)'
OVERRIDE_SECTIONS = (S_OVERRIDE, S_MODEL_OVERRIDE)
_VIDEO_DATA = {opt.key: replace(opt, modalities=()) for opt in VIDEO_DATA_OPTIONS}
INFER_OPTIONS = (
    Opt('--no_abstain', BOOL, 'Answer every clip (ignore "not sure")', "Ignore the run's abstention.json and give every "
        "clip's prediction.", False, (), S_TEST),
    Opt('--accept_weights_licence', MULTI, 'I accept the licence of these weights', 'Only for opt-in weights whose run did not '
        'record the acceptance.', (), HUB_LICENCES, S_TEST),
    Opt('--ensemble', LINES, 'Ensemble of exactly these runs', 'Video only: average the class probabilities of exactly the runs '
        'listed here, one trained run (folder or weights file) per line, at least two. The run selected above is a member '
        'only if it is listed. Each member uses its own recorded settings.', (), (), S_TEST, browse='folder'),
    Opt('--allow_unverified_labels', BOOL, 'Allow unverified class indices', 'With an ensemble: average runs even if their '
        'class indices cannot be verified. The result says so.', False, (), S_TEST),
    *(as_override(opt, S_OVERRIDE) for opt in (
        Opt('--test_clips', INT, 'Test clips per video', 'More clips per video, averaged.', 1),
        Opt('--test_crops', CHOICE, 'Test crops per clip', 'More spatial views per clip, averaged.', 1, (1, 3, 5)),
        *(_VIDEO_DATA[key] for key in ('video_backend', 'temporal_sampling', 'clip_stride', 'clip_duration_sec', 'num_workers',
                                       'frame_cache')),
        replace(ARCH_OPTIONS[[o.key for o in ARCH_OPTIONS].index('motion_cache')], modalities=()),
    )),
    *(as_override(opt, S_MODEL_OVERRIDE) for opt in (
        Opt('--label_mapping', TEXT, 'Label mapping file', "Class names for the indices; the run's own "
            "data_splits_*/label_mapping.json by default. A wrong file renames every prediction.", browse='file'),
        Opt('--model_version', TEXT, 'Model version', ''),
        Opt('--model_str', CHOICE, 'Text model family', '', None, ('BERT', 'gpt2')),
        Opt('--num_classes', INT, 'Number of classes', '', 2),
        Opt('--mode', CHOICE, 'Task', '', 'cls', ('cls', 'reg')),
        Opt('--added_layers', INT, 'Added layers', '', 2),
        Opt('--embed_size', INT, 'Embedding size', '', 1000),
        Opt('--lora_rank', INT, 'LoRA rank', ''),
        Opt('--freeze-backbone', BOOL, 'Freeze backbone', ''),
        Opt('--use-peft', BOOL, 'PEFT / LoRA model', ''),
        Opt('--num_frames', INT, 'Frames per clip', 'The clip length the model was built for.', 8),
        Opt('--pooling', TEXT, 'Pooling or architecture spec', 'e.g. mean, attention or tsm:tsm_fold_div=8.', 'mean'),
        *(replace(opt, modalities=()) for opt in ARCH_OPTIONS if opt.key != 'motion_cache'),
        Opt('--unfreeze_backbone', BOOL, 'Whole encoder trained', ''),
        *(_VIDEO_DATA[key] for key in ('motion_source', 'frame_cache_size', 'frame_resize')),
        Opt('--target', TEXT, 'Target column', '', 'labels'),
        Opt('--finetuning-mode', BOOL, 'TabPFN fine-tuned', ''),
        Opt('--time_idx_column', TEXT, 'Time index column', '', 'labels'),
        Opt('--max_encoder_length', INT, 'Max encoder length', '', 30),
        Opt('--group_ids', WORDS, 'Group id columns', ''),
        Opt('--history_df', LINES, 'History files (DeepAR)', 'One parquet per line.', browse='file'),
    )),
)
# With --ensemble, infer.py refuses every other flag (infer.ENSEMBLE_FLAGS).
INFER_ENSEMBLE_KEYS = ('accept_weights_licence', 'allow_unverified_labels', 'ensemble')
MODEL_OVERRIDE_KEYS = tuple(opt.key for opt in INFER_OPTIONS if opt.section == S_MODEL_OVERRIDE)


# ---------------------------------------------------------------------------------------------------------------------
# select_model.py (Model selection page)
# ---------------------------------------------------------------------------------------------------------------------
STRATEGIES = ('bayes', 'exhaustive', 'random', 'smart')  # handlers/model_selection/search.STRATEGIES, sorted as the CLI lists them
AUGMENTATIONS = ('off', 'aug', 'aug_mix', 'aug_flip', 'aug_mix_flip')  # handlers/model_selection/grid.AUGMENTATIONS
SELECTION_METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auc', 'mae', 'rmse', 'r2')
G_DATA, G_GRID, G_TRAIN, G_SMART, G_SAMPLED, G_OVERRIDES, G_LIMITS = (
    'Data', 'Grid', 'Training and scoring', 'Smart search', 'Random and bayes search', 'Overrides', 'Limits, resume and checks')
SELECT_OPTIONS = (
    Opt('--search', CHOICE, 'Strategy', 'smart (default): staged screening and tuning. exhaustive: every candidate. random: seeded '
        'draws. bayes: Optuna TPE (needs optuna). Only the final winner sees the sealed test set.', 'smart', STRATEGIES, G_DATA),
    Opt('--df', TEXT, 'Video dataset', 'A .parquet file as deeptune.py takes it, or a clip folder / CSV list with raw data.',
        None, (), G_DATA, required=True, browse='file'),
    Opt('--raw-data', BOOL, 'Raw data (clip folder or CSV list)', 'Converted to Parquet once, into <out>/data/.', False, (), G_DATA),
    Opt('--out', TEXT, 'Search folder', 'Rerunning with the same folder resumes: finished candidates are not trained again.',
        None, (), G_DATA, required=True, browse='folder'),
    Opt('--mode', CHOICE, 'Task', 'Classification or regression (default cls).', 'cls', ('cls', 'reg'), G_DATA),
    Opt('--num_classes', OPT_INT, 'Number of classes', 'Default: counted from the data.', None, (), G_DATA),
    Opt('--target', TEXT, 'Target column', "Default 'labels'.", 'labels', (), G_DATA),
    Opt('--grouper', TEXT, 'Group column', "Clips of one group stay together (default: 'group' if present).", None, (), G_DATA),
    Opt('--test_fraction', FLOAT, 'Sealed test share', 'Share of the groups set aside before anything trains (default 0.2, '
        'at most 0.5).', 0.2, (), G_DATA),
    Opt('--grid', TEXT, 'Grid file (JSON)', 'Axes: backbones, video_archs, poolings, train_modes, learning_rates, num_frames. '
        'A field below replaces the same axis.', None, (), G_GRID, browse='file'),
    Opt('--backbones', WORDS, 'Backbones', 'e.g. resnet18 r3d_18 vjepa2_vitl', (), (), G_GRID),
    Opt('--video_archs', WORDS, 'Temporal architectures', 'e.g. default tsm cnn_rnn', (), (), G_GRID),
    Opt('--poolings', WORDS, 'Poolings', 'e.g. default attention', (), (), G_GRID),
    Opt('--train_modes', WORDS, 'Train modes', 'frozen (default), full, lora:R or partial:COUNT', (), (), G_GRID),
    Opt('--learning_rates', WORDS, 'Learning rates', 'e.g. 1e-4 3e-4 (default 1e-4)', (), (), G_GRID),
    Opt('--num_frames', WORDS, 'Frames per clip', 'e.g. 8 16 (default 8)', (), (), G_GRID),
    Opt('--augmentations', MULTI, 'Augmentations', 'Searched like any axis. Default: off, aug, aug_mix for smart (off, aug for '
        'regression); off for the others. Flips need the next box.', (), AUGMENTATIONS, G_GRID),
    Opt('--allow_flip', BOOL, 'My labels do not depend on direction (allow flips)', 'Needed for aug_flip and aug_mix_flip.',
        False, (), G_GRID),
    Opt('--k', INT, 'Folds (k)', 'Cross-validation folds per candidate (default 5, at least 2).', 5, (), G_TRAIN),
    Opt('--metric', CHOICE, 'Ranking metric', 'Classification: accuracy (default), balanced_accuracy, macro_f1, weighted_f1, '
        'macro_auc. Regression: mae (default), rmse, r2.', None, SELECTION_METRICS, G_TRAIN),
    Opt('--epochs', INT, 'Epochs per training', 'Default 10.', 10, (), G_TRAIN),
    Opt('--early_stopping_patience', INT, 'Early stopping patience', 'Default 3; 0 turns it off.', 3, (), G_TRAIN),
    Opt('--batch_size', INT, 'Batch size', 'Default 16.', 16, (), G_TRAIN),
    Opt('--num_workers', INT, 'DataLoader workers', 'Default 0 (each worker re-imports torch on Windows).', 0, (), G_TRAIN),
    Opt('--seed', INT, 'Seed', 'Seed of the sealed split, the folds and training (default 42).', 42, (), G_TRAIN),
    Opt('--boot', INT, 'Bootstrap resamples', 'For every interval (default 2000).', 2000, (), G_TRAIN),
    Opt('--accept_weights_licence', MULTI, 'I accept the licence of these weights', 'Opt-in weights, e.g. CC-BY-NC-4.0 for '
        'VideoMAE. Recorded.', (), HUB_LICENCES, G_TRAIN),
    Opt('--deeptune_args', LINES, 'More deeptune.py flags (one word per line)', 'e.g. --precision on one line and bf16 on the '
        'next. Flags select_model.py sets itself are refused.', (), (), G_TRAIN, remainder=True),
    Opt('--screen', CHOICE, 'Screening', 'probe: cached frozen features + attentive probe. cv: short cross-validation. '
        'auto (default).', None, ('auto', 'probe', 'cv'), G_SMART),
    Opt('--screen_views', CHOICE, 'Screening views', 'auto (default): frozen and trainable views when the grid has both.',
        None, ('auto', 'frozen', 'trainable'), G_SMART),
    Opt('--screen_epochs', OPT_INT, 'Screening epochs', 'Epochs of a cv screening run (default 2).', None, (), G_SMART),
    Opt('--top_families', OPT_INT, 'Top families kept', 'Default 2.', None, (), G_SMART),
    Opt('--top_backbones', OPT_INT, 'Top backbones kept', 'Default 3.', None, (), G_SMART),
    Opt('--tune_epochs', OPT_INT, 'Tuning epochs', 'Default: the epochs above.', None, (), G_SMART),
    Opt('--recheck_models', OPT_INT, 'Re-checked models', 'Default 2; 0 turns it off.', None, (), G_SMART),
    Opt('--recheck_full_product', OPT_INT, 'Full-product re-check limit', 'Default 6; 0: runner-ups only.', None, (), G_SMART),
    Opt('--final_top', OPT_INT, 'Final ranking size', 'Candidates in the final grouped CV ranking (default 3).', None, (), G_SMART),
    Opt('--include', WORDS, 'Keep past the cuts', 'KEY=VALUE, e.g. backbone=r3d_18 family=vjepa2 (smart only).', (), (), G_SMART),
    Opt('--n_trials', OPT_INT, 'Trials', 'Candidates to train (default: until the time budget or the grid runs out).', None, (),
        G_SAMPLED),
    Opt('--bayes_startup', OPT_INT, 'Random startup trials', 'bayes only (default: a quarter of the trials, at least 3).', None,
        (), G_SAMPLED),
    Opt('--exclude', WORDS, 'Exclude', 'KEY=VALUE, e.g. backbone=swin3d_b train_mode=full family=videomae.', (), (), G_OVERRIDES),
    Opt('--fix', WORDS, 'Fix', 'KEY=VALUE, e.g. learning_rate=1e-4: other values are skipped.', (), (), G_OVERRIDES),
    Opt('--max_trainable_params', TEXT, 'Max trainable parameters', 'e.g. 5M (default: no limit).', None, (), G_LIMITS),
    Opt('--gpu_memory_gb', TEXT, 'GPU memory (GB)', 'auto (default): the visible GPU. none: no check. Or a number.', 'auto', (),
        G_LIMITS),
    Opt('--preflight', CHOICE, 'Preflight', 'full (default): build each model on the CPU (may download weights). params: count '
        'only. off: no build.', 'full', ('full', 'params', 'off'), G_LIMITS),
    Opt('--no_pilot', BOOL, 'Skip the timing pilot', 'The time estimate is then unknown.', False, (), G_LIMITS),
    Opt('--max_hours', FLOAT, 'Time budget (hours)', 'exhaustive: refuse to start above it. Others: a running budget (default 8).',
        8.0, (), G_LIMITS),
    Opt('--profile_clips', INT, 'Profile clips', 'Clips the data profile opens (default 16; 0: none).', 16, (), G_LIMITS),
    Opt('--yes', BOOL, 'Run even over the time budget', 'Or a large run without a time estimate.', False, (), G_LIMITS),
    Opt('--dry_run', BOOL, 'Dry run', 'Set by the Check grid button.', False, (), G_LIMITS, hidden=True),
    Opt('--estimate_only', BOOL, 'Pilot and estimate only', 'Run the pilot, print the estimate, and stop.', False, (), G_LIMITS),
    Opt('--allow_grid_change', BOOL, 'Resume with a changed grid', 'Keep results of candidates still in the grid.', False, (), G_LIMITS),
    Opt('--retry_failed', BOOL, 'Retry failed candidates on resume', '', False, (), G_LIMITS),
    Opt('--rescore_sealed_test', BOOL, 'Allow the sealed test to score a new winner', 'selection.json then says it was reused, '
        'and its score is no longer untouched.', False, (), G_LIMITS),
)
SELECT_SECTIONS = (G_DATA, G_GRID, G_TRAIN, G_SMART, G_SAMPLED, G_OVERRIDES, G_LIMITS)


# ---------------------------------------------------------------------------------------------------------------------
# ensemble.py (Ensemble page)
# ---------------------------------------------------------------------------------------------------------------------
E_MAIN = 'Ensemble'
ENSEMBLE_OPTIONS = (
    Opt('--runs', LINES, 'Trained runs (one per line)', 'Weights files or folders holding one model_weights.pth with '
        'cli_arguments.json beside it.', (), (), E_MAIN, browse='folder'),
    Opt('--predictions', LINES, 'Finished predictions.csv files (one per line)', 'Tables with probability_<k> columns and the '
        'same rows, e.g. from feature probes.', (), (), E_MAIN, browse='file'),
    Opt('--out', TEXT, 'Output folder', 'predictions.csv, ensemble.json, and inference_metrics.json when there are labels.',
        None, (), E_MAIN, required=True, browse='folder'),
    Opt('--data', TEXT, 'Data', 'With runs: a folder of clips, one clip, or a CSV/XLSX list of paths.', None, (), E_MAIN, browse='file'),
    Opt('--batch_size', INT, 'Batch size', 'With runs (default 16).', 16, (), E_MAIN),
    Opt('--mode', CHOICE, 'Task', 'With predictions files (default cls); runs read it from their record.', 'cls', ('cls', 'reg'), E_MAIN),
    Opt('--label_mapping', TEXT, 'Label mapping file', "A run's data_splits_*/label_mapping.json, for class names.", None, (),
        E_MAIN, browse='file'),
    Opt('--allow_unverified_labels', BOOL, 'Allow unverified class indices', 'Average members even if what their class indices '
        'mean cannot be verified. ensemble.json says so.', False, (), E_MAIN),
    Opt('--accept_weights_licence', MULTI, 'I accept the licence of these weights', 'With runs: opt-in weights whose run did '
        'not record the acceptance.', (), HUB_LICENCES, E_MAIN),
)


# ---------------------------------------------------------------------------------------------------------------------
# extract_features.py and train_feature_head.py (Features & probes page)
# ---------------------------------------------------------------------------------------------------------------------
F_MAIN = 'Extract features'
EXTRACT_OPTIONS = (
    Opt('--data', TEXT, 'Videos', "A CSV/XLSX manifest or a parquet with a 'videos' column.", None, (), F_MAIN, required=True,
        browse='file'),
    Opt('--out', TEXT, 'Feature folder', 'Already extracted videos in it are skipped, so a stopped run resumes.', None, (),
        F_MAIN, required=True, browse='folder'),
    Opt('--streams', MULTI, 'Streams', 'rgb (default) and/or motion (codec motion vectors or Farneback flow).', ('rgb',),
        ('rgb', 'motion'), F_MAIN),
    Opt('--model_version', TEXT, 'RGB backbone', 'Any DeepTune video or frame backbone, e.g. r3d_18, swin3d_t, resnet18, '
        'vjepa2_vitl (default r3d_18).', 'r3d_18', (), F_MAIN),
    Opt('--motion_backbone', TEXT, 'Motion backbone', 'Default r3d_18.', 'r3d_18', (), F_MAIN),
    Opt('--feature_weights', CHOICE, 'Weights', 'pretrained (default; may download) or none (random, for tests and timing).',
        'pretrained', ('pretrained', 'none'), F_MAIN),
    Opt('--snippet_frames', INT, 'Frames per snippet', 'Default 16.', 16, (), F_MAIN),
    Opt('--snippet_stride', OPT_INT, 'Snippet stride', 'Frames between snippet starts (default: no overlap).', None, (), F_MAIN),
    Opt('--cache_tokens', TEXT, 'Token grid', "'none' (default): one vector per snippet. 'T' or 'TxHxW', e.g. 8x4x4, for the "
        "attentive probe.", 'none', (), F_MAIN),
    Opt('--batch_size', INT, 'Snippets per pass', 'Default 8.', 8, (), F_MAIN),
    Opt('--motion_method', CHOICE, 'Motion method', 'auto (default): codec, else Farneback.', 'auto', ('auto', 'codec', 'farneback'),
        F_MAIN),
    Opt('--motion_gain', FLOAT, 'Motion gain', 'Default 0.05.', 0.05, (), F_MAIN),
    Opt('--feature_input_size', OPT_INT, 'Input size', "Default: the backbone's own preprocessing.", None, (), F_MAIN),
    Opt('--video_backend', CHOICE, 'Decoder', 'Default auto.', 'auto', ('auto', 'torchcodec', 'pyav', 'opencv'), F_MAIN),
    Opt('--accept_weights_licence', MULTI, 'I accept the licence of these weights', 'Opt-in weights such as VideoMAE.', (),
        HUB_LICENCES, F_MAIN),
)
P_MAIN = 'Train a probe'
PROBE_OPTIONS = (
    Opt('--features', TEXT, 'Feature folder', 'Written by Extract features.', None, (), P_MAIN, required=True, browse='folder'),
    Opt('--train_df', TEXT, 'Training rows', 'CSV/XLSX/parquet with videos (or video_id) and labels.', None, (), P_MAIN,
        required=True, browse='file'),
    Opt('--val_df', TEXT, 'Validation rows', '', None, (), P_MAIN, required=True, browse='file'),
    Opt('--test_df', TEXT, 'Test rows (optional)', 'Disjoint from train and validation; writes metrics.json and predictions.csv.',
        None, (), P_MAIN, browse='file'),
    Opt('--out', TEXT, 'Output folder', '', None, (), P_MAIN, required=True, browse='folder'),
    Opt('--head', CHOICE, 'Head', 'Default attentive_probe.', 'attentive_probe', ('attentive_probe', 'linear_probe'), P_MAIN),
    Opt('--streams', MULTI, 'Streams', 'Default: every stream in the folder.', (), ('rgb', 'motion'), P_MAIN),
    Opt('--target', TEXT, 'Label column', 'Numeric labels (default labels; e.g. label_idx).', 'labels', (), P_MAIN),
    Opt('--mode', CHOICE, 'Task', 'Default cls.', 'cls', ('cls', 'reg'), P_MAIN),
    Opt('--num_classes', INT, 'Number of classes', 'With classification (default 2).', 2, (), P_MAIN),
    Opt('--num_snippets', INT, 'Snippets per video', 'Default 8.', 8, (), P_MAIN),
    Opt('--batch_size', INT, 'Batch size', 'Default 16.', 16, (), P_MAIN),
    Opt('--num_epochs', INT, 'Epochs', 'Default 10.', 10, (), P_MAIN),
    Opt('--learning_rate', FLOAT, 'Learning rate', 'Default 1e-3.', 1e-3, (), P_MAIN),
    Opt('--probe_width', INT, 'Probe width', 'attentive_probe (default 256).', 256, (), P_MAIN),
    Opt('--probe_depth', INT, 'Probe depth', 'attentive_probe, 1 to 4 (default 1).', 1, (), P_MAIN),
    Opt('--probe_heads', INT, 'Probe heads', 'attentive_probe (default 4).', 4, (), P_MAIN),
    Opt('--num_workers', INT, 'DataLoader workers', 'Default 0.', 0, (), P_MAIN),
    Opt('--fixed-seed', BOOL, 'Use a reproducible seed (42)', '', False, (), P_MAIN),
)


# ---------------------------------------------------------------------------------------------------------------------
# track.py (Analyse Video page, "More tracking options"). The page's own controls set --video, --prompt, --out,
# --detector, --every_n_frames, --save_video and the thresholds of its one sensitivity setting; these add the rest, and
# a value here replaces what the sensitivity setting chose.
# ---------------------------------------------------------------------------------------------------------------------
T_MORE = 'More tracking options'
TRACK_OPTIONS = (
    Opt('--video_dir', TEXT, 'Track a whole folder instead', 'Every video in this folder (not recursive), each into its own '
        'sub-folder. The page then shows the log and the output folder, not one overlay.', None, (), T_MORE, browse='folder'),
    Opt('--detector_model', TEXT, 'Detector model', 'Hugging Face id or local folder (blank: the detector default).', None, (),
        T_MORE, browse='folder'),
    Opt('--box_threshold', OPT_FLOAT, 'Box threshold', 'Minimum box score (blank: from the score threshold above).', None, (), T_MORE),
    Opt('--text_threshold', OPT_FLOAT, 'Text threshold', 'Grounding DINO only (blank: from the score threshold above).', None, (),
        T_MORE),
    Opt('--nms_iou', FLOAT, 'NMS IoU', 'Per-label NMS on detector boxes; 1 or more disables (default 0.5).', 0.5, (), T_MORE),
    Opt('--max_frames', OPT_INT, 'Max frames', 'Stop after this many decoded frames.', None, (), T_MORE),
    Opt('--batch_size', INT, 'Frames per detector pass', 'Default 4.', 4, (), T_MORE),
    Opt('--device', CHOICE, 'Device', 'auto (default): CUDA if available.', 'auto', ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'), T_MORE),
    Opt('--track_high_thresh', OPT_FLOAT, 'Track high threshold', 'First association (blank: from the score threshold).', None, (),
        T_MORE),
    Opt('--track_low_thresh', OPT_FLOAT, 'Track low threshold', 'Second association (default: the box threshold).', None, (), T_MORE),
    Opt('--new_track_thresh', OPT_FLOAT, 'New track threshold', 'Blank: the score threshold above.', None, (), T_MORE),
    Opt('--match_thresh', FLOAT, 'Match threshold', 'IoU-distance gate of the first association (default 0.8).', 0.8, (), T_MORE),
    Opt('--track_buffer', INT, 'Track buffer', 'Steps a lost track can be re-found (default 30).', 30, (), T_MORE),
    Opt('--cross_label_nms', OPT_FLOAT, 'Cross-prompt merge IoU', 'Merge overlapping boxes of different prompts above this IoU, '
        'e.g. 0.7 (default off).', None, (), T_MORE),
    Opt('--class_agnostic', BOOL, 'Match boxes across labels', 'Each track takes the majority label.', False, (), T_MORE),
)

# ---------------------------------------------------------------------------------------------------------------------
# Device: not a flag of deeptune.py or the other training scripts (they take CUDA when torch sees it). The GUI hides
# the GPU from a run with CUDA_VISIBLE_DEVICES, the same switch CLAUDE.md documents for the command line.
# ---------------------------------------------------------------------------------------------------------------------
DEVICES = ('auto', 'cpu', 'cuda:0', 'cuda:1')


def device_env(device: str) -> dict:
    """Environment overrides for a run on `device`: auto changes nothing, cpu hides every GPU, cuda:N shows only GPU N."""
    if device in (None, '', 'auto'):
        return {}
    if device == 'cpu':
        return {'CUDA_VISIBLE_DEVICES': '-1'}
    if device.startswith('cuda:') and device[5:].isdigit():
        return {'CUDA_VISIBLE_DEVICES': device[5:]}
    raise ValueError(f'Unknown device {device!r}; choose one of {", ".join(DEVICES)}.')

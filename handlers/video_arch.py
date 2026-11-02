"""Temporal video architectures: CLI flags and the pooling spec (Phase 3). Torch-free, so the GUI can import it.

A run's architecture travels as one string, the *pooling spec*, through the
existing ``pooling`` setting that deeptune.py, the trainers, evaluators, embed
code, cli_arguments.json, the desktop Test page and infer.py already pass along:

    mean | attention | single_frame                 frame-sampling models, as before
    tsm:tsm_fold_div=8                               <architecture>[:option=value,...]
    cnn_rnn:bidirectional=0,rnn_hidden=256,rnn_layers=1,rnn_type=lstm

``--video_arch`` and its option flags compose the spec (``pooling_from_args``);
``utils.build_video_model`` parses it (``parse_pooling_spec``).
"""
from __future__ import annotations

from handlers.video_backbones import add_weights_licence_argument

FRAME_POOLINGS = ('mean', 'attention', 'single_frame')
TEMPORAL_ARCHITECTURES = ('late_fusion', 'early_fusion', 'tsm', 'cnn_rnn', 'temporal_transformer', 'two_stream')
# The heads of clip models with their own tokens (V-JEPA 2, VideoMAE): src/video/hub_backbones.py.
PROBE_ARCHITECTURES = ('attentive_probe', 'linear_probe')
VIDEO_ARCH_CHOICES = ('frame_pool', 'single_frame', 'slow_fusion', *TEMPORAL_ARCHITECTURES, *PROBE_ARCHITECTURES)
# Karpathy et al.'s slow fusion is a 3D CNN: --video_arch slow_fusion selects this native model.
SLOW_FUSION_DEFAULT = 'r2plus1d_18'

_OPTION_TYPES = {'num_frames': int, 'tsm_fold_div': int, 'rnn_type': str, 'rnn_hidden': int, 'rnn_layers': int,
                 'bidirectional': bool, 'tt_layers': int, 'tt_heads': int, 'tt_dim': int, 'fusion': str,
                 'motion_length': int, 'probe_width': int, 'probe_depth': int, 'probe_heads': int}
ARCH_OPTIONS = {
    'late_fusion': (),
    'early_fusion': ('num_frames',),
    'tsm': ('tsm_fold_div',),
    'cnn_rnn': ('rnn_type', 'rnn_hidden', 'rnn_layers', 'bidirectional'),
    'temporal_transformer': ('num_frames', 'tt_layers', 'tt_heads', 'tt_dim'),
    'two_stream': ('num_frames', 'fusion', 'motion_length'),
    'attentive_probe': ('probe_depth', 'probe_heads', 'probe_width'),
    'linear_probe': (),
}
_ALLOWED_VALUES = {'rnn_type': ('lstm', 'gru'), 'fusion': ('avg', 'fc')}
# What an option flag means when neither the flag nor a recorded spec sets it. The flags themselves default to None,
# so a recorded spec can tell "not given" from "given with the default value".
ARCH_OPTION_DEFAULTS = {'tsm_fold_div': 8, 'rnn_type': 'lstm', 'rnn_hidden': 256, 'rnn_layers': 1, 'bidirectional': False,
                        'tt_layers': 2, 'tt_heads': 4, 'tt_dim': 256, 'fusion': 'avg', 'probe_width': 256, 'probe_depth': 1, 'probe_heads': 4}

# Everything add_video_arch_arguments adds; recorded in cli_arguments.json.
VIDEO_ARCH_ARG_NAMES = ('video_arch', 'tsm_fold_div', 'rnn_type', 'rnn_hidden', 'rnn_layers', 'bidirectional',
                        'tt_layers', 'tt_heads', 'tt_dim', 'fusion', 'motion_length', 'motion_cache',
                        'probe_width', 'probe_depth', 'probe_heads', 'accept_weights_licence', 'unfreeze_backbone')
# Sources of the motion stream, chosen with --motion_source (which also names the --temporal_sampling motion curve).
CLIP_MOTION_SOURCES = ('farneback', 'mv')
_TWO_STREAM_SOURCE = {'auto': 'farneback', 'farneback': 'farneback', 'mv': 'mv', 'codec': 'mv'}
_CURVE_METHOD = {'farneback': 'framediff', 'mv': 'codec'}


def _convert(name: str, text: str):
    kind = _OPTION_TYPES[name]
    if kind is bool:
        if text not in ('0', '1'):
            raise ValueError(f'{name} must be 0 or 1, got {text!r}.')
        return text == '1'
    if kind is int:
        value = int(text)
        if value < 1:
            raise ValueError(f'{name} must be at least 1, got {value}.')
        return value
    if text not in _ALLOWED_VALUES[name]:
        raise ValueError(f'{name} must be one of {", ".join(_ALLOWED_VALUES[name])}, got {text!r}.')
    return text


def parse_pooling_spec(spec: str) -> tuple[str, dict]:
    """``'tsm:tsm_fold_div=4'`` -> ``('tsm', {'tsm_fold_div': 4})``. Raises ValueError on anything unknown."""
    name, _, rest = str(spec).partition(':')
    if name in FRAME_POOLINGS and not rest:
        return name, {}
    if name not in TEMPORAL_ARCHITECTURES + PROBE_ARCHITECTURES:
        raise ValueError(f'Unknown pooling {spec!r}; choose one of {", ".join(FRAME_POOLINGS + TEMPORAL_ARCHITECTURES + PROBE_ARCHITECTURES)}.')
    options = {}
    for item in filter(None, rest.split(',')):
        key, separator, value = item.partition('=')
        if not separator or key not in ARCH_OPTIONS[name]:
            raise ValueError(f'{spec!r}: {name} has the options {", ".join(ARCH_OPTIONS[name]) or "(none)"}, not {item!r}.')
        options[key] = _convert(key, value)
    return name, options


def format_pooling_spec(arch: str, **options) -> str:
    """The inverse of parse_pooling_spec. Options the architecture does not use, and None, are dropped."""
    if arch in FRAME_POOLINGS:
        return arch
    if arch not in TEMPORAL_ARCHITECTURES + PROBE_ARCHITECTURES:
        raise ValueError(f'Unknown architecture {arch!r}; choose one of {", ".join(TEMPORAL_ARCHITECTURES + PROBE_ARCHITECTURES)}.')
    parts = []
    for key in sorted(ARCH_OPTIONS[arch]):
        value = options.get(key)
        if value is not None:
            parts.append(f'{key}={int(value) if isinstance(value, bool) else value}')
    spec = f'{arch}:{",".join(parts)}' if parts else arch
    parse_pooling_spec(spec)
    return spec


def pooling_argument(text: str) -> str:
    """argparse ``type`` of --pooling: a frame pooling or a recorded architecture spec."""
    parse_pooling_spec(text)
    return text


pooling_argument.__name__ = 'pooling'


def _value(args, name):
    return args.get(name) if isinstance(args, dict) else getattr(args, name, None)


def pooling_from_args(args, num_frames_given: bool = True) -> str:
    """The pooling spec for parsed arguments. With ``video_arch`` set, each option comes from its flag if given, else
    from a spec of the same architecture in ``pooling`` (what a run recorded), else from its default.

    ``num_frames`` always has a value, so the caller says whether it was given (see ``flag_was_given``)."""
    arch, pooling = _value(args, 'video_arch'), _value(args, 'pooling') or 'mean'
    if arch in (None, 'frame_pool'):
        return pooling
    if pooling != 'mean' and pooling.partition(':')[0] != arch:
        raise ValueError(f'--pooling {pooling} does not apply to --video_arch {arch}: use one of them.')
    if arch == 'single_frame':
        return arch
    if arch == 'slow_fusion':
        return 'mean'
    recorded = parse_pooling_spec(pooling)[1] if pooling.partition(':')[0] == arch else {}
    options = {name: ARCH_OPTION_DEFAULTS.get(name) for name in ARCH_OPTIONS[arch]}
    options.update(recorded)
    for name in ARCH_OPTIONS[arch]:
        value = _value(args, name)
        if value is not None and (name != 'num_frames' or num_frames_given or name not in recorded):
            options[name] = value
    return format_pooling_spec(arch, **options)


def flag_was_given(parser, argv, name: str) -> bool:
    """Whether ``--name`` was on the command line, in any spelling argparse accepts. Parses again with no default."""
    default = parser.get_default(name)
    parser.set_defaults(**{name: None})
    try:
        return getattr(parser.parse_args(argv), name) is not None
    finally:
        parser.set_defaults(**{name: default})


def spec_num_frames(pooling: str, num_frames: int) -> int:
    """Clips must have the length the model was built for: the spec's frame count if it has one."""
    return parse_pooling_spec(pooling)[1].get('num_frames', num_frames)


def resolve_model_version(args, is_native):
    """slow_fusion keeps a native video model and otherwise selects the documented default."""
    version = _value(args, 'model_version')
    if _value(args, 'video_arch') == 'slow_fusion' and not is_native(version):
        print(f'--video_arch slow_fusion: using {SLOW_FUSION_DEFAULT} instead of the 2D model {version}.')
        return SLOW_FUSION_DEFAULT
    return version


def clip_motion_source(pooling, motion_source):
    """Motion stream to attach to clips: 'farneback', 'mv', or None for models without one."""
    if not isinstance(pooling, str) or pooling.partition(':')[0] != 'two_stream':
        return None
    motion_source = motion_source if isinstance(motion_source, str) else 'auto'
    if motion_source not in _TWO_STREAM_SOURCE:
        raise ValueError(f'two_stream needs --motion_source farneback or mv, not {motion_source!r}.')
    return _TWO_STREAM_SOURCE[motion_source]


def curve_method(motion_source: str) -> str:
    """--temporal_sampling motion has no dense-flow curve: farneback means frame differences, mv means the codec."""
    return _CURVE_METHOD.get(motion_source, motion_source)


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise ValueError('must be at least 1')
    return value


_positive_int.__name__ = 'positive integer'


def add_video_arch_arguments(parser):
    """Temporal architecture flags (video modality only; ignored elsewhere)."""
    g = parser.add_argument_group('temporal video architectures (video modality only)')
    g.add_argument('--video_arch', choices=VIDEO_ARCH_CHOICES, default=None,
                   help="""How the model fuses time, with --model_version as the backbone (default: frame_pool, set by --pooling).
With a recorded run, an option flag below replaces the recorded value only if you give it.
With --freeze_backbone, early_fusion and two_stream keep their inflated first convolution trainable.
frame_pool: pool per-frame embeddings. single_frame: middle frame only.
late_fusion: first and last frames, embeddings merged by FC layers.
early_fusion: all frames stacked on the first convolution (ResNet family).
slow_fusion: a native 3D CNN (--model_version if it names one, else r2plus1d_18).
tsm: Temporal Shift Module in each residual block (ResNet family).
cnn_rnn: per-frame embeddings into an LSTM or GRU.
temporal_transformer: per-frame embeddings into a small transformer with a CLS token.
two_stream: RGB middle frame + stacked motion fields (--motion_source; ResNet family).
attentive_probe: a learned query reads the encoder's tokens (V-JEPA 2 and VideoMAE; their default).
linear_probe: the mean of the tokens, then a linear layer (V-JEPA 2 and VideoMAE).""")
    g.add_argument('--tsm_fold_div', type=_positive_int, default=None,
                   help='tsm: shift 1/N of the channels forward and 1/N backward (default 8).')
    g.add_argument('--rnn_type', choices=_ALLOWED_VALUES['rnn_type'], default=None, help='cnn_rnn: recurrent cell (default lstm).')
    g.add_argument('--rnn_hidden', type=_positive_int, default=None, help='cnn_rnn: hidden size (default 256).')
    g.add_argument('--rnn_layers', type=_positive_int, default=None, help='cnn_rnn: layers (default 1).')
    g.add_argument('--bidirectional', action='store_true', default=None, help='cnn_rnn: read the clip in both directions.')
    g.add_argument('--tt_layers', type=_positive_int, default=None, help='temporal_transformer: encoder layers (default 2).')
    g.add_argument('--tt_heads', type=_positive_int, default=None, help='temporal_transformer: attention heads (default 4).')
    g.add_argument('--tt_dim', type=_positive_int, default=None,
                   help='temporal_transformer: token size, a multiple of --tt_heads (default 256).')
    g.add_argument('--fusion', choices=_ALLOWED_VALUES['fusion'], default=None,
                   help="two_stream: average the two streams' logits, or merge their embeddings with an FC head (default avg).")
    g.add_argument('--motion_length', type=_positive_int, default=None,
                   help='two_stream: motion fields stacked per clip, taken around the middle frame (default: --num_frames).')
    g.add_argument('--motion_cache', default=None, metavar='DIR',
                   help='two_stream: keep motion fields in this folder, keyed by video hash and parameters (default: off).')
    g.add_argument('--probe_width', type=_positive_int, default=None, help='attentive_probe: token width, a multiple of --probe_heads (default 256).')
    g.add_argument('--probe_depth', type=_positive_int, default=None, help='attentive_probe: cross-attention blocks, 1 to 4 (default 1).')
    g.add_argument('--probe_heads', type=_positive_int, default=None, help='attentive_probe: attention heads (default 4).')
    g.add_argument('--unfreeze_backbone', action='store_true',
                   help='V-JEPA 2 and VideoMAE: train the whole encoder. They are frozen by default; see the memory notes in the README.')
    add_weights_licence_argument(g)
    return g

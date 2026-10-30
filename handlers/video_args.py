"""Video data-layer CLI flags, shared by cli.py (deeptune.py and the
train/eval/embed scripts) and infer.py. Torch-free, so the desktop GUI can
import it too (it must never import torch)."""

# Shared by deeptune.py / the train, eval and embed scripts (DeepTuneVisionOptions)
# and infer.py, so inference rebuilds exactly the pipeline the run recorded.
VIDEO_DATA_ARG_NAMES = ('video_backend', 'video_storage', 'temporal_sampling', 'clip_stride', 'clip_duration_sec',
                        'motion_source', 'video_augment', 'video_flip', 'video_speed_perturb', 'test_clips',
                        'num_workers')


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise ValueError('must be at least 1')
    return value


def _positive_float(text):
    value = float(text)
    if not value > 0:
        raise ValueError('must be positive')
    return value


def _speed(text):
    value = float(text)
    if not 0 <= value < 1:
        raise ValueError('must be in [0, 1)')
    return value


def _workers(text):
    if str(text).lower() == 'auto':
        return 'auto'
    value = int(text)
    if value < 0:
        raise ValueError('must be 0 or more')
    return value


_positive_int.__name__ = 'positive integer'
_positive_float.__name__ = 'positive number'
_speed.__name__ = 'number in [0, 1)'
_workers.__name__ = "'auto' or an integer >= 0"


def add_video_data_arguments(parser, training: bool = True):
    """Video data-layer flags (video modality only; ignored elsewhere)."""
    g = parser.add_argument_group('video data (video modality only)')
    g.add_argument('--video_backend', choices=['auto', 'torchcodec', 'pyav', 'opencv'], default='auto',
                   help='Decoder: auto = torchcodec if installed, else PyAV (default), else OpenCV. All read exact frame indices.')
    g.add_argument('--temporal_sampling', choices=['uniform', 'segment', 'dense', 'motion'], default='uniform',
                   help="""Which frames form a clip (default uniform = evenly spaced over the clip, as before).
segment: TSN, random frame per segment in training, centre at test.
dense: --num_frames consecutive frames --clip_stride apart.
motion: MGSampler, evenly spaced along the cumulative motion curve (more frames where things move).""")
    g.add_argument('--clip_stride', type=_positive_int, default=None,
                   help='Frame step inside a dense clip (default 1).')
    g.add_argument('--clip_duration_sec', type=_positive_float, default=None,
                   help='Sample from a window this many seconds long (random position in training, centred at test). Default: whole clip.')
    g.add_argument('--motion_source', choices=['auto', 'codec', 'framediff'], default='auto',
                   help='Motion curve for --temporal_sampling motion: codec motion vectors (PyAV) or frame differences; auto = codec if available.')
    g.add_argument('--test_clips', type=_positive_int, default=1,
                   help='Clips per video at test/embedding/inference time; predictions average the clip probabilities (default 1).')
    g.add_argument('--num_workers', type=_workers, default='auto',
                   help="""DataLoader worker processes for video. auto (default): 0 on CPU; on Windows + GPU 0 for
datasets under 500 clips (each spawned worker re-imports torch, 15-40 s) and up to 4 above; min(8, cpu/2) elsewhere.""")
    if training:
        g.add_argument('--video_storage', choices=['paths', 'bytes'], default='paths',
                       help="""With --raw-data: keep file paths in the dataset parquet (default; clips are read lazily and
never copied) or embed the encoded clips as bytes (portable, the original format).""")
        g.add_argument('--video_augment', action='store_true',
                       help='Training only: temporal jitter, one random-resized crop and colour jitter per clip (same on every frame).')
        g.add_argument('--video_flip', action='store_true',
                       help="""With --video_augment: also flip clips horizontally (p=0.5). Off by default because a mirror turns
left motion into right motion, which corrupts direction-dependent labels.""")
        g.add_argument('--video_speed_perturb', type=_speed, default=0.0,
                       help='With --video_augment: random playback speed in [1-x, 1+x] (default 0 = off).')
    return g


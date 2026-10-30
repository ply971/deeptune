"""Video data-layer CLI flags, shared by cli.py (deeptune.py and the
train/eval/embed scripts) and infer.py. Torch-free, so the desktop GUI can
import it too (it must never import torch)."""

# Shared by deeptune.py / the train, eval and embed scripts (DeepTuneVisionOptions)
# and infer.py, so inference rebuilds exactly the pipeline the run recorded.
VIDEO_DATA_ARG_NAMES = ('video_backend', 'video_storage', 'temporal_sampling', 'clip_stride', 'clip_duration_sec',
                        'motion_source', 'video_augment', 'video_flip', 'video_speed_perturb', 'test_clips',
                        'num_workers', 'frame_cache', 'frame_cache_size')


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


def _non_negative_int(text):
    value = int(text)
    if value < 0:
        raise ValueError('must be 0 or more')
    return value


_positive_int.__name__ = 'positive integer'
_non_negative_int.__name__ = 'integer >= 0'
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
    g.add_argument('--frame_cache', default=None, metavar='DIR',
                   help="""Keep sampled, resized frames of deterministic clips in this folder (memory-mapped) and decode
them only once (default: off). Randomly sampled training clips are not cached; see datasets/frame_cache.py.""")
    g.add_argument('--frame_cache_size', type=_non_negative_int, default=256,
                   help="""With --frame_cache: shrink frames so the short side is this many pixels before the model's own
resize (default 256; never enlarges; 0 keeps the decoded size). Use a value at least as large as the model input.""")
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


# extract_features.py records these in <out>/cli_arguments.json.
SNIPPET_FEATURE_ARG_NAMES = ('data', 'out', 'streams', 'model_version', 'motion_backbone', 'feature_weights', 'snippet_frames',
                             'snippet_stride', 'cache_tokens', 'batch_size', 'motion_method', 'feature_input_size',
                             'video_backend')


def add_snippet_feature_arguments(parser):
    """Flags of extract_features.py (cached snippet features)."""
    parser.add_argument('--data', required=True,
                        help="Videos: a CSV/XLSX manifest or a parquet file with a 'videos' column (paths or bytes). An optional 'video_id' column names the feature files.")
    parser.add_argument('--out', required=True, help='Feature folder. Already extracted videos in it are skipped.')
    parser.add_argument('--streams', nargs='+', choices=['rgb', 'motion'], default=['rgb'],
                        help='Streams to extract (default rgb). motion = codec motion vectors or Farneback flow.')
    parser.add_argument('--model_version', default='r3d_18',
                        help='rgb backbone: any DeepTune video or frame backbone, e.g. r3d_18, swin3d_t, resnet18 (default r3d_18).')
    parser.add_argument('--motion_backbone', default='r3d_18',
                        help='motion backbone; its first convolution is inflated to 2 channels (default r3d_18).')
    parser.add_argument('--feature_weights', choices=['pretrained', 'none'], default='pretrained',
                        help='pretrained (default; may download) or none = fixed random weights, for tests and timing only.')
    parser.add_argument('--snippet_frames', type=_positive_int, default=16, help='Frames per snippet (default 16).')
    parser.add_argument('--snippet_stride', type=_positive_int, default=None,
                        help='Frames between snippet starts (default: --snippet_frames, no overlap).')
    parser.add_argument('--cache_tokens', default='none',
                        help="""What to store per snippet: 'none' (default) one vector [N, D]; 'T' or 'TxHxW', e.g. 8x4x4, a pooled
token grid [N, T, H, W, D] for the attentive probe. An axis is capped at what the backbone produces; the sidecar records the real grid.""")
    parser.add_argument('--batch_size', type=_positive_int, default=8, help='Snippets per forward pass (default 8).')
    parser.add_argument('--motion_method', choices=['auto', 'codec', 'farneback'], default='auto',
                        help='Motion source: codec motion vectors, Farneback flow, or auto = codec, else Farneback (default; recorded per video).')
    parser.add_argument('--feature_input_size', type=_positive_int, default=None,
                        help="Square input size. Default: the backbone's own preprocessing.")
    parser.add_argument('--video_backend', choices=['auto', 'torchcodec', 'pyav', 'opencv'], default='auto',
                        help='Decoder (default auto).')
    return parser


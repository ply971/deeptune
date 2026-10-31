"""The key assurance for Phase 3: on the held-out split of the Phase 2 direction task (scenes disjoint from training),
motion-aware models beat chance by a wide margin and appearance-only models do not, and the motion-aware models really
use time: reversing a clip reverses their answer and shuffling frames destroys it.

Data goes through the real decoding path (open_video_dataset, motion channels included). The model is tiny and untrained
(`resnet_tiny`), on CPU. Thresholds: with 200 test clips one standard error at chance is 0.035.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision as tv

from benchmarks.motion_features import read_frames
from benchmarks.render import write_mp4
from benchmarks.synthetic_motion import make_direction_dataset
from datasets.video_datasets import VideoDataOptions, open_video_dataset
from handlers.video_manifest import load_video_manifest
from utils import build_video_model

pytest.importorskip('av')

MOTION_AWARE_MIN = 0.85
CHANCE_MAX = 0.62  # 0.5 + 3.4 standard errors
REVERSED_MAX = 0.25  # the true label flips under reversal, so a motion model should now be wrong
FRAMES = 8
# Epochs and learning rate per model family. Training accuracy is not used anywhere below.
BUDGET = {'early_fusion': (15, 2e-3), 'tsm': (10, 2e-3), 'cnn_rnn': (10, 2e-3), 'two_stream': (15, 2e-3),
          'single_frame': (15, 2e-3), 'mean': (10, 2e-3), 'attention': (10, 2e-3), 'late_fusion': (25, 2e-3)}
TT_POOLING = 'temporal_transformer:num_frames=8,tt_dim=64,tt_heads=4,tt_layers=2'
TT_BUDGET = (15, 3e-4)
SLOW_SPEEDS = (0.5, 1.0)  # half the default speed, in px/frame at 32 px


def _transform(size):
    return tv.transforms.Compose([tv.transforms.Resize((size, size)), tv.transforms.ToTensor(),
                                  tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Task:
    """A generated direction benchmark, decoded once per motion source."""

    def __init__(self, root: Path, size: int, n_train: int, speed_range=(1.0, 2.0), seed: int = 0):
        self.root, self.size = root, size
        self.ds = make_direction_dataset(root, n_train=n_train, n_val=0, n_test=200, size=size, seed=seed, codec='mp4v',
                                         speed_range=speed_range)
        self._cache = {}

    def _decode(self, frame, motion):
        dataset = open_video_dataset(frame[['videos', 'labels']], transform=_transform(32), num_frames=FRAMES,
                                     options=VideoDataOptions(clip_motion=motion))
        return torch.stack([dataset[i][0] for i in range(len(dataset))]), torch.tensor(frame['labels'].to_numpy())

    def split(self, name, motion=None):
        if (name, motion) not in self._cache:
            frame = load_video_manifest(self.ds.splits[name])
            frame['labels'] = frame['label_idx'].astype(int)
            self._cache[name, motion] = self._decode(frame, motion)
        return self._cache[name, motion]

    def rewritten_test(self, motion, reorder, tmp: Path):
        """The test clips re-encoded with their frames reordered, so motion is computed from the new order. The labels
        are the original ones."""
        frame = load_video_manifest(self.ds.splits['test'])
        frame['labels'] = frame['label_idx'].astype(int)
        paths = []
        for i, video in enumerate(frame['videos']):
            path = tmp / f'{i}.mp4'
            original = tmp / f'{i}.source.mp4'
            original.write_bytes(video if isinstance(video, bytes) else Path(video).read_bytes())
            write_mp4(path, reorder(read_frames(original)), 10, 'mp4v')
            paths.append(str(path))
        return self._decode(frame.assign(videos=paths), motion)


def _fit(pooling, x, y, seed=0, budget=None, version='resnet_tiny'):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = build_video_model(version, 2, 1, 32, False, 'cls', pooling=pooling)
    epochs, lr = budget or BUDGET[pooling.partition(':')[0]]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), 16):
            batch = order[start:start + 16]
            optimiser.zero_grad()
            torch.nn.functional.cross_entropy(model(x[batch]), y[batch]).backward()
            optimiser.step()
    return model.eval()


def _accuracy(model, x, y):
    with torch.no_grad():
        predicted = torch.cat([model(x[i:i + 32]).argmax(1) for i in range(0, len(x), 32)])
    return (predicted == y).float().mean().item()


def passes_motion_oracle(accuracy, reversed_accuracy, shuffled):
    """The three checks every motion model's oracle makes, as one verdict."""
    return bool(accuracy >= MOTION_AWARE_MIN and reversed_accuracy <= REVERSED_MAX and np.mean(shuffled) <= accuracy - 0.2)


def _permutations(count=3, frames=FRAMES):
    """Orders that are neither the original nor its reverse."""
    generator = torch.Generator().manual_seed(1)
    found = []
    while len(found) < count:
        order = torch.randperm(frames, generator=generator)
        if (order[1:] - order[:-1]).abs().float().mean() > 2:  # far from any monotonic order
            found.append(order)
    return found


@pytest.fixture(scope='module')
def task(tmp_path_factory):
    return Task(tmp_path_factory.mktemp('direction'), 32, 240)


@pytest.fixture(scope='module')
def fitted(task):
    models = {}

    def get(pooling, motion=None):
        if (pooling, motion) not in models:
            models[pooling, motion] = _fit(pooling, *task.split('train', motion))
        return models[pooling, motion]
    return get


def test_train_and_test_scenes_are_disjoint(task):
    frames = {s: pd.read_csv(task.ds.splits[s]) for s in ('train', 'test')}
    assert not set(frames['train']['group']) & set(frames['test']['group'])
    assert not set(frames['train']['videos']) & set(frames['test']['videos'])
    assert np.isclose(frames['test']['label_idx'].mean(), 0.5)


@pytest.mark.parametrize('pooling', ['single_frame', 'mean', 'attention'])
def test_appearance_models_stay_at_chance(task, fitted, pooling):
    accuracy = _accuracy(fitted(pooling), *task.split('test'))
    print(f'{pooling}: held-out {accuracy:.3f}')
    assert accuracy <= CHANCE_MAX


@pytest.mark.parametrize('pooling', ['early_fusion:num_frames=8', 'tsm', 'cnn_rnn'])
def test_rgb_motion_models_beat_chance_and_depend_on_frame_order(task, fitted, pooling):
    model = fitted(pooling)
    x, y = task.split('test')
    accuracy = _accuracy(model, x, y)
    reversed_accuracy = _accuracy(model, x.flip(1), y)
    shuffled = [_accuracy(model, x[:, order], y) for order in _permutations()]
    print(f'{pooling}: held-out {accuracy:.3f}, reversed {reversed_accuracy:.3f}, shuffled {np.round(shuffled, 3)}')
    assert accuracy >= MOTION_AWARE_MIN
    assert reversed_accuracy <= REVERSED_MAX
    assert np.mean(shuffled) <= accuracy - 0.2


def test_two_stream_farneback_beats_chance_and_follows_the_frame_order_of_the_video(task, fitted, tmp_path):
    model = fitted('two_stream:num_frames=8', 'farneback')
    x, y = task.split('test', 'farneback')
    accuracy = _accuracy(model, x, y)
    reversed_clip, _ = task.rewritten_test('farneback', lambda f: f[::-1], tmp_path)
    reversed_accuracy = _accuracy(model, reversed_clip, y)
    blind = x.clone()
    blind[:, :, 3:] = 0
    blind_accuracy = _accuracy(model, blind, y)
    print(f'two_stream farneback: held-out {accuracy:.3f}, reversed video {reversed_accuracy:.3f}, no motion {blind_accuracy:.3f}')
    assert accuracy >= MOTION_AWARE_MIN
    assert reversed_accuracy <= REVERSED_MAX
    assert blind_accuracy <= CHANCE_MAX


def test_two_stream_codec_motion_vectors_beat_chance(tmp_path_factory, tmp_path):
    """48 px: at 32 px an MPEG-4 clip has only 4 macroblocks per frame (docs/video-plan/benchmark.md)."""
    task = Task(tmp_path_factory.mktemp('direction48'), 48, 240)
    model = _fit('two_stream:num_frames=8', *task.split('train', 'mv'))
    x, y = task.split('test', 'mv')
    accuracy = _accuracy(model, x, y)
    reversed_clip, _ = task.rewritten_test('mv', lambda f: f[::-1], tmp_path)
    reversed_accuracy = _accuracy(model, reversed_clip, y)
    print(f'two_stream mv: held-out {accuracy:.3f}, reversed video {reversed_accuracy:.3f}')
    assert accuracy >= MOTION_AWARE_MIN
    assert reversed_accuracy <= REVERSED_MAX


@pytest.mark.slow
def test_late_fusion_beats_chance_when_two_frames_can_decide(tmp_path_factory):
    """Late fusion sees the first and last frames only. On the torus, the default speeds (1-2 px/frame at 32 px, 11-22 px
    between those frames) make +d and -(32-d) nearly the same picture, so that task is undecidable from two frames (measured
    0.505-0.555 with 800 clips). At half the speed the displacement stays under half the frame, and two frames do decide."""
    task = Task(tmp_path_factory.mktemp('direction_slow'), 32, 800, speed_range=(0.5, 1.0))
    model = _fit('late_fusion', *task.split('train'))
    x, y = task.split('test')
    accuracy = _accuracy(model, x, y)
    reversed_accuracy = _accuracy(model, x.flip(1), y)
    print(f'late_fusion slow: held-out {accuracy:.3f}, reversed {reversed_accuracy:.3f}')
    assert accuracy >= MOTION_AWARE_MIN
    assert reversed_accuracy <= REVERSED_MAX


@pytest.mark.slow
def test_temporal_transformer_beats_chance_given_enough_clips(tmp_path_factory):
    """The most data-hungry model here: with 240 and 800 clips it memorises them and stays near chance (0.575, 0.575)."""
    task = Task(tmp_path_factory.mktemp('direction_many'), 32, 1500)
    model = _fit('temporal_transformer:num_frames=8,tt_dim=64,tt_heads=4,tt_layers=2', *task.split('train'), budget=(15, 3e-4))
    x, y = task.split('test')
    accuracy = _accuracy(model, x, y)
    reversed_accuracy = _accuracy(model, x.flip(1), y)
    shuffled = [_accuracy(model, x[:, order], y) for order in _permutations()]
    print(f'temporal_transformer: held-out {accuracy:.3f}, reversed {reversed_accuracy:.3f}, shuffled {np.round(shuffled, 3)}')
    assert accuracy >= MOTION_AWARE_MIN
    assert reversed_accuracy <= REVERSED_MAX
    assert np.mean(shuffled) <= accuracy - 0.2


def test_tsm_without_its_shift_cannot_see_motion(task):
    """The control for TSM: the same model and budget, with no channel shifted, is a frame average. The shift is what works."""
    model = _fit('tsm:tsm_fold_div=1000', *task.split('train'))
    accuracy = _accuracy(model, *task.split('test'))
    print(f'tsm without shift: held-out {accuracy:.3f}')
    assert accuracy <= CHANCE_MAX


@pytest.mark.slow
def test_slow_fusion_beats_chance_and_depends_on_frame_order(task):
    """--video_arch slow_fusion is a native 3D CNN; dummy_video3d is an untrained r3d_18, the same family as the default."""
    model = _fit('mean', *task.split('train'), budget=(8, 1e-3), version='dummy_video3d')
    x, y = task.split('test')
    accuracy = _accuracy(model, x, y)
    reversed_accuracy = _accuracy(model, x.flip(1), y)
    shuffled = [_accuracy(model, x[:, order], y) for order in _permutations()]
    print(f'slow_fusion: held-out {accuracy:.3f}, reversed {reversed_accuracy:.3f}, shuffled {np.round(shuffled, 3)}')
    assert accuracy >= MOTION_AWARE_MIN
    assert reversed_accuracy <= REVERSED_MAX
    assert np.mean(shuffled) <= accuracy - 0.2


def test_shuffled_labels_give_chance_so_the_split_does_not_leak(task):
    x, y = task.split('train')
    labels = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(3))]
    model = _fit('tsm', x, labels)
    accuracy = _accuracy(model, *task.split('test'))
    print(f'tsm on shuffled labels: held-out {accuracy:.3f}')
    assert accuracy <= CHANCE_MAX

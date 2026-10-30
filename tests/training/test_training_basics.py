"""Training basics in the shared Trainer (T1): precision, resume, early stopping,
schedules, accumulation, config files, logging, seeding and the expert recipe.
CPU, tiny models and tiny tensors. Weights are compared on the CPU, because the
Trainer moves the model to options.DEVICE."""
import copy
import json
import logging
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cli import DeepTuneVisionOptions
from evaluators.error_model import weights_fingerprint
from handlers.training_args import TrainingOptions, load_config_file
from options import DEVICE
from trainers import training_basics as tb
from trainers.vision.trainer import BEST_WEIGHTS_FILENAME, CHECKPOINT_FILENAME, RUN_RECORD_FILENAME, Trainer
from utils import RunType


class TinyNet(nn.Module):
    """Dropout and BatchNorm, so that both RNG state and buffers have to be restored."""

    def __init__(self, classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(4, classes)

    def forward(self, x, extract_embed=False):
        x = torch.relu(self.norm(self.conv(x))).mean(dim=(2, 3))
        return self.fc1(self.drop(x))


class TinyClipNet(nn.Module):
    """Takes [B, T, C, H, W] like every DeepTune video model."""

    def __init__(self, classes=2):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, 3, padding=1)
        self.fc1 = nn.Linear(4, classes)

    def forward(self, x, extract_embed=False):
        return self.fc1(torch.relu(self.conv(x.permute(0, 2, 1, 3, 4))).mean(dim=(2, 3, 4)))


class Crashes(Dataset):
    """Raises at the given read, like a killed process in the middle of an epoch."""

    def __init__(self, data, crash_at):
        self.data, self.crash_at, self.reads = data, crash_at, 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.reads += 1
        if self.reads == self.crash_at:
            raise KeyboardInterrupt('simulated crash')
        return self.data[index]


def images(n=8, classes=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 3, 8, 8, generator=generator), torch.arange(n) % classes)


def clips(n=8, classes=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return TensorDataset(torch.randn(n, 4, 3, 8, 8, generator=generator), torch.arange(n) % classes)


def loaders(train, val, batch_size=2):
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size)


def make_trainer(tmp_path, options=None, model=None, epochs=3, lr=1e-2, mode='cls', train=None, val=None, seed=0, error_model=False):
    torch.manual_seed(seed)
    train_loader, val_loader = loaders(train if train is not None else images(), val if val is not None else images(4, seed=1))
    trainer = Trainer(model if model is not None else TinyNet(), train_loader, val_loader, lr, mode, epochs, tmp_path,
                      training_options=options)
    trainer.fit_error_model = error_model
    return trainer


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_same_state(left, right):
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def read_run(path):
    return json.loads((Path(path) / RUN_RECORD_FILENAME).read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# Defaults keep today's behaviour
# ---------------------------------------------------------------------------
def test_default_options_match_a_plain_adam_loop(tmp_path):
    torch.manual_seed(5)
    train_loader, val_loader = loaders(images(), images(4, seed=1))
    trainer = Trainer(TinyNet(), train_loader, val_loader, 1e-2, 'cls', 2, tmp_path)
    trainer.fit_error_model = False
    trainer.train()

    # What the Trainer did before these features existed, including the validation pass per epoch.
    torch.manual_seed(5)
    train_loader, val_loader = loaders(images(), images(4, seed=1))
    model = TinyNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for _ in range(2):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                criterion(model(inputs.to(DEVICE)), labels.to(DEVICE))
    assert_same_state(cpu_state(trainer.model), cpu_state(model))
    assert not (tmp_path / CHECKPOINT_FILENAME).exists()
    assert not (tmp_path / BEST_WEIGHTS_FILENAME).exists()


def test_every_flag_default_equals_the_dataclass_default():
    args = DeepTuneVisionOptions(RunType.TRAIN, args=['--out', 'unused'])
    assert TrainingOptions.from_args(args) == TrainingOptions()
    assert TrainingOptions.from_args(args).non_default() == []


def test_from_args_on_an_object_without_the_flags_gives_defaults():
    from unittest.mock import MagicMock
    assert TrainingOptions.from_args(MagicMock()) == TrainingOptions()
    assert TrainingOptions.from_args(types.SimpleNamespace(precision='bf16')).precision == 'bf16'


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('precision', ['bf16', 'fp16'])
def test_precision_is_a_noop_with_a_log_line_on_cpu(precision):
    dtype, scaler, line = tb.resolve_precision(precision, torch.device('cpu'))
    assert dtype is None and scaler is False
    assert precision in line and 'no effect on cpu' in line


def test_precision_choices_on_cuda_without_needing_a_gpu():
    assert tb.resolve_precision('no', 'cuda') == (None, False, None)
    dtype, scaler, _ = tb.resolve_precision('bf16', 'cuda', bf16_supported=True)
    assert dtype == torch.bfloat16 and scaler is False
    dtype, scaler, _ = tb.resolve_precision('fp16', 'cuda')
    assert dtype == torch.float16 and scaler is True
    dtype, scaler, line = tb.resolve_precision('bf16', 'cuda', bf16_supported=False)
    assert dtype is None and scaler is False and 'not supported' in line


def test_trainer_logs_the_cpu_noop_and_trains(tmp_path, monkeypatch, caplog):
    import trainers.vision.trainer as trainer_module
    if DEVICE.type != 'cpu':
        pytest.skip('Checks the CPU no-op; this run has a GPU.')
    monkeypatch.setattr(trainer_module, 'DEVICE', torch.device('cpu'))
    with caplog.at_level(logging.INFO):
        trainer = make_trainer(tmp_path, TrainingOptions(precision='fp16'), epochs=1)
    assert 'no effect on cpu' in caplog.text
    assert trainer.scaler.is_enabled() is False
    trainer.train()
    assert read_run(tmp_path)['precision_active'] == 'float32'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Needs CUDA: autocast only runs on the GPU.')
@pytest.mark.parametrize('precision,dtype,scaler', [('bf16', 'bfloat16', False), ('fp16', 'float16', True)])
def test_autocast_on_gpu(tmp_path, precision, dtype, scaler):
    trainer = make_trainer(tmp_path, TrainingOptions(precision=precision), epochs=1)
    assert trainer.scaler.is_enabled() is scaler
    trainer.train()
    assert read_run(tmp_path)['precision_active'] == dtype


class SpyScaler:
    """A GradScaler double: follows the call order and can skip a step like a real one on inf gradients."""

    def __init__(self, skip_steps=()):
        self.calls, self.skip_steps, self.scale_value, self.steps = [], set(skip_steps), 1024.0, 0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        self.calls.append('scale')
        return loss * 2

    def step(self, optimizer):
        self.calls.append('step')
        self.steps += 1
        self.skipped = self.steps in self.skip_steps
        if not self.skipped:
            optimizer.step()

    def update(self):
        self.calls.append('update')
        if self.skipped:
            self.scale_value /= 2


def test_scaler_order_and_a_skipped_step_moves_neither_the_scheduler_nor_the_ema(tmp_path):
    options = TrainingOptions(lr_scheduler='cosine', ema_decay=0.9, grad_accum_steps=2)
    # 8 samples, batch 2, windows of 2: 2 updates per epoch, 2 epochs, so 4 steps.
    trainer = make_trainer(tmp_path, options, epochs=2)
    trainer.scaler = SpyScaler(skip_steps={2})
    trainer.train()
    assert trainer.scaler.calls[:4] == ['scale', 'scale', 'step', 'update']
    assert trainer.scaler.steps == 4
    assert trainer.global_updates == 3
    assert trainer.scheduler.last_epoch == 3
    assert trainer.ema.updates == 3


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
RESUME_CASES = {
    'plain': dict(),
    'everything': dict(lr_scheduler='cosine', warmup_epochs=1.0, grad_accum_steps=3, label_smoothing=0.1, mixup_alpha=0.4,
                       cutmix_alpha=0.4, mix_prob=0.8, layer_decay=0.8, ema_decay=0.9, select_best=True),
    'early_stopping': dict(early_stopping_patience=10, lr_scheduler='step', lr_step_size=1, lr_gamma=0.5),
}


@pytest.mark.parametrize('case', RESUME_CASES)
@pytest.mark.parametrize('kind', ['images', 'clips'])
def test_resumed_run_continues_exactly_like_an_uninterrupted_one(tmp_path, case, kind):
    data, net = (images, TinyNet) if kind == 'images' else (clips, TinyClipNet)
    settings = RESUME_CASES[case]
    epochs = 4
    # Same generators and so the same model at the start of both runs.
    def start():
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)

    # A) uninterrupted
    start()
    whole = make_trainer(tmp_path / 'whole', TrainingOptions(checkpoint_every=1, **settings), net(), epochs,
                         train=data(), val=data(4, seed=1))
    whole.train()
    whole_rng = (torch.get_rng_state(), random.getstate(), np.random.get_state()[1].copy())

    # B) crashes in the middle of epoch 3, then resumes with a fresh model and fresh generators
    start()
    crashing = Crashes(data(), crash_at=8 * 2 + 3)
    first = make_trainer(tmp_path / 'first', TrainingOptions(checkpoint_every=1, **settings), net(), epochs,
                         train=crashing, val=data(4, seed=1))
    with pytest.raises(KeyboardInterrupt):
        first.train()
    checkpoint = tmp_path / 'first' / CHECKPOINT_FILENAME
    assert torch.load(checkpoint, weights_only=True)['epoch'] == 2

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    second = make_trainer(tmp_path / 'second', TrainingOptions(resume=str(checkpoint), checkpoint_every=1, **settings),
                          net(), epochs, train=data(), val=data(4, seed=1), seed=999)
    second.train()

    assert_same_state(cpu_state(whole.model), cpu_state(second.model))
    assert whole.global_updates == second.global_updates
    assert whole._current_lr() == second._current_lr()
    for group_a, group_b in zip(whole.optimizer.state_dict()['state'].values(), second.optimizer.state_dict()['state'].values()):
        for key in group_a:
            assert torch.equal(torch.as_tensor(group_a[key]).cpu(), torch.as_tensor(group_b[key]).cpu()), key
    # The generators end up in the same place too, so anything after training also matches.
    assert torch.equal(torch.get_rng_state(), whole_rng[0])
    assert random.getstate() == whole_rng[1]
    assert np.array_equal(np.random.get_state()[1], whole_rng[2])
    log_a = pd.read_csv(tmp_path / 'whole' / 'training_log.csv')
    log_b = pd.read_csv(tmp_path / 'second' / 'training_log.csv')
    pd.testing.assert_frame_equal(log_a, log_b)
    rows_a = pd.read_csv(tmp_path / 'whole' / tb.METRICS_FILENAME)
    rows_b = pd.read_csv(tmp_path / 'second' / tb.METRICS_FILENAME)
    assert list(rows_b['epoch']) == [1, 2, 3, 4]
    for column in ('train_loss', 'val_loss', 'lr', 'optimizer_updates'):
        assert rows_a[column].tolist() == rows_b[column].tolist(), column
    if settings.get('select_best'):
        assert whole.best_epoch == second.best_epoch
        assert read_run(tmp_path / 'second')['selection']['best_epoch'] == whole.best_epoch
    assert read_run(tmp_path / 'second')['resumed_from'] == str(checkpoint)


def test_resume_restores_all_random_generators(tmp_path):
    trainer = make_trainer(tmp_path / 'a', TrainingOptions(checkpoint_every=1), epochs=1)
    trainer.train()
    after = (random.random(), np.random.random(), torch.rand(1).item())
    trainer_b = make_trainer(tmp_path / 'b', TrainingOptions(resume=str(tmp_path / 'a' / CHECKPOINT_FILENAME)), epochs=1, seed=3)
    random.seed(1)
    np.random.seed(1)
    trainer_b.train()  # nothing left to train, so the generators stay where the checkpoint left them
    assert (random.random(), np.random.random(), torch.rand(1).item()) == after


def test_checkpoint_is_written_every_n_epochs_and_after_the_last(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(checkpoint_every=2), epochs=3)
    seen = []
    original = trainer._save_checkpoint
    trainer._save_checkpoint = lambda: (seen.append(trainer.completed_epochs), original())
    trainer.train()
    assert seen == [2, 3]
    state = torch.load(tmp_path / CHECKPOINT_FILENAME, weights_only=True)
    expected = {'model', 'optimizer', 'scheduler', 'scaler', 'ema', 'epoch', 'global_updates', 'rng', 'best_metric',
                'best_epoch', 'epochs_without_improvement', 'options', 'data_identity'}
    assert expected <= set(state)
    assert set(state['rng']) == {'python', 'numpy', 'torch', 'cuda'}
    assert not list(tmp_path.glob('*.tmp'))


def test_num_epochs_can_be_extended_on_resume(tmp_path):
    first = make_trainer(tmp_path / 'a', TrainingOptions(checkpoint_every=1), epochs=2)
    first.train()
    before = cpu_state(first.model)
    second = make_trainer(tmp_path / 'b', TrainingOptions(resume=str(tmp_path / 'a' / CHECKPOINT_FILENAME)), epochs=3)
    second.train()
    assert second.completed_epochs == 3
    assert any(not torch.equal(before[key], value) for key, value in cpu_state(second.model).items())


def test_resume_warns_when_settings_change(tmp_path, caplog):
    make_trainer(tmp_path / 'a', TrainingOptions(checkpoint_every=1), epochs=1).train()
    trainer = make_trainer(tmp_path / 'b', TrainingOptions(resume=str(tmp_path / 'a' / CHECKPOINT_FILENAME), label_smoothing=0.1),
                           epochs=1, lr=1e-3)
    with caplog.at_level(logging.WARNING):
        trainer.train()
    assert 'label_smoothing: 0.0 -> 0.1' in caplog.text and 'learning_rate: 0.01 -> 0.001' in caplog.text


def test_resume_refuses_other_data_unless_told_otherwise(tmp_path, caplog):
    make_trainer(tmp_path / 'a', TrainingOptions(checkpoint_every=1), epochs=1).train()
    checkpoint = str(tmp_path / 'a' / CHECKPOINT_FILENAME)
    with pytest.raises(ValueError, match='the data differ'):
        make_trainer(tmp_path / 'b', TrainingOptions(resume=checkpoint), train=images(12), epochs=2).train()
    # Same size, other file content.
    data = tmp_path / 'train.parquet'
    data.write_bytes(b'one split')
    trainer = make_trainer(tmp_path / 'c', TrainingOptions(checkpoint_every=1), epochs=1)
    trainer.data_identity['train_file'] = tb.data_fingerprint(data)
    trainer.train()
    data.write_bytes(b'another split')
    checkpoint = str(tmp_path / 'c' / CHECKPOINT_FILENAME)
    other = make_trainer(tmp_path / 'd', TrainingOptions(resume=checkpoint), epochs=2)
    other.data_identity['train_file'] = tb.data_fingerprint(data)
    with pytest.raises(ValueError, match='train_file'):
        other.train()
    allowed = make_trainer(tmp_path / 'e', TrainingOptions(resume=checkpoint, resume_allow_data_change=True), epochs=2)
    allowed.data_identity['train_file'] = tb.data_fingerprint(data)
    with caplog.at_level(logging.WARNING):
        allowed.train()
    assert allowed.completed_epochs == 2 and 'the data differ' in caplog.text


def test_resume_gives_clear_errors(tmp_path):
    first = make_trainer(tmp_path / 'a', TrainingOptions(checkpoint_every=1), epochs=1)
    first.train()
    checkpoint = str(tmp_path / 'a' / CHECKPOINT_FILENAME)
    weights = tmp_path / 'weights.pth'
    torch.save(first.model.state_dict(), weights)
    with pytest.raises(ValueError, match='not a DeepTune training checkpoint'):
        make_trainer(tmp_path / 'b', TrainingOptions(resume=str(weights))).train()
    junk = tmp_path / 'junk.pt'
    junk.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='could not read'):
        make_trainer(tmp_path / 'c', TrainingOptions(resume=str(junk))).train()
    with pytest.raises(ValueError, match='does not fit this model'):
        make_trainer(tmp_path / 'd', TrainingOptions(resume=checkpoint), model=TinyNet(classes=3)).train()
    with pytest.raises(ValueError, match='no EMA state'):
        make_trainer(tmp_path / 'e', TrainingOptions(resume=checkpoint, ema_decay=0.9)).train()
    with pytest.raises(ValueError, match='no checkpoint file'):
        make_trainer(tmp_path / 'f', TrainingOptions(resume=str(tmp_path / 'missing.pt')))


# ---------------------------------------------------------------------------
# Early stopping and best-weights reload
# ---------------------------------------------------------------------------
def scripted(trainer, scores):
    """Replace validation with fixed scores and remember the weights at each pass."""
    snapshots, script = [], iter(scores)

    def validate():
        snapshots.append(cpu_state(trainer.model))
        return next(script)

    trainer.validate = validate
    return snapshots


def test_early_stopping_stops_on_patience_and_reloads_the_best_weights(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(early_stopping_patience=2), epochs=10)
    snapshots = scripted(trainer, [(1.0, 50.0), (0.5, 60.0), (0.7, 70.0), (0.6, 80.0), (0.1, 90.0)])
    trainer.train()
    assert trainer.completed_epochs == 4 and trainer.stopped_early and trainer.best_epoch == 2
    assert_same_state(cpu_state(trainer.model), snapshots[1])
    assert any(not torch.equal(snapshots[1][k], snapshots[3][k]) for k in snapshots[1])
    selection = read_run(tmp_path)['selection']
    assert selection == {'selection_set': 'validation', 'validation_used_for_model_selection': True,
                         'selection_metric': 'val_loss', 'best_epoch': 2, 'best_metric': 0.5, 'epochs_completed': 4,
                         'stopped_early': True, 'early_stopping_patience': 2}
    assert pd.read_csv(tmp_path / tb.METRICS_FILENAME)['is_best'].tolist() == [True, True, False, False]


def test_selection_on_accuracy_and_min_delta(tmp_path):
    trainer = make_trainer(tmp_path / 'acc', TrainingOptions(select_best=True, selection_metric='val_accuracy'), epochs=3)
    scripted(trainer, [(1.0, 50.0), (2.0, 90.0), (0.1, 60.0)])
    trainer.train()
    assert trainer.best_epoch == 2 and trainer.completed_epochs == 3

    trainer = make_trainer(tmp_path / 'delta', TrainingOptions(select_best=True, early_stopping_min_delta=0.1), epochs=3)
    scripted(trainer, [(1.0, 0.0), (0.95, 0.0), (0.85, 0.0)])
    trainer.train()
    assert trainer.best_epoch == 3


def test_nan_validation_loss_never_counts_as_best(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(select_best=True), epochs=3)
    snapshots = scripted(trainer, [(float('nan'), 0.0), (1.0, 0.0), (float('nan'), 0.0)])
    trainer.train()
    assert trainer.best_epoch == 2
    assert_same_state(cpu_state(trainer.model), snapshots[1])


def test_error_model_is_fitted_on_the_reloaded_best_weights_and_records_selection(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(early_stopping_patience=1), epochs=5, error_model=True)
    snapshots = scripted(trainer, [(1.0, 0.0), (0.2, 0.0), (0.9, 0.0)])
    trainer.train()
    fit = json.loads((tmp_path / 'error_model_fit.json').read_text(encoding='utf-8'))
    assert fit['weights_sha256'] == weights_fingerprint(snapshots[1])
    assert fit['weights_sha256'] != weights_fingerprint(snapshots[2])
    assert fit['validation_used_for_model_selection'] is True
    assert fit['selection_metric'] == 'val_loss' and fit['best_epoch'] == 2
    # What train.py would save next is what the error model describes.
    assert weights_fingerprint(trainer.model.state_dict()) == fit['weights_sha256']


def test_error_model_without_selection_records_that_too(tmp_path):
    trainer = make_trainer(tmp_path, epochs=1, error_model=True)
    trainer.train()
    fit = json.loads((tmp_path / 'error_model_fit.json').read_text(encoding='utf-8'))
    assert fit['validation_used_for_model_selection'] is False and fit['best_epoch'] is None


def test_resume_of_an_early_stopped_run_does_not_train_on(tmp_path):
    first = make_trainer(tmp_path / 'a', TrainingOptions(early_stopping_patience=1, checkpoint_every=1), epochs=5)
    snapshots = scripted(first, [(0.5, 0.0), (0.9, 0.0)])
    first.train()
    assert first.completed_epochs == 2
    second = make_trainer(tmp_path / 'b', TrainingOptions(early_stopping_patience=1,
                                                          resume=str(tmp_path / 'a' / CHECKPOINT_FILENAME)), epochs=5)
    second.train()
    assert second.completed_epochs == 2 and second.stopped_early and second.best_epoch == 1
    assert_same_state(cpu_state(second.model), snapshots[0])


def test_checkpoint_carries_its_own_best_weights(tmp_path):
    """The best file of the crashed run may be newer than the checkpoint; the checkpoint's own copy counts."""
    first = make_trainer(tmp_path / 'a', TrainingOptions(select_best=True, checkpoint_every=1), epochs=2)
    snapshots = scripted(first, [(0.5, 0.0), (0.9, 0.0)])
    first.train()
    second = make_trainer(tmp_path / 'b', TrainingOptions(select_best=True, resume=str(tmp_path / 'a' / CHECKPOINT_FILENAME)), epochs=2)
    second.train()
    assert_same_state(cpu_state(second.model), snapshots[0])
    assert (tmp_path / 'b' / BEST_WEIGHTS_FILENAME).is_file()


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------
def _lr_trace(tmp_path, options, epochs=4, n=8, batch=2):
    trainer = make_trainer(tmp_path, options, epochs=epochs, lr=0.1, train=images(n))
    trace = []
    original_step = trainer.optimizer.step

    def step(*args, **kwargs):
        trace.append(trainer.optimizer.param_groups[0]['lr'])
        return original_step(*args, **kwargs)

    trainer.optimizer.step = step
    trainer.train()
    return trace, trainer


def test_cosine_matches_torch_cosine_annealing(tmp_path):
    trace, _ = _lr_trace(tmp_path, TrainingOptions(lr_scheduler='cosine', min_lr=0.01))
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    oracle = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=0.01)
    expected = []
    for _ in range(16):
        expected.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        oracle.step()
    assert trace == pytest.approx(expected, rel=1e-9)


def test_step_schedule_and_warmup(tmp_path):
    trace, _ = _lr_trace(tmp_path / 'step', TrainingOptions(lr_scheduler='step', lr_step_size=1, lr_gamma=0.5), epochs=3)
    assert trace == pytest.approx([0.1] * 4 + [0.05] * 4 + [0.025] * 4)
    trace, _ = _lr_trace(tmp_path / 'warm', TrainingOptions(warmup_epochs=1.0), epochs=2)
    assert trace == pytest.approx([0.025, 0.05, 0.075, 0.1, 0.1, 0.1, 0.1, 0.1])
    trace, _ = _lr_trace(tmp_path / 'both', TrainingOptions(warmup_epochs=0.5, lr_scheduler='cosine'), epochs=2)
    assert trace[:2] == pytest.approx([0.05, 0.1])
    assert trace[2:] == pytest.approx([0.1 * 0.5 * (1 + math.cos(math.pi * k / 6)) for k in range(6)])


def test_warmup_longer_than_the_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match='leaves no epochs'):
        make_trainer(tmp_path, TrainingOptions(warmup_epochs=3.0), epochs=3)


def test_schedule_counts_optimizer_updates_not_batches(tmp_path):
    trace, trainer = _lr_trace(tmp_path, TrainingOptions(lr_scheduler='step', lr_step_size=1, lr_gamma=0.5, grad_accum_steps=3), epochs=2)
    assert trainer.updates_per_epoch == 2  # 4 batches in windows of 3: one full, one partial
    assert trace == pytest.approx([0.1, 0.1, 0.05, 0.05])


def test_lr_is_in_the_metrics_csv(tmp_path):
    _, trainer = _lr_trace(tmp_path, TrainingOptions(lr_scheduler='step', lr_step_size=1, lr_gamma=0.5), epochs=2)
    assert pd.read_csv(tmp_path / tb.METRICS_FILENAME)['lr'].tolist() == pytest.approx([0.05, 0.025])


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)

    def forward(self, x, extract_embed=False):
        return self.fc1(x)


@pytest.mark.parametrize('n,batch,accum,windows', [
    (8, 2, 2, [(0, 4), (4, 8)]),        # full windows only
    (10, 2, 3, [(0, 6), (6, 10)]),      # a shorter last window
    (7, 2, 2, [(0, 4), (4, 7)]),        # the last batch is smaller
    (5, 2, 8, [(0, 5)]),                # the window never fills up
])
def test_accumulated_gradient_equals_the_gradient_of_the_big_batch(tmp_path, n, batch, accum, windows):
    generator = torch.Generator().manual_seed(n)
    x, y = torch.randn(n, 4, generator=generator), torch.arange(n) % 3
    torch.manual_seed(0)
    trainer = Trainer(Linear(), DataLoader(TensorDataset(x, y), batch_size=batch), DataLoader(TensorDataset(x, y), batch_size=batch),
                      0.0, 'cls', 1, tmp_path, training_options=TrainingOptions(grad_accum_steps=accum))
    trainer.fit_error_model = False
    grads = []
    original_step = trainer.optimizer.step

    def step(*args, **kwargs):
        grads.append([p.grad.clone() for p in trainer.model.parameters()])
        return original_step(*args, **kwargs)

    trainer.optimizer.step = step
    trainer.train()  # lr 0: the weights stay put, so every window can be checked against the same point

    assert len(grads) == len(windows)
    assert trainer.global_updates == len(windows)
    for (start, end), measured in zip(windows, grads):
        oracle = copy.deepcopy(trainer.model)
        oracle.zero_grad()
        nn.functional.cross_entropy(oracle(x[start:end].to(DEVICE)), y[start:end].to(DEVICE)).backward()
        for mine, expected in zip(measured, oracle.parameters()):
            assert torch.allclose(mine, expected.grad, atol=1e-6), (start, end)


def test_accumulation_reports_the_true_loss_and_validates_options(tmp_path):
    x, y = torch.randn(8, 4), torch.arange(8) % 3
    trainer = Trainer(Linear(), DataLoader(TensorDataset(x, y), batch_size=2), DataLoader(TensorDataset(x, y), batch_size=2),
                      0.0, 'cls', 1, tmp_path, training_options=TrainingOptions(grad_accum_steps=4))
    trainer.fit_error_model = False
    trainer.train()
    with torch.no_grad():
        expected = nn.functional.cross_entropy(trainer.model(x.to(DEVICE)), y.to(DEVICE)).item()
    assert pd.read_csv(tmp_path / tb.METRICS_FILENAME)['train_loss'][0] == pytest.approx(expected, rel=1e-5)
    with pytest.raises(ValueError, match='grad_accum_steps'):
        TrainingOptions(grad_accum_steps=0).validate()


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------
@pytest.fixture(params=['json', 'yaml'])
def config_path(request, tmp_path):
    values = {'out': str(tmp_path / 'from_config'), 'num_epochs': 7, 'learning_rate': 0.5, 'freeze-backbone': True,
              'precision': 'bf16', 'trainable_params': '100k', 'lr_scheduler': 'cosine'}
    path = tmp_path / f'run.{request.param}'
    if request.param == 'json':
        path.write_text(json.dumps(values), encoding='utf-8')
    else:
        yaml = pytest.importorskip('yaml')
        path.write_text(yaml.safe_dump(values), encoding='utf-8')
    return path


def test_config_supplies_defaults_and_cli_flags_win(config_path, tmp_path):
    args = DeepTuneVisionOptions(RunType.TRAIN, args=['--config', str(config_path), '--num_epochs', '3', '--lr_scheduler', 'step'])
    assert args.num_epochs == 3 and args.lr_scheduler == 'step'  # explicit flags win
    assert args.learning_rate == 0.5 and args.precision == 'bf16' and args.freeze_backbone is True
    assert args.trainable_params == 100_000  # the flag's own parser ran on the value
    assert args.out == (tmp_path / 'from_config').resolve()  # --out is required, but the file provides it
    assert args.config == str(config_path)


def test_config_values_are_saved_with_the_run(config_path, tmp_path):
    args = DeepTuneVisionOptions(RunType.TRAIN, args=['--config', str(config_path)])
    args.save_args(tmp_path)
    saved = json.loads((tmp_path / 'cli_arguments.json').read_text(encoding='utf-8'))
    assert saved['precision'] == 'bf16' and saved['num_epochs'] == 7 and saved['config'] == str(config_path)
    for name in ('seed', 'resume', 'checkpoint_every', 'early_stopping_patience', 'grad_accum_steps', 'label_smoothing',
                 'mixup_alpha', 'layer_decay', 'ema_decay'):
        assert name in saved, name


@pytest.mark.parametrize('content,message', [
    ({'no_such_flag': 1}, 'unknown key'),
    ({'precision': 'fp8'}, 'precision must be one of'),
    ({'num_epochs': 'many'}, 'invalid value'),
    ({'freeze_backbone': 'yes'}, 'true or false'),
    ([1, 2], 'must hold a mapping'),
])
def test_bad_config_stops_with_a_clear_message(tmp_path, capsys, content, message):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(content), encoding='utf-8')
    with pytest.raises(SystemExit):
        DeepTuneVisionOptions(RunType.TRAIN, args=['--out', 'x', '--config', str(path)])
    assert message in capsys.readouterr().err


def test_config_errors_for_missing_file_and_wrong_type(tmp_path):
    with pytest.raises(ValueError, match='no such file'):
        load_config_file(tmp_path / 'nope.json')
    other = tmp_path / 'run.toml'
    other.write_text('x = 1', encoding='utf-8')
    with pytest.raises(ValueError, match='.json, .yaml or .yml'):
        load_config_file(other)
    broken = tmp_path / 'run.json'
    broken.write_text('{', encoding='utf-8')
    with pytest.raises(ValueError, match='not valid JSON'):
        load_config_file(broken)


def test_one_config_serves_train_and_eval_commands(tmp_path):
    path = tmp_path / 'shared.json'
    path.write_text(json.dumps({'batch_size': 4, 'precision': 'bf16', 'ema_decay': 0.9}), encoding='utf-8')
    args = DeepTuneVisionOptions(RunType.EVAL, args=['--out', 'x', '--config', str(path)])
    assert args.batch_size == 4 and not hasattr(args, 'precision')


# ---------------------------------------------------------------------------
# Logging and the run record
# ---------------------------------------------------------------------------
def test_metrics_csv_is_always_written(tmp_path):
    make_trainer(tmp_path, epochs=2).train()
    rows = pd.read_csv(tmp_path / tb.METRICS_FILENAME)
    assert rows['epoch'].tolist() == [1, 2]
    for column in ('train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr', 'seconds', 'peak_gpu_allocated_mb', 'peak_process_rss_mb'):
        assert column in rows.columns, column


def test_regression_metrics_have_no_accuracy(tmp_path):
    data = TensorDataset(torch.randn(8, 4), torch.randn(8))
    trainer = Trainer(Linear(), DataLoader(data, batch_size=2), DataLoader(data, batch_size=2), 1e-3, 'reg', 1, tmp_path)
    # Linear's three outputs broadcast against [B, 1]; only the bookkeeping matters here.
    trainer.fit_error_model = False
    with pytest.warns(UserWarning, match='target size'):
        trainer.train()
    rows = pd.read_csv(tmp_path / tb.METRICS_FILENAME)
    assert rows['val_accuracy'].isna().all() and rows['val_loss'].notna().all()


def test_missing_experiment_logger_falls_back_to_csv(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'mlflow', None)
    with caplog.at_level(logging.WARNING):
        make_trainer(tmp_path, TrainingOptions(experiment_logger='mlflow'), epochs=1).train()
    assert 'mlflow is not installed' in caplog.text
    assert (tmp_path / tb.METRICS_FILENAME).is_file()


def test_mlflow_gets_params_and_metrics_when_installed(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        start_run=lambda run_name=None: calls.append(('start', run_name)),
        log_params=lambda params: calls.append(('params', params)),
        log_metrics=lambda metrics, step=None: calls.append(('metrics', step, metrics)),
        end_run=lambda: calls.append(('end',)),
    )
    monkeypatch.setitem(sys.modules, 'mlflow', fake)
    make_trainer(tmp_path, TrainingOptions(experiment_logger='mlflow', experiment_name='exp1'), epochs=2).train()
    assert calls[0] == ('start', 'exp1')
    assert calls[1][0] == 'params' and calls[1][1]['learning_rate'] == 0.01
    metric_calls = [c for c in calls if c[0] == 'metrics']
    assert [c[1] for c in metric_calls] == [1, 2] and 'val_loss' in metric_calls[0][2]
    assert calls[-1] == ('end',)


def test_wandb_gets_metrics_when_installed(tmp_path, monkeypatch):
    calls = []
    run = types.SimpleNamespace(log=lambda metrics, step=None: calls.append(('log', step)), finish=lambda: calls.append(('finish',)))
    fake = types.SimpleNamespace(init=lambda **kwargs: (calls.append(('init', kwargs['name'])), run)[1])
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    make_trainer(tmp_path, TrainingOptions(experiment_logger='wandb', experiment_name='exp2'), epochs=1).train()
    assert calls == [('init', 'exp2'), ('log', 1), ('finish',)]


def test_a_failing_experiment_logger_does_not_stop_training(tmp_path, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise ConnectionError('no server')
    monkeypatch.setitem(sys.modules, 'mlflow', types.SimpleNamespace(start_run=lambda run_name=None: None, log_params=lambda p: None,
                                                                     log_metrics=broken, end_run=lambda: None))
    with caplog.at_level(logging.WARNING):
        trainer = make_trainer(tmp_path, TrainingOptions(experiment_logger='mlflow'), epochs=2)
        trainer.train()
    assert trainer.completed_epochs == 2 and 'no server' in caplog.text
    assert len(pd.read_csv(tmp_path / tb.METRICS_FILENAME)) == 2


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def test_seed_everything_covers_all_generators():
    tb.seed_everything(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    tb.seed_everything(123)
    assert (random.random(), np.random.random(), torch.rand(1).item()) == first
    tb.seed_everything(124)
    assert (random.random(), np.random.random(), torch.rand(1).item()) != first


def test_seed_worker_follows_the_worker_torch_seed(monkeypatch):
    torch.manual_seed(77)
    tb.seed_worker(0)
    first = (random.random(), np.random.random())
    torch.manual_seed(77)
    tb.seed_worker(0)
    assert (random.random(), np.random.random()) == first
    torch.manual_seed(78)
    tb.seed_worker(0)
    assert (random.random(), np.random.random()) != first


def test_worker_seeding_is_only_installed_where_it_applies():
    data = images(2)
    loader = DataLoader(data, batch_size=1)
    tb.install_worker_seeding(loader)
    assert loader.worker_init_fn is None
    loader = DataLoader(data, batch_size=1, num_workers=1)
    tb.install_worker_seeding(loader)
    assert loader.worker_init_fn is tb.seed_worker
    mine = lambda _: None
    loader = DataLoader(data, batch_size=1, num_workers=1, worker_init_fn=mine)
    tb.install_worker_seeding(loader)
    assert loader.worker_init_fn is mine


def test_seed_is_recorded_with_and_without_the_flag(tmp_path):
    tb.seed_everything(9)
    make_trainer(tmp_path / 'with', TrainingOptions(seed=9), epochs=1, seed=9).train()
    run = read_run(tmp_path / 'with')
    assert run['seed_requested'] == 9 and run['seed_effective'] == 9
    torch.manual_seed(31)
    train_loader, val_loader = loaders(images(), images(4, seed=1))
    trainer = Trainer(TinyNet(), train_loader, val_loader, 1e-2, 'cls', 1, tmp_path / 'without')
    trainer.fit_error_model = False
    trainer.train()
    run = read_run(tmp_path / 'without')
    assert run['seed_requested'] is None and run['seed_effective'] == 31


# ---------------------------------------------------------------------------
# Train scripts pass the options on
# ---------------------------------------------------------------------------
def _script_kwargs(tmp_path, args):
    return dict(train_df=tmp_path / 't.parquet', val_df=tmp_path / 'v.parquet', out=tmp_path / 'o', freeze_backbone=False,
                use_peft=False, fixed_seed=0, mode='cls', model_version='resnet18', batch_size=2, num_epochs=1,
                learning_rate=1e-3, added_layers=1, num_classes=2, embed_size=4, model_str='m', args=args)


@pytest.mark.parametrize('module_name', ['trainers.vision.train', 'trainers.video.train'])
def test_train_scripts_seed_and_hand_the_options_to_the_trainer(tmp_path, module_name):
    import importlib
    from unittest.mock import MagicMock, patch
    module = importlib.import_module(module_name)
    args = types.SimpleNamespace(seed=21, checkpoint_every=1, grad_accum_steps=2, no_error_model=True, save_args=MagicMock(),
                                 model_architecture='resnet', video_backend='auto', temporal_sampling='uniform', clip_stride=None,
                                 clip_duration_sec=None, motion_source='auto', test_clips=1, num_workers=0, video_augment=False,
                                 video_flip=False, video_speed_perturb=0.0)
    dataset = 'ParquetImageDataset' if 'vision' in module_name else 'ParquetVideoDataset'
    builder = 'get_model_cls' if 'vision' in module_name else 'build_video_model'
    with patch.object(module, dataset), patch.object(module, builder), patch.object(module, 'Trainer') as trainer_class, \
            patch.object(module, 'DataLoader') as loader_class, patch.object(module, 'seed_everything') as seeding:
        module.train(**_script_kwargs(tmp_path, args))
    seeding.assert_called_once_with(21)
    options = trainer_class.call_args.kwargs['training_options']
    assert options.checkpoint_every == 1 and options.grad_accum_steps == 2
    # An exact resume cannot use persistent workers.
    assert all(call.kwargs['persistent_workers'] is False for call in loader_class.call_args_list)


def test_data_fingerprint_sees_content_and_tolerates_missing_files(tmp_path):
    path = tmp_path / 'big.bin'
    path.write_bytes(b'a' * (3 << 20))
    first = tb.data_fingerprint(path)
    assert tb.data_fingerprint(path) == first
    path.write_bytes(b'a' * ((3 << 20) - 1) + b'b')
    assert tb.data_fingerprint(path) != first
    assert tb.data_fingerprint(tmp_path / 'missing.bin') is None


# ---------------------------------------------------------------------------
# Option checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kwargs,mode,message', [
    (dict(label_smoothing=0.1), 'reg', 'label_smoothing'),
    (dict(mixup_alpha=0.2), 'reg', 'mixup_alpha'),
    (dict(selection_metric='val_accuracy'), 'reg', 'val_accuracy'),
    (dict(label_smoothing=1.0), 'cls', 'label_smoothing'),
    (dict(ema_decay=1.0), 'cls', 'ema_decay'),
    (dict(layer_decay=0.0), 'cls', 'layer_decay'),
    (dict(lr_gamma=0.0), 'cls', 'lr_gamma'),
    (dict(precision='fp8'), 'cls', 'precision'),
    (dict(seed=-1), 'cls', 'seed'),
])
def test_bad_option_combinations_are_refused(kwargs, mode, message):
    with pytest.raises(ValueError, match=message):
        TrainingOptions(**kwargs).validate(mode)


# ---------------------------------------------------------------------------
# Expert recipe: label smoothing
# ---------------------------------------------------------------------------
def test_label_smoothing_changes_training_but_not_the_validation_loss(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(label_smoothing=0.2), epochs=1)
    trainer.fit_error_model = False
    trainer.train()
    # validate() reports plain cross-entropy, so scores stay comparable across settings.
    model = trainer.model.eval()
    x, y = zip(*[images(4, seed=1)[i] for i in range(4)])
    with torch.no_grad():
        plain = nn.functional.cross_entropy(model(torch.stack(x).to(DEVICE)), torch.stack(y).to(DEVICE)).item()
    assert trainer.validate()[0] == pytest.approx(plain, rel=1e-5)
    assert trainer.train_criterion.label_smoothing == 0.2
    logits = torch.tensor([[4.0, 0.0]])
    assert trainer.train_criterion(logits, torch.tensor([0])).item() > nn.functional.cross_entropy(logits, torch.tensor([0])).item()


# ---------------------------------------------------------------------------
# Expert recipe: mixup and CutMix
# ---------------------------------------------------------------------------
def _frame_coded_clips(batch=4, frames=5, size=16):
    """Each pixel of a frame holds clip_index * 100 + frame_index."""
    clip = torch.zeros(batch, frames, 3, size, size)
    for b in range(batch):
        for t in range(frames):
            clip[b, t] = b * 100 + t
    return clip


def test_mixup_uses_one_weight_for_all_frames_of_a_clip():
    torch.manual_seed(0)
    clip = _frame_coded_clips()
    labels = torch.arange(4)
    mixed, other, lam = tb.mix_batch(clip, labels, mixup_alpha=1.0)
    assert 0 < lam < 1
    for b in range(4):
        partner = int(other[b])
        for t in range(5):
            expected = lam * (b * 100 + t) + (1 - lam) * (partner * 100 + t)
            assert mixed[b, t].allclose(torch.full_like(mixed[b, t], expected), atol=1e-3)


def test_cutmix_uses_one_box_for_all_frames_and_lambda_is_the_area():
    torch.manual_seed(3)
    clip = _frame_coded_clips()
    labels = torch.arange(4)
    for _ in range(20):
        mixed, other, lam = tb.mix_batch(clip, labels, cutmix_alpha=1.0)
        for b in range(4):
            partner = int(other[b])
            if partner == b:
                continue
            masks = [mixed[b, t, 0] == partner * 100 + t for t in range(5)]
            assert all(torch.equal(masks[0], mask) for mask in masks), 'every frame shares one box'
            assert 1 - masks[0].float().mean().item() == pytest.approx(lam, abs=1e-6)
            rows, cols = masks[0].any(dim=1).nonzero(), masks[0].any(dim=0).nonzero()
            if len(rows):
                area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
                assert area.item() == masks[0].sum().item(), 'the replaced region is one rectangle'


def test_mixing_works_on_images_and_leaves_the_input_alone():
    torch.manual_seed(1)
    x = torch.rand(4, 3, 8, 8)
    original = x.clone()
    mixed, other, lam = tb.mix_batch(x, torch.arange(4), cutmix_alpha=1.0)
    assert mixed.shape == x.shape and other.shape == (4,) and torch.equal(x, original)


def test_mixing_off_or_skipped_returns_the_batch_unchanged():
    x, y = torch.rand(2, 3, 8, 8), torch.tensor([0, 1])
    state = torch.get_rng_state()
    mixed, other, lam = tb.mix_batch(x, y)
    assert mixed is x and other is None and lam == 1.0
    assert torch.equal(torch.get_rng_state(), state), 'off must not draw random numbers'
    seen = [tb.mix_batch(x, y, mixup_alpha=1.0, mix_prob=0.5)[1] is None for _ in range(200)]
    assert 60 < sum(seen) < 140


def test_mixed_training_loss_is_the_weighted_sum_of_both_labels(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(mixup_alpha=1.0), epochs=1)
    outputs = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    a, b = torch.tensor([0, 1]), torch.tensor([1, 0])
    expected = 0.3 * nn.functional.cross_entropy(outputs, a) + 0.7 * nn.functional.cross_entropy(outputs, b)
    assert trainer._training_loss(outputs, a, b, 0.3).item() == pytest.approx(expected.item())


def test_mixup_and_cutmix_train_a_video_model_end_to_end(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(mixup_alpha=0.4, cutmix_alpha=0.4), model=TinyClipNet(), epochs=1,
                           train=clips(), val=clips(4, seed=1))
    trainer.train()
    assert trainer.completed_epochs == 1


# ---------------------------------------------------------------------------
# Expert recipe: layer-wise LR decay
# ---------------------------------------------------------------------------
class Blocks(nn.Module):
    """Shaped like DeepTune's models: backbone under model., head beside it."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module('stem', nn.Conv2d(3, 4, 1))
        self.model.add_module('layer1', nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1)))
        self.model.add_module('layer2', nn.Sequential(nn.Conv2d(4, 4, 1)))
        self.fc1 = nn.Linear(4, 2)

    def forward(self, x, extract_embed=False):
        return self.fc1(self.model(x).mean(dim=(2, 3)))


@pytest.mark.parametrize('name,expected', [
    ('model.layer2.0.conv1.0.0.weight', 'layer2.0'),
    ('peftmodel.base_model.model.layer2.0.conv1.0.0.base_layer.weight', 'layer2.0'),
    ('peftmodel.base_model.model.layer2.0.conv1.0.0.lora_A.default.weight', 'layer2.0'),
    ('model.encoder.layers.encoder_layer_3.mlp.0.weight', 'encoder.layers.encoder_layer_3'),
    ('model.features.3.1.mlp.0.weight', 'features.3.1'),
    ('backbone.model.conv_proj.weight', 'conv_proj'),
    ('model.class_token', 'model.class_token'),
])
def test_layer_key(name, expected):
    assert tb.layer_key(name) == expected


def test_layer_decay_groups_get_smaller_rates_towards_the_input():
    model = Blocks()
    groups = tb.layer_wise_param_groups(model, 1.0, 0.5)
    by_layer = {group['layer']: group['lr'] for group in groups}
    assert by_layer['head'] == 1.0
    rates = [group['lr'] for group in groups]
    assert rates == sorted(rates) and rates[0] < rates[-1]
    grouped = {id(p) for group in groups for p in group['params']}
    assert grouped == {id(p) for p in model.parameters()}


def test_layer_decay_skips_frozen_parameters_and_trains(tmp_path):
    model = Blocks()
    for param in model.model.stem.parameters():
        param.requires_grad = False
    groups = tb.layer_wise_param_groups(model, 1.0, 0.5)
    assert all(not any(p is q for q in model.model.stem.parameters()) for g in groups for p in g['params'])
    trainer = make_trainer(tmp_path, TrainingOptions(layer_decay=0.5, lr_scheduler='cosine'), model=model, epochs=1)
    rates = [group['lr'] for group in trainer.optimizer.param_groups]
    assert rates[-1] == pytest.approx(1e-2) and rates[0] < rates[-1]
    trainer.train()


def test_layer_decay_off_builds_one_group(tmp_path):
    assert len(make_trainer(tmp_path, TrainingOptions(), model=Blocks()).optimizer.param_groups) == 1


# ---------------------------------------------------------------------------
# Expert recipe: EMA
# ---------------------------------------------------------------------------
def test_ema_follows_the_closed_form():
    model = nn.Linear(1, 1, bias=False)
    nn.init.constant_(model.weight, 0.0)
    ema = tb.ModelEma(model, decay=0.9)
    expected = 0.0
    for update, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        nn.init.constant_(model.weight, value)
        decay = min(0.9, (1 + update) / (10 + update))
        ema.update(model)
        expected = decay * expected + (1 - decay) * value
        assert ema.shadow['weight'].item() == pytest.approx(expected)


def test_ema_swap_restores_the_training_weights_and_handles_buffers():
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2))
    ema = tb.ModelEma(model, decay=0.5)
    model(torch.rand(2, 3, 4, 4))
    with torch.no_grad():
        model[0].weight.add_(1.0)
    ema.update(model)
    before = cpu_state(model)
    assert 'num_batches_tracked' in ''.join(ema.shadow)
    with ema.swapped_in(model):
        assert not torch.equal(model[0].weight.detach().cpu(), before['0.weight'])
    assert_same_state(cpu_state(model), before)


def test_ema_does_not_copy_frozen_parameters():
    model = Blocks()
    for param in model.model.parameters():
        param.requires_grad = False
    assert set(tb.ModelEma(model, 0.9).shadow) == {'fc1.weight', 'fc1.bias'}


def test_ema_weights_are_validated_saved_and_hand_on(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(ema_decay=0.9), epochs=2)
    original_validate = trainer.validate
    seen = []

    def validate():
        seen.append(cpu_state(trainer.model))
        return original_validate()

    trainer.validate = validate
    raw_at_end = {}
    original_reload = trainer._reload_final_weights

    def reload():
        raw_at_end.update(cpu_state(trainer.model))
        original_reload()

    trainer._reload_final_weights = reload
    trainer.train()
    final = cpu_state(trainer.model)
    assert_same_state(seen[-1], final)  # validation saw what is handed on
    assert any(not torch.equal(raw_at_end[key], final[key]) for key in final)  # which are not the raw weights


def test_best_selection_with_ema_keeps_the_averaged_weights(tmp_path):
    trainer = make_trainer(tmp_path, TrainingOptions(ema_decay=0.9, select_best=True), epochs=3)
    snapshots = scripted(trainer, [(1.0, 0.0), (0.2, 0.0), (0.7, 0.0)])
    trainer.train()
    assert_same_state(cpu_state(trainer.model), snapshots[1])


# ---------------------------------------------------------------------------
# Peak memory
# ---------------------------------------------------------------------------
def test_peak_memory_on_cpu_has_no_gpu_numbers(tmp_path):
    peak = tb.peak_memory_mb('cpu')
    assert peak['peak_gpu_allocated_mb'] is None and peak['peak_gpu_reserved_mb'] is None
    make_trainer(tmp_path, epochs=1).train()
    assert 'peak_process_rss_mb' in read_run(tmp_path)['peak_memory']


def test_cli_rejects_a_bad_combination_before_any_work(tmp_path, capsys):
    import deeptune
    df = tmp_path / 'df.parquet'
    df.write_bytes(b'')
    with pytest.raises(SystemExit):
        deeptune.main(['--modality', 'tabular', '--out', str(tmp_path), '--df', str(df), '--raw-data', '--seed', '1',
                       '--type', 'classification', '--target', 'x'])
    assert 'only supported for the images and video' in capsys.readouterr().err

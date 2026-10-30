"""A2 follow-ups from the wave-1b review: error model vs --test_clips (V-016),
infer.py falling back to the run's recorded settings (V-017), loader roles and
augmentation never touching evaluation clips (V-019)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

pytest.importorskip('av')

from datasets.video_synth import decode_frame_index, index_coded_clip


# ---------------------------------------------------------------------------
# V-016: the error model is fitted on what evaluation sees
# ---------------------------------------------------------------------------
class _Clips(Dataset):
    """[N clips, T, C, H, W] per item whose clips disagree, so averaging matters."""

    def __init__(self, n, clips, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, clips, 2, 1, 2, 2, generator=g)
        self.y = (self.x.mean(dim=(1, 2, 3, 4, 5)) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _Single(_Clips):
    def __getitem__(self, i):
        return self.x[i, 0], self.y[i]


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x, extract_embed=False):
        return self.fc(x.reshape(len(x), -1))


def test_trainer_fits_the_error_model_on_multi_clip_probabilities(tmp_path):
    from evaluators.error_model import ErrorModel, error_targets, prediction_features
    from src.video.multi_clip import MultiClipModel
    from trainers.vision.trainer import Trainer
    torch.manual_seed(0)
    model = _Head()
    train, val = _Single(32, 3, 0), _Single(30, 3, 1)
    trainer = Trainer(model, DataLoader(train, batch_size=8), DataLoader(val, batch_size=8), 0.01, 'cls', 1, tmp_path)
    multi = _Clips(30, 3, 1)
    trainer.error_fit_loader = DataLoader(multi, batch_size=8)
    trainer.error_fit_model = MultiClipModel(model, 'cls')
    trainer.error_fit_info = {'test_clips': 3}
    trainer.train()
    fit = json.loads((tmp_path / 'error_model_fit.json').read_text())
    assert fit['test_clips'] == 3 and fit['n_samples'] == 30 and fit['val_dataset_size'] == 30

    # Refit by hand on the 3-clip averaged probabilities: same model as the Trainer's.
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(MultiClipModel(model, 'cls')(multi.x), 1).numpy()
    expected = ErrorModel('cls').fit(prediction_features(probs, 'cls'), error_targets(probs, multi.y.numpy(), 'cls'),
                                     num_outputs=2)
    saved = json.loads((tmp_path / 'error_model.json').read_text())
    reference = json.loads(json.dumps(expected.to_dict()))
    for key in ('method', 'knots_x', 'knots_y', 'logistic', 'constant'):
        assert saved[key] == pytest.approx(reference[key]) if isinstance(reference[key], list) else saved[key] == reference[key], key
    # ...and it differs from fitting on the first clip only (the old behaviour), so the hook matters.
    single = torch.softmax(model(multi.x[:, 0]), 1).detach().numpy()
    assert not np.allclose(single.max(1), probs.max(1))


def test_trainer_without_hooks_is_unchanged(tmp_path):
    from trainers.vision.trainer import Trainer
    torch.manual_seed(0)
    val = _Single(20, 1, 1)
    trainer = Trainer(_Head(), DataLoader(_Single(16, 1, 0), batch_size=8), DataLoader(val, batch_size=8), 0.01, 'cls', 1,
                      tmp_path)
    trainer.train()
    fit = json.loads((tmp_path / 'error_model_fit.json').read_text())
    assert fit['n_samples'] == 20 and 'test_clips' not in fit


def test_mismatched_test_clips_warn_at_eval_and_inference(tmp_path):
    from src.video.multi_clip import error_model_clips_mismatch
    ckpt = tmp_path / 'model_weights.pth'
    ckpt.write_bytes(b'')
    assert not error_model_clips_mismatch(ckpt, 3)  # no error model at all
    (tmp_path / 'error_model.json').write_text(json.dumps({'fit_info': {}}))  # old run: fitted on 1 clip
    messages = []
    assert not error_model_clips_mismatch(ckpt, 1, warn=messages.append)
    assert error_model_clips_mismatch(ckpt, 3, warn=messages.append)
    (tmp_path / 'error_model.json').write_text(json.dumps({'fit_info': {'test_clips': 3}}))
    assert not error_model_clips_mismatch(ckpt, 3, warn=messages.append)
    assert error_model_clips_mismatch(ckpt, 1, warn=messages.append)
    assert len(messages) == 2 and 'fitted on 3 clip(s)' in messages[1]


def test_video_train_with_test_clips_records_them_in_the_fit(tmp_path, monkeypatch):
    import torchvision
    import trainers.video.train as video_train
    small = torchvision.transforms.Compose([torchvision.transforms.Resize((16, 16)), torchvision.transforms.ToTensor()])
    monkeypatch.setattr(video_train, 'transformations', small)
    rows = [{'videos': str(index_coded_clip(tmp_path / f'c{i}.mp4', 8 + i)), 'labels': i % 2} for i in range(6)]
    for name in ('train', 'val'):
        pd.DataFrame(rows).to_parquet(tmp_path / f'{name}.parquet')

    class Args:
        test_clips = 2
        no_error_model = False

        def save_args(self, outdir):
            Path(outdir, 'cli_arguments.json').write_text('{}')

    ckpt = video_train.train(train_df=tmp_path / 'train.parquet', val_df=tmp_path / 'val.parquet', out=tmp_path,
                             freeze_backbone=False, use_peft=False, fixed_seed=0, mode='cls', model_version='dummy_video3d',
                             batch_size=3, num_epochs=1, learning_rate=1e-3, added_layers=1, num_classes=2, embed_size=8,
                             model_str='dummy', args=Args(), num_frames=2)
    fit = json.loads((Path(ckpt).parent / 'error_model_fit.json').read_text())
    assert fit['test_clips'] == 2 and fit['n_samples'] == 6


# ---------------------------------------------------------------------------
# V-017: infer.py uses the run's recorded data-layer settings by default
# ---------------------------------------------------------------------------
def _recording_predict(monkeypatch):
    import inference.video as video_inference
    seen = {}

    def fake_predict(df, checkpoint_path, model_version, num_classes, added_layers, embed_size, freeze_backbone, mode,
                     use_peft, batch_size, num_frames=8, pooling='mean', label_mapping=None, video_options=None):
        seen.update(num_frames=num_frames, pooling=pooling, options=video_options)
        return pd.DataFrame({'sample_id': list(df['__sample_id']), 'predicted_class': 0, 'predicted_label': '0',
                             'confidence': 1.0})

    monkeypatch.setattr(video_inference, 'predict_video', fake_predict)
    return seen


def test_infer_cli_falls_back_to_the_runs_recorded_sampling(tmp_path, monkeypatch, capsys):
    import infer
    run = tmp_path / 'trainval_output_x'
    run.mkdir()
    (run / 'model_weights.pth').write_bytes(b'')
    (run / 'cli_arguments.json').write_text(json.dumps({
        'temporal_sampling': 'dense', 'clip_stride': 2, 'clip_duration_sec': None, 'test_clips': 3,
        'video_backend': 'pyav', 'motion_source': 'auto', 'num_frames': 4, 'pooling': 'attention', 'num_workers': 4}))
    clip = index_coded_clip(tmp_path / 'one.mp4', 10)
    seen = _recording_predict(monkeypatch)
    base = ['--modality', 'video', '--checkpoint', str(run / 'model_weights.pth'), '--data', str(clip),
            '--out', str(tmp_path / 'o1'), '--model_version', 'r3d_18']
    assert infer.main(base) == 0
    opts = seen['options']
    assert (opts.temporal_sampling, opts.clip_stride, opts.test_clips, opts.video_backend) == ('dense', 2, 3, 'pyav')
    assert seen['num_frames'] == 4 and seen['pooling'] == 'attention'
    assert opts.num_workers == 'auto'  # a resource setting of the training machine, not of the pipeline
    out = capsys.readouterr().out
    assert 'temporal_sampling=dense' in out and 'clip_stride=2' in out

    # Explicit flags win over the recording.
    assert infer.main([*base[:-2], '--out', str(tmp_path / 'o2'), '--model_version', 'r3d_18',
                       '--temporal_sampling', 'uniform', '--num_frames', '8']) == 0
    assert seen['options'].temporal_sampling == 'uniform' and seen['options'].clip_stride == 2
    assert seen['num_frames'] == 8


def test_infer_cli_without_a_recording_uses_defaults(tmp_path, monkeypatch):
    import infer
    (tmp_path / 'model_weights.pth').write_bytes(b'')
    clip = index_coded_clip(tmp_path / 'one.mp4', 10)
    seen = _recording_predict(monkeypatch)
    assert infer.main(['--modality', 'video', '--checkpoint', str(tmp_path / 'model_weights.pth'), '--data', str(clip),
                       '--out', str(tmp_path / 'o'), '--model_version', 'r3d_18']) == 0
    assert seen['options'].temporal_sampling == 'uniform' and seen['num_frames'] == 8


# ---------------------------------------------------------------------------
# V-019: augmentation never changes evaluation clips; loader roles
# ---------------------------------------------------------------------------
def test_video_augment_never_changes_validation_or_test_clips(tmp_path):
    from datasets.video_datasets import VideoDataOptions, open_video_dataset
    df = pd.DataFrame({'videos': [str(index_coded_clip(tmp_path / f'v{i}.mp4', 30 + i)) for i in range(3)]})
    plain = open_video_dataset(df, num_frames=6, train=False)
    for options in (VideoDataOptions(video_augment=True, video_flip=True, video_speed_perturb=0.5,
                                     temporal_sampling='segment'),):
        augmented = open_video_dataset(df, num_frames=6, train=False, options=options)
        assert augmented.augment is None and not augmented.spec.temporal_jitter and augmented.spec.speed_perturb == 0
        segment_eval = open_video_dataset(df, num_frames=6, train=False,
                                          options=VideoDataOptions(temporal_sampling='segment'))
        for i in range(3):
            for _ in range(3):  # and deterministic across calls / RNG states
                torch.manual_seed(np.random.randint(1000))
                assert torch.equal(augmented[i][0], segment_eval[i][0])
    assert [decode_frame_index((f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for f in plain[0][0]] == \
        np.linspace(0, 29, 6).astype(int).tolist()


def test_spilled_clip_file_cleanup_retries_and_sweeps(tmp_path, monkeypatch):
    import os
    import time
    from datasets import video_datasets as vd
    held = tmp_path / f'{vd.SPILL_PREFIX}held.arrow'
    held.write_bytes(b'x')
    real_unlink = os.unlink
    monkeypatch.setattr(vd.os, 'unlink', lambda p: (_ for _ in ()).throw(PermissionError('mapped by a worker')))
    vd._try_unlink(held)  # Windows: a persistent worker still maps the file
    assert held.exists() and str(held) in vd._PENDING_UNLINK
    monkeypatch.setattr(vd.os, 'unlink', real_unlink)
    vd._retry_pending_unlinks()  # what atexit runs
    assert not held.exists() and str(held) not in vd._PENDING_UNLINK
    stale, fresh = tmp_path / f'{vd.SPILL_PREFIX}old.arrow', tmp_path / f'{vd.SPILL_PREFIX}new.arrow'
    stale.write_bytes(b'x')
    fresh.write_bytes(b'x')
    old = time.time() - 2 * vd.STALE_SPILL_SECONDS
    os.utime(stale, (old, old))
    assert vd._sweep_stale_spills(tmp_path) == 1 and not stale.exists() and fresh.exists()


def test_loader_roles(monkeypatch, tmp_path):
    import sys
    from datasets import video_datasets as vd
    df = pd.DataFrame({'videos': [str(index_coded_clip(tmp_path / 'a.mp4', 8))]})
    dataset = vd.open_video_dataset(df, num_frames=2)
    assert vd.video_loader_kwargs(dataset, 2, True, 4)['persistent_workers'] is True
    assert vd.video_loader_kwargs(dataset, 2, False, 4)['persistent_workers'] is False  # one-shot pass
    assert vd.video_loader_kwargs(dataset, 2, False, 4, role='val')['persistent_workers'] is True
    monkeypatch.setattr('options.resolve_num_workers', lambda requested, n=None: requested if isinstance(requested, int) else 4)
    monkeypatch.setattr(sys, 'platform', 'win32')
    assert vd.video_loader_kwargs(dataset, 2, False, 'auto', role='val')['num_workers'] == vd.VAL_WORKERS_WINDOWS
    assert vd.video_loader_kwargs(dataset, 2, True, 'auto')['num_workers'] == 4
    assert vd.video_loader_kwargs(dataset, 2, False, 3, role='val')['num_workers'] == 3  # explicit wins
    with pytest.raises(ValueError):
        vd.video_loader_kwargs(dataset, 2, False, 0, role='eval')

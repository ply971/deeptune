"""Integration tests for the per-prediction error estimate: fitted by the
shared Trainer on the validation split, used by TestTrainer (eval), the
video/images inference functions and infer.py."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import evaluators.error_model as em
from evaluators.vision.evaluator import TestTrainer as Evaluator
from trainers.vision.trainer import Trainer


class CountingDataset(Dataset):
    """Records how often each split is read, to prove which data the error
    model was fitted on."""

    def __init__(self, inputs, labels):
        self.inputs, self.labels, self.reads = inputs, labels, 0

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        self.reads += 1
        return self.inputs[idx], self.labels[idx]


def _blobs(n, seed, num_classes=3):
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % num_classes
    centres = torch.eye(num_classes, 4) * 2.0
    return centres[labels] + torch.randn(n, 4, generator=g), labels


def _tiny_trainer(tmp_path, n_train=48, n_val=40, epochs=2, mode='cls'):
    torch.manual_seed(0)
    x_tr, y_tr = _blobs(n_train, 1)
    x_va, y_va = _blobs(n_val, 2)
    if mode == 'reg':
        y_tr, y_va = x_tr[:, 0].clone(), x_va[:, 0].clone()
    train_ds, val_ds = CountingDataset(x_tr, y_tr), CountingDataset(x_va, y_va)
    out_dim = 3 if mode == 'cls' else 1
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, out_dim))
    trainer = Trainer(model, DataLoader(train_ds, batch_size=8, shuffle=True),
                      DataLoader(val_ds, batch_size=8), 0.01, mode, epochs, tmp_path)
    return trainer, train_ds, val_ds


def test_trainer_fits_error_model_on_validation_split_only(tmp_path):
    trainer, train_ds, val_ds = _tiny_trainer(tmp_path, n_train=48, n_val=40, epochs=2)
    trainer.train()
    fit = json.loads((tmp_path / 'error_model_fit.json').read_text())
    assert fit['split'] == 'validation'
    assert fit['n_samples'] == len(val_ds) == fit['val_dataset_size'] != len(train_ds)
    # Training reads: one pass per epoch. Validation reads: one per epoch
    # plus exactly one extra pass for the error model.
    assert train_ds.reads == 2 * len(train_ds)
    assert val_ds.reads == 3 * len(val_ds)
    model = em.ErrorModel.load(tmp_path / 'error_model.json')
    assert model.mode == 'cls' and model.fit_info['num_outputs'] == 3


def test_error_model_does_not_change_training_results(tmp_path):
    results = []
    for enabled in (True, False):
        out = tmp_path / str(enabled)
        out.mkdir()
        trainer, _, _ = _tiny_trainer(out)
        trainer.fit_error_model = enabled
        trainer.train()
        results.append(({k: v.clone() for k, v in trainer.model.state_dict().items()},
                        torch.get_rng_state().clone(), trainer.model.training))
        assert (out / 'error_model.json').is_file() == enabled
    (w1, rng1, mode1), (w2, rng2, mode2) = results
    assert all(torch.equal(w1[k], w2[k]) for k in w1)
    assert torch.equal(rng1, rng2) and mode1 == mode2


def test_regression_trainer_writes_constant_error_model(tmp_path):
    trainer, _, val_ds = _tiny_trainer(tmp_path, mode='reg')
    trainer.train()
    model = em.ErrorModel.load(tmp_path / 'error_model.json')
    assert model.mode == 'reg' and model.method_used == 'constant'
    assert 'regression' in model.fit_info['fallback_reason']
    assert model.fit_info['n_samples'] == len(val_ds)


class _TinyNet(nn.Module):
    """Stands in for build_video_model / the images model class so the real
    train.py -> Trainer path runs in about a second on CPU."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))

    def forward(self, x, extract_embed=False):
        return self.net(x)


@pytest.mark.parametrize('modality', ['video', 'images'])
@pytest.mark.parametrize('flag', [False, True])
def test_no_error_model_flag_is_parsed_recorded_and_honoured(tmp_path, monkeypatch, modality, flag):
    """--no_error_model goes CLI -> DeepTuneVisionOptions -> train.py ->
    Trainer.fit_error_model, with no process-wide state: the real train()
    and Trainer run end to end; only data and model are tiny stand-ins."""
    from cli import DeepTuneVisionOptions
    from utils import RunType
    if modality == 'video':
        import trainers.video.train as train_module
        monkeypatch.setattr(train_module, 'build_video_model', lambda **kw: _TinyNet())
        dataset_attr = 'ParquetVideoDataset'
    else:
        import trainers.vision.train as train_module
        monkeypatch.setattr(train_module, 'get_model_cls', lambda **kw: _TinyNet)
        dataset_attr = 'ParquetImageDataset'
    splits = {'train': _blobs(12, 1), 'val': _blobs(9, 2)}

    class _FakeDataset:
        @staticmethod
        def from_parquet(parquet_file, **kwargs):
            return TensorDataset(*splits[Path(parquet_file).stem])

    monkeypatch.setattr(train_module, dataset_attr, _FakeDataset)
    extra = ['--no_error_model'] if flag else []
    args = DeepTuneVisionOptions(RunType.TRAIN, ['--out', str(tmp_path), '--model_version', 'resnet18'] + extra)
    assert args.no_error_model is flag
    onecall = DeepTuneVisionOptions(RunType.ONECALL, ['--out', str(tmp_path), '--modality', modality,
                                                      '--df', str(tmp_path / 'x.parquet')] + extra)
    assert onecall.no_error_model is flag
    # A later parse with the opposite setting must not leak into this run.
    DeepTuneVisionOptions(RunType.TRAIN, ['--out', str(tmp_path)] + ([] if flag else ['--no_error_model']))

    weights = Path(train_module.train(
        train_df=tmp_path / 'train.parquet', val_df=tmp_path / 'val.parquet', out=tmp_path,
        freeze_backbone=False, use_peft=False, fixed_seed=0, mode='cls', model_version='resnet18',
        batch_size=4, num_epochs=1, learning_rate=0.01, added_layers=1, num_classes=3, embed_size=8,
        model_str='tiny', args=args))
    run_dir = weights.parent if weights.suffix == '.pth' else weights
    assert (run_dir / 'training_log.csv').is_file()
    recorded = json.loads((run_dir / 'cli_arguments.json').read_text(encoding='utf-8-sig'))
    assert recorded['no_error_model'] is flag
    assert (run_dir / 'error_model.json').exists() is (not flag)
    assert (run_dir / 'error_model_fit.json').exists() is (not flag)
    if not flag:
        fit = json.loads((run_dir / 'error_model_fit.json').read_text())
        assert fit['n_samples'] == 9 and fit['split'] == 'validation'
        # The fingerprint matches the weights train.py saved afterwards.
        saved = torch.load(run_dir / 'model_weights.pth', map_location='cpu', weights_only=True)
        assert fit['weights_sha256'] == em.weights_fingerprint(saved)


def test_trainer_default_fits_and_explicit_false_skips(tmp_path):
    on, off = tmp_path / 'on', tmp_path / 'off'
    on.mkdir()
    off.mkdir()
    _tiny_trainer(on)[0].train()
    trainer = _tiny_trainer(off)[0]
    trainer.fit_error_model = False
    trainer.train()
    assert (on / 'error_model.json').is_file()
    assert not (off / 'error_model.json').exists() and not (off / 'error_model_fit.json').exists()
    assert (off / 'training_log.csv').is_file()


def test_eval_and_inference_refuse_error_model_fitted_on_other_weights(tmp_path, caplog):
    import logging
    from inference.common import attach_expected_error
    trainer, _, _ = _tiny_trainer(tmp_path)
    trainer.train()
    ckpt = tmp_path / 'model_weights.pth'
    trainer.saveModel(ckpt)
    x, y = _blobs(30, 5)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    good = Evaluator(trainer.model, 8, loader, 'cls', tmp_path / 'eval_good', device='cpu').test(ckpt)
    assert 'error_estimate' in good
    fitted = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    # Same folder, different weights (e.g. a later checkpoint): refused, loudly.
    changed = {k: v + 0.01 if v.is_floating_point() else v for k, v in trainer.model.state_dict().items()}
    other = tmp_path / 'other_weights.pth'
    torch.save(changed, other)
    with caplog.at_level(logging.WARNING):
        bad = Evaluator(trainer.model, 8, loader, 'cls', tmp_path / 'eval_bad', device='cpu').test(other)
    assert 'error_estimate' not in bad
    assert 'expected_error' not in pd.read_csv(tmp_path / 'eval_bad' / 'predictions.csv').columns
    assert any('different weights' in r.getMessage() for r in caplog.records)
    warnings = []
    refused = em.load_error_model_near(ckpt, 'cls', num_outputs=3, warn=warnings.append,
                                       weights_sha256=em.weights_fingerprint(changed))
    assert refused is None and 'different weights' in warnings[0]
    probs = np.array([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]])
    matching = pd.DataFrame({'sample_id': range(2)})
    assert attach_expected_error(matching, 'cls', ckpt, probs, fitted) is True
    assert matching['expected_error'].between(0, 1).all()
    mismatched = pd.DataFrame({'sample_id': range(2)})
    assert attach_expected_error(mismatched, 'cls', ckpt, probs, changed) is False
    assert 'expected_error' not in mismatched.columns


def test_weights_fingerprint_is_content_based():
    a = nn.Linear(3, 2).state_dict()
    same = {k: v.clone() for k, v in a.items()}
    assert em.weights_fingerprint(a) == em.weights_fingerprint(same)
    assert em.weights_fingerprint(a) == em.weights_fingerprint({k: v.numpy() for k, v in a.items()})
    bumped = {k: v.clone() for k, v in a.items()}
    bumped['bias'][0] += 1e-3
    assert em.weights_fingerprint(a) != em.weights_fingerprint(bumped)
    renamed = {('x' + k): v for k, v in a.items()}
    assert em.weights_fingerprint(a) != em.weights_fingerprint(renamed)
    assert len(em.weights_fingerprint({'w': torch.ones(2, dtype=torch.bfloat16)})) == 64


def _identity_eval(tmp_path, ckpt_dir):
    probs_like = torch.tensor([[6., 0.], [0., 6.], [4., 0.], [0., 4.], [1., 0.], [0., 1.]])
    loader = DataLoader(TensorDataset(probs_like, torch.tensor([0, 1, 0, 1, 1, 0])), batch_size=4)
    ckpt = ckpt_dir / 'model_weights.pth'
    torch.save(nn.Identity().state_dict(), ckpt)
    out = tmp_path / 'eval'
    return Evaluator(nn.Identity(), 4, loader, 'cls', out, device='cpu').test(ckpt), out


def test_eval_of_old_run_without_error_model_is_unchanged(tmp_path):
    ckpt_dir = tmp_path / 'run'
    ckpt_dir.mkdir()
    metrics, out = _identity_eval(tmp_path, ckpt_dir)
    assert 'error_estimate' not in metrics
    predictions = pd.read_csv(out / 'predictions.csv')
    assert 'expected_error' not in predictions.columns
    assert len(predictions) == 6 and predictions['correct'].sum() == 4
    assert not (out / 'error_calibration.json').exists()
    assert json.loads((out / 'full_metrics.json').read_text()) == metrics


def test_eval_adds_expected_error_and_calibration_report(tmp_path):
    ckpt_dir = tmp_path / 'run'
    ckpt_dir.mkdir()
    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1.0, 200)
    probs = np.column_stack([c, 1 - c])
    wrong = (rng.uniform(size=200) < (1 - c)).astype(float)
    em.ErrorModel('cls').fit(em.prediction_features(probs, 'cls'), wrong, num_outputs=2).save(
        ckpt_dir / 'error_model.json')
    metrics, out = _identity_eval(tmp_path, ckpt_dir)
    predictions = pd.read_csv(out / 'predictions.csv')
    assert predictions['expected_error'].between(0, 1).all()
    report = json.loads((out / 'error_calibration.json').read_text())
    assert report['split'] == 'test' and report['summary']['n'] == 6
    assert len(pd.read_csv(out / 'error_calibration.csv')) == report['summary']['n_bins']
    assert metrics['error_estimate']['method'] == 'isotonic'
    assert metrics['error_estimate']['n'] == 6
    assert json.loads((out / 'full_metrics.json').read_text()) == metrics


def test_regression_eval_writes_per_sample_abs_error(tmp_path):
    loader = DataLoader(TensorDataset(torch.tensor([[1.], [2.], [3.]]), torch.tensor([1., 2., 0.])), batch_size=2)
    Evaluator(nn.Identity(), 2, loader, 'reg', tmp_path, device='cpu').test()
    predictions = pd.read_csv(tmp_path / 'predictions.csv')
    assert predictions['abs_error'].tolist() == [0.0, 0.0, 3.0]
    assert 'expected_error' not in predictions.columns


# ---------------------------------------------------------------------------
# End to end on real (tiny, synthetic) video clips with dummy_video3d
# ---------------------------------------------------------------------------
SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])


def _clip(path: Path, level: int) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (32, 32))
    for i in range(6):
        writer.write(np.full((32, 32, 3), (level + 7 * i) % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _video_split(tmp_path, name, n, offset):
    rows = []
    for i in range(n):
        label = i % 2
        rows.append({'videos': _clip(tmp_path / f'{name}_{i}.mp4', 40 + 150 * label + offset + i),
                     'labels': label})
    path = tmp_path / f'{name}.parquet'
    pd.DataFrame(rows).to_parquet(path)
    return path


class _Args:
    def save_args(self, outdir):
        Path(outdir, 'cli_arguments.json').write_text('{}')


def test_video_end_to_end_train_eval_infer_with_expected_error(tmp_path, monkeypatch):
    import evaluators.video.evaluate as video_eval
    import infer
    import inference.video as video_inference
    from datasets.video_datasets import ParquetVideoDataset
    from utils import build_video_model

    monkeypatch.setattr(video_eval, 'transformations', SMALL)
    monkeypatch.setattr(video_inference, 'transformations', SMALL)
    train_pq = _video_split(tmp_path, 'train', 6, 0)
    val_pq = _video_split(tmp_path, 'val', 4, 3)
    test_pq = _video_split(tmp_path, 'test', 4, 5)

    torch.manual_seed(0)
    build = dict(model_version='dummy_video3d', num_classes=2, added_layers=1, embed_size=16,
                 freeze_backbone=False, mode='cls')
    model = build_video_model(**build)
    run_dir = tmp_path / 'trainval'
    run_dir.mkdir()
    train_loader = DataLoader(ParquetVideoDataset.from_parquet(train_pq, transform=SMALL, num_frames=2), batch_size=3)
    val_loader = DataLoader(ParquetVideoDataset.from_parquet(val_pq, transform=SMALL, num_frames=2), batch_size=2)
    trainer = Trainer(model, train_loader, val_loader, 1e-3, 'cls', 1, run_dir)
    trainer.train()
    ckpt = run_dir / 'model_weights.pth'
    trainer.saveModel(ckpt)
    fit = json.loads((run_dir / 'error_model_fit.json').read_text())
    assert fit['n_samples'] == 4 and fit['split'] == 'validation'
    assert fit['weights_sha256'] == em.weights_fingerprint(torch.load(ckpt, map_location='cpu', weights_only=True))
    fitted_json = (run_dir / 'error_model.json').read_text()

    metrics = video_eval.evaluate(eval_df=test_pq, out=tmp_path, model_weights=ckpt, num_classes=2,
                                  model_version='dummy_video3d', mode='cls', added_layers=1, embed_size=16,
                                  batch_size=2, freeze_backbone=False, args=_Args(), use_peft=False,
                                  model_str='dummy', num_frames=2)
    eval_dir = next(tmp_path.glob('eval_output_dummy_*'))
    predictions = pd.read_csv(eval_dir / 'predictions.csv')
    assert len(predictions) == 4
    assert predictions['expected_error'].between(0, 1).all()
    assert (eval_dir / 'error_calibration.json').is_file() and (eval_dir / 'error_calibration.csv').is_file()
    assert metrics['error_estimate']['n'] == 4
    # Evaluating never refits: the error model still holds validation-only data.
    assert (run_dir / 'error_model.json').read_text() == fitted_json

    test_df = pd.read_parquet(test_pq)
    direct = video_inference.predict_video(test_df, ckpt, batch_size=2, num_frames=2, use_peft=False, **build)
    assert direct['expected_error'].between(0, 1).all()
    np.testing.assert_allclose(direct['expected_error'], predictions['expected_error'])

    data_dir = tmp_path / 'new_data'
    for cls_name in ('a', 'b'):
        (data_dir / cls_name).mkdir(parents=True)
    for i, (blob, label) in enumerate(zip(test_df['videos'], test_df['labels'])):
        (data_dir / ('a' if label == 0 else 'b') / f'clip_{i}.mp4').write_bytes(blob)
    mapping = tmp_path / 'label_mapping.json'
    mapping.write_text(json.dumps({'a': 0, 'b': 1}))
    infer_out = tmp_path / 'infer_out'
    code = infer.main(['--modality', 'video', '--checkpoint', str(ckpt), '--data', str(data_dir),
                       '--out', str(infer_out), '--model_version', 'dummy_video3d', '--num_classes', '2',
                       '--added_layers', '1', '--embed_size', '16', '--num_frames', '2', '--batch_size', '2',
                       '--label_mapping', str(mapping)])
    assert code == 0
    inferred = pd.read_csv(infer_out / 'predictions.csv')
    assert inferred['expected_error'].between(0, 1).all()
    assert json.loads((infer_out / 'inference_metrics.json').read_text())['error_estimate']['n'] == 4
    assert (infer_out / 'error_calibration.csv').is_file()

    # An old run (no error_model.json next to the checkpoint): unchanged output.
    old_dir = tmp_path / 'old_run'
    old_dir.mkdir()
    (old_dir / 'model_weights.pth').write_bytes(ckpt.read_bytes())
    old = video_inference.predict_video(test_df, old_dir / 'model_weights.pth', batch_size=2, num_frames=2,
                                        use_peft=False, **build)
    assert 'expected_error' not in old.columns
    assert list(old.columns) == [c for c in direct.columns if c != 'expected_error']


def test_image_inference_adds_expected_error_only_with_error_model(tmp_path):
    import io
    from PIL import Image
    from inference.vision import predict_images
    from utils import get_model_cls
    torch.manual_seed(0)
    model_cls = get_model_cls('resnet', use_peft=False)
    model = model_cls(2, 'dummy', 1, 16, task_type='cls', freeze_backbone=False)
    ckpt = tmp_path / 'model_weights.pth'
    torch.save(model.state_dict(), ckpt)

    def png(level):
        buf = io.BytesIO()
        Image.fromarray(np.full((16, 16, 3), level, dtype=np.uint8)).save(buf, format='PNG')
        return buf.getvalue()

    df = pd.DataFrame({'images': [png(10), png(200)], 'labels': ['0', '1']})
    kwargs = dict(model_architecture='resnet', model_version='dummy', num_classes=2, added_layers=1,
                  embed_size=16, freeze_backbone=False, mode='cls', use_peft=False, batch_size=2)
    assert 'expected_error' not in predict_images(df, ckpt, **kwargs).columns
    em.ErrorModel('cls', method='constant').fit(em.prediction_features(np.array([[0.6, 0.4]] * 3), 'cls'),
                                                [0, 1, 0], num_outputs=2).save(tmp_path / 'error_model.json')
    with_estimate = predict_images(df, ckpt, **kwargs)
    np.testing.assert_allclose(with_estimate['expected_error'], (1 + 1) / (3 + 2))

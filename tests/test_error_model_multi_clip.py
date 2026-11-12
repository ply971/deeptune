"""End to end: per-prediction error model through A2's --test_clips wrapper
(Critic V-033). Trained with --test_clips 2 (the error model is fitted via the
Trainer's MultiClipModel hooks), then evaluated and inferred with 2 clips:
expected_error must be present, which proves the weights fingerprint matched
through the MultiClipModel wrapper. A --test_clips 1 eval must warn."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchvision

pytest.importorskip('av')

from datasets.video_synth import index_coded_clip  # noqa: E402

SMALL = torchvision.transforms.Compose([torchvision.transforms.Resize((16, 16)), torchvision.transforms.ToTensor()])


def _split(tmp_path, name, n, offset):
    rows = [{'videos': str(index_coded_clip(tmp_path / f'{name}_{i}.mp4', 8 + offset + i)), 'labels': i % 2}
            for i in range(n)]
    path = tmp_path / f'{name}.parquet'
    pd.DataFrame(rows).to_parquet(path)
    return path, rows


def test_error_model_survives_the_multi_clip_wrapper_end_to_end(tmp_path, monkeypatch, capsys):
    import evaluators.error_model as em
    import evaluators.video.evaluate as video_eval
    import infer
    import inference.video as video_inference
    import trainers.video.train as video_train
    from cli import DeepTuneVisionOptions
    from datasets.video_datasets import video_options_from_args
    from utils import RunType

    for module in (video_train, video_eval, video_inference):
        monkeypatch.setattr(module, 'transformations', SMALL)
    train_pq, _ = _split(tmp_path, 'train', 6, 0)
    val_pq, _ = _split(tmp_path, 'val', 6, 1)
    test_pq, test_rows = _split(tmp_path, 'test', 4, 2)
    build = dict(model_version='dummy_video3d', num_classes=2, added_layers=1, embed_size=8,
                 freeze_backbone=False, mode='cls', use_peft=False)
    clips2 = ['--test_clips', '2', '--num_workers', '0']

    # Train with --test_clips 2 through the real CLI options and train.py.
    torch.manual_seed(0)
    train_args = DeepTuneVisionOptions(RunType.TRAIN, ['--out', str(tmp_path), '--model_version', 'dummy_video3d'] + clips2)
    ckpt = Path(video_train.train(
        train_df=train_pq, val_df=val_pq, out=tmp_path, fixed_seed=0, batch_size=3, num_epochs=1,
        learning_rate=1e-3, model_str='dummy', args=train_args, num_frames=2,
        **{k: v for k, v in build.items() if k != 'model_version'}, model_version='dummy_video3d'))
    fit = json.loads((ckpt.parent / 'error_model_fit.json').read_text())
    assert fit['test_clips'] == 2 and fit['n_samples'] == 6 and fit['split'] == 'validation'
    assert fit['weights_sha256'] == em.weights_fingerprint(torch.load(ckpt, map_location='cpu', weights_only=True))
    capsys.readouterr()

    def evaluate(out, clips):
        args = DeepTuneVisionOptions(RunType.EVAL, ['--out', str(out), '--test_clips', str(clips), '--num_workers', '0'])
        return video_eval.evaluate(eval_df=test_pq, out=out, model_weights=ckpt, batch_size=2, args=args,
                                   model_str='dummy', num_frames=2,
                                   **{k: v for k, v in build.items()})

    # Eval with the matching --test_clips 2: TestTrainer scores the MultiClipModel,
    # whose state_dict is the inner model's, so the fingerprint matches.
    metrics = evaluate(tmp_path / 'eval2', 2)
    out = capsys.readouterr().out
    assert 'different weights' not in out and 'clip(s) per video' not in out
    assert metrics['error_estimate']['n'] == 4
    eval_dir = next((tmp_path / 'eval2').glob('eval_output_dummy_*'))
    predictions = pd.read_csv(eval_dir / 'predictions.csv')
    assert predictions['expected_error'].between(0, 1).all()
    assert (eval_dir / 'error_calibration.json').is_file()

    # Inference with 2 clips, directly and through infer.py (which inherits
    # nothing it is told explicitly).
    direct = video_inference.predict_video(pd.read_parquet(test_pq), ckpt, batch_size=2, num_frames=2,
                                           video_options=video_options_from_args({'test_clips': 2, 'num_workers': 0}),
                                           **build)
    assert direct['expected_error'].between(0, 1).all()
    assert 'clip_disagreement' in direct.columns
    pd.testing.assert_series_equal(direct['expected_error'], predictions['expected_error'], check_names=False)
    data_dir = tmp_path / 'new_data'
    for row in test_rows:
        target = data_dir / ('a' if row['labels'] == 0 else 'b') / Path(row['videos']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(row['videos']).read_bytes())
    mapping = tmp_path / 'label_mapping.json'
    mapping.write_text(json.dumps({'a': 0, 'b': 1}))
    infer_out = tmp_path / 'infer_out'
    capsys.readouterr()
    assert infer.main(['--modality', 'video', '--checkpoint', str(ckpt), '--data', str(data_dir), '--out', str(infer_out),
                       '--model_version', 'dummy_video3d', '--num_classes', '2', '--added_layers', '1',
                       '--embed_size', '8', '--num_frames', '2', '--batch_size', '2', '--label_mapping', str(mapping)]
                      + clips2) == 0
    assert 'different weights' not in capsys.readouterr().out
    assert pd.read_csv(infer_out / 'predictions.csv')['expected_error'].between(0, 1).all()
    assert json.loads((infer_out / 'inference_metrics.json').read_text())['error_estimate']['n'] == 4

    # A mismatched --test_clips 1 eval warns that the error model was fitted on 2 clips.
    evaluate(tmp_path / 'eval1', 1)
    assert 'fitted on 2 clip(s) per video' in capsys.readouterr().out

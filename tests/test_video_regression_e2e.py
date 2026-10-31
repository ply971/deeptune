"""deeptune.py --modality video --mode reg, from the command line to a new infer.py. It used to crash on every model, printing
the training log's 'Regression: No Accuracy' as a number (helpers.py)."""
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('av')

from datasets.video_synth import translating_clip
from helpers import print_training_log_table


def test_the_training_log_table_prints_a_regression_log(tmp_path, capsys):
    log = tmp_path / 'training_log.csv'
    pd.DataFrame({'epoch': [1], 'epoch_loss': [0.25], 'epoch_accuracy': ['Regression: No Accuracy'], 'val_loss': [0.5],
                  'val_accuracy': ['Regression: No Accuracy']}).to_csv(log, index=False)
    print_training_log_table(str(log))
    printed = capsys.readouterr().out
    assert '0.25' in printed and '0.5' in printed and 'No Accuracy' not in printed
    assert ' %' not in printed, 'regression has no accuracy to show as a percentage'


def test_the_training_log_table_still_prints_accuracy(tmp_path, capsys):
    log = tmp_path / 'training_log.csv'
    pd.DataFrame({'epoch': [1], 'epoch_loss': [0.25], 'epoch_accuracy': [75.0], 'val_loss': [0.5], 'val_accuracy': [50.0]}).to_csv(log, index=False)
    print_training_log_table(str(log))
    printed = capsys.readouterr().out
    assert '75.00 %' in printed and '50.00 %' in printed


@pytest.mark.parametrize('version,flags', [('resnet_tiny', ['--embed_size', '8']), ('vjepa2_tiny', ['--embed_size', '8'])])
def test_regression_trains_evaluates_embeds_and_infers(version, flags, tmp_path):
    import deeptune
    import infer
    rows = [{'videos': str(translating_clip(tmp_path / f'c{i}.mp4', 10, 2, 0, size=48, seed=i)), 'labels': float(i) / 2} for i in range(10)]
    manifest = tmp_path / 'clips.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    out = tmp_path / 'out'
    deeptune.main(['--modality', 'video', '--mode', 'reg', '--out', str(out), '--model_version', version, '--num_epochs', '1', '--batch_size', '2',
                   '--num_frames', '4', '--added_layers', '2', '--fixed-seed', '--df', str(manifest), '--raw-data', '--report_boot', '0', *flags])
    checkpoint = next(out.rglob('trainval_output_*/model_weights.pth'))
    recorded = json.loads((checkpoint.parent / 'cli_arguments.json').read_text(encoding='utf-8-sig'))
    assert recorded['mode'] == 'reg'
    predictions = tmp_path / 'infer'
    assert infer.main(['--modality', 'video', '--checkpoint', str(checkpoint), '--data', str(Path(rows[0]['videos'])), '--out', str(predictions),
                       '--mode', 'reg', '--model_version', version, '--num_classes', '1', '--added_layers', '2', '--embed_size', '8']) == 0
    assert 'predicted_value' in pd.read_csv(predictions / 'predictions.csv').columns

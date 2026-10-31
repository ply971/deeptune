"""Ensembles: the average is the hand-computed one, through the CLI and infer.py, with members of different kinds."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip('av')

from datasets.video_synth import translating_clip
from inference.ensemble import combine_tables, ensemble_runs, ensemble_tables, probability_columns, resolve_checkpoint
from utils import build_video_model


def _table(ids, probabilities, **extra):
    table = pd.DataFrame({'video_id': ids})
    for k, column in enumerate(np.asarray(probabilities, dtype=float).T):
        table[f'probability_{k}'] = column
    return table.assign(**extra)


def test_the_average_is_the_hand_computed_one():
    a = _table(['x', 'y'], [[0.9, 0.1], [0.2, 0.8]])
    b = _table(['x', 'y'], [[0.3, 0.7], [0.6, 0.4]])
    result = combine_tables([a, b])
    assert result['probability_0'].tolist() == pytest.approx([0.6, 0.4])
    assert result['probability_1'].tolist() == pytest.approx([0.4, 0.6])
    assert result['predicted_class'].tolist() == [0, 1]
    assert result['confidence'].tolist() == pytest.approx([0.6, 0.6])
    assert result['member_disagreement'].tolist() == pytest.approx([0.3, 0.2])


def test_it_averages_probabilities_not_logits():
    """One very confident member: the mean of logits picks class 0, the mean of probabilities (0.466, 0.534) class 1."""
    members = [[[0.999, 0.001]], [[0.2, 0.8]], [[0.2, 0.8]]]
    logits = np.log(np.asarray(members)).mean(axis=0)
    assert logits.argmax() == 0, 'the case only tests something if logit averaging would differ'
    result = combine_tables([_table(['x'], m) for m in members])
    assert result['predicted_class'].tolist() == [1]
    assert result['probability_1'].iloc[0] == pytest.approx((0.001 + 0.8 + 0.8) / 3)


def test_rows_are_matched_by_id_not_by_position():
    a = _table(['x', 'y'], [[1.0, 0.0], [0.0, 1.0]])
    b = _table(['y', 'x'], [[0.0, 1.0], [1.0, 0.0]])
    result = combine_tables([a, b])
    assert result['video_id'].tolist() == ['x', 'y']
    assert result['member_disagreement'].tolist() == pytest.approx([0.0, 0.0])
    assert result['predicted_class'].tolist() == [0, 1]


def test_scoring_against_targets():
    a = _table(['x', 'y'], [[0.9, 0.1], [0.9, 0.1]], target=[0, 1])
    result = combine_tables([a, a.copy()])
    assert result['correct'].tolist() == [True, False]


def test_scoring_against_class_names():
    a = _table(['x'], [[0.1, 0.9]], true_label=['right'])
    result = combine_tables([a, a.copy()], label_mapping={'left': 0, 'right': 1})
    assert result['predicted_label'].tolist() == ['right'] and result['correct'].tolist() == [True]


def test_regression_averages_values():
    a = pd.DataFrame({'sample_id': [0, 1], 'predicted_value': [1.0, 2.0], 'true_value': [2.0, 2.0]})
    b = pd.DataFrame({'sample_id': [0, 1], 'predicted_value': [3.0, 6.0], 'true_value': [2.0, 2.0]})
    result = combine_tables([a, b], 'reg')
    assert result['predicted_value'].tolist() == [2.0, 4.0]
    assert result['abs_error'].tolist() == [0.0, 2.0]
    assert result['member_disagreement'].tolist() == [1.0, 2.0]


@pytest.mark.parametrize('second,message', [
    (_table(['x', 'z'], [[0.5, 0.5], [0.5, 0.5]]), 'other rows'),
    (_table(['x', 'x'], [[0.5, 0.5], [0.5, 0.5]]), 'repeats'),
    (_table(['x', 'y'], [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]], target=[0, 1]), 'same classes'),
    (_table(['x', 'y'], [[0.5, 0.5], [0.5, 0.5]], target=[1, 1]), 'labels'),
])
def test_members_that_do_not_line_up_are_refused(second, message):
    first = _table(['x', 'y'], [[0.5, 0.5], [0.5, 0.5]], target=[0, 1])
    with pytest.raises(ValueError, match=message):
        combine_tables([first, second])


def test_one_member_is_not_an_ensemble():
    with pytest.raises(ValueError, match='at least two'):
        combine_tables([_table(['x'], [[1.0, 0.0]])])


def test_probability_columns_are_in_class_order_not_file_order():
    columns = [f'probability_{k}' for k in (10, 2, 0, 1, 3, 4, 5, 6, 7, 8, 9)]
    assert probability_columns(pd.DataFrame(columns=columns)) == [f'probability_{k}' for k in range(11)]
    with pytest.raises(ValueError, match='without gaps'):
        probability_columns(pd.DataFrame(columns=['probability_0', 'probability_2']))


# --- finished tables, the route for feature-head probes -----------------------------------------------------------------
def test_tables_through_the_command_line(tmp_path):
    import ensemble
    for name, rows, licence in (('a', [[0.9, 0.1], [0.2, 0.8]], {'rgb': 'MIT (a)'}), ('b', [[0.3, 0.7], [0.6, 0.4]], {'rgb': 'Apache-2.0 (b)'})):
        folder = tmp_path / name
        folder.mkdir()
        _table(['x', 'y'], rows, target=[0, 1]).to_csv(folder / 'predictions.csv', index=False)
        (folder / 'cli_arguments.json').write_text(json.dumps({'weights_licence': licence}), encoding='utf-8')
    out = tmp_path / 'out'
    assert ensemble.main(['--predictions', str(tmp_path / 'a' / 'predictions.csv'), str(tmp_path / 'b' / 'predictions.csv'), '--out', str(out)]) == 0
    result = pd.read_csv(out / 'predictions.csv')
    assert result['probability_0'].tolist() == pytest.approx([0.6, 0.4])
    assert json.loads((out / 'inference_metrics.json').read_text())['accuracy'] == 1.0
    record = json.loads((out / 'ensemble.json').read_text())
    assert [m['weights_licence']['rgb'] for m in record['members']] == ['MIT (a)', 'Apache-2.0 (b)']
    assert 'expected_error' not in result.columns


# --- trained runs -------------------------------------------------------------------------------------------------------
def _save_run(folder: Path, version: str, seed: int, num_frames: int, **recorded) -> tuple:
    torch.manual_seed(seed)
    settings = dict(model_version=version, num_classes=2, added_layers=2, embed_size=8, freeze_backbone=True, use_peft=False,
                    mode='cls', pooling='mean', num_frames=num_frames, modality='video', **recorded)
    model = build_video_model(version, 2, 2, 8, True, 'cls', pooling=settings['pooling'])
    folder.mkdir(parents=True)
    torch.save(model.state_dict(), folder / 'model_weights.pth')
    (folder / 'cli_arguments.json').write_text(json.dumps(settings), encoding='utf-8')
    return model.eval(), settings


def _videos(root: Path):
    root.mkdir()
    return [translating_clip(root / f'clip_{i}.mp4', 10, dx, 0, size=48, seed=i) for i, dx in enumerate((-2, 2))]


def _by_hand(model, version, num_frames, paths):
    """Probabilities computed without inference/: the dataset's clip, the model, a softmax."""
    from datasets.video_datasets import native_video_transform, open_video_dataset
    from helpers import transformations
    dataset = open_video_dataset(pd.DataFrame({'videos': [str(p) for p in paths]}), transform=transformations, num_frames=num_frames,
                                 clip_transform=native_video_transform(version))
    with torch.no_grad():
        return torch.stack([torch.softmax(model(dataset[i][0].unsqueeze(0)), dim=1)[0] for i in range(len(paths))]).numpy()


def test_members_of_different_kinds_average_to_the_hand_computed_value(tmp_path):
    """Other backbone, other frames, other preprocessing: each member must get its own, as recorded."""
    paths = _videos(tmp_path / 'clips')
    frames = {'vjepa2_tiny': 6, 'siglip2_tiny': 4}
    runs = {version: tmp_path / version for version in frames}
    expected = []
    for seed, (version, count) in enumerate(frames.items()):
        model, _ = _save_run(runs[version], version, seed, count)
        expected.append(_by_hand(model, version, count, paths))
    assert not np.allclose(expected[0], expected[1]), 'members must differ for the average to mean anything'
    out = tmp_path / 'out'
    result = ensemble_runs(list(runs.values()), tmp_path / 'clips', out, batch_size=1)
    computed = result.sort_values('sample_id')[['probability_0', 'probability_1']].to_numpy()
    assert computed == pytest.approx((expected[0] + expected[1]) / 2, abs=1e-5)
    assert result['predicted_class'].tolist() == ((expected[0] + expected[1]) / 2).argmax(axis=1).tolist()
    record = json.loads((out / 'ensemble.json').read_text())
    assert [m['model_version'] for m in record['members']] == list(frames)
    assert all(next(iter(m['weights_licence'].values())).startswith('none') for m in record['members'])
    assert 'expected_error' not in result.columns


def test_infer_reaches_the_same_ensemble(tmp_path):
    import infer
    paths = _videos(tmp_path / 'clips')
    first, _ = _save_run(tmp_path / 'a', 'siglip2_tiny', 0, 4)
    second, _ = _save_run(tmp_path / 'b', 'siglip2_tiny', 1, 4)
    out = tmp_path / 'out'
    assert infer.main(['--modality', 'video', '--ensemble', str(tmp_path / 'a'), str(tmp_path / 'b'), '--data', str(tmp_path / 'clips'), '--out', str(out)]) == 0
    written = pd.read_csv(out / 'predictions.csv')
    expected = (_by_hand(first, 'siglip2_tiny', 4, paths) + _by_hand(second, 'siglip2_tiny', 4, paths)) / 2
    assert written[['probability_0', 'probability_1']].to_numpy() == pytest.approx(expected, abs=1e-5)


def test_infer_still_needs_a_checkpoint_without_an_ensemble(tmp_path, capsys):
    import infer
    with pytest.raises(SystemExit):
        infer.main(['--modality', 'video', '--data', str(tmp_path), '--out', str(tmp_path / 'o')])
    assert '--checkpoint' in capsys.readouterr().err


def test_members_use_their_recorded_test_clips(tmp_path):
    """--test_clips is reused, not reimplemented: a member recorded with 2 clips averages them first."""
    from datasets.video_datasets import native_video_transform, open_video_dataset
    from helpers import transformations
    paths = _videos(tmp_path / 'clips')
    first, _ = _save_run(tmp_path / 'a', 'siglip2_tiny', 0, 4, test_clips=2)
    second, _ = _save_run(tmp_path / 'b', 'siglip2_tiny', 1, 4)
    result = ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', tmp_path / 'out', batch_size=1)
    dataset = open_video_dataset(pd.DataFrame({'videos': [str(p) for p in paths]}), transform=transformations, num_frames=4,
                                 clip_transform=native_video_transform('siglip2_tiny'), num_clips=2)
    with torch.no_grad():
        two_clips = torch.stack([torch.softmax(first(dataset[i][0]), dim=1).mean(dim=0) for i in range(2)]).numpy()
    expected = (two_clips + _by_hand(second, 'siglip2_tiny', 4, paths)) / 2
    assert result[['probability_0', 'probability_1']].to_numpy() == pytest.approx(expected, abs=1e-5)


def test_runs_that_cannot_be_averaged_are_refused_before_loading(tmp_path):
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'a', 'siglip2_tiny', 0, 4)
    other = tmp_path / 'b'
    _save_run(other, 'siglip2_tiny', 1, 4)
    arguments = json.loads((other / 'cli_arguments.json').read_text())
    (other / 'cli_arguments.json').write_text(json.dumps({**arguments, 'num_classes': 5}), encoding='utf-8')
    with pytest.raises(ValueError, match='same task'):
        ensemble_runs([tmp_path / 'a', other], tmp_path / 'clips', tmp_path / 'out')
    (other / 'cli_arguments.json').unlink()
    with pytest.raises(ValueError, match='cli_arguments'):
        ensemble_runs([tmp_path / 'a', other], tmp_path / 'clips', tmp_path / 'out')


def test_runs_with_different_class_meanings_are_refused(tmp_path):
    _videos(tmp_path / 'clips')
    for name, mapping in (('a', {'left': 0, 'right': 1}), ('b', {'left': 1, 'right': 0})):
        _save_run(tmp_path / name / 'trainval_output', 'siglip2_tiny', 0, 4)
        splits = tmp_path / name / 'data_splits_1'
        splits.mkdir()
        (splits / 'label_mapping.json').write_text(json.dumps(mapping), encoding='utf-8')
    with pytest.raises(ValueError, match='differently'):
        ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', tmp_path / 'out')


def test_a_folder_must_hold_one_set_of_weights(tmp_path):
    (tmp_path / 'r' / 'x').mkdir(parents=True)
    (tmp_path / 'r' / 'y').mkdir()
    with pytest.raises(ValueError, match='found 0'):
        resolve_checkpoint(tmp_path / 'r')
    (tmp_path / 'r' / 'x' / 'model_weights.pth').touch()
    assert resolve_checkpoint(tmp_path / 'r').name == 'model_weights.pth'
    (tmp_path / 'r' / 'y' / 'model_weights.pth').touch()
    with pytest.raises(ValueError, match='found 2'):
        resolve_checkpoint(tmp_path / 'r')


def test_tables_refuse_one_file(tmp_path):
    _table(['x'], [[1.0, 0.0]]).to_csv(tmp_path / 'p.csv', index=False)
    with pytest.raises(ValueError, match='at least two'):
        ensemble_tables([tmp_path / 'p.csv'], tmp_path / 'out')

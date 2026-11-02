"""Ensembles: the average is the hand-computed one, through the CLI and infer.py, with members of different kinds."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip('av')

from datasets.video_synth import translating_clip
from handlers.video_backbones import HUB_BACKBONES, WeightsLicenceNotAccepted, reset_accepted_licences
from inference.common import label_mapping_digest
from inference.ensemble import combine_tables, ensemble_runs, ensemble_tables, probability_columns, resolve_checkpoint, verify_label_mapping
from utils import build_video_model


LEFT_RIGHT = {'left': 0, 'right': 1}


@pytest.fixture(autouse=True)
def clean_acceptance():
    reset_accepted_licences()
    yield
    reset_accepted_licences()


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
    assert record['label_check']['result'].startswith('verified')
    assert 'expected_error' not in result.columns


# --- trained runs -------------------------------------------------------------------------------------------------------
def _save_mapping(folder: Path, name: str, mapping: dict) -> Path:
    splits = folder / f'data_splits_{name}'
    splits.mkdir(parents=True, exist_ok=True)
    path = splits / 'label_mapping.json'
    path.write_text(json.dumps(mapping), encoding='utf-8')
    return path


def _save_run(folder: Path, version: str, seed: int, num_frames: int, mapping=LEFT_RIGHT, **recorded) -> tuple:
    """A run folder as deeptune.py leaves it. mapping=None: a run with no recorded mapping."""
    torch.manual_seed(seed)
    if mapping is not None:
        path = _save_mapping(folder.parent, folder.name, mapping)
        recorded = {'label_mapping_path': str(path), 'label_mapping_sha256': label_mapping_digest(mapping), **recorded}
    mode = recorded.pop('mode_override', 'cls')
    settings = dict(model_version=version, num_classes=2, added_layers=2, embed_size=8, freeze_backbone=True, use_peft=False,
                    mode=mode, pooling='mean', num_frames=num_frames, modality='video', **recorded)
    model = build_video_model(version, 2, 2, 8, True, mode, pooling=settings['pooling'])
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
    assert all(m['weights_licence'].startswith('none') and 'none' in m['weights_revision'] for m in record['members'])
    assert record['label_check']['result'].startswith('verified')
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


def test_runs_with_different_class_meanings_are_refused_even_when_unverified_is_allowed(tmp_path):
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'a' / 'trainval_output', 'siglip2_tiny', 0, 4, mapping={'left': 0, 'right': 1})
    _save_run(tmp_path / 'b' / 'trainval_output', 'siglip2_tiny', 0, 4, mapping={'left': 1, 'right': 0})
    for allow in (False, True):
        with pytest.raises(ValueError, match='differently'):
            ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', tmp_path / 'out', allow_unverified_labels=allow)


def test_a_member_with_no_mapping_is_not_waved_through(tmp_path):
    """V-052 case A: its class order was never compared."""
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4)
    _save_run(tmp_path / 'b' / 'x', 'siglip2_tiny', 1, 4, mapping=None)
    for order in ([tmp_path / 'a', tmp_path / 'b'], [tmp_path / 'b', tmp_path / 'a']):
        with pytest.raises(ValueError, match='--allow_unverified_labels'):
            ensemble_runs(order, tmp_path / 'clips', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_two_runs_in_one_folder_are_each_checked_against_their_own_mapping(tmp_path):
    """V-052 case B: both used to read the first data_splits_* in the folder, so opposite orders looked the same."""
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'trainval_a', 'siglip2_tiny', 0, 4, mapping={'left': 0, 'right': 1})
    _save_run(tmp_path / 'trainval_b', 'siglip2_tiny', 1, 4, mapping={'left': 1, 'right': 0})
    with pytest.raises(ValueError, match='differently'):
        ensemble_runs([tmp_path / 'trainval_a' / 'model_weights.pth', tmp_path / 'trainval_b' / 'model_weights.pth'], tmp_path / 'clips', tmp_path / 'out')


def test_a_mapping_of_an_unrelated_run_nearby_verifies_nothing(tmp_path):
    """V-052 case C: the search climbed two folders into another run."""
    _save_run(tmp_path / 'exp' / 'run1' / 'x', 'siglip2_tiny', 0, 4, mapping=None)
    _save_mapping(tmp_path / 'exp', 'other', {'cat': 0, 'dog': 1})
    _save_mapping(tmp_path / 'exp' / 'run1', 'other', {'cat': 0, 'dog': 1})
    checkpoint = tmp_path / 'exp' / 'run1' / 'x' / 'model_weights.pth'
    check = verify_label_mapping(checkpoint, json.loads((checkpoint.parent / 'cli_arguments.json').read_text()))
    assert check['status'] == 'unverified' and check['mapping'] is None


def test_a_changed_mapping_file_is_not_verified(tmp_path):
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4)
    checkpoint = tmp_path / 'a' / 'x' / 'model_weights.pth'
    recorded = json.loads((checkpoint.parent / 'cli_arguments.json').read_text())
    assert verify_label_mapping(checkpoint, recorded)['status'] == 'verified'
    Path(recorded['label_mapping_path']).write_text(json.dumps({'left': 1, 'right': 0}), encoding='utf-8')
    assert verify_label_mapping(checkpoint, recorded)['status'] == 'unverified'


def test_a_moved_run_still_verifies_by_hash(tmp_path):
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4)
    checkpoint = tmp_path / 'a' / 'x' / 'model_weights.pth'
    recorded = json.loads((checkpoint.parent / 'cli_arguments.json').read_text())
    moved = {**recorded, 'label_mapping_path': str(tmp_path / 'gone' / 'label_mapping.json')}
    assert verify_label_mapping(checkpoint, moved)['status'] == 'verified'


def test_allowing_unverified_labels_runs_and_says_so(tmp_path):
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4, mapping=None)
    _save_run(tmp_path / 'b' / 'x', 'siglip2_tiny', 1, 4, mapping=None)
    ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', tmp_path / 'out', batch_size=1, allow_unverified_labels=True)
    check = json.loads((tmp_path / 'out' / 'ensemble.json').read_text())['label_check']
    assert 'not verified' in check['result']
    assert [m['status'] for m in check['members']] == ['unverified', 'unverified']


def test_a_given_mapping_that_contradicts_the_runs_is_refused(tmp_path):
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4)
    _save_run(tmp_path / 'b' / 'x', 'siglip2_tiny', 1, 4)
    other = _save_mapping(tmp_path, 'mine', {'left': 1, 'right': 0})
    with pytest.raises(ValueError, match='--label_mapping differs'):
        ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', tmp_path / 'out', label_mapping_path=other)


def test_tables_without_targets_cannot_be_verified(tmp_path):
    for name in 'ab':
        (tmp_path / name).mkdir()
        _table(['x'], [[0.5, 0.5]]).to_csv(tmp_path / name / 'p.csv', index=False)
    paths = [tmp_path / 'a' / 'p.csv', tmp_path / 'b' / 'p.csv']
    with pytest.raises(ValueError, match='--allow_unverified_labels'):
        ensemble_tables(paths, tmp_path / 'out')
    ensemble_tables(paths, tmp_path / 'out', allow_unverified_labels=True)
    assert 'not verified' in json.loads((tmp_path / 'out' / 'ensemble.json').read_text())['label_check']['result']


def test_regression_members_need_no_class_order(tmp_path):
    _videos(tmp_path / 'clips')
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4, mapping=None, mode_override='reg')
    _save_run(tmp_path / 'b' / 'x', 'siglip2_tiny', 1, 4, mapping=None, mode_override='reg')
    result = ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', tmp_path / 'out', batch_size=1)
    assert 'predicted_value' in result.columns and 'member_disagreement' in result.columns
    assert 'not applicable' in json.loads((tmp_path / 'out' / 'ensemble.json').read_text())['label_check']['result']


# --- scoring ------------------------------------------------------------------------------------------------------------
def test_names_with_no_mapping_are_not_scored_as_zero(tmp_path, capsys):
    """'left' against index 0 used to give a false accuracy of 0.0."""
    labelled = tmp_path / 'labelled'
    for k, name in enumerate(LEFT_RIGHT):
        (labelled / name).mkdir(parents=True)
        translating_clip(labelled / name / f'c{k}.mp4', 10, 2, 0, size=48, seed=k)
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4, mapping=None)
    _save_run(tmp_path / 'b' / 'x', 'siglip2_tiny', 1, 4, mapping=None)
    result = ensemble_runs([tmp_path / 'a', tmp_path / 'b'], labelled, tmp_path / 'out', batch_size=1, allow_unverified_labels=True)
    assert 'correct' not in result.columns
    assert 'true_label' in result.columns
    assert not (tmp_path / 'out' / 'inference_metrics.json').exists()
    assert 'Not scored' in capsys.readouterr().out


def test_names_with_a_mapping_are_scored(tmp_path):
    labelled = tmp_path / 'labelled'
    for k, name in enumerate(LEFT_RIGHT):
        (labelled / name).mkdir(parents=True)
        translating_clip(labelled / name / f'c{k}.mp4', 10, 2, 0, size=48, seed=k)
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4)
    _save_run(tmp_path / 'b' / 'x', 'siglip2_tiny', 1, 4)
    result = ensemble_runs([tmp_path / 'a', tmp_path / 'b'], labelled, tmp_path / 'out', batch_size=1)
    assert result['correct'].tolist() == [t == p for t, p in zip(result['true_label'], result['predicted_label'])]
    assert (tmp_path / 'out' / 'inference_metrics.json').exists()


def test_single_model_inference_does_not_invent_an_accuracy_either(tmp_path):
    import infer
    labelled = tmp_path / 'labelled'
    for k, name in enumerate(LEFT_RIGHT):
        (labelled / name).mkdir(parents=True)
        translating_clip(labelled / name / f'c{k}.mp4', 10, 2, 0, size=48, seed=k)
    _save_run(tmp_path / 'a' / 'x', 'siglip2_tiny', 0, 4)
    checkpoint = tmp_path / 'a' / 'x' / 'model_weights.pth'
    base = ['--modality', 'video', '--checkpoint', str(checkpoint), '--data', str(labelled), '--model_version', 'siglip2_tiny', '--embed_size', '8', '--freeze-backbone']
    assert infer.main([*base, '--out', str(tmp_path / 'without')]) == 0
    assert 'correct' not in pd.read_csv(tmp_path / 'without' / 'predictions.csv').columns
    assert not (tmp_path / 'without' / 'inference_metrics.json').exists()
    mapping = json.loads((checkpoint.parent / 'cli_arguments.json').read_text())['label_mapping_path']
    assert infer.main([*base, '--out', str(tmp_path / 'with'), '--label_mapping', mapping]) == 0
    assert 'correct' in pd.read_csv(tmp_path / 'with' / 'predictions.csv').columns
    assert (tmp_path / 'with' / 'inference_metrics.json').exists()


# --- flags --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('flags', [['--test_clips', '5'], ['--num_frames', '2'], ['--checkpoint', 'c.pth'], ['--model_version', 'resnet18'],
                                   ['--pooling', 'attention'], ['--test_crops', '3'], ['--use-peft'], ['--num_classes', '3'], ['--num_frames=2']])
def test_infer_ensemble_refuses_flags_it_would_ignore(flags, tmp_path, capsys):
    import infer
    with pytest.raises(SystemExit):
        infer.main(['--modality', 'video', '--ensemble', 'a', 'b', '--data', str(tmp_path), '--out', str(tmp_path / 'o'), *flags])
    error = capsys.readouterr().err
    assert 'would be ignored' in error and flags[0].split('=')[0] in error


def test_unverified_flag_means_nothing_without_an_ensemble(tmp_path, capsys):
    import infer
    with pytest.raises(SystemExit):
        infer.main(['--modality', 'video', '--checkpoint', 'c', '--data', str(tmp_path), '--out', str(tmp_path / 'o'), '--allow_unverified_labels'])
    assert '--allow_unverified_labels' in capsys.readouterr().err


# --- licences -----------------------------------------------------------------------------------------------------------
@pytest.fixture()
def opt_in_stub(monkeypatch):
    """videomae_b16 with its real gate and a tiny network in place of the download. Returns the list of loads."""
    from dataclasses import replace

    import src.video.hub_backbones as hub
    loads = []
    monkeypatch.setitem(HUB_BACKBONES, 'videomae_b16', replace(HUB_BACKBONES['videomae_b16'], input_size=32, resize_short=36, fixed_frames=4))

    def fake_load(info):
        loads.append(info.version)
        torch.manual_seed(0)
        return hub._tiny_model('videomae_tiny')

    monkeypatch.setattr(hub, '_load_videomae', fake_load)
    return loads


def _opt_in_run(folder, **recorded):
    from handlers.video_backbones import accept_weights_licences
    accept_weights_licences(['CC-BY-NC-4.0'])
    _save_run(folder, 'videomae_b16', 0, 4, **recorded)
    reset_accepted_licences()


@pytest.mark.parametrize('recorded_first', [True, False])
def test_one_members_acceptance_does_not_open_the_gate_for_another(recorded_first, tmp_path, opt_in_stub):
    """V-052: the first order used to pass and the second to refuse."""
    _videos(tmp_path / 'clips')
    _opt_in_run(tmp_path / 'yes' / 'x', accept_weights_licence=['CC-BY-NC-4.0'])
    _opt_in_run(tmp_path / 'no' / 'x')
    opt_in_stub.clear()
    order = [tmp_path / 'yes', tmp_path / 'no'] if recorded_first else [tmp_path / 'no', tmp_path / 'yes']
    with pytest.raises(WeightsLicenceNotAccepted, match='CC-BY-NC-4.0'):
        ensemble_runs(order, tmp_path / 'clips', tmp_path / 'out', batch_size=1)
    assert opt_in_stub == [], 'refused before any weights loaded'


def test_a_member_cannot_use_acceptance_left_over_in_the_process(tmp_path, opt_in_stub):
    from handlers.video_backbones import accept_weights_licences
    from inference.ensemble import _predict_member
    _videos(tmp_path / 'clips')
    _opt_in_run(tmp_path / 'no' / 'x')
    accept_weights_licences(['CC-BY-NC-4.0'])
    recorded = json.loads((tmp_path / 'no' / 'x' / 'cli_arguments.json').read_text())
    frame = pd.DataFrame({'videos': [str(next((tmp_path / 'clips').glob('*.mp4')))]})
    with pytest.raises(WeightsLicenceNotAccepted):
        _predict_member(tmp_path / 'no' / 'x' / 'model_weights.pth', recorded, frame, 1, None)
    from handlers.video_backbones import _accepted
    assert 'cc-by-nc-4.0' in _accepted, 'what the process accepted comes back afterwards'


def test_every_member_recorded_or_the_flag_passes_the_gate(tmp_path, opt_in_stub):
    _videos(tmp_path / 'clips')
    _opt_in_run(tmp_path / 'yes' / 'x', accept_weights_licence=['CC-BY-NC-4.0'])
    _opt_in_run(tmp_path / 'no' / 'x')
    opt_in_stub.clear()
    ensemble_runs([tmp_path / 'yes', tmp_path / 'no'], tmp_path / 'clips', tmp_path / 'out', batch_size=1, accept_licences=['CC-BY-NC-4.0'])
    assert opt_in_stub == ['videomae_b16', 'videomae_b16']


@pytest.mark.parametrize('recorded,passes', [(['CC-BY-NC-4.0'], True), (None, False), (['MIT'], False)])
def test_rebuilding_one_opt_in_run_follows_its_record(recorded, passes, tmp_path, opt_in_stub):
    import infer
    paths = _videos(tmp_path / 'clips')
    _opt_in_run(tmp_path / 'r' / 'x', **({} if recorded is None else {'accept_weights_licence': recorded}))
    opt_in_stub.clear()
    argv = ['--modality', 'video', '--checkpoint', str(tmp_path / 'r' / 'x' / 'model_weights.pth'), '--data', str(paths[0]), '--out', str(tmp_path / 'o'),
            '--model_version', 'videomae_b16', '--num_frames', '4', '--freeze-backbone']
    if passes:
        assert infer.main(argv) == 0
        assert opt_in_stub == ['videomae_b16']
        written = json.loads((tmp_path / 'o' / 'inference_run.json').read_text())
        assert 'CC-BY-NC-4.0' in written['weights_licence'] and written['model_version'] == 'videomae_b16'
    else:
        with pytest.raises(WeightsLicenceNotAccepted):
            infer.main(argv)
        assert opt_in_stub == []


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


# --- R2 abstention with ensembles and licensed backbones (V-053 item 6) -------------------------------------------------
def _abstention_beside(checkpoint: Path, threshold_target=0.4):
    """An abstention.json for exactly these weights, as a fitted run would hold it, so a single-model infer.py would apply it."""
    from evaluators.abstention import fit_abstention, mapping_fingerprint
    from evaluators.error_model import weights_fingerprint
    rng = np.random.default_rng(0)
    score = rng.uniform(0.5, 1.0, 400)
    rule = fit_abstention(score, rng.random(400) > score, [f'g{i}' for i in range(400)], target_error=threshold_target, num_classes=2,
                          provenance={'weights_sha256': weights_fingerprint(torch.load(checkpoint, weights_only=True)), 'label_mapping_sha256': mapping_fingerprint(LEFT_RIGHT)})
    rule.save(checkpoint.parent / 'abstention.json')


def test_an_ensemble_applies_no_members_abstention(tmp_path, capsys):
    """Each member's rule describes that member alone, so averaged probabilities get no decision column and no abstention message."""
    _videos(tmp_path / 'clips')
    for seed, name in enumerate('ab'):
        _save_run(tmp_path / name, 'siglip2_tiny', seed, 4)
        _abstention_beside(tmp_path / name / 'model_weights.pth')
    out = tmp_path / 'out'
    result = ensemble_runs([tmp_path / 'a', tmp_path / 'b'], tmp_path / 'clips', out, batch_size=1)
    assert 'decision' not in result.columns and 'answered' not in result.columns
    assert 'decision' not in pd.read_csv(out / 'predictions.csv').columns
    assert 'Abstention:' not in capsys.readouterr().out and not (out / 'abstention_summary.json').exists()
    assert json.loads((out / 'ensemble.json').read_text())['abstention'] == 'not applied to ensembles'


def test_no_abstain_is_refused_with_an_ensemble(tmp_path, capsys):
    import infer
    with pytest.raises(SystemExit):
        infer.main(['--modality', 'video', '--ensemble', 'a', 'b', '--data', str(tmp_path), '--out', str(tmp_path / 'o'), '--no_abstain'])
    error = capsys.readouterr().err
    assert 'would be ignored' in error and '--no_abstain' in error


@pytest.mark.parametrize('recorded,passes', [(['CC-BY-NC-4.0'], True), (None, False)])
def test_a_standalone_abstention_fit_follows_the_runs_licence_record(recorded, passes, tmp_path, opt_in_stub):
    """A run on opt-in weights can be calibrated later only under its own recorded acceptance, and abstention.json names the weights' terms."""
    from evaluators.video import abstention_fit
    paths = _videos(tmp_path / 'clips')
    _opt_in_run(tmp_path / 'r' / 'x', **({} if recorded is None else {'accept_weights_licence': recorded}))
    opt_in_stub.clear()
    calibration = tmp_path / 'held_out.parquet'
    pd.DataFrame({'videos': [str(p) for p in paths], 'labels': ['left', 'right'], 'group': ['g0', 'g1']}).to_parquet(calibration, index=False)
    checkpoint = tmp_path / 'r' / 'x' / 'model_weights.pth'
    argv = ['--checkpoint', str(checkpoint), '--calib_df', str(calibration), '--target_error', '0.4', '--split_name', 'held_out']
    if passes:
        assert abstention_fit.main(argv) == 0 and opt_in_stub == ['videomae_b16']
        info = json.loads((checkpoint.parent / 'abstention.json').read_text())
        assert 'CC-BY-NC-4.0' in info['weights_licence'] and info['weights_revision']
        assert info['label_mapping_sha256'] == label_mapping_digest(LEFT_RIGHT)  # names read through the run's verified mapping
    else:
        with pytest.raises(WeightsLicenceNotAccepted):
            abstention_fit.main(argv)
        assert opt_in_stub == []

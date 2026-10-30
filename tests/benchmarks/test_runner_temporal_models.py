"""The benchmark runner: the flow reference on periodic, and the Phase 3 model specs."""
import pandas as pd
import pytest

import scripts.run_motion_benchmark as bench

TINY = ['--n_train', '60', '--n_val', '20', '--n_test', '60', '--size', '32', '--resize', '32', '--codec', 'mp4v', '--threads', '2',
        '--batch_size', '8', '--lr', '2e-3']


def test_flow_probe_separates_periodic(tmp_path):
    """It used the mean flow here, which is about zero for both classes (0.5175). The honesty probe's summary reaches 1.0."""
    out = bench.main(['--task', 'periodic', '--out', str(tmp_path), '--models', 'flow_probe', *TINY])
    assert pd.read_csv(out / 'results.csv')['test_acc'].iloc[0] >= 0.9


def test_flow_probe_still_separates_direction(tmp_path):
    out = bench.main(['--task', 'direction', '--out', str(tmp_path), '--models', 'flow_probe', *TINY])
    assert pd.read_csv(out / 'results.csv')['test_acc'].iloc[0] >= 0.9


def test_every_temporal_spec_runs_and_reports_speed(tmp_path):
    specs = ['late_fusion:resnet_tiny', 'early_fusion:resnet_tiny', 'tsm:resnet_tiny', 'cnn_rnn:resnet_tiny',
             'temporal_transformer:resnet_tiny', 'two_stream:resnet_tiny', 'two_stream@mv:resnet_tiny', 'slow_fusion:dummy_video3d']
    out = bench.main(['--task', 'direction', '--out', str(tmp_path), '--models', *specs, '--epochs', '1', *TINY])
    table = pd.read_csv(out / 'results.csv')
    assert table['model'].tolist() == specs
    assert (table['test_clips_per_s'] > 0).all()
    assert table['test_acc'].between(0, 1).all()


def test_unknown_temporal_spec_is_refused():
    with pytest.raises(SystemExit):
        bench.parse_spec('fancy_fusion:resnet18')


@pytest.mark.parametrize('task,columns', [('scene_bias', ['biased_val_acc', 'unbiased_test_acc']), ('speed', ['test_r2', 'test_mae'])])
def test_other_tasks_run_the_temporal_models(tmp_path, task, columns):
    specs = ['tsm:resnet_tiny', 'two_stream:resnet_tiny']
    out = bench.main(['--task', task, '--out', str(tmp_path), '--models', *specs, '--epochs', '1', *TINY])
    table = pd.read_csv(out / 'results.csv')
    assert table['model'].tolist() == specs
    assert table[columns].notna().all().all()


def test_siglip_gets_its_own_preprocessing(tmp_path):
    """siglip_tiny takes 32 px only, so this fails if the runner's --resize transform reaches it."""
    out = bench.main(['--task', 'direction', '--out', str(tmp_path), '--models', 'mean:siglip_tiny', '--epochs', '1', *TINY, '--resize', '64'])
    assert pd.read_csv(out / 'results.csv')['test_acc'].between(0, 1).all()

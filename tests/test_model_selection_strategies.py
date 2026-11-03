"""select_model.py R3b: the smart, random and bayes strategies, the augmentation axis, overrides, budgets and the regret study.

The runner is handlers/model_selection/simulate.py's SurfaceRunner: it trains nothing, and writes what deeptune.py
--cv_only (and the feature probe, and the final refit) write, through the real run_validation, with each candidate right on
a known share of the rows. Its surface has an interaction (full fine-tuning is best at the low learning rate and worst at
the high one, and the base learning rate is the high one), so one-group-at-a-time tuning alone goes wrong. Everything else
(the sealed split, ledger, checks, ranking, the final step) is production code. CPU only, no clips decoded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import select_model
from handlers.model_selection import simulate
from handlers.model_selection.grid import COLLAPSE_REASON, Candidate, Grid, RunSettings, candidate_argv
from handlers.model_selection.search import STAGE_BIAS, ModelSearch

SEALED, DEV = 'sealed_test.parquet', 'dev.parquet'
try:
    import optuna  # noqa: F401  (optional: only --search bayes needs it)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
NEEDS_OPTUNA = pytest.mark.skipif(not HAS_OPTUNA, reason='--search bayes needs Optuna, an optional dependency (pip install optuna)')
BAYES = pytest.param('bayes', marks=NEEDS_OPTUNA)
GRID = ['--backbones', 'resnet_tiny', 'siglip_tiny', 'vjepa2_tiny', '--train_modes', 'frozen', 'lora:8', 'full',
        '--learning_rates', '1e-3', '1e-4', '--augmentations', 'off', 'aug']


def _argv(data, out, *extra, grid=GRID, search='smart'):
    return ['--search', search, '--df', str(data), '--out', str(out), '--k', '3', '--epochs', '4', '--boot', '30',
            '--gpu_memory_gb', 'none', '--no_pilot', '--profile_clips', '0', '--yes', *grid, *extra]


def _runner(seed=0, **kwargs):
    runner = simulate.SurfaceRunner(simulate.interaction_surface(seed), **kwargs)
    runner.full_epochs_value = 4
    return runner


def _search(argv, runner, say=None, **kwargs):
    args = select_model.build_parser().parse_args(argv)
    search = ModelSearch(select_model.options_from_args(args), runner=runner, footprint_fn=simulate.sim_footprint,
                         check_fn=simulate.quick_check, say=say, **kwargs)
    return search.run()


def _ledger(out):
    return [json.loads(line) for line in (Path(out) / 'ledger.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def _trained(out):
    """Keys of trained jobs, in ledger order (checks' lines left out)."""
    return [entry['key'] for entry in _ledger(out) if entry.get('stage') != 'checks']


def _configs(out):
    """The candidates of trained jobs, in ledger order: unlike keys, these do not change with the seed's data split."""
    return [json.dumps(entry['config'], sort_keys=True) for entry in _ledger(out) if entry.get('stage') != 'checks']


def _decisions(out):
    found = {}
    for decision in json.loads((Path(out) / 'stages.json').read_text()):
        found.setdefault(decision['stage'], []).append(decision['decision'])
    return found


@pytest.fixture()
def data(tmp_path):
    return simulate.make_dataset(tmp_path / 'data')


# -- the augmentation axis ---------------------------------------------------------------------------------------------------
def test_the_augmentation_axis_expands_and_collapses_on_frozen_backbones_with_a_reason():
    grid = Grid(backbones=['resnet18'], train_modes=['frozen', 'full', 'lora:8'], augmentations=['off', 'aug', 'aug_mix'])
    candidates = grid.expand()
    assert [(c.train_mode, c.augmentation) for c in candidates] == [
        ('frozen', 'off'), ('full', 'off'), ('full', 'aug'), ('full', 'aug_mix'), ('lora:8', 'off'), ('lora:8', 'aug'), ('lora:8', 'aug_mix')]
    assert grid.size() == 7 and grid.combinations() == 9
    folded = grid.collapsed()
    assert len(folded) == 2 and all(f['reason'] == COLLAPSE_REASON and 'frozen' in f['runs_as'] for f in folded)
    assert 'folded into others' in grid.formula() and grid.formula().endswith('= 7 candidates')
    settings = RunSettings()
    flags = {c.augmentation + c.train_mode: candidate_argv(c, settings, 'd.parquet', 'o', 'cv') for c in candidates}
    assert '--video_augment' not in flags['offfrozen'] and '--video_augment' not in flags['offfull']
    aug = flags['augfull']
    assert '--video_augment' in aug and aug[aug.index('--video_speed_perturb') + 1] == '0.1' and '--mixup_alpha' not in aug
    mix = flags['aug_mixlora:8']
    assert mix[mix.index('--mixup_alpha') + 1] == '0.2' and mix[mix.index('--cutmix_alpha') + 1] == '1.0' and '--video_flip' not in mix
    # Asking a frozen backbone for augmentation gives the frozen candidate itself, so its ledger key does not change.
    assert Candidate('resnet18', augmentation='aug').replace() == Candidate('resnet18')
    assert 'augmentation' not in Candidate('resnet18').config()  # R3a's keys, so R3a folders resume


def test_flip_values_need_allow_flip(tmp_path):
    base = ['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18', '--train_modes', 'full', '--augmentations', 'off', 'aug_flip']
    with pytest.raises(ValueError, match='left motion into right motion'):
        select_model.options_from_args(select_model.build_parser().parse_args(base))
    options = select_model.options_from_args(select_model.build_parser().parse_args(base + ['--allow_flip']))
    flipped = [c for c in options.grid.expand() if c.augmentation == 'aug_flip']
    assert flipped and '--video_flip' in candidate_argv(flipped[0], options.settings, tmp_path / DEV, tmp_path / 'r', 'cv')


def test_the_default_strategy_is_smart_and_it_searches_augmentation_by_default():
    parse = select_model.build_parser().parse_args
    options = select_model.options_from_args(parse(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18']))
    assert options.strategy == 'smart' and options.grid.augmentations == ['off', 'aug', 'aug_mix']
    options = select_model.options_from_args(parse(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18', '--mode', 'reg']))
    assert options.grid.augmentations == ['off', 'aug']  # mixup is for classification
    for strategy in ('exhaustive', 'random', 'bayes'):
        options = select_model.options_from_args(parse(['--search', strategy, '--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18']))
        assert options.grid.augmentations == ['off']


def test_augmentation_in_deeptune_args_conflicts_with_the_axis_and_is_respected_by_the_default(data, tmp_path):
    parse = select_model.build_parser().parse_args
    with pytest.raises(ValueError, match='use one of them'):
        select_model.options_from_args(parse(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18', '--augmentations', 'off', 'aug',
                                              '--deeptune_args', '--video_augment']))
    options = select_model.options_from_args(parse(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18',
                                                    '--deeptune_args', '--video_augment']))
    assert options.grid.augmentations == ['off']  # the user's own choice for every run stands
    said = []
    assert _search(_argv(data, tmp_path / 's', '--deeptune_args', '--video_augment', grid=['--backbones', 'resnet_tiny']), _runner(),
                   say=said.append) == 0
    assert any("reach every run, frozen backbones included" in line for line in said)


def test_frozen_candidates_get_no_augmentation_and_the_leaderboard_says_so(data, tmp_path):
    out = tmp_path / 's'
    runner = _runner()
    assert _search(_argv(data, out, search='exhaustive'), runner) == 0
    for role, argv in runner.calls:
        if '--freeze-backbone' in argv:
            assert '--video_augment' not in argv and '--mixup_alpha' not in argv
    board = pd.read_csv(out / 'leaderboard.csv')
    assert set(board.loc[board['train_mode'] == 'frozen', 'augmentation']) == {'not searched (frozen-backbone rule)'}
    assert set(board.loc[board['train_mode'] == 'full', 'augmentation']) == {'off', 'aug'}
    selection = json.loads((out / 'selection.json').read_text())
    assert len(selection['augmentation_collapsed']) == 3 * 2  # frozen x aug: per backbone and learning rate


class _Sentinel(Exception):
    pass


def test_augmentation_reaches_training_rows_only_never_validation_held_out_or_test_rows(tmp_path, monkeypatch):
    """deeptune.py builds the training rows' dataset with train=True and every other with train=False; a train=False
    dataset neither augments nor jitters, whatever the options say. So --cv_only folds augment only fold-training rows, and
    the final run never augments the sealed test rows. Mixup and CutMix run inside the training loop only."""
    import deeptune
    import trainers.video.train as video_train
    import evaluators.video.evaluate as video_evaluate
    from datasets.video_datasets import VideoDataOptions, VideoDataset
    candidate = Candidate('resnet_tiny', train_mode='full', augmentation='aug_mix', num_frames=4)
    pd.DataFrame({'videos': [f'{i}.mp4' for i in range(12)], 'labels': ['x', 'y'] * 6}).to_parquet(tmp_path / DEV, index=False)
    argv = candidate_argv(candidate, RunSettings(), tmp_path / DEV, tmp_path / 'run', 'cv')
    args, _, _ = deeptune.parse_args(argv)
    assert args.video_augment and args.video_speed_perturb == pytest.approx(0.1) and args.mixup_alpha == pytest.approx(0.2)
    built = []

    def record(parquet_file=None, *a, **kwargs):
        built.append((Path(parquet_file if parquet_file is not None else kwargs.get('parquet_file')).name, kwargs.get('train', False),
                      kwargs['options'].video_augment))
        if len(built) == 2:
            raise _Sentinel()
        return object()
    monkeypatch.setattr(video_train.ParquetVideoDataset, 'from_parquet', staticmethod(record))
    with pytest.raises(_Sentinel):
        video_train.train(train_df=tmp_path / 'train.parquet', val_df=tmp_path / 'val.parquet', out=tmp_path, freeze_backbone=False,
                          use_peft=False, fixed_seed=True, mode='cls', model_version='resnet_tiny', batch_size=2, num_epochs=1,
                          learning_rate=1e-4, added_layers=2, embed_size=8, num_classes=2, model_str='resnet_tiny', args=args, num_frames=4)
    assert built == [('train.parquet', True, True), ('val.parquet', False, True)]

    built.clear()
    seen = {}

    def record_eval(eval_df, *a, **kwargs):
        seen.update(kwargs)
        raise _Sentinel()
    monkeypatch.setattr(video_evaluate.ParquetVideoDataset, 'from_parquet', staticmethod(record_eval))
    weights = tmp_path / 'w' / 'model.pth'
    weights.parent.mkdir()
    weights.write_bytes(b'')
    with pytest.raises(_Sentinel):
        video_evaluate.evaluate(eval_df=tmp_path / 'eval.parquet', out=tmp_path / 'e', model_weights=weights, args=args, mode='cls',
                                num_classes=2, model_version='resnet_tiny', model_str='resnet_tiny', use_peft=False, added_layers=2,
                                embed_size=8, batch_size=2, freeze_backbone=False, num_frames=4, pooling='mean')
    assert not seen.get('train', False) and seen['options'].video_augment  # evaluation rows: train=False, options notwithstanding

    frame = pd.DataFrame({'videos': ['a.mp4', 'b.mp4'], 'labels': [0, 1]})
    options = VideoDataOptions(video_augment=True, video_speed_perturb=0.1)
    held_out = VideoDataset(frame, num_frames=4, train=False, options=options, check_files=False)
    assert held_out.augment is None and not held_out.spec.temporal_jitter and held_out.spec.speed_perturb == 0
    training = VideoDataset(frame, num_frames=4, train=True, options=options, check_files=False)
    assert training.augment is not None and training.spec.temporal_jitter and training.spec.speed_perturb == pytest.approx(0.1)


def test_mixup_and_cutmix_touch_training_batches_only(tmp_path, monkeypatch):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import trainers.vision.trainer as trainer_module
    from handlers.training_args import TrainingOptions
    torch.manual_seed(0)
    train_x, val_x = torch.randn(8, 2, 3, 8, 8), torch.randn(6, 2, 3, 8, 8)
    train = DataLoader(TensorDataset(train_x, torch.randint(0, 3, (8,))), batch_size=4)
    val = DataLoader(TensorDataset(val_x, torch.randint(0, 3, (6,))), batch_size=3)
    mixed, real_mix = [], trainer_module.mix_batch

    def counting(inputs, *a, **kw):
        mixed.append(inputs.shape[0])
        return real_mix(inputs, *a, **kw)
    monkeypatch.setattr(trainer_module, 'mix_batch', counting)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(2 * 3 * 8 * 8, 3))
    seen = []
    model.register_forward_pre_hook(lambda module, inputs: seen.append((module.training, inputs[0].detach().clone())))
    trainer = trainer_module.Trainer(model, train, val, 1e-3, 'cls', 1, output_dir=tmp_path,
                                     training_options=TrainingOptions(mixup_alpha=0.2, cutmix_alpha=1.0, mix_prob=1.0))
    trainer.fit_error_model = False
    trainer.train()
    assert mixed == [4, 4]  # every training batch, and nothing else
    evaluated = [x for training, x in seen if not training]
    assert evaluated and all(any(torch.equal(x, val_x[i:i + len(x)]) for i in range(0, len(val_x), 3)) for x in evaluated)


# -- every strategy uses development rows only -----------------------------------------------------------------------------------
@pytest.mark.parametrize('strategy', ['smart', 'random', BAYES, 'exhaustive'])
def test_every_strategy_reads_and_passes_development_rows_only(data, tmp_path, monkeypatch, strategy):
    import pyarrow.parquet as pq_module
    runner = _runner()
    state = {'role': None}
    reads = []
    real_pandas, real_arrow = pd.read_parquet, pq_module.read_table

    def tagged(fn):
        def wrapper(path, *a, **kw):
            reads.append((str(Path(path).resolve()) if isinstance(path, (str, Path)) else str(path), state['role'] or 'engine'))
            return fn(path, *a, **kw)
        return wrapper
    monkeypatch.setattr(pd, 'read_parquet', tagged(real_pandas))
    monkeypatch.setattr(pq_module, 'read_table', tagged(real_arrow))
    call, probe = runner.__call__, runner.probe

    class Tagging:
        env = None

        def __call__(self, argv, run_dir, role):
            state['role'] = role
            try:
                return call(argv, run_dir, role)
            finally:
                state['role'] = None

        def probe(self, job, run_dir, info):
            state['role'] = 'probe'
            try:
                assert Path(info['dev']).name == DEV
                return probe(job, run_dir, info)
            finally:
                state['role'] = None
    out = tmp_path / 's'
    extra = ['--n_trials', '6'] if strategy in ('random', 'bayes') else []
    assert _search(_argv(data, out, *extra, search=strategy), Tagging()) == 0
    sealed = str((out / 'data' / SEALED).resolve())
    roles = [role for role, _ in runner.calls]
    assert roles.count('final') == 1 and roles[-1] == 'final'
    for role, argv in runner.calls:
        if role not in ('final', 'probe'):  # a probe gets dev.parquet itself (asserted in Tagging.probe)
            assert SEALED not in ' '.join(argv) and Path(argv[argv.index('--df') + 1]).name == DEV and '--cv_only' in argv
    assert [role for path, role in reads if path == sealed] == ['final']  # read once, by the final run only
    sealed_videos = set(real_pandas(sealed)['videos'])
    for entry in _ledger(out):
        if entry.get('oof_path'):
            oof = pd.read_csv(out / entry['oof_path'])
            assert not set(oof['video_id']) & sealed_videos
    selection = json.loads((out / 'selection.json').read_text())
    if strategy != 'exhaustive':
        assert STAGE_BIAS in selection['notes']


# -- smart: stages, reasons, overrides, budgets ----------------------------------------------------------------------------------
def test_smart_records_a_reason_for_every_stage_and_tunes_train_mode_with_the_learning_rate(data, tmp_path):
    out, runner = tmp_path / 's', _runner()
    said = []
    assert _search(_argv(data, out), runner, say=said.append) == 0
    stages = json.loads((out / 'stages.json').read_text())
    by_stage = _decisions(out)
    assert 'probe' in by_stage['screen protocol (frozen view)'][0]
    assert 'short cross-validation' in by_stage['screen protocol (trainable view)'][0]
    assert 'probe protocol' in by_stage['screen families (frozen view)'][0]
    assert 'screen protocol' in by_stage['screen families (trainable view)'][0] and 'lora:8' in by_stage['screen families (trainable view)'][0]
    assert by_stage['screen families'][0].startswith('kept resnet, siglip; dropped vjepa2')
    assert 'not more than --top_backbones 3' in by_stage['screen backbones'][0]
    # Train mode and learning rate are one group, so the interaction (full is best at the low rate only) is found in stage 2.
    assert by_stage['tune what trains and learning rate (resnet_tiny)'][0].startswith('chose train_mode=full, learning_rate=0.0001')
    assert 'tune augmentation (resnet_tiny)' in by_stage and by_stage['tune augmentation (resnet_tiny)'][0].startswith('chose augmentation=aug')
    assert 'These are the ranking runs themselves' in by_stage['final'][0]
    for decision in stages:
        assert any(decision['decision'] in line for line in said)  # printed as well as recorded
    trained = [entry for entry in _ledger(out) if entry.get('stage') != 'checks']
    assert all(entry['why'] and entry['stage'] for entry in trained)
    assert {entry['protocol'] for entry in trained} == {'probe', 'screen', 'cv'}
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['winner']['config'] == {'backbone': 'resnet_tiny', 'video_arch': 'default', 'pooling': 'default', 'train_mode': 'full',
                                             'learning_rate': 1e-4, 'num_frames': 'default', 'augmentation': 'aug'}
    assert selection['stages'] == stages and STAGE_BIAS in selection['notes']
    assert any('never compete with cv scores' in note for note in selection['notes'])
    board = pd.read_csv(out / 'leaderboard.csv')
    probes = board[board['protocol'] == 'probe']
    assert len(probes) == 3 and probes['rank'].isna().all()  # screening scores are listed, never ranked with cv
    assert list(board.loc[board['rank'].notna(), 'protocol'].unique()) == ['cv']
    assert board.loc[0, 'rank'] == 1 and board.loc[0, 'key'] == selection['winner']['key']


def _frames_surface(runner_up_reveals: bool):
    """A test surface with an interaction between two groups (what trains and learning rate x frames): full fine-tuning at
    the low rate is the best only with 16 frames. With `runner_up_reveals`, that train mode and rate are the joint group's
    runner-up at 8 frames, so the runner-up pair finds it; otherwise they are third, and only the full product does."""
    joint = {('frozen', 1e-3): 0.60, ('frozen', 1e-4): 0.50, ('full', 1e-3): 0.30, ('full', 1e-4): 0.40}
    if runner_up_reveals:
        joint[('frozen', 1e-4)] = 0.35

    def surface(c):
        q = joint[(c.train_mode, c.learning_rate)]
        if c.num_frames == 16:
            q += 0.40 if (c.train_mode, c.learning_rate) == ('full', 1e-4) else -0.05
        return q
    return surface


FRAMES_GRID = ['--backbones', 'resnet_tiny', '--train_modes', 'frozen', 'full', '--learning_rates', '1e-3', '1e-4', '--num_frames', '8', '16',
               '--augmentations', 'off']


def _winner(out):
    return json.loads((Path(out) / 'selection.json').read_text())['winner']['config']


@pytest.mark.parametrize('runner_up_reveals', [True, False])
def test_stage_three_finds_an_interaction_between_groups_only_among_the_combinations_it_tries(data, tmp_path, runner_up_reveals):
    """Stage 3 catches exactly the combinations it tries: the runner-up pair when the two groups' values multiply to more
    than --recheck_full_product, every pair when they multiply to at most that. Here the joint group has 4 values and frames
    2, so 8 combinations: over the default 6, only the runner-up pair is tried."""
    best = {'train_mode': 'full', 'learning_rate': 1e-4, 'num_frames': 16}
    runner = lambda: simulate.SurfaceRunner(_frames_surface(runner_up_reveals), full_epochs_value=4)
    assert _search(_argv(data, tmp_path / 'off', '--recheck_models', '0', grid=FRAMES_GRID), runner()) == 0
    assert {k: _winner(tmp_path / 'off')[k] for k in best} == {'train_mode': 'frozen', 'learning_rate': 1e-3, 'num_frames': 8}
    assert _search(_argv(data, tmp_path / 'default', grid=FRAMES_GRID), runner()) == 0
    found = {k: _winner(tmp_path / 'default')[k] for k in best} == best
    assert found == runner_up_reveals  # the default re-check (runner-up pair) finds it only when the runner-ups reveal it
    decision = _decisions(tmp_path / 'default')['recheck (resnet_tiny)'][0]
    assert 'the runner-up values together (4 x 2 values)' in decision
    assert _search(_argv(data, tmp_path / 'full', '--recheck_full_product', '8', grid=FRAMES_GRID), runner()) == 0
    assert {k: _winner(tmp_path / 'full')[k] for k in best} == best  # every combination: found either way
    assert 'every combination (4 x 2 values)' in _decisions(tmp_path / 'full')['recheck (resnet_tiny)'][0]


def test_smart_screens_with_short_cv_runs_when_asked_and_keeps_the_stage_budgets(data, tmp_path):
    out, runner = tmp_path / 's', _runner()
    assert _search(_argv(data, out, '--screen', 'cv', '--screen_epochs', '1', '--top_families', '1'), runner) == 0
    screens = [entry for entry in _ledger(out) if entry.get('protocol') == 'screen']
    assert len(screens) == 6 and {entry['epochs'] for entry in screens} == {1}  # 3 families x (frozen view + trainable view)
    assert sorted(entry['config']['train_mode'] for entry in screens) == ['frozen'] * 3 + ['lora:8'] * 3
    screened = [argv for role, argv in runner.calls if role == 'cv' and argv[argv.index('--num_epochs') + 1] == '1']
    assert sum('--freeze-backbone' in argv for argv in screened) == 3 and sum('--use-peft' in argv for argv in screened) == 3
    tuned = {entry['config']['backbone'] for entry in _ledger(out) if entry.get('protocol') == 'cv'}
    assert tuned == {'resnet_tiny'}  # --top_families 1, and both views rank resnet first here
    decisions = _decisions(out)
    assert decisions['screen protocol (frozen view)'] == ['short cross-validation runs (1 epochs), as --screen cv asks.']
    assert decisions['screen families'][0].startswith('kept resnet;')


def test_smart_screens_backbones_within_families_when_there_are_more_than_top_backbones(data, tmp_path):
    out = tmp_path / 's'
    grid = ['--backbones', 'resnet_tiny', 'siglip_tiny', 'siglip2_tiny', 'vjepa2_tiny', '--train_modes', 'frozen', 'full',
            '--learning_rates', '1e-4']
    assert _search(_argv(data, out, '--top_families', '2', '--top_backbones', '2', grid=grid), _runner()) == 0
    stages = {d['stage']: d['decision'] for d in json.loads((out / 'stages.json').read_text())}
    assert stages['screen families'].startswith('kept resnet, siglip; dropped vjepa2')
    assert stages['screen backbones'].startswith('kept ') and '(--top_backbones 2 per view)' in stages['screen backbones']
    screened = [(e['config']['backbone'], e['config']['train_mode']) for e in _ledger(out) if e.get('stage') == 'screen backbones']
    # siglip2_tiny (fewer parameters) screened the siglip family; its screens are reused, so only siglip_tiny is new here.
    assert sorted(screened) == [('siglip_tiny', 'frozen'), ('siglip_tiny', 'full')]
    tuned = {e['config']['backbone'] for e in _ledger(out) if e.get('protocol') == 'cv'}
    kept = stages['screen backbones'].split(' (--top_backbones')[0].removeprefix('kept ').split(', ')
    views = [stages[f'screen backbones ({view} view)'] for view in ('frozen', 'trainable')]
    assert tuned == set(kept) and 2 <= len(kept) <= 4  # each view's best 2, united: 2 if they agree, up to 4
    assert all(b in ' '.join(views) for b in kept)


def test_overrides_include_exclude_and_fix(data, tmp_path):
    out = tmp_path / 's'
    assert _search(_argv(data, out, '--top_families', '1', '--include', 'family=vjepa2', '--exclude', 'backbone=siglip_tiny',
                         '--fix', 'learning_rate=1e-4'), _runner()) == 0
    stages = {d['stage']: d['decision'] for d in json.loads((out / 'stages.json').read_text())}
    assert 'plus vjepa2 (--include)' in stages['screen families']
    assert 'learning_rate=0.001' not in ' '.join(stages.values())  # fixed: no stage tried another rate
    entries = [e for e in _ledger(out) if e.get('stage') != 'checks']
    assert {e['config']['learning_rate'] for e in entries} == {1e-4}
    assert 'siglip_tiny' not in {e['config']['backbone'] for e in entries}
    skipped = pd.read_csv(out / 'candidates.csv')
    reasons = set(skipped.loc[skipped['backbone'] == 'siglip_tiny', 'reason'])
    assert reasons == {'excluded by --exclude backbone=siglip_tiny'}
    assert any(r.startswith('not the value fixed by --fix learning_rate') for r in skipped['reason'].dropna())
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['overrides']['include'] == {'family': ['vjepa2']}
    with pytest.raises(ValueError, match='--include'):
        select_model.options_from_args(select_model.build_parser().parse_args(
            ['--search', 'random', '--df', 'x.parquet', '--out', 'o', '--backbones', 'r3d_18', '--include', 'backbone=r3d_18']))
    with pytest.raises(ValueError, match='not a value of the grid'):
        select_model.options_from_args(select_model.build_parser().parse_args(
            ['--df', 'x.parquet', '--out', 'o', '--backbones', 'r3d_18', '--fix', 'learning_rate=0.5']))
    with pytest.raises(ValueError, match='key=value'):
        select_model.options_from_args(select_model.build_parser().parse_args(
            ['--df', 'x.parquet', '--out', 'o', '--backbones', 'r3d_18', '--exclude', 'colour=red']))


@pytest.mark.parametrize('strategy', ['smart', 'random', BAYES])
def test_max_hours_is_a_running_budget_for_adaptive_strategies(data, tmp_path, strategy):
    out = tmp_path / 's'
    runner = _runner(seconds=3600.0)  # every cv training takes an hour; probes a tenth of that
    assert _search(_argv(data, out, '--max_hours', '2.5', search=strategy), runner) == 0
    trained = [e for e in _ledger(out) if e.get('stage') != 'checks']
    assert sum(float(e['seconds']) for e in trained[:-1]) < 2.5 * 3600 <= sum(float(e['seconds']) for e in trained)
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['budget']['stopped_early'].startswith('stopped by --max_hours 2.5')
    assert selection['test'] is not None  # what finished is ranked, and the winner still gets its one sealed score


# -- random and bayes: reproducible, resumable ----------------------------------------------------------------------------------
class _Interrupting:
    """Wraps a runner: Ctrl+C on the n-th cross-validation."""

    def __init__(self, runner, on_call):
        self.runner, self.on_call, self.count = runner, on_call, 0

    def __getattr__(self, name):  # probe and probe_problem: the same screens as an uninterrupted run
        return getattr(self.runner, name)

    def __call__(self, argv, run_dir, role):
        if role == 'cv':
            self.count += 1
            if self.count == self.on_call:
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                raise KeyboardInterrupt
        return self.runner(argv, run_dir, role)


@pytest.mark.parametrize('strategy', ['random', BAYES])
def test_random_and_bayes_are_reproducible_and_resume_the_same_sequence(data, tmp_path, strategy):
    argv = lambda out, *extra: _argv(data, out, '--n_trials', '7', *extra, search=strategy)
    assert _search(argv(tmp_path / 'a'), _runner()) == 0
    assert _search(argv(tmp_path / 'b'), _runner()) == 0
    first, second = _trained(tmp_path / 'a'), _trained(tmp_path / 'b')
    assert len(first) == 7 and first == second  # the same seed, the same sequence
    assert _search(argv(tmp_path / 'c'), _Interrupting(_runner(), 4)) == 130
    assert _trained(tmp_path / 'c') == first[:3]
    resumed = _runner()
    assert _search(argv(tmp_path / 'c'), resumed) == 0
    assert _trained(tmp_path / 'c') == first  # it goes on exactly where it stopped
    assert len([role for role, _ in resumed.calls if role == 'cv']) == 4
    assert _search(argv(tmp_path / 'd', '--seed', '7'), _runner()) == 0
    assert _configs(tmp_path / 'd') != _configs(tmp_path / 'a')  # another seed draws other candidates, or in another order


def test_random_draws_without_replacement_and_stops_when_the_grid_ends(data, tmp_path):
    out = tmp_path / 's'
    assert _search(_argv(data, out, search='random'), _runner()) == 0
    keys = _trained(out)
    runnable = (pd.read_csv(out / 'candidates.csv')['check'] == 'runnable').sum()
    assert len(keys) == len(set(keys)) == runnable
    assert 'every runnable candidate of the grid was drawn' in json.loads((out / 'stages.json').read_text())[-1]['decision']


def test_bayes_proposals_are_runnable_untried_candidates_and_optuna_sees_dev_scores_only(data, tmp_path, monkeypatch):
    optuna = pytest.importorskip('optuna', reason='--search bayes needs Optuna, an optional dependency')
    values = []
    real = optuna.trial.create_trial

    def spy(**kwargs):
        values.append(kwargs.get('value'))
        return real(**kwargs)
    monkeypatch.setattr(optuna.trial, 'create_trial', spy)
    out = tmp_path / 's'
    assert _search(_argv(data, out, '--n_trials', '8', '--exclude', 'train_mode=lora:8', search='bayes'), _runner()) == 0
    entries = [e for e in _ledger(out) if e.get('stage') == 'bayes']
    assert len(entries) == len({e['key'] for e in entries}) == 8 and all(e['config']['train_mode'] != 'lora:8' for e in entries)
    dev_scores = {round(float(np.mean([f['accuracy'] for f in e['folds']])), 10) for e in entries}
    assert {round(v, 10) for v in values if v is not None} <= dev_scores  # only development CV means reach the sampler


def test_bayes_without_optuna_exits_with_an_install_hint_and_the_rest_works(data, tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'optuna', None)  # import optuna now raises ImportError
    with pytest.raises(SystemExit) as raised:
        select_model.main(_argv(data, tmp_path / 'b', search='bayes'), runner=_runner(), footprint_fn=simulate.sim_footprint)
    assert raised.value.code == 2 and 'pip install optuna' in capsys.readouterr().err
    assert not (tmp_path / 'b').exists()
    assert _search(_argv(data, tmp_path / 'r', '--n_trials', '3', search='random'), _runner()) == 0
    assert _search(_argv(data, tmp_path / 's', grid=['--backbones', 'resnet_tiny']), _runner()) == 0


# -- the data profile ------------------------------------------------------------------------------------------------------------
def test_the_profile_reads_development_rows_only_and_suggests_without_changing_anything(tmp_path):
    pytest.importorskip('cv2')
    from datasets.video_synth import translating_clip
    rows = []
    for label, dx in (('left', -3), ('right', 3), ('still', 0)):
        for i in range(8):
            path = translating_clip(tmp_path / f'{label}_{i}.mp4', 6, dx, 0, size=32, seed=i)
            rows.append({'videos': str(path.resolve()), 'labels': label, 'group': f'{label}-{i // 2}'})
    rows[1]['labels'] = 'Left '  # a spelling that differs only in case and spaces
    rows.append({**rows[2], 'labels': 'right'})  # one file listed with two labels
    data = tmp_path / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(data, index=False)
    before = data.read_bytes()
    out = tmp_path / 's'
    said = []
    assert _search(['--search', 'exhaustive', '--df', str(data), '--out', str(out), '--backbones', 'resnet_tiny', '--num_frames', '16',
                    '--k', '2', '--boot', '30', '--gpu_memory_gb', 'none', '--no_pilot', '--dry_run', '--profile_clips', '6'],
                   _runner(), say=said.append) == 0
    profile = json.loads((out / 'profile.json').read_text())
    assert data.read_bytes() == before and 'Nothing was changed' in profile['what']
    facts = profile['facts']
    assert facts['rows'] == json.loads((out / 'data' / 'split.json').read_text())['dev']['rows']  # dev rows, not the input
    from handlers.model_selection.profile import profile_development  # the label checks on a known frame
    whole = profile_development(data, {'unit': "column 'group'"}, RunSettings(k=2), Grid(backbones=['resnet_tiny']), 'cls', 0)
    issues = ' '.join(whole['facts']['label_issues'])
    assert 'differ only in case or spaces' in issues and 'listed with different labels' in issues
    probe = facts['clip_probe']
    assert probe['opened'] == 6 and probe['frames_median'] == 6 and probe['motion_median'] > 0
    suggestions = ' '.join(profile['suggestions'])
    assert 'fewer than the 16 frames the grid asks for' in suggestions
    assert any(line.startswith('  Suggestion: ') for line in said)
    sealed = set(pd.read_parquet(out / 'data' / SEALED)['videos'])
    assert facts['rows'] == len(pd.read_parquet(out / 'data' / DEV)) and sealed  # the sealed rows exist and were not profiled


# -- the regret study -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('surface', list(simulate.SURFACES))
def test_regret_study_on_each_surface(tmp_path, surface):
    """Against exhaustive on a small grid, seeds 0 and 1, on the builder's surface and the critic's A and B. Smart's pick must
    be exhaustive's best on each; on B (frozen features rank backbones opposite to fine-tuning) smart screened on the frozen
    view only must not be. random and bayes get smart's training count; their regret is reported, not required."""
    variants = {'smart': ['--search', 'smart'], 'smart_frozen_view_only': ['--search', 'smart', '--screen_views', 'frozen'],
                'random': ['--search', 'random']}
    if HAS_OPTUNA:
        variants['bayes'] = ['--search', 'bayes']
    rows = simulate.regret_study(tmp_path, [0, 1], GRID, variants=variants, surface_fn=simulate.SURFACES[surface])
    table = pd.DataFrame(rows)
    smart, exhaustive = table[table['strategy'] == 'smart'], table[table['strategy'] == 'exhaustive']
    assert (smart['regret'] <= 1e-9).all() and smart['picked_true_best'].all()
    assert (smart['cv_runs'].to_numpy() < exhaustive['cv_runs'].to_numpy()).all()
    if surface == 'B_screen_misleads':
        frozen_only = table[table['strategy'] == 'smart_frozen_view_only']
        assert (frozen_only['regret'] > 0.05).all() and (frozen_only['true_quality_regret'] > 0.05).all()
    for name in variants:
        if name in ('random', 'bayes'):
            sampled = table[table['strategy'] == name]
            assert (sampled['cv_runs'].to_numpy() <= (smart['cv_runs'] + smart['screen_runs']).to_numpy()).all()
            assert (sampled['regret'] >= 0).all()


def test_smart_with_cheaper_tuning_cross_validates_its_top_candidates_again_under_the_ranking_protocol(data, tmp_path):
    out = tmp_path / 'repro'  # the critic's repro (V-057): one backbone tunes, yet --final_top 3 must give 3 final runs
    assert _search(_argv(data, out, '--tune_epochs', '2', '--top_families', '1', '--final_top', '3'), _runner()) == 0
    final = [e for e in _ledger(out) if e.get('protocol') == 'cv']
    assert len(final) == 3 and len({e['key'] for e in final}) == 3 and {e['stage'] for e in final} == {'final'}
    assert json.loads((out / 'selection.json').read_text())['runner_up'] is not None
    tune = [e for e in _ledger(out) if e.get('protocol') == 'tune']
    scores = sorted((np.mean([f['accuracy'] for f in e['folds']]) for e in tune), reverse=True)
    finals = sorted((np.mean([f['accuracy'] for f in e['folds']]) for e in tune
                     if json.dumps(e['config'], sort_keys=True) in {json.dumps(f['config'], sort_keys=True) for f in final}), reverse=True)
    assert finals == scores[:3]  # the three best of all tune runs, not one per backbone
    out = tmp_path / 's'
    assert _search(_argv(data, out, '--tune_epochs', '2', '--final_top', '2'), _runner()) == 0
    entries = [e for e in _ledger(out) if e.get('stage') != 'checks']
    tune = [e for e in entries if e['protocol'] == 'tune']
    final = [e for e in entries if e['protocol'] == 'cv']
    assert tune and {e['epochs'] for e in tune} == {2}
    assert len(final) == 2 and {e['stage'] for e in final} == {'final'} and {e['epochs'] for e in final} == {4}
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['winner']['key'] in {e['key'] for e in final}  # only ranking-protocol scores pick the winner
    board = pd.read_csv(out / 'leaderboard.csv')
    assert board['rank'].notna().sum() == 2 and set(board.loc[board['protocol'] == 'tune', 'rank'].isna()) == {True}


def test_smart_resumes_with_the_same_decisions_after_an_interruption(data, tmp_path):
    assert _search(_argv(data, tmp_path / 'a'), _runner()) == 0
    assert _search(_argv(data, tmp_path / 'b'), _Interrupting(_runner(), 5)) == 130
    resumed = _runner()
    assert _search(_argv(data, tmp_path / 'b'), resumed) == 0
    assert _trained(tmp_path / 'b') == _trained(tmp_path / 'a')
    decisions = lambda out: [(d['stage'], d['decision']) for d in json.loads((out / 'stages.json').read_text())]
    assert decisions(tmp_path / 'b') == decisions(tmp_path / 'a')
    assert not any(role == 'probe' for role, _ in resumed.calls)  # the screens finished before Ctrl+C and are not run again
    rerun = _runner()
    assert _search(_argv(data, tmp_path / 'b'), rerun) == 0 and rerun.calls == []  # all done: nothing trains, no second sealed score


def test_the_probe_screens_only_frozen_backbones_the_feature_extractor_builds(data, tmp_path):
    from handlers.model_selection.probe import probe_problem
    assert probe_problem(Candidate('vjepa2_tiny', num_frames=4), RunSettings()) is None
    assert probe_problem(Candidate('r3d_18'), RunSettings()) is None
    assert 'torchvision' in probe_problem(Candidate('resnet_tiny'), RunSettings())  # deeptune.py has it; the extractor does not
    assert 'trains (lora:8)' in probe_problem(Candidate('r3d_18', train_mode='lora:8'), RunSettings())

    class NoProbeForSiglip(simulate.SurfaceRunner):
        @staticmethod
        def probe_problem(candidate, settings):
            return 'not extractable here' if candidate.backbone == 'siglip_tiny' else None
    runner = NoProbeForSiglip(simulate.interaction_surface(0))
    runner.full_epochs_value = 4
    out = tmp_path / 's'
    assert _search(_argv(data, out), runner) == 0
    decision = json.loads((out / 'stages.json').read_text())[0]
    assert decision['stage'] == 'screen protocol (frozen view)' and 'siglip_tiny: not extractable here' in decision['decision']
    protocols = {e['protocol'] for e in _ledger(out) if e.get('stage', '').startswith('screen')}
    assert protocols == {'screen'}  # one protocol for the whole screen: never a mix of probe and cv scores


# -- review fix round (ml-critic V-057) -------------------------------------------------------------------------------------------
def test_a_screen_on_other_folds_stops_the_search_before_the_sealed_test(data, tmp_path, capsys):
    """Same folds at every stage: a probe that cross-validated other folds (here, another fold seed) must stop the ranking,
    even though probe scores are never ranked; nothing reaches the sealed test set."""
    class OtherFoldsProbe(simulate.SurfaceRunner):
        def probe(self, job, run_dir, info):
            settings = info['settings']
            from dataclasses import replace
            return super().probe(job, run_dir, {**info, 'settings': replace(settings, seed=settings.seed + 1)})
    runner = OtherFoldsProbe(simulate.interaction_surface(0), full_epochs_value=4)
    out = tmp_path / 's'
    assert _search(_argv(data, out), runner) == 2
    assert 'not cross-validated on the same folds' in json.loads((out / 'selection.json').read_text())['reason']
    assert not (out / 'final.json').exists() and 'final' not in [role for role, _ in runner.calls]


@pytest.mark.parametrize('search, flags', [
    ('smart', ['--n_trials', '5']), ('smart', ['--bayes_startup', '3']), ('exhaustive', ['--n_trials', '5']),
    ('random', ['--top_families', '1']), ('random', ['--bayes_startup', '3']), ('bayes', ['--screen', 'cv']),
    ('exhaustive', ['--tune_epochs', '2', '--final_top', '2']), ('random', ['--screen_views', 'frozen']),
    ('bayes', ['--recheck_full_product', '9']),
])
def test_flags_for_another_strategy_are_refused_not_ignored(search, flags):
    args = select_model.build_parser().parse_args(['--search', search, '--df', 'x.parquet', '--out', 'o', '--backbones', 'r3d_18', *flags])
    with pytest.raises(ValueError, match=f'would be ignored by --search {search}'):
        select_model.options_from_args(args)
    for owner, flag in (('random', ['--n_trials', '5']), ('bayes', ['--n_trials', '5', '--bayes_startup', '3']), ('smart', ['--screen', 'cv'])):
        args = select_model.build_parser().parse_args(['--search', owner, '--df', 'x.parquet', '--out', 'o', '--backbones', 'r3d_18', *flag])
        assert select_model.options_from_args(args).strategy == owner  # their own strategy takes them


def test_an_explicit_probe_screen_that_cannot_run_is_refused_before_anything_trains(data, tmp_path, capsys):
    class NoProbe(simulate.SurfaceRunner):
        @staticmethod
        def probe_problem(candidate, settings):
            return 'not extractable here'
    runner = NoProbe(simulate.interaction_surface(0), full_epochs_value=4)
    said = []
    assert _search(_argv(data, tmp_path / 's', '--screen', 'probe'), runner, say=said.append) == 2
    assert runner.calls == [] and any(line.startswith('Refused: --screen probe cannot run') for line in said)
    said.clear()
    assert _search(_argv(data, tmp_path / 'f', '--screen', 'probe', '--fix', 'train_mode=full'), _runner(), say=said.append) == 2
    assert any('has no frozen train mode' in line for line in said)
    assert _search(_argv(data, tmp_path / 'a', '--screen', 'auto'), runner) == 0  # auto falls back to short runs, and says why
    assert 'not extractable here' in _decisions(tmp_path / 'a')['screen protocol (frozen view)'][0]


def test_resuming_an_r3a_folder_without_search_points_to_the_default_change(data, tmp_path):
    out = tmp_path / 's'
    grid = ['--backbones', 'resnet_tiny', '--train_modes', 'frozen', 'full']
    assert _search(_argv(data, out, grid=grid, search='exhaustive'), _runner()) == 0
    said = []
    args = [a for a in _argv(data, out, grid=grid) if a not in ('--search', 'smart')]
    assert _search(args, _runner(), say=said.append) == 2
    refusal = next(line for line in said if line.startswith('Refused:'))
    assert 'the default is now --search smart' in refusal and 'pass --search exhaustive' in refusal

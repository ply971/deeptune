"""select_model.py (R3a): the exhaustive search's orchestration, with a runner that trains nothing.

The fake runner writes what `deeptune.py --cv_only` writes, through the real `run_validation`, but its predictions are
right on a share of the rows that is a known function of the candidate's config (FakeRunner.quality). So the tests know
which candidate must win without any training, and everything else (checks by deeptune.py's parser, the sealed split,
ledger, ranking, intervals, the final step) is the production code. CPU only, no clips decoded.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import select_model
from handlers.model_selection.checks import Footprint
from handlers.model_selection.grid import Candidate, Grid, RunSettings, candidate_argv, parse_train_mode, train_mode_flags
from handlers.model_selection.ledger import Ledger
from handlers.model_selection.runner import RunResult, result_from_exit
from handlers.model_selection.search import ModelSearch, SearchOptions

SEALED, DEV = 'sealed_test.parquet', 'dev.parquet'


# -- fixtures ------------------------------------------------------------------------------------------------------------
def make_dataset(root: Path, per_class: int = 15, classes=('left', 'right', 'up')) -> Path:
    """Clip paths need not exist: nothing here decodes video."""
    rows = []
    for label in classes:
        for i in range(per_class):
            rows.append({'videos': str((root / f'{label}_{i}.mp4').resolve()), 'labels': label, 'group': f'{label}-src{i // 3}'})
    path = root / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _flag(argv, name, default=None):
    argv = [str(a) for a in argv]
    return argv[argv.index(name) + 1] if name in argv else default


class FakeRunner:
    """Trains nothing. A candidate's predictions are right on the rows whose fixed pseudo-random number is below its quality."""
    QUALITY = {'resnet_tiny': 0.85, 'siglip_tiny': 0.55, 'dummy_video3d': 0.7, 'vjepa2_tiny': 0.4}

    def __init__(self, out_of_memory=(), interrupt_on_call=None, seconds=12.0, quality=None, fold_quality=None, seed_offset=None):
        self.calls, self.reads, self.role = [], [], None
        self.out_of_memory, self.interrupt_on_call, self.seconds = set(out_of_memory), interrupt_on_call, seconds
        self.quality_by_backbone = {**self.QUALITY, **(quality or {})}
        self.fold_quality = dict(fold_quality or {})  # backbone -> one quality per fold, replacing the single one
        self.seed_offset = dict(seed_offset or {})  # backbone -> added to its fold seed, as a bug giving each candidate other folds would

    def quality(self, argv) -> float:
        rate = float(_flag(argv, '--learning_rate'))
        return self.quality_by_backbone[_flag(argv, '--model_version')] - (0.1 if rate > 1e-3 else 0.0)

    def __call__(self, argv, run_dir, role):
        self.calls.append((role, [str(a) for a in argv]))
        if self.interrupt_on_call is not None and len(self.calls) == self.interrupt_on_call:
            Path(run_dir).mkdir(parents=True, exist_ok=True)  # half a run, as Ctrl+C would leave it
            raise KeyboardInterrupt
        if _flag(argv, '--model_version') in self.out_of_memory:
            return RunResult('did_not_fit', 3.0, 'out of memory: torch.OutOfMemoryError: CUDA out of memory.', 1)
        self.role = role
        try:
            if role == 'final':
                self._final(argv, Path(run_dir))
            else:
                self._cv(argv, Path(run_dir))
        finally:
            self.role = None
        return RunResult('ok', self.seconds)

    @staticmethod
    def _correct(n, quality, salt):
        u = (np.arange(n) * 0.6180339887 + salt) % 1.0
        return u < quality

    def _probabilities(self, labels, correct, classes):
        predicted = np.where(correct, labels, (labels + 1) % classes)
        scores = np.full((len(labels), classes), 0.1 / (classes - 1))
        scores[np.arange(len(labels)), predicted] = 0.9
        return scores

    def _cv(self, argv, run_dir):
        from evaluators.video.validation import FoldScores, run_validation
        frame = pd.read_parquet(_flag(argv, '--df'))
        self.reads.append(('cv', str(Path(_flag(argv, '--df')).resolve()), frame['videos'].tolist()))
        codes, names = pd.factorize(frame['labels'], sort=True)
        quality, classes, backbone = self.quality(argv), len(names), _flag(argv, '--model_version')
        right = self._correct(len(frame), quality, 0.0)

        def fit_predict(train_rows, val_rows, eval_rows, name):
            correct = right[eval_rows]
            if backbone in self.fold_quality:
                correct = self._correct(len(frame), self.fold_quality[backbone][int(name.split('_')[1])], 0.0)[eval_rows]
            return FoldScores(self._probabilities(codes[eval_rows], correct, classes))
        seed = int(_flag(argv, '--seed')) + self.seed_offset.get(backbone, 0)
        run_validation('kfold', codes, fit_predict, run_dir / 'deeptune-x' / 'validation_kfold', mode='cls', ids=frame['videos'].tolist(),
                       groups=frame['group'].tolist(), k=int(_flag(argv, '--k')), seed=seed, ci_boot=60)

    def _final(self, argv, run_dir):
        from evaluators.video.report import build_report
        test = pd.read_parquet(_flag(argv, '--test_df'))
        self.reads.append(('final', str(Path(_flag(argv, '--test_df')).resolve()), test['videos'].tolist()))
        codes, names = pd.factorize(test['labels'], sort=True)
        scores = self._probabilities(codes, self._correct(len(test), self.quality(argv), 0.3), len(names))
        report = build_report('cls', codes, scores, test['group'].tolist(), n_boot=60)
        folder = run_dir / 'deeptune-x' / 'eval_output_fake'
        folder.mkdir(parents=True)
        (folder / 'evaluation_report.json').write_text(json.dumps(report, default=float), encoding='utf-8')


def fake_footprint(args, num_classes, measure):
    """Parameter counts that differ per backbone and train mode, without building anything."""
    total = {'resnet_tiny': 300_000, 'siglip_tiny': 2_000_000, 'dummy_video3d': 33_000_000, 'vjepa2_tiny': 800_000}[args.model_version]
    trainable = total if not args.freeze_backbone else 20_000
    return Footprint(total, trainable, activation_bytes_per_clip=total * 10, forward_flops_per_clip=total * 100)


def _argv(data, out, *extra, backbones=('resnet_tiny', 'siglip_tiny')):
    return ['--df', str(data), '--out', str(out), '--backbones', *backbones, '--k', '3', '--epochs', '2', '--boot', '60',
            '--gpu_memory_gb', 'none', *extra]


def _run(argv, runner=None, footprint_fn=fake_footprint):
    return select_model.main(argv, runner=runner if runner is not None else FakeRunner(), footprint_fn=footprint_fn)


def _ledger(out):
    return [json.loads(line) for line in (Path(out) / 'ledger.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


@pytest.fixture()
def data(tmp_path):
    return make_dataset(tmp_path)


# -- grid ----------------------------------------------------------------------------------------------------------------
def test_the_grid_is_the_cartesian_product_and_says_how_big_it_is():
    grid = Grid(backbones=['resnet18', 'r3d_18'], video_archs=['default', 'tsm'], train_modes=['frozen', 'lora:8', 'partial:100k'],
                learning_rates=[1e-4, 1e-3], num_frames=['default', 16])
    candidates = grid.expand()
    assert len(candidates) == grid.size() == 2 * 2 * 1 * 3 * 2 * 2
    assert len({c.config()['backbone'] + json.dumps(c.config(), sort_keys=True) for c in candidates}) == len(candidates)
    assert grid.formula().endswith(f'= {grid.size()} candidates')
    with pytest.raises(ValueError, match='repeats'):
        Grid(backbones=['resnet18', 'resnet18'])
    with pytest.raises(ValueError, match='Unknown train mode'):
        Grid(backbones=['resnet18'], train_modes=['half'])


def test_grid_file_and_flags_combine(tmp_path):
    path = tmp_path / 'grid.json'
    path.write_text(json.dumps({'backbones': ['resnet18'], 'learning_rates': [1e-3, 1e-4]}))
    grid = Grid.from_sources(path, backbones=['r3d_18', 'resnet18'], train_modes=None)
    assert grid.backbones == ['r3d_18', 'resnet18'] and grid.learning_rates == [1e-3, 1e-4] and grid.train_modes == ['frozen']
    path.write_text(json.dumps({'backbone': ['resnet18']}))
    with pytest.raises(ValueError, match='unknown grid keys'):
        Grid.from_sources(path)


def test_extra_deeptune_flags_reach_every_run(tmp_path):
    args = select_model.build_parser().parse_args(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18',
                                                   '--deeptune_args', '--precision', 'bf16', '--video_augment'])
    options = select_model.options_from_args(args)
    assert options.settings.deeptune_args == ['--precision', 'bf16', '--video_augment']
    argv = candidate_argv(Candidate('resnet18'), options.settings, tmp_path / DEV, tmp_path / 'run', 'cv')
    assert argv[argv.index('--precision') + 1] == 'bf16' and '--video_augment' in argv


def test_train_modes_translate_to_deeptune_flags():
    assert parse_train_mode('partial:1.5M') == ('partial', 1_500_000)
    assert train_mode_flags('frozen', 'resnet18') == ['--freeze-backbone']
    assert train_mode_flags('full', 'resnet18') == []
    assert train_mode_flags('full', 'vjepa2_tiny') == ['--unfreeze_backbone']  # frozen by default: full must say so
    assert train_mode_flags('lora:8', 'r3d_18') == ['--use-peft', '--lora_rank', '8']
    assert train_mode_flags('lora:8', 'resnet18') == ['--use-peft', '--embed_size', '8']  # the rank is the head's width there
    assert train_mode_flags('partial:100k', 'resnet18') == ['--trainable_params', '100000']


def test_cv_runs_get_cv_only_and_only_the_final_run_gets_a_test_file(tmp_path):
    candidate, settings = Candidate('resnet18'), RunSettings(k=4)
    cv = candidate_argv(candidate, settings, tmp_path / DEV, tmp_path / 'run', 'cv')
    assert '--cv_only' in cv and _flag(cv, '--k') == '4' and '--test_df' not in cv
    final = candidate_argv(candidate, settings, tmp_path / DEV, tmp_path / 'run', 'final', test_df=tmp_path / SEALED)
    assert _flag(final, '--test_df') == str(tmp_path / SEALED) and '--cv_only' not in final and '--validation' not in final
    with pytest.raises(ValueError):
        candidate_argv(candidate, settings, tmp_path / DEV, tmp_path / 'run', 'final')


# -- checks through deeptune.py's parser -----------------------------------------------------------------------------------
def test_invalid_candidates_are_refused_with_the_parsers_reason_and_listed(data, tmp_path, capsys):
    out, runner = tmp_path / 'search', FakeRunner()
    code = _run(_argv(data, out, '--video_archs', 'default', 'tsm', '--poolings', 'default', 'attention', '--dry_run',
                      backbones=('resnet_tiny', 'vjepa2_tiny')), runner)
    assert code == 0 and runner.calls == []
    candidates = pd.read_csv(out / 'candidates.csv')
    assert len(candidates) == 8
    refused = candidates[candidates['check'] == 'refused'].set_index('candidate')['reason']
    # Each reason is deeptune.py's own text, not a rule repeated here.
    assert refused['resnet_tiny | tsm | attention | frozen | lr=0.0001 | default'].endswith(
        '--pooling attention does not apply to --video_arch tsm: use one of them.')
    assert 'reads its tokens through a probe' in refused['vjepa2_tiny | default | attention | frozen | lr=0.0001 | default']
    assert 'reads its tokens through a probe' in refused['vjepa2_tiny | tsm | default | frozen | lr=0.0001 | default']
    runnable = set(candidates[candidates['check'] == 'runnable']['candidate'])
    assert runnable == {'resnet_tiny | default | default | frozen | lr=0.0001 | default',
                        'resnet_tiny | default | attention | frozen | lr=0.0001 | default',
                        'resnet_tiny | tsm | default | frozen | lr=0.0001 | default',
                        'vjepa2_tiny | default | default | frozen | lr=0.0001 | default'}
    assert {entry['status'] for entry in _ledger(out)} == {'refused'} and len(_ledger(out)) == 4  # never silently dropped
    printed = capsys.readouterr().out
    assert 'refused by checks: 4' in printed and printed.count('  refused: ') == 4


def test_a_combination_only_the_model_builder_refuses_is_refused_by_the_real_builder(data, tmp_path):
    out = tmp_path / 'search'
    code = select_model.main(_argv(data, out, '--video_archs', 'tsm', '--dry_run', '--preflight', 'params',
                                   backbones=('resnet_tiny', 'dummy_video3d')), runner=FakeRunner())
    assert code == 0
    candidates = pd.read_csv(out / 'candidates.csv').set_index('backbone')
    assert candidates.loc['dummy_video3d', 'check'] == 'refused'
    assert 'model build: ValueError: tsm needs a 2D image backbone' in candidates.loc['dummy_video3d', 'reason']
    assert candidates.loc['resnet_tiny', 'check'] == 'runnable' and candidates.loc['resnet_tiny', 'preflight_total_params'] > 0


def test_candidates_over_the_parameter_budget_or_memory_are_skipped_with_the_reason(data, tmp_path):
    out = tmp_path / 'search'
    code = _run(_argv(data, out, '--train_modes', 'frozen', 'full', '--max_trainable_params', '1M', '--dry_run'))
    assert code == 0
    table = pd.read_csv(out / 'candidates.csv').set_index('candidate')
    skipped = table[table['check'] == 'skipped']
    assert list(skipped.index) == ['siglip_tiny | default | default | full | lr=0.0001 | default']
    assert '2,000,000 trainable parameters, over the budget of 1,000,000' in skipped['reason'].iloc[0]
    out2 = tmp_path / 'search2'
    _run(_argv(data, out2, '--gpu_memory_gb', '0.6', '--dry_run'))
    table = pd.read_csv(out2 / 'candidates.csv').set_index('backbone')
    assert table.loc['siglip_tiny', 'check'] == 'skipped' and 'GB of GPU memory' in table.loc['siglip_tiny', 'reason']
    assert table.loc['resnet_tiny', 'check'] == 'runnable'


# -- estimate, dry run and limits ------------------------------------------------------------------------------------------
def test_dry_run_prints_counts_and_trains_nothing(data, tmp_path, capsys):
    out, runner = tmp_path / 'search', FakeRunner()
    assert _run(_argv(data, out, '--learning_rates', '1e-4', '1e-3', '--dry_run'), runner) == 0
    printed = capsys.readouterr().out
    assert runner.calls == [] and not (out / 'runs').exists() and not (out / 'ledger.jsonl').exists()
    assert '= 4 candidates' in printed and 'Trainings to run: 4 candidates x k=3 folds + 1 final refit = 13.' in printed
    assert 'Time estimate: unknown (a dry run trains nothing' in printed
    assert (out / 'data' / 'split.json').is_file()  # the split is made before anything else, and only once


def test_the_estimate_is_printed_after_a_pilot_and_before_any_candidate_trains(data, tmp_path):
    events = []

    class Recording(FakeRunner):
        def __call__(self, argv, run_dir, role):
            events.append(('run', role, _flag(argv, '--num_epochs'), _flag(argv, '--k')))
            return super().__call__(argv, run_dir, role)
    args = select_model.build_parser().parse_args(_argv(data, tmp_path / 'search'))
    search = ModelSearch(select_model.options_from_args(args), runner=Recording(), footprint_fn=fake_footprint,
                         say=lambda text: events.append(('say', text)))
    assert search.run() == 0
    runs = [e for e in events if e[0] == 'run']
    assert runs[0] == ('run', 'pilot', '1', '2')  # cheapest candidate, 1 epoch, 2 folds
    assert [e[1] for e in runs[1:]] == ['cv', 'cv', 'final']
    estimate = next(i for i, e in enumerate(events) if e[0] == 'say' and e[1].startswith('Time estimate: about'))
    trainings = next(i for i, e in enumerate(events) if e[0] == 'say' and e[1].startswith('Trainings to run'))
    first_cv = next(i for i, e in enumerate(events) if e[:2] == ('run', 'cv'))
    assert trainings < estimate < first_cv
    assert 'rough' in events[estimate][1] and 'pilot resnet_tiny' in events[estimate][1]


def test_over_max_hours_is_refused_without_yes(data, tmp_path, capsys):
    out = tmp_path / 'search'
    runner = FakeRunner(seconds=4 * 3600)  # the pilot takes 4 h, so the search would take far longer
    assert _run(_argv(data, out, '--max_hours', '10'), runner) == 3
    assert [role for role, _ in runner.calls] == ['pilot']
    assert 'over --max_hours 10' in capsys.readouterr().out
    runner = FakeRunner(seconds=4 * 3600)
    assert _run(_argv(data, out, '--max_hours', '10', '--yes'), runner) == 0
    assert [role for role, _ in runner.calls] == ['cv', 'cv', 'final']  # the recorded pilot is reused


def test_a_large_run_without_an_estimate_needs_yes(data, tmp_path, capsys):
    argv = _argv(data, tmp_path / 'search', '--learning_rates', *[f'{r}e-4' for r in range(1, 10)], '--no_pilot')
    runner = FakeRunner()
    assert _run(argv, runner) == 3 and runner.calls == []
    assert 'no time estimate' in capsys.readouterr().out


# -- ranking, the sealed test and the record ---------------------------------------------------------------------------------
def test_the_search_ranks_by_cross_validation_and_scores_only_the_winner_on_the_sealed_test(data, tmp_path):
    out, runner = tmp_path / 'search', FakeRunner()
    argv = _argv(data, out, '--learning_rates', '1e-4', '1e-2', '--no_pilot', backbones=('siglip_tiny', 'resnet_tiny', 'vjepa2_tiny'))
    assert _run(argv, runner) == 0
    board = pd.read_csv(out / 'leaderboard.csv')
    assert list(board['rank']) == [1, 2, 3, 4, 5, 6]
    assert board['cv_accuracy_mean'].is_monotonic_decreasing
    # The winner is the configuration the fake made best, found from the scores alone.
    qualities = {(b, lr): runner.quality(['--model_version', b, '--learning_rate', str(lr)])
                 for b in ('siglip_tiny', 'resnet_tiny', 'vjepa2_tiny') for lr in (1e-4, 1e-2)}
    best = max(qualities, key=qualities.get)
    assert (board.loc[0, 'backbone'], board.loc[0, 'learning_rate']) == best
    assert board['cv_accuracy_fold_sd'].notna().all() and (board['oof_accuracy_low'] <= board['cv_accuracy_mean'] + 0.2).all()
    assert set(board['interval_unit']) == {'group'}
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['winner']['key'] == board.loc[0, 'key'] and selection['runner_up']['key'] == board.loc[1, 'key']
    paired = selection['paired_difference']
    assert paired['metric'] == 'accuracy' and paired['unit'] == 'group' and paired['low'] <= paired['point'] <= paired['high']
    assert 'winner' in selection['reason'].lower() or 'Highest-ranked' in selection['reason']
    assert selection['test']['n'] == selection['split']['sealed_test']['rows'] and 'accuracy' in selection['test']['intervals']
    assert selection['sealed_test_uses'] == 1
    assert any("winner's curse" in note for note in selection['notes'])
    assert set(selection['weights_licences']) == {'siglip_tiny', 'resnet_tiny', 'vjepa2_tiny'}
    roles = [role for role, _ in runner.calls]
    assert roles.count('final') == 1 and roles[-1] == 'final'
    assert _flag(runner.calls[-1][1], '--model_version') == best[0]


def test_sealed_test_rows_never_reach_a_candidate_run_or_the_ranking(data, tmp_path, monkeypatch):
    import pyarrow.parquet as pq_module
    reads = []
    runner = FakeRunner()
    real_pandas, real_arrow = pd.read_parquet, pq_module.read_table

    def logged_pandas(path, *args, **kwargs):
        reads.append((str(Path(path).resolve()) if isinstance(path, (str, Path)) else str(path), runner.role or 'engine'))
        return real_pandas(path, *args, **kwargs)

    def logged_arrow(path, *args, **kwargs):
        reads.append((str(Path(path).resolve()) if isinstance(path, (str, Path)) else str(path), runner.role or 'engine'))
        return real_arrow(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', logged_pandas)
    monkeypatch.setattr(pq_module, 'read_table', logged_arrow)
    out = tmp_path / 'search'
    assert _run(_argv(data, out), runner) == 0
    sealed = str((out / 'data' / SEALED).resolve())
    for role, argv in runner.calls:
        if role != 'final':
            assert sealed not in argv and SEALED not in ' '.join(argv)  # not passed
    assert [role for path, role in reads if path == sealed] == ['final']  # read once, by the final run only
    sealed_videos = set(pd.read_parquet(sealed)['videos'])
    for role, path, videos in runner.reads:
        if role == 'cv':
            assert path == str((out / 'data' / DEV).resolve()) and not set(videos) & sealed_videos
    record = json.loads((out / 'data' / 'split.json').read_text())
    assert record['dev']['rows'] + record['sealed_test']['rows'] == 45 and record['seed'] == 42
    assignment = pd.read_csv(out / 'data' / 'split_assignment.csv')
    assert (assignment['part'] == 'sealed_test').sum() == record['sealed_test']['rows']
    dev_groups, test_groups = (set(pd.read_parquet(out / 'data' / name)['group']) for name in (DEV, SEALED))
    assert not dev_groups & test_groups


def test_a_run_that_does_not_fit_is_recorded_and_the_search_goes_on(data, tmp_path):
    out = tmp_path / 'search'
    runner = FakeRunner(out_of_memory={'resnet_tiny'})
    assert _run(_argv(data, out, '--no_pilot', backbones=('resnet_tiny', 'siglip_tiny', 'vjepa2_tiny')), runner) == 0
    statuses = {entry['config']['backbone']: entry for entry in _ledger(out)}
    assert statuses['resnet_tiny']['status'] == 'did_not_fit' and 'out of memory' in statuses['resnet_tiny']['reason']
    assert statuses['siglip_tiny']['status'] == statuses['vjepa2_tiny']['status'] == 'done'
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['winner']['config']['backbone'] == 'siglip_tiny' and selection['counts']['did_not_fit'] == 1
    board = pd.read_csv(out / 'leaderboard.csv').set_index('backbone')
    assert board.loc['resnet_tiny', 'status'] == 'did_not_fit' and pd.isna(board.loc['resnet_tiny', 'rank'])


def test_resume_after_an_interruption_runs_only_the_missing_candidates(data, tmp_path):
    out = tmp_path / 'search'
    argv = _argv(data, out, '--no_pilot', '--learning_rates', '1e-4', '1e-2', backbones=('resnet_tiny', 'siglip_tiny'))
    first = FakeRunner(interrupt_on_call=3)
    assert _run(argv, first) == 130
    finished = _ledger(out)
    assert len(finished) == 2 and {entry['status'] for entry in finished} == {'done'}
    interrupted_key = Path(first.calls[2][1][first.calls[2][1].index('--out') + 1]).name
    assert interrupted_key not in {entry['key'] for entry in finished}
    second = FakeRunner()
    assert _run(argv, second) == 0
    rerun = [_flag(a, '--out') for role, a in second.calls if role == 'cv']
    assert len(rerun) == 2 and Path(rerun[0]).name == interrupted_key
    assert not {Path(path).name for path in rerun} & {entry['key'] for entry in finished}
    assert len({entry['key'] for entry in _ledger(out)}) == 4
    third = FakeRunner()
    assert _run(argv, third) == 0 and third.calls == []  # all done, and the final score is not repeated


def test_a_changed_dataset_or_grid_is_refused_on_resume(data, tmp_path, capsys):
    out = tmp_path / 'search'
    assert _run(_argv(data, out, '--no_pilot')) == 0
    capsys.readouterr()
    runner = FakeRunner()
    assert _run(_argv(data, out, '--no_pilot', backbones=('resnet_tiny', 'siglip_tiny', 'vjepa2_tiny')), runner) == 2
    assert 'the grid differs' in capsys.readouterr().out and runner.calls == []
    assert _run(_argv(data, out, '--no_pilot', '--allow_grid_change', backbones=('resnet_tiny', 'siglip_tiny', 'vjepa2_tiny')), runner) == 0
    assert [_flag(a, '--model_version') for role, a in runner.calls] == ['vjepa2_tiny']  # the winner did not change
    runner = FakeRunner()
    assert _run(_argv(data, out, '--no_pilot', '--epochs', '5'), runner) == 2
    assert 'settings differ' in capsys.readouterr().out
    frame = pd.read_parquet(data)
    frame.loc[0, 'labels'] = 'right'
    frame.to_parquet(data, index=False)
    assert _run(_argv(data, out, '--no_pilot'), runner) == 2
    assert 'the dataset changed' in capsys.readouterr().out and runner.calls == []


def test_the_sealed_test_is_not_reused_for_a_new_winner_unless_asked(data, tmp_path, capsys):
    out = tmp_path / 'search'
    assert _run(_argv(data, out, '--no_pilot', backbones=('siglip_tiny',))) == 0
    runner = FakeRunner()
    assert _run(_argv(data, out, '--no_pilot', '--allow_grid_change', backbones=('siglip_tiny', 'resnet_tiny')), runner) == 0
    assert 'final' not in [role for role, _ in runner.calls] and 'already scored an earlier winner' in capsys.readouterr().out
    runner = FakeRunner()
    assert _run(_argv(data, out, '--no_pilot', '--rescore_sealed_test', backbones=('siglip_tiny', 'resnet_tiny')), runner) == 0
    selection = json.loads((out / 'selection.json').read_text())
    assert [role for role, _ in runner.calls] == ['final'] and selection['sealed_test_uses'] == 2
    assert any('no longer an untouched estimate' in note for note in selection['notes'])


# -- pieces ----------------------------------------------------------------------------------------------------------------
def test_ledger_skips_a_torn_line_and_keeps_appending_whole_lines(tmp_path, capsys):
    ledger = Ledger(tmp_path / 'ledger.jsonl')
    ledger.append({'key': 'a', 'status': 'done', 'config': {}})
    with open(ledger.path, 'a', encoding='utf-8') as stream:
        stream.write('{"key": "b", "status": "do')  # a crash in the middle of a write
    ledger.append({'key': 'c', 'status': 'failed', 'config': {}})
    assert set(ledger.load()) == {'a', 'c'} and ledger.skipped_lines == 1
    assert 'unreadable line' in capsys.readouterr().out
    with pytest.raises(ValueError):
        ledger.append({'key': 'd', 'status': 'maybe'})


def test_exit_codes_become_statuses():
    assert result_from_exit(0, 1.0, '').status == 'ok'
    oom = result_from_exit(1, 1.0, 'Traceback\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB')
    assert oom.status == 'did_not_fit' and 'CUDA out of memory' in oom.reason
    failed = result_from_exit(1, 1.0, 'stuff\nValueError: Number of classes is 3, but --num_classes is 2.\n')
    assert failed.status == 'failed' and failed.reason.startswith('ValueError: Number of classes')


def test_the_real_preflight_counts_what_trains_and_measures_a_clip(data):
    from handlers.model_selection.checks import check_argv, measure_footprint
    footprints = {}
    for mode in ('frozen', 'full'):
        argv = candidate_argv(Candidate('resnet_tiny', train_mode=mode, num_frames=4), RunSettings(), data, data.parent / 'o', 'cv')
        args, reason = check_argv(argv)
        assert reason is None
        footprints[mode] = measure_footprint(args, 3)
    assert footprints['frozen'].total_params == footprints['full'].total_params
    assert footprints['frozen'].trainable_params < footprints['full'].trainable_params == footprints['full'].total_params
    assert footprints['full'].activation_bytes_per_clip > 0 and footprints['full'].forward_flops_per_clip > 0
    assert footprints['full'].cost(4) > footprints['frozen'].cost(4)
    assert footprints['full'].memory_bytes(16) > footprints['full'].memory_bytes(1)


# -- deeptune.py's new flags -------------------------------------------------------------------------------------------------
def _deeptune_argv(data, tmp_path, *extra):
    return ['--modality', 'video', '--df', str(data), '--out', str(tmp_path / 'o'), '--model_version', 'resnet_tiny', *extra]


@pytest.mark.parametrize('extra, message', [
    (('--cv_only',), '--cv_only needs --validation kfold'),
    (('--cv_only', '--validation', 'kfold', '--test_df', '__data__'), 'cannot take --test_df'),
    (('--test_df', 'missing.parquet'), '--test_df must be an existing .parquet file'),
    (('--cv_only', '--validation', 'kfold', '--export_dfanalyze'), 'trains no final model'),
])
def test_deeptune_refuses_bad_cv_only_and_test_df_combinations(data, tmp_path, extra, message, capsys):
    import deeptune
    extra = [str(data) if value == '__data__' else value for value in extra]
    with pytest.raises(SystemExit):
        deeptune.parse_args(_deeptune_argv(data, tmp_path, *extra))
    assert message in capsys.readouterr().err


def test_split_with_test_file_keeps_the_given_test_rows_and_refuses_leaks(data, tmp_path):
    from handlers.split_dataset import split_with_test_file
    frame = pd.read_parquet(data)
    test_mask = frame['group'].isin(['left-src0', 'right-src0', 'up-src0'])
    dev_path, test_path = tmp_path / 'd.parquet', tmp_path / 't.parquet'
    frame[~test_mask].to_parquet(dev_path, index=False)
    frame[test_mask].to_parquet(test_path, index=False)
    train, val, test = split_with_test_file(dev_path, test_path, tmp_path / 'run', True, False, 'labels', 'video', None, 'cls')
    parts = {name: pd.read_parquet(path) for name, path in (('train', train), ('val', val), ('test', test))}
    assert sorted(parts['test']['videos']) == sorted(frame[test_mask]['videos'])
    assert sorted(pd.concat([parts['train'], parts['val']])['videos']) == sorted(frame[~test_mask]['videos'])
    assert not set(parts['train']['group']) & set(parts['val']['group'])
    mapping = json.loads((Path(train).parent / 'label_mapping.json').read_text())
    assert mapping == {'left': 0, 'right': 1, 'up': 2}
    assert dict(zip(frame[test_mask]['videos'], frame[test_mask]['labels'].map(mapping))) == dict(zip(parts['test']['videos'], parts['test']['labels']))
    leaky = pd.concat([frame[test_mask], frame[~test_mask].head(1)])
    leaky.to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match='share 1 .group. groups'):
        split_with_test_file(dev_path, test_path, tmp_path / 'run2', True, False, 'labels', 'video', None, 'cls')
    frame[~test_mask & (frame['labels'] != 'up')].to_parquet(dev_path, index=False)
    frame[test_mask].to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match="classes that --df lacks: \\['up'\\]"):
        split_with_test_file(dev_path, test_path, tmp_path / 'run3', True, False, 'labels', 'video', None, 'cls')


def test_prepare_whole_dataset_encodes_every_row_and_writes_no_test_file(data, tmp_path):
    from handlers.split_dataset import prepare_whole_dataset
    path = prepare_whole_dataset(data, tmp_path / 'run', False, 'labels', 'video', None, 'cls')
    pool = pd.read_parquet(path)
    assert len(pool) == len(pd.read_parquet(data)) and set(pool['labels']) == {0, 1, 2}
    assert not list(Path(path).parent.glob('test_split*')) and json.loads((Path(path).parent / 'split_summary.json').read_text())['strategy'] == 'cv_only'


# -- review fix round (ml-critic, R3a) -----------------------------------------------------------------------------------------
@pytest.mark.parametrize('extra', [
    ['--df', 'full.parquet'], ['--df=full.parquet'], ['--see', '7'], ['--seed', '7'], ['--test_df', 'x.parquet'], ['--cv_only'],
    ['--raw-data'], ['--learning_rate', '0.1'], ['--freeze-backbone'], ['--num_epochs', '99'], ['--config', 'c.yaml'], ['--d', 'x'],
])
def test_extra_flags_that_select_model_controls_are_refused(extra):
    args = select_model.build_parser().parse_args(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18', '--deeptune_args', *extra])
    with pytest.raises(ValueError, match='--deeptune_args'):
        select_model.options_from_args(args)


def test_extra_flags_must_be_separate_words_and_keep_windows_paths(tmp_path):
    args = select_model.build_parser().parse_args(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18',
                                                   '--deeptune_args=--precision bf16'])
    with pytest.raises(ValueError, match='separate words'):
        select_model.options_from_args(args)
    path = 'C:\\Users\\someone\\motion cache'
    args = select_model.build_parser().parse_args(['--df', 'x.parquet', '--out', 'o', '--backbones', 'resnet18',
                                                   '--deeptune_args', '--motion_cache', path])
    settings = select_model.options_from_args(args).settings
    assert settings.deeptune_args == ['--motion_cache', path]
    argv = candidate_argv(Candidate('resnet18'), settings, tmp_path / DEV, tmp_path / 'run', 'cv')
    assert argv[argv.index('--motion_cache') + 1] == path


def test_the_critics_repro_cannot_train_a_candidate_on_the_full_input(data, tmp_path):
    with pytest.raises(SystemExit):
        select_model.main(['--df', str(data), '--out', str(tmp_path / 's'), '--backbones', 'resnet_tiny',
                           '--deeptune_args', '--df', str(data), '--seed', '7'], runner=FakeRunner(), footprint_fn=fake_footprint)
    assert not (tmp_path / 's').exists()


def test_candidates_cross_validated_on_different_folds_are_not_ranked(data, tmp_path, capsys):
    out = tmp_path / 'search'
    assert _run(_argv(data, out, '--no_pilot'), FakeRunner(seed_offset={'siglip_tiny': 5})) == 2
    assert 'not cross-validated on the same folds' in capsys.readouterr().out
    assert not (out / 'final.json').exists()  # nothing reached the sealed test set


def test_ranking_uses_the_mean_over_folds_not_the_best_fold(data, tmp_path):
    out = tmp_path / 'search'
    runner = FakeRunner(fold_quality={'siglip_tiny': [1.0, 0.2, 0.2], 'resnet_tiny': [0.6, 0.6, 0.6]})
    assert _run(_argv(data, out, '--no_pilot'), runner) == 0
    folds = {entry['config']['backbone']: [f['accuracy'] for f in entry['folds']] for entry in _ledger(out)}
    assert max(folds['siglip_tiny']) > max(folds['resnet_tiny']) and np.mean(folds['siglip_tiny']) < np.mean(folds['resnet_tiny'])
    assert json.loads((out / 'selection.json').read_text())['winner']['config']['backbone'] == 'resnet_tiny'


@pytest.mark.parametrize('order', [('siglip_tiny', 'resnet_tiny'), ('resnet_tiny', 'siglip_tiny')])
def test_a_tie_keeps_grid_order(data, tmp_path, order):
    out = tmp_path / 'search'
    assert _run(_argv(data, out, '--no_pilot', backbones=order), FakeRunner(quality={'siglip_tiny': 0.85, 'resnet_tiny': 0.85})) == 0
    board = pd.read_csv(out / 'leaderboard.csv')
    assert board.loc[0, 'cv_accuracy_mean'] == board.loc[1, 'cv_accuracy_mean']
    assert list(board['backbone']) == list(order)


def test_the_paired_point_is_the_winner_minus_the_runner_up_and_the_refit_reads_dev_then_sealed(data, tmp_path):
    out, runner = tmp_path / 'search', FakeRunner()
    assert _run(_argv(data, out, '--no_pilot', backbones=('siglip_tiny', 'resnet_tiny', 'vjepa2_tiny')), runner) == 0
    selection = json.loads((out / 'selection.json').read_text())
    entries = {entry['key']: entry for entry in _ledger(out)}
    winner, runner_up = entries[selection['winner']['key']], entries[selection['runner_up']['key']]
    expected = winner['out_of_fold']['accuracy'] - runner_up['out_of_fold']['accuracy']
    assert expected != 0 and selection['paired_difference']['point'] == pytest.approx(expected)
    assert 'picked on these same rows' in selection['reason']
    role, final = runner.calls[-1]
    assert role == 'final'
    assert Path(_flag(final, '--df')).resolve() == (out / 'data' / DEV).resolve()
    assert Path(_flag(final, '--test_df')).resolve() == (out / 'data' / SEALED).resolve()


def test_the_code_version_is_recorded_and_a_change_is_flagged_on_resume(data, tmp_path, monkeypatch, capsys):
    import handlers.model_selection.search as search_module
    out = tmp_path / 'search'
    monkeypatch.setattr(search_module, 'code_version', lambda: {'commit': 'aaaa', 'commit_dirty': False})
    assert _run(_argv(data, out, '--no_pilot')) == 0
    assert {entry['code']['commit'] for entry in _ledger(out)} == {'aaaa'}
    capsys.readouterr()
    monkeypatch.setattr(search_module, 'code_version', lambda: {'commit': 'bbbb', 'commit_dirty': True})
    assert _run(_argv(data, out, '--no_pilot', '--allow_grid_change', backbones=('resnet_tiny', 'siglip_tiny', 'vjepa2_tiny'))) == 0
    printed = capsys.readouterr().out
    assert 'other code version' in printed and 'aaaa' in printed
    selection = json.loads((out / 'selection.json').read_text())
    assert selection['code']['commit'] == 'bbbb' and selection['code']['commit_dirty'] is True
    assert sorted(selection['code']['candidates_from_other_commits'].values()) == ['aaaa', 'aaaa']
    assert any('other code version' in warning for warning in selection['warnings'])


def test_imbalanced_classes_get_a_warning_about_accuracy(tmp_path, capsys):
    rows = [{'videos': str((tmp_path / f'{label}_{i}.mp4').resolve()), 'labels': label, 'group': f'{label}-src{i // 3}'}
            for label, count in (('left', 30), ('right', 12), ('up', 12)) for i in range(count)]
    data = tmp_path / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(data, index=False)
    assert _run(_argv(data, tmp_path / 'a', '--dry_run')) == 0
    printed = capsys.readouterr().out
    assert 'imbalanced' in printed and '--metric balanced_accuracy' in printed
    assert _run(_argv(data, tmp_path / 'b', '--no_pilot')) == 0
    assert any('imbalanced' in warning for warning in json.loads((tmp_path / 'b' / 'selection.json').read_text())['warnings'])
    capsys.readouterr()
    assert _run(_argv(data, tmp_path / 'c', '--dry_run', '--metric', 'balanced_accuracy')) == 0
    assert 'imbalanced' not in capsys.readouterr().out


def test_the_split_message_names_its_unit_and_warns_when_clips_are_treated_as_independent(tmp_path, capsys):
    rows = [{'videos': str((tmp_path / f'v_{label}_g{i // 3:02d}_c{i % 3 + 1:02d}.avi').resolve()), 'labels': label}
            for label in ('Archery', 'Bowling') for i in range(15)]
    data = tmp_path / 'clips.parquet'
    pd.DataFrame(rows).to_parquet(data, index=False)
    assert _run(_argv(data, tmp_path / 's', '--dry_run')) == 0
    printed = capsys.readouterr().out
    assert 'of the groups' not in printed and 'unit: row' in printed
    assert 'Warning' in printed and 'independent' in printed and 'UCF101' in printed
    assert any('UCF101' in w for w in json.loads((tmp_path / 's' / 'data' / 'split.json').read_text())['warnings'])


def test_a_preflight_that_crashes_is_failed_not_refused_and_is_tried_again(data, tmp_path):
    out = tmp_path / 'search'

    def flaky(args, num_classes, measure):
        if args.model_version == 'siglip_tiny':
            raise OSError('could not download the weights')
        return fake_footprint(args, num_classes, measure)
    assert _run(_argv(data, out, '--no_pilot'), footprint_fn=flaky) == 0
    table = pd.read_csv(out / 'candidates.csv').set_index('backbone')
    assert table.loc['siglip_tiny', 'check'] == 'failed' and 'preflight: OSError: could not download' in table.loc['siglip_tiny', 'reason']
    runner = FakeRunner()
    assert _run(_argv(data, out, '--no_pilot'), runner) == 0
    assert [_flag(a, '--model_version') for role, a in runner.calls if role == 'cv'] == ['siglip_tiny']


def test_test_file_overlap_compares_groups_as_text_and_paths_from_their_file(tmp_path):
    from handlers.split_dataset import split_with_test_file
    rows = pd.DataFrame({'videos': [str((tmp_path / f'{i}.mp4').resolve()) for i in range(30)], 'labels': ['a', 'b'] * 15,
                         'group': [i // 2 for i in range(30)]})
    dev, test = rows[rows['group'] < 12], rows[rows['group'] >= 11].assign(group=lambda f: f['group'].astype(str))
    dev.to_parquet(tmp_path / 'd.parquet', index=False)
    test.to_parquet(tmp_path / 't.parquet', index=False)
    with pytest.raises(ValueError, match="share 1 'group' groups"):
        split_with_test_file(tmp_path / 'd.parquet', tmp_path / 't.parquet', tmp_path / 'r1', True, False, 'labels', 'video', None, 'cls')
    folder = tmp_path / 'data'
    folder.mkdir()
    rows = rows.drop(columns='group')
    relative = rows.head(20).assign(videos=['../' + Path(p).name for p in rows.head(20)['videos']])  # relative to its own file
    relative.to_parquet(folder / 'd.parquet', index=False)
    rows.iloc[[0, 25]].to_parquet(folder / 't.parquet', index=False)  # row 0 is the same file as dev's ../0.mp4
    with pytest.raises(ValueError, match='share 1 video files'):
        split_with_test_file(folder / 'd.parquet', folder / 't.parquet', tmp_path / 'r2', True, False, 'labels', 'video', None, 'cls')


def test_test_file_of_clip_bytes_without_groups_warns_that_overlap_cannot_be_checked(tmp_path):
    from handlers.split_dataset import split_with_test_file
    rows = pd.DataFrame({'videos': [bytes([i]) * 4 for i in range(24)], 'labels': ['a', 'b'] * 12})
    rows.head(18).to_parquet(tmp_path / 'd.parquet', index=False)
    rows.tail(6).to_parquet(tmp_path / 't.parquet', index=False)
    with pytest.warns(UserWarning, match='cannot be checked'):
        train, _, _ = split_with_test_file(tmp_path / 'd.parquet', tmp_path / 't.parquet', tmp_path / 'r', True, False, 'labels', 'video',
                                           None, 'cls', seed=7)
    summary = json.loads((Path(train).parent / 'split_summary.json').read_text())
    assert 'cannot be checked' in summary['overlap_check'] and summary['validation_seed'] == 7


def test_a_ledger_key_belongs_to_one_dataset_and_one_set_of_training_settings():
    """Defence in depth: search.json already refuses changed data or settings, and the key must not match across them either."""
    from handlers.model_selection.ledger import candidate_key
    config, training = Candidate('resnet18').config(), RunSettings().training()
    key = candidate_key(config, 'data-a', training)
    assert candidate_key(config, 'data-b', training) != key
    assert candidate_key(config, 'data-a', {**training, 'epochs': training['epochs'] + 1}) != key
    assert candidate_key(config, 'data-a', dict(training)) == key

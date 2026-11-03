"""Automatic model selection for video: cross-validate candidates of a grid with grouped k-fold on development rows, rank
them, refit the winner and score it once on a sealed test set (R3a, R3b; handlers/model_selection/).

    python select_model.py --df clips.parquet --out search --backbones resnet18 r3d_18 vjepa2_vitl --train_modes frozen lora:8 full
    python select_model.py --df clips.parquet --out search --search exhaustive --grid grid.json --k 5 --yes
    python select_model.py --df clips.parquet --out search --search random --n_trials 20 --grid grid.json

Strategies: smart (default; staged: profile, screen, tune one group at a time, re-check interactions, final CV),
exhaustive (every candidate), random (seeded draws without replacement) and bayes (Optuna's TPE; optional dependency).
All of them share one sealed test set, ledger and resume. Exhaustive grids explode: the candidate count is the product of
the axis lengths, and every candidate trains k times. The run count and a rough time estimate are printed before anything
trains. Rerunning with the same --out resumes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'must be at least 1, got {value}')
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0 < value <= 0.5:
        raise argparse.ArgumentTypeError(f'must be in (0, 0.5], got {value}')
    return value


def build_parser() -> argparse.ArgumentParser:
    from handlers.model_selection.grid import AUGMENTATIONS, CUTMIX_ALPHA, MIX_PROB, MIXUP_ALPHA, SPEED_PERTURB
    from handlers.model_selection.ranking import DEFAULT_METRIC, METRICS
    from handlers.model_selection.search import OVERRIDE_KEYS, STRATEGIES
    from trainers.param_count import parse_param_count
    p = argparse.ArgumentParser(description='Pick the best video model of a grid by grouped cross-validation, with a sealed test set.',
                                formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=False)
    p.add_argument('--search', choices=sorted(STRATEGIES), default='smart',
                   help='smart (default): profile the data, screen backbone families then backbones cheaply, tune one setting group '
                        'at a time on the best, re-check pairs of settings together, then rank the top by grouped CV.\n'
                        'exhaustive: every candidate of the grid.\n'
                        'random: seeded random candidates without replacement, up to --n_trials or --max_hours.\n'
                        'bayes: Optuna TPE over the grid (needs `pip install optuna`), up to --n_trials or --max_hours.\n'
                        'Every strategy selects on development folds only; only the final winner sees the sealed test set.')
    data = p.add_argument_group('data')
    data.add_argument('--df', type=Path, required=True,
                      help='The video dataset, a .parquet file as deeptune.py takes it (or a clip folder / CSV list with --raw-data).')
    data.add_argument('--raw-data', '--raw_data', dest='raw_data', action='store_true',
                      help='--df is a clip folder or a CSV/XLSX list; it is converted to Parquet once, into <out>/data/.')
    data.add_argument('--out', type=Path, required=True,
                      help='Search folder. Rerunning with the same folder resumes: finished candidates are not trained again.')
    data.add_argument('--mode', choices=['cls', 'reg'], default='cls', help='Classification or regression (default cls).')
    data.add_argument('--num_classes', type=_positive_int, default=None, help='Classes (default: counted from the data).')
    data.add_argument('--target', default='labels', help="Target column (default 'labels').")
    data.add_argument('--grouper', default=None,
                      help="Group column: clips of one group stay together in the sealed split and in every fold (default: 'group' if present).")
    data.add_argument('--test_fraction', type=_fraction, default=0.2,
                      help='Share of the groups set aside as the sealed test set before anything trains (default 0.2).')
    grid = p.add_argument_group('grid (a flag replaces the same axis of --grid; "default" leaves the deeptune.py flag out)')
    grid.add_argument('--grid', type=Path, default=None,
                      help='JSON file of axes: {"backbones": [...], "video_archs": [...], "poolings": [...], "train_modes": [...], '
                           '"learning_rates": [...], "num_frames": [...]}.')
    grid.add_argument('--backbones', nargs='+', default=None, metavar='MODEL', help='--model_version values, e.g. resnet18 r3d_18 vjepa2_vitl.')
    grid.add_argument('--video_archs', nargs='+', default=None, metavar='ARCH', help='--video_arch values (default: default).')
    grid.add_argument('--poolings', nargs='+', default=None, metavar='POOLING', help='--pooling values (default: default).')
    grid.add_argument('--train_modes', nargs='+', default=None, metavar='MODE',
                      help='What trains: frozen (default), full, lora:R (LoRA rank R) or partial:COUNT (--trainable_params COUNT).')
    grid.add_argument('--learning_rates', nargs='+', type=float, default=None, metavar='LR', help='Learning rates (default 1e-4).')
    grid.add_argument('--num_frames', nargs='+', default=None, metavar='T', help='Frames per clip (default: default, which is 8).')
    grid.add_argument('--augmentations', nargs='+', default=None, choices=list(AUGMENTATIONS), metavar='AUG',
                      help='Training augmentation, searched like any axis: off; aug (--video_augment with --video_speed_perturb '
                           f'{SPEED_PERTURB}); aug_mix (aug plus mixup {MIXUP_ALPHA} / CutMix {CUTMIX_ALPHA} on {MIX_PROB} of the '
                           'batches, classification only); aug_flip and aug_mix_flip (also --video_flip, only with --allow_flip). '
                           'Applies only where backbone weights train: frozen candidates run once, without augmentation. '
                           'Default: off aug aug_mix for --search smart (off aug for regression), off for the others.')
    grid.add_argument('--allow_flip', action='store_true',
                      help='Allow aug_flip and aug_mix_flip. A mirror turns left motion into right motion: state with this flag that '
                           'your labels do not depend on direction.')
    run = p.add_argument_group('training and scoring')
    run.add_argument('--k', type=_positive_int, default=5, help='Cross-validation folds per candidate (default 5, at least 2).')
    run.add_argument('--metric', default=None,
                     help=f'Ranking metric. Classification: {", ".join(METRICS["cls"])} (default {DEFAULT_METRIC["cls"]}). '
                          f'Regression: {", ".join(METRICS["reg"])} (default {DEFAULT_METRIC["reg"]}, lower is better).')
    run.add_argument('--epochs', type=_positive_int, default=10, help='Epochs per training (default 10).')
    run.add_argument('--early_stopping_patience', type=int, default=3,
                     help='Stop a training after this many epochs without validation improvement (default 3; 0 turns it off).')
    run.add_argument('--batch_size', type=_positive_int, default=16, help='Batch size (default 16).')
    run.add_argument('--num_workers', type=int, default=0,
                     help='DataLoader workers per run (default 0: on this laptop each worker re-imports torch, and runs are sequential).')
    run.add_argument('--seed', type=int, default=42, help='Seed of the sealed split, the folds and training (default 42).')
    run.add_argument('--boot', type=int, default=2000, help='Group-bootstrap resamples for every interval (default 2000).')
    run.add_argument('--accept_weights_licence', nargs='+', default=[], metavar='LICENCE',
                     help='Accept opt-in weights licences by name, as deeptune.py spells them (e.g. CC-BY-NC-4.0). Recorded.')
    run.add_argument('--deeptune_args', nargs=argparse.REMAINDER, default=[], metavar='FLAG',
                     help='More deeptune.py flags for every run, as separate words, LAST on the line: --deeptune_args --precision bf16. '
                          'No shell-style splitting, so Windows paths keep their backslashes. Flags select_model.py sets itself '
                          '(data, split, seed, folds, grid axes, epochs, batch size...) are refused.')
    smart = p.add_argument_group('smart search (--search smart)')
    smart.add_argument('--screen', choices=['auto', 'probe', 'cv'], default='auto',
                       help='How stage 1 screens: probe = cached frozen features + attentive probe (extract_features.py, '
                            'train_feature_head.py); cv = short deeptune.py cross-validation (--screen_epochs). auto (default): '
                            'probe when every screened candidate is a frozen backbone the feature extractor takes, else cv.')
    smart.add_argument('--screen_epochs', type=_positive_int, default=2, help='Epochs of a cv screening run (default 2).')
    smart.add_argument('--top_families', type=_positive_int, default=2,
                       help='Backbone families kept after the first screen, one small model per family (default 2).')
    smart.add_argument('--top_backbones', type=_positive_int, default=3,
                       help='Backbones kept after the second screen, which tune (default 3).')
    smart.add_argument('--tune_epochs', type=_positive_int, default=None,
                       help='Epochs while tuning (default: --epochs, so tuning runs are the final ranking runs). Fewer makes '
                            'tuning cheaper; the --final_top best are then cross-validated again with --epochs.')
    smart.add_argument('--recheck_models', type=int, default=2,
                       help='Tuned models whose pairs of runner-up settings are re-checked together (default 2; 0 turns it off).')
    smart.add_argument('--final_top', type=_positive_int, default=3,
                       help='Candidates in the final grouped cross-validation ranking (default 3).')
    smart.add_argument('--include', nargs='+', default=[], metavar='KEY=VALUE',
                       help='Keep these past the screening cuts, e.g. backbone=r3d_18 or family=vjepa2 (smart only).')
    sampled = p.add_argument_group('random and bayes search')
    sampled.add_argument('--n_trials', type=_positive_int, default=None,
                         help='Candidates to train (default: until --max_hours or the grid is exhausted).')
    sampled.add_argument('--bayes_startup', type=_positive_int, default=None,
                         help='bayes: random trials before TPE models the results (default: a quarter of --n_trials, at least 3; 5 '
                              'without --n_trials).')
    overrides = p.add_argument_group('overrides (every strategy)')
    overrides.add_argument('--exclude', nargs='+', default=[], metavar='KEY=VALUE',
                           help='Skip candidates with this value, e.g. backbone=swin3d_b, train_mode=full or family=videomae. Keys: '
                                f'{", ".join(OVERRIDE_KEYS)}, family. Recorded as skipped with the reason.')
    overrides.add_argument('--fix', nargs='+', default=[], metavar='KEY=VALUE',
                           help='Pin an axis to a value, e.g. learning_rate=1e-4: other values are skipped, and smart does not tune it.')
    limits = p.add_argument_group('limits and checks')
    limits.add_argument('--max_trainable_params', type=parse_param_count, default=None, metavar='COUNT',
                        help='Skip candidates that would train more parameters than this (e.g. 5M). Default: no limit.')
    limits.add_argument('--gpu_memory_gb', default='auto',
                        help='GPU memory to plan for; candidates estimated above it are skipped. auto (default): the visible GPU, '
                             'or no check without one. none: no check.')
    limits.add_argument('--preflight', choices=['full', 'params', 'off'], default='full',
                        help='full (default): build each model once on the CPU to count parameters, FLOPs and activations. '
                             'params: count parameters only. off: no build, so no budget or memory skips and no cost scaling. '
                             'Building loads each backbone\'s weights, downloading them if they are not cached.')
    limits.add_argument('--no_pilot', dest='pilot', action='store_false',
                        help='Do not time a 1-epoch, 2-fold pilot of the cheapest candidate; the time estimate is then unknown.')
    limits.add_argument('--max_hours', type=float, default=8.0,
                        help='exhaustive: refuse to start if the estimate is longer than this, unless --yes. smart, random, bayes: '
                             'a running budget; no new training starts once the trainings in the ledger add up to it (default 8).')
    limits.add_argument('--profile_clips', type=int, default=16,
                        help='Development clips the data profile opens for clip length and a quick motion probe (default 16; 0: none).')
    limits.add_argument('--yes', action='store_true', help='Run even over --max_hours, or a large run without a time estimate.')
    limits.add_argument('--dry_run', action='store_true', help='Check the grid and print the counts and estimate; train nothing, not even the pilot.')
    limits.add_argument('--estimate_only', action='store_true', help='Run the pilot, print the estimate, and stop.')
    limits.add_argument('--allow_grid_change', action='store_true',
                        help='On resume with a changed grid: keep results of candidates still in it, train the new ones.')
    limits.add_argument('--retry_failed', action='store_true', help='On resume, train failed and did-not-fit candidates again.')
    limits.add_argument('--rescore_sealed_test', action='store_true',
                        help='Allow the sealed test set to score a new winner after it scored another. selection.json then says it was reused.')
    return p


def _gpu_memory(text: str):
    if text == 'none':
        return None
    if text != 'auto':
        return float(text)
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return None


def _default_augmentations(args):
    """--search smart tunes augmentation as one more group by default; the other strategies leave it off unless asked."""
    if args.augmentations is not None:
        return args.augmentations
    if args.grid is not None:
        import json
        if 'augmentations' in json.loads(Path(args.grid).read_text(encoding='utf-8')):
            return None  # the grid file's own axis
    if args.search == 'smart':
        from handlers.model_selection.grid import extra_augmentation_flags
        if extra_augmentation_flags(args.deeptune_args):
            return None  # the user set augmentation for every run through --deeptune_args: respected, with a warning
        return ['off', 'aug', 'aug_mix'] if args.mode == 'cls' else ['off', 'aug']
    return None


def options_from_args(args):
    from handlers.model_selection.grid import FLIP_AUGMENTATIONS, AXES, Grid, RunSettings, extra_args_problems, extra_augmentation_flags
    from handlers.model_selection.search import SearchOptions, parse_overrides
    problems = extra_args_problems(args.deeptune_args)
    if problems:
        raise ValueError('--deeptune_args: ' + ' '.join(problems))
    grid = Grid.from_sources(args.grid, backbones=args.backbones, video_archs=args.video_archs, poolings=args.poolings,
                             train_modes=args.train_modes, learning_rates=args.learning_rates, num_frames=args.num_frames,
                             augmentations=_default_augmentations(args))
    flips = sorted(set(grid.augmentations) & set(FLIP_AUGMENTATIONS))
    if flips and not args.allow_flip:
        raise ValueError(f'--augmentations {" ".join(flips)} flips clips, which turns left motion into right motion. Pass --allow_flip '
                         'only if your labels do not depend on direction.')
    if grid.augmentations != ['off'] and extra_augmentation_flags(args.deeptune_args):
        raise ValueError(f'--deeptune_args sets augmentation ({", ".join(extra_augmentation_flags(args.deeptune_args))}) while '
                         '--augmentations searches it: use one of them.')
    include = parse_overrides(args.include, '--include', ('family',))
    exclude = parse_overrides(args.exclude, '--exclude', ('family',))
    fix = parse_overrides(args.fix, '--fix')
    if include and args.search != 'smart':
        raise ValueError("--include keeps values past the smart search's screening cuts; the other strategies have no cuts. "
                         'Use --fix or --exclude.')
    axis_of = {field: axis for axis, field in AXES.items()}
    for key, values in fix.items():
        missing = sorted(map(str, values - set(getattr(grid, axis_of[key]))))
        if missing:
            raise ValueError(f"--fix {key}={missing[0]}: not a value of the grid's {axis_of[key]} axis {getattr(grid, axis_of[key])}.")
    settings = RunSettings(mode=args.mode, num_classes=args.num_classes, target=args.target, grouper=args.grouper, k=args.k,
                           seed=args.seed, epochs=args.epochs, early_stopping_patience=args.early_stopping_patience,
                           batch_size=args.batch_size, num_workers=args.num_workers, boot=args.boot,
                           accept_weights_licence=list(args.accept_weights_licence), deeptune_args=list(args.deeptune_args))
    strategy_options = {name: getattr(args, name) for name in ('screen', 'screen_epochs', 'top_families', 'top_backbones', 'tune_epochs',
                                                               'recheck_models', 'final_top', 'n_trials', 'bayes_startup')}
    return SearchOptions(df=args.df, out=args.out, grid=grid, settings=settings, strategy=args.search, metric=args.metric,
                         include=include, exclude=exclude, fix=fix, strategy_options=strategy_options, profile_clips=args.profile_clips,
                         raw_data=args.raw_data, test_fraction=args.test_fraction, max_trainable_params=args.max_trainable_params,
                         gpu_memory_gb=_gpu_memory(args.gpu_memory_gb), preflight=args.preflight, pilot=args.pilot, dry_run=args.dry_run,
                         estimate_only=args.estimate_only, yes=args.yes, max_hours=args.max_hours, allow_grid_change=args.allow_grid_change,
                         retry_failed=args.retry_failed, rescore_sealed_test=args.rescore_sealed_test)


def main(argv=None, runner=None, footprint_fn=None, probe_runner=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.k < 2:
        parser.error('--k must be at least 2.')
    if args.early_stopping_patience < 0 or args.num_workers < 0 or args.boot < 1:
        parser.error('--early_stopping_patience and --num_workers must be 0 or more, and --boot at least 1.')
    if args.recheck_models < 0 or args.profile_clips < 0:
        parser.error('--recheck_models and --profile_clips must be 0 or more.')
    if args.search == 'bayes':
        from handlers.model_selection.strategies import BayesStrategy
        try:
            BayesStrategy.require_optuna()  # optional: imported only for this strategy
        except ImportError as exc:
            parser.error(str(exc))
    try:
        options = options_from_args(args)
        from handlers.model_selection.search import ModelSearch
        search = ModelSearch(options, runner=runner, footprint_fn=footprint_fn, probe_runner=probe_runner)
    except ValueError as exc:
        parser.error(str(exc))
    return search.run()


if __name__ == '__main__':
    sys.exit(main())

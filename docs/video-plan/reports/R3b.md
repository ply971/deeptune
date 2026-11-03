# R3b: smart, random and bayes search, and augmentation as a search setting

Branch `video/R3b-smart-search`, on top of `video-analysis` 7c7228a (R3a merged). Everything here ran on the CPU, with
the simulated runner of `handlers/model_selection/simulate.py` or untrained tiny models on 32x32 synthetic clips. **No
number in this report describes a real model on real data.** The real-data GPU gate is at the end, for the coordinator to
run with the user. No GUI in this round (G1 comes next).

## What it adds

`select_model.py --search smart|exhaustive|random|bayes` (smart is now the default):

- **smart** (`smart.py`): stage 0 profiles the development rows (`profile.py`); stage 1 screens one small model per backbone
  family, then every backbone of the best families; stage 2 tunes each survivor one group at a time (temporal head, what
  trains, learning rate, frames, augmentation); stage 3 re-checks pairs of runner-up values together and, if one wins, runs
  one more pass from it; stage 4 ranks the top 3 by grouped CV. The winner goes to the sealed test exactly as in R3a.
- **random** (`strategies.py`): a seeded permutation of the grid, without replacement, up to `--n_trials` / `--max_hours`.
- **bayes** (`strategies.py`): Optuna TPE (multivariate) over the axes as categorical choices, one trial at a time, same
  budgets. Optuna is optional (`pip install optuna`, or the new `search` extra in `pyproject.toml`).
- **`--augmentations`** `off | aug | aug_mix | aug_flip | aug_mix_flip`, in every strategy (`grid.py`).
- **`--include` / `--exclude` / `--fix key=value`**, **`--profile_clips`**, smart's budget knobs, and `--max_hours` as a
  running budget for the adaptive strategies.

## Decisions

**One engine, so no strategy can bypass the sealing.** R3a's `batches(runnable, ledger)` became `jobs(ctx)`: a strategy
yields jobs, each a checked candidate plus a protocol (`jobs.py`), made by `StrategyContext.job()`. The context builds the
command line itself with `candidate_argv(..., role='cv')` on `data/dev.parquet`; it has no handle on the sealed file, the
final run or the runner, and serves back only ledger entries and development-fold means. The engine (`search.py`) runs the
jobs, skips those the ledger holds (resume), applies the budget, appends the ledger, checks folds and ranks. The final
refit and the one sealed-test score are R3a's code, unchanged. The sealing test of R3a (every Parquet read logged, the
sealed file read once, by the final run) now runs for all four strategies, and also checks that no out-of-fold row of any
job is a sealed row.

**Protocols, and how screening scores are compared.** Only the ranking protocol `cv` (deeptune.py `--cv_only`, the
search's `--epochs`, early stopping) picks the winner; its ledger keys are R3a's, so R3a folders resume. `screen` (fewer
epochs), `tune` (`--tune_epochs`) and `probe` (cached features + attentive probe) have their own keys (the protocol is in
the hash). A stage ranks only jobs of one protocol (`ctx.rank` refuses a mix). The leaderboard lists other protocols' rows
with a `protocol` column and no rank, and `selection.json` notes that they never compete with cv scores.

**Same folds at every stage.** All protocols use the same development file and seed. The probe reproduces deeptune.py's
folds by calling the same `prepare_whole_dataset` (label encoding), the same group choice and the same `run_validation`;
the engine's fold check now covers screening and tuning entries too, so a probe on other folds would stop the ranking.
The real e2e test asserts that probe and deeptune.py entries share one fold signature.

**Validity.** Every smart stage, and every random or bayes step, selects on the same development folds as the final
ranking, so the choices are optimistically biased as well as the winner's score. The output and `selection.json` say
so (`STAGE_BIAS`); the sealed test stays the honest number.

**Screening.** `--screen auto` uses the probe when every screened candidate is a frozen backbone `extract_features.py`
accepts (its own `check_feature_input`) and a probe runner exists; otherwise short deeptune.py CV (`--screen_epochs`, 2).
The choice and its reason are a recorded decision. A probe scores the backbone's frozen features under an attentive
probe, not the candidate's own head: it ranks backbones within the screen only. Features are cached per backbone and
frame count under `<out>/features/`, extracted from `dev.parquet` only.

**Smart's base settings and groups.** The first value of every axis, except that `frozen` and augmentation `off` are
preferred when present, and `--fix` wins. The family representative is the backbone with the fewest parameters by the
preflight. Groups: temporal head (`video_arch` x `pooling`), what trains, learning rate, frames, augmentation. A group
whose axis has one value is silent; one with a single runnable value (collapsed augmentation, `--fix`, refusals) records
why. Default budgets: `--top_families 2`, `--top_backbones 3`, `--recheck_models 2`, `--final_top 3`; a screen that
would keep everything is skipped and recorded.

**The re-check (stage 3).** For each pair of tuned groups, the tuned best with both groups at their runner-up values.
On the regret surface (full fine-tuning best at the low learning rate, worst at the high base one) one-group-at-a-time
picks `frozen` at the high rate; the pair (`lora:8`, low rate) beats it, and the extra pass then finds `full` and
augmentation. Without stage 3 the same search keeps `frozen` (see the regret results).

**Final stage.** When tuning runs under the ranking protocol (the default), the top 3 have already been cross-validated
on the same folds with the same seed; re-running them would reproduce the same numbers, so they are not trained again
and the decision says so. With `--tune_epochs`, the top `--final_top` are cross-validated again under the ranking
protocol and only those are ranked.

**Random and bayes resume.** Random permutes the whole grid (refused and skipped candidates included, so a changed check
does not shift the order) and replays it. Bayes stores no study: at each step it builds a fresh Optuna study from the
trials it has run so far (scores read from the ledger) with a sampler seeded from (`--seed`, step, attempt), and asks one
trial. An interrupted run therefore replays the same proposals and continues exactly as an uninterrupted one; the ledger
stays the only record. Proposals that are not runnable or already tried are asked again with the next attempt seed; after
25 attempts the step takes the next untried candidate of a seeded permutation, and says so in the job's `why`. Failed
trials are given to TPE as FAIL. Optuna only receives development CV means (a test spies on `create_trial`). It does not
warm-start from other strategies' results in the same folder (it reuses them if it proposes them).

**Augmentation.** `aug` = `--video_augment --video_speed_perturb 0.1`; `aug_mix` adds `--mixup_alpha 0.2 --cutmix_alpha
1.0 --mix_prob 0.5`. On a frozen candidate the axis collapses to `off` (`Candidate.replace`), each folded combination is
listed (`Grid.collapsed`, printed, `selection.json`), and the leaderboard's augmentation column says `n/a (frozen
backbone)`. A candidate with augmentation off keeps R3a's config and ledger key. Flips need `--allow_flip`, and deeptune.py's
own benchmark flip guard still applies through the parser check. Where augmentation can reach: deeptune.py builds the
training rows' dataset with `train=True` and the validation, fold-held-out and evaluation datasets with `train=False`, and
a `train=False` dataset neither augments nor jitters whatever the options say; mixup and CutMix run inside the training
loop only. Tests pin all three. The final refit trains with the winner's augmentation on its training rows; the sealed
rows are only evaluated.

**`--max_hours`.** Exhaustive keeps R3a's start gate. For smart, random and bayes it is a running budget: before each new
training, the seconds recorded in the ledger are summed, and no new training starts once they reach it; what finished is
ranked and the winner still gets its sealed score (`budget.stopped_early`).

**Data profile (stage 0, every strategy).** Reads `dev.parquet` only; opens a seeded sample of `--profile_clips` (16)
clip files with OpenCV for frame count, fps, size and a motion number (Farneback flow on 8 grey 64-px frames, from
`benchmarks/motion_features.py`, as a share of the frame width per frame). Label checks: a file with two labels, the same
file twice without trims, labels differing only in case or spaces, classes with fewer groups than k, groups with several
labels. Suggestions only; it changes nothing. The motion thresholds (0.002 and 0.02) are rules of thumb and say so.

## Deviations from the request

- **R3a's tests now pass `--search exhaustive`** (the `_argv` helper and the e2e command line), because the default
  changed to smart as requested. No assertion was changed or removed.
- **Smart's default augmentations** are `off aug aug_mix` (`off aug` for regression); the other strategies default to
  `off`, so R3a's grids keep their size. If `--deeptune_args` already sets augmentation, smart's default stays `off` and
  the run warns that those flags reach every run.
- **`--include`** is smart-only (other strategies have no cuts; they refuse it). For axes other than backbone and family it
  is recorded as having no effect, since stage 2 tries every value anyway.
- **The final stage** does not re-run the top 3 when tuning used the ranking protocol (see Decisions).
- **The regret study's budget** for random and bayes is smart's trainings including probe screens, which favours them a
  little (a probe costs less than a cross-validation).

## Regret study (CPU, simulated)

`python scripts/study_search_regret.py --seeds 0 1 2 3 4 5 6 7` (CPU, about 4 minutes). Grid: 3 backbones (resnet_tiny,
siglip_tiny, vjepa2_tiny; 3 families) x train modes frozen, lora:8, full x learning rates 1e-3, 1e-4 x augmentation off,
aug = 36 combinations, 30 candidates after the frozen collapse. k = 3, 45 clips in 15 groups, seed per run = the study
seed (split, folds, surface jitter, sampling). Surface (`simulate.interaction_surface`): backbone 0.55 / 0.50 / 0.45;
(train mode, learning rate) frozen 0 / -0.05, lora -0.03 / +0.08, full -0.20 / +0.20 (high / low rate; the high rate is
the base); augmentation +0.04 where weights train; +-0.02 jitter per candidate; screens and probes add +-0.04 noise.
Regret = exhaustive's best CV accuracy minus exhaustive's CV accuracy of the pick. random and bayes got smart's training
count (17) as `--n_trials`.

| strategy | seeds | mean regret | max regret | picked the true best | CV runs | screen (probe) runs |
|---|---|---|---|---|---|---|
| exhaustive | 8 | 0 | 0 | 8/8 | 30 | 0 |
| smart | 8 | 0 | 0 | 8/8 | 14 | 3 |
| smart without stage 3 (`--recheck_models 0`) | 8 | 0.267 | 0.306 | 0/8 | 8 | 3 |
| random (17 trials) | 8 | 0.024 | 0.139 | 6/8 | 17 | 0 |
| bayes (17 trials) | 8 | 0 | 0 | 8/8 | 17 | 0 |

Reading it: one group at a time from the base settings is misled by the interaction on every seed (it keeps `frozen` at the
high rate, 27 points worse); stage 3's pairwise re-check and the pass after it recover the true best on every seed, with
17 trainings against exhaustive's 30. Bayes with the same budget also found it every time; random missed it on 2 of 8.
**Caveats:** the surface was written to have exactly the kind of interaction stage 3 checks (a pair of groups), so smart's
perfect score here is a check that the mechanism works, not evidence that it beats bayes in general. A three-way
interaction, or a screen that ranks backbones differently from fine-tuning (frozen features of one family good, its
fine-tuning poor), would defeat it; the real-data gate below is the test that counts. The unit test
(`test_regret_study_smart_recovers_the_interaction_that_one_group_at_a_time_misses`) runs seeds 0-2 and asserts smart's
zero regret with fewer runs than exhaustive, and over 5 points of regret without stage 3.

## Tests

`tests/test_model_selection_strategies.py` (29 tests, CPU, simulated runner unless stated): the augmentation axis expands,
collapses on frozen backbones with a reason, and keeps R3a's keys; flips need `--allow_flip`; smart is the default and
searches augmentation by default; augmentation in `--deeptune_args` conflicts with the axis and is respected by the
default; frozen candidates never get augmentation flags and the leaderboard says `n/a`; **augmentation reaches training
rows only** (the real `train()` builds training rows with `train=True` and validation with `train=False`, the real
`evaluate()` builds evaluation rows without `train`, a `train=False` dataset neither augments nor jitters, and mixup /
CutMix run on training batches only, through the real Trainer); **every strategy (smart, random, bayes, exhaustive) reads
and passes development rows only** (Parquet reads logged, the sealed file read once by the final run, no out-of-fold
row is sealed); smart records and prints a reason for every stage, is misled by one group at a time and recovers through
stage 3, and without stage 3 picks the wrong candidate; the cv screen, `--screen_epochs` and `--top_families`; backbone
screening within families reuses the family screens; `--include` / `--exclude` / `--fix`; `--max_hours` as a running budget
(smart, random, bayes); random and bayes are reproducible and resume the same sequence after Ctrl+C; random draws without
replacement; bayes proposes runnable untried candidates and Optuna sees development CV means only; bayes without Optuna
exits with the install hint while the other strategies work; the profile reads development rows only and changes nothing;
`--tune_epochs`; smart resumes with the same decisions; the probe screens only extractor-built frozen backbones, and a
screen never mixes probe and cv; the regret study (seeds 0-2).

`tests/test_model_selection_e2e.py` gained `test_smart_search_end_to_end_with_a_real_feature_probe_screen` (`slow`): real
extract_features.py, train_feature_head.py and deeptune.py subprocesses on siglip_tiny and vjepa2_tiny, 4 frames, k = 2,
1 epoch: probe screens of both families, tuning, the final refit; probe and deeptune.py entries share one fold signature,
augmentation only on trainable candidates, no out-of-fold row sealed, the final test split is the sealed file.

Results (CPU, `CUDA_VISIBLE_DEVICES=-1`, from the worktree):

- `tests/test_model_selection_strategies.py tests/test_model_selection.py`: `82 passed, 2 warnings in 249.42s`
- `tests/test_model_selection_e2e.py`: `2 passed in 343.38s`
- `tests/test_video_followups.py tests/test_small_dataset_splits.py`: `26 passed in 28.79s`

Edits to existing tests: R3a's `_argv` helper in `tests/test_model_selection.py` and the R3a e2e command line now start with
`--search exhaustive`, because the default strategy changed. No assertion was changed or removed.

## Not done

- Any GUI (G1, next). `select_model.py` keeps one parser with help on every flag, and writes `selection.json`,
  `stages.json`, `profile.json`, `leaderboard.csv` and `candidates.csv` for it.
- Real-data runs of smart, random and bayes (the gate below). The regret numbers above are from a simulated surface.
- Nested CV, or a correction of the stages' selection bias: stated, not corrected.
- Bayes does not warm-start from other strategies' runs in the same folder; random and bayes do not use the probe.
- The time estimate of the adaptive strategies is an upper bound built from representative candidates, not calibrated.
- The full suite was not run (instructions: the laptop runs out of memory). Targeted files above.

## Real-data GPU gate (coordinator, with the user)

From the repo root, GPU visible, on the UCF101 10-class demo manifest (330 clips, `group` column). Steps 1 to 3 are the
regret check on real data: exhaustive is the reference, then each strategy on the same folder data and seed. Record
each step's wall time, its trainings (`selection.json` -> `budget.trainings_recorded`), its pick, and the pick's CV score in
the exhaustive leaderboard.

```powershell
$data = "C:\Users\Ahmed Magdy\deeptune-demo\train_clips.csv"
$grid = @('--backbones','r3d_18','r2plus1d_18','resnet18','--train_modes','frozen','lora:8','full','--learning_rates','1e-3','1e-4','--augmentations','off','aug','--num_frames','16','--k','5','--epochs','5','--batch_size','4')
$extra = @('--deeptune_args','--precision','bf16')  # always last on the line
# 0. Counts and the profile only; trains nothing
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3b_smart @grid --dry_run @extra
# 1. The reference: every candidate (add --yes if the estimate is over 8 h)
.venv\Scripts\python.exe select_model.py --search exhaustive --df $data --raw-data --out .artifacts\gate\r3b_exhaustive @grid --yes @extra
# 2. Smart (the default). Press Ctrl+C once mid-run, rerun the same line, and check stages.json is unchanged and nothing re-trains
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3b_smart @grid --max_hours 24 @extra
# 3. Random and bayes with smart's training count N (from step 2's selection.json)
.venv\Scripts\python.exe select_model.py --search random --df $data --raw-data --out .artifacts\gate\r3b_random @grid --n_trials N --max_hours 24 @extra
.venv\Scripts\python.exe select_model.py --search bayes --df $data --raw-data --out .artifacts\gate\r3b_bayes @grid --n_trials N --max_hours 24 @extra
# 4. The probe screen on its own: smart with frozen candidates only, to time feature extraction against short CV
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3b_probe --backbones r3d_18 r2plus1d_18 resnet18 vjepa2_vitl --top_families 1 --train_modes frozen --num_frames 16 --k 5 --epochs 5 --batch_size 4 @extra
```

Check also: the leaderboard's probe/screen rows have no rank; `n/a (frozen backbone)` on frozen rows; each `final/<key>/.../split_summary.json` names `sealed_test.parquet`; and in step 2 the printed upper bound against the trainings actually run.

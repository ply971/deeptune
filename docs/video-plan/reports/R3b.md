# R3b: smart, random and bayes search, and augmentation as a search setting

Branch `video/R3b-smart-search`, on top of `video-analysis` 7c7228a (R3a merged). Everything here ran on the CPU, with
the simulated runner of `handlers/model_selection/simulate.py` or untrained tiny models on 32x32 synthetic clips. **No
number in this report describes a real model on real data.** The real-data GPU gate is at the end, for the coordinator to
run with the user. No GUI in this round (G1 comes next).

## What it adds

`select_model.py --search smart|exhaustive|random|bayes` (smart is now the default):

- **smart** (`smart.py`): stage 0 profiles the development rows (`profile.py`); stage 1 screens one small model per backbone
  family, then every backbone of the best families, in a frozen and a trainable view (fix round); stage 2 tunes each
  survivor one group at a time (temporal head, what trains with the learning rate, frames, augmentation); stage 3
  re-checks pairs of groups (every combination of small pairs, the runner-up pair of larger ones) and, if one wins, runs
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
command line itself with `candidate_argv(..., role='cv')` on `data/dev.parquet`, and serves back only ledger entries and
development-fold means; the interface offers no way to the sealed file, the final run or the runner. That is a convention,
not a sandbox: the context holds the engine (`_search`), which a strategy could reach around the interface. The sealing
test enforces the convention for every shipped strategy. The engine (`search.py`) runs the
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

**Screening, in two views (fix round).** Stage 1 screens each backbone in a frozen view (base settings, frozen) and,
when the grid has a trainable mode, a trainable view (base settings at the grid's first trainable mode). Each view keeps
its own `--top_families` / `--top_backbones`, and the survivors are their union. Why: frozen features can rank backbones
in another order than fine-tuning (the critic's surface B), and which kind of training wins is only decided in stage 2,
so the screen should not presume it. Alternatives considered: screening only with a short fine-tune when the grid has
trainable modes (cheaper, but it misleads in the opposite direction when a frozen backbone wins, as on small data), or one
combined ranking (it hides which view a backbone was kept for). Cost: one more screen per backbone, and up to twice the
survivors when the views disagree; none when they agree. `--screen_views frozen|trainable` keeps one view. The frozen
view uses the probe when `--screen auto` finds every screened backbone probe-able (the extractor's own
`check_feature_input` and the names `SnippetBackbone` builds), else short deeptune.py runs (`--screen_epochs`, 2); an
explicit `--screen probe` that cannot run is refused before anything trains. The trainable view always uses short runs.
Each view's protocol and its reason are a recorded decision, `--screen cv` included. A probe scores the backbone's frozen
features under an attentive probe, not the candidate's own head: it ranks backbones within its view only. Features are
cached per backbone and frame count under `<out>/features/`, extracted from `dev.parquet` only; the probe's ledger key
includes its epochs and learning rate.

**Smart's base settings and groups.** The first value of every axis, except that `frozen` and augmentation `off` are
preferred when present, and `--fix` wins. The family representative is the backbone with the fewest parameters by the
preflight. Groups: temporal head (`video_arch` x `pooling`), what trains together with the learning rate (one group
since the fix round: the rate that suits a train mode depends on it, the most common interaction in fine-tuning; the
critic's surface A showed that separate groups miss it), frames, augmentation. A group
whose axis has one value is silent; one with a single runnable value (collapsed augmentation, `--fix`, refusals) records
why. Default budgets: `--top_families 2`, `--top_backbones 3`, `--recheck_models 2`, `--final_top 3`; a screen that
would keep everything is skipped and recorded.

**The re-check (stage 3).** For each pair of tuned groups on the `--recheck_models` best tuned models: when the
values each group's sweep finished multiply to at most `--recheck_full_product` (default 6), every combination of them,
with the rest at the tuned best; otherwise only the two runner-up values together. If one beats the tuned best, it is
adopted and one more one-group-at-a-time pass runs from it. **It catches exactly the combinations it tries.** An
interaction among three groups, or one that needs a third-best value of a pair over the threshold, is missed: the test
`test_stage_three_finds_an_interaction_between_groups_only_among_the_combinations_it_tries` shows both sides (what trains
and rate x frames, 4 x 2 = 8 combinations: missed at the default 6 when the runner-ups do not reveal it, found at 8).
The threshold 6 keeps a pair's cost at most one extra train-mode-and-rate sweep for a two-value axis; the plan and the
estimate count it.

**Final stage.** When tuning runs under the ranking protocol (the default), the tuning runs are the ranking runs
themselves, so nothing trains again and the decision says so. With `--tune_epochs`, the `--final_top` best of all tune
runs (across backbones; the first R3b kept one per backbone, a bug the critic found) are cross-validated under the ranking
protocol. Either way the engine ranks every ranking-protocol entry of the `--out` folder, runs of another strategy in the
same folder included, and the decision says that too.

**Random and bayes resume.** Random permutes the whole grid (refused and skipped candidates included, so a changed check
does not shift the order) and replays it. Bayes stores no study: at each step it builds a fresh Optuna study from the
trials it has run so far (scores read from the ledger) with a sampler seeded from (`--seed`, step, attempt), and asks one
trial. An interrupted run therefore replays the same proposals and continues exactly as an uninterrupted one; the ledger
stays the only record. Proposals that are not runnable or already tried are asked again with the next attempt seed; after
25 attempts the step takes the next untried candidate of a seeded permutation, and says so in the job's `why`. Failed
trials are given to TPE as FAIL. Optuna only receives development CV means (a test spies on `create_trial`). It does not
warm-start from other strategies' results in the same folder (it reuses them if it proposes them).

**Augmentation.** `aug` = `--video_augment --video_speed_perturb 0.1`; `aug_mix` adds `--mixup_alpha 0.2 --cutmix_alpha
1.0 --mix_prob 0.5`. **By rule**, augmentation is not searched on a frozen backbone: the axis collapses to `off`
(`Candidate.replace`), each folded combination is listed (`Grid.collapsed`, printed, `selection.json`), and the
leaderboard's augmentation column says `not searched (frozen-backbone rule)`. This is the user's rule, kept for cost and
to match the cached-feature probe, which never augments; it is not a property of deeptune.py, which builds a `train=True`
dataset for a frozen run too (trainers/video/train.py), so `--video_augment` would change what a frozen backbone's head
sees. (The first R3b text said otherwise; the critic corrected it.) The simulated surfaces assume augmentation adds
nothing to a frozen run, an assumption no score depends on, since the search never runs that combination. A candidate with augmentation off keeps R3a's config and ledger key. Flips need `--allow_flip`, and deeptune.py's
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
- **Flags of another strategy are refused** (`--n_trials` or `--bayes_startup` with smart, smart's knobs with random or
  bayes, and so on), as `--include` is; so their argparse defaults are now `None`, resolved by each strategy.
- **The regret study's budget** for random and bayes is smart's trainings including probe screens, which favours them a
  little (a probe costs less than a cross-validation).

## Regret study (CPU, simulated)

`python scripts/study_search_regret.py --seeds 0 1 2 3 4 5 6 7` (CPU, about 15 minutes; rerun after the fix round).
Grid: 3 backbones (resnet_tiny, siglip_tiny, vjepa2_tiny; 3 families) x train modes frozen, lora:8, full x learning rates
1e-3, 1e-4 x augmentation off, aug = 36 combinations, 30 candidates after the frozen collapse. k = 3, 45 clips in 15
groups; the study seed sets the split, folds, surface jitter and sampling. Regret = exhaustive's best CV accuracy minus
exhaustive's CV accuracy of the pick; true-quality regret is the same on the surface's noiseless quality. random and bayes
got smart's training count on each seed (screens included) as `--n_trials`. Surfaces (`simulate.SURFACES`):

- **interaction** (written by the builder): backbone 0.55 / 0.50 / 0.45; (train mode, rate) frozen 0 / -0.05, lora
  -0.03 / +0.08, full -0.20 / +0.20 (high / low rate; the high rate is the base); +0.04 for augmentation where weights
  train; +-0.02 jitter; screens and probes add +-0.04 noise. **Its interaction sits exactly where smart looks**: in R3b's
  first version, stage 3's runner-up pair; now, inside the joint train-mode-and-rate group.
- **A_lora_high_rate** (the critic's): as above, but LoRA is slightly better at the high rate (-0.03 against -0.04), so
  the runner-up pair (LoRA, low rate) no longer reveals full fine-tuning at the low rate.
- **B_screen_misleads** (the critic's): frozen features rank the backbones 0.55 / 0.50 / 0.45, LoRA and full the other
  way round (0.45 / 0.50 / 0.55, full 0.45 / 0.50 / 0.60).

Both critic surfaces are used exactly as V-057 wrote them; none was tuned to the fix.

| surface | strategy | mean regret | max regret | mean true-quality regret | picked the true best | CV runs | screen runs |
|---|---|---|---|---|---|---|---|
| interaction | exhaustive | 0 | 0 | 0 | 8/8 | 30 | 0 |
| interaction | smart | 0 | 0 | 0 | 8/8 | 14.5 | 6 |
| interaction | smart, frozen view only | 0 | 0 | 0 | 8/8 | 14.5 | 3 |
| interaction | smart without stage 3 | 0 | 0 | 0 | 8/8 | 14 | 6 |
| interaction | random (20.5 trials) | 0.024 | 0.139 | 0.027 | 6/8 | 20.5 | 0 |
| interaction | bayes (20.5 trials) | 0 | 0 | 0 | 8/8 | 20.5 | 0 |
| A | exhaustive | 0 | 0 | 0 | 8/8 | 30 | 0 |
| A | smart | 0 | 0 | 0 | 8/8 | 14 | 6 |
| A | smart, frozen view only | 0 | 0 | 0 | 8/8 | 14 | 3 |
| A | smart without stage 3 | 0 | 0 | 0 | 8/8 | 14 | 6 |
| A | random (20 trials) | 0.024 | 0.139 | 0.023 | 6/8 | 20 | 0 |
| A | bayes (20 trials) | 0.021 | 0.139 | 0.020 | 6/8 | 20 | 0 |
| B | exhaustive | 0 | 0 | 0 | 8/8 | 30 | 0 |
| B | smart | 0 | 0 | 0 | 8/8 | 21.5 | 6 |
| B | smart, frozen view only | 0.104 | 0.139 | 0.107 | 0/8 | 14.25 | 3 |
| B | smart without stage 3 | 0 | 0 | 0 | 8/8 | 21 | 6 |
| B | random (27.5 trials) | 0.010 | 0.083 | 0.011 | 7/8 | 27.5 | 0 |
| B | bayes (27.5 trials) | 0 | 0 | 0 | 8/8 | 27.5 | 0 |

Before the fix round (the critic's run, 4 seeds, same grid): on A smart's regret was 0.271 (4/4 seeds wrong) against
random 0.035 and bayes 0.049; on B smart's was 0.090 against bayes 0. On the interaction surface (8 seeds) smart was 0,
smart without stage 3 0.267, random 0.024, bayes 0.

**Reading it, with its limits.**
- A is now handled by stage 2: train mode and rate are one group, so every (mode, rate) pair is tried on each tuned
  backbone. That is also why smart without stage 3 now matches smart on all three surfaces: none of them has an
  interaction outside the joint group. Stage 3 is exercised by the dedicated test (what trains and rate x frames), not by
  these surfaces, and it catches only the combinations it tries.
- B is handled by the trainable screening view: smart with the frozen view only (R3b as first shipped) picks the wrong
  backbone on every seed (regret 0.104). The price is in the runs: on B the views disagree, three backbones survive instead
  of two, and smart used 21.5 cross-validations plus 6 screens, against exhaustive's 30. On the other surfaces it used
  14 to 14.5 plus 6 screens.
- **This grid is too small to separate the strategies.** With 30 candidates, smart's own budget is two thirds to nine
  tenths of the grid, so random and bayes at the same budget nearly cover it; bayes matched smart on two surfaces and lost
  on A (6/8). Smart's zero regret on surfaces written with knowledge of where it looks shows that its mechanisms work; it is
  not evidence that smart beats bayes in general. A three-group interaction, or a screen misled in both views, would beat
  it. The real-data gate below is the comparison that counts.
- The unit test `test_regret_study_on_each_surface` runs seeds 0 and 1 on each surface and asserts smart's zero regret
  with fewer CV runs than exhaustive, and over 5 points of regret for the frozen-view-only smart on B.

## Review fix round (ml-critic V-057: "merge after fixes")

| # | Item | Fix | Before -> after |
|---|---|---|---|
| 1 | CI fails without Optuna | Optuna-dependent tests skip with a reason (`BAYES` param mark, `importorskip`); the regret test drops bayes without it; CI installs `optuna>=4,<6` (`.github/workflows/test.yml`) | `-p nooptuna`: `5 failed, 24 passed` -> `40 passed, 4 skipped` |
| 2 | Stage 3 misses train mode x rate | Train mode and rate are one group; stage 3 tries every combination of two groups when their counts multiply to at most `--recheck_full_product` (6), else the runner-up pair; counted in the plan. Trainable screening view for B | A: 0.271 -> 0; B: 0.090 -> 0 (table above) |
| 3 | Regret claims | Regret section, README and smart.py rewritten: stage 3 catches exactly what it tries; the builder's surface put the interaction where smart looks; all three surfaces with true-quality regret | see above |
| 4 | "Augmentation cannot affect a frozen backbone" | Reworded everywhere as a rule (cost; the probe never augments); the leaderboard says `not searched (frozen-backbone rule)`; the simulation's assumption labelled | wording |
| 5 | `--tune_epochs` final kept one per backbone | The top `--final_top` of all tune runs | critic's repro: 1 final run, runner-up None -> 3 final runs, runner-up set |
| 6 | Tests that did not test their claim | A probe on other folds stops the search; random/bayes compare drawn candidates, not keys; `--screen cv` records its decision and the test checks it | mutants below |
| 7 | Flags of another strategy ignored | Refused with the strategy named (`refuse_foreign_flags`); explicit `--screen probe` that cannot run is refused before training | 9 parametrized cases |

Also found by the new tests: the second screening round labelled backbones by family, so two backbones of one family
collided and one was silently dropped from the screen (fixed: labels are per backbone).

Nice-to-haves done: the "no handle on the sealed file" wording is now a convention enforced by the sealing test; the final
decision says "these are the ranking runs themselves" and that the engine ranks every ranking-protocol entry of `--out`;
the probe's ledger key includes its epochs and learning rate (frames are the candidate's own axis); resuming an R3a
folder without `--search` names the default change and suggests `--search exhaustive`; `--include` survivors are in the
plan's upper bound. P-122 (one-standard-error ties) left for later, as asked.

**Mutants** (each with `-k` on its matching test in `tests/test_model_selection_strategies.py`; the critic's `mutplug.py`
and the builder's `fixmut.py`, both outside the repo):

| Mutant | What it breaks | Result |
|---|---|---|
| critic `fold_check_ranked_only` | the fold check sees cv entries only | killed: `1 failed` |
| critic `random_fixed_seed` | one permutation whatever the seed | killed: `1 failed, 1 passed` (the bayes case is unaffected) |
| `screen_cv_silent` | `--screen cv` records no decision | killed: `1 failed` |
| `recheck_runner_up_only` | stage 3 never tries the full product | killed: `2 failed` |
| `separate_mode_and_rate` | train mode and rate as separate groups again | killed by the stage-2 decision test: `1 failed, 3 passed`. The three regret tests pass under it: stage 3's full product (3 x 2 = 6) recovers A on its own, so the joint group is defence in depth there |
| `frozen_view_only` | stage 1 screens frozen features only | killed: `1 failed` (regret test, B) |
| `final_one_per_backbone` | the first R3b's `--tune_epochs` final | killed: `1 failed` |
| `foreign_flags_ignored` | no refusal of other strategies' flags | killed: `9 failed` |
| `explicit_probe_falls_back` | `--screen probe` silently falls back | killed: `1 failed` |

## Tests

`tests/test_model_selection_strategies.py` (44 test cases after the fix round, CPU, simulated runner unless stated): the
augmentation axis expands, collapses on frozen backbones with the rule's reason, and keeps R3a's keys; flips need
`--allow_flip`; smart is the default and searches augmentation by default; augmentation in `--deeptune_args` conflicts with
the axis and is respected by the default; frozen candidates never get augmentation flags and the leaderboard says so;
**augmentation reaches training rows only** (the real `train()` builds training rows with `train=True` and validation with
`train=False`, the real `evaluate()` builds evaluation rows without `train`, a `train=False` dataset neither augments nor
jitters, and mixup / CutMix run on training batches only, through the real Trainer); **every strategy reads and passes
development rows only**; smart records and prints a reason for every stage and tunes train mode with the rate; stage 3
finds an interaction between groups only among the combinations it tries (both sides); `--screen cv` in both views and its
recorded decision; backbone screening within families in both views; `--include` / `--exclude` / `--fix`; `--max_hours` as
a running budget; random and bayes are reproducible, draw other candidates with another seed, and resume the same sequence
after Ctrl+C; random draws without replacement; bayes proposes runnable untried candidates and Optuna sees development CV
means only; bayes without Optuna exits with the install hint while the other strategies work; the profile reads
development rows only; `--tune_epochs` (the critic's repro: 3 final runs, the 3 best of all tune runs); smart resumes with
the same decisions; the probe screens only extractor-built frozen backbones and a view never mixes protocols; a screen on
other folds stops the search before the sealed test; other strategies' flags are refused; an explicit `--screen probe`
that cannot run is refused; an R3a folder resumed without `--search` gets the hint; the regret study on each surface.

`tests/test_model_selection_e2e.py`: `test_smart_search_end_to_end_with_a_real_feature_probe_screen` (`slow`, now with
`--screen_views frozen` to keep the real probe path at its first cost; the two views are covered by the simulated tests):
real extract_features.py, train_feature_head.py and deeptune.py subprocesses on siglip_tiny and vjepa2_tiny.

Results (CPU, `CUDA_VISIBLE_DEVICES=-1`, OMP/MKL 2 threads, basetemp in the scratchpad):

- `tests/test_model_selection_strategies.py tests/test_model_selection.py`: `97 passed, 2 warnings in 339.40s`
- the same strategies file with Optuna hidden (`-p nooptuna`): `40 passed, 4 skipped, 1 warning in 178.38s` (before the
  fix: `5 failed, 24 passed`)
- `tests/test_video_followups.py tests/test_small_dataset_splits.py`: `26 passed in 51.84s`
- `tests/test_model_selection_e2e.py` (run once, at the end): `2 passed in 324.29s`

First round, for the record: `82 passed` (strategies + R3a), `2 passed` (e2e), `26 passed` (follow-ups + splits).

Edits to existing tests: R3a's `_argv` helper in `tests/test_model_selection.py` and the R3a e2e command line start with
`--search exhaustive`, because the default strategy changed. No assertion of an R3a test was changed or removed. In the
fix round, R3b's own tests changed where the design changed (stage names, two views, the joint group, the leaderboard's
wording); the stage-3 test that relied on runner-up pairs was replaced by one that checks both what stage 3 finds and what
it misses.

## Not done

- Any GUI (G1, next). `select_model.py` keeps one parser with help on every flag, and writes `selection.json`,
  `stages.json`, `profile.json`, `leaderboard.csv` and `candidates.csv` for it.
- Real-data runs of smart, random and bayes (the gate below). The regret numbers above are from simulated surfaces.
- A larger simulated grid, on which the strategies' budgets would differ enough to separate them.
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

Check also: the leaderboard's probe/screen rows have no rank; `not searched (frozen-backbone rule)` on frozen rows; `stages.json` has both screening views; each `final/<key>/.../split_summary.json` names `sealed_test.parquet`; and in step 2 the printed upper bound against the trainings actually run.

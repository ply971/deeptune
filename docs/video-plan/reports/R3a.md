# R3a: automatic model selection, the engine and its exhaustive mode

Branch `video/R3-model-selection`, on top of `video-analysis` eefe3a9. Everything here ran on the CPU, with a fake runner
or two untrained tiny models on 32x32 synthetic clips. **No number in this report describes a real model on real data.**
The real-data GPU gate is at the end, for the coordinator to run with the user. R3b (the smart, staged search) is not built.

## What it does

`select_model.py --search exhaustive` (handlers/model_selection/):

1. Sets a sealed test split aside, once, before anything else (`sealed.py`).
2. Expands the grid (`grid.py`), checks every candidate through deeptune.py's own parser and the trainer's own model
   builder (`checks.py`), and lists invalid ones as refused and over-budget or over-memory ones as skipped, with reasons.
3. Prints the candidate, refused and skipped counts, the trainings (runnable candidates x k + 1 final refit) and a time
   estimate with its basis (`estimate.py`). `--dry_run` stops before any training.
4. Trains each candidate as a deeptune.py subprocess with grouped k-fold on the development rows (`runner.py`), one ledger
   line per finished candidate (`ledger.py`).
5. Ranks (`ranking.py`), refits the winner on the development rows, and scores it once on the sealed test set.
6. Writes `leaderboard.csv`, `selection.json` and a short summary.

## Decisions

**The sealed test set, and how deeptune.py gets development data only.** deeptune.py always carved its own 70/10/20
split from `--df`, and its k-fold ran on the 80% train + validation part, followed by a final model scored on its own 20%
test. Three ways were possible:

- Pass the development file to plain `deeptune.py --validation kfold`. Rejected: each candidate would train k + 1 models,
  its folds would see only 80% of the development rows, and its run folder would hold a `test_split.parquet` and a test
  score that are easy to mistake for the sealed test.
- Drive the folds from select_model.py with one subprocess per fold. Rejected: deeptune.py would still carve a test split
  from each fold's rows, and the fold logic of `evaluators/video/validation.py` would be repeated.
- **Chosen:** two small video-only flags in deeptune.py. `--cv_only` (with `--validation kfold|bootstrap632`) runs the
  existing B3 cross-validation on every row of `--df` and stops: no test split is carved, so none exists to be confused
  (`prepare_whole_dataset` writes `data_splits_*/cv_pool.parquet` with the same label encoding and checks). `--test_df`
  makes a given Parquet file the test split; `--df` then gives training and validation (`split_with_test_file`, using the
  same `inner_validation_split` and 10% share as each fold, so the refit trains the way its folds did). It refuses a
  shared group, a shared video path, or a test class that `--df` lacks.

select_model.py splits once with the first fold of `kfold_splits` (stratified, grouped, `--seed`), writes
`data/dev.parquet` and `data/sealed_test.parquet` with absolute video paths, and records the split in `data/split.json` and
`data/split_assignment.csv`. Every candidate and the pilot get `--df data/dev.parquet --cv_only`; only the final run gets
`--test_df data/sealed_test.parquet`. Ranking reads only the ledger and the out-of-fold files of development rows. A test
checks this on the argument lists, on every Parquet read during a search (pandas and pyarrow are wrapped: the sealed file
is read once, by the final run), and on the rows each run received.

**Checks without repeated rules.** `deeptune.main`'s argument checks moved, unchanged, into `deeptune.parse_args(argv)`,
which `main` calls first. `checks.check_argv` runs it on each candidate's command line and keeps the parser's message.
Some combinations only the model builder refuses (tsm on a 3D CNN, LoRA on resnet_tiny); the preflight builds each model
with `trainers.video.train.build_training_model` (the trainer's own code, extracted from `train` without change), so those
are refused with the builder's message.

**Preflight and memory.** Each distinct model (learning rate aside) is built once on the CPU, then moved to the meta device
for one forward pass of one clip: shapes only, so a large model on a long clip cannot fill the laptop's RAM. It records
total and trainable parameters, the bytes saved for backward and the forward FLOPs. On resnet_tiny and r3d_18 the meta
numbers matched a CPU pass to within 0.2%. Memory estimate, fp32: 1.2 x (4 B x parameters + 12 B x trainable parameters)
+ 1.2 x activations x batch size + 0.5 GB CUDA context. It is not calibrated against real GPU peaks yet (gate item 3).
Building loads the weights, so the first preflight of a backbone downloads them if they are not cached.

**Time estimate.** One pilot: the cheapest candidate by preflight cost, through the same deeptune.py path, 1 epoch on 2
folds. Seconds per fold-epoch = wall time / 2, overhead included, scaled per candidate by cost ratio x rows per fold x k x
epochs; the final refit is one fold-training of the dearest candidate scaled to all rows. The printed line says it is
rough and gives its basis. `--dry_run` trains nothing, not even the pilot (the spec's "runs nothing"), so its time is
"unknown"; `--estimate_only` runs the pilot and stops. Over `--max_hours` (default 8), or over 50 trainings without an
estimate, the search refuses to start unless `--yes`. `--max_hours` is a start gate, not a deadline.

**Ledger and resume.** The key hashes the candidate config, the development file's fingerprint and the training settings.
A line is written in one append and fsynced after its candidate ends, so Ctrl+C loses only the running candidate, whose
half-written folder is removed before it reruns; a torn line is skipped with a warning. On resume: a changed input file
(fingerprint), changed training settings, a changed `--test_fraction` or modified split files are refused; a changed grid
needs `--allow_grid_change`. Failed and did-not-fit candidates are not retried unless `--retry_failed`.

**Out of memory.** The subprocess's log is scanned for CUDA and CPU allocation errors (including Windows' page-file error);
those end as `did_not_fit` and the search continues. Each run is its own process, so its memory is freed.

**Ranking.** By the mean of the metric over folds (`--metric`, default `accuracy`; `mae` for regression, lower is better),
with the fold SD and B3's group-bootstrap interval of the pooled out-of-fold metric. The paired difference of winner and
runner-up uses B3's `compare_models` on the same out-of-fold rows (both ran the same folds), resampling groups together.
Ties keep grid order.

**Reporting.** `selection.json` holds the grid and its formula, the settings and seeds, the split record, counts, winner,
reason, runner-up, paired difference, the sealed-test score with its interval, the weights licence of every backbone that
finished, and notes: the winner's CV score is optimistically biased by the selection (winner's curse) and the sealed-test
score is not; neither promises accuracy elsewhere. The sealed test scores one winner per folder: a later different winner
is not scored unless `--rescore_sealed_test`, and then the record says the set was reused.

**Strategy plug-in.** `STRATEGIES` maps `--search` names to classes with `plan(runnable)` (for the count and estimate) and
`batches(runnable, ledger)`, a generator that can read the ledger between batches. R3b adds `smart` there and makes it the
default.

## Deviations from the request

- The final refit trains on all development rows except 10% (whole groups) held for checkpoint selection and early
  stopping, as each fold does. A refit with no validation rows would need a fixed epoch count chosen from the folds.
- `lora:R` on 2D backbones passes `--embed_size R`, because deeptune.py ties their LoRA rank to the head width there.
- `--dry_run` does not run the pilot; `--estimate_only` does (see above).
- No ledger line for runnable candidates that were never run; they appear in the leaderboard as "not run" only if a
  strategy chose not to run them (cannot happen with exhaustive).

## Not done

- R3b (smart search, measured against exhaustive) and any GUI page.
- Nested cross-validation: the selection bias of the winner's CV score is stated, not corrected. The sealed test is the
  unbiased number.
- `bootstrap632` candidates (only k-fold), parallel runs, and a run-time deadline.
- The memory estimate and the time estimate are not yet checked against real GPU runs (gate items 2 and 3).
- The full suite was not run (instructions: the laptop runs out of memory). Targeted files below.

## Tests

`tests/test_model_selection.py` (27 tests, about 20 s on CPU): a fake runner writes what `deeptune.py --cv_only` writes,
through the real `run_validation`, with predictions right on a share of rows that is a known function of the config.
It covers grid expansion; refusal reasons from the real parser and the real builder; budget and memory skips; dry run;
the estimate printed after the pilot and before the first candidate; `--max_hours` and the large-run refusal without
`--yes`; ranking against the fake's known best; the paired difference; sealed rows never passed or read; did-not-fit;
resume after a simulated Ctrl+C running only the missing candidates; a changed dataset, grid or settings on resume; the
sealed test not reused for a new winner; a torn ledger line; exit-code classification; the real preflight on resnet_tiny;
deeptune.py's refusals for the new flags; `split_with_test_file` and `prepare_whole_dataset`.

`tests/test_model_selection_e2e.py` (1 test, about 45 s, `slow`): real deeptune.py subprocesses, resnet_tiny and vjepa2_tiny,
4 frames, k = 2, 1 epoch, 20 clips: ledger, leaderboard, selection, that no cross-validated row is a sealed row, and that
the final run's test split is exactly the sealed file.

Results (CPU, `CUDA_VISIBLE_DEVICES=-1`):

- `tests/test_model_selection.py`: `27 passed, 1 warning in 19.84s`
- `tests/test_model_selection_e2e.py`: `1 passed in 43.79s`
- Existing files for the touched modules, batch 1 (splits, validation schemes, param budget, video trainer, training basics,
  evaluation / abstention / video / regression / flip-guard / training-basics end to end): `283 passed, 2 skipped, 49 warnings in 172.42s`
- Batch 2 (error model, frame cache, modern backbones, temporal, data layer, follow-ups, workflow, desktop analyse):
  `2 failed, 354 passed, 16 warnings in 83.39s`. The two failures, `test_video_followups.py::test_infer_cli_falls_back_to_the_runs_recorded_sampling`
  and `::test_infer_cli_without_a_recording_uses_defaults`, are not from this branch: R2 (debc35a) made `infer.py` pass
  `abstain=` to the predictor, and these tests' fake predictor does not take it (`TypeError: ... unexpected keyword
  argument 'abstain'`). This branch does not touch `infer.py`, `inference/` or that test. Left as is, per CLAUDE.md rule 3.

## Real-data GPU gate (coordinator, with the user)

From the repo root, GPU visible, on the UCF101 10-class demo manifest (330 clips, `group` column):

```powershell
$data = "C:\Users\Ahmed Magdy\deeptune-demo\train_clips.csv"
$grid = @('--backbones','r3d_18','r2plus1d_18','resnet18','--train_modes','frozen','lora:8','--num_frames','16','--k','5','--epochs','5','--batch_size','4')
$extra = @('--deeptune_args','--precision','bf16')  # always last on the line
# 1. Counts and refusals only; trains nothing
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a @grid --dry_run @extra
# 2. The pilot and the estimate; write down the estimate
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a @grid --estimate_only @extra
# 3. The search (add --yes if the estimate is over 8 h). Press Ctrl+C once mid-run, then rerun the same line to check resume
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a @grid @extra
# 4. Out of memory on purpose: one large model at a large batch, memory check off, to see did_not_fit and the search going on
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a_oom --backbones swin3d_b r3d_18 --train_modes full --num_frames 16 --k 2 --epochs 1 --batch_size 16 --gpu_memory_gb none --no_pilot --yes
```

Record in PROGRESS.md: the printed estimate against the real wall time; each candidate's `estimated_gpu_gb` in
`candidates.csv` against its peak GPU memory (`peak_gpu_reserved_mb` in each run's `training_run.json`, from T1); the leaderboard, the paired difference and the
sealed-test score with its interval; and that `final/<key>/.../split_summary.json` names `sealed_test.parquet`.

## Review fix round (ml-critic: "merge after fixes")

`video-analysis` was merged in first (668f480), which brings 92dc079, the fix for the two `test_video_followups.py`
failures above. With the new tests on the unfixed code: 22 failed, 30 passed. The tests of items 1-3 and 5-11 failed before their fixes; those of items 4 and 12 pin behaviour that was already right and were written to kill the mutants.

| # | Item | Fix | Test(s) in `tests/test_model_selection.py` |
|---|---|---|---|
| 1 | `--deeptune_args` could undo the sealing (`--df full --seed 7`) | `grid.extra_args_problems`: every dashed word is matched to a deeptune.py flag as argparse matches it (exact, else a unique prefix), and the flags select_model sets (data, split, seed, folds, grid axes, epochs, batch size, target, grouper, boot, licences, `--config`, `--raw-data`) are refused with the reason. None of the grid-axis flags was found safe to pass. | `test_extra_flags_that_select_model_controls_are_refused` (12 cases, incl. `--see`, `--d`, `--df=`), `test_the_critics_repro_cannot_train_a_candidate_on_the_full_input` |
| 2 | Windows paths lost backslashes (`shlex.split`) | `--deeptune_args` is `nargs=REMAINDER`, last on the line, word by word; a word holding a space is refused with advice | `test_extra_flags_must_be_separate_words_and_keep_windows_paths`, `test_extra_deeptune_flags_reach_every_run` |
| 3 | "Same folds" never checked | Each ledger line has a `fold_signature` (hash of row to fold); finished candidates on different folds are refused before ranking or the sealed test; the paired difference also compares the `fold` column | `test_candidates_cross_validated_on_different_folds_are_not_ranked` |
| 4 | Untested ranking claims | none needed | `test_ranking_uses_the_mean_over_folds_not_the_best_fold` (M5), `test_the_paired_point_is_the_winner_minus_the_runner_up_...` (M9), `test_a_tie_keeps_grid_order` (M11, both orders) |
| 5 | Code version | `code` (commit, dirty) on every ledger line and in `selection.json`, from T1's `run_environment`; resume warns about lines from other commits, and the summary repeats it | `test_the_code_version_is_recorded_and_a_change_is_flagged_on_resume` |
| 6 | `--test_df` overlap checks | groups compared as text; relative paths resolved from their Parquet file's folder; bytes with no group column warn that overlap cannot be checked, recorded as `overlap_check` | `test_test_file_overlap_compares_groups_as_text_and_paths_from_their_file`, `test_test_file_of_clip_bytes_without_groups_...` |
| 7 | "20% of the groups" said for rows | the split message names the real unit; loud warning when clips are treated as independent, plus a UCF101 `_gNN_cNN` hint; in `split.json` and `selection.json` | `test_the_split_message_names_its_unit_and_warns_...` |
| 8 | Refit inner split ignored `--seed` | `split_with_test_file(seed=)`, passed from `--seed` only on the `--test_df` path, so plain deeptune.py runs keep 42; `validation_seed` in `split_summary.json` and `selection.json` `seeds.refit_validation` | e2e now runs `--seed 7` and checks both; the bytes test checks `validation_seed` |
| 9 | "Interval excludes 0" without caveat | the reason says the top two were picked on the same rows, so the interval flatters the gap and is not a test | asserted in the M9 test |
| 10 | Imbalanced classes with accuracy | warning (largest/smallest > 1.5 in the development rows) suggesting `--metric balanced_accuracy`, kept in `selection.json` | `test_imbalanced_classes_get_a_warning_about_accuracy` |
| 11 | Preflight crash labelled "refused" | `ValueError` from the builder stays refused; any other exception is `failed` ("preflight: ..."), marked `stage: checks`, and is tried again on the next run | `test_a_preflight_that_crashes_is_failed_not_refused_and_is_tried_again` |
| 12 | M12 survived | none needed | the M9 test also asserts the refit's `--df` is `dev.parquet` and its `--test_df` the sealed file |

**Edits to existing R3a tests (mine, unmerged).** `test_extra_deeptune_flags_reach_every_run` now passes the flags as
separate words (`--deeptune_args --precision bf16 --video_augment`) instead of one `=` string, because item 2 changed the
interface; its assertions are unchanged. The other changed lines extend `FakeRunner` (per-backbone quality, per-fold
quality, a fold-seed offset) with defaults that keep its old behaviour. No assertion was removed or weakened.

**Mutants** (critic's `mutplug.py`, run with `-x` on `tests/test_model_selection.py`):

| Mutant | Result |
|---|---|
| M1 sealed to cv, M1b full input to cv, M4 drop refused, M6 sealed ignores groups, M7 test_df without group check, M8 pilot on sealed, M10 no refusal on settings change | killed (as before) |
| M2 fold seed per candidate | **killed** (fold check) |
| M5 rank by max fold | **killed** |
| M9 runner-up is the winner | **killed** |
| M11 ties reversed | **killed** |
| M12 final refit on full input | **killed** |
| M3 key without data fingerprint, M3b key without training settings | **killed** by a new direct test of `candidate_key`; through the search they are caught earlier by `search.json`'s refusals |

## Proposals (not built)

- Tie-break by trainable parameters (fewer first) instead of grid order, so a tie goes to the cheaper model.
- A one-standard-error rule: among candidates within one fold SE of the best mean, pick the smallest. It counters the
  winner's curse somewhat, but fold SEs are too narrow (folds share data), so it should be reported, not silently applied.
- Nested cross-validation as an opt-in, to estimate the whole selection procedure's error instead of the winner's.

Results after the fixes (CPU, `CUDA_VISIBLE_DEVICES=-1`):

- `tests/test_model_selection.py`: `53 passed, 1 warning in 41.70s`
- `tests/test_model_selection_e2e.py`: `1 passed in 43.08s`
- `tests/test_small_dataset_splits.py tests/test_video_followups.py tests/test_evaluation_pipeline_e2e.py tests/test_abstention_pipeline.py tests/test_validation_schemes.py tests/test_video_pipeline_e2e.py`: `96 passed, 15 warnings in 233.34s`

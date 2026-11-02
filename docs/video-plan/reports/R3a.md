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
$grid = @('--backbones','r3d_18','r2plus1d_18','resnet18','--train_modes','frozen','lora:8','--num_frames','16','--k','5','--epochs','5','--batch_size','4','--deeptune_args=--precision bf16')
# 1. Counts and refusals only; trains nothing
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a @grid --dry_run
# 2. The pilot and the estimate; write down the estimate
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a @grid --estimate_only
# 3. The search (add --yes if the estimate is over 8 h). Press Ctrl+C once mid-run, then rerun the same line to check resume
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a @grid
# 4. Out of memory on purpose: one large model at a large batch, memory check off, to see did_not_fit and the search going on
.venv\Scripts\python.exe select_model.py --df $data --raw-data --out .artifacts\gate\r3a_oom --backbones swin3d_b r3d_18 --train_modes full --num_frames 16 --k 2 --epochs 1 --batch_size 16 --gpu_memory_gb none --no_pilot --yes
```

Record in PROGRESS.md: the printed estimate against the real wall time; each candidate's `estimated_gpu_gb` in
`candidates.csv` against its peak GPU memory (`peak_gpu_reserved_mb` in each run's `training_run.json`, from T1); the leaderboard, the paired difference and the
sealed-test score with its interval; and that `final/<key>/.../split_summary.json` names `sealed_test.parquet`.

# A4: Per-prediction error estimate ("adaptive assurance")

Branch `video/A4-error-estimate`, cut from `video-analysis` @ 7782b0b.
Scope: Phase 5 task 4 only, plus the reporting it needs. Tasks 1–3 and 5–7 are
left to the Phase 5 agent. The API is built so that agent can feed in
out-of-fold predictions and extra features.

## Plan (as executed)
1. A new torch-free module, `evaluators/error_model.py` (numpy + pandas only):
   features, `ErrorModel`, calibration table, JSON I/O, run lookup.
2. Fit on the validation split at the end of `Trainer.train()` (one marked
   block plus a helper method). Add an opt-out `--no_error_model` flag.
3. Use the model in TestTrainer (eval), `predict_video`, `predict_images` and
   `infer.py`. Write the calibration report on eval, and on inference when the
   new data has labels.
4. Tests (CPU, fast), a README section, and this report.

## Done
- **`evaluators/error_model.py`**
  - `prediction_features(scores, mode, extra=None, from_logits=False)`.
    Classification gives `max_prob`, `margin` (top1 − top2) and `entropy`
    (normalised by log K). Regression gives `predicted_value`. `extra` (a
    DataFrame, a dict or an array) appends columns, for TTA disagreement, kNN
    distance or evidential uncertainty later.
  - `error_targets(scores, labels, mode)` returns is_wrong (cls) or the
    absolute error (reg).
  - `ErrorModel(mode, method='auto'|'isotonic'|'logistic'|'constant', score_feature=None, features=None, increasing='auto')`
    with `.fit(features, target, num_outputs=None)`, `.predict(features)`,
    `.to_json()/.from_json()` and `.save()/.load()`. The stored format is plain
    JSON with a `format`/`version` header and no pickle.
    - **isotonic** (the default) uses a weighted PAV written in numpy, checked
      against `sklearn.isotonic.isotonic_regression`. It runs on one score
      (`max_prob` for cls, with direction from a known-feature map). Smoothing:
      each block is shrunk toward the base rate with `PRIOR_STRENGTH` (2 at
      first, tuned to 0.5 in the follow-ups)
      pseudo-samples, the blocks are re-pooled to stay monotone, and
      predictions interpolate linearly between block centres (constant past
      the ends).
    - **logistic** is L2-regularised Newton/IRLS on standardised features.
      It is cls only and opt-in, meant for when extra features exist.
    - **constant / fallback** uses the Laplace-smoothed base rate
      `(errors+1)/(n+2)` for cls and the MAE for reg. The fallback applies
      when `n < MIN_FIT_SAMPLES = 30`; for cls when there are fewer than
      `MIN_OUTCOME_COUNT = 5` wrong or 5 right predictions; for regression
      with no `score_feature`; and when the score or target is constant.
      `fit_info['fallback_reason']` records why.
    - Rows with a NaN required feature get the fallback value.
      `num_outputs` is recorded, so a model is not applied to a network with
      a different class count.
  - `calibration_table(expected_error, observed, n_bins=10, mode=None)` uses
    equal-count bins (deciles). Each bin has n, stated min/max/mean, observed
    mean, gap, and a 95% CI (Wilson for cls, normal approximation for reg)
    with a `within_95ci` flag. The summary holds n, mean stated/observed,
    `calibration_error` (count-weighted mean |gap|, an equal-mass ECE),
    `max_abs_gap` and `bins_within_95ci`. For cls it adds `brier` and
    `auroc_error_detection`.
  - `save_report(report, out_dir, stem='error_calibration', extra=None)`
    writes the `.json` and `.csv`. `metrics_summary()` produces the flat dict
    for metrics files.
  - Other helpers: `find_error_model(ckpt)`, `load_error_model_near(...)`
    (warns and returns None on a corrupt file or a mode/class-count
    mismatch), and `observed_errors(predictions_df, mode)`.
  - Process-wide switch: `set_fit_by_default()` / `fit_by_default()` (removed in the follow-ups).
- **Trainer** (`trainers/vision/trainer.py`): one marked block at the end of
  `train()` plus the helper `_fit_error_model()`. Nothing changed in
  `__init__`.
  - The helper makes one extra pass over `val_loader` under `no_grad`/`eval()`,
    using the final weights. These are the weights `train.py` saves as
    `model_weights.pth`; `Trainer` keeps no separate "best" checkpoint.
  - Python, numpy, torch and CUDA RNG states are restored afterwards, and so
    is train/eval mode. A test shows that weights and RNG state are identical
    with and without the fit.
  - It writes `error_model.json` and `error_model_fit.json` (n_samples,
    n_errors, base_error, method, fallback_reason, split=`validation`,
    n_batches, val_dataset_size) into `output_dir`, which is the folder that
    holds `model_weights.pth`.
  - Any exception is logged as a warning and never breaks a finished
    training run.
  - It is enabled by `trainer.fit_error_model` (per instance) when that is
    set, and otherwise by `fit_by_default()`.
- **CLI** (`cli.py`): `--no_error_model` (store_true, with help text) sits in
  `_add_default_args`, so every run type has it, ONECALL and TRAIN included.
  It is stored as `self.no_error_model`, so `save_args` records it in
  `cli_arguments.json`. It calls `set_fit_by_default(not no_error_model)`.
- **Eval** (`evaluators/vision/evaluator.py`, shared by images and video):
  `test()` now also writes **`predictions.csv`** (one row per test sample).
  - cls columns: sample_index, true_label, predicted_class, confidence,
    correct, and prob_k when K ≤ 50.
  - reg columns: sample_index, true_value, predicted_value, abs_error.
  - When there is an `error_model.json` next to the checkpoint (or an explicit
    `error_model_path=`), it adds `expected_error`, writes
    `error_calibration.json/.csv` (split=`test`) and adds `error_estimate`
    (the flat summary) to `full_metrics.json` and to the returned dict.
  - `evaluators/video/evaluate.py` needed no change.
- **Inference**: `predict_video` and `predict_images` now keep the full
  probability rows (cls) or raw outputs (reg) and call
  `inference.common.attach_expected_error(...)`. `infer.py` writes
  `error_calibration.json/.csv` and `inference_metrics.json['error_estimate']`
  when the new data has labels and an `expected_error` column.
- **README**: new section "Per-prediction error estimate (video & images)",
  covering what `expected_error` means and how to read the calibration table.

## Files
New: `evaluators/error_model.py`, `tests/test_error_model.py`,
`tests/test_error_model_pipeline.py`, this report.
Edited: `trainers/vision/trainer.py` (end of `train()` + helper),
`evaluators/vision/evaluator.py`, `inference/common.py`, `inference/video.py`,
`inference/vision.py`, `infer.py`, `cli.py` (one grouped flag),
`README.md` (one section).

## Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`; `options` → `cpu 0`)
- `tests/test_error_model.py`: `18 passed in 2.50s`. Covers features, targets,
  per-decile calibration on held-out synthetic data, the logistic model,
  isotonic/logistic monotonicity, PAV vs scikit-learn, the tiny-set fallback,
  the no-error fallback, regression constant vs adaptive, JSON round-trip
  (identical predictions for every method), missing features, calibration
  edge cases and report files, run lookup with mismatch warnings, and a
  subprocess check that importing `evaluators.error_model` and
  `inference.common` does not import torch.
- `tests/test_error_model_pipeline.py`: `9 passed`.
  - The fit uses only validation data. Bookkeeping: `n_samples ==
    len(val) != len(train)`; train reads = epochs × len(train); val reads =
    (epochs + 1) × len(val), so exactly one extra pass.
  - Training results are unchanged.
  - Regression gives a constant model.
  - `--no_error_model` is parsed, recorded in `to_dict` and honoured (no files
    written).
  - Eval of an old run is unchanged: no column, no `error_estimate`, no
    calibration file.
  - Eval with a model writes `expected_error` ∈ [0,1], the calibration
    json/csv and `full_metrics.json == returned`.
  - Regression eval writes per-sample `abs_error`.
  - End-to-end video with `dummy_video3d` on real synthetic mp4 clips:
    Trainer → `error_model.json` (fit on the 4 validation clips) → video
    `evaluate()` → `predictions.csv` with `expected_error` and the calibration
    files; eval does not touch `error_model.json`; `predict_video` matches the
    eval values; `infer.main` writes `expected_error`, `error_calibration.csv`
    and `inference_metrics.json.error_estimate`. An old-run checkpoint gives
    exactly the previous columns.
  - Image inference adds the column only when the model file exists.
- Targeted regression run (my tests plus the existing tests for the files I
  touched: test_evaluation, test_inference_data, training/test_video_trainer,
  training/test_vision_trainer, test_workflow_reliability,
  models/test_desktop_config, models/test_video):
  **`112 passed, 4 warnings in 21.76s`**.
  As instructed, I did not run the full suite.

## Calibration numbers (synthetic check, `test_stated_error_matches_observed_per_decile_on_held_out_data`)
> **Superseded.** This fixture was flawed (V-010) and the reading of it below
> was incomplete. See "A4 follow-ups" at the end for the corrected fixture,
> the re-derived table and the honest interpretation.

The synthetic model is 3-class and over-confident: true P(wrong) =
min(0.95, 1.6·(1 − max_prob)). The error model is fitted on 4000 samples and
scored on a separate held-out set of 4000.

| decile | n | stated | observed | gap | in 95% CI |
|---|---|---|---|---|---|
| 1 | 400 | 0.102 | 0.035 | −0.067 | no |
| 2 | 400 | 0.156 | 0.177 | +0.021 | yes |
| 3 | 400 | 0.251 | 0.253 | +0.002 | yes |
| 4 | 400 | 0.350 | 0.405 | +0.055 | no |
| 5 | 400 | 0.467 | 0.407 | −0.060 | no |
| 6 | 400 | 0.587 | 0.632 | +0.046 | yes |
| 7 | 400 | 0.708 | 0.698 | −0.011 | yes |
| 8 | 400 | 0.752 | 0.787 | +0.036 | yes |
| 9 | 400 | 0.844 | 0.873 | +0.028 | yes |
| 10 | 400 | 0.917 | 0.943 | +0.025 | yes |

| summary | isotonic error model | logistic error model | raw 1 − max_prob |
|---|---|---|---|
| calibration_error (ECE) | **0.035** | 0.030 | 0.208 |
| max abs gap | 0.067 | 0.053 | 0.363 |
| bins within 95% CI | 7/10 | 7/10 | 1/10 |
| Brier | 0.163 | 0.163 | 0.220 |
| AUROC (error detection) | 0.842 | 0.842 | 0.842 |

Test tolerances: every decile |gap| < 0.10, ECE < 0.05, raw ECE > 0.15,
AUROC > 0.75. The fixed seed makes the test deterministic. The isotonic model
is a little pessimistic in the most confident decile, because the last block
spans a wide confidence range and the shrinkage pulls it toward the base
rate. This is the known end-effect of isotonic calibration, and it errs on the
safe side.

## Decisions
- **Process-wide opt-out switch** (*removed in the follow-ups; see the end*). I could not edit `trainers/*/train.py`
  or `deeptune.py`, and `Trainer.__init__` is A1's. So `--no_error_model`
  reaches the Trainer through `evaluators.error_model.set_fit_by_default`,
  which `DeepTuneVisionOptions` calls after parsing. A per-instance
  `trainer.fit_error_model = True/False` overrides it.
- **The fit uses the final weights**, because those are what `train.py`
  saves. If a later phase adds best-epoch checkpointing, the helper should be
  called after the best weights are loaded back.
- **numpy PAV instead of sklearn.** It keeps the module light for the
  desktop and makes JSON serialisation trivial. A test checks it against
  scikit-learn.
- **Laplace-smoothed base rate** for the classification fallback, so a
  validation set with zero errors does not claim zero risk.
- **Regression has no intrinsic signal**, so it uses a constant (the MAE)
  until `extra` plus `score_feature` is supplied. `predicted_value` is a
  feature, but it is not used by default, because error is usually not
  monotone in it.
- **Eval `predictions.csv`** now always exists, since it needs a home for
  `expected_error`. prob_k columns are capped at K ≤ 50 to keep the file
  small for Kinetics-size heads.
- **Unreadable or mismatched `error_model.json`** produces a warning and no
  column. It does not crash the run.

## Open problems / limits
- With DeepTune's default 70/10/20 split, a small dataset often has fewer
  than 30 validation samples, so the model falls back to the constant base
  rate. That is honest but not adaptive. Out-of-fold predictions (Phase 5
  task 1) fix this.
- The e2e test uses only 4 validation clips, so it exercises the constant
  path end to end. The adaptive paths are covered by the unit tests.
- No TTA or kNN features yet. The `extra` hook is ready for them.

## Notes for the Phase 5 agent
- Fitting on out-of-fold predictions:
  `ErrorModel(mode).fit(prediction_features(oof_probs, mode, extra=...), error_targets(oof_probs, oof_labels, mode), num_outputs=K).save(run_dir / 'error_model.json')`.
  This overwrites the validation-fit model. Set `fit_info['split'] = 'oof'`
  before saving.
- Extra signals: pass `extra={'tta_disagreement': ..., 'knn_dist': ...}`.
  Use `method='logistic'` for several features, or `score_feature=` for a 1-D
  isotonic fit. Extras default to "larger means more uncertain".
  **Inference must then supply the same extra columns.** Today
  `attach_expected_error` only builds the built-in features, so extend it to
  pass `extra` as well.
- Temperature scaling (task 3) should be applied before the features are
  computed, both at fit time and at prediction time, or the error model will
  see a shifted `max_prob`.
- Task 7 (evidential uncertainty): `calibration_table(uncertainty, observed)`
  already reports per decile and needs no change.
- `predictions.csv` in the eval folder carries prob_k for ROC and PR curves
  (task 2).

## Notes for the GUI agent (Phase 7)
- **Test page:** show the `expected_error` column from `predictions.csv`, as
  a percentage "chance wrong" for cls and in target units for reg. Sorting
  or colouring by it is useful. The column is absent for old runs, so handle
  that case.
- **Results page:** read `eval*/error_calibration.json` or `.csv` and show
  the decile table (stated vs observed, with the CI) plus the summary line
  (calibration_error, bins_within_95ci, auroc_error_detection). A
  reliability-style plot of stated vs observed is the natural chart.
  `full_metrics.json['error_estimate']` is already flattened by
  `metric_rows`.
- **Reading files without torch:** `evaluators.error_model` is torch-free
  (a test enforces it), so the desktop can import it to read these files.
- **Train form:** add a checkbox "Fit per-prediction error model" (default
  on) that maps to `--no_error_model` when unchecked.

## Merge notes
- `trainers/vision/trainer.py`: my changes come after the epoch loop in
  `train()` and add a new method. They do not touch `__init__`, so they
  should not conflict with A1's `__init__` edits.
- `cli.py`: two small hunks, one after `self.fixed_seed = ...` in `__init__`
  and one after the `--pooling` argument. A1 also edits `cli.py`, so expect a
  possible textual conflict near those lines; resolve it by keeping both.
- Optional after the merge: wire the flag explicitly in
  `trainers/{video,vision}/train.py` with
  `trainer.fit_error_model = not getattr(args, 'no_error_model', False)`.
  The global then becomes a fallback. (Done in 05cf682; the global itself is
  removed in the follow-ups below.)
- No new dependencies.

## A4 follow-ups (branch `video/A4-fixes`, from `video-analysis` @ ac32792)
These answer the Critic's retroactive review: V-010, V-011 and V-012.

### V-010: the fixture, and what the original table really showed
- **The fixture was wrong.** `_overconfident` drew P(wrong) from column 0,
  which was not the arg-max in about 14% of samples (those with c < 0.5).
  The error therefore was not purely a function of `max_prob`.
- **Fix.** Column 0 is now always the arg-max: the other shares use
  a ∈ [1 − c/(1−c), c/(1−c)] ∩ [0, 1]. A new test,
  `test_synthetic_fixture_error_depends_only_on_max_prob`, asserts this.
  The fixture can also return the true P(wrong) (the *oracle*).
- **My original reading was incomplete, and it was wrong to imply the misses
  were benign.** I attributed only decile 1 to the isotonic end effect and
  said nothing about deciles 4–5, where decile 4 is optimistic.
- **The fixture bug did not cause the decile 4–5 misses either.** Those
  deciles hold samples with c > 0.5, where the old and new fixtures give
  identical draws. The re-derived numbers for those two deciles are
  unchanged.
- **What does explain deciles 4–5: test-set sampling noise.** A noise-free
  comparison of the stated error with the true P(wrong) of the same samples
  shows the stated error was right there. The table below uses fixture seed 0
  after the fix, with the new default prior (0.5, see below):

| decile | stated | true P(wrong) | observed | stated − true | observed − true |
|---|---|---|---|---|---|
| 1 | 0.069 | 0.051 | 0.035 | +0.018 | −0.016 |
| 2 | 0.154 | 0.159 | 0.177 | −0.005 | +0.019 |
| 3 | 0.246 | 0.263 | 0.253 | −0.018 | −0.011 |
| 4 | 0.348 | 0.366 | 0.405 | −0.018 | **+0.039** |
| 5 | 0.466 | 0.468 | 0.407 | −0.002 | **−0.061** |
| 6 | 0.568 | 0.570 | 0.613 | −0.002 | +0.042 |
| 7 | 0.672 | 0.673 | 0.645 | −0.001 | −0.028 |
| 8 | 0.772 | 0.782 | 0.787 | −0.010 | +0.005 |
| 9 | 0.890 | 0.901 | 0.925 | −0.010 | +0.024 |
| 10 | 0.947 | 0.950 | 0.963 | −0.003 | +0.013 |

- In deciles 4 and 5 the stated error is within 0.02 of the truth. The
  observed rate is what moved (+0.039 and −0.061 from the truth, about 1.6
  and 2.5 binomial SDs at n = 400).
- **Two real biases remain.** The prior pulls decile 1 toward the base rate
  (+0.018), and most deciles sit about 0.01–0.02 optimistic. The latter is
  fitting noise from 4000 samples spread over about 25 isotonic blocks.
- **Summary on the same draws:**

| | ECE vs observed | max abs gap | bins in 95% CI |
|---|---|---|---|
| error model (prior 0.5) | 0.032 | 0.058 | 6/10 |
| oracle (true P(wrong)) | 0.026 | 0.061 | 9/10 |
| raw 1 − max_prob | 0.193 | 0.362 | 1/10 |

  The logistic model gets 0.038, 0.060 and 4/10 on the same draws.
- **Over 200 seeds** (fit 4000 / test 4000, scratchpad `a4fix\coverage.py`):

| | bins within 95% CI | mean ECE |
|---|---|---|
| error model, prior 0.5 | 80.6% | 0.024 |
| error model, prior 2 (before tuning) | 70.3% | 0.035 |
| oracle | 95.1% | 0.016 |

- **So `within_95ci` is not a pass/fail.** The interval covers only
  test-set sampling noise, not the error model's own fitting noise. A
  well-fitted model therefore lands below 95% coverage, and here it does
  (81%).
- **Changes to the decile test.**
  - Kept unchanged: every decile |gap vs observed| < 0.10, ECE < 0.05, raw
    ECE > 0.15, AUROC > 0.75.
  - Added a **noise-free check against the truth**: per-decile
    |stated − true|. A first version used single-seed bounds (max < 0.04,
    mean < 0.015) that held only for seed 0; 78/100 seeds failed them
    (Critic, V-030). It now asserts the **median over seeds 0–9**: worst
    decile < 0.07, mean decile < 0.025.
  - These bounds come from the multi-seed spread. Over 200 seeds, a single
    seed's worst decile has median 0.044 (p95 0.071, max 0.088) and its mean
    decile has median 0.016 (p95 0.022). The 10-seed medians range
    0.035–0.052 and 0.014–0.019 over 20 disjoint seed blocks.
  - So **the typical worst decile is about 0.05 off the true error rate**
    (mean decile about 0.016) at 4000 fit samples. The seed-0 figures quoted
    above (0.018 worst) were a lucky draw.
  - A paired test confirms the check catches real mis-calibration. Stating
    every error 15% too low gives 10-seed medians of about 0.145 worst /
    0.08 mean (0.144–0.152 and 0.078–0.084 over the 20 blocks), and the test
    asserts that this fails.
  - Added: ECE ≤ oracle ECE + 0.02.
- **One stricter check I tried fails, so I did not add it:** "≥ 8/10 bins
  within the 95% CI". Seed 0 gives 6/10 and the oracle itself gives 9/10.
  Over seeds the expected value is about 8.1/10, so the check would be
  flaky for the reason above.

### V-011: which weights the error model belongs to
- `error_model_fit.json` and `error_model.json` (`fit_info`) now record
  `weights_sha256` (method `state_dict-sha256-v1`). This is a SHA-256 over
  the state_dict, built from sorted names, dtype, shape and raw tensor
  bytes, via `evaluators.error_model.weights_fingerprint` (still
  torch-free).
- It hashes content, not the file. At fit time `model_weights.pth` does not
  exist yet: `train.py` saves it after `train()`. A test checks that the
  recorded hash equals the fingerprint of the saved file loaded back.
- **Eval:** `TestTrainer` fingerprints the weights it actually scores.
  **Inference:** `predict_video` and `predict_images` fingerprint the loaded
  model.
- If the hash differs from the error model's, `load_error_model_near` warns
  ("fitted on different weights … No expected_error is reported") and the
  error model is **not applied**: no column, no calibration report. Applying
  it would print a number that belongs to other weights.
- Error models from before this change (no hash) are still applied,
  unchecked.
- **TODO hook for T1** is in `Trainer.train()`: with early stopping or a best
  checkpoint, reload the best weights before `_fit_error_model()`, and fit on
  a calibration split or out-of-fold predictions instead of `val_loader`.
  Validation then becomes a selection set.
- **Saying so in the output (V-011 note).** When the model is the constant
  fallback, eval now logs "expected_error is the same for every sample (base
  rate): <reason>". `error_estimate` in the metrics files also gains `note`
  and `fit_split`.

### Process-wide switch removed
- `set_fit_by_default` and `fit_by_default` are deleted, along with their
  call in `cli.py`.
- `--no_error_model` is still parsed and recorded (`self.no_error_model`,
  saved in `cli_arguments.json`). It now reaches the Trainer only through
  `trainers/{video,vision}/train.py`: `trainer.fit_error_model = not
  args.no_error_model` (05cf682).
- The Trainer fits by default (`getattr(self, 'fit_error_model', True)`).
- **The flag test is now stronger.** It is parametrised over
  video/images × flag on/off and runs the real `train()` and `Trainer` (only
  data and model are tiny stand-ins). For each case it checks:
  - the TRAIN and ONECALL parse;
  - `cli_arguments.json` holds the flag;
  - `error_model*.json` exists exactly when the flag is off;
  - a later opposite parse does not leak into the run;
  - the fitted hash matches the saved weights.
- The autouse fixture that guarded the global is gone.

### PRIOR_STRENGTH: tuned 2 → 0.5 on held-out checks, not the fixture
- **Setup.** A different 5-class generator, not the test fixture: max
  prob ~ U(0.4, 1) and three true-error curves, 1 − m^1.8, 1 − m and a step
  0.02 + 0.5·[m < 0.6]. Fit sizes 50 to 4000, 30 seeds each, scored by mean
  |stated − true P(wrong)| on 10k held-out draws.
- **Reproduce** (CPU, seeded, about 8 s):
  `python scripts/study_error_model_prior.py --seeds 30 --n_test 10000 --fit_sizes 50 200 1000 4000`.
  The default run takes seconds and reaches the same conclusion. The first
  version of this table came from scratch scripts with a different RNG
  order. The table below is the committed script's exact output and
  replaces it.

| curve, n_fit | prior 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| 1 − m^1.8, 50 | 0.0938 | **0.0664** | 0.0753 | 0.0946 |
| 1 − m^1.8, 200 | 0.0605 | **0.0511** | 0.0573 | 0.0681 |
| 1 − m^1.8, 1000 | 0.0317 | **0.0310** | 0.0343 | 0.0419 |
| 1 − m^1.8, 4000 | **0.0185** | 0.0198 | 0.0223 | 0.0269 |
| 1 − m, 50 | 0.0892 | 0.0692 | **0.0676** | 0.0731 |
| 1 − m, 200 | 0.0637 | 0.0528 | **0.0517** | 0.0565 |
| 1 − m, 1000 | 0.0285 | **0.0280** | 0.0299 | 0.0348 |
| 1 − m, 4000 | **0.0171** | 0.0181 | 0.0202 | 0.0242 |
| step, 50 | **0.0910** | 0.0993 | 0.1100 | 0.1274 |
| step, 200 | **0.0749** | 0.0847 | 0.0918 | 0.1023 |
| step, 1000 | **0.0429** | 0.0450 | 0.0479 | 0.0578 |
| step, 4000 | 0.0311 | **0.0290** | 0.0336 | 0.0379 |

- **Reading.**
  - Prior 2 is never the best and loses to 0.5 in all 12 settings.
  - Prior 0 is best for large fits and the step curve. It is clearly worse
    on smooth curves at small n (0.094 vs 0.066, 0.089 vs 0.069).
  - Best-in-row counts: prior 0: 5, 0.5: 5, 1: 2, 2: 0.
  - 0.5 is the robust choice and is now the default.
- On the fixture, prior 0.5 moved seed-0 ECE from 0.036 to 0.032 and the
  200-seed mean ECE from 0.035 to 0.024.

### Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`)
- `tests/test_error_model.py` + `tests/test_error_model_pipeline.py`:
  `34 passed, 8 warnings in 23.34s` (first follow-up). After the V-030 fix
  (multi-seed truth-gap test plus the mis-calibration test): see the end of
  the section.
- Targeted run: those two files plus test_evaluation, test_inference_data,
  training/test_video_trainer, training/test_vision_trainer,
  training/test_param_budget, test_workflow_reliability,
  models/test_desktop_config and models/test_video:
  `175 passed, 18 warnings in 30.46s`.

### Second follow-up (Critic V-030/V-031)
- The single-seed truth-gap bounds are replaced by 10-seed medians with
  bounds taken from the multi-seed spread (see the V-010 section above).
  `test_truth_gap_check_catches_a_miscalibrated_model` shows that a model
  stating every error 15% too low fails the check.
- The ECE ≤ perfect-estimator ECE + 0.02 check is kept on seed 0 (the
  Critic measured 100/100 seeds passing).
- `scripts/study_error_model_prior.py` (numpy only, seeded) reproduces the
  PRIOR_STRENGTH table exactly.
- Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`):
  - `tests/test_error_model.py` + `tests/test_error_model_pipeline.py`:
    `36 passed, 8 warnings in 31.67s`.
  - The same targeted run as above: `177 passed, 18 warnings in 31.18s`.

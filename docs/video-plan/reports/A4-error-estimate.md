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
      each block is shrunk toward the base rate with `PRIOR_STRENGTH = 2`
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
  - Process-wide switch: `set_fit_by_default()` / `fit_by_default()`.
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
- **Process-wide opt-out switch.** I could not edit `trainers/*/train.py`
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
  The global then becomes a fallback.
- No new dependencies.

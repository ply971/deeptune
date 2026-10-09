# Phase 5 — Rigorous evaluation & adaptive assurance

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 5 only. Show your plan first.

## Goal
Every video result DeepTune reports should come with an honest estimate of how
well it will do on data it has never seen — overall *and* per prediction.

## Tasks
1. **Validation schemes** (`--validation holdout|kfold|bootstrap632`):
   - holdout (current 70/10/20, keep default);
   - stratified k-fold and **grouped** k-fold when a `group` column exists
     (`--k`, default 5); the holdout test set stays sealed and is only scored
     once at the end;
   - Efron .632+ bootstrap (`--n_boot`) for small datasets.
   Each fold/bootstrap trains a fresh model with the same config, results
   aggregated with mean ± std and 95% CI. Save per-fold metrics and
   out-of-fold predictions.
2. **Metrics report** `evaluators/video/report.py` (shared helper is fine):
   classification → confusion matrix, accuracy, balanced accuracy,
   macro/weighted F1, per-class sensitivity, specificity, PPV, NPV,
   ROC curves + AUC (one-vs-rest for multiclass), PR-AUC;
   regression → MAE, RMSE, R², residual plot. JSON + CSV + PNG.
3. **Calibration:** reliability diagram, ECE, Brier score; temperature scaling
   fitted on validation (or out-of-fold) only; report before/after.
4. **Per-prediction error estimate (adaptive assurance):** fit an error model
   on out-of-fold predictions that maps per-sample features (max prob,
   entropy, margin, test-clip disagreement from multi-clip TTA, and optionally
   embedding distance to training set) → probability the prediction is wrong
   (cls) or expected absolute error (reg). Validate it on held-out data:
   report its calibration (stated vs observed error per decile) so the
   number is trustworthy. Output an `expected_error` column in every
   predictions file (eval, inference, timeline).
5. **df-analyze export:** `--export_dfanalyze` writes the clip embeddings +
   label (+ group) as a df-analyze-ready CSV with consistent split
   identifiers, so classical models, feature selection and stacking can be run
   on video embeddings. Document the command to run df-analyze on it.
6. **Localisation metrics** (`evaluators/video/localization_report.py`):
   ActivityNet-style detection mAP at tIoU 0.1–0.9 (report avg 0.1–0.5,
   0.1–0.7, all — the THUMOS'14 convention CtxPL uses), per-class AP,
   classification mAP over video-level scores, and mean boundary error.
   Must agree with the ActivityNet reference implementation
   (`compute_average_precision_detection` + `interpolated_prec_rec`, as used
   in CtxPL `eval/eval_detection.py`) on fixtures. Replaces the minimal
   version from Phase 4 if one was added.
7. **Evidential uncertainty as an error estimate:** CtxPL produces a Dirichlet
   uncertainty per snippet and per video. Validate it like the error model in
   task 4: does stated uncertainty track observed error (video-level label
   errors, and segment false positives) per decile on held-out data? Report
   honestly whether it is calibrated, and feed it into `expected_error` for
   localisation outputs if it is.

## Tests
- Metrics match scikit-learn exactly on fixed hand-made predictions
  (including multiclass ROC and NPV/PPV edge cases with zero denominators).
- Grouped k-fold: no group in both train and validation of any fold; holdout
  untouched by CV (assert indices).
- .632+ bootstrap: formula unit test against a hand-computed example.
- Temperature scaling reduces ECE on a synthetic over-confident model and
  never uses test data (assert via data-split bookkeeping).
- Error model: on synthetic data where true error rate depends on confidence,
  stated vs observed error per decile within tolerance.
- Old runs without these files still render in Results (no crash).
- Localisation mAP: matches the ActivityNet reference on hand-made
  prediction/ground-truth tables, including duplicate detections (only one
  TP per ground truth), predictions in videos with no ground truth, empty
  prediction sets, and segments that touch but do not overlap.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1`, paste the summary, and paste a
5-fold report on the Phase 2 benchmark for two models (one strong, one at
chance) including the calibration table. If anything fails twice, stop and
report.

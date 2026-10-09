# Phase 8 — Performance, real-data experiments, final report

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 8 only. Show your plan first, including the dataset choice,
download size and licence, and an estimate of GPU/CPU hours per experiment.

## Part A — Training performance & robustness
- Mixed precision (`--amp`, CUDA only, safe fallback), gradient accumulation
  (`--grad_accum`), early stopping on validation (`--patience`), LR scheduler
  (cosine / plateau), gradient clipping, checkpoint **resume**
  (`--resume RUN_DIR`), best-checkpoint saving, GPU auto-detect, DataLoader
  worker/prefetch defaults tuned with the Phase 1 benchmark.
- Tests: resume continues from the same epoch with identical optimizer state;
  early stopping stops at the right epoch on a scripted loss curve; AMP
  path skipped cleanly on CPU.

## Part B — Experiments (reproducible)
`scripts/run_experiments.py --config experiments/*.yaml` regenerates every
table and figure. Each experiment uses Phase 5 validation (k-fold with CI,
grouped where relevant) and saves raw per-fold results.
1. **Synthetic motion benchmark** (full 700/400 setting from the slide):
   all fusion strategies → reproduce the slide's finding (appearance-only ≈
   chance, motion-aware ≫ chance) with error bars.
2. **Real dataset:** e.g. 10-class subset of UCF101 or HMDB51 (confirm
   licence and that the official train/test splits — which are grouped by
   actor — are respected). Compare: single_frame, frame_pool (mean/attn),
   late, early, cnn_rnn, temporal_transformer, two_stream, r2plus1d_18,
   one video transformer if compute allows.
3. **Ablations:** frames per clip (4/8/16/32), temporal sampling
   (uniform vs segment), transfer-learning regime (head only vs full vs
   PEFT/LoRA — trainable params, time, accuracy), optical flow on/off,
   test-time clips 1 vs 3 vs 5.
4. **Unsupervised & assurance results:** clustering ARI on real embeddings,
   anomaly detection AUC with a held-out class as "anomaly", calibration
   before/after, expected-error calibration table.
5. **Explainability gallery:** a few correct and incorrect predictions with
   saliency + temporal importance.

## Part C — Deliverables
- `docs/video-plan/RESULTS.md` with all tables/figures and an honest
  limitations section (what doesn't work, compute limits, dataset biases).
- README and docs updated; a short demo video of the GUI workflow.
- Rebuild the Windows installer (`packaging/`) and check it launches.
- Final `scripts/check.ps1` run, tag `v-video-1.0`.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Every number in RESULTS.md must be produced by
`run_experiments.py` from saved raw outputs — no hand-typed results. Re-run
one experiment from scratch to show it reproduces within CI. Run
`scripts/check.ps1` and paste the summary. If anything fails twice, stop and
report.

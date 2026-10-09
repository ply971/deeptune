# Phase 8 — GPU performance, real-data experiments, final report

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 8 only. Show your plan first, including dataset choices,
download sizes and licences, and an estimate of GPU hours per experiment on
the RTX 5060 Laptop GPU (8 GB). Ask before any download.

## Part A — Training performance & robustness (GPU)
- Mixed precision (`--amp`, bf16/fp16 on CUDA, safe fallback on CPU),
  gradient accumulation (`--grad_accum`), early stopping (`--patience`), LR
  scheduler (cosine / plateau), gradient clipping, checkpoint **resume**
  (`--resume RUN_DIR`), best-checkpoint saving, GPU auto-detect,
  `channels_last` where it helps, DataLoader worker/prefetch defaults tuned
  with the Phase 1 benchmark. Document batch sizes that fit in 8 GB for each
  model family.
- Tests: resume continues from the same epoch with identical optimizer state;
  early stopping stops at the right epoch on a scripted loss curve; AMP path
  skipped cleanly on CPU.

## Part B — CtxPL reproduction on THUMOS'14 (do this first)
Data: the 2048-d I3D features + annotations used by CtxPL (from CO2-Net, linked
in the CtxPL README) and the released CtxPL checkpoint (47.6 avg mAP). Check
sizes and terms, then ask before downloading. Keep data outside the repo.
1. **Original code, original checkpoint:** run CtxPL's own `test.py` in a
   separate environment (its pinned torch 2.6+cu118 may not support Blackwell —
   if so, use the closest working torch and record it). Record mAP@0.1–0.7.
2. **Port equivalence:** load the same checkpoint into DeepTune's
   `CtxPLHead`, run DeepTune's evaluation → same mAP within a small tolerance
   (state the tolerance and why). Any difference must be explained (e.g.
   soft-NMS ties, ambiguous-segment filtering).
3. **Retrain the port from scratch** with the paper's defaults over ≥ 3 seeds
   → mean ± std vs 47.6. Report honestly; note which deviations from the
   public code (see `docs/video-plan/ctxpl-port.md`) might matter.
4. **Ablations (3 seeds each):** no context prototypes; with/without HNCPs;
   context vector used for both insert and update; `ctx_a/ctx_b` sweep;
   with/without snippet uncertainty in `m_ctx`.

## Part C — Experiments (reproducible)
`scripts/run_experiments.py --config experiments/*.yaml` regenerates every
table and figure. Each experiment uses Phase 5 validation (k-fold with CI,
grouped where relevant) and saves raw per-fold results.
1. **Synthetic benchmarks** (full 700/400 direction setting from the slide,
   untrimmed, scene_bias): all fusion strategies + TSM → reproduce the
   slide's finding with error bars; localisation A vs B; background-reliance
   scores (Phase 6) with/without context prototypes.
2. **Real classification dataset:** 10-class subset of UCF101 or HMDB51
   (confirm licence; respect official splits, which are grouped by actor).
   Compare: single_frame, frame_pool (mean/attn), late, early, tsm, cnn_rnn,
   temporal_transformer, two_stream (farneback vs mv), r2plus1d_18, one video
   transformer if it fits in 8 GB.
3. **DeepTune features for localisation (stretch):** extract RGB + motion
   snippet features with DeepTune backbones (e.g. r2plus1d_18 or tsm RGB; mv or
   Farneback motion) on a THUMOS'14 subset, train the CtxPL head, compare with
   the I3D-feature result. Only if raw-video download size and GPU time allow.
4. **Speed table:** decode backends, frame cache, feature cache vs pixels,
   Farneback vs codec MVs vs RAFT, TSM vs 3D CNN, token merging (if built),
   motion-guided vs uniform sampling — clips/sec, GPU memory, accuracy.
5. **Ablations:** frames per clip (4/8/16/32), temporal sampling
   (uniform/segment/motion), transfer regime (head / full / PEFT-LoRA:
   trainable params, time, accuracy), test-time clips 1/3/5.
6. **Unsupervised & assurance results:** clustering ARI on real embeddings,
   anomaly AUC with a held-out class as "anomaly", calibration before/after,
   expected-error calibration, evidential-uncertainty calibration (CtxPL).
7. **Explainability gallery:** correct and incorrect predictions with
   saliency + temporal importance; localisation examples with CAS curves.

## Part D — Deliverables
- `docs/video-plan/RESULTS.md` with all tables/figures and an honest
  limitations section (what doesn't work, compute limits, dataset biases,
  differences from the CtxPL paper).
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

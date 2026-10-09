# Phase 4 — Long-video analysis, CtxPL action localisation, unsupervised

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 4 only. This phase is large: show your plan first and split it
into steps (A, B1–B4, C), committing and updating PROGRESS.md after each.

## Goal
Go from "one label per clip" to *analysing* video: **where** in a long video
each action happens — learned from video-level labels only and without being
fooled by the background — which videos look alike, and which are unusual.

## Part A — Sliding-window timeline (baseline)
1. `analysis/timeline.py`: sliding-window inference over a video of any
   length using Phase 1 `dense` sampling (`--window_sec`, `--stride_sec`),
   batched, streaming (never decode the whole video into memory).
   Per window: start/end time, class probabilities (or regression value),
   embedding (optional).
2. Smoothing + segmentation: median filter or min-duration smoothing
   (`--min_segment_sec`), merge consecutive windows into **event segments**
   `{start, end, label, confidence}`. Export `timeline.csv`, `segments.json`.
3. Optional `--skip_static` (SCSampler-style scouting): use the cheap Phase 1
   motion score to skip windows with no motion; report the speed-up and any
   accuracy change.
4. Optional segment-level training labels: annotation CSV
   `video,start_sec,end_sec,label` → training clips cut from those ranges.
5. CLI: `infer.py --modality video --analyse_timeline ...` (or a new
   `analyse.py`), recorded in the run folder.

## Part B — Weakly-supervised action localisation with context prototypes (CtxPL)
Reference: Hussain & Gondra, *CtxPL: Context-based Prototype Learning for
Weakly-Supervised Temporal Action Localization*, CRV 2026 —
`github.com/moayadeldin/CtxPL` (MIT; builds on DELU, ECCV 2022, and CO2-Net,
MM 2021). Re-implement it cleanly inside DeepTune (`localization/`), with
attribution in the module docstring and README. Do **not** vendor the repo.

**B1 — Data.** `SnippetFeatureDataset` over the Phase 1 feature cache:
per video `[N, D_rgb + D_motion]`, video-level multi-hot labels, optional
ground-truth segments (eval only). Training batches follow CtxPL: `num_similar`
pairs of same-class videos + random videos (pairs are required by the
co-activity contrastive loss), snippet resampling to `max_seqlen` with random
perturbation in training, all snippets at test time. Batch size is a free
parameter.

**B2 — Model** `CtxPLHead` (keep the original module names `vAttn`, `fAttn`,
`fusion`, `classifier` and shapes so the released checkpoint loads):
- cross-modal attention between the two streams (each stream's features are
  filtered by attention computed from the other), fusion conv, classifier
  conv → class activation sequence (CAS, `C+1` with background) and
  attention;
- losses: MIL top-k (with and without background), evidential (EDL) loss with
  exp evidence and annealing, uncertainty-guided attention loss with the
  epoch-dependent curve, co-activity contrastive loss, mutual v/f attention
  loss, norm and guide losses;
- **context-based prototypes**: per class, up to `max_prototypes_per_class`
  prototypes built from the confident videos' context-weighted representation
  `W_t = a·m_fg(t) + b·m_ctx(t)` where `m_fg` = soft top-k of the class CAS
  and `m_ctx = (1 − m_fg)(1 − p_norm)(1 − u_t)`; momentum updates; replace the
  most-uncertain prototype when full;
- **HNCPs** (`--hncp`): videos above the uncertainty threshold become
  hard-negative context prototypes;
- prototype **alignment** loss and **attraction–repulsion** loss (InfoNCE-like,
  temperature 0.3).
Defaults = the paper's `options.py` values; all exposed as CLI flags and saved
with `save_args`. Flags for ablations: `--no_context_protos`, `--hncp`,
`--ctx_a/--ctx_b`, `--uncertainty_threshold`.

**Known issues in the public CtxPL code — handle explicitly and document each
choice in `docs/video-plan/ctxpl-port.md`:**
- `self.use_snip_unc` is read in `_contextual_video_repr` but never defined
  (training raises on the first confident video; `test.py` never hits it).
  Make it a real option (`--use_snippet_uncertainty`, default ON as the paper
  describes) and note the deviation.
- `torch.set_default_tensor_type('torch.cuda.FloatTensor')` and `.cuda()` calls
  → device-agnostic code (CPU tests, GPU training).
- `curve.repeat(10, 1)` hard-codes batch size 10 → use the real batch size.
- `Contrastive` assumes the first 6 items are 3 same-class pairs → derive
  pairs from the batch sampler.
- New prototypes use the context-weighted vector, momentum updates use the
  plain mean → keep the paper's behaviour by default, flag the inconsistency,
  add an option to use the context vector for both, ablate in Phase 8.
- `feature_size` 2048 = 1024 RGB + 1024 flow is hard-coded → configurable
  `D_rgb`, `D_motion`.
- `proposal_methods.py` uses an undefined `logger`; `AntgetLocMAP` has a
  leftover `pdb.set_trace()`.
- Duplicate `forward` definitions in `CTXPL` (identical; keep one).
- Prototypes live in Python dicts (not in `state_dict`) → save/load them
  explicitly alongside the checkpoint.

**B3 — Proposals & outputs.** Multi-threshold proposals on the attention
signal, outer-inner-contrast score + `gamma`·video score, per-class Gaussian
soft-NMS, optional ambiguous-segment filter. Convert snippet indices to
seconds using the feature sidecar (fps, snippet_frames, stride — CtxPL's
`25/16` factor must not be hard-coded). Output the same `segments.json` /
`timeline.csv` format as Part A (CAS as the timeline) so the GUI reuses it.

**B4 — CLI.** `deeptune.py --modality video --task localization ...` (train on
cached features), `infer.py ... --localize` (extract features if missing →
head → segments). Recorded in the run folder.

## Part C — Unsupervised analysis on clip embeddings
In `analysis/unsupervised.py`: PCA (explained variance), LDA when labels
exist (t-SNE/UMAP only if already installable — ask before adding deps);
K-Means (k-means++, restarts, elbow + silhouette) and agglomerative clustering
(Ward/average/complete, dendrogram data), ARI/NMI vs labels when present;
**anomaly detection** with a small autoencoder on embeddings (threshold from
validation percentile / target FPR), per clip and per timeline window.
Outputs CSV/JSON + PNG plots in the run folder.

## Tests
- Timeline (A): smoothing/merging unit tests on hand-written probability
  sequences (single window, all same class, flicker); on a synthetic long
  video, boundary error ≤ one stride using a stub or Phase-3-trained model;
  streaming: decode called per window, memory doesn't grow with length.
- CtxPL (B), on hand-made tensors with hand-computed expected values: soft
  top-k foreground mask, context weights `W_t`, EDL loss, MIL top-k loss,
  attraction–repulsion loss, alignment loss; prototype insert / momentum update
  / replace-most-uncertain / HNCP routing; batch sizes other than 10 work;
  model runs on CPU.
- **Checkpoint equivalence:** a CtxPL-format `state_dict` (generated in the
  test with the original module layout and random weights) loads into
  `CtxPLHead` and gives identical CAS/attention to a reference forward
  implemented from the original code. (The real released checkpoint is
  checked in Phase 8.)
- Proposals: OIC score and soft-NMS against hand-computed examples; snippet →
  seconds conversion against the sidecar.
- **Oracle:** on the Phase 2 `untrimmed` task (CI size, tiny features from a
  dummy backbone or cached fixtures), the head trained with video-level
  labels only reaches mAP@tIoU=0.5 above a set threshold and above the Part A
  baseline's random-proposal control. Use the Phase 5 mAP code if it exists
  yet; otherwise a minimal ActivityNet-style mAP in `localization/metrics.py`
  that Phase 5 will verify.
- Scene-bias: run with and without context prototypes on the `scene_bias`
  task and **report** the difference (do not assert that CtxPL wins — this is
  an experiment, not a unit test).
- Clustering: synthetic blobs → ARI ≥ 0.9; silhouette picks the right k.
- Anomaly: injected out-of-distribution clips → AUC ≥ threshold.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1`, paste the summary, attach an
example `segments.json` and plots from the synthetic untrimmed videos, the
mAP table (A vs B, B with/without context prototypes/HNCPs), and
`docs/video-plan/ctxpl-port.md`. If anything fails twice, stop and report.

# DeepTune Video Analysis — Term Project Plan

**Course:** CSCI 555 Data Mining & Machine Learning (StFX, Fall 2026)
**Repo:** `ply971/deeptune`, branch `GUI` (last commit 3072bd6, 23 Sep 2026)
**Goal:** grow DeepTune's existing video *classification* into a full
**video analysis** platform — the network should learn from motion, analyse long
videos over time, work without labels, report trustworthy error estimates,
explain itself, and be usable end-to-end from the desktop GUI.

---

## 1. Where the project stands (audit of the `GUI` branch)

**Already done — keep and build on it**
- Video modality wired into `deeptune.py` train → eval → embed pipeline.
- Two model families via `utils.build_video_model`:
  - *Frame pooling* (`src/video/frame_pool.py`): any 2D backbone (ResNet,
    EfficientNet, DenseNet, Swin, ViT, ConvNeXt, VGG, ± PEFT), shared across
    frames, then mean or attention pooling.
  - *Native 3D* (`src/video/video3d.py`, `video3d_peft.py`): r3d_18, mc3_18,
    r2plus1d_18, mvit_v2_s, swin3d_t/s/b with Kinetics-400 weights + presets.
- `ParquetVideoDataset`, CSV/XLSX video manifests, saved-model inference
  (`inference/video.py`, `infer.py`), desktop app with MP4 picker, labelling,
  player, Results and Test pages, Windows installer, ~117 tests.

**Gaps and risks found**
| # | Finding | Why it matters |
|---|---------|----------------|
| G1 | CI (`.github/workflows/test.yml`) triggers on `main`, but the fork's branches are `master`, `GUI`, `development`, `package` | CI never runs on your work |
| G2 | `docs/_build`, `docs/build`, `docs/source/_build` (incl. demo `.mp4`s) are committed | Repo bloat, noisy diffs |
| G3 | Whole clips are stored as bytes **inside parquet** and decoded per sample by writing a temp file and seeking with OpenCV | Will not scale past small datasets (RAM + disk I/O); OpenCV seeking on compressed video is slow and can be inaccurate |
| G4 | One uniformly-sampled clip per video; no temporal jitter/augmentation; no multi-clip test-time averaging | Weak generalisation; long videos lose most of their content |
| G5 | Attention pooling weights **class logits**, not embeddings; embeddings are always mean-pooled | Attention can't learn much; temporal information is discarded |
| G6 | Only 2 of the 4 fusion strategies on the *Learning from Video* slide exist in practice (frame pooling ≈ late fusion of predictions; 3D ≈ slow fusion). No single-frame baseline, early fusion, two-stream/optical flow, or recurrent/transformer temporal head | The core idea of the slide — *what you predict is in the motion* — isn't testable yet |
| G7 | Task = whole-clip label only | No timeline, event detection, anomaly detection or clustering of videos |
| G8 | Evaluation is a single 70/10/20 holdout | No k-fold / bootstrap / per-prediction error estimate; no group-aware splits for "many clips from the same source video/person" (leakage) |
| G9 | No explainability for video | Can't show *where* and *when* the model looked |
| G10 | Verification of video so far is on random generated clips ("execution, not accuracy") | Nothing yet proves a model actually learns motion |

---

## 2. How the slides map onto the project

| Slide | Becomes |
|---|---|
| **Learning from Video** (four fusion points, two-stream ConvNet, "motion is the whole signal" experiment, 3D conv / I3D / space-time attention) | Phases 2–3: the synthetic motion benchmark *and* the full fusion model zoo |
| Recurrent Networks / One Machine, Many Modalities (CNN front end + recurrent back end) | Phase 3: CNN+LSTM/GRU and temporal-transformer heads over frame embeddings |
| Transfer Learning (head / full / PEFT) | Kept for every new model; compared in Phase 8 |
| Autoencoder (anomaly detection), K-Means, Hierarchical Clustering, PCA, LDA | Phase 4: unsupervised video analysis on clip embeddings |
| Model Validation (holdout, k-fold, .632+ bootstrap), Classification Performance (ROC, sens/spec, PPV/NPV) | Phase 5: evaluation suite |
| Explainable AI (adaptive error estimation, calibrated risk), DeepTune "Adaptive Assurance" | Phase 5 (per-prediction error) + Phase 6 (saliency) |
| df-analyze, Feature Selection, Stacking & Super Learners | Phase 5: export clip embeddings to df-analyze for classical models/stacking |
| **Best Practices: AI Coding Agents** | The way this whole plan is run (section 4) |

---

## 3. Phases

Rough schedule assumes ~13 weeks. Each phase has a ready-to-paste prompt in
`docs/video-plan/prompts/`. Every phase ends with a green suite + commit.

| Phase | Name | Weeks | Prompt |
|---|---|---|---|
| 0 | Foundation & audit | 0.5–1 | `phase-0-foundation.md` |
| 1 | Scalable video data layer | 2 | `phase-1-data-layer.md` |
| 2 | Motion benchmark & baselines (the test oracle) | 1 | `phase-2-motion-benchmark.md` |
| 3 | Temporal model zoo (fusion taxonomy, two-stream, RNN, transformer) | 2.5 | `phase-3-temporal-models.md` |
| 4 | Long-video & unsupervised analysis | 2 | `phase-4-video-analysis.md` |
| 5 | Rigorous evaluation & adaptive assurance | 1.5 | `phase-5-evaluation.md` |
| 6 | Explainability | 1 | `phase-6-explainability.md` |
| 7 | GUI: Video Analysis workspace | 2 | `phase-7-gui.md` |
| 8 | Real-data experiments, performance, report | 1.5 | `phase-8-experiments.md` |

### Phase 0 — Foundation & audit
Fix CI triggers (G1), clean committed build output (G2), add `CLAUDE.md`,
`PROGRESS.md`, a one-command check script, record the baseline test count and
timings. Create a `video-analysis` branch off `GUI`. **Done when:** CI runs
on push to your branches and is green; baseline recorded.

### Phase 1 — Scalable video data layer
Path-based video manifests (store paths + metadata, not bytes) alongside the
existing bytes-in-parquet format; a decoder abstraction (TorchCodec or PyAV
preferred, OpenCV fallback) with accurate frame-index reads; temporal sampling
strategies (uniform, random-segment/TSN-style, dense sliding window with
stride); clip-level augmentation (temporal jitter, consistent spatial
crop/flip across frames, speed perturbation); multi-clip test-time averaging;
optional decoded-frame cache; `group` column support so clips from the same
source never cross splits; video metadata probing (fps, duration, resolution,
codec) with clear errors. **Done when:** a 1,000-clip manifest trains without
loading every clip into RAM, old parquet datasets still work, and timing
benchmarks are recorded.

### Phase 2 — Motion benchmark & baselines
Recreate the slide's measured experiment as a generator: textured blob drifting
across a static textured background, two classes that differ **only in the
sign of horizontal velocity**, trajectory centred on the middle frame, so no
single frame carries the label. Add a second synthetic task (e.g. speed
regression or periodic vs. aperiodic motion). Add the explicit **single-frame
baseline** model. **Done when:** the benchmark exists as a fixture + CLI
script, and tests assert *single-frame ≈ chance* (shows the data is honest).
These tests become the oracle for Phase 3.

### Phase 3 — Temporal model zoo
Implement the fusion taxonomy as selectable models: single frame, late fusion
(two distant frames merged at the top), early fusion (first layer spans the
clip), slow fusion (progressive), plus: fix attention pooling to act on
embeddings (G5); CNN→LSTM/GRU head; small temporal-transformer head;
**two-stream** (RGB stream + optical-flow stream, Farneback via OpenCV by
default, optional torchvision RAFT, late fusion); flow pre-computation/caching.
All ± PEFT where it makes sense. **Done when:** every motion-aware model beats
chance by a wide margin on the Phase 2 benchmark while single-frame does not —
asserted in tests — and everything trains/evaluates/embeds/infers through the
existing pipeline.

### Phase 4 — Long-video & unsupervised analysis
Sliding-window inference over arbitrarily long videos → per-window predictions
and probabilities → smoothed **timeline** with event segments (start/end/label/
confidence) exported as CSV/JSON; optional segment-level training labels
(`start,end,label` annotations). Unsupervised: clip embeddings → PCA / LDA
projections, K-Means and hierarchical clustering with silhouette/elbow,
and **video anomaly detection** with an autoencoder on embeddings
(reconstruction-error threshold). **Done when:** a long synthetic video with
known event boundaries is segmented with measured boundary error, and
clustering recovers known groups on synthetic data.

### Phase 5 — Rigorous evaluation & adaptive assurance
Stratified and **grouped** k-fold CV, .632+ bootstrap, full metric report
(confusion matrix, ROC/AUC, sensitivity, specificity, PPV, NPV, macro-F1;
MAE/RMSE/R² for regression) with confidence intervals; calibration (reliability
curve, ECE, temperature scaling); a **per-prediction error estimate** (adaptive
error, in the spirit of the course's Adaptive Assurance work — e.g. a held-out
model of error as a function of confidence/embedding, validated on out-of-fold
data); export of clip embeddings in a df-analyze-ready CSV for classical ML,
feature selection and stacking. **Done when:** metrics are verified against
scikit-learn on fixed fixtures and the error estimate is shown to be
calibrated on held-out data.

### Phase 6 — Explainability
Spatio-temporal Grad-CAM for 3D models and frame models; temporal importance
(attention weights over time; temporal occlusion — blank a window, measure
drop); rendered overlay MP4 / GIF and a per-frame importance curve.
**Done when:** on the motion benchmark, saliency concentrates on the moving
blob (measured overlap with the known blob mask beats a random baseline).

### Phase 7 — GUI: Video Analysis workspace
New desktop pages/panels (still torch-free in the GUI process): model family
picker with all Phase 3 models and plain-language descriptions; sampling and
augmentation controls; **Analyse video** page — open a long video, run a saved
model, scrub a timeline coloured by predicted class with confidence and error
band, jump to events; saliency overlay playback; embedding explorer (2D
PCA/LDA scatter, cluster colouring, click → play clip); evaluation report
viewer (CV, ROC, calibration); experiment comparison table.
**Done when:** UI tests drive each new callback, and a scripted walkthrough
from import → train → analyse → explain works on the synthetic data.

### Phase 8 — Real-data experiments, performance, report
Performance: AMP mixed precision, gradient accumulation, early stopping,
LR scheduler, GPU detection, dataloader worker tuning, checkpoint resume.
Experiments on a real public dataset (e.g. a 10-class subset of UCF101 or
HMDB51 — confirm licence/download size) plus the synthetic benchmark: fusion
strategy comparison, frames-per-clip ablation, head vs full vs PEFT,
flow vs no flow, with CV error bars. Final docs, demo video, installer
rebuild, and a results write-up for the course. **Done when:** a reproducible
`scripts/run_experiments.py` regenerates every table/figure in the report.

---

## 4. How we run this with AI (from the *Best Practices* slide)

1. **Two-AI workflow.** Claude (chat) = planner/reviewer; Claude Code (VS Code)
   = builder. Both see the repo. After each phase, paste Claude Code's final
   summary + test output back into the planner chat for review before the next
   phase.
2. **Phased plan.** This file. Each phase has its own prompt; the planner may
   revise the next prompt based on what the last phase revealed.
3. **Assurance in every request.** Every prompt ends with: check your work,
   finish it, add all relevant assurance checks, test it rigorously.
4. **Project setup.** `CLAUDE.md` instructions file, one-command tests, git
   commit per phase.
5. **Clear specs.** Each prompt states goal, constraints, files to touch, and
   definition of done.
6. **Verify, don't trust.** Run it end-to-end yourself; watch for weakened tests
   and hard-coded results (look at the diff of `tests/`).
7. **Manage context.** Fresh Claude Code session per phase (`/clear`);
   `PROGRESS.md` is the handoff.
8. **Permissions.** Auto-approve safe actions (read, run tests); keep approval on
   for deletes, installs, git push. No secrets in prompts.
9. **Iterate.** Redirect early; after 2 failed attempts at the same thing, reset
   and re-prompt with the real error text.
10. **Reuse & ownership.** Modular, re-runnable code — and *you* must be able to
    explain every part of it in your presentation.

# DeepTune Video Analysis — Term Project Plan

**Course:** CSCI 555 Data Mining & Machine Learning (StFX, Fall 2026)
**Repo:** `ply971/deeptune`, branch `GUI` (last commit 3072bd6, 23 Sep 2026)
**Goal:** grow DeepTune's existing video *classification* into a full
**video analysis** platform — the network should learn from motion, analyse long
videos over time, **localise actions in untrimmed video from video-level labels
without being fooled by the background (context)**, work without labels, run
fast (decode once, extract features once, cheap motion), report trustworthy
error estimates, explain itself, and be usable end-to-end from the desktop GUI.

**Hardware:** development machine has an NVIDIA RTX 5060 Laptop GPU (8 GB,
Blackwell). All training/experiments use it. CI and unit tests stay CPU-only.

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
| G2 | Build output is committed: `docs/_build`, `docs/build`, `docs/source/_build` **and `docs/source/build`** (424 files incl. demo `.mp4`s). The real demo videos live in `docs/source/_static` and are what the docs link to | Repo bloat, noisy diffs |
| G3 | Whole clips are stored as bytes **inside parquet** and decoded per sample by writing a temp file and seeking with OpenCV | Will not scale past small datasets (RAM + disk I/O); OpenCV seeking on compressed video is slow and can be inaccurate |
| G4 | One uniformly-sampled clip per video; no temporal jitter/augmentation; no multi-clip test-time averaging | Weak generalisation; long videos lose most of their content |
| G5 | Attention pooling weights **class logits**, not embeddings; embeddings are always mean-pooled | Attention can't learn much; temporal information is discarded |
| G6 | Only 2 of the 4 fusion strategies on the *Learning from Video* slide exist in practice (frame pooling ≈ late fusion of predictions; 3D ≈ slow fusion). No single-frame baseline, early fusion, two-stream/optical flow, or recurrent/transformer temporal head | The core idea of the slide — *what you predict is in the motion* — isn't testable yet |
| G7 | Task = whole-clip label only | No timeline, event detection, action localisation, anomaly detection or clustering of videos |
| G8 | Evaluation is a single 70/10/20 holdout. *(Phase 0 audit: grouped holdout splits already exist via `--grouper` → `StratifiedGroupKFold` in `handlers/split_dataset.py`, including for video.)* | No k-fold / bootstrap / per-prediction error estimate |
| G9 | No explainability for video | Can't show *where* and *when* the model looked |
| G10 | Verification of video so far is on random generated clips ("execution, not accuracy") | Nothing yet proves a model actually learns motion |
| G11 | The project `.venv` has `torch 2.14.0+cpu` although the machine has an RTX 5060 GPU | Every run so far is CPU-only; GPU is unused |
| G12 | Every epoch decodes pixels again; there is no "extract features once" path and no cheap motion source (optical flow would be computed per sample) | Training cost is dominated by decoding/flow, not learning |
| G13 | Nothing measures whether a model uses the **action** or the **background/scene** | Video models are known to classify by scene context; results could be right for the wrong reason |
| G14 | 2D frame path uses one fixed transform `Resize((224,224))` (helpers.py): no augmentation at all, and non-square/portrait video is squashed | Distorted inputs; nothing to regularise training |
| G15 | On CUDA, `options.py` turns on `NUM_WORKERS = min(8, cpu/2)`; on Windows each worker process gets a copy of a dataset that holds every clip's bytes | RAM multiplies with workers once the GPU is used |

---

## 2. How the slides and research map onto the project

| Source | Becomes |
|---|---|
| **Learning from Video** slide (four fusion points, two-stream ConvNet, "motion is the whole signal" experiment, 3D conv / I3D / space-time attention) | Phases 2–3: synthetic motion benchmark *and* the full fusion model zoo |
| Recurrent Networks / One Machine, Many Modalities | Phase 3: CNN+LSTM/GRU and temporal-transformer heads |
| Transfer Learning (head / full / PEFT) | Kept for every new model; compared in Phase 8 |
| Autoencoder, K-Means, Hierarchical Clustering, PCA, LDA | Phase 4 Part C: unsupervised analysis on clip embeddings |
| Model Validation, Classification Performance | Phase 5: evaluation suite |
| Explainable AI / Adaptive Assurance | Phase 5 (per-prediction error) + Phase 6 (saliency, background reliance) |
| df-analyze, Feature Selection, Stacking | Phase 5: export clip embeddings for df-analyze |
| **CtxPL** (Hussain & Gondra, CRV 2026 — `github.com/moayadeldin/CtxPL`, MIT) | Phase 4 Part B: weakly-supervised temporal action localisation with context (background) prototypes; Phase 2 background-bias benchmark; Phase 8 reproduction |
| Best Practices: AI Coding Agents | The way this whole plan is run (section 5) |

---

## 3. Research basis (what we implement, and why)

**Context = the background.** A model can recognise "long jump" from the
running track instead of the jump. CtxPL attacks this directly; the other
items make the whole pipeline fast enough to run it on real video.

| Method | Paper | What it gives us | Phase |
|---|---|---|---|
| **CtxPL** — context-based prototype learning for weakly-supervised temporal action localisation (WTAL) | Hussain & Gondra, CRV 2026 | Localise actions in untrimmed video from video-level labels. Foreground snippets (soft top-k of the class activation sequence) and confident *context* (background) snippets build multiple class prototypes; uncertain videos become hard-negative context prototypes (HNCPs); attraction–repulsion + alignment losses pull videos to action prototypes and push them from background ones. Built on DELU (evidential uncertainty, ECCV 2022) and CO2-Net features (RGB+flow, MM 2021). 47.6 avg mAP on THUMOS'14 | 4B, 8 |
| Scene-bias literature | Choi et al., "Why Can't I Dance in the Mall?" (NeurIPS 2019); SOAR (ICCV 2023) | Motivation and evaluation protocol for background reliance | 2, 6 |
| **Codec motion vectors** (compressed-domain video) | CoViAR, Wu et al. CVPR 2018 | H.264/MPEG-4 already store per-block motion; FFmpeg exports it during normal decoding (`+export_mvs`). Nearly free motion stream instead of costly TV-L1/Farneback flow | 1, 3 |
| **Motion-guided sampling** | MGSampler, Zhi et al. ICCV 2021 | Sample frames uniformly along the *cumulative motion* curve, not uniformly in time | 1 |
| Salient-clip scouting | SCSampler, Korbar et al. ICCV 2019 | Skip static windows in long videos using a cheap motion score | 4A |
| **Temporal Shift Module** | TSM, Lin et al. ICCV 2019 | Temporal modelling inside a 2D ResNet at zero extra FLOPs/params | 3 |
| Token merging | ToMe, Bolya et al. ICLR 2023; "Video, How Do Your Tokens Merge?" CVPRW 2025 | ~2× faster ViT/Swin backbones without retraining | 3 (stretch) |
| Fast decoding | TorchCodec (PyTorch), PyAV | Accurate frame-index reads, optional NVDEC GPU decoding | 1 |
| Feature-level training | standard in WTAL (CO2-Net, DELU, CtxPL) | Decode once → extract snippet features once → train heads in minutes | 1, 4B |

Every speed or accuracy number quoted from a paper is a *claim to check*, not a
result: we measure it on our own benchmark before it goes in the report.

---

## 4. Phases

Each phase has a ready-to-paste prompt in `docs/video-plan/prompts/`. Every
phase ends with a green suite + commit. Items marked **(stretch)** are done only
if the phase finishes early; they never block "done".

| Phase | Name | Weeks | Prompt |
|---|---|---|---|
| 0 | Foundation, audit & GPU setup | 0.5–1 | `phase-0-foundation.md` |
| 1 | Scalable video data layer + motion vectors + feature cache | 2 | `phase-1-data-layer.md` |
| 2 | Motion, untrimmed & background-bias benchmarks (the test oracle) | 1 | `phase-2-motion-benchmark.md` |
| 3 | Temporal model zoo (fusion taxonomy, TSM, two-stream, RNN, transformer) | 2.5 | `phase-3-temporal-models.md` |
| 4 | Long-video analysis, CtxPL action localisation, unsupervised | 3 | `phase-4-video-analysis.md` |
| 5 | Rigorous evaluation & adaptive assurance | 1.5 | `phase-5-evaluation.md` |
| 6 | Explainability & background reliance | 1 | `phase-6-explainability.md` |
| 7 | GUI: Video Analysis workspace | 2 | `phase-7-gui.md` |
| 8 | GPU performance, real-data experiments (incl. CtxPL reproduction), report | 2 | `phase-8-experiments.md` |

### Phase 0 — Foundation, audit & GPU setup
Fix CI triggers (G1), untrack committed build output (G2), add a one-command
check script, record the baseline. Install the CUDA build of the **same** torch
version into `.venv` so the RTX 5060 is usable (G11; Blackwell needs CUDA ≥ 12.8
wheels). Create a `video-analysis` branch off `GUI`. **Done when:** CI runs on
push to your branches and is green; `torch.cuda.is_available()` is true locally;
baseline recorded.

### Phase 1 — Scalable video data layer + motion vectors + feature cache
Path-based manifests alongside bytes-in-parquet; decoder abstraction
(TorchCodec / PyAV / OpenCV fallback) with exact frame-index reads; **codec
motion-vector extraction** (PyAV `export_mvs`) as a cheap motion field; temporal
sampling (uniform, segment/TSN, dense, **motion-guided/MGSampler**); clip
augmentation; multi-clip TTA; decoded-frame cache; **snippet feature extraction
+ cache** (decode once, run a frozen backbone once, store `[N_snippets, D]` per
video, separately for an RGB and a motion stream); `group` splits; metadata
probing; GPU-aware loading. **Done when:** a 1,000-clip manifest trains without
loading every clip into RAM, old parquet datasets still work, and the benchmark
table (decode backends, MV vs Farneback cost, feature-cache speed-up) is recorded.

### Phase 2 — Motion, untrimmed & background-bias benchmarks
(a) The slide's direction task (classes differ only in the sign of horizontal
velocity) + speed regression + periodic task; (b) **untrimmed videos** with
background-only stretches and known action boundaries, labelled only at video
level (WTAL oracle); (c) **background-bias** task: background texture family
correlates with the class in training but not in testing. Single-frame
baseline model. **Done when:** tests prove the data is honest (single frame ≈
chance on (a); flow/MV features ≥ 95%; background-only classifier high on the
biased split and ≈ chance on the unbiased one).

### Phase 3 — Temporal model zoo
Single frame, late, early, slow fusion; attention pooling fixed to act on
embeddings (G5); CNN→LSTM/GRU; temporal transformer; **TSM**; **two-stream**
with `--motion_source farneback|mv|raft`; token merging for ViT/Swin
(stretch). **Done when:** every motion-aware model beats chance by a wide
margin on the Phase 2 direction task while single-frame does not — asserted in
tests — and everything trains/evaluates/embeds/infers through the pipeline.

### Phase 4 — Long-video analysis, CtxPL action localisation, unsupervised
**A.** Sliding-window timeline with smoothing → event segments (baseline),
optional static-window skipping from motion scores.
**B.** A clean DeepTune port of **CtxPL** running on Phase 1 snippet features
(RGB stream + motion stream): DELU-style base (cross-modal attention, class
activation sequence, evidential uncertainty, MIL losses), context prototypes,
HNCPs, attraction–repulsion and alignment losses, multi-threshold proposals +
soft-NMS. Device-, batch- and feature-size-agnostic; checkpoint-compatible with
the released CtxPL weights. Same `segments.json` output as A.
**C.** Unsupervised: PCA/LDA, K-Means/hierarchical, autoencoder anomaly
detection on embeddings.
**Done when:** on the Phase 2 untrimmed task, both A and B are scored with
mAP@tIoU and boundary error; B's losses pass hand-computed unit tests; the
background-bias task is reported with and without context prototypes;
clustering recovers known groups.

### Phase 5 — Rigorous evaluation & adaptive assurance
Stratified/grouped k-fold, .632+ bootstrap, full metric report with CIs,
calibration (ECE, temperature scaling), per-prediction error estimate,
df-analyze export; **localisation metrics** (mAP at tIoU 0.1–0.7, verified
against the ActivityNet reference implementation) and validation of CtxPL's
evidential uncertainty as a per-segment error estimate. **Done when:** metrics
match scikit-learn / the reference on fixtures and the error estimate is shown
to be calibrated on held-out data.

### Phase 6 — Explainability & background reliance
Spatio-temporal Grad-CAM, temporal importance (attention, occlusion), rendered
overlays; a **background-reliance score** (fraction of saliency on the known
action mask vs background; for localisation, CAS mass inside vs outside the
true segments). **Done when:** saliency concentrates on the moving blob beyond
a random baseline, passes the randomised-weights sanity check, and the
background-reliance score is reported with/without context prototypes.

### Phase 7 — GUI: Video Analysis workspace
Model family picker (all Phase 3 models + localisation model), sampling,
motion-source and feature-cache options, Analyse-video page with timeline /
detected actions / saliency, embedding explorer, evaluation viewer, run
comparison. GUI stays torch-free. **Done when:** UI tests drive each new
callback and a scripted walkthrough works on synthetic data.

### Phase 8 — GPU performance, real-data experiments, report
AMP, grad accumulation, early stopping, schedulers, resume — tuned for the
8 GB RTX 5060. **CtxPL reproduction on THUMOS'14 features**: original
`test.py` with the released checkpoint → the DeepTune port loads the same
checkpoint and matches its mAP → retrain the port from scratch. Real-data
classification experiments (UCF101/HMDB51 subset), ablations (fusion, frames,
transfer regime, flow vs MV vs none, context prototypes on/off, HNCPs on/off,
speed table). **Done when:** `scripts/run_experiments.py` regenerates every
table/figure in the report.

---

## 5. How we run this with AI (from the *Best Practices* slide)

1. **Two-AI workflow.** Claude (chat) = planner/reviewer; Claude Code (VS Code)
   = builder. After each phase, paste Claude Code's final summary + test output
   back into the planner chat for review before the next phase.
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

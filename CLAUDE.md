# CLAUDE.md — DeepTune Video Analysis (CSCI 555 term project)

Read this file at the start of every session. Then read `docs/video-plan/PLAN.md`
and `docs/video-plan/PROGRESS.md` before touching code.

## What this project is
DeepTune is a one-command deep-learning library (images, text, tabular,
timeseries, video) with a Dear PyGui desktop app. This fork (`ply971/deeptune`,
branch `GUI`) already has: video classification/regression via frame pooling
(2D backbones + mean/attention pooling) and native 3D models (r3d_18, mc3_18,
r2plus1d_18, mvit_v2_s, swin3d_t/s/b), PEFT/LoRA, embeddings, inference, and a
desktop GUI with MP4 import and preview.

The term project turns this into a **video analysis platform**: scalable video
data handling (decode once, extract features once, codec motion vectors), the
full temporal-fusion taxonomy (single frame / late / early / slow fusion, TSM,
two-stream, CNN+RNN, temporal transformer), long-video timeline analysis,
**weakly-supervised action localisation with context (background) prototypes
— a DeepTune port of CtxPL** (`github.com/moayadeldin/CtxPL`), unsupervised
analysis, rigorous evaluation, explainability, and a GUI to drive all of it.

## Hardware
Development machine: NVIDIA RTX 5060 Laptop GPU, 8 GB (Blackwell — needs a
CUDA ≥ 12.8 torch build; see PROGRESS.md for the exact install command once
Phase 0 records it). Use the GPU for all training, feature extraction and
experiments. **Unit tests and CI must still pass on CPU only**: never require
CUDA in a test; skip GPU-only paths cleanly with a reason.

## CtxPL port rules
- Re-implement, don't vendor. Credit Hussain & Gondra (CRV 2026, MIT) in
  docstrings and README.
- Keep module names/shapes compatible with the released checkpoint.
- Every deviation from the public code goes in `docs/video-plan/ctxpl-port.md`
  with the reason.

## How we work (non-negotiable)
1. **One phase per session.** Work only on the phase you were given. If you
   notice something for a later phase, write it in PROGRESS.md under "Notes for
   later" — do not build it.
2. **Plan first.** Before editing, post a short plan: files you will touch,
   new files, tests you will add, risks. Wait for approval if the prompt says so.
3. **Verify, don't trust.** Every feature ships with tests. Never weaken,
   skip, `xfail`, or delete an existing test to make the suite pass. Never
   hard-code expected outputs or special-case test inputs in production code.
   If a test is truly wrong, stop and explain why instead of changing it.
4. **Run the full suite before you say "done"**, and paste the real summary
   line (e.g. `142 passed, 3 warnings`). "Should pass" is not acceptable.
5. **Two failed attempts at the same fix = stop.** Write what you tried and
   what you saw into PROGRESS.md and ask for direction.
6. **Commit per phase step** with a clear message (`video: <what>`). Small,
   reviewable commits. Never force-push. Never commit data, weights, `.venv`,
   `.artifacts/`, or build output.
7. **Update `docs/video-plan/PROGRESS.md`** at the end of every session:
   what was done, test counts, decisions made, open problems, next step.
8. **No secrets** in code, prompts, logs or commits (HF tokens etc.).

## Commands (Windows, from repo root)
```powershell
.venv\Scripts\python.exe -m pytest tests -q            # full suite
.venv\Scripts\python.exe -m pytest tests -q -k video    # video-focused
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts\smoke_video.py --model resnet18 --manifest
.venv\Scripts\python.exe scripts\smoke_video.py --model r3d_18
.venv\Scripts\python.exe desktop\app.py                 # GUI
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # GPU check
```
Once Phase 0 adds it, `scripts\check.ps1` runs all of the above in one go.

## Architecture rules
- Video clips flow through the code as tensors shaped **[B, T, C, H, W]**.
  Models that need [B, C, T, H, W] permute internally (see `src/video/video3d.py`).
- Every video model exposes `forward(x, extract_embed=False)` and returns
  `[B, num_classes]` (cls) or `[B, output_dim]` (reg), so the shared
  `Trainer` / `TestTrainer` / embedding / inference code keeps working.
  New models are registered through `utils.build_video_model`, never by
  special-casing the trainers.
- The desktop GUI process must **never import torch** (it launches
  `deeptune.py` / `infer.py` as subprocesses — see `desktop/runner.py`).
  New GUI features follow that rule.
- New CLI flags go in `cli.py` with help text and sane defaults, and are
  recorded by `args.save_args` so inference/Test can rebuild the model.
- Old parquet-with-bytes video datasets must keep working (backward compatible).
- Tests must run on CPU in CI in reasonable time: use tiny synthetic clips
  (e.g. 8 frames, 32×32 or 64×64) and `weights=None` / dummy models in unit
  tests. Real pretrained-weight runs belong in `scripts/` smoke checks.

## Definition of done (every phase)
- Feature works end-to-end through the CLI (and the GUI if the phase says so).
- New unit + integration tests; full suite green; `pip check` clean.
- Assurance checks from the phase prompt all pass, with evidence pasted.
- README / desktop README / docs updated for anything user-visible.
- PROGRESS.md updated; work committed.

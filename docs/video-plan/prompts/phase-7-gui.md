# Phase 7 — GUI: Video Analysis workspace

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md`,
and `desktop/README.md` first. Work on Phase 7 only. Show your plan first
with a sketch (ASCII is fine) of each new page/panel.

## Goal
Everything built in Phases 1–6 is usable from the desktop app by someone who
never touches the CLI. The GUI process stays torch-free: all heavy work runs
through `desktop/runner.py` subprocesses (`deeptune.py`, `infer.py`,
analysis/explain commands) with live logs and cancel.

## Tasks
1. **Dataset page (video):** manifest import with `group`, `start/end`
   columns; segment-annotation CSV import; per-video metadata (fps,
   duration, resolution) shown from a lightweight probe (subprocess or
   OpenCV-only, no torch); warnings for very short / odd-fps / unreadable
   clips; one-click "generate synthetic motion benchmark" for demos.
2. **Model & Training page:** model-family picker covering every Phase 3
   architecture with a one-line plain-language description (e.g. "Early
   fusion — the first layer sees the whole clip"); only valid backbone/option
   combinations selectable (extend `desktop/config.py` rules); temporal
   sampling, augmentation, test clips, optical flow options; validation
   scheme (holdout / k-fold / bootstrap). Equivalent CLI command shown, as
   today.
3. **New "Analyse video" page:** pick a trained run + a (long) video →
   run timeline analysis → video player with a **timeline bar** coloured by
   predicted class, confidence and expected-error band underneath; click a
   segment to jump there; list of detected events; export CSV/JSON.
   Toggle saliency overlay playback and the temporal-importance curve
   (from Phase 6 outputs).
4. **Embedding explorer:** 2D PCA/LDA scatter of clip embeddings coloured by
   label / cluster / anomaly score; choose k, see silhouette; click a point
   to play that clip; list top anomalies.
5. **Results page:** evaluation report viewer — CV table with mean ± CI,
   ROC, confusion matrix, calibration diagram; **compare runs** table (select
   several runs → side-by-side metrics) and export.
6. Keep settings save/restore and session logs working for all new options.

## Tests
- Extend `tests/test_desktop_ui.py` style: dispatch every new callback
  registered with Dear PyGui headlessly; assert state changes, generated CLI
  commands, and that invalid combinations are blocked.
- Test that importing `desktop.app` does not import torch
  (`'torch' not in sys.modules` after import).
- Timeline widget logic (time ↔ pixel mapping, segment hit-testing) unit tests.
- A scripted end-to-end walkthrough (can be `@pytest.mark.slow`): generate
  benchmark → train tiny model via runner → analyse a long synthetic video →
  explain → results/compare pages load the outputs.
- Old run folders without new files still load without errors.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1`, paste the summary, and take
screenshots of every new page (save under `.artifacts/ui/`, list paths).
Update `desktop/README.md`. If anything fails twice, stop and report.

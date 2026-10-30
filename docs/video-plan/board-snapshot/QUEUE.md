# Coordinator queue (written by the coordinator only)

Wave 1 status as of 2026-10-30. A1 and A4 were merged before reviews existed.
- Running fix rounds:
  - S1a: branch `video/S1a-zeroshot-track`; must-fix V-001, V-002, V-003
  - A1: branch `video/A1-fixes`; V-006, V-007 interim, V-015
  - A4: branch `video/A4-fixes`; V-010, V-011 prep, remove the global switch
- When a fix round reports: ask the Critic to re-check exactly those items, then merge on its OK.

Update: **A1 fixes MERGED** (1e2e0ff, Critic V-027; 124 passed on targeted tests).
Update: **A2 MERGED** (966a5c1, V-033) and **A4 fixes MERGED** (d517c15; trainer conflict resolved per V-033). Targeted tests 414 passed, 3 skipped.
  Now running: the A3 merge-in + fix round (V-021..V-023 + flip guard), and A4 adding the multi-clip hash e2e test (branch video/A4-wrapper-test).
  After both: merge → full suite on CPU → PROGRESS.md handoff → STOP (Wave 2 in a fresh session).
Update: A2 fix round done (f944c7e..5b76953, V-032; 245 passed targeted) → the Critic re-checks it, including the hash-on-wrapped-model question for the A2 × A4 merge.
Plan agreed with the user's question on limits: finish Wave 1 here, write the PROGRESS.md handoff, then a FRESH session plus a fresh Critic and Scout for Wave 2.
Update: Critic V-030/V-031 on A4-fixes says "merge after one small fix" (truth-gap bounds from the multi-seed spread, commit the prior study script, fix report line 34) → sent to A4.
  Merge note: A2 conflicts with A4-fixes in trainers/vision/trainer.py (fit_info.update block). Keep A2's fit_loader size AND A4's weights_sha256 keys.
Update: A4 fix round done (c73b4b8..671626f, V-029; 175 passed targeted) → the Critic re-checks it and adjudicates A4's disagreement on the V-010 diagnosis.
Update: the desktop visual upgrade was COMMITTED as-is at the user's request (680db84). **S1a MERGED** (71dce3d, Critic V-028;
tests/spatial + desktop 110 passed). Remaining Wave 1: A4 fixes (running) → Critic → merge; A2 fixes (running) → Critic → merge; then A3.
(resolved) WARNING: the main tree had UNCOMMITTED changes from another Claude session (desktop visual upgrade at ~04:05:
desktop/app.py, desktop/README.md, tests/test_desktop_ui.py, PROGRESS.md entry). They are not ours. Don't commit,
stash or revert them; ask the user. Merges that touch those files (A2/A3 desktop/config.py is fine; desktop/app.py is not) must wait.
Update: S1a fix round done (cd39d45, 6687930, b900983, b8c538a; V-026; tests/spatial 67 passed) → the Critic re-checks it after A1.
Update: A1 fix round done (3ec722c, 97202c1, c7c5b02; V-025) → the Critic is re-checking it. The A2 fix round has STARTED.

## Queued (start when a builder slot frees; max 3 builders)
1. **A2 fix round (started)** (agent A2, worktree `.claude/wt/A2`, branch `video/A2-data-layer`). First merge or rebase the latest
   `video-analysis` into it (A1/A4 fixes may have landed). Must-fix:
   - V-016: `--test_clips` breaks calibration. Fit the error model on validation with `num_clips=test_clips` through the
     multi-clip wrapper, record test_clips in error_model_fit.json, warn on mismatch, add a test.
   - V-017: `infer.py` CLI ignores the recorded sampling flags. Fall back to the checkpoint's cli_arguments.json when a
     flag isn't given, print what was used, add a test.
   - V-018 (suspected): trimmed `-c copy` MP4s. Skip `packet.is_discard` when indexing, check that the first decoded
     frame's pts matches, fall back to sequential reading; add a test if a fixture can be made with PyAV.
   - Nice-to-have (V-019): only the train pool persistent on Windows; a test that `--video_augment` never touches
     val/eval clips; Windows memmap cleanup.
   Then the Critic re-checks → **merge A2**.
2. **A3 rebase + fix round** (agent A3, worktree `.claude/wt/A3`), only after A2 is merged. Rebase or merge onto
   `video-analysis`. Expect one-line conflicts on the `--pooling` line in cli.py and infer.py; keep `single_frame`.
   Must-fix:
   - V-021: the runner validates on test. Generate a val split for every task (seeded), or set fit_error_model=False
     and label it as test. Correct the report's "validation" numbers.
   - V-022: keep the single-frame claim scoped to "linear and frozen probes"; the fine-tuned single_frame GPU run is
     a coordinator task that needs the user's OK. Do not write "pre-registered thresholds".
   - V-023: accept `single_frame` in desktop/config.py pooling (minimal), or note it for Phase 7.
   - Follow-up after both merge: wire A2's `--video_flip` to `benchmarks.guards.check_flip_augmentation`.
   Then the Critic re-checks → **merge A3** → run the full suite on CPU → PROGRESS.md → stop and report to the user.

## Needs the user's OK (do not start without it)
- GPU: `scripts\check.ps1`; `scripts\bench_video_io.py --clips 200 --nvdec`; `scripts\smoke_track.py` (downloads ~1.3 GB);
  the fine-tuned single_frame:resnet18 run (V-022).
- pip install `trackers` (P-110 parity check).
- Starting Wave 2 (B1, B2, B3 + T1, R1, R2, P-105, P-106, P-108).

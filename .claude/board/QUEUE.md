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

### 2026-10-30 - Codex A3 continuation stopped by user
- Read the rules and handoff; A2 is merged, so A3 is the first unfinished item.
- Worked in .claude/wt/A3. Verified CPU configuration: cpu 0 with CUDA_VISIBLE_DEVICES=-1.
- Merged current video-analysis (5cb6293) into A3 without conflicts (commit 3be531b). Existing uncommitted A3 documentation/test-docstring edits preserved. No implementation edits made.
- Pytest: not run in this continuation; no new pytest summary exists.
- Read-only Critic review interrupted at the user's stop request. Preliminary findings: A3 report has stale open-problem bullets about desktop single_frame validation and flip-guard wiring; src/video/single_frame.py still says it must stay at chance, beyond the V-022 evidence scope.
- Next: resolve those documentation inconsistencies, finish review, run targeted and full CPU tests, then merge only after review approval. The sync merge used Git's default message; correct it to the required video: prefix when resuming, before publishing. No merge into video-analysis, GPU jobs, installs, downloads or pushes.

## 2026-10-30: WAVE 1 CLOSED (coordinator)
- A3 merged (5bf2588, V-038). Full suite: 506 passed, 3 skipped, 26 warnings in 281.41s.
- Next: Wave 2 in a fresh session (see PROGRESS.md). Needs the user's OK for the GPU checks, the `trackers` install, and the Wave 2 start.

## 2026-10-30: Wave 2 session (coordinator)
- The user said "start Wave 2, GPU checks first". GPU checks run one at a time: check.ps1, bench_video_io, smoke_track, fine-tuned single_frame + lossless. `trackers` install was NOT approved.
- New user request: Analyse Video workspace, steps V1a..V1e in PLAN.md 4b. V1a joins Wave 2 once smoke_track passes on the GPU. V1d's model choice goes to the Wave 2 Scout.
- GPU checks done (PROGRESS.md). smoke_track passed, so V1a is unblocked.
- Resolved: the user approved the `model.cpu()` fix in `tests/test_video_followups.py`; committed on video-analysis. Re-run `check.ps1` on the GPU at the end of the wave.
- Running: T1, B1 + Scout. Only 2 builders for now: free RAM was 2 GB at launch (limit is otherwise 3). Worktree `.claude/wt/B2` is ready, not started. Next as slots free, in order: B2, V1a, B3, R1, R2 (+ P-105, P-106, P-108). Each branch: ml-critic review, then merge. Full suite after the wave.
- For B2: fix the two bugs in PROGRESS.md (silent FourCC fallback in benchmarks/render.py; flow_probe on periodic in the runner) and the nits `benchmarks/guards.py:38` utf-8-sig and `benchmark.md` "val.csv for scene_bias".
- Scout done: F-012..F-015 (appended by the coordinator; the Scout's write was refused), P-115..P-118 open, full table in scout-wave2.md. V1d default Qwen3.5-2B, fallback Qwen3.5-0.8B: neither fits-proven, and the fit gate needs the user's OK for the download. P-115..P-118 need the user's decision.
- The user approved everything still open (not pushing): P-115..P-118 recorded in PLAN.md 4b, P-116 sent to T1, `trackers` installed, Qwen download approved for V1d. B2 still held: free RAM under 1 GB.
- B1 done (a714204..0f35e71): ml-critic review running. B2 started from 2f57425. Coordinator GPU run still due for B1: `bench_video_io.py --clips 200 --feature_clips 100 --feature_model r3d_18 --feature_pretrained --feature_gpu`.
- T1 done (a714204..ff3aea7): ml-critic review running. Merge order: B1, then T1 (conflicts in datasets/video_datasets.py, desktop/config.py, handlers/video_args.py, infer.py; keep both sides' options). V1a started. Coordinator decision for T1's fix round: lower the default LoRA rank for native 3D PEFT (1000 gives about 97M trainable); value decided after the review.
- R3 moved up: right after R1 + R2, with --search smart|exhaustive (PLAN.md 4b). User request.
- B1 review V-039: merge after fixes, 12 must-fix items. Fix round running on video/B1-feature-cache, then ml-critic re-checks, then merge B1 before T1. P-119 open (probe evaluation parity); its core is item 12 of the fix round.

## 2026-10-30 19:25: resumed after the Claude usage limit (coordinator)
- LOCK removed as stale: Codex's session log ends at 17:06:37 with usage_limit_exceeded; no B1 file changed after 17:06:47. Codex left no QUEUE entry and its B1 edits are uncommitted in .claude/wt/B1; they are kept and built on.
- The Claude limit also killed the B2 and V1a builders (last edits 16:55) and the T1 ml-critic review (no entry after V-039). All restarted from their worktrees. Uncommitted work kept.
- Running: B1 fix round (V-039), B2, V1a, ml-critic on T1. Merge order unchanged: B1, then T1, then B2, V1a. Then B3, R1, R2, R3 + P-105, P-106, P-108.
- V1a done (430f738..bbd8e47, 146 passed targeted): ml-critic review running. B3 worktree created from 3c29de9.
- T1 review V-040: merge after fixes. Coordinator decision: --lora_rank default 16, absent = embed_size; r=8/16/32 compared at the UCF101 gate. T1 fix round started, includes the 2D Swin PEFT fix. B3 still held on RAM.
- B1 fix round done (0f35e71..229649a, 294 passed targeted): ml-critic re-check running, then merge B1. B3 starts from the merged tree. allow_abbrev=False in infer.py kept (coordinator). Open from B1: vit_h_14 pretrained rgb probably needs --feature_input_size 518 (by reading, not run); rest of P-119 (no infer path for probes).
- V1a review V-041: merge after fixes. Coordinator decisions: frames step default 1; fuse_score=False for zero-shot detectors. V1a fix round running.
- B1 re-check V-042: merge after 3 small fixes (A case-colliding ids, B bytes-parquet trim leak, C motion geometry). Second fix round running.

## 2026-10-31 00:25: resumed after the second Claude usage limit (coordinator)
- All agents stopped around 20:13. Committed and awaiting ml-critic re-check: B1 second fix round (229649a..37e4c89), T1 fix round (ff3aea7..6fca8e3), V1a fix round (bbd8e47..a0cb16d). B2 has uncommitted work in its worktree, kept and built on. B3 worktree has not started.
- Running: ml-critic re-checks of B1 and T1, B2 builder (max 3 agents, free RAM 4.1 GB). V1a re-check next. Merge order unchanged: B1, T1, B2, V1a. Then B3, R1, R2, R3 + P-105, P-106, P-108.
- B1 merged (05b401b) after V-043, V-047 fix rounds; the last round (f333f25..555a38a) was verified by the coordinator with the critic's repro scripts, not a fresh ml-critic. T1 merged (8314d32, V-044). V1a: V-045 fix by coordinator (6a6d41e), merges ahead of B2 because it is clean. B2: V-046 fix round. late_fusion done-when goes to the user.

## 2026-10-31: resumed after the third Claude usage limit (coordinator)
- All agents stopped around 01:47. B2 committed V-046 items 3-7 (4cda654, merged with 94c5ecd); items 1, 2 and the late_fusion pin are left. B3 has 2d64a37 plus uncommitted validation.py, localization_report.py and their tests, kept and built on.
- Coordinator decision on late_fusion (the question from V-046): keep the half-speed test as the done-when for late_fusion, pin the 0.548 limit at default speed in a test, and say so in PLAN and the report. Reported to the user; they can overrule.
- Running: B2 finish, B3 finish. Next: ml-critic on each, merge B2 then B3, then R1, R2, P-105, P-106, P-108.
- B3 built (2d64a37..14c0f11, 418 passed targeted). Review V-048: merge after fixes. Fix round on video/B3-evaluation. Coordinator decisions: follow Eq. 32 and correct B3's own new test (it never merged, and it pins the deviation); df-analyze default test for finetuned/peft, all for pretrained, group auto-included, --grouper printed. Merge order stays B2, then B3; the pretrained semantic fixes need B2 merged first.
- B3 fix round done (14c0f11..5a429f4, 457 passed targeted). Re-check V-049: merge after 3 small fixes, plus one at the B2 merge. Second fix round running. Split default for straddling sources waits for P-105 (warning only for now).
- B2 fix round done (4cda654..464d71b; test_temporal_oracle 18 passed in 967.82s). Measured: cnn_rnn 800 clips 9/10, temporal_transformer 1500 clips 5/10, late_fusion half speed 7/10, default speed 0/10. CORRECTION to my 2026-10-31 note: B2 puts the two-frame limit near 0.663, not 0.548; ml-critic is checking it. Use the checked figure in PLAN. Fresh ml-critic re-check running.
- B3 re-check V-050: one wording fix, done by coordinator (793d59d, tests/test_localization_report.py 64 passed). B3 is ready to merge after B2. At that merge: refuse pretrained + two_stream early in embed/video/embed.py:80-83. For P-105: pre-cut UCF101 clips get no shared-source warning. Group-aware null before any per-segment uncertainty claim (Phase 4).
- B2 re-check V-051: merge after small fixes, no training bug. Last fix round running on video/B2-temporal-models (floor=1, seed_study marker, docs, closed-form limit 0.66347, --model_seed, seeds for the other models). Phase 3 done-when goes to the user with a recommended rewording. CLAUDE.md commands not edited: the critic suggested it, and that needs the user.
- B2's last fix round stopped on the monthly spend limit (HTTP 429). Resume by messaging the B2 agent, or restart from V-051 and this queue if it is gone. Uncommitted edits in .claude/wt/B2 are kept. B3 is ready at 793d59d; it merges after B2, with the pretrained + two_stream refusal. Waiting on the user's spend limit and the Phase 3 done-when.
- User approved adding seed_study to the CLAUDE.md commands: committed on video/B2-temporal-models. The marker and check-script deselection are still to be done in B2's fix round (V-051 item 2); they must land before B2 merges.
- User accepted the reworded Phase 3 done-when (PLAN.md). The B2 report and README claim stands as V-051 item 7. Resuming B2.
- B2 merged (569790b), B3 merged (cb5d761), pretrained + two_stream fix (b42c308). Full suite run was STOPPED by Claude Code for low system memory; re-run only when the user asks. R1, R2, P-105, P-106, P-108 wait on it.
- Full suite after B2 + B3: 1656 passed, 5 skipped (CPU). R1 and R2 started, worktrees from the same commit; 2 builders because of memory. Then P-105, P-106, P-108, R3. check.ps1 on the GPU and the real-data gates wait for the user.
- User request 2026-10-31: device choice in the desktop app (Auto / GPU / CPU) for training, testing and analysis. Today options.get_device() picks automatically and only track.py has --device. Design: runner.py sets CUDA_VISIBLE_DEVICES=-1 in the subprocess env for CPU, keeps the GUI torch-free, and refuses GPU when none is visible. Small; start after the GPU checks, per the user's order.
- User said to ignore check.ps1 and continue. It stays recorded as CHECK FAIL: two test-assumption failures are open and the tests were not changed. Running the GPU checks one at a time in their own windows, logs in .artifacts/gate: B1, T1, B2, B3 5-fold, V1a.
- User cut P-105, P-106, P-107, P-108, P-109 and R3 from the plan (PLAN.md 4b). Do not start them. Older queue lines naming them are superseded.
- Correction: the user did NOT mean to cut P-105 to P-109 and R3; they were asking how long they take. Restored in PLAN.md. All six stay in the plan, in their earlier order.
- User, 21:09: GPU checks stopped. P-105 to P-109 and R3 moved to a later extension. Order: V1a real clip + page 07, R1, R2, Wave 3. Device setting stays queued. Older queue lines are superseded.
- R1 built (8512a09..87a2ff2). Review V-052: merge after fixes. Fix round running. Coordinator decisions: ensembles refuse unverifiable label mappings unless --allow_unverified_labels; explicit --pooling mean refused on clip models. R2 review running. Whichever merges second applies abstain=False to ensemble members.
- R2 built (8512a09..01b27f6). Review V-053: merge after fixes, maths correct. Fix round running. Coordinator decision: the default is named Learn then Test, not conformal; PLAN gets reworded at merge. P-120, P-121 open.

### 2026-10-31 - Codex custom tracking intake
- User requests training for a specific object, integration into page 07, and separate videos processed one at a time.
- Read-only inspection confirms pretrained Grounding DINO/OWLv2 plus ByteTrack; fine-tuned detector integration is described as S1b in the S1a report.
- Waiting for the object name and training/test data locations. No implementation, training, downloads or GPU jobs performed.
- Pytest: not run (intake only); no pytest summary exists.
- Next: identify target and available labelled data, then scope training and page 07 integration under this user request.


### 2026-10-31 - Codex animal tracking scope
- User has no dataset and delegates target choice, suggesting an animal. Proposed cats, using Oxford-IIIT Pet segmentation masks to derive whole-animal boxes (provided XML boxes cover heads only).
- Proposed compact torchvision Faster R-CNN baseline plus existing ByteTrack, local checkpoint support on page 07, and independently sourced cat clips processed sequentially. Compare pretrained and fine-tuned performance; no accuracy claim yet.
- Dataset/model/video downloads and bounded GPU training require user OK under AGENTS.md. No downloads, training or implementation performed.
- Pytest: not run (research only); no pytest summary exists.
- Next: obtain download/GPU authorization, implement training and local detector loading with CPU tests, then train and evaluate on separate data.


### 2026-10-31 - Codex cat detector implementation; dataset download blocked
- User approved choosing cats, dataset/model/demo downloads, and bounded GPU training. Worked in .claude/wt/cat-tracking on video/cat-tracking; main application code unchanged.
- Added a local torchvision cat detector backend, page 07 checkpoint field, reproducible Oxford Pet training script, four regression tests, and report docs/video-plan/reports/cat-tracking.md. Whole-animal boxes derive from trimaps; dog negatives; disjoint image splits; validation selects epochs; baseline/test AP50 reports; batch 2, workers 0, GPU allocator cap 55%.
- CPU verified cpu 0. Targeted suite: 219 passed in 87.73s (0:01:27). pip check: No broken requirements found. git diff --check clean. Earlier default-temp test run had 59 passed, 160 setup errors (sandbox temp access); an attempted --basetemp lacked its parent (2 passed, 1 error); creating the workspace artifact parent resolved this without changing tests.
- Full CPU suite launched with session 59111, log .claude/wt/cat-tracking/.artifacts/cat-full-pytest.log; still running at handoff, no final summary yet. Do not assume it passed or run another full suite concurrently.
- Oxford original and alternate thor dataset hosts both failed with WinError 10060 after sandbox network escalation. Stopped retries under the two-attempt rule and asked user whether to proceed using a checksum-verified mirror. No dataset, trained checkpoint, GPU training, demo download, or real video tracking produced.
- Next: receive direction on mirror; inspect full-suite log; obtain data with known MD5 verification; train/evaluate; run and inspect two independent cat videos sequentially; review and merge only when checks pass. No pushes.
- Draft implementation committed as f000a33 on video/cat-tracking; not merged or pushed.

### 2026-11-01 - Cat tracking cancelled by user
- User said nvm; stopped task-owned train_cat_detector.py processes and removed task lock. Existing code and downloaded artifacts preserved.
- No trained checkpoint or completed video tracking was produced in this session. Main checkout contains prepared detector/page07 code; full prior CPU suite: 1662 passed, 5 skipped, 92 warnings in 2017.52s; follow-ups: 152 passed in 66.91s and 2 passed in 1.08s.
- Next: resume only if requested; inspect data preparation before starting training.

### 2026-11-01 - Cat experiment resumed at user's request (verification running)
- The user requested continuation, superseding the earlier cancellation. Five bounded GPU epochs completed on 897 train / 239 validation / 300 test photos, seed 42; epoch 4 selected by validation AP50. Test AP50 0.991724606 baseline versus 0.993064677 trained; precision@0.5 0.865454545 versus 0.963562753; recall@0.5 0.991666667 for both. Four explicit missing/invalid annotation exclusions are recorded, with the original test cohort unchanged.
- Trained and baseline checkpoints each processed the two downloaded cat videos sequentially; trained overlays exist under .artifacts/cat-tracking/trained-video01 and trained-video02. Installed main saved_models/cat-detector/best.pt; page07 discovers it without importing torch. Counts are fragmented track IDs, not unique cats; no video accuracy claim.
- Code committed eb18fef on video/cat-tracking; reviews V-054..V-056 accepted fixes and experiment/report. New preflight/mirror tests: 3 passed in 0.17s; earlier combined targeted run: 9 passed in 11.86s. pip check clean.
- First current full CPU suite ended in disk-full write failures without a pytest summary. Approved cleanup removed only old/failed generated pytest temp folders (about 10.8 GiB), preserving all logs/data/checkpoints/videos. Full CPU rerun active: tool session 33964; log .claude/wt/cat-tracking/.artifacts/resume-check/full-pytest.log, JUnit full-results.xml. Do not start another full suite concurrently or assume this rerun passed.
- Next: finish full suite, record actual summary and final handoff, commit report, release LOCK. Main's prepared cat source edits remain preserved; no merge or push performed.


### 2026-11-02 - Cat experiment verification complete (Codex)
- Continued the cat-tracking handoff in .claude/wt/cat-tracking, branch video/cat-tracking. The previous full-suite process was no longer running and its XML was incomplete; ran one fresh full CPU suite with CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=2 and MKL_NUM_THREADS=2. Device preflight: cpu 0.
- Full pytest summary: 1668 passed, 5 skipped, 92 warnings in 2076.31s (0:34:36). Exit code 0; includes seed_study tests. Log/JUnit: .claude/wt/cat-tracking/.artifacts/final-check-20261102/pytest.log and results.xml. pip check: No broken requirements found. Existing tests were not weakened or newly skipped.
- Finished the reviewed experiment report with the actual verification result. Added six page 07 usage steps to desktop/README.md. Main checkout's prepared integration now includes the reviewed training-data validation/exclusion fixes and their regression tests, plus the same usage guide.
- Rechecked installed saved_models/cat-detector/best.pt against experiment.json (matching SHA256) and existence of all four baseline/trained video overlays. Existing V-054 through V-056 cover implementation, saved cohorts/checkpoints, report and real-video observations. No new training, GPU jobs or downloads in this continuation.
- Explained how to use Trained cat detector with prompt cat, Frames step 1 and Find, track and count. Live GUI demonstration was unavailable because the computer-use native pipe was absent; no new live-GUI verification is claimed.
- Cat experiment verification is complete. No merge or push in this continuation; main's prepared integration and other existing files are preserved. Next project item remains R1's V-052 fix/review round, followed by R2's V-053 fix/review round; broader B2/B3 GPU gates remain open as recorded in PLAN.md. Do not infer tracking accuracy or unique-animal counts from unlabelled demo videos.
- Saved the final report, page 07 guide and task handoff in local commit af441fe (video/cat-tracking).

### 2026-11-02 - Fresh cat-video test sources (Codex)
- User requested videos not used by the cat model or earlier demonstrations. Selected new independent sources: Moomin1 (Tet, recorded 2021-01-31, CC0-1.0, about 19 s) and Cat chasing mouse in a room (Tilman Piesk/Watchduck, recorded 2019, CC-BY-SA-4.0, about 35 s). Sources: https://commons.wikimedia.org/wiki/File:Moomin1.webm and https://commons.wikimedia.org/wiki/File:Cat_chasing_mouse_in_a_room.webm.
- These sources were not used in this project's Oxford photo fine-tuning or previous video runs. No detector inference, GPU job, training or model change was performed. Original backbone pretraining overlap is not fully audited; a newly recorded phone video is the clearest never-before-existing test input.
- Prepared an uncommitted asset-only helper under .artifacts/cat-tracking/fresh-test-videos/prepare.py. Wikimedia API metadata succeeded after sandbox network access, but the 360p transcode and original-file fallback each returned HTTP 429. Stopped after these two download attempts under the repo rule; no new video was saved or decode-verified. Manual source-page download links supplied to user.
- Pytest not rerun: data-source intake/download attempts only, no production code or test edits. Last completed full CPU suite remains 1668 passed, 5 skipped, 92 warnings in 2076.31s (0:34:36), from the earlier verification, not this asset preparation.
- Next for this request: user may download a new source manually or record a phone clip, then load it on page 07 with Trained cat detector and prompt cat. Project queue order otherwise unchanged. No commit, merge or push for this asset-only request.

### 2026-11-02 - New local cat test clip supplied (Codex)
- User requested a usable local path after Wikimedia downloads were rate-limited. Downloaded a different Pexels clip (6864989, cottonbro studio) into .artifacts/cat-tracking/fresh-test-videos/new-white-cat.mp4: 7.8 seconds, 11.92 MiB, 1440x2732. First and last frames decoded successfully with OpenCV. Source/licence/hash and provenance are in adjacent ATTRIBUTION.json; download helper is artifact-only.
- This clip was not used in this project's fine-tuning or earlier model/video runs. No detector inference was performed on it during preparation; user can test it first in page 07. Full original backbone pretraining overlap remains unaudited.
- No production code or test changes, GPU work, training, installs, commits, merges or pushes. Pytest not rerun for this asset-only request; last completed full CPU summary remains 1668 passed, 5 skipped, 92 warnings in 2076.31s (0:34:36), from earlier verification.
- Next: load the supplied path on page 07 with Trained cat detector and prompt cat. Other queued work unchanged.

## 2026-11-02: resumed (coordinator)
- State on resume: cat-tracking reached video-analysis in the user's commit 5940e5f. The R1 and R2 fix-round agents had died, leaving uncommitted work. Free RAM is about 1 GB, so at most 2 agents run at once.
- R1: the V-052 fix round was complete but uncommitted. Targeted CPU tests: 184 passed, 4 warnings in 62.86s. Committed as 50fa8d1. An ml-critic re-check is running.
- R2: uncommitted edits cover V-053 items 1-5, 7, 9, 10 and the stale split_summary / eval-output warning. A builder is resuming it for the tests, item 8 (mutants delta/8, score-derived grid, simulations through fit_abstention), the correction to R2's own `test_few_answered_groups_say_loose` (it pins the old wording that item 9 removed; it is not a pre-R2 test) and the report.
- Merge order: R1 after its re-check, then R2. Item 6 (members' abstain=False) and the PLAN:242,298 rewording to Learn then Test are applied at the R2 merge. After both merges: full suite on CPU.
- R1 re-check (ml-critic): merge after one fix. All 7 V-052 mutants were killed, and 13 of 14 new ones. Regression: the GUI always sent `--pooling mean`, and the new refusal stopped every GUI-built vjepa2/videomae run. Coordinator fixed it in f1b8705, along with a test for the untested infer.py refusal (the surviving mutant). Targeted plus desktop tests: 350 passed. The critic's repro now passes. **R1 MERGED 4e3e1a6.** Open nice-to-haves (recorded in R1.md): the `--predictions` "verified" check covers only the classes present; `pooling: mean` in a --config is replaced; abbreviated flags are refused under --ensemble.
- FOUND, pre-existing, not R1: 11 page-07 tests in tests/test_desktop_analyse.py fail whenever `saved_models/cat-detector/best.pt` is installed, because analyse_page.py:741-742 preselects the cat detector. Reproduced on a clean HEAD checkout with the file linked in: 11 failed, 136 passed. Without the file: 225 passed. The full suite in the main checkout therefore fails on this machine. Needs the user's call: (a) test isolation, with a fixture that hides the installed checkpoint plus one test of the preselection, or (b) keep the zero-shot default and only suggest the cat detector.
- R2 fix round done (45283db, f313431): V-053 items 1-5 and 7-11 are fixed. Every new test fails on 01b27f6. Mutants delta/8, delta/2 and both score-derived grid mutants were killed. Only R2's own `test_few_answered_groups_say_loose` changed, and it keeps the same checks. Targeted: 155 passed, 3 deselected; seed_study: 3 passed. An ml-critic re-check is running and will also give the item-6 merge instructions. Still open: Holm or a coarser grid, betting bounds, and classes_missing not recomputed after carving.
- R2 re-check (ml-critic): merge after fixes. Items 1-5 and 7-11 are fixed; 158 passed, including seed_study. 31 of 32 mutants were killed; BI3 (no recorded weights hash) survived. Must-fix: (1) a CSV calibration file reads group ids such as '001' as int, so overlap with the parquet splits is missed and the result is a false guarantee_holds; (2) class names that are numbers different from their indices are read as indices with only a warning. Also R2.md:149 cites the clip-weighted error. The item-6 merge plan is in the second fix round. The R2 builder was resumed for that round and for merging video-analysis (R1) into R2. The coordinator rewords PLAN.md 223, 242 and 298 ("conformal" becomes Learn then Test for the default) at the merge.
- R2 second round done (2e03a0d..6c4ddb6, with video-analysis merged into R2 at 0648e30). The coordinator re-ran both critic repros: a CSV calibration file now voids, and ambiguous numeric class names are refused. **R2 MERGED 5ac8f79**, with PLAN.md 223/242/298 reworded to Learn then Test (V-053 item 11). Merged tree targeted: 432 passed, 3 deselected.
- Next: full CPU suite (it needs the user's OK, and RAM is low); the user's choice on the page-07 tests (a) or (b); then Wave 3. R2 still open: Holm or a coarser grid, betting bounds, classes_missing recompute.
- User, 2026-11-02: **R3 (auto model selection) now, right before Wave 3.** Split into R3a (engine, exhaustive mode, ledger and resume, sealed test, CLI) and then R3b (smart staged mode, data profile, regret study against exhaustive). One builder at a time because of RAM; an ml-critic review on each. Worktree .claude/wt/R3, branch video/R3-model-selection. The full suite on the merged tree is still owed (it was stopped by low memory).
- R3a built (eefe3a9..1d50af8): select_model.py --search exhaustive, sealed test via the new deeptune.py --cv_only / --test_df, ledger and resume, pilot time estimate. Tests: 27 + 1 passed. ml-critic review running.
- Regression found by the R3a builder and fixed on video-analysis in 92dc079: R2 always passed abstain=, which broke 2 V-017 tests in tests/test_video_followups.py (a stand-in with the old signature). The test was not edited. Lesson: include test_video_followups.py in the targeted set for any merge that touches infer.py.
- User asked about augmentation: it exists (off by default, CLI only, never reaches frozen-feature probes). Proposed adding it as a search setting in R3b; waiting for the user's answer.
- R3a review (ml-critic): merge after fixes. The design is sound: sealing and grouping are correct, and the deeptune.py refactor changes no behaviour. Must-fix: (1) `--deeptune_args` can override `--df`/`--seed` and so undo the sealing; (2) shlex in POSIX mode drops Windows backslashes; (3) equal folds are never checked (M2 survived); (4) ranking claims are untested (M5, M9, M11 survived); (5) no code version in the ledger. Nice-to-haves 6-12 were also sent: group dtype, the rows-vs-groups wording, the refit seed, the caveat on the reason, an imbalance warning, failed vs refused, and an M12 guard. The builder was resumed for the fix round and will first merge video-analysis.
- User, 2026-11-02: YES to augmentation as an R3 search setting and to random and Bayesian search strategies (PLAN.md 4b, "R3 additions"). They go in R3b, after the R3a merge. Optuna's install needs the user's OK when R3b reaches it.
- 2026-11-03: the R3a builder was stopped by the usage limit mid fix round (its merge of video-analysis is committed as 668f480; its new tests are uncommitted). Resumed.
- User, 2026-11-03: **every feature goes in the desktop app.** Recorded in PLAN.md 4b with a GUI coverage test (every CLI flag must be reachable, apart from a documented allow-list). From now on, a feature is not done until it has its GUI control. The catch-up of features that have already landed needs scheduling (proposed: after R3, before Wave 3).
- User, 2026-11-03: order = R3a, then R3b, then **G1 GUI catch-up** (everything landed so far, plus the coverage test), then Wave 3.
- R3a fix round done (89b6b0c..fd30824). The coordinator verified it with the critic's tools: the bypass is refused, backslashes survive, and the mutants M2, M5, M9, M11 and M12 are killed. **R3a MERGED.** Merged tree, broad targeted set: 225 passed. `--deeptune_args` now takes separate words and comes last. Next: R3b (smart, augmentation setting, random, bayes), then G1, then Wave 3. The full suite is still owed.
- User approved installing Optuna: optuna 5.0.0 installed in .venv, pip check clean. It is an optional dependency, imported lazily by --search bayes.
- R3b started on video/R3b-smart-search (worktree .claude/wt/R3, from 7c7228a): smart as the default, random, bayes (Optuna TPE), the augmentation axis, and a regret study against exhaustive. No GUI in this round (it comes in G1). An ml-critic review follows, then the merge, then G1.

- 2026-11-03 coordinator: R3b builder done (99b4755, 43e3a51, e74841a on video/R3b-smart-search; targeted 82+2+26 passed per builder). ml-critic review launched.

### 2026-11-03 - Codex full real-data benchmark campaign (user priority override)
- User requested full UCF101 / THUMOS14 action evaluation and tracking benchmarks, and explicitly approved downloads and sequential GPU jobs. This supersedes G1 for this session.
- Worktree .claude/wt/real-benchmarks, branch video/real-benchmarks; coordinator LOCK held. Data/output D:\DeepTuneBenchmarks. User freed D: space (about 116 GiB observed).
- Full MOT17 public-detection baseline completed: all 21 labelled sequence/detector combinations, official pinned TrackEval; ablations completed. Association-only, no detector accuracy or challenge-test claim.
- Full UCF101 mirror download in progress with source SHA256; no classification training yet. Full THUMOS14 features and CtxPL checkpoint downloaded; preparing upstream reference evaluation. DeepTune CtxPL port remains absent.
- New targeted CPU tests: 19 passed in 16.47s. Full suite pending; no existing tests changed. No merge/push.

### 2026-11-03 - Full benchmark results and active UCF101 queue (Codex)
- User approved full-dataset/model downloads and sequential GPU experiments, prioritising this over GUI catch-up. Benchmark branch/worktree: video/real-benchmarks / .claude/wt/real-benchmarks. Saved code/protocol/report commit 780ac26; no merge/push. No installs or existing test modifications.
- Full CPU verification: 2090 passed, 5 skipped, 99 warnings in 2215.31s (0:36:55). Final targeted benchmark tests after two added checks: 21 passed in 41.76s; pip check clean. CPU device verified cpu 0; two threads, temporary files on D:. Full log/JUnit: D:/DeepTuneBenchmarks/verification/full-cpu-01.log and full-cpu-01.xml.
- Completed upstream CtxPL checkpoint reference on the exact full author release (200 training / 212 test, 20 classes, original RGB+flow I3D features). Average mAP tIoU 0.1-0.7 47.6124093%; 0.1-0.5 57.7984844%; 66.14 s, about 954 MiB peak reserved GPU. Not DeepTune port/retraining or the 213-video/raw-recognition cohort. Source 9209139, matching annotation hashes, process-local compatibility changes recorded in result.json.
- Completed full MOT17 public-detection ByteTrack evaluation: default plus no_low/no_fuse/zero_buffer on all 7 videos x 3 streams, all 5316 source frames. Default HOTA/MOTA/IDF1: FRCNN 47.36/49.40/55.20, SDP 54.41/64.06/64.91, DPM 31.66/30.31/39.90. Official TrackEval 12c8791, GT preprocessing on; DPM sigmoid transform declared. Saved 2000 whole-video bootstrap CIs and paired ablation differences. Association-only, not detector accuracy or hidden-test results.
- Full UCF101 downloaded (mirror SHA256 verified), extracted and audited: all 13320 videos/101 classes across official splits 1/2/3; train/test counts 9537/3783, 9586/3734, 9624/3696; zero group/path overlap. Complete official cohorts, no tiny subset.
- ACTIVE GPU queue: D:/DeepTuneBenchmarks/results/ucf101-campaign01; 37 configurations x 3 official splits x 3 seeds = 333 jobs, up to 30 epochs with validation-only best selection and early stopping. Started frozen r3d_18/split1/seed42; first epoch training observed progressing, about 2.45 GiB total GPU usage and 3 GiB available RAM. No UCF accuracy result yet. Logs queue.log and r3d18_frozen/split1/seed42/run.log. Runner session 21140. Never launch a duplicate queue or concurrent GPU job. RUNNING.lock is a process lock; code/manifest identity sealed to 780ac26, so do not edit/commit Python code in this worktree during the campaign. Failed/interrupted jobs require inspection and explicit resume; automatic stops on failure or low disk/RAM. No automatic checkpoint deletion or silent cohort reduction.
- Decoder warnings observed: some original UCF AVI container counts exceed decodable frames; existing loader repeats final frames and logs each case. Record this preprocessing limitation rather than claiming all frames decode exactly.
- Full MOT17 JPEGs and both Apache detector models are ready on D:. All 5316 labelled frames extracted once without transcoding. End-to-end detector+tracking runs remain to be scheduled sequentially after classification yields the GPU; do not call public-detection results detector validation. Cat detector needs cat-labelled video GT, not MOT17 person GT.
- Generated report: .claude/wt/real-benchmarks/docs/video-plan/reports/real-benchmarks.md; operation protocol REAL_BENCHMARKS.md beside it. Raw data/models/outputs/cache remain in D:/DeepTuneBenchmarks. User freed D: space. Pre-merge ml-critic review still pending; no merge authorised by this handoff. Separate feature-cache/probe/ensemble/CV/search protocols and DeepTune CtxPL port remain open.

### 2026-11-03 - UCF101 queue stopped on a decoder defect; repair verification active
- Supersedes the ACTIVE queue note above: campaign01 stopped at 32% of its first epoch, with no UCF result/checkpoint. Runner exited and removed its RUNNING.lock. Never describe it as still training.
- Real clip GolfSwing/v_GolfSwing_g22_c03.avi contains 348 timestamped packets but 275 displayed frames. Packet-based metadata falsely claimed exact frame counts; seek fallback merged incompatible packet/display ordinals, then raised for missing frame 277.
- First fix in benchmark worktree datasets/video_io.py: build index/count from actual decoded frames (no RGB conversion); retain presentation order; clear partial seek maps before sequential fallback. Regression on non-display packets added without altering existing tests. Failing real clip now samples 16 frames pixel-equal to independent sequential decoding.
- Targeted decoder/data suite: 76 passed, 3 skipped, 2 warnings in 16.28s. Full-data independent pixel audit of all 13320 UCF videos active (session 50661), log D:/DeepTuneBenchmarks/verification/ucf-decode-audit01.log. Fresh full CPU suite active (session 36284), log D:/DeepTuneBenchmarks/verification/full-cpu-02.log. Do not start another full suite or GPU training until these gates pass.
- The reader fix adds an actual decode pass for exact metadata, so previous IO speed figures do not describe its new cost. Source changes are not yet committed; no merge/push. Preserve campaign01 logs. After passing gates, commit the fix and use a new campaign output directory because the code fingerprint changed. MOT detector images and both checkpoints are ready; no end-to-end detector benchmark run yet.

- 2026-11-03 coordinator: R3b critic V-057 (merge after fixes); fix round f3d546d, a1be5fb, 0fd063c; coordinator re-check: tune_final repro gives 3 final runs, foreign --n_trials refused, no-Optuna 40 passed 4 skipped, critic surfaces A/B smart regret 0 on 4/4 seeds. Merged 34de59f. Merged-tree targeted: 437 passed (desktop, eval, ensemble, hub, splits, followups) + 97 passed (model selection). Background run was first killed for low memory; rerun in foreground. G1 worktree .claude/wt/G1 (video/G1-gui-catchup) created.

### 2026-11-03 - Full UCF pixel audit PASSED; guarded sequential continuation installed
- All 13320 UCF101 videos / 101 classes passed independent sequential pixel comparison at all 16 sampled positions (213120 samples), no excluded clips, in 648.33 s. Evidence: D:/DeepTuneBenchmarks/verification/ucf-decode-audit01/result.json and videos.jsonl.
- Decoder fix and independent regression/audit committed as 9da38fa on video/real-benchmarks. Decoder/data suite 76 passed, 3 skipped; benchmark+new regression suite 23 passed in 14.05s. Current fresh full CPU suite still running in session 36284 (actual pytest PID 25516), log full-cpu-02.log; no pass assumed.
- Task-owned coordinator session 39791 waits for that exact full-suite process to exit 0, verifies JUnit, the full-data audit, source revision 9da38fa and absence of uncommitted Python changes, and waits for >=3 GiB RAM before each GPU stage. It then runs full MOT17 OWLv2-person and Grounding-DINO-person end-to-end benchmarks sequentially, followed by the fresh full UCF101 campaign02 (333 jobs). Detection failures are saved independently, never silently retuned; classification can still run. No GPU stages overlap within this controller.
- Coordinator script (artifact-only): .claude/wt/real-benchmarks/.artifacts/benchmark-tests/continue_after_verification.py. Live state: D:/DeepTuneBenchmarks/results/continuation-status.json; log continuation-coordinator.log. Do not launch a second controller, full suite, or concurrent GPU job. Do not change/commit Python code in the benchmark worktree while this verified campaign proceeds.
- All original MOT17 training JPEGs and both detector snapshots are available on D:. This continuation does not make prior public-detection scores end-to-end detector results. The frozen-UCF first attempt failed before an epoch and is preserved in campaign01, with no accuracy claim.
- Full requested campaign remains unfinished. Main-branch code has not been merged or pushed; pre-merge review remains required. Background verification/experiments continue from their task-owned tool sessions. Future status must read continuation-status.json and job logs, not assume training is active or all stages passed.

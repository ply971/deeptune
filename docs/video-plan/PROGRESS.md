# PROGRESS — DeepTune Video Analysis

Claude Code: update this file at the end of **every** session. Newest entry on top.
Keep entries short and factual. This is the handoff between sessions.

## Status board
| Phase | Status | Tests (passed/total) | Commit |
|---|---|---|---|
| 0 Foundation & GPU setup | done (1 known GPU smoke timeout → Phase 1) | 173/173 CPU and GPU | `video-analysis` |
| 1 Data layer + MVs + feature cache | merged (A2, B1); B1 GPU and real-video gate open | full suite 1191 passed, 5 skipped (CPU) | `966a5c1`, `05b401b` |
| 2 Motion / untrimmed / scene-bias benchmarks | merged (A3); fine-tuned single_frame and lossless checks done on the GPU (one seed) | full suite 506 passed, 3 skipped (CPU) | `5bf2588` |
| + Param budget (A1), error estimate (A4), zero-shot detect+track (S1a) | merged with review fixes | see session log | `1e2e0ff`, `158ecdb`, `71dce3d` |
| + Training basics (T1), Analyse Video (V1a) | merged with review fixes; GPU and real-video gate open | full suite 1191 passed, 5 skipped (CPU) | `8314d32`, `94c5ecd` |
| 3 Temporal models | merged (B2); target reworded by the user; GPU and real-video gate open | full suite 1656 passed, 5 skipped (CPU) | `569790b` |
| 4 Timeline + CtxPL localisation + unsupervised | not started | | |
| 5 Evaluation | merged (B3); 5-fold assurance report, `check.ps1`, GPU and real-video gate open | full suite 1656 passed, 5 skipped (CPU) | `cb5d761` |
| 6 Explainability & background reliance | not started | | |
| 7 GUI | not started | | |
| 8 GPU experiments, CtxPL reproduction, report | not started | | |

## Session log

### 2026-10-31 — B2 and B3 reviews; Phase 3 target changed
- **Merged, later the same day:** B2 (569790b), B3 (cb5d761), and the pretrained + `two_stream` fix (b42c308: pretrained embeddings use RGB, and the new test fails without the fix). B2's last fix round (464d71b..c564763) and b42c308 were checked by the coordinator, with no fresh ml-critic.
- **Full suite (CPU, `CUDA_VISIBLE_DEVICES=-1`, at 12493ed):** `1656 passed, 5 skipped, 92 warnings in 1225.02s (0:20:25)`. An earlier run was stopped by Claude Code for low system memory, with no results. `check.ps1` on the GPU is still due.
- **B3:** ready at `793d59d` after reviews V-048, V-049, V-050. Targeted CPU tests: `474 passed, 51 warnings` at `f68175c`, then `64 passed` for `tests/test_localization_report.py` on the last wording fix (coordinator's, no fresh review).
- **B2:** review V-051 says merge after small fixes and found no training bug. The last fix round is done: `tests/models/test_temporal_oracle.py` gave `19 passed, 3 deselected in 308.90s` with `-m "not seed_study"` and `3 passed, 19 deselected in 310.70s` with `-m seed_study`. The marker, the check-script change and the `CLAUDE.md` command are all in.
- **Measured, CPU, tiny backbone, direction task:** `cnn_rnn` 800 clips 9/10, 240 clips 2/10; `temporal_transformer` 1500 clips 5/10; `late_fusion` half speed 7/10, default speed 0/10. The two-frame limit at default speed is 0.66347; V-046's 0.548 was an error.
- **Decision (user):** Phase 3's done-when is reworded in PLAN.md to what the evidence supports.
- **Not run:** none of B2 or B3 has run on the GPU, real video, pretrained weights or df-analyze.
- **Open for later:**
  - P-105: pre-cut clips such as UCF101's get no shared-source warning.
  - Phase 4: a group-aware check before any per-segment uncertainty claim.
- **R1 and R2 stopped at the user's request, so the GPU checks come first.** Their uncommitted files stay in `.claude/wt/R1` and `.claude/wt/R2`.
- **`check.ps1` on the GPU (at cbd57bb): `CHECK FAIL`.** `pytest: 4 failed, 1650 passed, 4 skipped, 3 deselected, 96 warnings in 941.91s`; pip check and all three smoke runs ok. It ran in its own window, because Claude Code's memory safeguard stopped it from the session.
  - **Fixed, real bug:** `run_motion_benchmark.py` didn't set cuDNN determinism, so one seed gave 0.991, 0.942, 0.989 on the GPU. With it set, all three runs matched exactly.
  - **Fixed, real bug:** non-video loaders spawned 8 workers on Windows, and each loads torch's CUDA DLLs. That ran out of page file (WinError 1455) in 2 of 7 runs of one test. `options.NUM_WORKERS` now follows the Windows cap of 4: 4 of 4 passes, so this is improved, not proven gone. `CLAUDE.md`'s `NUM_WORKERS = min(8, cpu/2)` is now out of date.
  - **Open, test assumptions, pass on CPU:** `test_features_equal_a_direct_backbone_forward` compares a GPU result with a CPU one and differs by up to 6e-5. `test_validation_loss_is_computed_in_float32` feeds CPU tensors to a model on the GPU. Waiting on the user, per CLAUDE.md rule 3.
- **The user chose to skip the two test fixes and move on, so `check.ps1` stays `CHECK FAIL`.** It has not been re-run since `5bae1be`. GPU jobs run in their own windows with logs in `.artifacts/gate/` (not committed). Claude Code's memory safeguard stopped only one watcher during these.
- **B1 gate, GPU (generated clips, not real video; timing only):** `bench_video_io.py --clips 200 --feature_clips 100 --feature_model r3d_18 --feature_pretrained --feature_gpu`, 1.8 min. One epoch of a frozen r3d_18 + head at 112 px, batch 4: from pixels 2.67 s; `--frame_cache` 2nd epoch 2.79 s (1st 2.83 s, no gain); cached features 0.78 s (`linear_probe`), 1.08-1.18 s (`attentive_probe`). Extraction 80-82 ms/video.
- **T1 gate, GPU, real video (UCF101 10-class subset, `group` column, 66 grouped test clips, r3d_18, 16 frames, seed 42, one seed):** all 11 runs exit 0, 15.7 min in total. Single-frame-separable classes, so this shows transfer, not motion. No published baseline is comparable.

  | Run | Test accuracy (95% group CI) | Peak GPU reserved |
  |---|---|---|
  | frozen, float32 | 0.955 (0.881-1.000) | 932 MB |
  | frozen, bf16 | 0.955 (0.881-1.000) | 650 MB |
  | frozen, bf16, cosine, accum 2, select_best, 6 epochs | 0.970 (0.903-1.000) | 650 MB |
  | fine-tune, batch 4, `--freeze_bn off` | 1.000 (zero-width) | 1,076 MB |
  | fine-tune, batch 4, `--freeze_bn auto` | 0.939 (0.869-1.000) | 1,066 MB |
  | LoRA rank 8 / 16 / 32, batch 4 | 0.970 / 1.000 / 0.955 | 1,032 / 1,054 / 1,136 MB |

  - Frozen 5 epochs took about 1.5 min, against 11 min on CPU (2026-10-21, a different code version).
  - Resume is exact on the GPU: stopping at epoch 2 and resuming to 5 gave identical weights (max diff 0.0) and identical logs. Two bf16 runs with one seed also gave the same metrics.
  - `--freeze_bn` and the LoRA rank stay undecided: one seed and 66 clips can't separate them.
  - Not covered here: fp16, resume with DataLoader workers (all runs used `--num_workers 0`), and early stopping actually stopping (`c` ran all 6 epochs).
- **B2 gate, GPU, in progress** (`run_motion_benchmark.py --task direction`, 64 px, 700 train, 400 test, 5 epochs, batch 8, pretrained, single dataset, chance 0.5). Test accuracy:

  | Model | Seed 0 | Seed 1 | Seed 2 |
  |---|---|---|---|
  | `single_frame` | 0.498 | 0.500 | 0.498 |
  | `mean` | 0.508 | 0.500 | 0.493 |
  | `attention` | 0.510 | 0.500 | 0.500 |
  | `late_fusion` (limit about 0.663) | 0.500 | 0.493 | 0.500 |
  | `early_fusion` | 1.000 | 1.000 | 0.985 |
  | `tsm` | 1.000 | 1.000 | 1.000 |
  | `slow_fusion:r2plus1d_18` | 1.000 | 1.000 | 1.000 |
  | `cnn_rnn` | 0.648 | 1.000 | 0.503 |
  | `temporal_transformer` | 0.500 | 0.500 | 0.505 |
  | `two_stream` farneback | 0.963 | 1.000 | 0.993 |
  | `two_stream` mv | 1.000 | 0.998 | 0.990 |
  | `flow_probe` | 0.993 | 0.993 | 0.993 |

  84 min in total. Single dataset, so the seeds vary only the model and training order. `temporal_transformer` stayed near 50% on training as well as validation on all three seeds, so it did not memorise, unlike its tiny-backbone failures. This run is 5 epochs with the runner's default settings, not B2's `tt_dim=64`, lr 3e-4 oracle settings, and the cause is not investigated: possibly a settings problem or a bug. Phase 3's reworded target says each architecture can learn direction, and on real ResNets `temporal_transformer` has not shown that here. `late_fusion` has not either, as expected at this speed. Results: `.artifacts/gate/b2/direction/`. `scene_bias` and `periodic` are running.
- **User decision at 21:09:** stop the GPU checks. P-105 to P-109 and R3 move to a later extension (PLAN.md 4b). B2's `scene_bias` seed 0 finished in 24.6 min and seed 1 was stopped. Still open: `scene_bias` seeds 1 and 2, `periodic`, UCF101 through the pipeline and `infer.py`, B3's 5-fold report, and the `temporal_transformer` investigation.
- **B2 `scene_bias`, seed 0, GPU (biased validation / unbiased test):** `single_frame` 0.935 / 0.505; `mean` 0.910 / 0.493; `attention` 0.940 / 0.500; `late_fusion` 0.815 / 0.483; `cnn_rnn` 0.940 / 0.500; `temporal_transformer` 0.940 / 0.495; `early_fusion`, `tsm`, `slow_fusion:r2plus1d_18` 1.000 / 1.000; `two_stream` farneback 0.990 / 0.985; `two_stream` mv 0.830 / 0.905; `flow_probe` 0.995 / 0.995.
- **V1a gate, `smoke_track.py` on real UCF101 clips (320x240), GPU:**
  - **Bug fixed (0879567):** written track boxes ran up to 2.3 px past the frame edge, from the Kalman state. Re-run on the baby clip: `SMOKE TRACK PASS`.
  - **"baby" on `v_BabyCrawling_g03_c01`, one baby in the clip:** every frame: Grounding DINO 2 tracks (one lasts a single frame, so it is a false extra), OWLv2 1. Every 5th frame: 1 and 1. The main track covers the whole clip. 215 frames in about 55 s with each detector when run at every frame.
  - **"person" on `v_BandMarching_g05_c02`, a crowd of small people:** 0 tracks from both detectors at both steps, though the frame is full of people. It is a real miss, not a crash: Grounding DINO's top score on frame 40 was 0.178 for "person.", under its 0.2 box threshold, while "people." gave 0.738 and "a person." 0.344. So a count of 0 does not mean none are there, results depend heavily on the wording, and small objects at this resolution are a weak spot.
- **V1a gate, page 07's own code on a real run and real clips, GPU (no clicks; screen control is not available here):** the same functions its Predict, Track and Stop buttons call, in a torch-free process. 6.4 min in total.
  - **Predict:** `h_finetune_bn_off` gave BabyCrawling 98.6% and BandMarching 94.9%, both correct, with "3 of 33 validation clips were wrong, an estimate of 11.4% after smoothing", stated as the run-wide rate. The October demo run has no `error_model.json`: "not available", no number.
  - **Track "baby":** step 1: Grounding DINO 2 (the false one-frame extra), OWLv2 1. Step 5: 1 and 1. The overlay frame checked by eye boxes the baby correctly.
  - **Track on the band clip, Grounding DINO:** "person" gave 0 with a message giving the best score (0.31), the 0.45 needed to start a track, and what to try. "people" gave 1 track, the whole crowd as one object, which is a wrong count for a crowd.
  - **Stop:** cancelled in 0.9 s, and GPU memory fell back from 7,501 MB to 1,456 MB.
  - **Memory:** tracking with Grounding DINO peaked at 7,501 of 8,151 MB in use on the GPU, including about 1,500 MB already in use. This is a total from `nvidia-smi`, and may include cached memory. It leaves little room for another GPU job.
  - **Still not checked:** the buttons by hand, and how the page displays these results. Not a pass of the V1a step until someone clicks through it.
- **Next:** R1 and R2 reviews (running), then Wave 3.

### 2026-10-31 — Wave 2: B1, T1, V1a merged
- **Merged into `video-analysis`:**
  - B1, feature cache and snippet features (05b401b). Reviews V-039, V-042, V-043, V-047.
  - T1, training basics and LoRA coverage (8314d32). Reviews V-040, V-044.
  - V1a, Analyse Video workspace (94c5ecd). Reviews V-041, V-045.
- **Full suite (CPU, `CUDA_VISIBLE_DEVICES=-1`, at 94c5ecd):** `1191 passed, 5 skipped, 69 warnings in 368.29s`. `check.ps1` on the GPU has not been re-run.
- **Review limits, stated plainly:**
  - B1's last round (f333f25..555a38a) was checked by the coordinator with the critic's repro scripts, not by a fresh ml-critic.
  - V1a's last fix (6a6d41e, the V-045 count wording) was made and tested by the coordinator, with no fresh review.
- **Not done:** none of B1, T1 or V1a has run on the GPU or on real video. Per the PLAN 4b rule, they don't count as done until the real-data GPU gate passes. Commands are in `docs/video-plan/reports/`.
- **Still open from the reviews:**
  - B1: `--motion_gain` default undecided until real val clips; same size and mtime on a rewritten file goes undetected; `load_snippet_features` accepts an array with no sidecar.
  - T1: a budget overrides an explicit `--lora_rank`, so don't combine them at the r = 8/16/32 gate; changing the selection metric on resume keeps the old best metric.
  - V1a: zero-shot false-box counts with `fuse_score=False` need checking on real video.
- **Running:** B2 fix round (V-046), B3 builder.
- **Decision for the user:** `late_fusion` provably can't meet PLAN's direction done-when at default speed (Bayes limit 0.548 from two frames). See V-046.
- **Next:** merge B2 and B3 after review, then R1, R2, R3, then the Wave 2 GPU gates with the user present.

### 2026-10-30 — Wave 2 start: GPU checks
- **`scripts\check.ps1` on the GPU: `CHECK FAIL`.**
  - pip check ok. All three smoke runs ok. Total 353 s. The old r3d_18 smoke timeout (G15) is gone.
  - pytest: `1 failed, 505 passed, 3 skipped, 30 warnings in 295.66s`.
  - The failure is `tests/test_video_followups.py::test_trainer_fits_the_error_model_on_multi_clip_probabilities`. After `Trainer` moves the model to `cuda:0`, the test's hand refit passes CPU tensors. It fails with the GPU visible and passes on CPU.
  - The user approved the one-line fix, `model.cpu().eval()`. `tests/test_video_followups.py`: `9 passed` with the GPU visible and `9 passed` on CPU. `check.ps1` has not been re-run since; it is due at the end of the wave.
- **`scripts\bench_video_io.py --clips 200 --nvdec`** (generated clips: H.264, 112 px, 64 frames):
  - 8 frames, uniform: PyAV 60.9 clips/s, OpenCV 54.0. Real 1080p files: PyAV 1.8 clips/s, OpenCV 1.0.
  - PyAV + NVDEC, all 64 frames: 10.4 clips/s. There is no software all-frames row, so this says nothing about an NVDEC speed-up. TorchCodec is not installed.
  - Dataset `__getitem__`: old 42.3 clips/s, new bytes 77.5, new paths 75.3. Worker pickle: 2.92 MB falls to 4.6 kB.
  - Motion field, 8 frames: codec 12.8 ms/clip vs Farneback 21.7 (H.264); 11.1 vs 17.8 (MPEG-4).
  - DataLoader epoch, 200 clips: workers 0 / 1 / 2 = 2.68 / 4.69 / 4.28 s. On clips this cheap, worker start-up costs more than it saves.
- **`scripts\smoke_track.py --save_video` on the GPU: PASS.** Grounding DINO and OWLv2 each gave `red square=1, blue circle=1`, each track covering all 48 frames, in 28.5 s and 9.6 s excluding model loading and download. This is a two-shape synthetic clip, so it proves plumbing, not real-video accuracy.
- **Fine-tuned single-frame adversary (V-022, V-038): at chance on all three tasks**, with and without codec artefacts. The table, the controls and the limits are in `docs/video-plan/benchmark.md`, and `scripts/check_single_frame_adversary.py` reproduces them.
- **Bugs found, assigned to B2:**
  - OpenCV silently writes H.264 for FourCCs it can't encode (seen with FFV1, ffv1, `png `, HFYU, I420, RGBA), while `dataset.json` records the requested codec.
  - The runner's `flow_probe` uses mean flow on classification tasks, so it can't separate periodic (0.5175), although the flow summary reaches 1.000 in the honesty table.
- **Decisions:**
  - `trackers` is not installed: the user's go-ahead covered the GPU checks and Wave 2, and this was a separate OK.
  - New user request: an Analyse Video workspace (PLAN.md 4b, steps V1a–V1e). V1a joins Wave 2.
- **Next:** builders T1, B1, B2, then V1a, B3, R1, R2, P-105, P-106, P-108 as slots free; the Scout also picks V1d's model.

### 2026-10-30 — Wave 1 CLOSED
- **Merged:**
  - A3, Phase 2 benchmarks (5bf2588). Review chain V-020..V-024 → V-036 → **V-038 merge** (ml-critic).
  - A seeded val split for every task; the runner never validates on test.
  - A speed regression task in the runner.
  - The flip guard is wired into `deeptune.py --video_flip`.
  - The single-frame claim is scoped to linear pixel probes plus a frozen resnet18, on direction only.
- **Full suite (CPU, `CUDA_VISIBLE_DEVICES=-1`, cpu 0):** `506 passed, 3 skipped, 26 warnings in 281.41s`. The 3 skips are TorchCodec (not installed).
- **Codex note (while Claude was rate-limited):**
  - It merged video-analysis 5cb6293 into A3 (3be531b, default merge message, kept as is).
  - It made no code edits and ran no tests.
  - It flagged two stale doc notes, which A3 then fixed.
  - It worked without creating `.claude/board/LOCK`. Next time, create it first.
- **Reviewer:**
  - The Critic is now the `ml-critic` agent type (`.claude/agents/ml-critic.md`, ad1836c), an ML/data-analysis methodologist.
  - It loads in new sessions. In this session it ran as a general-purpose agent reading that file.
- **Pending the user's OK:**
  - GPU: `check.ps1`; `bench_video_io.py --clips 200 --nvdec`; `smoke_track.py` (~1.3 GB); the fine-tuned `single_frame:resnet18` on direction/speed/periodic at 700/400, plus a lossless-codec variant (V-022/V-038).
  - `pip install trackers`.
  - Wave 2.
- **Nice-to-haves left:**
  - `benchmarks/guards.py:38` should read `utf-8-sig`.
  - `benchmark.md:121` "val.csv for scene_bias" is stale.
  - The untrimmed benchmark was not re-audited in the last pass.
  - GUI pooling radio lacks `single_frame` (Phase 7).
  - `infer.py allow_abbrev`.
- **Next:** Wave 2 in a FRESH session, with a fresh ml-critic and Scout: T1, B1, B2, B3 (+P-105, P-106, P-108), R1, R2. Start with the GPU checks above, with the user present.

### 2026-10-30 — Wave 1 with parallel agents (coordinator session; HANDOFF)
- **How the work runs:**
  - Approved plan: Phases 1–8 via parallel agents in waves, coordinator-merged. PLAN.md §4b holds the scope extension, licences, the 13 accepted board proposals and the professional-quality rules.
  - Agents work in manual git worktrees `.claude/wt/<task>`; the Agent tool's worktree isolation is broken here.
  - They communicate through the board `.claude/board/` (README rules, FINDINGS, PROPOSALS, REVIEWS, INTERFACES, **QUEUE.md = exact next steps**).
  - A read-only Critic reviews every branch before merge; a Research Scout researches and proposes.
- **Merged into `video-analysis`:**
  - A1, trainable-parameter budget + fixes (1e2e0ff)
  - A4, per-prediction `expected_error` + fixes + multi-clip test (d517c15, 158ecdb); the A2×A4 trainer conflict was resolved per Critic V-033
  - S1a, zero-shot detect + ByteTrack + fixes (71dce3d); the ID-swap bug V-001 is fixed
  - A2, Phase 1 data layer (966a5c1)
  - desktop visual upgrade from a parallel session, committed as-is at the user's request (680db84)
- **Tests:** targeted CPU run after the A2+A4 merges: `414 passed, 3 skipped, 36 deselected, 24 warnings in 95.55s`. The **full suite has not been run since these merges**; it is due after A3.
- **Open:**
  - A3's merge-in + fix round is running in `.claude/wt/A3`: V-021 runner validated on test, V-022 single-frame claim scope, V-023 GUI single_frame, plus the flip guard.
  - After A3: Critic re-check → merge → full suite → close Wave 1.
- **Decisions:**
  - The plan's licences were corrected: V-JEPA 2 is MIT; VideoMAE/TimeSformer are CC-BY-NC.
  - The user accepted all 13 proposals.
  - Wave 2 starts in a FRESH session with a fresh Critic and Scout.
- **Needs the user's OK:**
  - GPU runs: `check.ps1`, `bench_video_io.py --clips 200 --nvdec`, `smoke_track.py` (~1.3 GB download), and the fine-tuned single_frame check (V-022)
  - `pip install trackers` (P-110)
  - starting Wave 2
- **Handoff to another tool (e.g. Codex):**
  - Read AGENTS.md and create `.claude/board/LOCK` while working, so Claude's hourly check-ins only report.
  - Continue the first open item in QUEUE.md.
- **Known nits for later:**
  - CLAUDE.md's "mimic CI" tip should say `'-1'`, not `''`.
  - LoRA coverage on 3D models (T1).
  - `infer.py allow_abbrev` (V-033 nit).
  - Re-tune the S1a tracker thresholds with S4.

### 2026-10-30 ? Desktop visual upgrade (UI only)
- Verified checkout: `video-analysis`, one commit ahead of origin; tracked tree was clean.
  Left Claude's untracked board and worktrees untouched; no branch switch.
- Added an ink/mint/violet studio visual system, overview artwork and workflow links,
  larger typography, refined cards/controls, breadcrumb navigation, striped metric
  tables, and distinct training/validation curves. Backend/analysis code unchanged.
- Added compact form layouts, enabled native keyboard navigation/focus, fit video
  display to its panel without stretching, and replaced activity spinners with static
  text. No decorative motion; existing playback controls remain available.
- Verification: full CPU suite `258 passed, 13 warnings in 168.36s`; final desktop
  UI rerun after visual refinements `14 passed in 1.64s`; pip check clean.
  Two new regression tests cover responsive navigation/config preservation and
  portrait/landscape preview fitting without changing texture pixels.
- Visually inspected native framebuffer captures at 1320x920 and 900x720, all six
  pages, a repository MP4 preview, and fixture-populated metrics/curves. Review
  captures and harness are local ignored artifacts in `.artifacts/`.
- Limitations: computer-use helper failed (`CreateProcessWithLogonW failed: 2`),
  so inspection used native framebuffer export rather than automated mouse/keyboard
  interaction. Screen-reader support remains limited by Dear PyGui. No new GPU
  training experiment was needed for this visual-only change. This does not mark
  the planned Phase 7 analysis features as implemented.
- Next: review the uncommitted UI diff alongside the existing agent work.

<!-- Template:
### YYYY-MM-DD — Phase N, step X
- Done:
- Tests: `NNN passed, N warnings` (full suite), runtime Ns
- Assurance checks: (each one + result)
- Decisions:
- Problems / open questions:
- Next step:
-->

### 2026-10-21 — Project update 1: slides + demo (no phase work, no code changes)
- Done: a 16-slide deck for the first in-class update (GUI-branch video work, ML theory,
  results, roadmap). A class demo kit outside the repo in `C:\Users\Ahmed Magdy\deeptune-demo`:
  the UCF101 10-class subset (HF `sayakpaul/ucf101-subset`, 405 clips), a manifest with
  a `group` column (class + UCF101 gNN), 75 test clips whose class-groups never appear in training,
  and an R3D-18 run trained through `deeptune.py` (frozen backbone, 16 frames, 5 epochs,
  lr 1e-3, `--grouper group`, CPU, 11 min).
- Results (first real-data run): grouped split with 155/21/44 groups and no group shared between splits;
  holdout 66/66; unseen test via the Test-page code path (`desktop.config` + `runner.start_infer_run`)
  74/75 = 98.7 %, the only error ApplyEyeMakeup→ApplyLipstick at 0.57 confidence. These classes
  overlap Kinetics-400 and are mostly single-frame separable, so this shows transfer works,
  not motion understanding (Phase 2's job).
- Problem found: in Windows PowerShell 5.1, `$env:CUDA_VISIBLE_DEVICES=''` removes the
  variable, so the GPU stays visible and 8 workers spawn (G15). Use `'-1'`. CLAUDE.md's
  "mimic CI" tip still shows `''`; it should be corrected.
- Tests: suite not run (no code touched).
- Next step: Phase 1, unchanged.

### 2026-10-10 — Phase 0, foundation & GPU setup
- Done (branch `video-analysis` from `GUI`):
  - CI: triggers on master/GUI/development/video-analysis + manual; checkout,
    setup-python and upload-artifact v7; pip cache; CPU-only torch 2.14.0 /
    torchvision 0.29.0 from the PyTorch CPU index; runs `scripts/check.sh --quick`;
    uploads the JUnit report; wheel build kept. README badge → ply971/deeptune.
    YAML validated with PyYAML; `act` not installed, so the first real run is on push.
  - Hygiene: 424 tracked build files untracked (`docs/_build` 92, `docs/build` 117,
    `docs/source/_build` 118, `docs/source/build` 97) and ignored. Nothing linked
    into them; demo videos stay in `docs/source/_static`.
  - `scripts/check.ps1` / `scripts/check.sh`: pip check → pytest (JUnit in
    `.artifacts/check/`, `--durations=10`) → two video smoke runs (`-Quick` /
    `--quick` skips) → one `CHECK PASS|FAIL | …` line with the device.
  - GPU: `.venv` switched from `torch 2.14.0+cpu` to `torch 2.14.0+cu130`,
    `torchvision 0.29.0+cu130` (index `https://download.pytorch.org/whl/cu130`).
    Verified: RTX 5060 Laptop GPU, capability (12, 0), arch list includes sm_120,
    8.5 GB, cuDNN 92400; 2048² matmul max |GPU−CPU| = 5.0e-4; conv fwd/bwd finite;
    r3d_18 fwd/bwd; bf16 autocast. `pip check` clean.
- Baseline (Python 3.12.10, torch 2.14.0, torchvision 0.29.0):
  - CPU (`+cpu`, unchanged code): **173 passed, 3 warnings in 433.51s**.
    Slowest: test_vit peft init 115.7s, vit init 52.9s, vit peft reg 52.7s,
    vit reg 27.9s, densenet peft 14.8s, vgg peft 12.8s, vgg peft reg 8.8s,
    swin 8.2s, swin peft 7.6s, densenet peft reg 7.2s.
  - CPU `check.ps1` (full): CHECK PASS — pytest 173 passed; smoke resnet18
    --manifest ok; smoke r3d_18 ok. That pytest took 5049s because one test
    (`test_vit_peft_model_regression_mode_initialization`, dummy ViT, no network)
    took 4766s vs ~53s elsewhere — most likely the laptop slept; not reproduced.
  - GPU (`+cu130`) `check.ps1` (full): **CHECK FAIL** — pip check ok; pytest
    **173 passed, 3 warnings in 431.14s**; smoke resnet18 --manifest ok;
    **smoke r3d_18 FAIL: TimeoutError, exceeded 600 s** during the eval/embed
    step. The log shows a new DataLoader worker process starting every 15–40 s
    (G15: `options.py` sets `NUM_WORKERS = min(8, cpu/2)` on CUDA; Windows spawns
    each worker and re-imports torch). Same run passes on CPU (`NUM_WORKERS = 0`).
    Not fixed here (Phase 0 = no behaviour changes) → Phase 1 task.
    Confirmed slow, not broken: re-run with `--timeout 1800` (and
    `PYTHONIOENCODING=utf-8`) → **passed in 811 s with 33 worker-process
    start-ups** (train, eval, embed all ok). Side note: running a smoke script
    with stdout redirected to a file under the default cp1252 encoding crashes on
    tqdm's block characters (`UnicodeEncodeError`) — set `PYTHONIOENCODING=utf-8`
    when capturing output (check.ps1 itself was unaffected).
  - 3 warnings (both platforms): `torch.jit.script` deprecation; pandas
    mixed-type column names in `test_desktop_config` (×2).
- Audit (G1–G15 in PLAN.md): G1–G7, G9–G13 confirmed. G8 corrected: grouped
  holdout splits already exist (`--grouper` → StratifiedGroupKFold, also for
  video). New: G14 fixed `Resize((224,224))` (no augmentation, aspect squashed),
  G15 worker processes on Windows + CUDA. Device is picked automatically
  (`options.get_device()`: cuda → mps → cpu). `desktop.app` imports without torch
  (checked: `'torch' in sys.modules` is False).
- Decisions: CI uses CPU wheels; local dev uses cu130; no library code touched.
- Problems / open: GPU r3d_18 smoke timeout (above). First CI run happens on the
  first push (not pushed yet — needs approval).
- Next step: Phase 1 in a fresh session; include fixing G15 (worker defaults on
  Windows / persistent workers) so `check.ps1` passes on the GPU.

### 2026-10-10 — Planning (before Phase 0)
- Done: plan copied into the repo (commit `docs: add video analysis term plan`),
  then revised: context = background → CtxPL-based weakly-supervised action
  localisation (Phase 4B), untrimmed + scene-bias benchmarks (Phase 2),
  background-reliance score (Phase 6), localisation mAP (Phase 5), CtxPL
  reproduction on THUMOS'14 (Phase 8); speed work: codec motion vectors,
  motion-guided sampling, snippet feature cache (Phase 1), TSM + two-stream
  motion source + ToMe stretch (Phase 3); GPU setup (Phase 0).
- Read in full: `moayadeldin/CtxPL` (main, train, test, model, options,
  edl_loss, base, wsad_dataset, proposal_methods, utils, eval/*).
- Environment found: RTX 5060 Laptop GPU (8 GB, driver CUDA 13.4) present, but
  `.venv` has `torch 2.14.0+cpu` → GPU unused (G11).
- Decisions: re-implement CtxPL in DeepTune rather than vendor it; THUMOS'14
  downloads wait until Phase 8.
- Next step: Phase 0 in a fresh session.

## Notes for later phases
- **CtxPL public-code issues** (from reading, not yet run) — handle in Phase 4B,
  document in `docs/video-plan/ctxpl-port.md`:
  - `self.use_snip_unc` read in `_contextual_video_repr` but never defined →
    AttributeError on the first confident video during training (initial video
    uncertainty ≈ 20/42 ≈ 0.48 < threshold 0.7, so immediately). `test.py`
    doesn't build prototypes, so evaluating the released checkpoint works.
  - Global `torch.set_default_tensor_type('torch.cuda.FloatTensor')` and
    `.cuda()` calls; `curve.repeat(10, 1)` hard-codes batch 10; `Contrastive`
    assumes the first 6 batch items are 3 same-class pairs (from
    `num_similar=3, similar_size=2`).
  - New prototypes use the context-weighted vector; momentum updates and
    replacements use the plain mean.
  - `proposal_methods.py`: undefined `logger`; `eval/detectionMAP.py`
    `AntgetLocMAP` has `pdb.set_trace()`; duplicate `forward` in `CTXPL`.
  - Snippet = 16 frames @ 25 fps (factor 25/16 hard-coded in eval/proposals);
    features = 1024 RGB + 1024 flow (CO2-Net I3D); evaluation filters
    THUMOS "ambiguous" segments; THUMOS split = validation→train, test→test.
  - Prototypes are Python dicts, not in `state_dict`.
- CtxPL's pinned `torch==2.6.0+cu118` probably lacks Blackwell (sm_120)
  kernels → the original `test.py` may need a newer torch in Phase 8.
- Schedule: plan assumes ~15 weeks of work; check against the course deadline
  and cut **(stretch)** items first (ToMe, RAFT, DeepTune-feature THUMOS run).

## Known limitations
-

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

### 2026-11-01 - Cat tracking cancelled by user
- User said nvm; stopped task-owned train_cat_detector.py processes and removed task lock. Existing code and downloaded artifacts preserved.
- No trained checkpoint or completed video tracking was produced in this session. Main checkout contains prepared detector/page07 code; full prior CPU suite: 1662 passed, 5 skipped, 92 warnings in 2017.52s; follow-ups: 152 passed in 66.91s and 2 passed in 1.08s.
- Next: resume only if requested; inspect data preparation before starting training.

### 2026-11-01 - Cat experiment resumed at user's request (verification running)
- Five bounded GPU epochs completed on 897 training / 239 validation / 300 test photos, seed 42; epoch 4 selected by validation AP50. The original test cohort is unchanged. Four explicit missing/foreground-free annotation exclusions are recorded in .artifacts/cat-tracking/run01/report.json and splits.json.
- Test AP50 baseline 0.991724606 versus trained 0.993064677; precision@0.5 0.865454545 versus 0.963562753; recall@0.5 0.991666667 for both (238/240 cats detected, 37 versus 9 false positives). Single seed and photo subset; no significance or video accuracy claim.
- Two real cat videos processed sequentially with each checkpoint; trained overlays saved in .artifacts/cat-tracking/trained-video01 and trained-video02. Track IDs fragment (3 each), so these are not unique-animal counts. Contact sheets inspected; attribution/hash manifests saved. Main page07 discovers installed saved_models/cat-detector/best.pt without importing torch.
- Code eb18fef on video/cat-tracking; reviews V-054..V-056 accepted code fix, saved splits/checkpoints and experiment report. Preflight/mirror tests: 3 passed in 0.17s; earlier combined targeted run: 9 passed in 11.86s. Pip check clean.
- First full CPU attempt stopped with disk-full write failures and no final pytest summary. Approved cleanup removed only about 10.8 GiB of generated old/failed pytest temp folders, preserving logs, data, checkpoints and videos. Full CPU rerun active in tool session 33964; .claude/wt/cat-tracking/.artifacts/resume-check/full-pytest.log and full-results.xml. Do not claim completion yet or launch a duplicate full suite.
- Next: finish full suite, record actual summary, commit report/handoff and release LOCK. No merge or push performed; main's prepared cat integration edits preserved.


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

### 2026-11-02 - R1 and R2 merged (coordinator)
- Resumed after the R1 and R2 fix-round agents died. R1's V-052 round was complete but uncommitted. Targeted: 184 passed, committed as 50fa8d1. The ml-critic re-check found one regression: the GUI always sent `--pooling mean`, which the new refusal blocks on vjepa2/videomae. Fixed in f1b8705, with a test for the untested infer.py refusal. **R1 merged 4e3e1a6.**
- R2: the builder finished V-053 (45283db, f313431). The re-check found two remaining false-guarantee paths (CSV group ids read as ints; ambiguous numeric class names) and one surviving mutant. Second round 2e03a0d..6c4ddb6 fixed them, merged R1 into R2 and resolved item 6: ensembles never abstain, the standalone fit has the licence gate, and there is one mapping hash. **R2 merged 5ac8f79.** PLAN.md now names the default Learn then Test, not conformal.
- Tests (CPU): merged-tree targeted set 432 passed, 3 deselected, 16 warnings in 110.88s; seed_study on the R2 files 3 passed. **Full suite not yet run on the merged tree.** The last full run is 1668 passed, 5 skipped, from before R1 and R2.
- Open problem: 11 page-07 tests in tests/test_desktop_analyse.py fail on this machine whenever `saved_models/cat-detector/best.pt` is installed, because analyse_page.py:741-742 preselects the cat detector. This predates R1 and R2: reproduced on a clean checkout of 5940e5f with the file linked in (11 failed, 136 passed); without the file, 225 passed. The user must choose (a) test isolation plus a test of the preselection, or (b) a zero-shot default with a hint.
- Next: full CPU suite, the page-07 fix, then Wave 3.
- Page 07 (user's choice b, c794f89): the page always starts on the general detector, and an installed cat detector gets a hint. `test_installed_cat_is_selected_in_page07` was replaced by `test_installed_cat_is_offered_not_selected_in_page07` plus a no-hint test. Desktop + spatial tests with the real checkpoint installed: 240 passed (previously 11 failed).
- The full CPU suite was started on c794f89 and stopped by Claude Code for low system memory after about 65 tests (no failures up to that point; no summary). It was not restarted. **The full suite on the merged tree is still outstanding**; rerun it when the user asks, ideally with memory-heavy apps closed or with CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1.

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

### 2026-11-03 — R3b merged (coordinator)
- R3b (smart default, random, bayes via optional Optuna, augmentation axis, regret study) built on `video/R3b-smart-search`; ml-critic V-057: merge after fixes (CI without Optuna, stage 3 missed train-mode x rate interaction, overstated regret claims, false frozen-augmentation reason, `--tune_epochs` final stage, vacuous tests, silently ignored foreign flags).
- Fix round f3d546d, a1be5fb, 0fd063c: joint train-mode x learning-rate group, full-product re-check (`--recheck_full_product`, default 6), two screening views (frozen and trainable), foreign flags refused, Optuna tests skip cleanly and CI installs Optuna.
- Coordinator verification: critic surfaces A and B, smart regret 0 on 4/4 seeds (A was 0.271); no-Optuna run `40 passed, 4 skipped`; tune_final repro 3 final runs.
- Merged `34de59f`. Merged tree: `437 passed, 4 warnings` (desktop, evaluation, ensemble, hub, splits, followups) and `97 passed, 2 warnings` (model selection). e2e on branch: `2 passed`.
- Honest limit: the 30-candidate simulated grid is too small to separate smart from bayes; real comparison is the GPU gate in reports/R3b.md (needs the user).
- A background test run was killed by Claude Code for low memory; reran in the foreground in two batches. Full suite still owed.
- Next: G1 GUI catch-up (worktree `.claude/wt/G1`, branch `video/G1-gui-catchup`).

### 2026-11-03 — G1 GUI catch-up built (builder, branch `video/G1-gui-catchup`)
- Every flag of deeptune.py, infer.py, select_model.py, ensemble.py, extract_features.py, train_feature_head.py and track.py (273) has a GUI control; allow-list empty. Specs in `desktop/cli_options.py`; new pages Model Selection (dry run gates the search; sealed-test score shown as the honest number), Ensemble, Features & Probes; Advanced settings on the Model page; Test options and overrides; More tracking options; sidebar Device (`CUDA_VISIBLE_DEVICES`); reports read from disk on Results/Test.
- `tests/test_gui_coverage.py` drives each flag's control through the GUI's argv builder and the real parser; mutation proof: dropping `--use-peft` or `--save_video` fails it. `tests/test_gui_pages.py` covers the pages. Targeted runs: `98 passed` (new), `315 passed, 2 warnings` (desktop + hub), `9 passed` (followups), `254 passed, 2 skipped` (CLI neighbours). Full suite owed.
- For review: hub backbones offered via `GUI_MODEL_VERSIONS` so `test_hub_backbones.py`'s "GUI list is Phase 7" assertion still holds; its message is now stale. Details in `reports/G1.md`.
- Next: click through the new pages once, a GPU check of a GUI-launched run, ml-critic review, then Wave 3.

### 2026-11-03 — G1 fix round after V-058 (builder)
- Must-fixes done: sealed-test wording from `selection.json` facts; Load recipe file instead of `--config` (allow-listed with reason); Test-page overrides split into test-time and model-defining (confirmation tick, log warning, `overrides.json`, three-way yes/no); coverage test now presses each page's Start handler and parses the captured argv (deeptune.py lines through `deeptune.parse_args`, video flags in a video run); ensemble = exactly the listed runs, labels only from a member; frozen-build launches in model selection. Nice-to-haves done (readers, device records, licences unsaved, strategy-flag sync, resume+validation, Seed tooltip).
- Critic mutants: all 8 caught; my two earlier mutants caught. `test_hub_backbones.py` GUI-list test reworded on its own commit (user decision).
- Runs: `121 passed` (G1 tests), `315 passed, 2 warnings` (desktop + hub), `9 passed` (followups), `53 passed, 1 warning` (model selection). Full suite still owed.
- Disk: C: hit 0 MB free during the round; I deleted only my own pytest temp folders (~800 MB). The coordinator should free space before the full suite.

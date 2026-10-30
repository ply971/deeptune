# PROGRESS — DeepTune Video Analysis

Claude Code: update this file at the end of **every** session. Newest entry on top.
Keep entries short and factual. This is the handoff between sessions.

## Status board
| Phase | Status | Tests (passed/total) | Commit |
|---|---|---|---|
| 0 Foundation & GPU setup | done (1 known GPU smoke timeout → Phase 1) | 173/173 CPU and GPU | `video-analysis` |
| 1 Data layer + MVs + feature cache | tasks 1–6, 9–12 merged (A2); 7–8 feature cache = B1 (Wave 2) | 414 passed, 3 skipped (targeted, CPU) | `966a5c1` |
| 2 Motion / untrimmed / scene-bias benchmarks | merged (A3); fine-tuned single_frame and lossless checks done on the GPU (one seed) | full suite 506 passed, 3 skipped (CPU) | `5bf2588` |
| + Param budget (A1), error estimate (A4), zero-shot detect+track (S1a) | merged with review fixes | see session log | `1e2e0ff`, `158ecdb`, `71dce3d` |
| 3 Temporal models | not started | | |
| 4 Timeline + CtxPL localisation + unsupervised | not started | | |
| 5 Evaluation | not started | | |
| 6 Explainability & background reliance | not started | | |
| 7 GUI | not started | | |
| 8 GPU experiments, CtxPL reproduction, report | not started | | |

## Session log

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

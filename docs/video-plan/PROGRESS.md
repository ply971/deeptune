# PROGRESS — DeepTune Video Analysis

Claude Code: update this file at the end of **every** session. Newest entry on top.
Keep entries short and factual. This is the handoff between sessions.

## Status board
| Phase | Status | Tests (passed/total) | Commit |
|---|---|---|---|
| 0 Foundation & GPU setup | done (1 known GPU smoke timeout → Phase 1) | 173/173 CPU and GPU | `video-analysis` |
| 1 Data layer + MVs + feature cache | not started | | |
| 2 Motion / untrimmed / scene-bias benchmarks | not started | | |
| 3 Temporal models | not started | | |
| 4 Timeline + CtxPL localisation + unsupervised | not started | | |
| 5 Evaluation | not started | | |
| 6 Explainability & background reliance | not started | | |
| 7 GUI | not started | | |
| 8 GPU experiments, CtxPL reproduction, report | not started | | |

## Session log
<!-- Template:
### YYYY-MM-DD — Phase N, step X
- Done:
- Tests: `NNN passed, N warnings` (full suite), runtime Ns
- Assurance checks: (each one + result)
- Decisions:
- Problems / open questions:
- Next step:
-->

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

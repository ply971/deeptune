# PROGRESS — DeepTune Video Analysis

Claude Code: update this file at the end of **every** session. Newest entry on top.
Keep entries short and factual. This is the handoff between sessions.

## Status board
| Phase | Status | Tests (passed/total) | Commit |
|---|---|---|---|
| 0 Foundation & GPU setup | not started | baseline: ? | |
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

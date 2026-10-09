# Phase 3 — Temporal model zoo

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 3 only. Show your plan first (model list, class names, CLI
names, how each plugs into `build_video_model`, and the test list).

## Goal
Implement the "four places to fuse time" from the course slide (Karpathy et
al., CVPR 2014), the two-stream ConvNet (Simonyan & Zisserman, 2014), and
recurrent/attention temporal heads — all as first-class DeepTune video models
that train, evaluate, embed and infer through the existing pipeline.

## Models (CLI `--video_arch`, combined with `--model_version` backbone)
Keep current names working. Add:
1. `single_frame` — done in Phase 2; keep.
2. `late_fusion` — shared 2D backbone on two distant frames (first & last
   sampled), embeddings concatenated, merged by FC layers at the top.
3. `early_fusion` — first conv layer widened to take all T frames stacked on
   channels (`3T` in-channels); initialise from the pretrained 2D weights by
   replicating/averaging the first-layer kernel across time (I3D-style
   inflation logic); rest of the 2D backbone unchanged. Support at least the
   ResNet family; error clearly for unsupported backbones.
4. `slow_fusion` — native 3D models already cover this (r3d_18, r2plus1d_18,
   mvit, swin3d). Map the name to a documented default (r2plus1d_18) and
   document the mapping. No new network needed unless trivial.
5. `frame_pool` (existing) — **fix G5**: attention pooling must weigh the
   per-frame **embeddings**, then classify the pooled embedding; mean pooling
   also pools embeddings. Keep the old logits-pooling behaviour loadable for
   old checkpoints (version the checkpoint or detect by keys) — old saved
   runs must still infer.
6. `cnn_rnn` — 2D backbone per frame → LSTM or GRU (`--rnn_type`,
   `--rnn_hidden`, `--rnn_layers`, `--bidirectional`) → head on last/mean
   hidden state.
7. `temporal_transformer` — 2D backbone per frame → learned positional
   encoding + small TransformerEncoder (`--tt_layers`, `--tt_heads`) → CLS
   token head.
8. `two_stream` — spatial stream (2D backbone on an RGB frame) + temporal
   stream (2D CNN on a stack of `L` optical-flow fields, 2L channels,
   first-layer inflation as in early fusion) → late fusion (average of
   softmax scores, or learned FC; `--fusion avg|fc`).
   Optical flow: `datasets/optical_flow.py` with Farneback (OpenCV, default,
   no weights) and optional torchvision RAFT (`raft_small`) behind a flag;
   flow is computed once and cached to disk keyed by video hash + params;
   flow normalisation documented.

All new models: `forward(x, extract_embed=False)` with x `[B,T,C,H,W]`;
`extract_embed=True` returns the clip embedding the head sees.
PEFT/LoRA where the backbone supports it. Freezing options respected.
Register every model in `utils.build_video_model` and record every new
option with `save_args` so `infer.py` and the GUI Test page can rebuild it.

## Tests
- Unit: output shapes for cls/reg, `extract_embed` shapes, frozen params
  actually frozen, PEFT param count small, checkpoint save→load→same output.
- Early-fusion inflation: with identical frames, output equals the 2D
  backbone's output on that frame (up to tolerance) — proves inflation math.
- Optical flow: on a synthetic translating texture, mean flow ≈ known shift
  (sign and magnitude within tolerance); cache hit returns identical arrays.
- **Oracle tests (the key assurance):** on the Phase 2 `direction` task at CI
  size, with fixed seeds and a small budget: `single_frame` and mean
  `frame_pool` stay near chance; `two_stream`, `cnn_rnn`,
  `temporal_transformer`, `early_fusion` reach clearly above chance
  (choose thresholds with margin, e.g. ≥ 80%). Use small/dummy backbones
  (`weights=None` tiny CNN) so CI stays fast; mark slower variants
  `@pytest.mark.slow`.
- Backward compatibility: a checkpoint from the old frame-pool attention
  model still loads and predicts.

## Deliverables
- Updated `scripts/run_motion_benchmark.py` covering all models; results table
  in PROGRESS.md and `docs/video-plan/benchmark.md`.
- README "Models DeepTune Supports" table extended with the video models and
  a one-line explanation of each fusion strategy.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1` and paste the summary, paste the
benchmark table, show `git diff --stat tests/`. Do not lower oracle
thresholds to get green — if a model can't beat chance, investigate and
report. If anything fails twice, stop and report.

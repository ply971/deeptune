# Phase 3 — Temporal model zoo

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 3 only. Show your plan first (model list, class names, CLI
names, how each plugs into `build_video_model`, and the test list).

## Goal
Implement the "four places to fuse time" from the course slide (Karpathy et
al., CVPR 2014), the two-stream ConvNet (Simonyan & Zisserman, 2014), the
Temporal Shift Module (Lin et al., ICCV 2019), and recurrent/attention
temporal heads — all as first-class DeepTune video models that train,
evaluate, embed and infer through the existing pipeline, and that can serve
as the RGB / motion snippet feature extractors for Phase 4.

## Models (CLI `--video_arch`, combined with `--model_version` backbone)
Keep current names working. Add:
1. `single_frame` — done in Phase 2; keep.
2. `late_fusion` — shared 2D backbone on two distant frames (first & last
   sampled), embeddings concatenated, merged by FC layers at the top.
3. `early_fusion` — first conv layer widened to take all T frames stacked on
   channels (`3T` in-channels); initialise from the pretrained 2D weights by
   replicating/averaging the first-layer kernel across time (I3D-style
   inflation); rest of the backbone unchanged. Support at least the ResNet
   family; error clearly for unsupported backbones.
4. `slow_fusion` — map the name to a documented native 3D default
   (r2plus1d_18). No new network needed unless trivial.
5. `frame_pool` (existing) — **fix G5**: attention pooling must weigh the
   per-frame **embeddings**, then classify the pooled embedding; mean pooling
   also pools embeddings. Keep the old logits-pooling behaviour loadable for
   old checkpoints (version the checkpoint or detect by keys).
6. `tsm` — ResNet-family 2D backbone with a temporal shift inserted in each
   residual block (shift 1/8 of channels forward, 1/8 backward, residual
   "shift-in-branch" variant; `--tsm_fold_div`, default 8). Zero extra
   parameters; pretrained 2D weights load unchanged.
7. `cnn_rnn` — 2D backbone per frame → LSTM or GRU (`--rnn_type`,
   `--rnn_hidden`, `--rnn_layers`, `--bidirectional`) → head.
8. `temporal_transformer` — 2D backbone per frame → learned positional
   encoding + small TransformerEncoder (`--tt_layers`, `--tt_heads`) → CLS head.
9. `two_stream` — spatial stream (2D backbone on an RGB frame) + temporal
   stream (2D CNN on a stack of `L` motion fields, 2L channels, first-layer
   inflation as in early fusion) → late fusion (`--fusion avg|fc`).
   Motion source `--motion_source farneback|mv|raft`:
   `farneback` (OpenCV, default, no weights), `mv` (Phase 1 codec motion
   vectors, upsampled to the stream's input size — the cheap option), `raft`
   (torchvision `raft_small`, optional, GPU). Motion fields cached to disk
   keyed by video hash + params; normalisation documented. `extract_embed`
   must be able to return the RGB and motion embeddings **separately** (Phase 4
   localisation needs both streams).
10. **(stretch)** `--token_merging R` for ViT/Swin backbones (ToMe, ICLR 2023):
    merge R tokens per block, training-free; report speed-up and accuracy
    change on the benchmark.

All new models: `forward(x, extract_embed=False)` with x `[B,T,C,H,W]`;
`extract_embed=True` returns the clip embedding the head sees. PEFT/LoRA where
the backbone supports it. Freezing options respected. Register every model in
`utils.build_video_model`; record every new option with `save_args`.

## Tests
- Unit: output shapes for cls/reg, `extract_embed` shapes, frozen params
  frozen, PEFT param count small, checkpoint save→load→same output.
- Early-fusion inflation: with identical frames, output equals the 2D
  backbone's output on that frame (up to tolerance).
- TSM: with `fold_div` such that nothing shifts, or with identical frames,
  output equals the plain 2D backbone; parameter count identical to the 2D
  backbone; a shift moves channel content exactly one step in time.
- Motion: on a synthetic translating texture, mean Farneback flow and mean
  codec MV ≈ known shift (sign and magnitude within tolerance); cache hit
  returns identical arrays.
- **Oracle tests (the key assurance):** on the Phase 2 `direction` task at CI
  size, fixed seeds, small budget: `single_frame` and mean `frame_pool` stay
  near chance; `tsm`, `two_stream` (farneback and mv), `cnn_rnn`,
  `temporal_transformer`, `early_fusion` reach clearly above chance (choose
  thresholds with margin, e.g. ≥ 80%). Use small/dummy backbones
  (`weights=None` tiny CNN) so CI stays fast; mark slower variants
  `@pytest.mark.slow`.
- Backward compatibility: a checkpoint from the old frame-pool attention
  model still loads and predicts.

## Deliverables
- Updated `scripts/run_motion_benchmark.py` covering all models (run on GPU),
  including the `scene_bias` task; results table with accuracy, train time and
  inference clips/sec in PROGRESS.md and `docs/video-plan/benchmark.md`.
- README "Models DeepTune Supports" table extended with the video models and
  a one-line explanation of each.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1` and paste the summary, paste the
benchmark table, show `git diff --stat tests/`. Do not lower oracle
thresholds to get green — if a model can't beat chance, investigate and
report. If anything fails twice, stop and report.

# Phase 1 — Scalable video data layer

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 1 only. Start in plan mode: show me your design (classes,
files, flags, data formats, test list) and wait for my OK before coding.

## Goal
Video datasets of thousands of clips must train without holding every clip
in memory, decode frames accurately and fast, sample time sensibly, and
avoid leakage between splits. The existing bytes-in-parquet format keeps working.

## Current state (verify)
`ParquetVideoDataset` stores whole encoded clips as bytes in parquet, writes
each to a temp file per `__getitem__`, and seeks with OpenCV
(`CAP_PROP_POS_FRAMES`) to uniformly spaced indices. One clip per video.
`handlers/video_manifest.load_video_manifest` reads every file into RAM.

## Tasks
1. **Decoder abstraction** `datasets/video_io.py`:
   `VideoReader(path_or_bytes).metadata()` → fps, frame_count, duration,
   width, height, codec; `.get_frames(indices)` → uint8 `[T,H,W,3]` RGB.
   Backends in priority order: TorchCodec (if it installs cleanly with our
   pinned torch on Windows), PyAV, OpenCV (fallback, sequential-read when
   seeking is unreliable). Check what actually works with our pinned
   versions and record the decision in PROGRESS.md. Bytes input must not
   require a temp file if the backend supports in-memory decoding.
2. **Path-based dataset** `VideoDataset` that reads from a manifest of paths
   (`videos`, `labels`, optional `group`, `start_sec`, `end_sec`, extra columns
   passed through as today). Lazy decoding in `__getitem__`. Keep
   `ParquetVideoDataset` as a thin compatibility path. `deeptune.py`,
   `handlers/split_dataset.py`, trainer/evaluator/embed/inference must accept
   either format; detect automatically.
3. **Temporal sampling** (`--temporal_sampling`): `uniform` (current default),
   `segment` (TSN-style: split into T segments, random frame per segment in
   train, centre in eval), `dense` (fixed stride/window, used later for
   sliding windows). Also `--clip_stride` / `--clip_duration_sec` options.
4. **Clip augmentation** (train only, `--video_augment`): temporal jitter,
   spatial random-resized-crop and horizontal flip applied **identically to
   every frame of a clip**, mild colour jitter, optional speed perturbation.
   Note: horizontal flip must be disable-able because it flips the meaning of
   left/right motion (Phase 2 benchmark depends on it) — default OFF for
   flip, and document why.
5. **Multi-clip test-time averaging** (`--test_clips N`): average logits over
   N temporal clips per video in eval/inference.
6. **Decoded-frame cache** (optional, `--frame_cache DIR`): cache sampled,
   resized frames (e.g. `.npy`/uint8) keyed by file hash + sampling params.
7. **Group-aware splits:** if a `group` column exists, `split_dataset` keeps
   each group in one split (stratified-group where possible). Error clearly if
   a split would be empty.
8. **Robustness:** clear errors for unreadable files, zero-length videos,
   variable-fps clips, very short clips (fewer frames than requested — pad by
   repeating last frame, logged), mixed resolutions, portrait videos.
9. Save every new option via `args.save_args` so inference and the GUI Test
   page rebuild the same pipeline.

## Tests (add, don't weaken)
- Generate tiny MP4s in fixtures (few colours that change per frame so the
  frame index is recoverable from pixel values). Assert `get_frames` returns
  the **exact** requested frames for each backend available.
- Sampling: shapes, determinism with seed, segment sampling covers the clip,
  eval sampling deterministic.
- Augmentation: same crop/flip on all frames of a clip (check via pixel
  correspondence); flip off by default.
- Group splits: no group appears in two splits (property test over random
  group layouts).
- Backward compat: existing parquet-bytes datasets + saved runs still train,
  evaluate and infer.
- Memory: a manifest of 1,000 tiny clips constructs the dataset without
  reading clip bytes (assert no file reads at construction, e.g. by mocking).

## Benchmark
Add `scripts/bench_video_io.py`: clips/sec for each backend and for old vs new
dataset on ~200 generated clips. Record numbers in PROGRESS.md.

## Definition of done
All of the above through CLI; video smoke scripts pass for the old parquet
format and the new manifest format; docs (README video section,
desktop/README) updated; full suite green; committed in small steps.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1` and paste the real summary. Show
the benchmark table. Show `git diff --stat tests/` and confirm no existing
test was weakened or deleted. If anything fails twice, stop and report.

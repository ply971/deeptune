# Phase 1 — Scalable video data layer + motion vectors + feature cache

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 1 only. Start in plan mode: show me your design (classes,
files, flags, data formats, test list) and wait for my OK before coding.

## Goal
Video datasets of thousands of clips must train without holding every clip
in memory, decode frames accurately and fast, sample time sensibly, avoid
leakage between splits, and — the main speed lever — **decode once and
extract features once** so later heads (Phase 4 localisation) train on cached
features instead of pixels. Motion should come cheaply from the codec where
possible. The existing bytes-in-parquet format keeps working.

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
   pinned torch on Windows; try its CUDA/NVDEC decoding on the RTX 5060 too),
   PyAV, OpenCV (fallback, sequential-read when seeking is unreliable). Check
   what actually works with our pinned versions and record the decision in
   PROGRESS.md. Bytes input must not require a temp file if the backend
   supports in-memory decoding.
2. **Codec motion vectors** (CoViAR, CVPR 2018): `.get_motion_vectors(indices)`
   using PyAV with `flags2=+export_mvs`, returning a dense per-block field
   `[T, 2, H/b, W/b]` (dx, dy; I-frames → zeros + a flag). Document block size,
   sign convention, and normalisation. Clear error / automatic Farneback
   fallback when the codec exposes no MVs (e.g. all-intra files).
   Verify it on our own generated clips and on an H.264 file.
3. **Path-based dataset** `VideoDataset` that reads from a manifest of paths
   (`videos`, `labels`, optional `group`, `start_sec`, `end_sec`, extra columns
   passed through as today). Lazy decoding in `__getitem__`. Keep
   `ParquetVideoDataset` as a thin compatibility path. `deeptune.py`,
   `handlers/split_dataset.py`, trainer/evaluator/embed/inference must accept
   either format; detect automatically.
4. **Temporal sampling** (`--temporal_sampling`): `uniform` (current default),
   `segment` (TSN: random frame per segment in train, centre in eval), `dense`
   (fixed stride/window, used later for sliding windows), and `motion`
   (MGSampler, ICCV 2021: build the cumulative motion curve from codec MVs or
   frame differences and sample uniformly along it). Also `--clip_stride` /
   `--clip_duration_sec`.
5. **Clip augmentation** (train only, `--video_augment`): temporal jitter,
   spatial random-resized-crop and horizontal flip applied **identically to
   every frame of a clip**, mild colour jitter, optional speed perturbation.
   Horizontal flip must be disable-able because it flips the meaning of
   left/right motion (Phase 2 benchmark depends on it) — default OFF for flip,
   and document why. If MVs are used, a flip must also negate dx.
6. **Multi-clip test-time averaging** (`--test_clips N`).
7. **Decoded-frame cache** (`--frame_cache DIR`): sampled, resized uint8
   frames in a memory-mapped array, keyed by file hash + sampling params.
8. **Snippet feature extraction + cache** (`extract_features.py` or
   `deeptune.py --extract_snippet_features`): split each video into snippets
   (`--snippet_frames`, default 16, `--snippet_stride`), run a frozen backbone
   once on the GPU, store per-video `[N_snippets, D]` float16 arrays plus a
   JSON sidecar (fps, snippet_frames, stride, backbone, weights, stream).
   Support two streams: `rgb` (any DeepTune video/frame backbone) and `motion`
   (codec MVs or Farneback flow through a small/inflated 2D or 3D backbone).
   Layout must be loadable by the Phase 4 localisation head (concatenated
   `[N, D_rgb + D_motion]` with the split recorded). Batched, resumable,
   skips already-extracted videos.
9. **Group-aware splits:** if a `group` column exists, `split_dataset` keeps
   each group in one split (stratified-group where possible). Error clearly if
   a split would be empty.
10. **Robustness:** clear errors for unreadable files, zero-length videos,
    variable-fps clips, very short clips (pad by repeating last frame,
    logged), mixed resolutions, portrait videos.
11. **GPU-aware loading:** `pin_memory` + `non_blocking` transfers when CUDA is
    used; worker count default sensible on Windows.
12. Save every new option via `args.save_args` so inference and the GUI Test
    page rebuild the same pipeline.

## Tests (add, don't weaken)
- Generate tiny MP4s in fixtures (colours that change per frame so the frame
  index is recoverable from pixel values). Assert `get_frames` returns the
  **exact** requested frames for each backend available.
- Motion vectors: on a generated clip of a texture translating by a known
  (dx, dy), the median MV over the moving region has the right sign and
  magnitude within tolerance; a static clip gives ≈ 0; flip negates dx.
- Sampling: shapes, determinism with seed, segment sampling covers the clip,
  eval sampling deterministic; motion sampling puts more samples where the
  motion is (clip that is static for half its length).
- Augmentation: same crop/flip on all frames of a clip; flip off by default.
- Feature cache: extracting twice gives identical arrays and the second run
  does not decode (assert by mock); snippet count matches
  `ceil((frames - snippet_frames)/stride)+1` (or documented formula).
- Group splits: no group appears in two splits (property test).
- Backward compat: existing parquet-bytes datasets + saved runs still train,
  evaluate and infer.
- Memory: a manifest of 1,000 tiny clips constructs the dataset without
  reading clip bytes (assert no file reads at construction).
- All tests must pass on CPU; GPU-only paths are skipped cleanly without CUDA.

## Benchmark
Add `scripts/bench_video_io.py` on ~200 generated clips + a few real H.264
files: clips/sec per decode backend (CPU and NVDEC if available); old vs new
dataset; **MV extraction vs Farneback flow cost** per clip; epoch time from
pixels vs from cached features. Record the table in PROGRESS.md.

## Definition of done
All of the above through CLI; video smoke scripts pass for the old parquet
format and the new manifest format; docs (README video section,
desktop/README) updated; full suite green; committed in small steps.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1` and paste the real summary. Show
the benchmark table. Show `git diff --stat tests/` and confirm no existing
test was weakened or deleted. If anything fails twice, stop and report.

# A2 — Phase 1 data layer (tasks 1–6, 9–12)

Agent A2, branch `video/A2-data-layer`, based on `video-analysis` @ 05cf682 (A1 + A4 merged).
Tasks 7 (frame cache) and 8 (snippet feature cache) are left to B1; the hooks are described below.

## Plan (written first; executed without waiting, as instructed)
- **New** `datasets/video_io.py` (decoder abstraction + codec motion vectors), `datasets/video_sampling.py`
  (uniform / segment / dense / motion), `datasets/video_augment.py` (clip-consistent augmentation),
  `src/video/multi_clip.py` (test-time clip averaging), `handlers/video_args.py` (shared CLI flags),
  `datasets/video_synth.py` (index-coded / translating clips for tests and the benchmark),
  `scripts/bench_video_io.py`.
- **Rewrite** `datasets/video_datasets.py`: one shared base, `VideoDataset` (paths) and `ParquetVideoDataset` (bytes),
  storage auto-detected; bytes never reach worker pickles (G15).
- **Wire** `cli.py`, `deeptune.py`, `infer.py`, `trainers/video/train.py`, `evaluators/video/evaluate.py`,
  `embed/video/embed.py`, `inference/video.py`, `inference/data.py`, `handlers/raw_to_parquet_dataset.py`,
  `handlers/video_manifest.py`, `handlers/split_dataset.py`, `options.py`, `desktop/config.py` (InferenceConfig only),
  `non_blocking` transfers in the shared `Trainer`/`TestTrainer`.
- Tests: exact frames per backend, MV sign/magnitude/static/flip/fallback, sampling properties, augmentation
  consistency, 1,000-clip construction without reads, worker pickles, group-split property test, flags recorded
  and carried to the Test page, end-to-end train/eval/embed/infer for both storage formats.
- Risks: breaking the existing parquet path and saved runs (kept and tested); existing tests that pin defaults
  (`raw_to_parquet` returns bytes — kept as the function default, the CLI passes `--video_storage paths`).

## Done
1. **Decoder** `VideoReader(path_or_bytes, backend)`: `metadata()`, `get_frames(indices, clamp_to_end)`,
   `iter_frames(width, gray)`, `frame_times()`, `time_to_index()`, `get_motion_vectors()`, `motion_curve()`,
   `fingerprint()`. PyAV decodes bytes in memory (tested: no temp file). Exact frame indices: sequential decode
   for short spans; for long videos a demux-only timestamp index + **keyframe-aware** seeking (seek only when a
   later keyframe lets us skip). Display rotation applied (matches OpenCV, tested with a rotated file).
2. **Codec motion vectors** via PyAV `flags2=+export_mvs` → `[T, 2, ceil(H/b), ceil(W/b)]`, b=8 default.
   Conventions: pixels of the stored frame; sign = later minus earlier (+dx right, +dy down, same as Farneback
   prev→next); divided by the display distance to the reference anchor; I-frames → zeros + `is_intra`.
   `method='auto'` falls back to Farneback (logged) when the codec has none (tested on FFV1);
   `method='codec'` raises `MotionVectorsUnavailable`. Verified on our generated clips: H.264 (x264 refs=1)
   and MPEG-4 Part 2 give the exact translation (±0.25 px), B-frames keep the sign, static ≈ 0, codec and
   Farneback agree within 0.3 px. Multi-reference H.264 (x264 default refs=3) yields vectors spanning 2–3 frames
   that FFmpeg cannot attribute; `reencode_for_motion_vectors()` (CoViAR's MPEG-4, P-only) fixes it (tested).
3. **Path-based `VideoDataset`** + `ParquetVideoDataset` (bytes, compatible) behind `open_video_dataset()` /
   `ParquetVideoDataset.from_parquet()` (auto-detects by the parquet schema / first value). Manifest columns:
   `videos`, `labels`, optional `group`, `start_sec`, `end_sec`, passthrough extras (as before).
4. **Temporal sampling** `--temporal_sampling uniform|segment|dense|motion`, `--clip_stride`, `--clip_duration_sec`,
   `--motion_source auto|codec|framediff`. `uniform` eval is bit-identical to the old `linspace(...).astype(int)`.
5. **Augmentation** `--video_augment` (train only): temporal jitter, one random-resized crop + colour jitter per clip,
   `--video_speed_perturb`; flip only with `--video_flip` (default off; documented why). `ClipAugment.apply_to_motion`
   applies the same crop (scales vectors by the zoom) and flip (mirrors grid, negates dx).
6. **Multi-clip TTA** `--test_clips N` in eval, embed and inference: softmax probabilities averaged **before** the
   A4 error model; returned as `log(mean p)` so every existing consumer is unchanged. `infer.py` adds a
   `clip_disagreement` column.
9. **Group-aware splits**: a `group` column is used automatically for video (`resolve_grouper`), the split is checked
   (`check_group_split`: clear error if a split is empty, assertion if a group spans two splits, warning listing classes a
   split lacks), and `split_summary.json` records clip and group counts.
10. **Robustness**: clear `VideoDecodeError` (a `ValueError`) for missing / 0-byte / corrupt files and streams with no
    frames, reported with the dataset row; short clips padded by repeating the last frame (logged once per row);
    variable-fps clips detected (timestamps used for seconds→frames, warning logged); mid-stream resolution changes
    resized; mixed resolutions across clips fail with a readable collate error when no resizing transform is used;
    portrait and rotated videos keep the right shape; string labels rejected with a hint.
11. **GPU-aware loading / G15**: `--num_workers auto|N` (`options.resolve_num_workers`), `pin_memory` on CUDA,
    `non_blocking` transfers in Trainer/TestTrainer/embed/inference, persistent workers. Path datasets pickle as paths;
    bytes datasets move clips to a memory-mapped Arrow IPC file before workers start (`prepare_for_workers`), and bytes
    parquet files > 256 MB are memory-mapped from the start (sidecar in `.deeptune_video_cache/`, git-ignored).
12. Every new option is a CLI flag recorded by `save_args`; `infer.py` accepts the test-time ones and the desktop Test
    page (`InferenceConfig.video_options`) passes a run's recorded values back (only non-defaults, so old runs and
    existing argv expectations are unchanged).

## New CLI flags (`handlers/video_args.py`, video modality only)
| Flag | Default | Where |
|---|---|---|
| `--video_backend auto\|torchcodec\|pyav\|opencv` | auto | all + infer.py |
| `--temporal_sampling uniform\|segment\|dense\|motion` | uniform | all + infer.py |
| `--clip_stride N` | 1 (dense) | all + infer.py |
| `--clip_duration_sec S` | whole clip | all + infer.py |
| `--motion_source auto\|codec\|framediff` | auto | all + infer.py |
| `--test_clips N` | 1 | all + infer.py |
| `--num_workers auto\|N` | auto | all + infer.py |
| `--video_storage paths\|bytes` | paths | deeptune.py `--raw-data` |
| `--video_augment`, `--video_flip`, `--video_speed_perturb X` | off, off, 0 | training |

## Data formats
- **paths** parquet/CSV/XLSX: `videos` = string paths (absolute; relative ones resolve against the manifest folder).
  Default for `--raw-data` ingestion now; the run's dataset parquet for 40 clips is < 20 kB.
- **bytes** parquet: `videos` = binary (the original format, still produced by `--video_storage bytes` and by
  `raw_to_parquet(...)` called without the new argument).
- Memory-mapped sidecar: Arrow IPC, one `large_binary` column `videos`, `<parquet dir>/.deeptune_video_cache/<stem>-<sha1[:12]>.videos.arrow`.

## Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`, `options` prints `cpu 0`)
- New files only: `98 passed, 3 skipped, 3 warnings in 26.99s` (the 3 skips = TorchCodec not installed).
- `pytest tests -k "video or split or manifest or dataset or desktop_config or inference_data or evaluation or workflow"`:
  `234 passed, 3 skipped, 120 deselected, 12 warnings in 56.08s`.
- `pytest tests/training/test_param_budget.py tests/test_error_model_pipeline.py tests/training/test_vision_trainer.py tests/training/test_video_trainer.py`:
  `68 passed, 10 warnings in 34.33s`.
- `pip check`: `No broken requirements found.`
- CPU smoke runs (all exit 0, `"passed": true`): `smoke_video.py --model resnet18 --manifest` 28 s,
  `--model resnet18 --parquet_bytes` (new option) 24 s, `--model r3d_18` 23 s.
- Full suite **not run** (coordinator's instruction).

`git diff --stat 05cf682 -- tests/` (only new files; no existing test edited, skipped or deleted):
```
 tests/test_video_data_layer.py   | 358 +++
 tests/test_video_io.py           | 257 +++
 tests/test_video_pipeline_e2e.py | 117 +++
 tests/test_video_sampling.py     | 181 +++
 4 files changed, 913 insertions(+)
```

## CPU benchmark (`scripts/bench_video_io.py --clips 20 --workers 0 1 2`, CPU only)
python 3.12.10, torch 2.14.0+cu130 (GPU hidden), av 19.0.2, opencv 5.0.0, 16 logical CPUs, Windows. Generated clips:
H.264 112×112, 64 frames; real files: the repo's 4 demo videos (1080p H.264, ~1,100–2,600 frames, GOP ≈ 250).

| Measurement | Variant | Result | Notes |
|---|---|---|---|
| decode generated H.264 | torchcodec | n/a (not installed) | |
| decode generated H.264 | pyav | 37.6 clips/s | 26.6 ms/clip, 8 frames uniform |
| decode generated H.264 | opencv | 20.2 clips/s | 49.5 ms/clip |
| decode real 1080p files (4) | pyav | 1.4 clips/s | 715 ms/clip |
| decode real 1080p files (4) | opencv | 0.8 clips/s | 1185 ms/clip (sequential read) |
| decode | pyav + NVDEC | n/a | not run (CPU only; `--nvdec` tries it) |
| dataset `__getitem__` | old: bytes + temp file + OpenCV seek | 26.4 clips/s | 37.9 ms/clip |
| dataset `__getitem__` | new: bytes, PyAV in memory | 63.1 clips/s | 15.9 ms/clip |
| dataset `__getitem__` | new: paths, PyAV | 65.9 clips/s | 15.2 ms/clip; construct 2 ms |
| worker pickle | bytes dataset before → after `prepare_for_workers` | 0.29 MB → 1.8 kB | paths dataset 3.2 kB |
| motion field, 8 frames | codec / H.264 vs Farneback | 19.6 vs 38.1 ms/clip | |
| motion field, 8 frames | codec / MPEG-4 vs Farneback | 16.1 vs 33.6 ms/clip | |
| motion curve, whole clip | codec vs framediff (H.264) | 38.8 vs 21.9 ms/clip | MGSampler curve |
| DataLoader epoch, 20 clips | workers 0 / 1 / 2 | 0.32 / 2.84 / 2.97 s | first batch 0.07 / 2.60 / 2.84 s |
| epoch from cached snippet features | — | n/a until B1 | |

Worker start-up for a real run: `import deeptune` (what each spawned worker re-imports under `deeptune.py`) takes
**15.4 s / 14.8 s** (warm), `import torch` alone 2.1 s (PowerShell `Measure-Command`). With ~15 ms of decode per
small clip, spawning pays off only after thousands of clip-epochs, hence the Windows default (0 workers under 500
clips, else up to 4). The coordinator should rerun with `--clips 200` (and `--nvdec`) on the GPU.

## Backend decisions
- **PyAV is the default** (installed, 1.6–1.9× faster than OpenCV here, in-memory bytes, exact seeking, the only
  source of codec MVs, display rotation). **OpenCV** stays as the fallback (sequential reads only; `CAP_PROP_POS_FRAMES`
  is not trusted). **TorchCodec** is implemented behind an import guard (`seek_mode='exact'`), auto-preferred when
  installed, skipped cleanly otherwise. Coordinator: worth a *try* in a throwaway venv (`pip install torchcodec` matching
  torch 2.14); on Windows it needs FFmpeg shared libraries on PATH, and its CUDA/NVDEC path needs the CUDA build. Not
  required — PyAV covers everything in Phase 1.
- `av==19.0.2` added to `requirements.txt` (CI installs from it) and to `packaging/deeptune.spec` metadata.

## Bug found and fixed during benchmarking
The first PyAV seek rule ("re-seek if the next target is > 96 frames ahead") looped forever when a GOP is longer
than that (the seek lands on the same keyframe again). The tests missed it because their clips had GOP 48.
Fixed with keyframe-aware seeking (commit f7d9d11) and a regression test with GOP 250.

## Open problems
- GPU checks not run (by instruction): `scripts\check.ps1` on the GPU should now pass, because 40-clip smoke runs use
  0 workers on Windows; please confirm, and run `bench_video_io.py --clips 200 --nvdec`.
- `deeptune.py` prints `[ERROR] CSV file does not exist ... training_log.csv` on every images/video run (pre-existing,
  wrong folder level; board V-015). Cosmetic.
- Spawned workers re-import all of `deeptune.py`'s imports (~15 s). Lazy imports in `deeptune.py`/`utils.py` would make
  workers cheap; out of scope.
- Images still use `options.NUM_WORKERS` (unchanged; same G15 pattern on Windows + CUDA).
- G14 (per-frame `Resize((224,224))` squashes portrait video) is unchanged: portrait clips decode correctly but are
  still squashed by the default transform.
- Codec MV magnitude is approximate for multi-reference H.264/HEVC (documented; re-encode helper available).
- Path datasets need the files to stay where they were at training time (documented in both READMEs);
  `--video_storage bytes` gives a self-contained dataset.
- Motion curves are cached per dataset instance (per worker); B1's cache should persist them.

## Notes for later agents
- **B1 frame cache (task 7)**: subclass or wrap `_VideoClipDataset._decode_frames(i, reader, indices)`; key with
  `clip_key(i)` (sha1 of bytes, or path+size+mtime) + `num_frames`, the sampling spec and the resize. Eval sampling is
  deterministic, so eval clips cache cleanly; training clips (segment/augment) should cache decoded frames, not tensors.
- **B1 snippet features (task 8)**: `VideoReader.iter_frames()` streams every frame; `sample_clip_indices` with
  `SamplingSpec('dense', snippet_frames, clip_stride=...)` and `num_clips` gives windows; motion stream:
  `get_motion_vectors(indices, method='auto')` (convention above), or `reencode_for_motion_vectors` first for exact MVs.
- **B2 temporal models / two-stream**: clips stay `[B, T, C, H, W]`; with `--test_clips` the model sees
  `[B, N, T, C, H, W]` only through `MultiClipModel`. A motion input must use `ClipAugment.apply_to_motion` with the
  clip's params so flips negate dx.
- **Phase 2 (A3)**: keep `--video_flip` off for the direction task; `datasets/video_synth.translating_clip` makes clips with
  known motion.
- **Phase 5 / R2**: `clip_disagreement` (0 = clips agree; cls: mean total-variation distance of each clip's probabilities
  to the mean; reg: std of clip outputs) is a candidate extra feature for the error model / conformal scores; it is only
  written by `infer.py` / `predict_video` when `--test_clips > 1`. The error model itself is unchanged.
- **S1a**: replace `spatial/io.py::iter_frames()` with `enumerate(VideoReader(p).iter_frames())` (board I-001).
- **GUI (Phase 7)**: `RunConfig` has no widgets/fields for the new flags yet; `handlers/video_args.VIDEO_DATA_ARG_NAMES`
  lists them and is torch-free.

## Commits
dfa70c9, 39208f5, 10ab77c, 06ba79c, f7d9d11, a1fe13c, c80e500 (report).

## A2 follow-ups (Critic wave-1b, V-016..V-020: "merge after fixes")
Step 0: merged `video-analysis` (308945e, plan docs only; no conflicts) → f944c7e.

- **V-016 (must-fix) — error model vs `--test_clips`: fixed** (ed29f10). The video trainer now fits the A4 error
  model on validation with `num_clips=test_clips`, through `MultiClipModel`, so the fit sees the same averaged
  probabilities as evaluation. The Trainer change is three optional hooks inside A4's marked block, following
  A4's own `getattr` pattern: `error_fit_loader`, `error_fit_model` and `error_fit_info`. They default to the old
  behaviour (a test checks this). `error_model_fit.json` / `fit_info` now record `test_clips`.
  `src.video.multi_clip.error_model_clips_mismatch()` warns at eval and inference when the run's error model was
  fitted on a different clip count; old runs without the key count as 1.
  Tests:
  - the Trainer's saved model equals a hand fit on the 3-clip averaged probabilities, and those differ from the
    1-clip ones;
  - a real `trainers.video.train` run with `test_clips=2` records it;
  - the mismatch warning fires both ways;
  - the e2e run asserts `fit['test_clips'] == 2`.
- **V-017 (must-fix) — infer.py ignores recorded sampling: fixed** (1bbd943). When a flag is not on the command
  line, `infer.py` takes it from the checkpoint run's `cli_arguments.json`. This covers `num_frames`, `pooling`,
  `video_backend`, `temporal_sampling`, `clip_stride`, `clip_duration_sec`, `motion_source` and `test_clips`.
  `num_workers` is not inherited, because it describes the training machine. Recorded values are re-validated
  by the flag's own parser, and infer.py prints the settings it used.
  Tests:
  - a dense / stride-2 / 3-clip / 4-frame / attention run inferred with no flags uses exactly those settings;
  - explicit flags win;
  - with no recording, the defaults are used;
  - e2e: plain `infer.py` on a `--test_clips 2` run writes `clip_disagreement`.
- **V-018 (suspected → reproduced and fixed)** (71f6bc4). The bug was reproduced with PyAV alone.
  `datasets.video_synth.trim_with_edit_list` remuxes a clip with its first 3 packets before the MP4 edit list,
  as a `-c copy` trim does. On the H.264 copy, 3 packets are `is_discard` and the decoder shows 37 of 40 frames.
  - Before the fix: `frame_count` was 40, and both the 40-frame (sequential) and 390-frame (seek path) tests fail
    (checked by swapping in the old file).
  - Now: discarded packets are left out of the index, but their keyframes are kept as seek landing points. If the
    first decoded frame's pts differs from the index start, the reader falls back to exact sequential reads.
  - OpenCV shows the same frames, but `CAP_PROP_FRAME_COUNT` still counts hidden samples. OpenCV counts are now
    marked `frame_count_exact=False`; reads clamp.
- **V-019 nice-to-haves (done except e)** (1bbd943):
  - (a) Test: `--video_augment --video_flip --video_speed_perturb` never changes validation/test clips; they stay
    deterministic under any RNG state.
  - (b) `video_loader_kwargs(role=...)`:
    - train: persistent;
    - val: persistent, but on Windows with `--num_workers auto` capped at 2 workers (train + val ≤ 4 + 2 alive
      instead of 8; an explicit count wins);
    - one-shot loaders (eval, embed, inference, error-model fit): not persistent.

    Making val non-persistent instead would re-spawn its pool every epoch (≈15 s per worker) without lowering
    the peak. My own earlier test (`test_loader_kwargs_pin_memory_only_on_cuda_and_prepare_workers`, added by A2)
    now expects `persistent_workers=False` for a one-shot loader.
  - (c) When Windows refuses to delete a still-mapped spill file, the delete is retried at exit. Each new spill
    sweeps `deeptune-clips-*.arrow` files older than 24 h from %TEMP%. Test included.
  - (d) Unchanged: the split summary already prints the auto-used `group` column.
  - (e) Not done, as instructed: wiring `--video_flip` to A3's flip guard waits for the A3 merge.
- Tests after the follow-ups (CPU, `cpu 0`):
  - A2 test files: `109 passed, 3 skipped, 3 warnings in 24.67s`.
  - `-k "video or split or manifest or dataset or desktop_config or inference_data or evaluation or workflow"`:
    `245 passed, 3 skipped, 120 deselected, 12 warnings in 41.62s`.
  - param_budget + error_model_pipeline + error_model + vision/video trainer:
    `86 passed, 10 warnings in 26.81s`.
  - `pip check` clean.
  - `git diff --stat 05cf682 -- tests/` still shows only A2's own new test files (now 5).

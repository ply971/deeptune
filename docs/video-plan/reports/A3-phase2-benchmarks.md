# A3: Phase 2 report (motion, untrimmed and scene-bias benchmarks)

Agent A3 worked on branch `video/A3-phase2-benchmarks`, cut from `video-analysis` at `7782b0b`.
All runs were on CPU (`CUDA_VISIBLE_DEVICES=-1`, `options.DEVICE == cpu`, `NUM_WORKERS == 0`).

## Plan (written before the work started)

- **New package `benchmarks/`**
  - `render.py` holds the textures, the textured blob on a torus, the grey distractor, the MP4
    writer and the codec choice.
  - `synthetic_motion.py` has five seeded generators plus the CLI
    `python -m benchmarks.synthetic_motion --task ... --out DIR`.
  - `motion_features.py` holds the Farneback, appearance, FFT-texture and codec-MV features.
  - `probes.py` holds the honesty probes.
  - `guards.py` holds the flip-augmentation guard.
- **Output per dataset:** MP4s, `manifest.csv` plus per-split CSVs (readable by
  `handlers/video_manifest.py`, with a `group` column), `dataset.json` (ground truth), and
  `masks.npz` (blob masks). Untrimmed datasets also get `gt_segments.csv`
  (`video-id,t-start,t-end,label`) and `gt_activitynet.json`.
- **Single-frame baseline:** `src/video/single_frame.py`, registered in
  `utils.build_video_model` through `pooling="single_frame"`. That needs a minimal, grouped edit
  to `utils.py`, plus `--pooling` choices in `cli.py` and `infer.py`.
- **Scripts:**
  - `scripts/run_motion_benchmark.py` produces the results table.
  - `scripts/check_benchmark_honesty.py` reports the probe numbers at any size.
  - `scripts/make_benchmark_figures.py` draws the doc figures.
- **Tests** live in new files only:
  - `tests/benchmarks/test_synthetic_benchmarks.py` covers formats, determinism and loader
    compatibility.
  - `tests/benchmarks/test_benchmark_honesty.py` covers the honesty checks.
  - `tests/benchmarks/test_run_motion_benchmark.py` covers the runner.
  - `tests/models/test_single_frame.py` covers the model.
- **Risks I saw up front:**
  - The OpenCV wheel has no H.264 encoder, so the files fall back to `mp4v`.
  - PyAV might be missing, in which case the codec-MV test skips.
  - Single-frame leakage if the trajectories are not symmetric. I made the world a torus and
    centred the trajectories so that it cannot happen.
  - Laptop load, so I kept every run small and used 4 torch threads.

## What was done

1. Generators for all five tasks. They are deterministic per clip, seeded by
   `(seed, task, split, index)`, and fast: about 1.5 s for 500 clips at 32×32 × 12 frames.
2. Honesty probes and tests. All pass. The numbers are below.
3. The single-frame baseline: `--pooling single_frame`, which works with any 2D backbone with
   or without PEFT, and with the shared `Trainer`.
4. The benchmark runner: `scripts/run_motion_benchmark.py`, which reports accuracy, AUC, time
   and device. Each neural model goes through `build_video_model`, the manifest loader,
   `ParquetVideoDataset` and `Trainer`. A `flow_logreg` reference row is included.
5. The flip guard. Benchmarks record `hflip_safe` and `vflip_safe`.
   `benchmarks.guards.check_flip_augmentation` raises `UnsafeAugmentationError` for a flip on
   direction-labelled data. The runner's `--hflip` calls it.
6. Docs: `docs/video-plan/benchmark.md` covers all five tasks, the formats and the numbers.
   It includes a GIF per class for direction, periodic and speed, an untrimmed timeline figure,
   and a biased vs unbiased example figure. The README links to it, and `pyproject.toml`
   now packages `benchmarks*`.

## Files

**New:**
- `benchmarks/__init__.py`
- `benchmarks/render.py`
- `benchmarks/synthetic_motion.py`
- `benchmarks/motion_features.py`
- `benchmarks/probes.py`
- `benchmarks/guards.py`
- `src/video/single_frame.py`
- `scripts/run_motion_benchmark.py`
- `scripts/check_benchmark_honesty.py`
- `scripts/make_benchmark_figures.py`
- `tests/benchmarks/test_synthetic_benchmarks.py` (17 tests)
- `tests/benchmarks/test_benchmark_honesty.py` (11)
- `tests/benchmarks/test_run_motion_benchmark.py` (5)
- `tests/models/test_single_frame.py` (11)
- `docs/video-plan/benchmark.md` and `docs/video-plan/img/*` (6 GIFs and 2 PNGs, 480 KB)
- this report

**Edited (small, grouped):**
- `utils.py`: import `VideoSingleFrameClassifier`, add `VIDEO_POOLING_CHOICES`, add two
  `pooling == "single_frame"` branches in `build_video_model`. That is +22 lines.
- `cli.py`: one line, `--pooling` choices and help.
- `infer.py`: one line, `--pooling` choices.
- `pyproject.toml`: add `benchmarks*` to `packages.find`.
- `README.md`: one feature bullet edited, one added.

I did not touch `datasets/`, `options.py`, `handlers/`, `trainers/`, `evaluators/`, `inference/`
or `desktop/`. I also did not touch `PROGRESS.md`, as instructed.

## Data formats

See `docs/video-plan/benchmark.md` § "Output format" for the full schema. Summary:

- **Layout:** `out/videos/<video_id>.mp4`, `manifest.csv`, `train.csv` / `val.csv` /
  `test.csv`, `dataset.json`, `masks.npz`. Untrimmed datasets add `gt_segments.csv`
  (`video-id,t-start,t-end,label`) and `gt_activitynet.json` (an ActivityNet "database" with
  `subset` training/validation).
- **Manifest columns:**
  - `videos` (relative path), `labels`, `label_idx` (sorted-class index, the same order as
    `LabelEncoder`), `group`, `split`, `video_id`, `background_family`.
  - scene_bias adds `family_matches_class`.
  - untrimmed adds `label_<class>` (multi-hot), `num_frames`, `fps`, `duration_sec` and
    `background_motion`.
  - There are deliberately no `start_sec` / `end_sec` columns, because A2's loader treats those
    as clip trims (I-001).
  - All of them load through `read_video_manifest` / `load_video_manifest` /
    `raw_to_parquet` / `ParquetVideoDataset` unchanged (tested).
- **Ground truth:** `dataset.json` per video holds `centers[T]`, `velocity` or `motion`, and the
  family. Untrimmed videos also hold `segments` (`start_frame`, `end_frame` exclusive, `t_start`,
  `t_end`), `background_intervals` (the complement), the `distractor` track and `multi_hot`.
- **Masks:** `load_blob_masks(root, vid)` returns `bool[T, H, W]` (True = blob, i.e. the action),
  stored as `np.packbits`. This is the background-only mask for Phase 6.
- **Codec:** `auto` tries `avc1` once per process. OpenCV's wheel here has no OpenH264, so it
  falls back to `mp4v` (MPEG-4 Part 2, which also carries motion vectors). The codec is recorded
  in `dataset.json`.

## Tests

Targeted run, CPU, `CUDA_VISIBLE_DEVICES=-1`, from the worktree root:

```
python -m pytest tests/benchmarks tests/models/test_single_frame.py tests/models/test_video.py tests/test_video_manifest.py tests/training/test_video_trainer.py -q
60 passed in 145.86s
```

After the follow-up round, which merged `video-analysis` and fixed V-021 to V-023 plus the flip
wiring (CPU, `options.DEVICE == cpu`, `NUM_WORKERS == 0`):

```
python -m pytest tests/benchmarks tests/models/test_single_frame.py -q
49 passed in 95.53s
python -m pytest tests/test_video_pipeline_e2e.py tests/models/test_video.py tests/models/test_desktop_config.py tests/test_video_manifest.py tests/training/test_video_trainer.py -q
47 passed, 3 warnings in 28.05s
```

Of the 60, 44 are new. The other 16 are existing video model, manifest and trainer tests,
re-run because `utils.py` changed. The honesty file takes about 55 s. The codec-MV test ran
because PyAV 19.0.2 is now in the shared `.venv`; without PyAV it skips with a reason. The full
suite was not run, per the coordinator's rule.

## Honesty-check numbers

Seed 0, `mp4v`, 12 frames (periodic 16). Held-out accuracy unless R² is stated. The thresholds
were set before the first full run of the probes. They were **not fixed in advance of every result** (the honesty
test file has a single commit, made after the generator), and the post-failure adjustments to the
probes and the generator are listed under Decisions. The thresholds themselves were not changed.

**Scope:** "a single frame carries no signal" is shown only for linear probes on pixels (and FFT
texture features) and for a frozen resnet18. A fine-tuned `single_frame:resnet18` at 700/400 and
a lossless-codec variant (to exclude direction-asymmetric MPEG-4 artefacts) are **pending a GPU
run by the coordinator** (Critic V-022). No slide should make the claim before that.

| Probe | CI size (32×32; 300/200, speed and periodic 200/200, scene_bias 300/200/200) | Slide size (64×64; 700/400, scene_bias val 200) | Test requires |
|---|---|---|---|
| direction: mean horizontal Farneback flow (11 numbers) → LR | 1.000 | 0.9925 | ≥ 0.95 |
| direction: first / middle / last frame pixels → LR | 0.500 / 0.485 / 0.540 | 0.4825 / 0.4775 / 0.545 | ≤ 0.60 |
| direction: frame average → LR | 0.515 | 0.560 | ≤ 0.60 |
| direction: codec MVs (PyAV) → LR | 0.985 (48×48) | 1.000 | ≥ 0.95 |
| speed: flow summary → ridge R² / middle frame R² | 0.805 / −0.07 | 0.874 / −0.13 | ≥ 0.7 / ≤ 0.2 |
| periodic: flow summary / middle frame | 1.000 / 0.445 | 0.9975 / 0.495 | ≥ 0.9 / ≤ 0.60 |
| scene_bias: background only (blob masked out), biased val / unbiased test | 0.915 / 0.520 | 0.940 / 0.4975 | ≥ 0.80 / ≤ 0.60 |
| scene_bias: flow, val / test | 1.000 / 1.000 | 0.995 / 0.995 | ≥ 0.95 |
| untrimmed: decoded red pixels > 0 ⇔ inside a segment; mask IoU ≥ 0.6; multi-hot = segment label set | holds on all frames | n/a | exact |

To reproduce the slide-size column (CPU, about 4 min):
`python scripts/check_benchmark_honesty.py --size 64 --mv_size 64 --n_train 700 --n_test 400 --out .artifacts/honesty-slide`.

## First benchmark table (CPU, small)

These numbers come from `scripts/run_motion_benchmark.py`, seed 0, `--size 64 --resize 112
--threads 4 --codec mp4v --added_layers 1` (the default). "Frozen" means only the head is
trained, on ImageNet (2D) or Kinetics-400 (r3d_18) features, with `--epochs 5 --batch_size 8
--lr 1e-3`.

**direction** (train 300 / test 200, 12 frames, frozen):

| model | family | device | train_s | test_acc | test_auc |
|---|---|---|---|---|---|
| flow_logreg | probe | cpu (sklearn) | 16.7 | 1.0 | 1.0 |
| single_frame:resnet18 | single_frame | cpu | 89.3 | 0.52 | 0.5331 |
| mean:resnet18 | mean | cpu | 231.4 | 0.51 | 0.5362 |
| attention:resnet18 | attention | cpu | 232.1 | 0.505 | 0.5196 |
| r3d_18 (frozen) | native | cpu | 640.4 | 0.505 | 0.5543 |

**direction, r3d_18 fully fine-tuned** (train 200 / test 200, 12 frames, `--epochs 3 --lr 1e-4`, no `--freeze_backbone`):

| model | family | device | train_s | test_acc | test_auc |
|---|---|---|---|---|---|
| r3d_18 (fine-tuned) | native | cpu | 599.0 | 1.0 | 1.0 |

**scene_bias** (train 300 / biased val 200 / unbiased test 200, bias 0.9, frozen):

| model | family | device | train_s | biased_val_acc | biased_val_auc | unbiased_test_acc | unbiased_test_auc |
|---|---|---|---|---|---|---|---|
| flow_logreg | probe | cpu (sklearn) | 25.1 | 0.995 | 0.99 | 0.995 | 0.9902 |
| single_frame:resnet18 | single_frame | cpu | 86.1 | 0.935 | 0.9401 | 0.505 | 0.4953 |

**Reading the tables:**
- Single frame and frame pooling (mean and attention), all frozen, are at chance on direction,
  as expected. Fine-tuned single-frame and pooling runs are pending (GPU).
- The frozen r3d_18 is **also at chance**. Kinetics features plus a linear head do not separate a
  1–2 px/frame drift direction at 112 px with 300 clips. Fine-tuning the whole r3d_18 instead
  gives **1.000** on held-out test after 3 epochs. *Correction (V-021):* the per-epoch
  "validation" of 96 / 99.5 / 100 % in that run was **test** accuracy. Direction had no val
  split then, so the runner logged on test. There was no model selection, so the final number
  is unaffected, but it was watched during training. The runner now always validates on a
  seeded val split. That meets the PLAN's expectation that a 3D model is well above chance while
  single-frame and frame pooling are not, but only once its spatio-temporal filters are trained.
- On scene_bias, the single-frame model shows the textbook failure. It scores 0.935 on the biased
  val split by reading the background, and collapses to 0.505 on the unbiased test split. Flow
  stays at 0.995 on both.
- The real table belongs to the coordinator or the Phase 3 run: 700/400 clips, full fine-tuning,
  on the GPU:
  `scripts/run_motion_benchmark.py --task direction --models flow_logreg single_frame:resnet18 mean:resnet18 attention:resnet18 r3d_18 --n_train 700 --n_test 400 --size 64 --epochs 5 --lr 1e-4`.

## Decisions

- **Torus world, with trajectories centred on the middle frame.** This makes every *rendered*
  single frame and the frame average class-independent exactly. On the encoded clips this is
  checked only with linear probes and a frozen resnet18; codec artefacts remain a possible
  channel until the lossless and fine-tuned checks run. Clips are grouped by scene: two clips per shared background, one per class, so the
  background can never carry the label.
- **The blob is red-dominant and the background is muted.** The untrimmed "blob present ⇔ in
  segment" test therefore runs on *decoded pixels*, independently of the generator.
- **Classes are sorted.** That makes `label_idx` equal DeepTune's `LabelEncoder` order.
- **The single-frame model is selected via `--pooling single_frame`**, not via a new
  model_version, so it combines with every 2D backbone and PEFT without new registry entries.
  Native 3D models raise a `ValueError` if asked for it.
- **The runner uses the existing `ParquetVideoDataset` (bytes) path and the shared `Trainer`.**
  The Phase 1 path dataset (A2's `open_video_dataset`) can replace it later. Since the V-021 fix
  it validates on a seeded val split, never on test.
- **Fixes I made to probes and generator after investigating failures** (thresholds unchanged):
  1. *Codec MV area overflow.* PyAV's `w`/`h` are uint8, and 16·16 overflowed to 0, which gave
     NaN. I fixed it by widening, and also switched to `motion_x / motion_scale` for sub-pixel
     precision.
  2. *Blob texture.* A smooth blob gave a GT-masked Farneback speed correlation of only 0.6,
     because flow inside a textureless disc is under-determined. I gave the blob a fine,
     high-contrast texture, which brought it to 0.85.
  3. *Speed statistic.* Whole-frame flow percentiles had outliers on striped backgrounds. On
     tiny frames Farneback spreads flow along the stripes (aperture problem); this happens on the
     raw frames too, so it is not the codec. I gated it on changing pixels (|Δgrey| > 40), which
     moved R² from 0.33 to 0.80. Farneback is now `levels=3, winsize=5, iterations=5`.
  4. *Codec MV at 32×32.* The probe reached only 0.81, a codec property (4 macroblocks per
     frame), so the MV test runs at 48×48 and gets 0.985. I record both numbers rather than
     lowering the threshold.
  5. *Distractor visibility.* The distractor colour is chosen to contrast with the background,
     so it visibly moves.
- **One wording slip:** the message of commit `eaebdb2` says it "gives the blob a fine
  high-contrast texture". That change actually landed in `3d92edb`.

## Open problems

- **H.264 is not produced on this machine**: the OpenCV wheel has no OpenH264 encoder and there
  is no system ffmpeg. PyAV is now in `.venv` and could write `libx264`, but I kept OpenCV (task
  rule). Phase 1 can switch `render.write_mp4` to PyAV when it is available.
- **The 2D models were only run frozen.** The fine-tuned single-frame and frame-pooling rows,
  and the 700/400 sizes, still need the GPU run (command above). On frozen features every model
  except flow is at chance.
- **The desktop GUI has no widget for `single_frame` yet.** That is Phase 7 work. Its config
  validation already accepts it (fixed in the follow-up below).
- **The flip guard is wired only in `deeptune.py`**, where the raw dataset path is known (see the
  follow-up below). A direct `trainers/video/train.py` call on parquet splits has no
  `dataset.json` to check, so the guard cannot run there.
- **The single-frame "no signal" claim is only partly checked.** Fine-tuned single-frame models
  and a lossless-codec variant are pending (V-022, GPU).
- **CI does not install PyAV**, so there the codec-MV honesty test skips with a reason.

## Notes for later phases

- **Phase 3:** use `direction` as the oracle. Motion-aware models should approach `flow_logreg`
  (≥ 0.99), while `single_frame` stays ≤ 0.60. Don't use codec-MV motion on clips under about
  48 px. At 32 px, Farneback needs gating to give magnitudes.
- **Phase 4:** untrimmed `gt_segments.csv` uses ActivityNet's eval column names.
  - Times are `frame / fps` with an exclusive end.
  - In the ActivityNet JSON, test videos are `subset: validation`, following the THUMOS / CtxPL
    convention.
  - `background_intervals` and the distractor track allow CAS-mass-outside-segment and
    false-positive-on-distractor metrics.
- **Phase 6:** use `load_blob_masks` for saliency-in-mask, and the drop from scene_bias biased val
  to unbiased test as the background-reliance gap. See P-001 (SCUFO static probe) on the board.
- **Board:** I posted I-002 (interfaces), F-001 (honesty numbers) and P-001 (re F-007, the SCUFO
  probe). Nothing in REVIEWS.md or PROPOSALS.md was addressed to A3.

## A3 follow-ups (after the Critic's review, V-020..V-024)

- **Merge:** I merged `video-analysis` at d517c15 (A1, A2, A4, S1a, desktop) into this branch,
  in commit `b558baf`.
  - The `--pooling` one-liners in `cli.py` and `infer.py` keep `single_frame` next to A2's
    video-data arguments and A4's `--no_error_model`.
  - `--pooling` still lives in `cli.py` / `infer.py`, not in `handlers/video_args.py`.
  - `desktop/config.py` RunConfig validation now accepts `single_frame` (V-023). The GUI widgets
    are Phase 7. `InferenceConfig` does not validate pooling.
- **V-021:**
  - `make_direction_dataset`, `make_speed_dataset` and `make_periodic_dataset` take `n_val`
    (CLI `--n_val`, default 200). The val split is seeded per `(seed, task, 'val', index)` with
    disjoint scenes.
  - `scripts/run_motion_benchmark.py` always generates a val split and refuses `--data` without
    one, or `--n_val 0`. It gives the Trainer `loaders["val"]`, so `error_model.json` is fitted
    on val. A test checks `error_model_fit.json` reports `val_dataset_size` = n_val, which
    differs from n_test.
  - The r3d_18 "validation" sentence above is corrected.
- **V-022:** the claim is scoped and the thresholds are described as set before the first full run, with the later adjustments listed (above
  and in `benchmark.md`). Fine-tuned single-frame (direction, speed, periodic at 700/400) and a
  lossless-codec variant are pending the coordinator's GPU run.
- **Flip guard wired:** `deeptune.py` calls `benchmarks.guards.check_flip_augmentation(args.df,
  hflip=True)` when `--video_flip` is set, and turns a refusal into a normal argparse error
  (exit 2).
  - Tests (`tests/benchmarks/test_flip_guard_pipeline.py`): a direction benchmark manifest is
    refused. A flip-safe benchmark (periodic) and ordinary data without `dataset.json` pass the
    guard.
  - The trainers only see parquet splits, so the check happens where the raw path is known.

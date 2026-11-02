# Cat detector and page 07 integration

User requested an animal-specific trained detector and sequential independent
video tracking; chose cats and approved downloads and bounded GPU training.

## Implementation

- `scripts/train_cat_detector.py`: Oxford-IIIT Pet images and whole-animal boxes
  from trimap foreground/boundary; dogs as negative images. Official trainval
  split supplies disjoint training and validation; official test is held out.
  Deterministic subset IDs are saved. Batch 2, zero workers, two CPU threads,
  frozen MobileNet backbone, GPU allocator capped at 55%. Default five epochs.
- COCO Faster R-CNN MobileNetV3 320 FPN head starts with background and cat
  weights. Baseline uses that same two-class head. Epoch selection uses validation
  AP50; test is evaluated for baseline and selected checkpoint. Metrics are
  all-points interpolated AP at IoU 0.5 and precision/recall at score 0.5,
  **not COCO mAP or tracking accuracy**. Single seed and photo subsets are limits.
- `spatial/pet_detector.py`: strict local checkpoint loading with
  `weights_only=True`, cat-only prompts, lazy torch imports and original-frame
  boxes. Existing ByteTrack and output formats are reused.
- Page 07 offers **Trained cat detector** and a checkpoint path field. One
  video runs at a time; each action has a separate output folder. GUI stays
  torch-free. Training is currently through the script, not a GUI training form.

## Reproduce (from repository root)

Use the exact training command below for the saved mirror subset and its
recorded exclusions. With a checkpoint already installed:

```powershell
.venv\Scripts\python.exe track.py --video path/to/cat.mp4 --prompt cat --detector pet_fasterrcnn --detector_model saved_models/cat-detector/best.pt --batch_size 1 --save_video --out .artifacts/cat-tracking/new-video01
```

For page 07, select the trained cat detector, paste the absolute `best.pt` path,
load a video, and press **Find, track and count**. Repeat with the next video.

## Experiment completed 2026-11-01

Run artifacts are under the main checkout's `.artifacts/cat-tracking/`.
`run01/report.json` contains measured photo metrics, `run01/splits.json` the
exact cohort, and `experiment.json` the checkpoint hashes and video observations.
The selected model is installed at `saved_models/cat-detector/best.pt`; page 07
discovers it automatically and its import remains torch-free.

Oxford hosts timed out previously. The saved images came from the
[original-file mirror](https://huggingface.co/datasets/Shiym/ViT-FineTune/tree/main/oxford-iiit-pet).
The annotations archive matches Oxford's MD5 `95a8c909bbe2e81eed6a22bccdf3f68f`.
Individual images were decoded and their SHA256 hashes recorded locally;
they were not independently verified against the original images archive.
Mirror metadata does not replace Oxford's source terms.

Before training, explicit data-quality exclusions removed `Birman_160` and
`Egyptian_Mau_117` (missing from the mirror), `Egyptian_Mau_165` (training mask
has no foreground), and `Egyptian_Mau_162` (validation mask has no foreground).
No replacements or split reassignment: 897 train / 239 validation / 300 test
images, including 717 / 191 / 240 cats; remaining images are dog negatives.
The test cohort is unchanged. All remaining images and cat masks passed
preflight; no split shares IDs or exact image-byte hashes.

Five GPU epochs, seed 42, batch 2, zero workers, two CPU threads, 55% CUDA
allocator cap. Epoch 4 maximised validation AP50; epoch times were 30.4–35.8 s
including validation. Frozen backbone tensors remain identical to baseline;
head tensors changed. PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130.

| Held-out photos (300 images, 240 cats) | Transferred two-class baseline | Fine-tuned epoch 4 |
|---|---:|---:|
| AP50, all-points interpolated | 0.991725 | 0.993065 |
| Precision at score 0.5 | 0.865455 | 0.963563 |
| Recall at score 0.5 | 0.991667 | 0.991667 |
| True positives / false positives at score 0.5 | 238 / 37 | 238 / 9 |

The baseline transfers COCO background and cat classifier/regressor weights
into the same two-class head; it is not untouched 91-class COCO inference.
AP50 improves only 0.134 percentage points. Single seed, no interval or
significance claim, possible COCO pretraining overlap, and no video accuracy
inference from these photo metrics. The experiment does not establish a
general improvement over pretrained detection.

### Exact training command

Run from the main checkout (prepared data already present):

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:TORCH_HOME=(Join-Path (Get-Location) '.artifacts/cat-tracking/torch-cache')
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.venv\Scripts\python.exe .claude/wt/cat-tracking/scripts/train_cat_detector.py --data .artifacts/cat-tracking/data --out .artifacts/cat-tracking/run02 --epochs 5 --device cuda --exclude-image Birman_160 --exclude-image Egyptian_Mau_117 --exclude-image Egyptian_Mau_165 --exclude-image Egyptian_Mau_162
```

Use a new output directory: an existing checkpoint is never overwritten.
`--mirror-subset --prepare-only` downloads the selected images while respecting
the explicit exclusions; mirror provenance records them. `--download` instead
requests the full original archives, which previously timed out here.

### Separate real-video runs

All frames processed on GPU, batch 1, one video at a time. Baseline and trained
checkpoints used identical default thresholds and ByteTrack settings.

| Video | Frames | Baseline track IDs | Trained track IDs | Trained wall time |
|---|---:|---:|---:|---:|
| Black Cat walking | 453 | 3 | 3 | 26.6 s |
| Cat Play | 1139 | 4 | 3 | 43.6 s |

These are track-fragment counts, not unique animals. Eight evenly spaced
frames from each original and each overlay were inspected: the visible cat
is boxed in many frames, but close approaches, cropping and changing pose
cause misses, duplicate boxes and identity fragmentation. A frame with any
detection is not a correct-detection metric. No HOTA, IDF1, detection AP or
tracking accuracy is claimed without labelled video ground truth.

Trained outputs: `trained-video01/black-cat-walking/overlay.mp4` and
`trained-video02/cat-play/overlay.mp4`, each alongside detections, tracks and
summary JSON/CSV. Baseline outputs have the corresponding `baseline-video*`
paths. `*-overlay-review.jpg` stores the inspected contact sheets.

Example tracking command (repeat only after the previous video finishes):

```powershell
.venv\Scripts\python.exe track.py --video .artifacts/cat-tracking/videos/black-cat-walking.mov --prompt cat --detector pet_fasterrcnn --detector_model saved_models/cat-detector/best.pt --device cuda --batch_size 1 --save_video --out .artifacts/cat-tracking/new-video01
```

Attribution is saved in `videos/ATTRIBUTION.json`, including input hashes and
the added boxes/IDs, silent output and resolution changes. Derived Cat Play
overlays and contact sheets retain CC BY-SA 4.0:

- [Cat Play](https://commons.wikimedia.org/wiki/File:Cat_Play.webm),
  NairobiPapel(Kamaa), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- [Black Cat walking](https://commons.wikimedia.org/wiki/File:Black_Cat_walking.webm),
  Dzkouslavia, [CC0](https://creativecommons.org/publicdomain/zero/1.0/).

## Verification

Required reviews V-054 through V-056 accepted the mirror exclusion fix, strict
CPU checkpoint reload, saved splits, epoch selection, reported metrics and
real-video observations.
New preflight/mirror tests: 3 passed in 0.17s. Earlier combined targeted run:
9 passed in 11.86s. Final full CPU suite, 2026-11-02:
**1668 passed, 5 skipped, 92 warnings in 2076.31s (0:34:36)**, exit code 0.
This includes the multi-seed studies. Pip check: No broken requirements found.

From `.claude/wt/cat-tracking`:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
../../../.venv/Scripts/python.exe -m pytest tests -q --basetemp=.artifacts/final-check-20261102/pytest-tmp --junitxml=.artifacts/final-check-20261102/results.xml
../../../.venv/Scripts/python.exe -m pip check
```

The saved log is `.artifacts/final-check-20261102/pytest.log` in that worktree;
JUnit records 1673 tests, zero failures, zero errors and five skips. No existing
test was weakened or newly skipped. The earlier incomplete run is superseded
by this result. The installed checkpoint hash and all four saved overlays were
rechecked. Page 07 usage is documented in `desktop/README.md`.

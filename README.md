# *DeepTune*

[![deeptune tests](https://github.com/ply971/deeptune/actions/workflows/test.yml/badge.svg)](https://github.com/ply971/deeptune/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/deeptune/badge/?version=latest)](https://deeptune.readthedocs.io/en/latest/)

***DeepTune*** is a full compatible library to automate Computer Vision, Natural Language Processing, Tabular, Time Series, and Video state-of-the-art deep learning algorithms for cross-modal applications on image, text, tabular, time series, and video datasets. The library is designed for use in different applied machine learning domains, including but not limited to medical imaging, natural language understanding, time series analysis, providing users with powerful, ready-to-use CLI tool that unlock the full potential of their case studies through just a one simple command.

***DeepTune*** is primarily presented for undergraduate and graduate computer science students community at St. Francis Xavier University (StFX) in Nova Scotia, Canada. We aspire to seeing this software adopted broadly across the computer science research community all over the world.

## DeepTune Demo

**Images Demo:** The data sample used in this demo is a subset of the Chest X-Ray Images (Pneumonia) dataset available at: [Chest X-ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

https://github.com/user-attachments/assets/3a2e62a2-4715-4cab-b5ff-b23f4684f212

More cross-modal video demos can be found in the documentation's [demo page](https://deeptune.readthedocs.io/en/latest/guides/demos.html).

## Desktop GUI

Use Python 3.12 and install the project from its repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

On Windows, the default `torch` wheel is CPU-only. To train on an NVIDIA GPU, replace
it with the CUDA build of the same versions (CUDA 13.0 also covers RTX 50-series / Blackwell cards):

```powershell
.venv\Scripts\python.exe -m pip install "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

DeepTune picks the GPU automatically when one is available.

Prefer a graphical interface over the command line? Run:

```powershell
.venv\Scripts\python.exe desktop\app.py
```

to pick a modality and dataset, choose a model and its hyperparameters, launch the run, and browse results, all without typing CLI flags. The desktop app includes saved settings, automatic full logs, bounded background previews, cancellation, and metrics CSV export. See [desktop/README.md](desktop/README.md) for details and repeatable video checks.

The installed commands are `deeptune` and `deeptune-desktop`. Run automated checks
with `.venv\Scripts\python.exe -m pytest tests -q`, or everything at once (pip check,
tests, video smoke runs) with `scripts\check.ps1` (`-Quick` skips the smoke runs).

Don't want to set up Python at all? See [packaging/README.md](packaging/README.md)
for building `DeepTune-Setup.exe`, a standalone Windows installer of the desktop
app with every dependency bundled in.

## Features

- Fine-tuning state-of-the-art Computer Vision algorithms (ResNet, DenseNet, etc.) for image classification.
- Fine-tuning state-of-the-art NLP (BERT, GPT-2) algorithms for text classification.
- End-to-end training for tabular and time-series algorithms.
- Fine-tuning video classifiers on raw video clips, either by sampling frames through the existing 2D vision backbones (with pooling over time), or through native spatio-temporal architectures (3D-CNNs and video transformers) that learn motion directly.
- Providing PEFT with LoRA support for Computer Vision and Video algorithms implemented, enabling state-of-the-art models that typically require substantial computational resources to perform efficiently on lower-powered devices. This approach not only reduces computational overhead but also enhances performance.
- Leveraging fine-tuned and pretrained state-of-the-art vision, video, and language models to generate robust knowledge representations for downstream visual, temporal, and textual tasks.
- Reporting how many of a model's parameters are actually trained, and letting you cap that number (images and video).

## Trainable-parameter budget (images and video)

Every image and video training run logs one line such as
`trainable params: 98,304 || all params: 33,231,000 || trainable%: 0.2958` and writes
`parameter_summary.json` (total, trainable, frozen, and a backbone / LoRA / head
breakdown) next to `model_weights.pth`.

`--trainable_params COUNT` sets how many parameters may be trained. COUNT accepts
`250000`, `100k`, `1.5M`, `2m` (underscores and thousands commas are fine).

- **Without `--use-peft`**: the new head (`fc1`/`fc2`, and the attention layer of
  frame-pooling video models) is always trained; then backbone layers are unfrozen
  from the output side down while they fit. The rest stays frozen.
- **With `--use-peft`**: DeepTune picks the largest LoRA rank that fits. In DeepTune
  the LoRA rank is also the head's hidden width (`--embed_size`), so the chosen rank
  is saved as `embed_size` in `cli_arguments.json`, and evaluation, embedding and
  the Test page rebuild the same model.

The budget overrides `--freeze-backbone`. If even the head (or LoRA rank 1) does not
fit, the run stops with the smallest achievable count and suggests `--added_layers 1`
or a smaller `--embed_size`. SigLIP, text, tabular and timeseries runs do not support it.

```powershell
# images: train the head plus as many top ResNet layers as fit in 500k parameters
.venv\Scripts\python.exe deeptune.py --modality images --df data\images --raw-data --out results --model_version resnet18 --trainable_params 500k
# video: LoRA on r3d_18, rank chosen so that at most 100k parameters train
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version r3d_18 --use-peft --trainable_params 100k
```
  
## Models *DeepTune* Supports (Up to Date)


<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Transfer Learning with Adjustable Embedding Layer?</th>
      <th>Support PEFT with Adjustable Embedding Layer?</th>
      <th>Support Embeddings Extraction?</th>
      <th>Task</th>
      <th>Modality</th>
      <th>Supported Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'</td>
    </tr>
    <tr>
      <td>DenseNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'densenet121', 'densenet161', 'densenet169', or 'densenet201'</td>
    </tr>
    <tr>
      <td>Swin</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'swin_t', 'swin_s', or 'swin_b'</td>
    </tr>
    <tr>
      <td>EfficientNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', or 'efficientnet_b7'</td>
    </tr>
    <tr>
      <td>VGGNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'vgg11', 'vgg13', 'vgg16', or 'vgg19'</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32' or 'vit_h_14'</td>
    <tr>
    </tr>
    <tr>
      <td>SiGLiP</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification</td>
      <td>Image</td>
      <td>'siglip'</td>
    <tr>
      <td>Frame-Pooling Video</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Video</td>
      <td>Any of the ResNet/DenseNet/Swin/EfficientNet/VGGNet/ViT/ConvNeXt versions above, applied frame-by-frame and pooled over time (mean or attention)</td>
    <tr>
      <td>Native Video (3D-CNN / Video Transformer)</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Video</td>
      <td>'r3d_18', 'mc3_18', 'r2plus1d_18', 'mvit_v2_s', 'swin3d_t', 'swin3d_s', or 'swin3d_b'</td>
    <tr>
      <td>GPT</td>
      <td>✅</td>
      <td>—</td>
      <td>✅</td>
      <td>Text Classification</td>
      <td>Text</td>
      <td>GPT-2</td>
    </tr>
    <tr>
      <td>BERT</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Sentiment Analysis</td>
      <td>Text</td>
      <td>bert-base-multilingual-cased</td>
    </tr>
<tr>
  <td>GANDALF</td>
  <td colspan="2" style="text-align:center; font-weight:bold; font-size:16px;">Supports End-to-End Conventional Training</td>
  <td>✅</td>
  <td>Classification & Regression</td>
  <td>Tabular</td</td>
  <td>GANDALF</td>
</tr>
  <tr>
  <td>TabPFN</td>
  <td colspan="2" style="text-align:center; font-weight:bold; font-size:16px;">Supports Both Fine-tuning & Conventional End-to-End Training</td>
  <td>✅</td>
  <td>Classification & Regression</td>
  <td>Tabular</td</td>
  <td>tabpfn</td>
</tr>
<tr>
  <td>DeepAR</td>
  <td colspan="2" style="text-align:center; font-weight:bold; font-size:16px;">Supports Conventional End-to-End Training</td>
  <td>✅</td>
  <td>Time Series Forecasting</td>
  <td>Time Series</td>
  <td>DeepAR</td>
</tr>
</tbody>
</table>

## Video data: decoding, sampling, augmentation, test-time clips

**Storage.** With `--raw-data`, a class-folder dataset or a CSV/XLSX list of clips
(`videos` = file paths, `labels`, optional `group`, `start_sec`, `end_sec`, any extra
columns) is stored as **file paths** (`--video_storage paths`, the default): clips are
decoded lazily per sample and never copied, so thousands of clips train without being
held in memory. `--video_storage bytes` embeds the encoded clips in the parquet file
(portable; DeepTune's original format). Parquet files of either kind, and saved runs of
either kind, are detected automatically everywhere (train, evaluate, embed, `infer.py`).

**Decoding** (`--video_backend auto|torchcodec|pyav|opencv`, `datasets/video_io.py`):
PyAV by default (in-memory, frame-exact reads, display rotation applied); TorchCodec is
used first if installed; OpenCV is the sequential-read fallback.

**Temporal sampling** (`--temporal_sampling`):

| value | frames per clip |
|---|---|
| `uniform` (default) | evenly spaced over the clip (same as before) |
| `segment` | TSN: one random frame per segment while training, segment centres at test |
| `dense` | `--num_frames` consecutive frames `--clip_stride` apart |
| `motion` | MGSampler: evenly spaced along the cumulative motion curve (codec motion vectors or frame differences, `--motion_source`) |

`--clip_duration_sec S` samples from an S-second window (random position while training,
centred at test).

**Augmentation** (training only, `--video_augment`): temporal jitter, one random-resized
crop and one colour jitter per clip, applied identically to every frame; optional
`--video_speed_perturb 0.2` (playback speed 0.8–1.2×). **Horizontal flip is off unless
you add `--video_flip`**, because mirroring turns "moves left" into "moves right" and so
corrupts any label that depends on direction.

**Test-time clips** (`--test_clips N`): evaluation, embeddings and inference average the
class probabilities of N clips per video; `infer.py` also writes a `clip_disagreement`
column (0 = all clips agree).

**Workers** (`--num_workers auto|N`): `auto` uses 0 workers on CPU; on Windows with a GPU
0 workers below 500 clips (each spawned worker re-imports torch, which costs 15–40 s) and
up to 4 above; `min(8, cpu/2)` elsewhere. Workers receive file paths, never clip bytes.

**Splits.** A `group` column (for example the source video or the subject) is used
automatically for video: every group stays in one split; `split_summary.json` records the
per-split clip and group counts.

**Codec motion vectors** (CoViAR) are available as
`VideoReader(path).get_motion_vectors(indices)`: a `[T, 2, H/8, W/8]` field of dx, dy in
pixels per frame (positive = right / down), zero with `is_intra=True` on I-frames, and an
automatic Farneback fallback for codecs without motion vectors. They are exact for
single-reference encoders (MPEG-4 Part 2, x264 `refs=1`);
`datasets.video_io.reencode_for_motion_vectors` converts other H.264 files.

All of these options are saved in `cli_arguments.json`, so the desktop Test page and
`infer.py` rebuild the same pipeline. `scripts/bench_video_io.py` measures decode speed,
old vs new dataset, and motion-vector vs Farneback cost.

```powershell
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version r3d_18 --num_frames 16 --temporal_sampling segment --video_augment --test_clips 3
```

## Per-prediction error estimate (video & images)

Every video (and image) prediction can carry an **`expected_error`** column:

- **classification:** the probability that *this* prediction is wrong;
- **regression:** the expected absolute error of *this* prediction, in target units.

How it works: after training, DeepTune runs the final model once more over the
**validation split** (never the test set) and fits a small error model
(`error_model.json`, plain JSON, saved next to `model_weights.pth`, with fit
details in `error_model_fit.json`). For classification it maps the max softmax
probability onto the error rate actually observed at that confidence
(smoothed isotonic regression, so a less confident prediction never gets a
lower stated error). With fewer than 30 validation samples, or fewer than 5
wrong or 5 right predictions, it falls back to the validation error rate
`(errors + 1) / (n + 2)` for every sample. A regression network has no built-in
confidence signal, so for regression the estimate is the validation mean
absolute error for every sample until a later phase adds an uncertainty
signal. Pass `--no_error_model` to skip the fit. Runs without
`error_model.json` (old runs, or `--no_error_model`) just have no
`expected_error` column.

Where it appears: the eval folder's `predictions.csv` (one row per test
sample), `infer.py`'s `predictions.csv`, and a summary under `error_estimate`
in `full_metrics.json` / `inference_metrics.json`.

**Reading `error_calibration.csv`** (written on eval, and on inference when
the new data has labels): test samples are sorted by stated error and split
into 10 equal-count bins (deciles). Each row gives the bin's mean stated error
(`stated_mean`), the error actually observed in that bin (`observed_mean`), the
difference (`gap`), and a 95% interval for the observed value
(`observed_ci_low`/`observed_ci_high`; `within_95ci` says whether the stated value
falls inside it). If the estimate is trustworthy, `stated_mean ≈ observed_mean`
in every row. `calibration_error` in the summary is the count-weighted mean
|gap|. For example, 0.03 means the stated error is off by 3 percentage points
on average. For classification, `auroc_error_detection` measures how well
`expected_error` ranks wrong predictions above right ones (0.5 means no better
than chance). With a small test set the bins are small and the intervals wide,
so read the table as a whole rather than one bin at a time.

## Documentation

DeepTune is being under active development and mainteneance with a user-friendly comprehensive documentation for easier usage. The documentation can be accessed [here](https://deeptune.readthedocs.io/en/latest/).

## Acknowledgments
This software package was developed as part of work done at Medical Imaging Bioinformatics lab under the supervision of Jacob Levman at St. Francis Xavier Univeristy (StFX), Nova Scotia, Canada.

Thanks to Xuchen for providing their parameter-efficient fine-tuned Swin implementation [SwinTransformerWithPEFT](https://github.com/XuchenGuo/SwinTransformerWithPEFT)

## Citation

If you find *DeepTune* useful, please give us a star ⭐ on GitHub for support.

Also if you find this repository helpful, please cite it as follows:

```bibtex
@software{DeepTune,
author  = {Moayadeldin Hussain, John Kendall and Jacob Levman},
title   = {DeepTune: Cutting-edge Tool automating state-of-the-art deep learning models for cross-modal applications},
year = {2025},
url = {https://github.com/moayadeldin/deeptune},
version = {1.1.0}
}

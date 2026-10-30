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

## Detect and track objects in video (zero-shot)

Describe what to find in words. No labels or training needed:

```bash
python track.py --video lobsters.mp4 --prompt lobster --out runs/lobsters --save_video
python track.py --video_dir clips/ --prompt lobster --prompt fish --detector owlv2 --out runs/clips
```

An open-vocabulary detector finds each prompt in each frame:
**Grounding DINO** (`IDEA-Research/grounding-dino-tiny`, the default) or
**OWLv2** (`google/owlv2-base-patch16-ensemble`), both Apache-2.0 and loaded
through 🤗 `transformers`. The weights download on first use (about 0.6–0.7 GB
each). A re-implementation of **ByteTrack** (Zhang et al., ECCV 2022) then links
the boxes into tracks. It uses a constant-velocity Kalman filter, a Hungarian
(IoU) match of high-score boxes, and a second match of low-score boxes, which
keeps objects that are briefly blurred or partly hidden. Each track gets a
stable integer ID. A lost track keeps its ID if it comes back within
`--track_buffer` steps.

Each video gets its own folder (`<out>/<video name>/`):

| File | Contents |
|---|---|
| `detections.json` | every sampled frame: `box_xyxy` (pixels), `score`, `label` |
| `tracks.csv` | MOTChallenge format, no header: `frame,id,x,y,w,h,score,class,-1,-1` (frame is 1-based; class is the 1-based prompt index) |
| `track_labels.csv` | `track_id,class_id,label` |
| `summary.json` | per prompt: `count` (number of unique tracks), first/last seen (seconds), and each track's frames, duration and number of observations |
| `overlay.mp4` | with `--save_video`: the video with boxes and `label #id` drawn on it |

`<out>/counts.csv` lists video, label and count, and `<out>/track_arguments.json`
records every setting used.

Useful options:
- `--box_threshold`: default 0.2 for Grounding DINO, 0.1 for OWLv2.
- `--text_threshold`: Grounding DINO only.
- `--every_n_frames`: detect on every n-th frame only, which is faster.
- `--batch_size`, `--device auto|cpu|cuda`.
- `--track_high_thresh` / `--track_low_thresh` / `--new_track_thresh`, `--track_buffer`.
- `--class_agnostic`: let one track switch between prompts and take the majority label.

Run `python track.py --help` for the full list. `scripts/smoke_track.py` runs
both detectors with real weights on a generated clip.

**Accuracy depends on the prompt and the footage.** Short, concrete nouns work
best ("lobster", not "the animal on the left"). Small, blurred or unusual
objects may be missed or confused. Two same-label objects that stay hidden
behind each other for a long time can come back with a new ID, because
association uses box overlap and motion only, not appearance. These results
have not been measured yet. Detection mAP and tracking HOTA / MOTA / IDF1
arrive with the spatial metrics task (S4). Check `overlay.mp4` before you
trust the counts.

## Documentation

DeepTune is being under active development and mainteneance with a user-friendly comprehensive documentation for easier usage. The documentation can be accessed [here](https://deeptune.readthedocs.io/en/latest/).

## Acknowledgments
This software package was developed as part of work done at Medical Imaging Bioinformatics lab under the supervision of Jacob Levman at St. Francis Xavier Univeristy (StFX), Nova Scotia, Canada.

Thanks to Xuchen for providing their parameter-efficient fine-tuned Swin implementation [SwinTransformerWithPEFT](https://github.com/XuchenGuo/SwinTransformerWithPEFT)

Video tracking (`spatial/track.py`) re-implements ByteTrack (Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022; reference code MIT). Zero-shot detection uses Grounding DINO (Liu et al., ECCV 2024) and OWLv2 (Minderer et al., NeurIPS 2023) through Hugging Face `transformers` (Apache-2.0 checkpoints).

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

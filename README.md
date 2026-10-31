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
- Fine-tuning video classifiers on raw video clips, either by sampling frames through the existing 2D vision backbones (with pooling over time, or `--pooling single_frame` for the middle-frame baseline), through temporal architectures on those backbones (`--video_arch`: late and early fusion, TSM, CNN + RNN, temporal transformer, two-stream), or through native spatio-temporal architectures (3D-CNNs and video transformers) that learn motion directly.
- Seeded synthetic video benchmarks with exact ground truth (motion direction, speed, periodic motion, untrimmed videos with action segments, scene-biased backgrounds): `python -m benchmarks.synthetic_motion --task direction --out DIR`, then `scripts/run_motion_benchmark.py` to compare models. See [docs/video-plan/benchmark.md](docs/video-plan/benchmark.md).
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
  frame-pooling video models) is always trained; then backbone parameter tensors are
  unfrozen in reverse registration order (roughly output side first; for ViT, for
  example, `class_token`/`pos_embedding` are registered first, so the order is
  approximate) while they fit. The rest of the parameters stay frozen. BatchNorm
  running statistics are buffers, not parameters, so they still update in frozen
  layers during training.
- **With `--use-peft`**: DeepTune picks the largest LoRA rank that fits.
  - On native video models (`r3d_18`, `swin3d_t`, ...) the rank is `--lora_rank` (default
    16) and only the rank is searched: `--embed_size` stays the head's hidden width. The
    chosen rank is saved as `lora_rank` in `cli_arguments.json`. A budget too small for
    rank 1 plus the head stops with an error instead of narrowing the head.
  - Other PEFT models ignore `--lora_rank` and say so. Their rank is also the head's
    hidden width, so the chosen rank is saved as `embed_size`.
  - Evaluation, embedding and the Test page rebuild the same model from those records. A
    run recorded without `lora_rank` used `--embed_size` as its rank, and still rebuilds.
  - The log and `parameter_summary.json` (`lora_coverage`) report how many backbone
    conv/linear modules received LoRA. On r3d_18 that is all 20 Conv3d layers. Runs saved
    before this was fixed had LoRA on 4 of them; their weights still load, and the other
    adapters start with no effect, so their predictions do not change.
  - Swin's attention `qkv` and `proj` get no LoRA, because torchvision reads their weights
    directly and an adapter there would do nothing. swin3d_t therefore reports 28 of 52.
    Old Swin weights load with those adapters dropped and unchanged predictions. On 2D
    Swin those two were the only targets, so old `--use-peft` Swin runs trained the head
    alone; LoRA now goes on the patch embedding, MLPs and patch merging.
  - Each LoRA rank on r3d_18 costs about 98k trainable parameters, because the adapter's
    first layer keeps the conv's 3x3x3 kernel, so the default rank of 16 costs about 1.6M.

The head alone has `fc1_in × embed_size + embed_size + embed_size × classes + classes`
parameters with the default `--added_layers 2`. With the default `--embed_size 1000`
a ResNet-18 head is already about 515k, so choose the budget and `--embed_size`
together. The budget overrides `--freeze-backbone`. If even the head (or LoRA rank 1)
does not fit, the run stops with the smallest achievable count and suggests
`--added_layers 1` or a smaller `--embed_size`. SigLIP, text, tabular and timeseries
runs do not support it.

```powershell
# images: a 256-wide head (~132k) plus as many top ResNet-18 layers as fit in 500k parameters
.venv\Scripts\python.exe deeptune.py --modality images --df data\images --raw-data --out results --model_version resnet18 --embed_size 256 --trainable_params 500k
# video: LoRA on r3d_18 with a 256-wide head, rank chosen so that at most 1M parameters train (rank 8)
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version r3d_18 --use-peft --embed_size 256 --trainable_params 1M
```
`tests/training/test_param_budget.py` checks the arithmetic of every example in this
block, so keep it in sync when you change them.

## Training options (images and video)

Everything here is off by default and does not change a plain run. All flags are saved
in `cli_arguments.json`. SigLIP has its own training loop and refuses them, and text,
tabular and timeseries runs do not support them.

| Flags | What they do |
|---|---|
| `--precision bf16\|fp16` | Mixed precision on CUDA (fp16 with a GradScaler). On CPU it does nothing and logs that. |
| `--checkpoint_every N`, `--resume FILE` | Write `last_checkpoint.pt` every N epochs and after the last one, and continue from it at the next epoch boundary. |
| `--select_best`, `--early_stopping_patience N`, `--early_stopping_min_delta`, `--selection_metric` | Keep the best validation weights and reload them after training. Early stopping also stops the run after N epochs without improvement. |
| `--lr_scheduler cosine\|step`, `--warmup_epochs`, `--lr_step_size`, `--lr_gamma`, `--min_lr` | Learning-rate schedule, updated at every optimizer step. |
| `--grad_accum_steps N` | Update every N batches. The gradient is the mean over the samples of the window, and a shorter last window is still applied. BatchNorm still sees the small batches. |
| `--freeze_bn auto\|on` | Keep backbone BatchNorm in eval mode with frozen parameters, so small batches do not disturb the pretrained statistics. `auto` does this when `--batch_size` is under 16 (a starting guess, not a measured cutoff) and the backbone has BatchNorm and pretrained weights. |
| `--seed N` | Seeds python, numpy, torch, CUDA and DataLoader workers. In `deeptune.py` it also fixes the data split, so it varies training and keeps the test set. |
| `--config FILE` | JSON or YAML defaults, keyed by flag name without the dashes. Flags on the command line win, and an unknown key is an error. |
| `--experiment_logger mlflow\|wandb`, `--experiment_name` | Also log to MLflow or W&B if already installed. DeepTune does not install them. |
| `--label_smoothing` | Training loss only. Validation reports plain cross-entropy, so scores stay comparable. |
| `--mixup_alpha`, `--cutmix_alpha`, `--mix_prob` | Mixup and CutMix. All frames of a clip share one weight and one box, so the motion is kept. |
| `--layer_decay` | Each trainable block one step further from the output gets a learning rate this many times smaller. Depth follows module registration order. |
| `--ema_decay` | Validate, select and save an exponential moving average of the weights. |
| `--test_crops 3\|5` | Spatial views at test time, video only. Multiplies with `--test_clips`. Each view is a 7/8 window of the already-resized frame, so it is not the larger-frame 3-crop protocol of the Kinetics papers. |

```powershell
# video: fp16, cosine schedule with warmup, early stopping on the best weights, resumable
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version r3d_18 --seed 7 --precision fp16 --lr_scheduler cosine --warmup_epochs 1 --early_stopping_patience 5 --checkpoint_every 1
# the same run after a crash; --num_epochs stays the total
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version r3d_18 --seed 7 --precision fp16 --lr_scheduler cosine --warmup_epochs 1 --early_stopping_patience 5 --checkpoint_every 1 --resume results\deeptune-...\trainval_output_...\last_checkpoint.pt
```

**Resume.** A resumed run restores the model, optimizer, scheduler, scaler, EMA, best-so-far,
logs and all random generators, and on CPU it ends with exactly the same weights as an
uninterrupted run. It starts again from the last checkpoint, not from the middle of an
epoch. The run stops if the train/val data differ from the checkpoint's, because the model
would then have trained on samples now in validation or test: with `deeptune.py`, use
`--fixed-seed` or `--seed` in both runs. `--resume_allow_data_change` overrides this for a
deliberate warm start. Changed settings give a warning that names them. With checkpointing
on, workers restart every epoch, which costs time on Windows.

**Selection.** With `--select_best` or early stopping the validation split selects the
weights, so its scores are optimistic and it is no longer exchangeable with test data.
`training_run.json`, `error_model_fit.json` and the test `error_calibration.json` record
this. The per-prediction error model is fitted on the reloaded best weights, on that same
split, so use a separate split for any calibration that needs a coverage guarantee. A
`--resume` of a run that selected must select too, otherwise it stops with an error.

**Files next to the weights.** `training_metrics.csv` (per epoch: losses, accuracy, learning
rate, updates, seconds, GPU and process memory) and `training_run.json` (the seed in effect,
the options, what selected the weights, the run's memory maximum, and `environment`: commit,
torch version, cuDNN determinism flags) are always written. `peak_process_rss_mb` is sampled
at the end of each epoch and is not a peak of that epoch: on Windows it is the process's
lifetime high-water mark, elsewhere the current RSS. `peak_memory_meaning` in
`training_run.json` says the same.

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
      <td>Temporal Video (--video_arch)</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Video</td>
      <td>'late_fusion' (first and last frames), 'cnn_rnn' (LSTM or GRU over frames) and 'temporal_transformer' (transformer over frames): any 2D version above, or 'siglip'. 'early_fusion' (frames stacked on the first convolution), 'tsm' (temporal shift in each residual block) and 'two_stream' (RGB + motion fields): ResNet versions. 'slow_fusion': a native 3D CNN, 'r2plus1d_18' by default. 'single_frame': the middle frame, as a baseline</td>
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

**Frame cache** (`--frame_cache DIR`, off by default): sampled frames are shrunk so their
short side is `--frame_cache_size` pixels (default 256; `0` keeps the decoded size) and
stored as memory-mapped arrays, keyed by the file, the sampling settings, the frame indices,
the size and the decoder. A hit does not decode the sampled frames again, but each one still
probes the file's metadata, which demuxes it and decodes one frame. Only deterministic clips are cached:
all of validation, test, embedding and inference, and training with plain `uniform`
sampling. Training with `segment`, `dense`, `motion`, `--clip_duration_sec` or
`--video_augment` draws different frames every epoch, so those clips are decoded each time
(a warning says so) rather than reusing one draw. The first read of a clip shrinks it once,
so with a cache the frames the model's resize sees differ slightly from an uncached run.
The size belongs to the pipeline and the folder to the machine: `infer.py` inherits the size
(`--frame_resize`, which also works without a cache; `0` turns it off) but not the folder, so
pass `--frame_cache` again to use one, and it warns if the size differs from training. Interrupted or
concurrent writes never leave a readable partial entry. The cache has no size limit; delete
the folder to reclaim the space.

All of these options are saved in `cli_arguments.json`, so the desktop Test page and
`infer.py` rebuild the same pipeline. `scripts/bench_video_io.py` measures decode speed,
old vs new dataset, motion-vector vs Farneback cost, and an epoch from pixels vs the frame
cache vs cached features.

```powershell
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version r3d_18 --num_frames 16 --temporal_sampling segment --video_augment --test_clips 3
```

## Video architectures: how a model fuses time

`--video_arch` picks the architecture and `--model_version` its 2D backbone. The run records it
in `cli_arguments.json` (as `--pooling`), so evaluation, embedding and `infer.py` rebuild it.

| `--video_arch` | What it does | Options |
|---|---|---|
| `frame_pool` (default) | Pools per-frame embeddings, then classifies the pooled embedding. | `--pooling mean\|attention` |
| `single_frame` | Middle frame only. A baseline: it cannot see motion. | |
| `late_fusion` | First and last frames, embeddings merged by FC layers. | |
| `early_fusion` | All frames stacked on an inflated first convolution. ResNet family; the frame count is fixed at build time. With `--freeze_backbone` that convolution stays trainable: its inflated weights cannot tell frame order until trained. | |
| `slow_fusion` | A native 3D CNN: `--model_version` if it names one, otherwise `r2plus1d_18`. | |
| `tsm` | Temporal Shift Module in each residual block. ResNet family, no extra parameters. | `--tsm_fold_div` |
| `cnn_rnn` | Per-frame embeddings into an LSTM or GRU. | `--rnn_type --rnn_hidden --rnn_layers --bidirectional` |
| `temporal_transformer` | Per-frame embeddings into a small transformer with a CLS token. Passed about half its benchmark runs, see below. | `--tt_layers --tt_heads --tt_dim` |
| `two_stream` | RGB middle frame + stacked motion fields, fused late. ResNet family. With `--freeze_backbone` the motion stream's inflated first convolution stays trainable: frozen, it cannot separate dx from dy. | `--motion_source farneback\|mv --fusion avg\|fc --motion_length --motion_cache DIR` |

```powershell
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version resnet18 --video_arch tsm
.venv\Scripts\python.exe deeptune.py --modality video --df clips.csv --raw-data --out results --model_version resnet18 --video_arch two_stream --motion_source mv --motion_cache .artifacts\motion
```

- **Motion source.** `farneback` is OpenCV optical flow and works on any codec. `mv` reads codec
  motion vectors: it is cheaper but needs a codec that stores them, and clips of about 48 px or more.
  Fields are stored in pixels per frame. For the model, 5 % of the frame's size per frame maps to 1,
  clipped to [-1, 1]; no mean is subtracted. `--motion_cache` keeps them on disk, keyed by the video's
  hash and the parameters.
- **SigLIP.** `--model_version siglip` uses SigLIP as the frame backbone (with `frame_pool`,
  `single_frame`, `late_fusion`, `cnn_rnn`, `temporal_transformer`, and `--use-peft`), from the command
  line only. Its weights download on first use.
- **Old checkpoints.** Frame-pooling checkpoints from before embeddings were pooled load and predict
  as before (they keep pooling logits).
- **DenseNet embeddings.** On video, a DenseNet frame model with `--added_layers 1` now embeds to its
  features, not its class logits, so embeddings saved by earlier runs have a different shape. Weights and
  predictions are unchanged. The images embedder still refuses `added_layers` below 2 on DenseNet.
- **What the benchmark shows** (`tests/models/test_temporal_oracle.py`, untrained tiny backbone, CPU,
  held-out scenes): on one seed, `early_fusion`, `tsm` and `two_stream` learn the direction of motion from
  240 clips, as does `slow_fusion` (on an untrained r3d_18, not the default r2plus1d_18), and `single_frame`,
  `mean`, `attention` and `tsm` with its shift turned off stay at chance. Three models were also run over
  several seeds, counting a run as passed at 85 % held-out accuracy or more, with the reversal and frame-order
  checks. `cnn_rnn` passed 2 of 10 runs from 240 clips and 9 of 10 from 800. `temporal_transformer` passed 5 of 10
  from 1500 clips (`tt_dim=64`, learning rate 3e-4); the other runs stayed at 0.50-0.625, so the benchmark does not
  show that it reliably learns direction. `late_fusion` passed 7 of 10 from 800 clips, on a task at half the default
  speed only. At the default speed it passed 0 of 10 (best 0.585), and it cannot pass: on the benchmark's wrap-around
  world, no rule that sees only two frames can exceed about 0.663. This is evidence that the models are wired
  correctly, not a ranking of them or a statement of how much data they need. Real ResNets are not yet measured
  on these models.

## Video features: decode once, extract once

`extract_features.py` splits each video into snippets, runs a frozen backbone on each one
once and stores the result per video. Heads then train in minutes without decoding a frame.
Finished videos are skipped, so an interrupted run continues where it stopped.

```powershell
.venv\Scripts\python.exe extract_features.py --data clips.csv --out features --model_version r3d_18 --streams rgb motion
.venv\Scripts\python.exe train_feature_head.py --features features --train_df train.csv --val_df val.csv --test_df test.csv --out probe --num_classes 5 --head linear_probe
```

| flag | default | meaning |
|---|---|---|
| `--data` | required | CSV/XLSX manifest or parquet (paths or bytes); an optional `video_id` column names the files. Within one manifest an id cannot stand for two videos or trims. Ids may not differ only by case, across runs and streams too. A later run may give an id another video in one stream only; loading both streams then refuses, as does a row whose file or clip bytes are not the ones extracted, or a row of one kind against features from the other. Files are compared by path, size and modified time, so a touched or re-copied file needs a new extraction or a row with only a `video_id`, which is not compared at all; a file rewritten with the same size and modified time goes undetected |
| `--out` | required | feature folder; a folder holds one configuration per stream |
| `--streams` | `rgb` | `rgb` and/or `motion` |
| `--model_version` | `r3d_18` | `rgb` backbone: any DeepTune video or frame backbone |
| `--motion_backbone` | `r3d_18` | `motion` backbone; dx, dy and their magnitude go through its RGB filters, which were not trained on motion |
| `--motion_method` | `auto` | `codec` motion vectors, `farneback` flow, or `auto` (codec, else Farneback; recorded per video) |
| `--motion_gain` | 0.05 | multiplies motion after clipping to +-20 input pixels per frame; recorded in the config and sidecar. The default has not been chosen on real clips |
| `--feature_weights` | `pretrained` | `none` = fixed random weights, for tests and timing only |
| `--snippet_frames` / `--snippet_stride` | 16 / same | frames per snippet / between snippet starts |
| `--cache_tokens` | `none` | `none` = one vector `[N, D]`; `8x4x4` = pooled token grid `[N, 8, 4, 4, D]` |
| `--batch_size` | 8 | snippets per forward pass |
| `--feature_input_size` | backbone's own | square input size; keeps that backbone's normalisation and squashes the whole frame. ViT and `mvit_v2_s` accept only their own size (`mvit_v2_s` also only 16-frame snippets), and other values are refused up front |
| `--video_backend` | `auto` | decoder |

**Snippets.** `N = ceil((frames - snippet_frames) / stride) + 1` (1 for a short video), with the
last snippet padded by repeating the final frame. With a stride larger than the snippet, no
snippet starts after the last frame.

**Layout.** `<out>/rgb/<video_id>.npy` and `<out>/motion/<video_id>.npy` are float16 arrays, each
with a `.json` sidecar (fps, snippet_frames, stride, backbone, weights, the licence of exactly those
weights where torchvision states one, the input size really fed, the part of the frame kept
(`field_of_view`), stream, frame range and trim, and float16 rounding stats; motion also records
`motion_gain` and `motion_weights_trained_on_motion: false`). Motion is cut to the same part of the frame
as `rgb` (native presets centre-crop), so a cell of a token grid covers the same place in both streams;
streams that see different parts refuse to concatenate. `datasets.snippet_features.load_snippet_features(out,
video_id)` returns `[N, D_rgb + D_motion]` plus the split. A value that float16 cannot hold (NaN,
inf, above 65504) fails that video instead of being stored; failures go to `<out>/failed.json`, and
the command exits with status 1. Run it again to retry them.

**Training a head.** `train_feature_head.py` needs only the feature folder and manifests with `videos`
or `video_id` plus labels; the video files may be gone. It refuses splits that share an id, a source
video (two trims of one video count) or a value of the `group` column, and warns if a split has no
`group` column. It also refuses motion features from an older preprocessing version. `--test_df` writes
`metrics.json` (accuracy, balanced accuracy, macro-F1, confusion matrix; MSE, RMSE, MAE, R2 for regression;
a `chance` row for a model that only learned the training majority or mean) and per-video `predictions.csv`,
the input to a paired test. `cli_arguments.json` records the seed, the feature config hash and whether the
motion weights were trained on motion.

**Token grids and the attentive probe.** A mean-pooled vector throws away where and when something
happened. `--cache_tokens 8x4x4` keeps a coarse grid, and `train_feature_head.py --head
attentive_probe` (the default) reads it through a learned query (`src/video/feature_heads.py`). An
axis is capped at what the backbone produces, for example `r3d_18` on 16 frames gives at most
`2x7x7`, and the sidecar records the real grid. A grid costs storage:

| stored per 16-frame snippet (`r3d_18`, 512-d) | size |
|---|---|
| vector | 1 kB |
| `2x4x4` | 33 kB |
| `2x7x7` (unpooled) | 100 kB |

Whether the finer grid pays off in accuracy depends on the data, so measure it on yours.

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
`expected_error` column. `error_model_fit.json` also records a fingerprint
(`weights_sha256`) of the weights the error model was fitted on. If eval or
inference loads different weights, DeepTune prints a warning and reports no
`expected_error`, rather than a number that belongs to other weights.

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
- `--cross_label_nms 0.7`: when boxes of *different* prompts overlap (Grounding DINO
  often finds one object for both "lobster" and "crab"), keep only the best-scoring
  one, so the object is counted once. This is off by default, so such an object is
  counted under each prompt.

Run `python track.py --help` for the full list. Bad arguments stop the run before
any video is read or weights load (exit code 2). The exit code is 1 when any video
fails; failed videos are listed in `<out>/errors.json`. Prompts may contain dots
("St. Bernard"). `scripts/smoke_track.py` runs both detectors with real weights on
a generated clip.

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

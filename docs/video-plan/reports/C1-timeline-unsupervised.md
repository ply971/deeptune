# C1: timeline (Phase 4 Part A) and unsupervised analysis (Phase 4 Part C)

Branch `video/C1-timeline-analysis`, from `video-analysis` @ 2fd72a4. Built overnight on 2026-11-12
("finish the project"). CtxPL (Part B) is not in this branch.

## What was built

| Piece | Files | What it does |
|---|---|---|
| Timeline | `analysis/timeline.py` | Windows over a video of any length (`make_windows`; the last one ends at the video's end); running-median smoothing per class (`median_smooth`); each window owns the time nearest its centre, so segments partition the video (`owned_intervals`); argmax labels, merging equal neighbours, absorbing segments shorter than `min_segment_sec` into the longer neighbour, dropping background labels (`segments_from_timeline`); optional skipping of still windows from the motion curve (`static_windows`, SCSampler-style); `timeline.csv` / `segments.json` per video, `segments.csv` + `timeline_run.json` for a batch; mAP and boundary error with `evaluators/video/localization_report.py` when ground truth is given. Torch-free; the scorer is injected. |
| CLI | `infer.py --timeline` | `--window_sec`, `--stride_sec`, `--smooth_windows`, `--min_segment_sec`, `--background_label`, `--min_confidence`, `--skip_static`, `--timeline_motion`, `--ground_truth`. Windows are DataFrame rows with `start_sec`/`end_sec`, scored by `predict_video`, so the run's recorded sampling, frame count, preprocessing, test clips and crops apply unchanged. All flags are checked before any model or video loads (exit 2); timeline flags without `--timeline` are refused. |
| Segment-level training labels | `handlers/video_manifest.py`, `handlers/split_dataset.py` | A manifest may now list one file on several rows when each row has its own finite `start_sec` < `end_sec` range and all rows of a file share one `group`. Before, any repeated file was refused, which made Part A item 4 impossible. The old refusal still applies without ranges (its test is unchanged). Every split path (train/val/test, `--cv_only`, `--test_df`) refuses repeated video files whose clips have different values of the grouper in use (e.g. `--grouper actor`), and the group-split check also asserts no video file is in two splits (review V-061). Without a grouper a Parquet table with repeated sources is still split by row with the existing warning (stdout and evaluation report), as an existing test requires. |
| Unsupervised | `analysis/unsupervised.py`, `analyse_embeddings.py` | PCA (explained variance, components for 90%), LDA when labelled, K-Means sweep (k-means++, restarts; inertia elbow + silhouette, suggested k = best silhouette), agglomerative clustering (ward/average/complete/single) with the SciPy linkage matrix, ARI/NMI against labels, and a small MLP autoencoder for anomaly scores with a threshold at the `1 - target_fpr` quantile of held-out validation scores; known outliers (`--outlier_label`) are never used for fitting and give AUC. Roles are drawn by whole `--group_column` groups (default `group` when present), and the test false-alarm rate comes with a Clopper-Pearson interval. CSV/JSON + PNG plots. |
| GUI | `desktop/cli_options.py`, `config.py`, `tools.py`, `tool_pages.py`, `app.py` | Test page: a **Timeline** section (every flag), the same checks as `infer.py`, and a results view (segments table, mAP/boundary error, per-video windows). Features & probes page: card **3. Cluster clips and find unusual ones** (every `analyse_embeddings.py` flag), results listing the most unusual clips first. The packaged app runs it through its tool worker. The GUI still imports no torch. |
| Experiment | `scripts/timeline_untrimmed.py` | Phase 2 untrimmed task end to end: training windows cut from the training videos only, `deeptune.py` training, `infer.py --timeline` on held-out videos, compared with a random control and a pixel oracle; skip-static; clustering of the run's test embeddings. |

## Results

### Timeline on the synthetic untrimmed task

**Not measured yet with a real model.** The planned run (`scripts/timeline_untrimmed.py --model r3d_18
--epochs 6`, 40 videos, on the GPU) was stopped by Claude Code at the start of training because the machine
was critically low on memory (15 GB total; other programs held most of it). It was not restarted. The script
now reuses only a finished run, so it can simply be run again:

    python scripts/timeline_untrimmed.py --out D:/DeepTuneBenchmarks/results/c1-timeline-untrimmed01 --model r3d_18 --epochs 6 --extra --num_workers 2

What exists is a **pipeline check on the CPU** (8 videos, an untrained `dummy_video3d` trained for one epoch on
218 windows, 2 held-out videos with 6 actions; window 1.6 s, stride 0.4 s, median 3, min segment 0.8 s). It shows
the whole path works (training on time ranges, `infer.py --timeline`, scoring, skip-static, clustering); the
numbers say nothing about how good a timeline can be:

| Method | mAP@0.5 | avg mAP 0.1-0.7 | boundary error (s) | windows scored | seconds |
|---|---|---|---|---|---|
| model (1 epoch, tiny) | 0.278 | 0.310 | 0.35 | 129/129 | 24.8 |
| model + `--skip_static 0.5` | 0.278 | 0.310 | 0.35 | 65/129 | 12.9 |
| random control | 0.000 | 0.019 | - | 129/129 | - |
| pixel oracle (sees the blob, always says the majority class) | 0.167 | 0.167 | 0.05 | 129/129 | - |

Skip-static halved the scoring time with identical segments on these videos, because half the windows were
still background. The oracle tests in `tests/test_timeline.py` (mAP@0.5 >= 0.9, boundary error <= one stride
with a pixel-reading scorer) check the segmentation itself.

### Clustering real clip embeddings: UCF101 split 1, r3d_18

`analyse_embeddings.py` on the 3,783 test clips of the existing `ucf101-campaign03/r3d18_frozen/split1/seed42`
run, k from 90 to 110, 4 restarts, seed 0, fit/val/test roles drawn by UCF101 `group` (actor + scene; 707
groups, none in two roles). Command:

    python analyse_embeddings.py --embeddings <run>/embed_output_r3d_18_20261104_1824/r3d_18_cls_embeddings.parquet \
        --out D:/DeepTuneBenchmarks/results/c1-unsupervised-ucf101-r3d18-groups --k_min 90 --k_max 110 --n_init 4 --seed 0

**What the embedding is.** The 256 numbers per clip are the output of the first layer DeepTune added on top of a
frozen Kinetics-400 r3d_18, and that layer was **trained on UCF101 split-1 training labels** (2 added layers,
157,285 trainable parameters). So only the clustering step uses no labels; the features were shaped by the same
101 classes the clusters are compared with. This measures how class-structured a trained embedding is, not
label-free discovery.

| | Value |
|---|---|
| PCA: variance in the first two components | 16.6% |
| K-Means vs labels (k = 99): ARI / NMI | 0.536 / 0.819 |
| Ward hierarchical (k = 99) vs labels: ARI / NMI | 0.581 / 0.851 |
| Chance level (random cluster labels, k = 99, 20 draws): ARI / NMI | 0.000 / 0.259 |
| LDA training accuracy (fitted and scored on the same clips: an upper bound, not a held-out accuracy) | 0.983 |
| Autoencoder: false alarms on clips of unseen groups at a 5% target | 5.5% (43 of 775; 95% Clopper-Pearson 4.0-7.4%; one seed) |

**The suggested k is not a finding.** The k range 90-110 was chosen knowing there are 101 classes, and the
silhouette is flat across it (0.131-0.143): the suggested k was 109 in a first run with random row roles and 99
here, where only the rows used to standardise changed. Read the ARI/NMI as "at a k near the number of classes".

**False alarms.** The threshold is the 95% quantile of the validation groups' scores; on clips of groups the
autoencoder never saw it flags 5.5%, within sampling error of the 5% target. A first run that drew roles per clip
(so 98.7% of test clips shared a group with fitted clips) gave 4.5% (34/756); that number described new clips of
seen actors and scenes, so it is replaced. No outliers are known in UCF101, so there is no detection rate here; on
synthetic data with injected outliers the tests require AUC >= 0.95. Outputs:
`D:/DeepTuneBenchmarks/results/c1-unsupervised-ucf101-r3d18-groups/`.

## Tests

Full CPU suite at 1bdf491 (`CUDA_VISIBLE_DEVICES=-1`, `-m "not seed_study"` as in `check.ps1`, run in nine
chunks because long background jobs were being stopped for low memory): **2246 passed, 5 skipped, 0 failed**
(sum of the nine JUnit files in `.artifacts/c1-suite/`). `pip check`: no broken requirements.

- `tests/test_timeline.py` (29): windows (cover, last-window shift, short videos, bad arguments); owned
  intervals partition the video; smoothing on hand-written sequences (flicker removed, real changes kept,
  width 1, even widths refused); segments (single window, all one class with mean confidence, flicker
  absorbed, longer neighbour wins, background dropped but still separating, no overlaps on random labels);
  chunked scoring, shape checks, `min_confidence`, static skipping; outputs. **Oracle:** synthetic untrimmed
  videos with a stub that reads the window's pixels: mAP@0.5 >= 0.9 and start/end boundary error <= one
  stride; every scoring call decodes at most one window of frames. Skip-static on still-background videos:
  fewer windows scored, mAP unchanged. The real clip pipeline decodes only each window's frames
  (`get_frames` spied). `infer.py --timeline` end to end with a trained `dummy_video3d` run, and every
  flag check before loading.
- `tests/test_unsupervised.py` (17): K-Means and agglomerative recover blobs (ARI >= 0.9); silhouette picks
  k = 2, 3, 5; elbow on a hand-made curve; PCA explained variance on data of known structure; LDA;
  threshold and AUC on hand-made scores; outliers kept out of fitting; seeded autoencoder; injected
  out-of-distribution clips found (AUC >= 0.95); CLI end to end and its argument checks.
- `tests/test_video_manifest.py` (+5): ranges of one video in one group accepted; repeated ranges,
  missing or split groups, and repeats without ranges refused.
- `tests/test_desktop_timeline.py` (6), `tests/test_desktop_embedding_analysis.py` (2), and
  `tests/test_gui_coverage.py` now covers `infer.py`'s timeline flags (tried with the timeline on, as
  deeptune prerequisites are) and every `analyse_embeddings.py` flag.

## Decisions and limits

- **Windows are scored with the trained clip pipeline**, not a separate decoder, so a timeline sees
  exactly what training saw. Cost: `predict_video` builds the model once per video.
- **Smoothing is a median over windows**, not over time; with a short stride it spans less time.
- **Skip-static thresholds are in the units of the motion signal** (codec motion-vector magnitude, or a
  0-255 frame difference); there is no automatic threshold.
- Timeline window **embeddings are not exported yet**, so anomaly scores are per clip, not per window
  (the plan's "per timeline window" anomaly item is open).
- **CtxPL (Part B) is not here.** The timeline is the baseline it will be compared with; both write the
  same `segments.csv` columns.

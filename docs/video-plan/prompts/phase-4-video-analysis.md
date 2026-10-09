# Phase 4 — Long-video & unsupervised analysis

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 4 only. Show your plan first.

## Goal
Go from "one label per clip" to *analysing* video: what happens when, which
videos look alike, and which ones are unusual.

## Part A — Timeline analysis of long videos
1. `analysis/timeline.py`: sliding-window inference over a video of any
   length using Phase 1 `dense` sampling (`--window_sec`, `--stride_sec`),
   batched, streaming (never decode the whole video into memory).
   Output per window: start/end time, class probabilities (or regression
   value), embedding (optional).
2. Smoothing + segmentation: moving average / median filter or HMM-style
   min-duration smoothing (`--min_segment_sec`), then merge consecutive
   windows into **event segments** `{start, end, label, mean_confidence}`.
   Export `timeline.csv`, `segments.json`.
3. Optional segment-level labels for training: annotation CSV
   `video,start_sec,end_sec,label` → training clips cut from those ranges
   (uses Phase 1 `start_sec/end_sec`).
4. CLI: `infer.py --modality video --analyse_timeline ...` (or a new
   `analyse.py`), recorded in run folder.
5. Synthetic check: extend the Phase 2 generator with
   `make_long_video(events=[(label, duration), ...])` that concatenates motion
   segments with known boundaries.

## Part B — Unsupervised analysis on clip embeddings
Use embeddings from any trained or pretrained video model (existing embed
step). In `analysis/unsupervised.py`:
1. Dimensionality reduction: PCA (explained variance), LDA when labels exist;
   optional t-SNE/UMAP if already installable (don't add heavy deps silently —
   ask first).
2. Clustering: K-Means (k-means++, multiple restarts, elbow + silhouette to
   suggest k) and agglomerative hierarchical clustering (Ward/average/complete,
   dendrogram data). If labels exist, report ARI/NMI vs labels.
3. **Video anomaly detection:** small autoencoder on clip embeddings trained
   on "normal" clips; reconstruction-error threshold chosen on validation
   (percentile or target false-positive rate); per-clip and per-window
   (timeline) anomaly scores.
4. Outputs: CSV/JSON + PNG plots in the run folder (no GUI work yet).

## Tests
- Timeline: on a synthetic long video with known boundaries, segment boundary
  error ≤ one stride and label accuracy over time above a set threshold, using
  a model trained on the Phase 2 task (or a deterministic stub model for the
  pure segmentation logic unit tests).
- Streaming: peak memory doesn't grow with video length (test with a long
  generated video vs short; assert decode called per window).
- Smoothing/merging unit tests with hand-written probability sequences
  (edge cases: single window, all same class, flicker).
- Clustering: on synthetic embeddings with known blobs, K-Means/hierarchical
  recover them (ARI ≥ 0.9); silhouette picks the right k.
- Anomaly: inject out-of-distribution clips (e.g. different motion pattern);
  AUC of anomaly score ≥ threshold.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1`, paste the summary, attach an
example `segments.json` and plots from the synthetic long video. If anything
fails twice, stop and report.

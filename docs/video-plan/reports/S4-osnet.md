# S4: a person Re-ID model (OSNet-AIN) for tracking — no measurable gain on MOT17 dev

Branch `video/S4-osnet-reid` (from `video/S3-botsort` @ c359eb3), 2026-11-12/13, after the user approved
the download ("OK for all"). **Not merged**: the development videos show no gain, so it stays an option on
a branch until someone decides it is worth shipping.

## What was built

- `spatial/osnet.py`: OSNet-AIN x1.0 (Zhou et al., TPAMI 2021), ported from Torchreid (MIT, notice kept),
  classifier removed; module names match upstream so the released weights load with `strict=True`
  (2,169,508 parameters, every tensor matched).
- `spatial/reid.py`: `--reid osnet` (`OsnetEmbedder`, 256x128 crops, ImageNet normalisation, 512-d). Weights:
  `LibreYOLO/LibreReID-osnet` @ 98f8d865, `osnet_ain_x1_0.pt` (8.9 MB, MIT; converted unchanged from the
  Torchreid multi-source MSMT17 + DukeMTMC-reID + CUHK03 model), SHA-256 checked before loading. Never chosen
  by `auto`: the training datasets carry research-use terms. No MOT17 in its training data.
- `track.py` help, GUI choice, `scripts/benchmark_mot17_botsort.py --reid cnn|osnet` (keys get `_osnet`).
- Tests (`tests/spatial/test_spatial_reid.py`): 512-d non-negative output; own state dict reloads strictly and
  a missing tensor fails; wrong checksum refused at the pinned revision; 256x128 crops reach the model.
  Two existing tests used `osnet` as their example of an *unknown* Re-ID name; now that it is valid they use
  `resnet50` (same check: an unknown name is refused).
- Not verified bit-exact against Torchreid (not installed; installing needs the user's OK).

## Results

**Crop check** (MOT17 dev 02/04/05 ground-truth boxes, visible >= 0.6, height >= 40 px; 352 crops; same-ID
pairs from different frames vs different-ID pairs in the same video; ROC AUC of cosine similarity):
ResNet-18 ImageNet 0.895, OSNet-AIN 0.896, HSV colour histogram 0.929. Feeding BGR instead of RGB, or other
feature layers, changed OSNet's AUC by at most 0.01, so this is not an input-format slip.

**Tracking, MOT17 dev videos 02/04/05, public detections, TrackEval** (`D:/DeepTuneBenchmarks/results/mot17-osnet-dev01`,
code 0887ccf; embeddings cached per model, everything else identical):

| Detections | Variant | HOTA cnn -> osnet | IDF1 cnn -> osnet | ID switches cnn -> osnet |
|---|---|---|---|---|
| FRCNN | Re-ID only | 47.92 -> 47.92 | 56.36 -> 56.37 | 140 -> 140 |
| FRCNN | botsort | 49.36 -> 49.39 | 57.91 -> 57.94 | 119 -> 113 |
| SDP | Re-ID only | 55.02 -> 55.46 | 68.43 -> 69.22 | 326 -> 328 |
| SDP | botsort | 57.17 -> 57.35 | 70.50 -> 70.83 | 243 -> 230 |
| DPM | Re-ID only | 32.37 -> 31.84 | 40.90 -> 40.28 | 274 -> 280 |
| DPM | botsort | 34.27 -> 33.70 | 42.62 -> 41.97 | 157 -> 176 |

Within noise for FRCNN and SDP, slightly worse with DPM's loose boxes. By the dev-only rule this does not
justify a new default, so the held-out videos were **not** run (there is no decision left for them to check).

## Why so little, and what might help

- In the fused BoT-SORT cost, appearance only counts for boxes that already overlap (IoU >= 0.5) and look
  alike within `appearance_thresh` 0.25. On the crop check, 80% of *different*-person pairs are already under
  0.25 with OSNet (85% with ResNet-18): the gate rarely separates anyone, so swapping the model changes little.
  BoT-SORT's 0.25 was set for FastReID features trained on the MOT data itself.
- Next step if wanted: sweep `--appearance_thresh` / `--link_appearance_thresh` per model on the dev videos
  (cached embeddings make it cheap), then run the held-out videos once with the chosen values.
- Colour histograms separated people best on the crop check (0.929) and are what a fresh install gets; they
  were never run through the tracking benchmark (also flagged in review V-062).

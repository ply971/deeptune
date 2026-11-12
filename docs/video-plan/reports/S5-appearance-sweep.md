# S5: appearance thresholds and colour Re-ID on MOT17 — the defaults stay

Branch `video/S5-appearance-thresh` (from `video-analysis` @ ed50e14), 2026-11-13. Follow-up to S4 (OSNet gave
no gain) and review V-062 ("run `--reid color` once"). `scripts/benchmark_mot17_botsort.py` gained
`--reid cnn|color`, `--appearance_thresh` and `--link_appearance_thresh` (keys get `_color`, `_a<x>`, `_l<x>`).
Public detections, TrackEval, cached ResNet-18 vectors and camera-motion affines from
`D:/DeepTuneBenchmarks/results/mot17-botsort01/cache` (colour caches reuse the same affines).
Decisions on the dev videos 02/04/05 (`mot17-appearance-dev01`); one held-out check on 09/10/11/13
(`mot17-appearance-heldout01`). The defaults reproduce the S3 numbers exactly (49.36 / 57.17 / 34.27 dev HOTA).

## Dev sweep (full `botsort`, HOTA; FRCNN / SDP / DPM)

| Setting | FRCNN | SDP | DPM |
|---|---|---|---|
| default (tracker gate 0.25, link gate 0.3, ResNet-18) | 49.36 | 57.17 | 34.27 |
| tracker gate 0.05 | 49.26 | 56.61 | 34.71 |
| tracker gate 0.15 | 49.36 | 57.17 | 34.30 |
| tracker gate 0.35 or 0.5 | 49.36 | 57.17 | 34.27 |
| link gate 0.1 | 50.18 | 56.46 | 34.53 |
| link gate 0.2 | 49.20 | 57.17 | 33.49 |
| link gate 0.5 | 49.36 | 57.17 | 34.27 |
| colour histograms instead of ResNet-18 | 49.60 | 57.75 | 34.31 |

- **The tracker's gate barely matters.** BoT-SORT's fused cost is min(IoU cost, appearance cost): appearance
  can only make a match cheaper, never veto one, so tightening the gate only removes help and loosening it past
  0.25 changes nothing here.
- **The link gate is noise-level.** 0.1 helps FRCNN (+0.8) and hurts SDP (-0.7); 0.2 is worse than both 0.1 and
  0.3 for two streams. No consistent direction, so no change.
- **Colour was at least as good as ResNet-18 on all three dev streams** (+0.04 to +0.58), so it went to the
  held-out check.

## Held-out check (full `botsort`, ResNet-18 vs colour, same code and run)

| Detections | HOTA | IDF1 | ID switches |
|---|---|---|---|
| FRCNN | 48.56 -> 48.57 | 55.65 -> 55.72 | 310 -> 305 |
| SDP | 54.07 -> 54.85 | 64.59 -> 65.38 | 219 -> 219 |
| DPM | 34.73 -> 33.58 | 43.84 -> 42.41 | 108 -> 120 |

Mixed (+0.8 to -1.2), 4 videos, no interval: the two appearance models are about equal on MOT17. That answers
V-062's question: a fresh install, where `--reid auto` falls back to colour histograms, is not measurably worse.
Nothing about the defaults changes.

## What this means for "better tracking"

With S3, S4 and S5 together, the appearance side (which model, which thresholds) moves MOT17 HOTA by about
+-1 point at most, while the detector moved it by 4.5 (Grounding DINO -> D-FINE, end to end, same tracker).
Further tracking gains are most likely in detection (DetA): a stronger or fine-tuned detector for the user's
own footage, or a higher `--box_threshold` against false tracks, rather than in appearance features.

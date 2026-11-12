# S3: better tracking (BoT-SORT parts, offline repair, D-FINE detector)

Branch `video/S3-botsort`. The user asked for this overnight on 2026-11-12: "the tracking ... is not that
good ... it's not using everything and the models aren't good enough". Two problems, two fixes:

1. **The tracker used only box overlap.** It ignored camera motion and appearance, and it never
   repaired a broken track, even though the whole video is available before anything is written.
2. **The detector was the weak part.** Open-vocabulary Grounding DINO (tiny) was the default even for
   everyday objects such as people and cats.

## What changed

| Part | File | What it does |
|---|---|---|
| Camera-motion compensation | `spatial/cmc.py` | Optical flow (Shi-Tomasi corners, pyramidal Lucas-Kanade) on the background, with detection boxes masked out; a RANSAC partial-affine fit; every predicted track (Kalman mean and covariance) moves with the camera. Identity when there is too little texture. From BoT-SORT (Aharon et al., 2022). |
| Appearance (Re-ID) | `spatial/reid.py`, `spatial/track.py` | One appearance vector per box: ImageNet ResNet-18 features (`--reid cnn`; `auto` uses them when the weights are already cached) or an HSV colour histogram (`color`, no torch). EMA track features (alpha 0.9). Association cost = min(IoU distance, appearance distance) where the boxes overlap enough (`proximity_thresh` 0.5) and look alike (`appearance_thresh` 0.25), as in BoT-SORT. Without vectors the tracker is exactly the old ByteTrack (existing tests unchanged). |
| Offline repair | `spatial/postprocess.py` | After the whole video: drop tracks shorter than `--min_track_len` (3); join a track that ends to one that starts within `--link_max_gap_s` (3 s) when the jump fits the motion, the sizes match, the appearance matches and the pair is **unambiguous** (exactly one candidate either way); fill gaps up to `--interpolate_max_gap_s` (1 s) by linear interpolation; write each track's first box (ByteTrack confirms a track one frame late); renumber. Linking follows the tracklet-association idea of StrongSORT's AFLink (Du et al., 2023), with hand-written gates instead of a learned model. |
| Detector | `spatial/detect.py` | **D-FINE-L** (Objects365 → COCO, `ustc-community/dfine-large-obj2coco-e25`, Apache-2.0, 0.13 GB) for the 80 COCO classes, with plurals and synonyms (people, kitten, ...). `--detector auto` (new default) picks D-FINE when every prompt is a COCO class and Grounding DINO otherwise. |
| CLI | `track.py` | `--tracker botsort` (default) or `bytetrack` (old behaviour); each part can also be switched on its own: `--cmc`, `--reid`, `--appearance_thresh`, `--proximity_thresh`, `--link_max_gap_s`, `--link_appearance_thresh`, `--interpolate_max_gap_s`, `--min_track_len`. `summary.json` gains a `tracking` block (parts used, links made, tracks dropped, boxes filled). `track_arguments.json` records what `auto` resolved to. |
| GUI | `desktop/analyse.py`, `desktop/analyse_page.py`, `desktop/cli_options.py` | D-FINE in the detector list (Grounding DINO stays first, so nothing changes for users who did not choose); a tip appears when an open-vocabulary detector is used for COCO objects; every new flag is under "More tracking options" (GUI coverage test). |

## How it was measured

`scripts/benchmark_mot17_botsort.py` scores trackers with the official **TrackEval** (pinned commit
12c8791) on the labelled MOT17 training videos. Detections are the MOT17 public ones (FRCNN, SDP, DPM),
so only the tracker differs. Decisions were made on videos **02, 04, 05** (dev); the numbers that count
are the four **held-out** videos **09, 10, 11, 13**, run once after the decisions were fixed.
Appearance vectors are ResNet-18 ImageNet features (`--reid cnn`); camera motion is computed on the real frames.
A fresh install without the ResNet-18 weights gets `--reid auto` = colour histograms, which were **not
measured** here.

**Disclosure (review V-062).** The held-out videos were not untouched before the design was final. The
first run (`D:/DeepTuneBenchmarks/results/mot17-botsort01`, before the `--sequences` option existed) scored
all 7 videos, held-out ones included, with the first, permissive linking: botsort 45.81 vs ByteTrack 47.28
HOTA (FRCNN, all 7). Linking was then made unambiguous. The development videos alone showed the same harm
(table below: IDF1 56.6 -> 49.6), so the decision does not depend on the held-out videos, but "decided on dev
only" is not literally true. No later setting was changed after a held-out number was seen.

One protocol detail: the new pipeline clips boxes to the frame. The old benchmark wrote boxes partly
outside the frame, which moved the old ByteTrack baseline from 47.36 to 47.28 HOTA (FRCNN, **all 7 videos**,
not dev; the dev baseline is 47.84). All comparisons below use the clipped baseline, so the only difference
is the tracker.

### Development set (02, 04, 05; FRCNN detections)

| Variant | HOTA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|
| bytetrack (old) | 47.84 | 56.27 | 47.82 | 158 |
| + camera motion | 47.95 | 56.49 | 47.93 | 133 |
| + Re-ID only | 47.92 | 56.36 | 47.87 | 140 |
| online (camera + Re-ID) | 48.06 | 56.63 | 47.94 | 133 |
| online + interpolation | 49.04 | 57.53 | 49.40 | 123 |
| online + linking, **first version (permissive)** | 45.69 | 49.57 | 48.00 | 88 |
| online + linking, **unambiguous** | 48.06 | 56.62 | 47.94 | 130 |
| **botsort, final** (all parts, unambiguous linking) | **49.36** | **57.91** | **49.66** | **119** |

The first linking rule joined every end to its cheapest start. In crowds that joins different people:
fewer ID switches (88) but IDF1 fell by 7 points. **Decision:** only link pairs with no rival
candidate either way, and drop short tracks before linking. In crowds the step then does almost
nothing (it cannot tell people apart), which is the safe failure.

### Held-out set (09, 10, 11, 13): run once, after the decisions

| Detections | Tracker | HOTA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|
| FRCNN | bytetrack (old) | 46.23 | 53.14 | 51.86 | 483 |
| FRCNN | **botsort** | **48.56** | **55.65** | **53.18** | **310** |
| SDP | bytetrack (old) | 50.10 | 59.71 | 61.48 | 460 |
| SDP | **botsort** | **54.07** | **64.59** | **64.16** | **219** |
| DPM | bytetrack (old) | 30.34 | 38.50 | 31.71 | 186 |
| DPM | **botsort** | **34.73** | **43.84** | **32.88** | **108** |

Per-part ablation on the held-out set (HOTA / IDF1 / ID switches):

| Variant | FRCNN | SDP | DPM |
|---|---|---|---|
| bytetrack | 46.23 / 53.14 / 483 | 50.10 / 59.71 / 460 | 30.34 / 38.50 / 186 |
| + camera motion | 47.44 / 54.67 / 356 | 53.21 / 64.00 / 320 | 33.05 / 42.05 / 139 |
| + Re-ID only | 46.37 / 53.39 / 441 | 50.90 / 60.97 / 434 | 31.09 / 39.77 / 214 |
| online (camera + Re-ID) | 47.65 / 55.07 / 337 | 52.86 / 63.78 / 266 | 32.99 / 42.33 / 162 |
| online + interpolation | 48.56 / 55.65 / 306 | 54.02 / 64.66 / 220 | 34.62 / 43.77 / 108 |
| online + linking | 47.39 / 54.73 / 336 | 52.76 / 63.60 / 265 | 32.98 / 42.30 / 162 |
| online + first box | 47.78 / 55.18 / 348 | 52.96 / 63.87 / 274 | 33.17 / 42.55 / 166 |
| online + min length | 47.65 / 55.08 / 327 | 52.88 / 63.79 / 260 | 33.00 / 42.33 / 160 |
| offline parts only (no camera, no Re-ID) | 46.74 / 53.24 / 482 | 50.96 / 60.13 / 388 | 31.86 / 40.12 / 149 |
| **botsort (all)** | **48.56 / 55.65 / 310** | **54.07 / 64.59 / 219** | **34.73 / 43.84 / 108** |

What this says:
- **Camera-motion compensation is the largest single gain** (+1.2 to +3.1 HOTA): MOT17 has moving
  cameras (10, 11, 13), and a predicted box that does not move with the camera misses its person.
- **Interpolation** adds about +1 HOTA everywhere: it fills the frames where a person was missed.
- **Re-ID alone is small** on pedestrians (+0.1 to +0.8): many people in dark clothes look alike to an
  ImageNet network that was not trained for Re-ID. It helps more together with camera motion.
- **Linking is about neutral on pedestrians** (−0.3 HOTA on FRCNN, −0.1 SDP, 0.0 DPM). It is kept because
  it is the part that fixes the user's case (one animal leaving and coming back, below) and it is
  unambiguous by construction; in crowds it does nothing. It can be turned off (`--link_max_gap_s 0`).
- No part was tuned on the held-out videos.

### End to end: DeepTune's own detectors on the raw frames

`scripts/benchmark_mot17_e2e.py` runs each `--detector` with prompt `person` on the raw held-out frames
(same four videos), then each tracker with exactly the settings `track.py` resolves for that pair.
TrackEval scores only pedestrians (sitting people, reflections etc. are removed as distractors).

| Detector | Tracker | HOTA | DetA | AssA | IDF1 | MOTA | ID switches | Detector frames/s |
|---|---|---|---|---|---|---|---|---|
| Grounding DINO tiny | bytetrack (old default) | 42.0 | 38.1 | 46.7 | 49.9 | 41.3 | 137 | 3.3 |
| Grounding DINO tiny | botsort | 43.3 | 39.6 | 47.5 | 50.6 | 40.9 | 79 | 3.3 |
| OWLv2 base | bytetrack | 39.0 | 39.0 | 39.4 | 48.1 | 37.2 | 317 | 2.0 |
| OWLv2 base | botsort | 41.1 | 39.5 | 43.1 | 49.6 | 31.8 | 203 | 2.0 |
| D-FINE-L | bytetrack | 44.5 | 44.2 | 45.5 | 54.0 | 46.6 | 221 | 20.8 |
| D-FINE-L | botsort (**new default**) | **46.5** | 45.3 | 48.2 | **56.2** | 46.5 | 179 | 20.8 |

The old default was Grounding DINO + ByteTrack; the new default for COCO prompts is D-FINE + BoT-SORT:
**HOTA 42.0 -> 46.5, IDF1 49.9 -> 56.2, MOTA 41.3 -> 46.5, and a detector 6x faster** (20.8 vs 3.3 frames/s on
the RTX 5060 laptop GPU, batch 4).

- **The detector is the bigger part of the gain:** D-FINE with the *old* tracker already beats Grounding DINO
  with the new one (44.5 vs 43.3 HOTA). DetA rises from 38-39 to 44-45.
- **BoT-SORT raises HOTA and IDF1 with every detector** (+1.2 to +2.1 HOTA, +0.7 to +2.2 IDF1, 20-43% fewer
  ID switches) **but not MOTA**: flat for D-FINE (46.6 -> 46.5) and Grounding DINO (41.3 -> 40.9), and down
  for OWLv2 (37.2 -> 31.8). Interpolation and the first-box emission also extend tracks that are false
  positives: OWLv2's false positives rise from 7,337 to 11,516. MOTA counts every false box; HOTA and IDF1
  weigh identity, which is what a count of objects depends on. With a noisy detector, raise
  `--box_threshold` or set `--interpolate_max_gap_s 0`.
- Prompted with `person`, every detector also finds people that MOT17 does not label as pedestrians
  (sitting, reflections), which TrackEval removes only where they overlap a distractor box.

### The cat videos (observation, no labels)

The two cat clips used before (Pexels / Wikimedia, attribution files beside them). Counts are from
`summary.json`; the contact sheets were checked by eye.

| Setup | black-cat-walking | cat-play |
|---|---|---|
| trained cat detector + bytetrack (old) | 3 IDs | 3 IDs |
| trained cat detector + botsort | 1 ID (404 of 453 frames) | 2 IDs (one extra 10-frame box around the basin and the cat) |
| `--detector auto` (D-FINE) + botsort | **1 ID on all 453 frames** | **1 ID on all 1139 frames** |

## Tests

New: `tests/spatial/test_spatial_cmc.py` (a known camera translation is recovered, in full-resolution
pixels when downscaled; detection boxes are masked; identity on the first and textureless frames; one ID
through sudden camera pans), `test_spatial_reid.py` (vector shape and norm; degenerate boxes; `auto` picks
the CNN only when its weights are cached; appearance keeps the identity on a lookalike, but cannot join
far-apart boxes; EMA features), `test_spatial_postprocess.py` (linking rules incl. rivals either way, far
jumps, sizes, labels, appearance veto, chains; interpolation on the sampled grid; short tracks; majority
labels; renumbering), `test_track_botsort_cli.py` (presets vs ByteTrack on a long miss, birth frame, false
positives dropped, records, usage errors), `test_spatial_dfine.py` (stub model: exact pixel boxes and
labels, the 80 COCO classes and synonyms, `auto`), `tests/test_desktop_dfine.py`. The tracker's old tests are unchanged and pass: without appearance
vectors and camera motion, `ByteTracker` is the same code path.

Full CPU suite at 7181c51 (`CUDA_VISIBLE_DEVICES=-1`, `-m "not seed_study"` as in `check.ps1`, in nine chunks): **2261 passed, 5 skipped, 0 failed** (sum of `.artifacts/s3-suite/chunk-*.xml`); `pip check` clean.

### Held-out, per video (HOTA, ByteTrack -> BoT-SORT)

| Video | FRCNN | SDP | DPM |
|---|---|---|---|
| 09 | 47.2 -> 49.0 | 47.3 -> 50.2 | 35.1 -> 38.5 |
| 10 | 39.7 -> 42.7 | 45.6 -> 52.9 | 28.8 -> 33.6 |
| 11 | 53.7 -> **53.0** | 55.9 -> 59.6 | 38.3 -> 43.2 |
| 13 | 45.8 -> 50.5 | 50.7 -> 52.1 | 19.8 -> 23.7 |

11 of 12 pairs improve; MOT17-11 with FRCNN detections loses 0.7. The three detection streams share the same
four videos, so this is **4 independent videos with no confidence interval**; a one-sided sign test over the
videos gives p = 0.0625 (all four improve on average). Read the gains as consistent but not established.

## Limits

- MOT17 is pedestrians from street cameras. Animals, sports and handheld phone video can behave
  differently; the cat clips are two videos, not a benchmark.
- Re-ID uses ImageNet features, not a Re-ID-trained network (none with a clean licence was cached;
  downloads need the user's OK). A pedestrian Re-ID model (e.g. OSNet) would likely help crowds.
- The end-to-end table compares each detector's own `track.py` defaults (thresholds, input resolution):
  defaults against defaults, not matched settings.
- Gap filling and first-box emission add false positives (held-out FRCNN false positives 1,406 -> 3,217).
- Long videos (review V-062): the offline pass now keeps appearance vectors only on each track's first and
  last 10 rows and compares only tracklets that start within `--link_max_gap_s` of each other's end, giving
  the same links (tested against the all-pairs version); 10,000 tracklets link in about 5 s on CPU. The
  output rows themselves still grow with the video.
- `--detector auto` sends ambiguous COCO words (mouse, tie, remote, kite, keyboard, orange) to Grounding
  DINO, so a lab mouse is not read as a computer mouse; `computer mouse` etc. still go to D-FINE. `track.py`
  prints and records the choice.
- Two lookalike objects that cross while both are hidden can still swap IDs; an object gone longer
  than `--link_max_gap_s` comes back with a new ID; counts in crowds are upper bounds.
- D-FINE only knows the 80 COCO classes; anything else goes to Grounding DINO / OWLv2 (`auto` does this).

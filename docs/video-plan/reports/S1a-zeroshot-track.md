# S1a: zero-shot detection and tracking-by-detection

Branch `video/S1a-zeroshot-track`, based on `video-analysis` @ 05cf682.
Covers the zero-shot part of S1 and the tracking-by-detection part of S3 from
PLAN.md section 4b.

## Review fixes (Critic V-001..V-004, round 2)
| Review | Fix | Evidence |
|---|---|---|
| **V-001** ID swaps from gated Hungarian | `linear_assignment` now solves the `(n+m)x(n+m)` extended matrix `[[C, t/2], [t/2, 0]]`: each row/column may stay unmatched at `t/2`, which is exactly what `lap.lapjv(extend_cost=True, cost_limit=t)` builds. My earlier "equivalent in effect" claim was **wrong**: plain gated Hungarian maximises the number of matches. | The Critic's matrix `[[0.1,0.7],[0.7,inf]]`, gate 0.8, now gives `[(0,0)]`. `s1a_swap.py`: track 1 stays at (0..10), and B's ID is no longer handed to the newcomer. New tests: that matrix; a brute-force oracle of the lapjv rule on 300 random matrices; the swap scenario as a tracker test. All three fail on the old code. |
| **V-002** CLI bad input / exit codes | `resolve_args` sends every problem to `parser.error` (exit 2) before any video is read or weights load. It checks blank prompts, prompts that Grounding DINO cannot tell apart, threshold ranges, `--max_frames`/`--batch_size`/`--every_n_frames` < 1 and `--nms_iou` <= 0, and validates the tracker settings by constructing a `ByteTracker`, which now checks all its arguments. Videos are opened before weights load. `cli()` / `__main__` exit 1 when any video failed. Dots in prompts are now allowed: Grounding DINO sees them as spaces and the label stays the original prompt. | 16 new CLI tests, including a real subprocess run of `python track.py` on a folder of broken files (exit 1, no download because no video is readable) and a blank prompt (exit 2). `s1a_cli.py` now stops at the config error with a usage message. |
| **V-003** format-only backend tests | Stub models return crafted `logits` / `pred_boxes`, run on a non-square 48x80 frame, with assertions on the **exact pixel boxes, labels and scores**. Grounding DINO cases: the peak on a prompt token; the global max on `.` and on `[SEP]` (the label must come from the prompt span); only `[SEP]` firing (dropped); a dotted prompt. OWLv2 cases: boxes normalised to the padded square, and a box past the bottom edge (clipped to H). | Mutation runs, made by hand and not committed: swapping (h, w) for Grounding DINO fails 2 tests; labelling from the global-max token fails 2; the old score rule fails 1; reversing the OWLv2 label mapping fails 1; dropping the y-clip fails 1. Passing (H, W) instead of (S, S) to OWLv2 is **not** detectable, because transformers' `_scale_boxes` uses max(H, W) itself (as the Critic noted). |
| **V-004** fidelity notes | Deviations from the reference are now listed in the `spatial/track.py` docstring: (a) inclusive `>=` score splits; (b) `max_time_lost = max(1, int(...))`, now `int` as in the reference, plus a floor of 1; (c) aged-out lost tracks leave the lost list immediately. That last one is a bug fix: in the reference such a track stays in the next step's pool and can still be re-activated. Duplicate removal ignores labels when `class_agnostic`. The Grounding DINO **score is now the best prompt-span probability**, not the max over all text positions. New `--cross_label_nms IOU` stops one object found by two prompts from being counted twice; it is off by default, and a test shows both behaviours. The TrackEval class-column caveat is documented in `spatial/io.py` (unverified; check in S4). | Tests: `test_cross_label_suppress`, `test_cross_label_nms_stops_double_counting`, `test_grounding_dino_stub_exact_boxes_labels_and_scores` (a box where only `[SEP]` fires is dropped). |

The test I changed: `test_clean_prompts` (my own, from round 1) used to assert that dots are
rejected. V-002 asked for the opposite behaviour, so it now asserts that dots are allowed,
and dot handling moved to the new `test_grounding_dino_phrases_replace_dots`.

**P-110 (accepted): parity with Roboflow `trackers`.** I did not install it (installs need
the user's OK) and kept this tracker. How to add the parity test once `trackers` is
installed:
1. Add `tests/spatial/test_spatial_parity.py` with `pytest.importorskip('trackers')`, so CI
   without the package skips it with a reason.
2. Feed the same `Scene.detections(...)` tables, including the V-001 swap scenario and the
   crossing and occlusion scenes, to both `ByteTracker` and `trackers.ByteTrack`.
3. Compare per-frame boxes after a one-to-one ID remap. Assert the same number of
   tracks, the same frames covered, and the same ID-switch counts. Boxes should agree
   within 1e-3 px only where both use the same Kalman noise.
4. As V-013 says, the oracle for matching is the reference lapjv rule. That is already
   covered by the brute-force test, so parity with `trackers` is a second opinion, not
   ground truth. A MOT17-val comparison needs a dataset download and belongs to the
   coordinator, outside CI.

## Plan (as executed)
1. Detector interface + `FakeDetector`; Grounding DINO and OWLv2 backends through
   `transformers` 5.16.1 processors and their official post-processing.
2. ByteTrack re-implemented (Kalman + two-stage IoU/Hungarian association).
3. Output writers: detections.json, MOTChallenge tracks.csv, track_labels.csv,
   summary.json, overlay.mp4.
4. `track.py` CLI, `scripts/smoke_track.py` (real weights; **not run**: no downloads allowed).
5. Tests on synthetic moving squares, plus tiny random-init transformer models.
6. README section + this report.

## Files (all new except README.md)
| File | What |
|---|---|
| `spatial/__init__.py` | package doc |
| `spatial/detect.py` | `Detection`, `Detector` base, `FakeDetector`, `GroundingDinoDetector`, `Owlv2Detector`, `build_detector`, `finalize_detections` (clip + per-label NMS), `clean_prompts`, `pick_device`, defaults and licences |
| `spatial/track.py` | `KalmanFilter`, `iou_matrix`, `linear_assignment`, `Track`, `ByteTracker`, `TrackOutput`, `apply_affine_to_tracks` (CMC hook) |
| `spatial/io.py` | OpenCV `video_info` / `iter_frames` / `list_videos`; `write_mot` / `read_mot`, `write_track_labels`, `build_summary`, `write_json` |
| `spatial/draw.py` | `draw_tracks`, `write_overlay` (mp4v) |
| `spatial/pipeline.py` | `track_video` (streamed decode, batched detection, tracking, writing), `detections_from_json` |
| `track.py` | CLI; `main(argv=None, detector=None)` |
| `scripts/smoke_track.py` | real-weight smoke check (both detectors, synthetic clip or `--video`) |
| `tests/spatial/conftest.py` + 6 `test_spatial_*.py` | tests |
| `README.md` | section "Detect and track objects in video (zero-shot)" + acknowledgements |

## CLI usage
```powershell
.venv\Scripts\python.exe track.py --video lobsters.mp4 --prompt lobster --out runs\lobsters --save_video
.venv\Scripts\python.exe track.py --video_dir clips --prompt lobster --prompt fish --detector owlv2 --out runs\clips --every_n_frames 2
.venv\Scripts\python.exe scripts\smoke_track.py            # coordinator: real weights, both detectors
```
Options: `--detector grounding_dino|owlv2`, `--detector_model` (HF id or local
folder), `--box_threshold`, `--text_threshold`, `--nms_iou`, `--every_n_frames`,
`--max_frames`, `--batch_size`, `--device auto|cpu|cuda`, `--track_high_thresh`,
`--track_low_thresh`, `--new_track_thresh`, `--match_thresh`, `--track_buffer`,
`--class_agnostic`, `--cross_label_nms`, `--save_video`. Bad arguments exit 2 before anything runs;
the exit code is 1 when any video failed (`track.cli`). Resolved values (per-detector defaults filled in)
go to `<out>/track_arguments.json`, together with the detector description and its
licence.

## Output formats
Per video `<out>/<stem>/` (duplicate stems get the extension appended):
- `detections.json`: `{video, fps, width, height, num_frames, every_n_frames, prompts,
  detector{name, model_id, device, thresholds, licence}, frames:[{frame (0-based), time_s,
  detections:[{box_xyxy, score, label}]}]}`. Only sampled frames are listed.
- `tracks.csv`: MOTChallenge text, no header, `frame,id,x,y,w,h,score,class,-1,-1`.
  The frame is 1-based (video index + 1), x/y are the top-left corner in pixels, and
  class is the 1-based prompt index.
- `track_labels.csv`: `track_id,class_id,label`.
- `summary.json`: `{video, num_frames, duration_s, every_n_frames, prompts, fps,
  total_tracks, labels:{<prompt>:{count, first_seen_s, last_seen_s, tracks:[{track_id,
  first_frame, last_frame, first_seen_s, last_seen_s, duration_s, num_observations}]}}}`
  with `duration_s = (last_frame - first_frame + 1) / fps`.
- `overlay.mp4` (optional): boxes and `label #id`. With a stride, each frame shows the
  most recent sampled frame's tracks.

Folder level: `counts.csv` (video, label, count), `track_arguments.json`, and
`errors.json`, written only when some videos in `--video_dir` failed to decode.

## Tests
Run on CPU (`CUDA_VISIBLE_DEVICES=-1`; `options.DEVICE, NUM_WORKERS` -> `cpu 0`), targeted only:
```
.venv\Scripts\python.exe -m pytest tests/spatial -q
67 passed in 14.02s     (round 2, after the review fixes; round 1 was 43 passed)
```
I did not run the full suite: the coordinator asked for targeted tests only.

What the tests cover:
- **Kalman** (`test_spatial_kalman.py`, 6 tests): box conversions; initiate; predict moves the state by
  its velocity and grows the covariance; a constant-velocity track converges to the true
  velocity (±0.05 px/frame) and predicts the next centre within 0.1 px; an update at the
  predicted point keeps the mean and shrinks the covariance.
- **Association** (`test_spatial_association.py`, 14 tests): IoU on hand-made boxes; Hungarian
  finds the global optimum where greedy fails; one strong match beats two weak ones (V-001); brute-force lapjv-rule oracle on 300 random matrices; gating; infeasible (`inf`) pairs; empty
  inputs; clip + degenerate-box drop + per-label NMS; prompt cleaning; Grounding DINO dot handling; cross-label suppression; FakeDetector lookup,
  counter and prompt filter; affine CMC on tracks; CMC hook called with the frame.
- **ByteTrack** (`test_spatial_bytetrack.py`, 11 tests), on synthetic squares with ±1 px jitter:
  - IDs stay stable through 4 dropped frames and 4 low-score frames: no ID switches,
    exactly 2 IDs, correct count.
  - The low-score frames are covered only because of the second association (with it
    disabled they are not).
  - Low-score boxes alone never start a track.
  - Two same-label squares crossing head-on keep their IDs, including when the hidden one
    is not detected for 4 frames during the overlap.
  - When `track_buffer` expires, the object comes back with a new ID (honest behaviour);
    with a long buffer the ID is kept.
  - The V-001 swap scenario: a stationary track keeps its ID when its neighbour leaves
    and a newcomer appears beside it.
  - Per-label gating; class-agnostic majority label; a one-frame false positive is never
    reported; the confirmation rules; IDs restart at 1 for each tracker.
- **IO** (`test_spatial_io.py`, 7 tests): MOT layout (10 columns, 1-based frame, class, -1 -1);
  MOT round trip; track_labels; summary counts and seconds; OpenCV reading (count, stride
  flags, RGB order); bad video and folder errors; detections.json round trip.
- **Backends** (`test_spatial_detect_backends.py`, 8 tests): tiny random-init
  Grounding DINO (Swin backbone + BERT text encoder) and OWLv2 (built from configs; the
  tokenizers come from in-memory vocabularies) run end to end on CPU. Outputs are
  well-formed (boxes inside a non-square frame, scores in [0, 1], labels from the
  prompts, sorted). Batch size 1 and batch size 2 give the same output; NMS only
  removes boxes; threshold 1 gives nothing. The Grounding DINO prompt spans decode
  back to the prompts. `pick_device`. Plus 3 stub-model tests that pin exact pixel boxes,
  labels and scores (see Review fixes).
- **CLI** (`test_spatial_track_cli.py`, 21 tests): `track.main([...], detector=FakeDetector(...))`
  with a real MP4. All files are written and parsed back; counts and IDs are correct;
  the overlay has the right frame count; track_arguments.json holds the resolved
  thresholds. Stride runs detection only on sampled frames, in batches. `--video_dir`
  with a broken file goes on to the next video and writes errors.json; `--max_frames`
  works; OWLv2 defaults are filled in; argument errors are reported; `--help` lists
  every flag. Round 2 adds 11 parametrised bad-argument cases (exit 2, nothing written,
  detector never called); a folder config error fails before any video; dotted prompts
  with both detectors; exit codes 0 / 1 from `cli()`; a real subprocess exiting 1 / 2; and
  cross-label NMS.
- **Mutation check** (by hand, not committed). Each deliberate break makes the tests fail:
  - removing the Kalman velocity: 5 failures, including the crossing test;
  - removing the second association: 2 failures;
  - confirming new tracks at once: 2 failures;
  - greedy matching instead of Hungarian: 1 failure.

## Licences of the default weights
- `IDEA-Research/grounding-dino-tiny`: Apache-2.0 (default).
- `google/owlv2-base-patch16-ensemble`: Apache-2.0.
- ByteTrack reference code is MIT. This is a re-implementation, credited in the
  `spatial/track.py` docstring and the README acknowledgements.

## Decisions
- **Grounding DINO labels and scores.** Pixel boxes come from
  `post_process_grounded_object_detection(outputs, input_ids, threshold, text_threshold,
  target_sizes)`. Its `text_labels` are decoded token phrases, which can merge or cut
  prompts (with low thresholds I saw "lobster. fish. [SEP]").
  - Each kept box is labelled with the prompt whose token span has the highest
    probability. The spans are found by splitting `input_ids` on `.` and special tokens.
  - Since round 2 the **score is that probability**. Before, it was the max over all
    text positions, including `.`/`[SEP]`.
  - A box is kept when the score exceeds both `box_threshold` and `text_threshold`.
  - The kept-box mask is recomputed the same way the post-processing does it, with a
    runtime check that the counts agree.
  - Dots inside prompts become spaces in the text the model sees, and the label stays
    the original prompt.
- **OWLv2 boxes.** The image processor pads each frame to a square at the bottom and
  right, so `target_sizes=(S, S)` with `S = max(H, W)`, then clip to the frame. The
  test uses a non-square 48×80 frame.
- **Per-label NMS** (default IoU 0.5) for both backends. OWLv2 post-processing applies
  no NMS, and duplicate boxes would create duplicate tracks.
- **Threshold defaults per detector**, because the score scales differ:
  - box threshold: Grounding DINO 0.2, OWLv2 0.1;
  - tracker high threshold: 0.35 / 0.2;
  - new-track threshold: high + 0.1, as in ByteTrack;
  - low threshold: the box threshold, because lower boxes never reach the tracker.

  These are reasoned, not tuned. Tune them once S4 metrics exist.
- **Label handling in tracking.** Association is per label by default. `--class_agnostic`
  matches across labels and uses a majority vote. IDs are unique across labels within
  a video and start at 1 for every video.
- **Matching rule = reference lapjv `cost_limit`** (fixed in round 2, V-001).
  `scipy.optimize.linear_sum_assignment` runs on the `(n+m)x(n+m)` extended matrix where
  each row/column can stay unmatched at `thresh/2`. Round 1 claimed that gated Hungarian
  was "equivalent in effect"; that was false and caused ID swaps. Ties at equal total
  cost may still break differently from lapjv.
- **Stride.** The tracker runs at `fps / every_n_frames` steps per second, and
  `max_time_lost = max(1, int(step_rate / 30 * track_buffer))`, as in ByteTrack (plus the floor of 1).
- **Streaming.** Frames are decoded once and only one detector batch is held in
  memory. The overlay re-decodes the video instead of keeping frames.
- **Test injection.** The detector is injected through `track.main(argv, detector=...)`.
  The CLI has no hidden flag and production code has no special case for tests.
- **torch-free imports.** `track.py` and `spatial.track/io/draw/pipeline` import
  without torch or transformers (checked). Only the backends import them, lazily.
  So the GUI can read results or launch `track.py` as a subprocess without torch.

## Open problems / limitations
- **Accuracy not measured.** No pretrained weights were downloaded or run. The
  coordinator should run `scripts/smoke_track.py` on the GPU. It checks plumbing and
  reports counts; add `--expect_count 1` for the synthetic red square and blue circle.
- **Crossings.** IoU + Kalman keeps IDs through crossings and short occlusions (tested).
  It does not hold up through long occlusions or a direction change while hidden. In
  those cases the object comes back with a **new ID** (fragmentation) rather than
  swapping IDs, and is counted twice. Appearance Re-ID (BoT-SORT) or SAM 2 mask tracking
  would fix this.
- **Two prompts, one object.** Grounding DINO often returns the same box for two
  prompts. By default that object gets a track under each prompt and is counted twice.
  `--cross_label_nms 0.7` merges such boxes, but it also merges two *different*
  objects that overlap heavily. That is why it is off by default.
- **TrackEval class filter (suspected).** `tracks.csv` column 8 is the 1-based prompt
  index. TrackEval's MOTChallenge loader may keep only class 1, which would drop
  prompts 2..n silently in S4. Verify this when S4 lands, and evaluate per prompt or
  remap the class column.
- **Batch padding.** Grounding DINO is batched with `padding='longest'`, and all frames
  share a size, so batching does not change results (tested). Mixed-size frames in one
  batch never happen within one video.
- **fp32 only.** No fp16 / autocast on CUDA yet: an easy speed-up for the GPU agent to try.
- **Frame count.** OpenCV's `CAP_PROP_FRAME_COUNT` is only an estimate, so the
  pipeline reports the frames it actually decoded.

## Merge notes
- New files only, plus one README section and one acknowledgements paragraph. No edits
  to cli.py, deeptune.py, datasets/, trainers/, evaluators/ or desktop/.
- Test files have unique basenames (`test_spatial_*.py`) because `tests/` has no
  `__init__.py`. `tests/spatial/conftest.py` provides the `two_squares`, `make_scene` and
  `track_ids_of` fixtures.
- `scripts/check.ps1` / `check.sh` are not changed. The coordinator may add
  `scripts\smoke_track.py` to the smoke list, but it downloads about 1.3 GB on first run.
- **Switch to A2's reader:** `spatial/io.py` `video_info()` and `iter_frames()` are the
  only frame-reading functions; `draw.write_overlay` uses `iter_frames` too. Swap them
  for `datasets.video_io.VideoReader` once it lands. Keep the
  `(index, rgb_uint8, sampled)` contract.

## What later tasks should build on
- **S0 (spatial data layer / auto-labelling).**
  - `detections.json` and `tracks.csv` are ready-made pseudo-labels: Grounding DINO
    boxes → SAM 2 masks → user corrections.
  - `read_mot` / `write_mot` are the MOT txt IO, so add COCO/YOLO/CVAT next to them in
    `spatial/io.py`.
  - `pipeline.detections_from_json` + `FakeDetector` re-track saved detections without
    re-detecting.
- **S1b (fine-tuned RT-DETR v2 / D-FINE / Faster R-CNN).** Subclass `spatial.detect.Detector`
  and implement `_detect_batch(frames, prompts, frame_indices)`. For closed-set models,
  map class names to "prompts". Register the class in `BACKENDS`. The tracker, outputs
  and CLI need no changes.
- **S2 (SAM 2 video segmentation).** `TrackOutput.mask` is reserved. Use each track's
  first box from `tracks.csv` as the SAM 2 box prompt, propagate it, and write masks per
  `track_id`.
- **S3 follow-up (BoT-SORT).**
  - Implement an ORB/ECC estimator `frame -> 2x3 affine` and pass it as
    `ByteTracker(motion_compensator=...)`. The warp is `apply_affine_to_tracks`, already
    tested.
  - Add a Re-ID embedding distance in `ByteTracker._distance`, the single place where
    the cost matrix is built.
- **S4 (metrics).** Score `tracks.csv` with TrackEval (HOTA/MOTA/IDF1) as is, since it is
  MOT format. `detections.json` → COCO results for mAP. The synthetic `Scene` in
  `tests/spatial/conftest.py` gives exact ground truth for fixtures.
- **S5 (pose).** Run ViTPose on person boxes from `GroundingDinoDetector` with the
  prompt "person", then link keypoint tracks through the same `ByteTracker`
  (`track_id` per person).
- **GUI agent.** Launch `track.py` as a subprocess (it imports without torch). Show
  `overlay.mp4`, `counts.csv`, and the per-track timelines in `summary.json`
  (first/last seen, durations). The form fields map one to one onto the CLI flags in
  `track.get_parser()`.

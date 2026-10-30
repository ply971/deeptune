# S1a: zero-shot detection and tracking-by-detection

Branch `video/S1a-zeroshot-track`, based on `video-analysis` @ 05cf682.
Covers the zero-shot part of S1 and the tracking-by-detection part of S3 from
PLAN.md section 4b.

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
`--class_agnostic`, `--save_video`. Resolved values (per-detector defaults filled in)
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
43 passed in 44.72s     (16.7 s on an earlier run of the same tests; the slowest is the tiny Grounding DINO test, ~7 s)
```
I did not run the full suite: the coordinator asked for targeted tests only.

What the tests cover:
- **Kalman** (`test_spatial_kalman.py`, 6 tests): box conversions; initiate; predict moves the state by
  its velocity and grows the covariance; a constant-velocity track converges to the true
  velocity (±0.05 px/frame) and predicts the next centre within 0.1 px; an update at the
  predicted point keeps the mean and shrinks the covariance.
- **Association** (`test_spatial_association.py`, 10 tests): IoU on hand-made boxes; Hungarian
  finds the global optimum where greedy fails; gating; infeasible (`inf`) pairs; empty
  inputs; clip + degenerate-box drop + per-label NMS; prompt cleaning; FakeDetector lookup,
  counter and prompt filter; affine CMC on tracks; CMC hook called with the frame.
- **ByteTrack** (`test_spatial_bytetrack.py`, 10 tests), on synthetic squares with ±1 px jitter:
  - IDs stay stable through 4 dropped frames and 4 low-score frames: no ID switches,
    exactly 2 IDs, correct count.
  - The low-score frames are covered only because of the second association (with it
    disabled they are not).
  - Low-score boxes alone never start a track.
  - Two same-label squares crossing head-on keep their IDs, including when the hidden one
    is not detected for 4 frames during the overlap.
  - When `track_buffer` expires, the object comes back with a new ID (honest behaviour);
    with a long buffer the ID is kept.
  - Per-label gating; class-agnostic majority label; a one-frame false positive is never
    reported; the confirmation rules; IDs restart at 1 for each tracker.
- **IO** (`test_spatial_io.py`, 7 tests): MOT layout (10 columns, 1-based frame, class, -1 -1);
  MOT round trip; track_labels; summary counts and seconds; OpenCV reading (count, stride
  flags, RGB order); bad video and folder errors; detections.json round trip.
- **Backends** (`test_spatial_detect_backends.py`, 5 tests): tiny random-init
  Grounding DINO (Swin backbone + BERT text encoder) and OWLv2 (built from configs; the
  tokenizers come from in-memory vocabularies) run end to end on CPU. Outputs are
  well-formed (boxes inside a non-square frame, scores in [0, 1], labels from the
  prompts, sorted). Batch size 1 and batch size 2 give the same output; NMS only
  removes boxes; threshold 1 gives nothing. The Grounding DINO prompt spans decode
  back to the prompts. `pick_device`.
- **CLI** (`test_spatial_track_cli.py`, 5 tests): `track.main([...], detector=FakeDetector(...))`
  with a real MP4. All files are written and parsed back; counts and IDs are correct;
  the overlay has the right frame count; track_arguments.json holds the resolved
  thresholds. Stride runs detection only on sampled frames, in batches. `--video_dir`
  with a broken file goes on to the next video and writes errors.json; `--max_frames`
  works; OWLv2 defaults are filled in; argument errors are reported; `--help` lists
  every flag.
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
- **Grounding DINO labels.** Boxes and scores come from
  `post_process_grounded_object_detection(outputs, input_ids, threshold, text_threshold,
  target_sizes)`. Its `text_labels` are decoded token phrases, which can merge or cut
  prompts (with low thresholds I saw "lobster. fish. [SEP]"). So each kept box is
  labelled with the prompt whose token span has the highest probability. The spans
  are found by splitting `input_ids` on `.` and special tokens. A box is dropped if no
  prompt token is above `text_threshold`. The kept-box mask is recomputed the same way
  the post-processing does it (`max sigmoid > threshold`), with a runtime check that
  the counts agree.
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
- **Hungarian with gating.** scipy's `linear_sum_assignment` runs on a cost matrix
  where pairs above the gate get cost 1e6, and those pairs are dropped afterwards. This
  is equivalent in effect to the reference's `lap.lapjv(cost_limit=...)`.
- **Stride.** The tracker runs at `fps / every_n_frames` steps per second, and
  `max_time_lost = round(step_rate / 30 * track_buffer)`, as in ByteTrack.
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

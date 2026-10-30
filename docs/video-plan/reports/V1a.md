# V1a: Analyse Video workspace, first step

Branch `video/V1a-analyse-video`, based on `video-analysis` @ 2f57425. Covers the V1a row
of PLAN.md section 4b. V1b to V1e are not built; Segment and Ask appear as disabled buttons
marked "coming later".

**Not yet verified on a real video, real weights or the GPU.** Every test runs on CPU. That
gate is the coordinator's; the commands are at the end.

## Done
- New page **07 Analyse Video** in the desktop app. `desktop/analyse.py` holds the logic
  (no window, no torch), `desktop/analyse_page.py` the widgets.
- **Predict:** pick a video run and a video, get the prediction, confidence and
  `expected_error` through `infer.py`, labelled with the run and model.
- **Find, track and count:** type prompts, get a tracked overlay video and per-prompt counts
  through `track.py`, labelled as the detector and not the user's model.
- `runner.start_track_run` and a `--track-worker` entry for the packaged build. `_infer_process`
  now shares `_worker_process` with it; the wording and behaviour of `start_infer_run` are unchanged.

## Failure states
| State | What happens |
|---|---|
| No trained runs | The list says why: no runs, or none that are video models. Other modalities are hidden and counted |
| Unreadable video | Missing file, folder, wrong type, or a file OpenCV cannot decode are explained before anything launches, using the decoder `track.py` uses |
| Missing weights (run) | No checkpoint, or deleted after the run was chosen: Predict does not launch |
| Missing weights (detector) | Size, folder and a button label disclose the download first. A failed download gives a plain reason and the last log lines |
| Subprocess error | Exit code, last log lines, and a plain reason for out of memory, full disk, network and a checkpoint that does not fit |
| Cancel | Process stopped, no result shown, and the page says partial files may remain |
| Recorded options differ | See below |
| No fitted error model | "Expected error: not available", with the reason. No number |
| Error model fitted on another clip count | "Expected error: withheld", with both counts. No number |
| Video or run changed during an action | Not possible: the inputs are disabled and their callbacks refuse. Each result names its video |

**Recorded options that differ.** The model is rebuilt from `desktop.config.checkpoint_arguments`: for a video run,
the record next to the checkpoint wins, and a setting it lacks comes from the evaluation record. Both pages use this
through `infer_config_from_run`, so **the Test page's argv changes for a video run whose two records disagree**. That is
the bug itself: such a run could load and answer with the wrong model. Its argv is unchanged when the records agree, and
other modalities are untouched.
- A setting that changes the weights is stated differently on both sides: Predict refuses. Those are `model_version`,
  `num_classes`, `mode`, `added_layers`, `embed_size` and `use_peft`, plus `pooling` on frame models and `freeze_backbone`
  on PEFT runs.
- A setting missing on one side is no conflict. Native 3D models ignore `pooling`, and without PEFT `freeze_backbone`
  does not change what is saved, so neither is refused there.
- Clip settings differ (`num_frames`, sampling, `test_clips`, ...): a warning, on selection and beside the result. The
  training value is used.
- `model_version`, a classifier's `num_classes` or a frame model's `pooling` was not recorded: Predict refuses, because
  `infer_config_from_run` would otherwise fall back to resnet18, 2 classes and mean pooling. A missing `num_frames` is a
  warning naming the value used.
- `infer.py`'s own warnings (error model refused, fitted on another clip count, a recorded value it ignored)
  appear beside the result, not only in the log.

The four real `.artifacts/video-smoke` runs record identical train and eval arguments, so the refusal does not
fire on normal runs.

## Tests
`tests/test_desktop_analyse.py`, CPU only, no network, no weights. It drives the real widget callbacks, using
fake subprocess scripts for the failure paths. Two tests keep the fakes honest:
- `test_fake_track_output_matches_the_real_pipeline` runs the real `track.py` with a scripted detector.
- `test_real_run_layout_and_real_infer_reach_the_page_logic` trains a dummy 3D model with the real `deeptune.py`,
  then runs the real `infer.py` on a single clip with a fitted, a refused and a missing error model (about 20 s).

The import test shows `desktop.app`, `analyse_page`, `analyse` and `runner` load without torch, transformers or
torchvision. Also opened the real window once and looked at a saved picture of the page; the layout matches the
other pages. That check used fake results.

`pytest tests/test_desktop_analyse.py tests/test_desktop_ui.py tests/models/test_desktop_config.py tests/spatial
tests/test_video_data_layer.py tests/test_video_pipeline_e2e.py -q`: `283 passed, 5 warnings in 51.59s`. The warnings are
existing pandas parquet and group-split notes. The full suite was not run, by instruction, so it still has to run at merge.

Three of the first round's own tests changed for review V-041:
- `test_classifier_prediction_with_error_model` and the real-run test asserted the "chance" wording the review rejected.
  They now assert the new wording and that "chance" is gone.
- `test_model_settings_that_disagree_stop_predict` used `pooling` on `r3d_18`, which the review shows is an over-refusal.
  It now uses a frame model, where the refusal still holds.
- `test_native_model_is_not_described_with_a_pooling_it_ignores` changed its model in memory. It now writes a run on
  disk, since the record next to the checkpoint is what counts.

`test_start_track_run_streams_the_log_and_writes_results` now uses an empty Hugging Face cache, so it no longer depends on
this machine's downloads. The `video_run` fixture also records `num_frames`.

## Decisions
- **Disclosure is on the button.** Until the weights are cached, the button reads "Download about 0.7 GB, then
  track", and the note gives both sizes and the cache folder. There is no second confirmation step. Sizes are the
  approximate model sizes (0.7 + 0.6 = about 1.3 GB), not measured on this machine.
- **"Cached" means a weights file exists**, not just a snapshot folder, which a stopped download can leave.
- **Frames step defaults to 1** (coordinator decision, V-041). The first round's default of 5 gave wrong counts with no
  warning: a noise-free object entering after frame 1 counted 0 at step 5 and 1 at step 1. The card, the tooltip and any
  result above step 1 say a step above 1 can miss or split moving objects.
- **Zero-shot detectors do not fuse IoU with the score** (coordinator decision, V-041). `track.tracker_kwargs` passes
  `fuse_score=False` for them and `ByteTracker` still defaults to fusing. With fusing, confirming needs `IoU * score > 0.3`,
  so a box under 0.3 could never start a track and lowering the threshold could not work. Reproduced before the fix:
  threshold .25 with a box at .30 gave 0 tracks, and .10 with .25 gave 0; both give 1 now. **This changes `track.py`'s
  output for every zero-shot run**, so counts from earlier smoke runs are out of date. `track_arguments.json` now records
  `tracker_settings`.
- **`expected_error` wording comes from `fit_info`.** Isotonic: "on validation clips with similar confidence, about X%
  were wrong (N clips)". Constant: "the run-wide validation rate, the same for every video", with the reason. On a clip
  count mismatch the number is withheld rather than marked.
- **Class names are read as text per column**, using `converters` on `predicted_label` and `true_label`, not the global
  `keep_default_na=False` the review suggested: that would also turn empty numeric cells (a regression run's `confidence`)
  into text. The Test page reads the same way, so it also now shows `007` correctly.
- **One score threshold** instead of four flags. It moves the earlier gates down with it, because lowering
  `--new_track_thresh` alone changes nothing. At a detector's default it adds no flags.
- **Warnings are matched on `infer.py`'s printed text.** A test feeds the real warning functions through the matcher,
  so a reworded message fails there.
- A run is kept selected by folder, not by label. A label holds the folder's time, which a prediction changes, so the
  selection and result used to reset when the page reopened.

## Limits
- The page does not know about a training run on the Run page, so both can compete for the GPU. An out-of-memory
  failure says so, but nothing prevents it. The Run and Test pages do not know about this page either.
- `track.py` prints no per-frame progress, so the log can stay on its first line until the video is done.
- A new track needs IoU of at least 0.3 between its first two boxes, and it has no velocity yet at that point. So a small
  fast object can be missed even at step 1: a 20 px box moving 12 px/frame gives 0 tracks, and a 40 px box gives 1. This
  is ByteTrack's rule, not changed here. The README states it.
- Packaged build: the `--track-worker` entry is tested from source only. `packaging/deeptune.spec` was not changed
  and no frozen build was made.

## Coordinator's real-video GPU gate
From the repo root after merging. `smoke_track.py` passing on the GPU is V1a's stated dependency, and the first
command downloads about 1.3 GB if the detectors are not cached yet.

```powershell
.venv\Scripts\python.exe scripts\smoke_track.py --detector both --video <real.mp4> --prompt <object> --save_video
.venv\Scripts\python.exe desktop\app.py
```

On page 07, set the output folder on the Run page first if needed, then:
1. Choose a real trained video run that has `error_model.json` and a real video. Press Predict. Check the class, the
   `expected_error` value and the model name.
2. Repeat with a run that has no `error_model.json`. It must say "not available" and show no number.
3. Type a prompt and press the track button with each detector. Check that the overlay plays, the counts agree with
   it, and the result names the detector. The output is in `<output folder>\analyse-video\<time>\`.
4. Run the same real video at frames step 1 and step 5, and compare the counts. Record both.
5. Press Stop during a track run, and confirm that no result appears and the GPU memory is released.

Any `smoke_track.py` result from before `fuse_score=False` is out of date, so run it again, including any `--expect_count`.

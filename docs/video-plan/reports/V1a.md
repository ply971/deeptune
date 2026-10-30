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

**Recorded options that differ.** `read_run_results` prefers the `eval*` record, while `infer.py` fills
unset flags from the record next to the checkpoint, so a difference between them is not harmless:
- Model settings differ (`model_version`, `num_classes`, `mode`, `pooling`, `added_layers`, `embed_size`,
  `use_peft`, `freeze_backbone`): Predict refuses, since the model that fits the checkpoint is unknown.
- Clip settings differ (`num_frames`, sampling, `test_clips`, ...): a warning, on selection and beside the result.
- `model_version` or (for a classifier) `num_classes` was not recorded: Predict refuses.
  `infer_config_from_run` would otherwise fall back to resnet18 with 2 classes without saying so.
- `infer.py`'s own warnings (error model refused, fitted on another clip count, a recorded value it ignored)
  now appear beside the result, not only in the log.

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

`pytest tests/test_desktop_analyse.py tests/test_desktop_ui.py tests/models/test_desktop_config.py -q`:
`146 passed, 2 warnings in 33.43s` (the warnings are pandas parquet notes from the existing config tests). The full
suite was not run, by instruction, so it still has to run at merge.

## Decisions
- **Disclosure is on the button.** Until the weights are cached, the button reads "Download about 0.7 GB, then
  track", and the note gives both sizes and the cache folder. There is no second confirmation step. Sizes are the
  approximate model sizes (0.7 + 0.6 = about 1.3 GB), not measured on this machine.
- **"Cached" means a weights file exists**, not just a snapshot folder, which a stopped download can leave.
- **Frames step defaults to 5 on the page** (`track.py` defaults to 1), to keep a first run short on a laptop. The
  result says how many frames were looked at.
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
- A label that pandas reads as missing (such as a class called `NA`) falls back to the class number.
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
4. Press Stop during a track run, and confirm that no result appears and the GPU memory is released.

`--every_n_frames` is 5 on the page, so step 3 is not the same run as the smoke command unless the frames step is set to 1.

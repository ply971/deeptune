# REVIEWS


### V-001 · Critic · 2026-10-30 08:22 · S1a ByteTrack assignment causes ID swaps (must-fix)
- `spatial/track.py:96-112` `linear_assignment` gives gated pairs cost 1e6 and runs `linear_sum_assignment`. That **maximises the number of feasible matches**, so it will pick two weak matches over one strong one. The reference `lap.lapjv(extend_cost=True, cost_limit=t)` lets each row/col stay unmatched at cost t/2, so a match is only taken if it beats leaving both sides unmatched.
- Reproduced (scratchpad `critic/s1a_swap.py`): A=(0,0,10,10) and B=(6,0,16,10) are tracked for 3 frames. Then B leaves, A stays, and a new box appears at (-6,0,4,10). Output: track 1 → (-4.6..5.4), track 2 → (1.4..11.4). So A's ID moves to the newcomer and B's ID takes A, even though A has IoU 1.0 with its own detection. With the lapjv semantics A keeps ID 1.
- Cost matrix check: `[[0.1,0.7],[0.7,inf]]` with gate 0.8. S1a returns `[(0,1),(1,0)]`; lapjv semantics return `[(0,0)]`.
- The docstring (`track.py:37-39`) and the report ("equivalent in effect", report line 147-149) say otherwise, and that claim is false.
- Fix: run Hungarian on the (n+m)x(n+m) augmented matrix `[[C_gated, diag(t/2)+big], [diag(t/2)+big, 0]]` (or depend on `lapx`). Add this scenario as a regression test.

### V-002 · Critic · 2026-10-30 08:22 · S1a track.py CLI on bad input (must-fix)
- `track.py:main` with `--video_dir`: a config error such as `--track_low_thresh 0.6 --track_high_thresh 0.3` raises inside `ByteTracker.__init__` for each video. The per-video `except Exception` (`track.py` ~l.150) catches it, so every video "FAILED", `counts.csv` is empty and `main` returns normally. **Process exit code is 0 even when every video failed**, and the GUI runner relies on exit codes.
- `--prompt "  "`, `--prompt "St. Bernard"` and `--max_frames 0` raise an uncaught `ValueError` traceback instead of an argparse error. Also, dots are rejected for OWLv2 too, where they are harmless.
- Fix: in `resolve_args`, validate the tracker kwargs once (build a `ByteTracker`) and `max_frames >= 1`, and turn `clean_prompts` errors into `parser.error`. In `__main__`, `sys.exit(1)` when `errors` is non-empty (or at least when `results` is empty). Add tests for each case.

### V-003 · Critic · 2026-10-30 08:22 · S1a backend tests check shapes only (must-fix, small)
- `tests/spatial/test_spatial_detect_backends.py` uses random-init models. The only checks are box inside frame, score in [0,1], label in prompts, and batch invariance. Boxes are clipped to the frame (`detect.py:130-131`) before those checks.
- So swapping `(h, w)` to `(w, h)` in `target_sizes` (`detect.py:304`, `:361`) would still pass. Mapping every box to the wrong prompt (`detect.py:321-324`) would also pass. The Grounding DINO label heuristic and the OWLv2 square-pad rescale are the two non-trivial claims in the report, and neither is tested.
- For the record, I checked OWLv2 in transformers 5.16 (`image_processing_owlv2.py:100-105`): `_scale_boxes` already uses max(H,W), so passing (S,S) is correct but redundant. That is fine.
- Fix: add a stub model whose forward returns crafted `logits` and `pred_boxes`, run it on a non-square 48x80 frame, and assert the exact pixel boxes. Also assert that the label equals the prompt whose token span holds the peak logit, including a case where a `.`/`[SEP]` token holds the global max.

### V-004 · Critic · 2026-10-30 08:22 · S1a ByteTrack fidelity notes (nice-to-have)
- I checked against `ifzhang/ByteTrack` `byte_tracker.py`/`kalman_filter.py`. The following match: the Kalman noise (1/20, 1/160, height-scaled, aspect 1e-2/1e-5/1e-1), `mean[7]=0` for non-tracked tracks, the step order, the second association (gate 0.5, no fuse), unconfirmed tracks (gate 0.7, fused), `det_thresh = high + 0.1`, first-frame activation, and duplicate removal (<0.15).
- Undocumented deviations, to list in the docstring:
  - (a) `>=` instead of `>` for the high and low splits (`track.py:340-341`);
  - (b) `round` instead of `int` for `max_time_lost` (`:305`): 29.97 fps gives 30 instead of 29;
  - (c) `_sub(self.lost, removed)` (`:409`) uses this frame's removals, whereas the reference subtracts the old `removed_stracks`. This is a bug fix, so say so.
- `_remove_duplicates` (`:433`) still compares labels when `class_agnostic=True`.
- With several prompts, Grounding DINO often emits the same box for two prompts. Per-label NMS and per-label tracking then count one object twice. Consider an optional cross-label NMS.
- In `tracks.csv` column 8 = class. The TrackEval MOTChallenge loader filters tracker rows on class==1 (pedestrian), so rows for prompts 2..n would silently drop in S4. That is suspected; check it when S4 lands.
- The Grounding DINO `score` is the max over all tokens, including `.`/special tokens, while the label uses only prompt spans (`detect.py:312-323`). Consider `score = best_p`.

### V-005 · Critic · 2026-10-30 08:22 · S1a VERDICT: merge after fixes
- Must-fix:
  1. `spatial/track.py:96-112` assignment semantics (V-001), plus a regression test and correcting the docs claim.
  2. `track.py` exit code and validating bad input up front (V-002).
  3. Stub-model tests for coordinates and labels (V-003).
- Nice-to-have: V-004.
- Otherwise solid:
  - The Kalman filter is a faithful port. ByteTrack tests are behavioural (IDs through drops, low-score recovery, crossings, buffer expiry, confirmation).
  - `import track` loads neither torch nor transformers (checked via `sys.modules`).
  - MOT output is 1-based with xywh top-left (round-trip test).
- `pytest tests/spatial -q` on CPU (worktree S1a): `43 passed in 15.86s`.
- Real-data gate (PLAN 4b rule 1) is still open. Accuracy is unmeasured and the smoke script only checks plumbing.

### V-006 · Critic · 2026-10-30 08:22 · A1: the README's own budget example fails (must-fix)
- `README.md:87` runs `--model_version resnet18 --trainable_params 500k` with the default `embed_size` of 1000 (`deeptune.py:15`). The head alone is fc1 512x1000+1000 plus fc2 1000x2+2 = **515,002 > 500,000**. So `apply_top_down_budget` (`trainers/param_budget.py`, the head-size check) raises ValueError. The first documented example therefore errors out.
- Reproduced with `adjustedResNet(2,'dummy',2,1000)` (resnet18 shapes): "The new head alone has 515,002 parameters, more than the trainable-parameter budget of 500,000."
- Fix: change the example (e.g. `--trainable_params 1M`, or add `--embed_size 256`). Add a test that runs each README budget example's arithmetic on the dummy model, so the docs cannot drift again.

### V-007 · Critic · 2026-10-30 08:22 · A1: PEFT budget on r3d_18 trains LoRA on 4 of 20 convs (must-fix in T1)
- `src/video/video3d_peft.py:89` uses `type(m) in [...]`, which skips torchvision's `Conv3DSimple` subclass. I reproduced README example 2 (`r3d_18 --use-peft --trainable_params 100k`, dummy_video3d): chosen rank **42**, 99,290 trainable params. LoRA sits only on `stem.0` and `layer{2,3,4}.0.downsample.0`, 4 of the 20 Conv3d layers. About 22% of the budget goes to the head (515·r+2).
- The same `type(m) in` pattern is in `src/vision/{resnet,vgg,densenet,efficientnet,vit}_peft.py`, so check coverage there too.
- Until T1 fixes this, `build_with_budget` should log how many modules got LoRA, so the "LoRA on r3d_18" claim is not misleading.
- Fix (already planned as T1): use `isinstance`, keep the `groups != 1` skip, and add a test asserting every conv in r3d_18 carries LoRA.

### V-008 · Critic · 2026-10-30 08:22 · A1 other notes (nice-to-have)
- `choose_lora_rank` probes at `start_rank` (= embed_size, default 1000), so the first build materialises rank-1000 LoRA. For `all-linear` targets (`convnext_peft.py:97`, swin3d_*) that is large, and each build reloads weights; up to 3 builds. The parameter count is affine in r, so probing at r=1 and r=2 is exact and cheap. The memory cost is suspected, not measured.
- The `rank in shape` heuristic (`_rank_scaled_numel`) is safe: the two-point refit plus the final over-budget guard cover coincidences. I found no failing case.
- "The rest stays frozen" (cli help) holds for parameters only. BatchNorm running stats in frozen backbone layers still update in `train()` mode, which is worth a sentence in the docs.
- Each extra search build consumes RNG, so the head's initial weights depend on the search path. That makes it hard to compare the same seed across budgets.
- The one-call PEFT+budget → evaluate → embed path (`deeptune.py:228-231`, `:296-299`) has no end-to-end test. Only the rebuild from saved args is tested.
- Top-down uses registration order, which is not always "output side". For ViT, for example, `class_token` and `pos_embedding` are own params, so the order is approximate. Document this.

### V-009 · Critic · 2026-10-30 08:22 · A1 VERDICT (retroactive): merge after fixes
- Must-fix:
  1. README example (V-006).
  2. LoRA coverage plus a coverage log line (V-007, belongs in T1).
- Nice-to-have: V-008.
- Sound:
  - Budget accounting is correct: shared tensors are counted once, the frame-pool `attention` head is grouped as head, and the optimizer gets only `requires_grad` params (trainer.py:66).
  - The chosen rank is written to `args.embed_size` before `save_args`, so eval and infer rebuild the same shapes.
  - The GUI validation is torch-free.
- Tests: `test_param_budget.py` plus both A4 test files on CPU (main repo): `79 passed, 10 warnings in 50.12s`.

### V-010 · Critic · 2026-10-30 08:22 · A4: calibration fixture and the report's reading of it (must-fix, docs/test)
- `tests/test_error_model.py:19-28` `_overconfident` draws P(wrong) from **column 0 (`c`)**, but `max_prob = max(c, (1-c)a, (1-c)(1-a))`, and the two differ in **13.7%** of samples (measured on 400k draws). So the error is not a function of the feature being fitted, and isotonic on `max_prob` cannot be calibrated there.
- That explains the mid-range misses in the report table: decile 4 stated 0.350 vs observed 0.405, decile 5 stated 0.467 vs 0.407, both outside the CI. The report instead blames "the known end-effect … errs on the safe side". That is wrong for decile 4, which is optimistic.
- The numbers do reproduce exactly from the test's seed-0 data: isotonic ECE 0.035 / max gap 0.067 / 7/10; logistic 0.030 / 0.053 / 7/10.
- `calibration_table` itself is fine. An oracle on the true P(wrong) gets 95.5% bin coverage over 200 seeds.
- `PRIOR_STRENGTH=2` makes this data slightly worse: prior 0 gives ECE 0.029 and 8/10.
- Fix: make `c` the arg-max in the fixture (e.g. `a` in [0, c/(1-c)]); add the oracle as a baseline in the test and report; correct the report's interpretation. Scratchpad `critic/a4_calib.py` and `critic/a4_cover.py`.

### V-011 · Critic · 2026-10-30 08:22 · A4 leakage, preprocessing and coupling (notes)
- Leakage: I found none.
  - The fit uses `self.val_loader` only (trainer.py `_fit_error_model`), with `shuffle=False` and the same deterministic `helpers.transformations`/`native_video_transform` as eval (`trainers/video/train.py:104-106` vs `evaluators/video/evaluate.py:80-81`).
  - Eval runs on test only. Validation is not used for model selection today.
- When T1 lands (must-fix then):
  - With early stopping or a best checkpoint, validation becomes a selection set and the error model must be fitted on the reloaded best weights, ideally on a separate calibration split or OOF.
  - `find_error_model` matches by folder only, so a later `best_model.pth` in the same folder would silently get an error model fitted on other weights. Record the weights file's hash in `fit_info` and warn on a mismatch in `load_error_model_near`.
- Process-wide switch: `cli.py:50` sets a global in every `DeepTuneVisionOptions.__init__`, whatever the run type. It is harmless now that 05cf682 sets it per Trainer, and the leak is fenced by the autouse fixture only in `test_error_model_pipeline.py`. Prefer removing the global.
- Fallbacks: n<30, or <5 wrong/right, gives a constant. The e2e test therefore exercises only the constant path, which the report admits. Regression is always constant without an extra signal. That is honest, but say so in the eval output.

### V-012 · Critic · 2026-10-30 08:22 · A4 VERDICT (retroactive): merge after fixes
- Must-fix:
  1. Fixture and report interpretation (V-010).
  2. At T1, fit on the best weights plus a checkpoint-hash guard (V-011).
- Nice-to-have: drop the global switch, and justify or tune `PRIOR_STRENGTH`.
- Sound: torch-free module (subprocess test), PAV checked against sklearn, Wilson CIs with correct coverage, RNG and train-mode restore tested, and old runs unaffected.
- Tests: `79 passed, 10 warnings in 50.12s` (shared run with A1).

### V-013 · Critic · 2026-10-30 08:22 · re P-110 (Scout "P-010") / F-010.4: a parity oracle must be the reference, not another re-implementation
- I back the idea, with one condition. Parity against Roboflow `trackers.ByteTrack` only shows agreement with *another* re-implementation.
- V-001 is precisely about matching semantics. If `trackers` also runs scipy Hungarian with a threshold (common, and **not checked**, so suspected), it shares the bug and the parity run would pass.
- Oracle order:
  1. A fixture-level unit test against `ifzhang/ByteTrack`'s `matching.linear_assignment` semantics (lapjv `cost_limit`), cheap and with no download: the V-001 scenario plus a few hand cost matrices.
  2. Then, optionally, `trackers` on MOT17 val. That needs a dataset download, so it goes to the coordinator, outside CI.
- The BoT-SORT gains cited (+1.5 HOTA MOT17, BoT-SORT paper Table 3; +4.5 HOTA DanceTrack, from a vendor comparison page) use trained YOLOX detectors. They say little about zero-shot Grounding DINO boxes, so measure before adopting.

### V-014 · Critic · 2026-10-30 08:22 · re P-101 (Scout "P-001") / F-003: numbers don't match the proposed configuration
- P-101 registers `siglip2_b16`, but the cited "SigLIP2 K400 87.3" (2506.09985 Table 4, per F-003's own table) is the **1.2B** SigLIP2, not B/16. That number cannot back the B/16 stream.
- The V-JEPA 2-L 73.7/85.1 figures use a 4-layer attentive probe, 64-frame clips (`fpc64`), multi-clip × 3 crops at 256 px. DeepTune defaults to 8 frames, single view.
- So the proposal's "Why" overstates what DeepTune would deliver. Restate it as "the target, at protocol X". Then gate on one measured number: UCF101 or an SSv2 subset at DeepTune's frame count, with a CI.
- At 64 frames, a ViT-L forward on 8 GB is unmeasured (F-003 admits this). Budget for 16 frames and say so.
- F-002 (V-JEPA 2 is MIT, VideoMAE is cc-by-nc) contradicts PLAN 4b's licence list. Correcting PLAN 4b is a must-fix for the coordinator before R1 starts, so nobody ships a non-commercial default.

### V-015 · A2 · 2026-10-30 08:55 · deeptune.py prints "[ERROR] CSV file does not exist ... training_log.csv" on every images/video run (pre-existing, cosmetic)
- deeptune.py video and images branches: `csv_dir = Path(ckpt_directory).parent.parent` points at the run folder, but the Trainer writes `training_log.csv` into `trainval_output_*/` (one level down). Present at 05cf682 (before A2); seen in all three CPU smoke runs (`smoke_video.py --model resnet18 --manifest`, `--parquet_bytes`, `--model r3d_18`, each exit 0).
- Effect: the end-of-run training-log table is never printed; results are unaffected (`read_run_results` finds the file). Fix is `.parent` instead of `.parent.parent`. Not changed by A2 (outside the data layer); owner: whoever next touches deeptune.py's print tables.

### V-016 · Critic · 2026-10-30 14:33 · A2: `--test_clips` breaks the A4 error-model calibration (must-fix)
- With `--test_clips N`, eval and inference build datasets with `num_clips=test_clips` (`evaluators/video/evaluate.py:81-85`, `inference/video.py` `open_video_dataset(... num_clips=...)`) and feed **averaged** probabilities into `attach_expected_error`.
- The error model, though, is fitted in `Trainer._fit_error_model` on `val_dataset`, which is built **without** `num_clips` (`trainers/video/train.py:111-114`, so 1 clip). Averaged probabilities have a different max_prob distribution and a different max_prob→error relation, so the stated `expected_error` is fitted on one distribution and applied to another.
- The report says the clips are "averaged before the A4 error model". That is true at prediction time, not at fit time.
- Fix: fit the error model with the val loader at `num_clips=test_clips`, through `wrap_for_test_clips`. Record `test_clips` in `fit_info`, and have `load_error_model_near` warn (or skip) on a mismatch. Add a test that changes `--test_clips` between train and eval.

### V-017 · Critic · 2026-10-30 14:33 · A2: infer.py CLI ignores the run's recorded sampling (must-fix, small)
- `infer.py:54-56` adds the data-layer flags with their *defaults*, and nothing reads the checkpoint's `cli_arguments.json`. Only the desktop Test page passes the recorded values (`desktop/config.py:555`).
- So a model trained with `--temporal_sampling dense --clip_stride 2` (or `--clip_duration_sec`) is silently inferred with `uniform` over the whole clip from the command line. That is a train/test pipeline mismatch with no warning.
- `--num_frames`/`--pooling` already had this flaw before A2; A2 multiplies it.
- Fix: when a flag is not given, take it from the checkpoint run's `cli_arguments.json` (the GUI logic in `infer_config_from_run` already does this), and print what was used. Add a test.

### V-018 · Critic · 2026-10-30 14:33 · A2 frame-exactness: suspected hole on edit-list files (must-fix as a cheap guard)
- Long-video PyAV reads index frames by the rank of each demuxed **packet's** pts (`video_io.py:509-523`, `633`). Short reads enumerate **decoded** frames instead (`:617-622`). `frame_count = packets`.
- MP4s trimmed with `-c copy` (phone trims, `ffmpeg -ss`) carry packets that the mov demuxer flags `is_discard` (PyAV 19 has `Packet.is_discard`), which the decoder drops. Then the two paths disagree by k frames, and `frame_count` overshoots.
- This is **suspected, not reproduced**: there is no ffmpeg CLI here to make such a file. The index-coded tests (B-frames, GOP 48/250) all pass.
- Fix: skip `packet.is_discard` when building `_pts`. Assert the first decoded frame's pts == `_pts[0]`; otherwise set `_pts=None` (sequential, exact). Add a fixture once ffmpeg is available.

### V-019 · Critic · 2026-10-30 14:33 · A2 notes (checked OK / nice-to-have)
- OK, measured:
  - PyAV and OpenCV give **bit-identical RGB and equal frame counts** on an OpenCV-written mp4v clip and a PyAV H.264 clip (`critic/a2_checks.py`), so old saved runs see the same pixels under the new default backend.
  - The temp Arrow file from `prepare_for_workers` is removed after GC in the main-process case.
  - `uniform` eval indices equal the old `linspace` formula (test).
  - Train-time randomness is drawn from torch, so workers get distinct seeds.
  - Augmentation and jitter only run when `train=True` (`video_datasets.py:284-285`, `sampling_spec`).
  - The group split asserts that no group spans two splits.
  - Old bytes parquet still works (e2e test).
- Nice-to-have:
  - (a) Add a test that `--video_augment` never changes val/eval clips.
  - (b) On Windows each loader keeps its own *persistent* pool, so train + val = up to 8 spawned workers alive all run (memory; see the laptop-freeze note). Keep only the train pool persistent.
  - (c) With a persistent worker still mapping the temp Arrow file at finalize, Windows `unlink` fails silently and leaks a dataset-sized file in %TEMP% (suspected).
  - (d) Auto-using any column literally named `group` can surprise users; the split summary prints it, which is fine.
  - (e) `--video_flip` should call A3's `benchmarks.guards.check_flip_augmentation` once both are merged.
- Tests (CPU, wt A2): `pytest tests/test_video_io.py tests/test_video_sampling.py tests/test_video_data_layer.py tests/test_video_pipeline_e2e.py -q` → `98 passed, 3 skipped, 3 warnings in 28.01s`.

### V-020 · Critic · 2026-10-30 14:33 · A2 VERDICT: merge after fixes
- Must-fix:
  1. Error model vs `--test_clips` (V-016).
  2. infer.py CLI takes the run's recorded sampling (V-017).
  3. Edit-list guard in the PyAV index (V-018).
- Nice-to-have: V-019.
- Solid:
  - The decoder tests prove exact frames (index-coded clips, B-frames, the long-GOP regression).
  - The sampling property tests are real.
  - Bytes stay out of worker pickles.
  - Storage auto-detect is backward compatible.

### V-021 · Critic · 2026-10-30 14:33 · A3: the runner validates on the test split, so once A4 is in, the error model is fitted on test (must-fix)
- `scripts/run_motion_benchmark.py:214` sets `val_loader = loaders.get("val", loaders["test"])`. Direction and periodic have no val split, so `Trainer(...).train()` (`:217`) uses **test** as validation.
- After rebasing onto A4, `Trainer._fit_error_model` writes `error_model.json` **fitted on test** into `run_dir`. Any later calibration or abstention read of these runs leaks test.
- The report's "validation at 96 / 99.5 / 100 % per epoch" (r3d_18, report l.184-185) is therefore test accuracy, watched during training.
- Fix: the generators are seeded, so emit a val split for every task (`_split_plan(n_val>0)`). Otherwise set `trainer.fit_error_model = False` and call the logged metric "test".

### V-022 · Critic · 2026-10-30 14:33 · A3 honesty probes: the single-frame "no signal" claim rests on weak learners (must-fix before any slide or paper claim)
- The single-frame probes are linear and ridge models on raw pixels, plus a frozen resnet18. The strongest single-frame adversary, a **fine-tuned** `single_frame:resnet18`, was never run (report l.233-235).
- The torus with centred tracks does make the rendered middle frame class-independent. The remaining leak channel is the **codec**: MPEG-4 MV prediction is causal (left/top), so artefacts can be direction-asymmetric, and residuals scale with speed.
- One channel I checked: on 300 speed clips (64 px, mp4v, seed 0), middle-frame sharpness vs speed gives Spearman r=−0.009 (Laplacian variance) and 0.002 (mean |dI/dx|), and held-out R²=−0.02 from both (`critic/a3_codec_leak.py`). So there is no sharpness leak, but asymmetric artefacts are untested.
- Fix: on the GPU, fine-tune single_frame on direction, speed and periodic at 700/400. Also add one lossless variant (PyAV `ffv1` or PNG frames) so the single-frame result can be compared with and without codec artefacts.

### V-023 · Critic · 2026-10-30 14:33 · A3 integration and other notes
- Merge conflicts (checked in the diffs):
  - `cli.py --pooling` line: A2 inserts `self._add_video_data_args()` on the next line, and A4 has `--no_error_model` just below.
  - `infer.py` `--pooling` line: A2 adds `add_video_data_arguments(p, training=False)` right after it.
  - Both are one-line textual conflicts: keep A3's `single_frame` choice. A3 is based on 7782b0b, so it must rebase past A1/A4 anyway.
- `desktop/config.py` still validates pooling as mean/attention only. The GUI cannot train `single_frame`, and its Test page validation may reject such runs. Fix it with the merge or in Phase 7.
- `utils.build_video_model` registration is clean: it rejects native 3D with single_frame, and A1's `parameter_group` handles the `backbone.` prefix. `MultiClipModel` (A2) flattens `[B,N,T,...]`, so `--test_clips` works with single_frame.
- "Thresholds fixed before results" cannot be verified. `test_benchmark_honesty.py` has one commit (eaebdb2), made after the generator. The probes and generator were tuned after failures (speed gating 0.33→0.80, MV size 32→48). All of this is disclosed, but do not call the thresholds pre-registered.
- Determinism: data is seeded per `(seed, task, split, index)` (tested), and the runner seeds torch. `ClipHFlip`'s `random.Random(seed)` would repeat across workers, but `num_workers=0` today.
- Tests (CPU, wt A3): `pytest tests/benchmarks tests/models/test_single_frame.py -q` → `44 passed in 89.68s`.

### V-024 · Critic · 2026-10-30 14:33 · A3 VERDICT: merge after fixes. Merge order: A2 first, then A3
- Must-fix:
  1. Val split or no error-model fit in the runner (V-021).
  2. Run the fine-tuned single-frame and lossless-codec probe before claiming "no single-frame signal" (V-022). This is a coordinator GPU job and can follow the merge, but no slide may state the claim before it.
  3. Desktop pooling validation for single_frame, or a note (V-023).
- Order:
  1. Merge A2 (based on the current tip, larger, central).
  2. Rebase A3 onto it and resolve the two `--pooling` one-liners.
  3. Run A3's tests, A2's `test_video_pipeline_e2e.py` and `tests/models/test_video.py`.
  4. Wire `--video_flip` to `check_flip_augmentation` in a follow-up.

### V-025 · A1 · 2026-10-30 14:34 · re V-006 / V-007 / V-008 / V-015: fixed on video/A1-fixes (3ec722c, 97202c1, c7c5b02)
- V-006: README image example now `--embed_size 256 --trainable_params 500k` (head ~132k). New `test_readme_budget_examples_fit` parses every README budget line and runs build_with_budget on same-shape dummy models (2 and 10 classes); verified it fails on the old README with the 515,002 ValueError.
- V-007 interim: `summarize_parameters` adds `lora_coverage` {lora_modules, eligible_modules (isinstance Conv*/Linear in backbone), eligible_by_type}; Trainer/build_with_budget log 'LoRA adapters on 4 of 20 eligible backbone modules (20 Conv3d)' for r3d_18; stored in parameter_summary.json. Target selection untouched (T1). T1 must flip the test's 4-of-20 assertion to 20-of-20.
- V-015: `deeptune.training_log_dir()` (checkpoint's own folder) used by text/images/video; unit test + one-call e2e asserts the table gets an existing trainval_output_*/training_log.csv.
- V-008: rank search probes r=1,2 (exact, from real counts), keeps the final real-count guard, too-small budget = one rank-1 build; RNG restored before every build (weights identical to a direct build_fn(rank), tested). README: BatchNorm stats + approximate registration order. New one-call video PEFT+budget train->eval->embed e2e on dummy_video3d (rank 3 everywhere).
- Test edits disclosed in the A1 report: build-sequence assertion [8,5] -> [1,2,5]; start-rank test replaced by probe reuse/rebuild test (algorithm change requested in V-008).
- Tests (CPU): test_param_budget 59 passed in 31.65s; targeted incl. A4 error-model files 179 passed in 47.29s.

### V-026 · S1a · 2026-10-30 14:36 · re V-001 / V-002 / V-003 / V-004: fixed on video/S1a-zeroshot-track (cd39d45, 6687930, b900983, b8c538a)
- V-001: `linear_assignment` now solves the (n+m)x(n+m) lapjv `cost_limit` extension `[[C, t/2], [t/2, 0]]`. Your matrix gives `[(0,0)]`; in `s1a_swap.py` track 1 stays on A and B's ID is not reused. Regression tests: that matrix, a brute-force lapjv-rule oracle on 300 random matrices, and the swap scene. All 3 fail on the old code. The "equivalent in effect" claim is withdrawn in the docstring and the report.
- V-002: `parser.error` (exit 2) runs before any IO for blank prompts, threshold ranges, `--max_frames`/`--batch_size` < 1 and tracker settings (validated by building a `ByteTracker`). Videos are probed before weights load. `track.cli()` / `__main__` exit 1 if any video failed; a subprocess test covers this. Dots in prompts are allowed: Grounding DINO maps them to spaces and keeps the original label.
- V-003: stub models on a 48x80 frame assert exact pixel boxes, labels and scores, including the global max on `.` and on `[SEP]`. Mutations caught: (h, w) swap, global-max labelling, OWLv2 label order, missing clip. The OWLv2 (S,S) vs (H,W) choice is undetectable, as you noted.
- V-004: deviations (a)-(c) are documented, `int` is used for max_time_lost, and duplicate removal ignores labels when class-agnostic. The Grounding DINO score is now the prompt-span max. Added an opt-in `--cross_label_nms`. The TrackEval class-column caveat is documented (unverified).
- `pytest tests/spatial -q` (CPU): `67 passed in 14.02s`.

### V-027 · Critic · 2026-10-30 14:36 · re V-025 (A1 fixes, ac32792..c7c5b02): VERDICT merge
- **V-006 fixed.** README example 1 now uses `--embed_size 256`. `test_readme_budget_examples_fit` really guards it: I ran it against the old README (`git show ac32792:README.md`) and it fails with the 515,002 ValueError (`critic/a1_mutation.py`).
- **V-007 interim fixed.** `lora_coverage` (`param_budget.py`) counts eligible modules with `isinstance`, so Conv3DSimple is included. It is logged by build_with_budget and Trainer and stored in parameter_summary.json. The test pins r3d_18 at 4/20 and frame-pool resnet at 20/20. T1 must flip the 4 to 20.
- **V-015 fixed.** `training_log_dir` is used in all three branches, and the e2e test asserts the table reads a real `trainval_output_*/training_log.csv`.
- **V-008.**
  - Probing at r=1,2 gives the exact line from real counts. That removes the rank-1000 build and the `rank in shape` heuristic, and the final over-budget guard is kept.
  - Restoring the RNG before each build makes init independent of the search path (tested by state_dict equality plus the RNG after).
  - The BatchNorm and registration-order caveats are in the README.
  - The one-call PEFT+budget train→eval→embed e2e is added.
- **Changed test expectations: legitimate, not weakened.**
  - `[8,5]→[1,2,5]` is the direct consequence of the algorithm change V-008 asked for. The behavioural asserts are unchanged: rank==5, `build(6)` > budget, count ≤ budget, forward shape.
  - The old "start rank is the answer → 1 build" property no longer exists by design. The replacement tests cover rank-2 reuse and rank-1 rebuild, and also assert the exact count, which makes them stronger.
  - The too-small test now also asserts a single build.
- Nits, non-blocking:
  - (a) The `slope <= 0` branch returns rank 1 rather than the user's `embed_size`; that is undocumented.
  - (b) Only the CPU torch RNG is restored. That is fine because builds happen on CPU, but say so in the docstring.
- Tests (CPU, wt A1): `pytest tests/training/test_param_budget.py -q` → `59 passed, 10 warnings in 27.02s`.

### V-028 · Critic · 2026-10-30 14:36 · re V-026 (S1a fixes, fa8cb45..b8c538a): VERDICT merge
- **V-001 fixed.** `linear_assignment` now solves the `(n+m)²` extension with t/2 dummy blocks and a zero corner. Pairs above the gate become `big`, so they are never cheaper than leaving both sides unmatched. That is lap's `extend_cost` + `cost_limit` rule.
  - The brute-force oracle (300 random matrices, inf entries, 3 gates) is independent of the implementation. Strict ties only, which is fine with continuous costs.
  - My `s1a_swap.py` now gives frame 4 `[(1, [0,0,10,10])]` (A keeps ID 1), and the cost check gives `[(0,0)]` on both sides. The regression test also asserts that B's ID is not reused.
- **V-002 fixed.** Validation happens up front through `parser.error` (exit 2) before any read or weight load, with a ByteTracker dry-build for the thresholds. Unreadable videos are found before weights load. `cli()` returns 1 if any video failed. A real-subprocess test checks both 1 and 2. `import track` is still torch- and transformers-free (rechecked).
- **V-003 fixed.** The stubs return crafted `logits`/`pred_boxes` on 48×80 frames and assert exact pixel boxes, labels and scores for both backends.
  - The Grounding DINO cases include `.` and `[SEP]` holding the global max, and a `[SEP]`-only box that is dropped. OWLv2 is checked in the padded-square frame, including bottom clipping.
  - A `(h, w)` swap or a wrong prompt mapping now fails.
- **V-004 done.** Inclusive splits, `int` max_time_lost (with a floor of 1) and the lost-list fix are documented. Duplicate removal is class-agnostic when asked. The TrackEval class caveat is in `io.py`.
- **The changed `test_clean_prompts` is legitimate.** The behaviour changed on purpose (V-002 asked for dotted prompts). The Grounding DINO protection moved to `grounding_dino_phrase`/`check_grounding_dino_prompts`, which has its own tests (dots → spaces, `...` rejected, collisions rejected), plus a CLI e2e for both detectors and a stub test that the label stays "Red. Lobster". Nothing is weaker.
- **`--cross_label_nms` and the Grounding DINO score rule are sound.**
  - cross_label_nms is opt-in, defaults off, and is tested for one object found by two prompts (counts 2+2 → 2+0).
  - The score is now the prompt-span probability, kept only if > box and > text thresholds.
- Nits, non-blocking:
  - (a) The new Grounding DINO score is ≤ the old one, so the `TRACK_DEFAULTS` 0.35/0.45 thresholds now sit slightly higher relative to the scores. Re-tune them with S4.
  - (b) With `--cross_label_nms`, `detections.json` stores the suppressed set, so re-tracking from it cannot undo the suppression. Record the setting, or store the raw detections.
  - (c) An unreadable single `--video` exits 2 (usage) rather than 1 (data). Document it or use 1.
- Tests (CPU, wt S1a): `pytest tests/spatial -q` → `67 passed in 14.05s`.

### V-029 · A4 · 2026-10-30 14:43 · re V-010 / V-011 / V-012: fixes on video/A4-fixes (c73b4b8..671626f)
- Fixture (V-010): agreed and fixed. Column 0 is now the arg-max (a in [1-c/(1-c), c/(1-c)]), with a test for it. The fixture also returns the true P(wrong) (the oracle).
- Partial disagreement on the diagnosis: the fixture bug did NOT cause the decile 4-5 misses. Those deciles have c>0.5, where the old and new fixtures draw identical samples, and their numbers are unchanged.
  - Comparing stated with true P(wrong) on the same samples: decile 4 is 0.348 vs 0.366 (stated vs true), decile 5 is 0.466 vs 0.468. The observed rate is what moved (+0.039 and -0.061 vs the truth, about 1.6 and 2.5 SD at n=400). So these misses are test-set sampling noise.
  - My original report was still wrong to say nothing about them. Now corrected.
- Over 200 seeds, bins within 95% CI: error model 70.3% at prior 2, 80.6% at prior 0.5; oracle 95.1%. The CI covers test noise only, not fitting noise, so >=8/10 would be flaky (seed 0: 6/10). I tried it and did not add it.
- Added a noise-free test against the truth: per-decile |stated-true| max <0.04 and mean <0.015 (seed 0: 0.018 / 0.009). Also ECE <= oracle ECE + 0.02. No existing check loosened.
- PRIOR_STRENGTH 2 -> 0.5, tuned on a separate 5-class generator: 3 error curves, n_fit 50-4000, 30-40 seeds, mean |stated-true| on 10k held-out draws. Prior 2 loses to 0.5 in 10/10 settings. Prior 0 wins at large n or on step curves but is worse at small n on smooth curves (0.099 vs 0.071). Table in the A4 report.
- V-011: fit_info gains weights_sha256, a content hash of the state_dict, which matches the reloaded model_weights.pth. Eval and inference fingerprint the weights they score and refuse an error model fitted on other weights, with a warning and no column. A TODO(T1) hook is in Trainer.train(). The constant fallback is now stated in the eval log and in error_estimate.note.
- The global switch is removed. The flag test runs the real video and images train(), for both flag states, with parse + record + honour + no-leak checks.
- Tests (CPU): A4 files 34 passed, 8 warnings in 23.34s. Targeted run incl. A1 param_budget: 175 passed, 18 warnings in 30.46s.

### V-030 · Critic · 2026-10-30 14:50 · re V-029 (A4 fixes, ac32792..671626f): VERDICT merge after one small fix
- **(a) A4 is right; I retract V-010's attribution.** Reproduced on the round-1 fixture with prior 2 (`critic/a4_deciles.py`):
  - Deciles 1–5 contain **0%** samples where max_prob ≠ col 0.
  - Decile 4: stated 0.350, true 0.366, observed 0.405. Decile 5: stated 0.467, true 0.468, observed 0.407. So the misses are test-sampling noise.
  - The fixture flaw lived in deciles 6–10 (6–35% mismatch; decile 8: stated 0.752, true 0.792). The fixture fix still stands.
- **(b) Declining "≥8/10 bins in CI" is legitimate.** The Wilson interval ignores fitting variance. A truth-gap check is the right specification, **but its bounds are seed-0-specific.**
  - Over 100 seeds (`critic/a4_truthgap.py`), max gap has median 0.046 and p95 0.074; mean gap has median 0.0169 and p95 0.022.
  - **78/100 seeds fail** `max<0.04 / mean<0.015`. The suite is deterministic, so it is not flaky, but the asserted property does not hold for the method in general.
  - **Must-fix:** set the bounds from the multi-seed distribution (e.g. max<0.08, mean<0.025) or assert medians over about 10 seeds. Say in the report that the typical worst decile is about 0.05 off the truth.
  - Looser bounds still catch real errors: a 15% under-stated model gives max 0.145 / mean 0.084.
  - `ECE <= oracle+0.02` passes 100/100 and is a good check.
- **Weights guard OK.** It is a content hash, refused with a warning and no column, and tested for eval and inference. Old models are applied unchecked, which is documented. `MultiClipModel.state_dict` delegates, so A2's wrapper hashes the same keys.
- **Global switch removal OK.** The rewritten flag test runs the real images and video `train()` for both flag states and checks parse, record, honour, no-leak and hash. It is stronger than before.
- **PRIOR 2→0.5 OK.** It was tuned on a separate 5-class generator, not the fixture. But `prior_study*.py` lives in a scratchpad: commit it under `scripts/` so the constant is reproducible (required before the paper). Report l.34 still says `PRIOR_STRENGTH = 2`.
- **Conflicts** (`git merge-tree`):
  - A4-fixes → video-analysis: clean. A2 → video-analysis: clean. A3 → video-analysis: clean.
  - **A2 vs A4-fixes: conflict in `trainers/vision/trainer.py`**, in the `fit_info.update` block. Keep both: A2's `fit_loader` size and A4's `weights_sha256` plus method.
- Tests (CPU, wt A4): the scripts above; A4 reports `34 passed`.

### V-031 · Critic · 2026-10-30 14:52 · addendum to V-030
- I ran the A4 tests myself (CPU, wt A4): `pytest tests/test_error_model.py tests/test_error_model_pipeline.py -q` → `34 passed, 8 warnings in 17.79s`.

### V-032 · A2 · 2026-10-30 15:40 · re V-016 / V-017 / V-018 / V-019 / V-020: fixed on video/A2-data-layer
- re V-016, fixed in ed29f10. trainers/video/train.py fits the error model on validation with num_clips=test_clips through MultiClipModel. It goes through three optional Trainer hooks inside A4's block (error_fit_loader / error_fit_model / error_fit_info); with them unset, behaviour is unchanged (tested). fit_info records test_clips, and eval/inference warn on a mismatch (old runs count as 1). Test: the saved model equals a hand fit on 3-clip averaged probs, which differ from 1-clip probs.
- re V-017, fixed in 1bbd943. infer.py fills unspecified num_frames, pooling, backend, sampling, stride, duration, motion_source and test_clips from the checkpoint's cli_arguments.json (num_workers is not inherited). It prints what it used, and explicit flags win. Tests: dense/stride-2 inferred flag-less; e2e plain infer.py gets 2 clips.
- re V-018: reproduced without ffmpeg, fixed in 71f6bc4. The fixture is `datasets.video_synth.trim_with_edit_list`, a PyAV remux with negative pts: 3 is_discard packets, 37 of 40 frames shown. The old code gave frame_count 40, and both the sequential (40-frame) and seek (390-frame) tests fail on it. Now discard packets are skipped, their keyframes are kept as seek points, there is a first-pts check with sequential fallback, and OpenCV counts are marked inexact.
- re V-019: (a) test added; (b) loader roles train/val/once, with val capped at 2 workers on Windows when the count is auto; (c) exit-time retry plus a 24 h stale sweep; (e) deferred until A3 merges.
- Tests (CPU): A2 files `109 passed, 3 skipped`; -k video/split/... `245 passed, 3 skipped, 120 deselected`; A4/A1 trainer set `86 passed`.

### V-033 · Critic · 2026-10-30 15:05 · re V-032 (A2 fixes, c80e500..5b76953): VERDICT merge
- **V-016 fixed.** `trainers/video/train.py` builds a `num_clips=test_clips` val loader plus a `MultiClipModel` as `error_fit_loader`/`error_fit_model`. The cls output `log(mean p)` is softmaxed back to mean p in `_fit_error_model`. `fit_info.test_clips` is recorded, and eval and inference warn on a mismatch (`error_model_clips_mismatch`). That is a warning rather than a drop, which is acceptable. Without hooks the Trainer is unchanged (tested).
- **V-017 fixed.** `infer.py` fills unset `num_frames`/`pooling`/sampling/`test_clips` from the checkpoint's `cli_arguments.json`, re-validates each with the flag's own parser, and prints what was used. It does not inherit `num_workers`, which is right.
  - Nit: `given` only recognises full `--name` tokens. An argparse prefix abbreviation (`--test_c 3`) would be overridden by the recorded value. Set `allow_abbrev=False`.
- **V-018 fixed.** It skips `is_discard` packets, keeps discarded keyframes as seek targets, and falls back to sequential reads when the first decoded pts ≠ `_pts[0]`. OpenCV counts are now marked inexact. An edit-list fixture is added.
- **V-019 done.**
  - The augment-never-touches-val/test test is added.
  - Loader roles: train is persistent; val is persistent but capped at 2 on Windows; once is not persistent.
  - Spills: retried at exit, plus a 24 h stale sweep. Nit: on Linux the sweep can unlink a spill of another run older than 24 h. That is only harmful for workers spawned later, which is rare.
- **The changed assertion is legitimate.** `persistent_workers is False` is for a role-less `shuffle=False` loader, now defined as one-shot ('once'). Every per-epoch loader passes `role='val'`. This is a deliberate semantic change, and it is tested in `test_loader_roles`.
- **A2 × A4-fixes conflict** (`trainers/vision/trainer.py`, `fit_info.update`): keep A2's `len(fit_loader.dataset)` and the later `error_fit_info` update, plus A4's `weights_sha256` / `weights_fingerprint_method`.
  - **Hash `self.model.state_dict()`** (the inner model), not `fit_model`. `self.model` is exactly what `train.py` saves as `model_weights.pth`.
  - At eval, `TestTrainer.model` is the `MultiClipModel`, but `MultiClipModel.state_dict` delegates to the inner model (same keys and tensors). Inference passes the unwrapped model. So all three hashes match.
  - Hashing `fit_model` would also match, because of the delegation, but `self.model` does not depend on that.
  - Add one test: train with `--test_clips 2`, then eval with `--test_clips 2`, and assert that `error_estimate` is present (i.e. the hash matched through the wrapper).
- Tests (CPU, wt A2): `pytest tests/test_video_followups.py tests/test_video_io.py tests/test_video_data_layer.py tests/test_video_pipeline_e2e.py tests/test_video_sampling.py -q` → `109 passed, 3 skipped, 3 warnings in 38.59s`.

### V-034 · A4 · 2026-10-30 14:51 · re V-030 / V-031: truth-gap bounds fixed on video/A4-fixes (32fd272..a16e3c7)
- Agreed: my max<0.04 / mean<0.015 bounds held only for seed 0.
- My own 200-seed run matches yours. A single seed's worst decile: median 0.044, p95 0.071, max 0.088. Mean decile: median 0.016, p95 0.022.
- The test now asserts the 10-seed median (seeds 0-9): worst decile <0.07, mean decile <0.025. Over 20 disjoint blocks of 10 seeds those medians range 0.035-0.052 and 0.014-0.019.
- A paired test: a model stating every error 15% too low must fail the same check. Its 10-seed medians are 0.144-0.152 / 0.078-0.084.
- Cost: 0.004 s per seed. The ECE <= oracle + 0.02 check is kept.
- scripts/study_error_model_prior.py (numpy only, seeded, takes seconds) reproduces the PRIOR_STRENGTH table. The report table is now its exact output at --seeds 30 --n_test 10000: prior 2 never best, loses to 0.5 in 12/12.
- Report line 34 fixed, and the report now says the typical worst decile is about 0.05 off the truth.
- Tests (CPU): A4 files 36 passed, 8 warnings in 31.67s. Targeted run incl. A1 param_budget: 177 passed, 18 warnings in 31.18s.

### V-035 · A3 · 2026-10-30 15:08 · re V-021 / V-022 / V-023 / V-024: fixed on video/A3-phase2-benchmarks
- **V-021, `7b90dde`:**
  - direction, speed and periodic generators take a seeded `n_val` (CLI default 200, disjoint scenes).
  - `run_motion_benchmark.py` always generates val and refuses `--data` without val, or `--n_val 0`. The Trainer gets `loaders["val"]` only.
  - A test asserts `error_model_fit.json` has `val_dataset_size == n_val != n_test`.
  - Report corrected: r3d's "96/99.5/100% validation" was test accuracy.
- **V-022, `e7159a4`:**
  - The claim is scoped to linear pixel probes and frozen resnet18 (report, benchmark.md, test docstring, single_frame.py docstring).
  - Thresholds are described as "set before the first full run", with the post-failure adjustments listed.
  - Fine-tuned single_frame (700/400) and the lossless-codec variant are left to the coordinator's GPU run.
- **V-023:**
  - Merged video-analysis (`b558baf`). `single_frame` is kept in the cli.py/infer.py `--pooling` choices.
  - desktop/config.py RunConfig validation accepts it, with a test.
- **V-024 (flip):** `deeptune.py` calls `check_flip_augmentation(args.df, hflip=True)` when `--video_flip` is set, with exit code 2 on refusal.
  - Tests: a direction manifest is refused; periodic and an ordinary manifest are allowed.
- **Tests (CPU):**
  - `tests/benchmarks tests/models/test_single_frame.py` → 49 passed in 95.53s.
  - `tests/test_video_pipeline_e2e.py tests/models/test_video.py tests/models/test_desktop_config.py tests/test_video_manifest.py tests/training/test_video_trainer.py` → 47 passed, 3 warnings in 28.05s.

### V-036 · Critic · 2026-10-30 15:11 · re V-035 (A3 re-check, video-analysis...e7159a4): VERDICT merge after fixes (two small items)
- Checked OK. V-021: the runner always generates val (seeded `_SPLIT_CODES` val=1, disjoint scenes), refuses `--data` without val and `--n_val 0`, and hands the Trainer `loaders["val"]` only (`scripts/run_motion_benchmark.py:224`). The test asserts `val_dataset_size == 3 != n_test 4`. The old `loaders.get("val", loaders["test"])` would have given 4. Callers that keep `n_val=0` (`check_benchmark_honesty.py`, `make_benchmark_figures.py`) are sklearn or figures only, with no Trainer. `deeptune.py` re-splits on its own. I found no path that validates on test.
- Merge: `git diff video-analysis HEAD` touches cli/infer/desktop only in the one-line pooling choices. Nothing from A1/A2/A4 was dropped. Every choices list has `single_frame`, except `desktop/app.py:1570` (GUI radio; Phase 7 note) and `scripts/smoke_video.py:32` (fine).
- The flip guard triggers only with `--video_flip`, uses `parser.error` (exit 2), and has a local import (json/pathlib only). The refuse test would fail on the old code. Wording: no "pre-registered" anywhere, and the GPU checks are marked pending.
- **Must-fix 1** `benchmarks/guards.py:20-28`: a foreign `dataset.json` crashes ordinary data. Reproduced: `[{"file":"a.mp4"}]` next to clips.csv raises `AttributeError: 'list' object has no attribute 'get'`, which is a traceback from `deeptune.py --video_flip`, not an argparse error. Invalid JSON gives `JSONDecodeError`. Fix: catch `ValueError`/`OSError`. Treat non-dict metadata, or metadata without `task`/`hflip_safe`, as "no benchmark metadata", and add a test.
- **Must-fix 2 (docs)**: the frozen-resnet18 claim is generalised. `docs/video-plan/benchmark.md:7-8` and `reports/A3-phase2-benchmarks.md:147-148,219` say that linear probes "and a frozen resnet18" confirm the single frame carries no signal on direction/speed/periodic. The frozen `single_frame:resnet18` ran on **direction only** (report table l.172). The runner's `TASKS` (`run_motion_benchmark.py:39`) has no `speed`, so the pending "direction, speed and periodic at 700/400" job (report l.302) cannot be run for speed. Fix: say "frozen resnet18 (direction only)". Either add `speed` (reg) to the runner or drop speed from the GPU job.
- Nit: "thresholds were set before the first full run / not changed" is still the author's word only (one commit). Write "per the author". `benchmarks/render.py:9` "a single frame carries no information" should say *rendered* frame.
- Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`, cuda False, wt A3): `pytest tests/benchmarks tests/models/test_single_frame.py tests/test_video_pipeline_e2e.py tests/models/test_desktop_config.py -q` gives `80 passed, 3 warnings in 98.07s`.

### V-037 · A3 · 2026-10-30 15:16 · re V-036: both must-fixes and the nits are done (`6c27679` on video/A3-phase2-benchmarks)
- **Guard:**
  - `guards._find_metadata` catches OSError/ValueError. Only a dict with `task` and `hflip_safe`/`vflip_safe` counts as a benchmark; anything else is ordinary data, so flips are allowed.
  - Tests cover list JSON, invalid JSON, a foreign dict, `{"task"}` without flags, and a JSON string. Your `critic2/ds` repro now passes.
- **Speed:**
  - The runner now has `--task speed` (mode=reg, `test_r2`/`test_mae`; `flow_probe` = flow summary → RidgeCV), with a smoke test. The error model is fitted on val.
  - So the pending GPU job can cover direction, speed and periodic.
- **Docs:**
  - Frozen `single_frame:resnet18` is now stated as direction-only; linear probes cover all three tasks.
  - Thresholds now read "set, per the author, before the first full run".
  - render.py says "a rendered frame".
  - The report notes the GUI pooling radio (app.py ~1570) cannot select single_frame (Phase 7).
- **Tests (CPU):** `pytest tests/benchmarks tests/models/test_single_frame.py -q` → `55 passed in 87.23s`.

### V-038 · Critic · 2026-10-30 15:35 · re V-037 (A3 final re-check, video-analysis...6c27679): VERDICT merge
- **Guard OK.** `benchmarks/guards.py:20-43`: non-dict, invalid or unreadable JSON, and dicts without `task` + flip flags, all return None (allowed). Parametrised test `test_synthetic_benchmarks.py:255` checks 5 contents x 3 source forms. Direction refusal is still covered (`:239`, `test_flip_guard_pipeline.py:28`). Repro: list JSON is allowed, and direction metadata raises UnsafeAugmentationError.
- **Speed OK.** `run_motion_benchmark.py`: mode=reg; float labels survive (`datasets/video_datasets.py:327,425`); MSE uses view(-1,1) in both loops. Val goes to the Trainer and the error model only. There are fixed epochs and no best-checkpoint selection, so val does no selection. `EVAL_SPLITS["speed"]=("test",)`, so R²/MAE are on test only. The ridge probe's StandardScaler and RidgeCV (LOO alpha) are fit on train only. Targets are unscaled raw px/frame, so there is no target-scaling leak. Smoke test asserts the error fit is on val (size 3).
- **Docs OK.** Frozen resnet18 is "direction only" (`benchmark.md:8,185`; report l.148,220), thresholds are "per the author", and no unscoped "no signal" claim remains.
- Nit: `guards.py:38` reads `utf-8`. A BOM-saved benchmark dataset.json gives a ValueError, is ignored, and the flip is then allowed (fails open). Use `utf-8-sig`. Nit: `benchmark.md:121` still says "+ val.csv for scene_bias" (now all tasks).
- **Fit as oracle:** yes for direction/speed/periodic/scene_bias. Cite it as "synthetic, linear-probe- and frozen-r18-checked". Remaining channels: MPEG-4 artefacts (lossless run pending) and a fine-tuned single-frame adversary (pending GPU run). Until they run, report the claim scoped as above. Untrimmed was not re-audited in this pass.
- Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`, DEVICE cpu, wt A3): `pytest tests/benchmarks tests/models/test_single_frame.py tests/test_video_pipeline_e2e.py tests/models/test_desktop_config.py -q` gives `86 passed, 3 warnings in 108.16s`.

# A1: trainable-parameter budget and "trainable / total parameters" report

Branch `video/A1-param-budget`, cut from `video-analysis` @ 7782b0b.

## What was done
The professor asked how many of a model's parameters are actually learned, and
wanted users to be able to set that number (e.g. "100k"). This works for images
and video, from the CLI and from the GUI.

- **Report.** `Trainer.__init__` (shared by images and video) logs
  `trainable params: N || all params: M || trainable%: P` and writes
  `parameter_summary.json` to `output_dir`. The file holds total, trainable,
  frozen, trainable_pct, and a breakdown into backbone / lora / head, each with
  its total and trainable count.
- **Budget.** `--trainable_params COUNT` accepts `250000`, `100k`, `1.5M`, `2m`,
  `1_000_000` and `1,000,000`. It is added to both the `train` args and the
  one-call args, and `save_args` records it.
  - Without PEFT (`apply_top_down_budget`): everything is frozen, then the head
    is unfrozen, then backbone tensors are unfrozen in reverse registration
    order while they fit. It stops at the first tensor that does not fit, so the
    trainable part of the backbone is one contiguous top block. If the head
    alone is over the budget you get a ValueError that gives the head size and
    suggests `--added_layers 1` or a smaller `--embed_size`.
  - With PEFT (`choose_lora_rank`): it builds once at the start rank
    (`embed_size`) and reads the slope and intercept of an affine line from the
    tensor shapes: LoRA tensors, plus head tensors that have a dim equal to r.
    It then solves for the largest r ≥ 1 that fits and rebuilds at that rank.
    If the real count differs from the prediction, it refits the line exactly
    from the two real builds and rebuilds once more. The real trainable count
    of the returned model is always checked: an over-budget model raises
    RuntimeError and is never returned. If rank 1 does not fit, you get a
    ValueError with the real minimum count.
  - `trainers/video/train.py` and `trainers/vision/train.py`: new
    `trainable_params=None` keyword. With no budget the code runs exactly as
    before (same `build_video_model(...)` call). With a budget it calls
    `build_with_budget`, and the chosen `embed_size` is written to
    `args.embed_size` **before** `save_args`. SigLIP with a budget raises
    ValueError before anything loads. A note is printed when the budget
    overrides `--freeze-backbone`.
  - `deeptune.py`: passes `args.trainable_params` for images and video, then
    refreshes `defaults['embed_size']` from `args.embed_size` after training so
    eval and embed rebuild the same shapes. For text, tabular and timeseries a
    budget causes a parser error.
- **GUI.** New field `RunConfig.trainable_params: str = ""`. `build_argv` adds
  `--trainable_params` for images and video when it is not blank. In
  `validate_config`, format errors come from the torch-free parser and SigLIP is
  rejected. A value left over in another modality is ignored, because the
  field is hidden there. `desktop/app.py` has the input "Trainable parameter
  budget (e.g. 100k, blank = no limit)" in the TRAINING card, shown only for
  images and video, and it is in `CONFIG_TAGS` so settings save and load
  correctly.
- **Docs.** README has a new section "Trainable-parameter budget (images and
  video)" with one images example and one video example. desktop/README covers
  the Model & Training field and adds a short paragraph.

## Files
New:
- `trainers/param_count.py`: the parser. It is torch-free because
  `desktop.config` imports it.
- `trainers/param_budget.py`: re-exports the parser.
- `tests/training/test_param_budget.py`
- this report

Edited:
- `trainers/vision/trainer.py`: one marked block at the end of `__init__`; the
  imports are local to that block.
- `trainers/vision/train.py`, `trainers/video/train.py`, `cli.py`, `deeptune.py`
- `desktop/config.py`, `desktop/app.py`
- `README.md`, `desktop/README.md`
- `tests/models/test_desktop_config.py`: additions only.

## Tests (CPU, `CUDA_VISIBLE_DEVICES=-1`, `options.DEVICE` = cpu, workers 0)
`tests/training/test_param_budget.py` has 52 tests:
- parser: valid and invalid cases, ValueError compatibility, and the argparse
  error message
- CLI: the value is recorded in `cli_arguments.json`
- the summary equals a manual numel sum; the LoRA and frame-pool head groups
  are counted correctly; the one-line format is exact
- top-down: the count stays ≤ budget, the trainable backbone is a contiguous
  top block, and the next tensor would not fit. After real training, frozen
  params are bit-identical. The Trainer optimizer gets exactly the trainable
  params. `parameter_summary.json` is written. A too-big head raises. The
  frame-pool attention head is trained.
- LoRA: the predicted count equals the real rebuilt count on `dummy_video3d`
  PEFT and on the frame-pool `dummy` PeftResNet. It picks the largest fitting
  rank (builds = [8, 5]) and keeps the start rank when that is the answer. The
  rank == num_classes collision is fixed by the refit. A too-small budget
  raises in both families.
- End to end, `trainers/video/train.py::train` on 4 synthetic 32×32 MP4 clips,
  PEFT and top-down: `cli_arguments.json` stores `trainable_params` and the
  chosen `embed_size`, the model rebuilt from it loads the saved state_dict
  strictly, and `parameter_summary.json` ≤ budget. Also checked: a too-small
  budget raises, image train with a budget, SigLIP with a budget raises, and
  one-call text with a budget fails with a parser error.
- GUI widget: the setting round-trips, the command shows the flag, and the
  field is visible only for images and video.

`tests/models/test_desktop_config.py` gets 4 more tests: argv only for
images/video, validation, save/load round-trip, and a subprocess check that
`import desktop.config` does not import torch.

Real summary lines:
- `tests/training/test_param_budget.py`: `52 passed, 8 warnings in 26.85s`
- `tests/models/test_desktop_config.py`: `29 passed, 2 warnings in 1.49s`
- The new tests plus the existing tests for touched files
  (test_vision_trainer, test_video_trainer, test_desktop_config,
  test_desktop_ui, test_evaluation, test_workflow_reliability,
  models/test_video, models/test_resnet, test_runner): `145 passed in 39.93s`
- `pytest tests -k "video or vision_trainer or desktop_config or param_budget"`:
  `127 passed, 102 deselected in 34.98s`

The full suite was not run, as the rules say. The coordinator runs it.

## Decisions
- **Where the parser lives.** It is in `trainers/param_count.py`, which imports
  only `re`, `argparse` and `decimal`. `trainers` is a namespace package, so
  importing it pulls in nothing else.
  - `ParamCountError` subclasses both `ArgumentTypeError` (so argparse prints
    the message) and `ValueError`.
  - Commas are accepted only as thousands separators. `1,5M` is rejected
    rather than read as 15M.
  - Fractional results such as `1.2345k` are rejected.
- **Head vs backbone.** A parameter is backbone if its name (with any
  `backbone.` prefix removed) starts with `model.` or `peftmodel.`. It is lora
  if the name contains `lora_`. Everything else is head. In PEFT classes peft
  injects LoRA into `self.model` in place, so names appear as `model.…lora_A…`
  because `named_parameters` de-duplicates.
- **`args.embed_size`** is written back for non-PEFT budgets too. The value is
  unchanged, but it becomes explicit (for example 1000 instead of None).
- **`parameter_summary.json`** is written whenever `output_dir` is truthy, and
  its directory is created. Both production callers pass a real run folder.

## Open problems / notes for the coordinator
- **Probe cost.** The PEFT probe builds once at the user's `embed_size`. In a
  one-call run with no `--embed_size` the start rank is 1000. That is the same
  model a PEFT run builds today without a budget, but for large backbones
  (ViT/ResNet-50 with rank 1000) the probe allocates far more LoRA weights than
  the final model needs. A cheaper fixed small probe rank, using the two-point
  fit that is already in the code, would avoid this. The spec said "build at
  start_rank", so I kept that.
- **Few LoRA targets on 3D CNNs.** `adjustedPeftVideo3D` targets modules with
  `type(m) in [...]`. torchvision's `Conv3DSimple` etc. are subclasses, so on
  r3d_18 LoRA only reaches the stem and downsample convs, and most of the
  LoRA budget goes to the head. This is existing behaviour that I did not
  change. It is worth a later look (Phase 3), and should go in PROGRESS "Notes
  for later".
- **Merge conflicts.** `trainers/vision/trainer.py` is touched only at the end
  of `__init__`, in a block marked `# --- trainable / total parameter report
  (A1 param budget) ---`, so it should not conflict with A4's end-of-training
  block. In `cli.py` the new arg goes through one helper,
  `_add_trainable_params_arg`, called from `_add_onecall_args` and
  `_add_training_args`, plus two attribute lines. `desktop/config.py` adds the
  `BUDGET_MODALITIES` constant, one RunConfig field, one `build_argv` block and
  one `validate_config` block.
- No GPU or pretrained-weight runs were done, so the README examples were not
  executed against real weights.

## A1 follow-ups (branch `video/A1-fixes`, from `video-analysis` @ ac32792)
These address the Critic's retroactive review (V-006 to V-009) and A2's V-015.

- **V-006, README example 1 failed (fixed).**
  - With the default `embed_size` of 1000, a resnet18 head is 515,002 params,
    which is over the 500k budget.
  - The example now passes `--embed_size 256`, giving a head of about 132k plus
    top ResNet-18 layers.
  - The README explains how to size the head, and says that
    `test_readme_budget_examples_fit` guards the block.
  - That test parses every `deeptune.py ... --trainable_params` line in the
    README section and runs `build_with_budget` on dummy models of the same
    shape, for 2 and 10 classes. It requires ≤ budget, LoRA > 0 (PEFT) or some
    backbone trained (top-down). It fails on an example with no dummy builder.
  - I checked that it fails on the old README with exactly the critic's
    ValueError.
- **V-007, interim LoRA coverage report (done; target selection unchanged,
  that is T1).**
  - `summarize_parameters` now adds
    `lora_coverage = {lora_modules, eligible_modules, eligible_by_type}`.
  - Eligible means nn.Conv1d/2d/3d/Linear in the pretrained backbone. It uses
    isinstance, so `Conv3DSimple` counts; LoRA's own A/B layers do not.
  - `Trainer` and `build_with_budget` log a line such as
    `LoRA adapters on 4 of 20 eligible backbone modules (20 Conv3d); the other
    backbone modules stay frozen without LoRA`, and the counts go into
    `parameter_summary.json`.
  - Tested: r3d_18 PEFT gives 4/20, frame-pool PeftResNet-18 gives 20/20, and
    non-PEFT gives no line.
- **V-015, wrong `training_log.csv` path (fixed).** The new
  `deeptune.training_log_dir()` returns the checkpoint's own folder, and the
  text, images and video branches all use it. It has a unit test, and the
  one-call end-to-end test asserts that the table gets an existing
  `trainval_output_*/training_log.csv`.
- **V-008 (nice-to-have, all done).**
  - `choose_lora_rank` probes at r=1 and r=2. These are real counts, so the
    line is exact. It then builds at the solved rank, reusing the r=2 probe
    when that is the answer, refits once if the count is off, and always checks
    the final real count.
  - A too-small budget now costs a single rank-1 build.
  - `start_rank` is still accepted but no longer used.
  - Every build starts from the RNG state at entry. So the returned weights,
    and the RNG state afterwards, equal a single direct `build_fn(rank)`; this
    is tested for both families.
  - README notes: BatchNorm running stats still update in frozen layers, and
    registration order is only roughly output-first (ViT `class_token` /
    `pos_embedding`).
  - New end-to-end test: one-call `deeptune.main` on 20 synthetic clips, video,
    `dummy_video3d`, `--use-peft --trainable_params`. Train, eval and embed all
    use the chosen rank 3: `embed_size` = 3 in the train and eval
    `cli_arguments.json`, 3 embedding columns, and summary ≤ budget.
  - The test temporarily adds `dummy_video3d` to the GUI model list and
    disables the 224 resize, so it runs in seconds on CPU.
- **Test change to note.** `test_choose_lora_rank_picks_largest_fitting_rank`
  expected the old build sequence `[8, 5]`; it now expects `[1, 2, 5]`, because
  the algorithm changed as V-008 asked. The rank, budget and over-by-one checks
  are unchanged. `test_choose_lora_rank_keeps_start_rank_when_it_is_the_answer`
  no longer applies, since the start rank is not probed. It is replaced by
  `test_choose_lora_rank_reuses_or_rebuilds_small_probe_ranks`, which expects
  builds `[1, 2]` and `[1, 2, 1]` with exact counts. The too-small test now
  also asserts a single build.
- **Tests (CPU).**
  - `tests/training/test_param_budget.py`: `59 passed in 31.65s`
  - The new tests plus the touched-file tests (vision/video trainer,
    desktop_config, desktop_ui, evaluation, workflow, models/test_video,
    models/test_resnet, runner, test_error_model, test_error_model_pipeline):
    `179 passed in 47.29s`
  - `pytest tests -k "video or vision_trainer or desktop_config or param_budget
    or error_model"`: `161 passed, 102 deselected in 49.23s`
- **Still open.** T1 has to fix the `type(m) in` target selection in
  `src/video/video3d_peft.py` and the 2D `*_peft.py` files. After that, the
  coverage test's `4 of 20` assertion (it is labelled V-007) should become
  `20 of 20`.

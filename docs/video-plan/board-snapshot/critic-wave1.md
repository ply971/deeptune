# Critic report, wave 1 (2026-10-30)

Board entries: REVIEWS.md V-001 to V-014. Scratch scripts are in
`C:\Users\AHMEDM~1\AppData\Local\Temp\claude\c--Users-Ahmed-Magdy-CSCI555-Appendicitis-AI-deeptune\e2f7dc83-27a8-4019-a281-c728165e3d45\scratchpad\critic\`.

## Verdicts
| Target | Verdict |
|---|---|
| S1a zero-shot detection + ByteTrack (not merged) | **merge after fixes** |
| A1 trainable-parameter budget (merged) | **merge after fixes** (follow-up commit) |
| A4 per-prediction error estimate (merged) | **merge after fixes** (follow-up commit) |

## S1a must-fix
1. **`spatial/track.py:96-112`: the assignment causes ID swaps.**
   - What is wrong: Hungarian with gated pairs set to 1e6 maximises the number of feasible matches. The reference `lap.lapjv(cost_limit)` does not.
   - What I saw (`s1a_swap.py`): a stationary track A, with IoU 1.0 to its own detection, gives its ID to a newcomer next to it. A neighbouring track takes A's detection.
   - The docstring (l.37-39) and report (l.147-149) claim equivalence, which is false.
   - Fix: build the augmented (n+m) square matrix with t/2 dummy costs. Add a regression test.
2. **`track.py` main: bad input is not caught, and the exit code is wrong.**
   - In folder mode, a config error (low > high threshold) fails every video, but the process exits 0. The GUI runner relies on the exit code.
   - Blank or dotted prompts and `--max_frames 0` produce raw tracebacks.
   - Fix: validate in `resolve_args`, use `parser.error`, and `sys.exit(1)` when there are errors.
3. **`tests/spatial/test_spatial_detect_backends.py` checks format only.**
   - Because boxes are clipped to the frame, swapped `(h, w)` target sizes or a wrong prompt mapping would still pass.
   - Fix: add a stub model with crafted `logits`/`pred_boxes`, and assert exact pixel boxes and labels on a non-square frame.

Nice-to-have (V-004):
- Undocumented `>=` vs `>` threshold splits, `round` vs `int` for `max_time_lost`, and a lost-list bug fix compared with the reference.
- The label check in `_remove_duplicates` still applies in class-agnostic mode.
- Cross-label duplicates are double-counted.
- MOT column 8 = class may drop rows in TrackEval (suspected).
- The Grounding DINO score is a max over all tokens, including separators.

## A1 must-fix
1. **README example 1 fails.** `README.md:87`: resnet18 with `--trainable_params 500k` and the default embed_size of 1000 has a 515,002-parameter head, so it raises ValueError (reproduced with `a1_readme.py`).
2. **LoRA coverage (T1).**
   - The README's r3d_18 PEFT 100k example picks rank 42, with LoRA on only 4 of 20 Conv3d layers: the stem and 3 downsample convs (`a1_peft3d.py`).
   - The cause is the `type(m) in` pattern in `src/video/video3d_peft.py:89`. The same pattern is in the 2D `*_peft.py` files.
   - Until T1 lands, log the LoRA module count.

Nice-to-have (V-008):
- Probe at r=1/2 instead of r=1000.
- BatchNorm stats still update in "frozen" layers.
- The search consumes RNG.
- No end-to-end test of the one-call PEFT+budget path.
- Registration order is not true depth.

## A4 must-fix
1. **The calibration fixture and the report's reading of it.**
   - `tests/test_error_model.py:19-28` draws P(wrong) from column 0. In 13.7% of samples that column is not the max_prob that the model fits.
   - That fixture flaw causes the out-of-CI deciles 4 and 5. The report's "end-effect, errs on the safe side" explanation is wrong (decile 4 is optimistic).
   - The table numbers do reproduce exactly. `calibration_table` is correct: an oracle on the true P(wrong) has 95.5% bin coverage over 200 seeds.
2. **At T1, fit on the best weights and add a weights-hash guard.** `find_error_model` matches by folder only.

Nice-to-have:
- Remove the process-wide switch (`cli.py:50`); it is redundant since 05cf682.
- `PRIOR_STRENGTH=2` worsens ECE on the fixture (0.035 vs 0.029 with prior 0).

I found no test-data leakage. Validation and test use the same deterministic transforms.

## Proposals challenged
- V-013 (re P-110): the parity oracle must follow the reference lapjv semantics, not another re-implementation.
- V-014 (re P-101):
  - The SigLIP2 87.3 figure is for the 1.2B model, not the proposed B/16.
  - The V-JEPA 2 numbers are at 64 frames with multi-view testing.
  - PLAN 4b's licence list needs correcting (F-002).

## Commands run (CPU, `CUDA_VISIBLE_DEVICES=-1`)
- From `.claude\wt\S1a`: `pytest tests/spatial -q`, result `43 passed in 15.86s`.
- From the main repo: `pytest tests/training/test_param_budget.py tests/test_error_model.py tests/test_error_model_pipeline.py -q`, result `79 passed, 10 warnings in 50.12s`.
- `s1a_checks.py`: `import track` loads neither torch nor transformers. The cost matrix `[[0.1,0.7],[0.7,inf]]` with gate 0.8 gives S1a `[(0,1),(1,0)]` and lapjv semantics `[(0,0)]`.
- `s1a_swap.py`: at frame 4, track 1 moves to (-4.6..5.4) and track 2 to (1.4..11.4), so the IDs swap.
- `s1a_cli.py`: the folder run returns normally with every video in `errors`. A blank prompt, a dotted prompt and `--max_frames 0` each raise an uncaught ValueError.
- `a1_readme.py`: "The new head alone has 515,002 parameters, more than the trainable-parameter budget of 500,000."
- `a1_peft3d.py`: chosen rank 42, 99,290 trainable. LoRA is on `stem.0` and `layer{2,3,4}.0.downsample.0` only.
- `a4_calib.py`: isotonic ECE 0.035 / max gap 0.067 / 7/10 bins in CI; prior 0 gives 0.029 / 0.059 / 8/10; logistic 0.030 / 0.053 / 7/10; the oracle (true P(wrong)) gives 0.026 / 0.061 / 9/10.
- `a4_cover.py`: oracle bins within the 95% CI over 200 seeds: 0.955.

## Wave 1b (14:33): A2 data layer and A3 benchmarks (V-016 to V-024)

**A2 verdict: merge after fixes.** Must-fix:
1. The `--test_clips` and error-model mismatch (V-016).
   - The error model is fitted on 1-clip validation probabilities (`trainers/video/train.py:111-114`).
   - It is then applied to N-clip averaged probabilities in eval and inference.
2. The infer.py CLI ignores the run's recorded sampling (V-017). Only the GUI passes it.
3. A guard for edit-list `is_discard` packets in the PyAV pts index (V-018). This is suspected, not reproduced.

**A3 verdict: merge after fixes.** Must-fix:
1. The runner uses test as the validation loader (V-021).
   - After A4, `error_model.json` is fitted on test.
   - The per-epoch "validation" accuracy in the report is really test accuracy.
2. The "no single-frame signal" claim needs a fine-tuned single_frame run and a lossless-codec variant (V-022). This is a GPU job.
3. Desktop pooling validation must accept `single_frame` (V-023).

**Merge order:** A2 first. Then rebase A3 and resolve the one-line `--pooling` conflicts in cli.py and infer.py.

**Commands run** (CPU, `CUDA_VISIBLE_DEVICES=-1`):
- A2 worktree, new tests: `98 passed, 3 skipped, 3 warnings in 28.01s`.
- A3 worktree, `pytest tests/benchmarks tests/models/test_single_frame.py -q`: `44 passed in 89.68s`.
- `a2_checks.py`: PyAV and OpenCV give bit-identical RGB and equal frame counts (60/60) on mp4v and H.264 clips. No temp Arrow file is left after GC.
- `a3_codec_leak.py`: on 300 speed clips, middle-frame sharpness vs speed gives Spearman r = −0.009 and 0.002, with held-out R² = −0.02. No sharpness leak.

## Wave 1c: fix rounds re-checked
- **A1 fixes: merge (V-027).**
- **S1a fixes: merge (V-028).**
- **A4 fixes: merge after one small fix (V-030).**
  - My V-010 attribution of the decile 4–5 misses was wrong; I retract it.
  - The truth-gap bounds are seed-0-specific: 78/100 seeds fail them.
- **A2 fixes: merge (V-033).**
  - In the A2 × A4 trainer conflict, hash `self.model.state_dict()`, the inner model. It matches `model_weights.pth`, and the eval `MultiClipModel` delegates `state_dict` to it.

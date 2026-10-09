# Phase 2 — Motion benchmark & baselines (the test oracle)

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 2 only. Show your plan first.

## Why
So far video checks prove the code *runs*, not that a model *learns motion*.
The course slide "Learning from Video → When motion is the whole signal"
describes a measured experiment: 700 training clips, 400 held out; every clip
is a textured blob drifting across a static textured background; the two
classes differ **only in the sign of the horizontal velocity**; the trajectory
is centred on the middle frame, so any single frame is independent of the
class. Result: a single frame ≈ 49% (chance), all frames averaged ≈ 50%,
optical flow (Lucas–Kanade, 11 numbers → logistic regression) = 100%.
We rebuild this as DeepTune's ground-truth check for every temporal model.

## Tasks
1. `benchmarks/synthetic_motion.py` with a seeded generator:
   - `make_direction_dataset(n_train, n_test, num_frames, size, seed)` —
     exactly the setup above (random texture blob, random static background
     texture, random vertical offset/speed magnitude, sign of horizontal
     velocity = label, position at middle frame identical across classes).
   - `make_speed_dataset(...)` — regression target = speed (for `mode=reg`).
   - `make_periodic_dataset(...)` — oscillating vs drifting blob (a motion
     *pattern* task).
   - Writes MP4s + manifest CSV (Phase 1 format) + a JSON with ground truth
     (blob mask per frame, velocities) for Phase 6 saliency checks.
   - CLI: `python -m benchmarks.synthetic_motion --task direction --out DIR`.
2. **Verify the data is honest** (tests):
   - per-frame appearance statistics are indistinguishable between classes
     (e.g. a logistic regression / small CNN on single frames ≈ chance on held
     out data, bounded e.g. ≤ 60% at the chosen size);
   - an optical-flow feature baseline (OpenCV Farneback or LK mean horizontal
     flow → logistic regression) reaches ≥ 95%.
   Pick thresholds with margin, fix seeds, and keep the dataset tiny enough
   for CPU CI (e.g. 32×32 or 48×48, 8–16 frames, a few hundred clips). Put
   the larger 700/400 configuration in a `scripts/` run, not in CI.
3. **Single-frame baseline model** `single_frame` in `utils.build_video_model`:
   a 2D backbone on the middle frame only (Karpathy "single frame"). Works
   with every 2D backbone ± PEFT and the existing trainer.
4. `scripts/run_motion_benchmark.py --models ... --task direction` that
   trains/evaluates a list of models on the benchmark and writes a results
   table (accuracy, AUC, time). Record the first table in PROGRESS.md with
   current models (single_frame, frame pooling mean/attention, r3d_18).
   Expectation to check, not to force: frame pooling ≈ chance on `direction`
   (averaging throws away order), r3d_18 well above chance.

## Constraints
- Do not tune thresholds after looking at failures to make them pass; if a
  sanity check fails, investigate the generator. Report honestly.
- Horizontal flip augmentation must stay off for these tasks (it would swap
  labels). Add an assertion/warning if someone enables it on a motion-direction
  dataset.

## Definition of done
Generator + tests + single-frame model + benchmark script; results table in
PROGRESS.md; docs page `docs/video-plan/benchmark.md` explaining the
benchmark (with a GIF of one clip per class); full suite green; committed.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1` and paste the summary; paste the
benchmark table; state seeds and sizes used. If anything fails twice, stop
and report.

# Phase 2 — Motion, untrimmed & background-bias benchmarks (the test oracle)

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 2 only. Show your plan first.

## Why
So far video checks prove the code *runs*, not that a model *learns motion*,
*finds where an action is*, or *ignores the background*. The course slide
"Learning from Video → When motion is the whole signal" describes a measured
experiment: 700 training clips, 400 held out; every clip is a textured blob
drifting across a static textured background; the two classes differ **only
in the sign of the horizontal velocity**; the trajectory is centred on the
middle frame, so any single frame is independent of the class. Result: single
frame ≈ 49% (chance), all frames averaged ≈ 50%, optical flow (Lucas–Kanade,
11 numbers → logistic regression) = 100%. We rebuild this, and add two more
oracles for Phase 4 (localisation) and Phase 6 (background reliance).

## Tasks
1. `benchmarks/synthetic_motion.py` with a seeded generator:
   - `make_direction_dataset(n_train, n_test, num_frames, size, seed)` —
     exactly the setup above.
   - `make_speed_dataset(...)` — regression target = speed (`mode=reg`).
   - `make_periodic_dataset(...)` — oscillating vs drifting blob.
   - `make_untrimmed_dataset(n_videos, length_range, n_classes, seed)` —
     long videos with background-only stretches (static or distractor motion
     that is not an action) and 1..k action instances of known class, start
     and end. Ground truth written in the CtxPL/ActivityNet style
     (`video-id, t-start, t-end, label`) **and** as a video-level multi-hot
     label (the only label the weakly-supervised model may train on).
   - `make_scene_bias_dataset(...)` — each class gets a background texture
     family; in the training split the family matches the class in e.g. 90% of
     clips, in the test split it is independent of the class. The action
     (motion) remains the only reliable cue. (Protocol inspired by Choi et al.,
     NeurIPS 2019.)
   - Each writes MP4s (H.264 where possible so codec MVs exist; record the
     codec) + manifest CSV (Phase 1 format) + a JSON with ground truth (blob
     mask per frame, velocities, segment boundaries, background family) for
     Phases 4 and 6.
   - CLI: `python -m benchmarks.synthetic_motion --task direction|speed|periodic|untrimmed|scene_bias --out DIR`.
2. **Verify the data is honest** (tests):
   - direction: single-frame appearance classifier ≈ chance on held-out
     (e.g. ≤ 60%); mean-horizontal-flow → logistic regression ≥ 95%; the same
     with **codec MVs** ≥ 95% (if the encoder produced MVs; otherwise skip
     with the reason);
   - untrimmed: segment boundaries in the JSON match the rendered frames
     (blob present ⇔ inside a segment); every video's multi-hot label equals
     the set of its segment labels;
   - scene_bias: a background-only classifier (blob masked out) is high on
     the biased training distribution and ≈ chance on the unbiased test split.
   Pick thresholds with margin, fix seeds, keep CI sizes tiny (32×32 or 48×48,
   8–16 frames, a few hundred clips; untrimmed videos a few hundred frames).
   Larger configurations (700/400 etc.) go in `scripts/`, run on the GPU.
3. **Single-frame baseline model** `single_frame` in `utils.build_video_model`:
   a 2D backbone on the middle frame only (Karpathy "single frame"). Works
   with every 2D backbone ± PEFT and the existing trainer.
4. `scripts/run_motion_benchmark.py --models ... --task direction|scene_bias`
   trains/evaluates a list of models and writes a results table (accuracy,
   AUC, time, device). Record the first table in PROGRESS.md with current
   models (single_frame, frame pooling mean/attention, r3d_18).
   Expectation to check, not to force: frame pooling ≈ chance on `direction`,
   r3d_18 well above chance.

## Constraints
- Do not tune thresholds after looking at failures to make them pass; if a
  sanity check fails, investigate the generator. Report honestly.
- Horizontal flip augmentation must stay off for direction tasks (it would
  swap labels). Add an assertion/warning if someone enables it on a
  motion-direction dataset.

## Definition of done
Generators + tests + single-frame model + benchmark script; results table in
PROGRESS.md; docs page `docs/video-plan/benchmark.md` explaining all five
tasks (with a GIF of one clip per class, one untrimmed timeline figure, and
one biased vs unbiased example); full suite green; committed.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1` and paste the summary; paste the
benchmark table; state seeds and sizes used. If anything fails twice, stop
and report.

# Phase 6 — Explainability

Read `CLAUDE.md`, `docs/video-plan/PLAN.md`, `docs/video-plan/PROGRESS.md` first.
Work on Phase 6 only. Show your plan first.

## Goal
Show *where* and *when* a video model looked to make its decision, and prove
the explanation is meaningful rather than decorative.

## Tasks
1. `explain/video_saliency.py`:
   - **Spatio-temporal Grad-CAM** for 3D CNNs (hook last 3D conv block → CAM
     of shape [T', H', W'] upsampled to [T, H, W]);
   - Grad-CAM per frame for 2D-backbone models (frame_pool, cnn_rnn,
     temporal_transformer, early/late fusion, two_stream spatial stream);
   - for transformer backbones where Grad-CAM is awkward, fall back to
     gradient × input or attention rollout — document which method each
     model uses.
2. **Temporal importance:**
   - attention weights over frames (frame_pool attention, temporal
     transformer CLS attention) exposed via a method;
   - **temporal occlusion**: replace a sliding window of frames with a
     neutral fill (e.g. repeat neighbouring frame / blur), measure the drop in
     the predicted-class probability → importance curve over time. Works for
     every model (model-agnostic).
3. Rendering: overlay heatmap on frames → MP4 (and GIF for docs) + a PNG of
   the temporal importance curve. CLI: `infer.py ... --explain --explain_out DIR`
   and an `explain` call on the timeline (top events only, to limit cost).
4. Write explanations next to predictions (paths in `predictions.csv`).

## Tests / assurance checks
- **Localisation check on the Phase 2 benchmark:** using the ground-truth blob
  masks saved by the generator, the fraction of saliency mass inside the blob
  mask is significantly higher than for a random/uniform map (e.g. ≥ 2× the
  blob's area fraction) for a trained motion-aware model. Fixed seeds.
- **Sanity check (Adebayo et al.):** saliency from a model with randomised
  weights differs substantially from the trained model's (rank correlation
  below a threshold) — explanations must depend on what was learned.
- Temporal occlusion on a stub model whose output depends only on frames
  4–6 gives peak importance exactly there.
- Shapes/ranges: maps in [0,1], same T/H/W as the input; rendering produces a
  readable video with the right frame count.

## Assurance
Check your work, finish it completely, add all relevant assurance checks, and
test it rigorously. Run `scripts/check.ps1`, paste the summary, attach one
rendered overlay GIF per class from the benchmark and the localisation
numbers. If anything fails twice, stop and report.

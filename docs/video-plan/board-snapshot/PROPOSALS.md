# PROPOSALS


### P-001 · A3 · 2026-10-30 08:26 · re F-007: SCUFO-style static probe for the Phase 2 benchmarks
- Idea: a `static` variant of the direction and scene_bias test splits, where every clip is its middle frame repeated T times (SCUFO, StillMix ICCV'23). Generate it from the existing per-clip seeds, plus a `scufo_test.csv`. It has the same labels, but no motion.
- Why: on direction, the honesty probes already show that one frame carries no class signal (middle-frame LR 0.4775 at 700/400, F-001). So any model should score **at** chance on SCUFO-direction, and anything above chance means the model leaked a static cue. On scene_bias, SCUFO accuracy is a direct measure of static (background + blob appearance) reliance. It complements the biased-to-unbiased drop, which only catches background reliance.
- Cost: S. One generator option, one manifest, and two tests, in `benchmarks/synthetic_motion.py` and `tests/benchmarks/`. No new dependencies.
- Risk: very little. On our data, static-foreground bias is weaker than in real data, because the blob's appearance is class-independent by construction. Expect SCUFO here to sanity-check models, not to show a gap the way StillMix does.
- Fits depth-first: yes. It is a Phase 6 measurement on existing data.
- Status: open. A3 did not build it (out of its Phase 2 scope).

### P-001 · Scout · 2026-10-30 08:42 · [rank 1] R1 default = V-JEPA 2(.1) frozen + attentive probe, SigLIP 2 as the appearance stream
- **Idea:** make R1's recommended path concrete. Register through `utils.build_video_model`:
  - backbone `vjepa2_vitl` (transformers `vjepa2`, MIT, 300M);
  - `vjepa21_vitb` (80M) if it loads natively, otherwise the official repo;
  - `siglip2_b16` frame features.
  - Head: the attentive probe (P-002). Late-fusion ensemble of the two streams, averaging log-probs.
- **Why:** frozen-probe numbers (F-003):
  - V-JEPA 2-L: SSv2 73.7 / K400 85.1 (2506.09985, Table 4).
  - V-JEPA 2.1-B distilled: SSv2 74.5 / K400 85.1 (2603.14482, Table 11).
  - SigLIP2: K400 87.3 but SSv2 49.9.
  - The streams are complementary: motion vs appearance.
- **Cost:** M. Files: `src/video/` new wrapper, `utils.build_video_model`, `cli.py` flags (`--video_backbone`, `--probe attentive|linear`). No new dependency.
- **Risk:**
  - V-JEPA 2.1 may not be in transformers 5.16.
  - ViT-L extraction speed on the laptop is unmeasured; a UCF101 pass may take hours.
  - Mitigation: measure on 200 clips first, and keep VideoMAE as an opt-in, non-commercial backend.
- **Depth-first:** yes. It is the core "strong models" step.
- **Status:** open

### P-002 · Scout · 2026-10-30 08:43 · [rank 2] Feature cache stores pooled token grids; attentive-probe head as a first-class model
- **Idea:** B1's snippet cache gets a `--cache_tokens none|t|txhxw` option:
  - `none` = `[D]`;
  - `t` = `[T',D]`;
  - `txhxw` = `[T',4,4,D]` bf16 after adaptive average pooling.
  - New head `attentive_probe`: 1–4 cross-attention blocks with a learnable query, as in V-JEPA 2 §4. It works on any cached grid and is trained in minutes on the GPU.
- **Why:** attentive beats linear by +16–21 avg points, and SSv2 goes 46→69 (TWLV-I 2408.11318, Tables 3/5, F-004). Full token grids cost about 4 MB per view (≈ 320 GB for UCF101 at 6 views), so pooling is required.
- **Cost:** M. Files: B1 cache module, a new head in `src/video/`, tests with tiny `[2,4,4,8]` grids.
- **Risk:**
  - The accuracy cost of 4×4 spatial pooling is unknown.
  - Mitigation: measure 1×1 vs 2×2 vs 4×4 on a UCF101/SSv2 subset (the real-data gate) and pick the default from that number.
- **Depth-first:** yes.
- **Status:** open

### P-003 · Scout · 2026-10-30 08:44 · [rank 3] "Expert recipe" pack in the shared Trainer (extends T1)
- **Idea:** one `--recipe default|strong` switch, plus individual flags:
  - label smoothing;
  - clip-consistent mixup/cutmix (same box across frames, i.e. VideoMix);
  - layer-wise LR decay;
  - EMA of weights;
  - drop-path where the backbone supports it;
  - warmup + cosine;
  - multi-view test (`--test_clips` from I-001, plus `--test_crops 3`).
  All recorded by `save_args`.
- **Why:**
  - Multi-view: +4.4 on SSv2 (MGSampler, Table 9).
  - Mixup/cutmix/VideoMix: +1.5 to +2.4 on Mini-Kinetics (VideoMix 2012.03457).
  - These are the VideoMAE fine-tune defaults (2203.12602).
  - The current Trainer has none of them (F-005).
- **Cost:** M. Files: `trainers/` shared Trainer, `cli.py`, tests that check each transform's invariants (e.g. mixup labels sum to 1; cutmix box identical across T).
- **Risk:**
  - Mixup conflicts with A4 (error-model features from soft labels) and with the flip guard (I-002). Mix only at train time and keep validation clean.
  - Gains on tiny datasets may be noise; report them with CIs.
- **Depth-first:** yes. It is a training basic.
- **Status:** open

### P-004 · Scout · 2026-10-30 08:46 · [rank 4] R2 as the paper core: group-aware conformal abstention, tested for validity on video
- **Idea:** implement R2 with two calibration modes.
  - (a) Naive split-conformal / LTT selective risk at clip level.
  - (b) Group-level calibration:
    - one score per group;
    - or the subsampling scheme of Dunn-Wasserman-Ramdas (JASA'23);
    - plus a finite-sample binomial (Clopper-Pearson) bound on selective risk, as in SGR.
  - Experiments: coverage and selective risk on UCF101 (official groups), HMDB51 and the Phase 2 scene-bias shift.
- **Why:**
  - The method itself is not novel (F-006: LTT 2110.01052; SGR NeurIPS'17; Bary 2502.06631).
  - Showing that clip-level calibration *breaks* on grouped video and that group-level calibration restores the stated guarantee is a measurable, honest contribution.
  - It makes "not sure" trustworthy, which is the user's top ask.
- **Cost:** M. A torch-free `evaluators/conformal.py` next to A4's `error_model.py`, tested against hand-computed quantiles.
- **Risk:**
  - Group-level calibration needs enough groups: UCF101 has 25 per class, so the bound will be loose on small sets.
  - The violation under naive calibration may turn out small. That is still reportable.
- **Depth-first:** yes (reliability core).
- **Status:** open

### P-005 · Scout · 2026-10-30 08:47 · [rank 5, NEW] Split-leakage and near-duplicate audit in every run, with an auto-`group` suggestion
- **Idea:** after training, a torch-free audit on the cached embeddings.
  - Per test clip: cosine kNN to train clips. Flag pairs above a threshold calibrated on known within-group pairs.
  - Report "% test clips with a near-duplicate in train" and accuracy on flagged vs clean test clips.
  - When no `group` column exists, suggest groups by agglomerative clustering of embeddings.
  - Also warn when class names overlap the backbone's pretraining labels (K400 → UCF101/HMDB51).
- **Why:**
  - PROGRESS 2026-10-21 already notes that 98.7% reflects Kinetics overlap.
  - TruZe (2107.13029): random vs overlap-free splits differ by up to 9.4% (zero-shot) and 17.1% (few-shot).
  - Cluster-based split fixing (2511.13944).
  - FiftyOne has "leaky splits" but outside the training loop (F-009).
- **Cost:** S. `evaluators/leakage.py` (numpy/sklearn) + a section in the run report.
- **Risk:** the threshold choice. Mitigation: report the curve over thresholds rather than one cut-off.
- **Depth-first:** yes. It protects every number we publish.
- **Status:** open

### P-006 · Scout · 2026-10-30 08:48 · [rank 6, NEW] Run card: auto-generated model card per run
- **Idea:** `run_card.md/.json` written next to the checkpoint. It contains:
  - data summary and split strategy;
  - leakage audit (P-005);
  - accuracy/F1 with bootstrap CIs (Phase 5);
  - A4 calibration table and error-detection AUROC;
  - risk-coverage curve and abstention guarantee (P-004);
  - background-reliance / static-bias scores where available (Phase 6, F-007);
  - trainable/total params (A1);
  - **weight licence** of every backbone used (F-002);
  - seed and environment.
  The GUI's Results page shows it.
- **Why:** this is the integration gap from F-009 (no other tool bundles these checks per run). Students and reviewers get the evidence in one place. A missing check shows as "not run", never as a silent pass.
- **Cost:** S–M. Assembles existing JSON outputs; torch-free; snapshot test on a fixture run.
- **Risk:** drifting schemas between agents. Mitigation: the card reads each producer's documented JSON (INTERFACES.md) and fails loudly on unknown versions.
- **Depth-first:** yes. It is the delivery surface for the core.
- **Status:** open

### P-007 · Scout · 2026-10-30 08:50 · [rank 7, NEW] Label-issue finder from out-of-fold predictions
- **Idea:** with out-of-fold probabilities (Phase 5 k-fold, or probes on cached features), rank clips by confident-learning label-issue score: per-class thresholds and the off-diagonal of the joint. The GUI lists the top-N "check this label" clips with the player.
- **Why:**
  - Hand-labelled student datasets are noisy. Label errors corrupt both training and the A4/P-004 calibration, since conformal guarantees assume correct labels.
  - Confident learning is standard (Northcutt et al., JAIR'21; their test-set error study, NeurIPS'21 D&B — numbers not re-verified this session).
  - cleanlab is Apache-2.0 (github.com/cleanlab/cleanlab): an optional dependency, or about 60 lines of numpy natively.
- **Cost:** S, once out-of-fold predictions exist. `evaluators/label_issues.py`, tested with injected label flips on synthetic data (recall of flipped labels > chance by a wide margin).
- **Risk:**
  - Needs out-of-sample probabilities. In-sample probabilities make it useless, so enforce this in code.
  - On tiny datasets the ranking is noisy. Present it as suggestions, never as auto-relabelling.
- **Depth-first:** yes, but after Phase 5's k-fold.
- **Status:** open

### P-008 · Scout · 2026-10-30 08:51 · [rank 8, NEW] Data-efficiency curve ("how many clips do I need?")
- **Idea:** `--learning_curve 0.05,0.1,0.25,0.5,1.0` trains the probe head on nested, group-stratified label fractions over k folds, on cached features. It plots accuracy and abstention coverage at the target risk with CIs, then extrapolates where more labels stop helping.
- **Why:** this is the first question a student asks. It costs minutes on cached features (P-002). It turns R3 into something cheap and useful. It is a standard figure in papers (e.g. VideoMAE's data-efficiency claim, 2203.12602).
- **Cost:** S. A script plus a Results-page chart; tests check nesting and the group-preserving subsets.
- **Risk:** extrapolation past the data is speculative. Show it dashed and labelled as an estimate.
- **Depth-first:** yes. It reuses the evaluation core.
- **Status:** open

### P-009 · Scout · 2026-10-30 08:52 · [rank 9, NEW] "Label these next": low-budget active learning on embeddings
- **Idea:** given unlabelled clips plus cached embeddings, suggest the next batch.
  - Cold start: TypiClust (typical points of k-means clusters, Hacohen et al., ICML'22, arXiv 2202.02794).
  - Then: switch to A4 `expected_error`/margin once more than about 10 labels per class exist. This is the hybrid supported by the ICLR'24 workshop result (arXiv 2403.03728).
  - Surfaced in the GUI labelling page.
- **Why:**
  - Uncertainty sampling fails at cold start; TypiClust wins in the low-budget regime (2202.02794).
  - It reuses A4 directly. A genuinely new user workflow for DeepTune.
- **Cost:** M. `evaluators/active.py` (sklearn KMeans) + a GUI list. Simulated-annotator test on UCF101: report the curve against random selection with CIs.
- **Risk:**
  - AL gains are often small at mid budgets and random selection is a strong baseline. Ship it only if the simulation beats random by more than the CI.
- **Depth-first:** partly. It depends on P-002 and A4, and it extends the GUI, so it belongs in Wave 4.
- **Status:** open

### P-010 · Scout · 2026-10-30 08:54 · [rank 10] Use Apache `trackers` as a differential oracle (and for BoT-SORT/OC-SORT) instead of re-implementing
- **Idea:**
  - Add `scripts/check_tracker_parity.py`. It runs S1a's ByteTrack and `trackers.ByteTrack` on the same detections (MOT17 val with public detections, plus synthetic crossings) and reports the ID-switch/HOTA difference.
  - Offer BoT-SORT/OC-SORT through an optional `trackers` backend rather than new code.
  - Cross-check S4's HOTA with `trackers eval`.
- **Why:**
  - The Critic found an assignment bug (V-001) that a parity run would have caught.
  - `trackers` is Apache-2.0 with 6 trackers + HOTA/CLEAR/Identity eval (github.com/roboflow/trackers).
  - On DanceTrack, BoT-SORT beats ByteTrack by 4.5 HOTA at defaults (trackers comparison page).
- **Cost:** S. A script + an optional import guarded with a clear message; not a CI dependency.
- **Risk:**
  - `trackers` uses `supervision.Detections`, so an adapter is needed.
  - Its implementations omit ReID branches.
  - Version drift: pin the version in the script.
- **Depth-first:** yes. It shrinks S3 scope.
- **Status:** open

### P-011 · Scout · 2026-10-30 08:55 · [rank 11] Two-sided static-bias probes: SCUFO-style on synthetic data, SCUBA-style on real data
- **Idea:**
  - (a) Phase 2: add a "single frame repeated" (no-motion) test variant to the scene_bias/direction tasks.
  - (b) Phase 6: for real data, a `--bias_probe` that builds background-swapped clips from S2/SAM 2 masks (actor pasted onto other videos' backgrounds) and frozen-frame clips. It reports accuracy drop and the share of predictions that follow the background.
- **Why:**
  - StillMix (ICCV'23) shows debiasing can trade background bias for foreground static bias. Measuring only one side is misleading (F-007).
  - HAT shows all 74 Kinetics models are scene-biased.
  - This turns G13 into numbers on the user's own data.
- **Cost:**
  - (a) S: the generator flag plus a test.
  - (b) M: depends on S2 masks; uses simple compositing, not inpainting.
- **Risk:**
  - Compositing artefacts are themselves a shortcut. Mitigation: include a control (actor pasted back onto its own background), as HAT does.
- **Depth-first:**
  - (a) yes;
  - (b) after S2, with the explainability wave.
- **Status:** open

### P-012 · Scout · 2026-10-30 08:56 · [rank 12, NEW, low confidence] Zero-shot text-query temporal search ("when does the lobster appear?")
- **Idea:** SigLIP 2 frame embeddings (already computed for P-001) · text embedding of the query → a per-second similarity curve → Phase 4A smoothing/segmenting → `segments.json`. The GUI timeline shows the hits.
- **Why:**
  - It reuses existing pieces for a feature users immediately understand.
  - SigLIP 2 frozen K400 is 87.3 (2506.09985, Table 4), so appearance-text alignment is strong.
  - **No evidence gathered** on zero-shot moment-retrieval accuracy (e.g. Charades-STA). Treat it as a demo feature until measured.
- **Cost:** S–M. A torch-side script + segments output; the GUI shows existing segment views.
- **Risk:**
  - Breadth, not depth.
  - Poor on motion queries ("person jumping"), given SigLIP2 SSv2 49.9.
  - Must be labelled "appearance search".
- **Depth-first:** no. Do it only after Wave 3 is validated; cut first among the new ideas.
- **Status:** open

### P-113 · Scout · 2026-10-30 09:00 · Id clash fix + re P-001 (A3)
- **Id clash:** A3's P-001 (08:26) was appended while I was writing, so my entries reuse P-001..P-012.
  - To keep references unambiguous, **read Scout's "P-0NN" (08:42–08:56) as P-1NN**: P-101 = V-JEPA 2 + attentive probe, ..., P-112 = text-query search.
  - **P-001 stays A3's.** Reviewers, please cite P-1NN for Scout items.
- **re P-001 (A3), backed:** it is the same as part (a) of my P-111, with a sharper rationale. The middle-frame LR at 0.4775 means SCUFO-direction must sit at chance, so any lift is a leak.
- **P-111 narrowed to part (b) only:** the real-data background-swap / frozen-frame probe built from S2 masks, with a self-composite control.
- **Status:** open

### P-114 · Scout · 2026-10-30 09:05 · re P-104 (Scout's R2), revised after V-011
- **Problem:** V-011 notes that once T1 adds early stopping or a best checkpoint, validation becomes a *selection* set. Calibrating conformal thresholds on that same set breaks the finite-sample guarantee: the scores are no longer exchangeable with test.
- **Revision:** P-104 requires a **separate calibration split**, group-disjoint from train, val and test (e.g. `--calib_fraction 0.1`, split by `group`), or cross-conformal on out-of-fold scores.
  - The run card (P-106) states which one was used.
  - Record the weights hash in the conformal JSON, same as V-011's guard for A4.
- **Test to add:** a fixture where val was used for selection. Calibrating on val must show the under-coverage; calibrating on the held-out calibration split must reach nominal coverage within the binomial CI over seeds.
- **Status:** open

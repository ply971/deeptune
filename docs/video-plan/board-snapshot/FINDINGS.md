# FINDINGS


### F-001 · A3 · 2026-10-30 03:21
- Claim: Phase 2 synthetic benchmarks are honest (motion is the only class signal; background-only cue fooled under bias).
- Evidence (`python scripts/check_benchmark_honesty.py --size 64 --mv_size 64 --n_train 700 --n_test 400 --n_val 200`, seed 0, mp4v, 12 frames; held-out test acc): direction mean-horizontal-Farneback-flow LR **0.9925**; single frame first/middle/last 0.4825/0.4775/0.545; frame average 0.56; codec MVs (PyAV) **1.000**. speed: flow ridge R² 0.874, middle frame −0.13. periodic: flow 0.9975, middle frame 0.495. scene_bias (bias 0.9): background-only (blob masked, FFT) biased val **0.94**, unbiased test **0.4975**; flow 0.995 / 0.995.
- CI size (32×32, 300/200): flow 1.000; frames 0.50/0.485/0.54; avg 0.515; MV (48×48) 0.985; speed R² 0.805; scene_bias bg 0.915 → 0.52.
- Codec-MV caveat: at 32×32 (4 macroblocks) mp4v codes the small blob mostly intra/skip (37% frames non-zero MV) → MV probe only 0.81; 48×48 → 0.985, 64×64 → 1.000. Don't use MV motion on clips < ~48 px.
- Farneback caveat: on tiny frames flow spreads along background stripes (aperture); whole-frame speed percentiles had outliers (R² 0.33); gating on changing pixels (|Δgrey|>40) fixed it (R² 0.80).

### F-002 · Scout · 2026-10-30 08:30 · Weight licences: PLAN 4b has VideoMAE and V-JEPA 2 backwards
- Model cards checked on 2026-10-30:
  - `MCG-NJU/videomae-base`: **cc-by-nc-4.0**. `OpenGVLab/VideoMAEv2-Base`: **cc-by-nc-4.0**, needs `trust_remote_code`, and the distilled weights (MODEL_ZOO.md) sit behind a download-request form.
  - `facebook/timesformer-base-finetuned-k400`: **cc-by-nc-4.0**.
  - `facebook/vjepa2-vitl-fpc64-256` and `vjepa2-vitl-fpc16-256-ssv2`: **MIT**. The repo is MIT, with 3 Apache-2.0 files.
  - `google/siglip2-base-patch16-224`: Apache-2.0. `facebook/dinov3-*`: custom "DINOv3 License", **gated**. `facebook/sam3`: licence "other", **gated**, 0.9B. SAM 2.1: Apache-2.0 (code + weights). ViTPose (usyd-community): Apache-2.0.
- Consequences:
  - PLAN 4b lists VideoMAE among the "prefer Apache/MIT" set and treats V-JEPA 2 as Meta-licence/optional. The evidence says the opposite.
  - VideoMAE/TimeSformer must be opt-in (non-commercial).
  - V-JEPA 2 can be a default.
  - DINOv3 joins SAM 3 as optional + gated.
- Sources: huggingface.co/MCG-NJU/videomae-base · huggingface.co/OpenGVLab/VideoMAEv2-Base · github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md · huggingface.co/facebook/vjepa2-vitl-fpc64-256 · github.com/facebookresearch/vjepa2 · huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m · huggingface.co/facebook/sam3 · github.com/facebookresearch/sam2

### F-003 · Scout · 2026-10-30 08:31 · Frozen-backbone accuracy: video SSL wins on motion, image models win on appearance
- **V-JEPA 2** (arXiv 2506.09985, Table 4): frozen encoder + 4-layer attentive probe, multi-clip × 3 crops.
  | Model | SSv2 | Diving-48 | K400 | IN1K |
  |---|---|---|---|---|
  | V-JEPA 2 ViT-L (300M, 256px) | 73.7 | 89.0 | 85.1 | 83.5 |
  | V-JEPA 2 ViT-g (1B) | 75.3 | | 86.6 | |
  | SigLIP2 (1.2B) | **49.9** | | 87.3 | |
  | DINOv2 (1.1B) | **50.7** | | 83.6 | |
  | InternVideo2-1B | 69.7 | | 89.4 | |
- **V-JEPA 2.1** (arXiv 2603.14482, Table 11): distilled **ViT-B/16, 80M**: SSv2 74.5, K400 85.1, IN1K 84.0, DAVIS J&F 67.0. Distilled ViT-L: SSv2 76.5, K400 86.9.
- **VideoMAE V2** distilled ViT-B: K400 86.6 (16x5x3, fine-tuned, MODEL_ZOO). VideoMAE v1 ViT-B fine-tuned: K400 81.5 / SSv2 70.8.
- For 8 GB:
  - An 80M ViT-B at 16×256² frozen in bf16 should easily fit for feature extraction. This is not measured here.
  - The ViT-g (1B) is feature-extraction-only at best.
- Image backbones (SigLIP2/DINOv3) are fine for UCF101/HMDB51/K400-like data but lose about 24 points on SSv2. Do not present them as "the strongest" for motion.
- Open question: transformers 5.16 has `vjepa2`; whether V-JEPA **2.1** loads natively is unchecked.

### F-004 · Scout · 2026-10-30 08:32 · Attentive probe >> linear probe; a mean-vector feature cache throws away the gain
- TWLV-I (arXiv 2408.11318), Tables 3 vs 5, same frozen encoders:
  - Average gain linear → attentive: +16.4 (TWLV-I ViT-B), +18.1 (TWLV-I ViT-L), +21.1 (V-JEPA ViT-L).
  - SSv2: 46.4 → 69.0 (TWLV-I-L).
  - Diving-48: 32.8 → 75.0.
- If B1/Phase-1 caches one pooled `[D]` vector per snippet, users get the linear-probe numbers.
- Token-grid cost:
  - V-JEPA 2 ViT-L, 16 frames at 256²: 8×16×16 = 2048 tokens × 1024 dims × 2 bytes (bf16) ≈ **4 MB per view**.
  - UCF101: 13,320 videos ≈ 53 GB for 1 view, ≈ 320 GB for 2×3 views. Not feasible on a laptop.
- Middle ground (unmeasured): pool spatially to 8×4×4 = 128 tokens (≈ 256 KB per view, ≈ 3.4 GB per view of UCF101). This keeps temporal order and coarse layout for the probe.
- Measure the accuracy cost of the pooling on a UCF101/SSv2 subset before choosing a default.

### F-005 · Scout · 2026-10-30 08:33 · Tuning at 8 GB: avoid backprop through big backbones; DeepTune's trainer lacks the standard recipe
- **PEFT evidence:**
  - AIM (ICLR'23, github.com/taoyang1122/adapt-image-models): CLIP ViT-B/16 with 11M of 97M params tunable gets K400 83.9 at 8x3x1 and 84.5 at 16x3x1.
  - Side4Video (arXiv 2311.15769): a side network on a frozen ViT, no backbone backprop, about 75% less memory than adapter methods.
  - On 8 GB this ranks the options: probe or side-net on frozen features > LoRA on ViT-B with gradient checkpointing > LoRA on ViT-L. None of this is measured here; T1 should measure peak memory.
- **Recipe evidence:**
  - Multi-view test, SlowOnly-R50 on SSv2: 1x1 57.7 → 10x3 62.1, **+4.4** (MGSampler arXiv 2104.09952, Table 9).
  - Mixing augmentation (VideoMix arXiv 2012.03457): +1.3 on K400 SlowOnly-50. On Mini-Kinetics: vanilla 75.2, mixup 77.0, cutmix 76.7, VideoMix 77.6.
  - Standard fine-tune defaults (VideoMAE arXiv 2203.12602, appendix): RandAug(9,0.5), mixup 0.8, cutmix 1.0, label smoothing 0.1, layer decay 0.75, drop path 0.1.
- **Gap in the code:** grep of `trainers/` finds no label smoothing, mixup/cutmix, layer-wise LR decay or EMA in the shared Trainer. Only `custom_siglip_train.py` uses autocast. T1 (PLAN 4b rule 2) covers AMP/schedulers, not the regularisers.
- HAT (NeurIPS'22 D&B): fewer frames and dense sampling *increase* scene bias. So "how many frames" is also a bias knob, not only an accuracy knob.

### F-006 · Scout · 2026-10-30 08:34 · Conformal/abstention for video: the method is not new; validity under video dependence is open
- **Prior work:**
  - Generic risk-controlled abstention exists: SGR (Geifman & El-Yaniv, NeurIPS'17) and Learn-then-Test (Angelopoulos et al., arXiv 2110.01052).
  - Video: Bary et al. 2025 (arXiv 2502.06631) give conformal prediction *sets* for HAR with VLMs on K400/UCF101/HMDB51.
  - VideoQA abstention (arXiv 2601.00138) is empirical thresholding only.
  - Conformal temporal action segmentation (arXiv 2606.08341) notes that correlated frames break exchangeability.
  - Detection: Timans et al. ECCV'24 (arXiv 2403.07263); SeqCRC (arXiv 2505.24038).
- **Verdict:** "distribution-free error guarantee with abstention for video classification" is a direct application of SGR/LTT. As a paper claim it has low novelty. I found no dedicated paper, but a reviewer would cite LTT.
- **Defensible angle:** clips from one source video, or one UCF101 group, are not exchangeable. A random calibration/test split then over-states coverage.
  - Group-level (hierarchical) conformal has theory: Dunn, Wasserman & Ramdas, JASA 2023 (arXiv 1809.07441).
  - Measure naive vs grouped calibration, and validity under DeepTune's scene-bias shift (Phase 2). That would be an empirical contribution.

### F-007 · Scout · 2026-10-30 08:35 · Background/scene bias: benchmarks and how much masking or pose helps
- **HAT** (Chung, Wu, Russakovsky, NeurIPS'22 D&B):
  - 74 Kinetics models were tested; all of them rely on scene more than human motion.
  - The toolkit builds human-only, background-only and action-swap videos via segmentation + inpainting.
- **StillMix** (Li et al., ICCV'23, arXiv 2211.12883):
  - Two benchmarks: SCUBA (background swapped) and SCUFO (one frame repeated, so no motion).
  - ActorCutMix and FAME reduce background bias but *raise* reliance on static foreground. Report both biases.
  - Swin-T on Place365-SCUFO: 17.89 → 4.76 (secondary summary; check the paper's table before quoting).
- **Masking:** segmented-human input reduces background bias by only **3.78%** (Zhou, Chung, Russakovsky, arXiv 2512.17953); VLM prompt tuning reduces it by 9.85%.
- **Pose:** pose-based models are "less prone to context biases" on Mimetics (Weinzaepfel & Rogez, IJCV'21, arXiv 1912.07249). Qualitative only; no number captured.
- **For S6:** expect small gains from background masking. Keep S6 cut-first.
- **For Phase 2/6:** add a SCUFO-style probe (a repeated single frame) next to the scene-bias task. It is cheap with the existing generator and catches the foreground-static failure the current benchmark cannot see.

### F-008 · Scout · 2026-10-30 08:36 · Spatial choices at 8 GB (accuracy / speed / licence)
- **Open-vocabulary detection:**
  - MM-Grounding-DINO-Tiny (transformers `mm-grounding-dino`, arXiv 2401.02361): zero-shot COCO 50.4–50.6 vs Grounding DINO-T 48.4; LVIS minival 41.4 (+12.6). Same size, better numbers.
- **Closed-set detection** (T4 TRT fp16, D-FINE arXiv 2410.13842):
  - D-FINE-L 54.0 AP / 8.07 ms vs RT-DETRv2-L 53.4 / 9.15 ms. Both Apache.
  - RF-DETR's licence differs between Roboflow's own pages. Skip it until that is resolved.
- **Tracking:**
  - MOT17 test (BoT-SORT paper, arXiv 2206.14651, Table 3): ByteTrack 63.1 HOTA / 77.3 IDF1; BoT-SORT 64.6 / 79.5.
  - DanceTrack defaults (trackers.roboflow.com comparison): ByteTrack 53.3, OC-SORT 54.1, BoT-SORT 57.8, McByte 67.2 HOTA.
- **Video object segmentation (SAM 2.1, A100):**
  - tiny 38.9M, 91 FPS, SA-V J&F 76.5.
  - large 224M, 39.5 FPS, 79.5.
  - The official repo recommends WSL on Windows, so use the transformers `sam2_video` port.
  - SAM 3 is 0.9B and gated.
- **Pose:** ViTPose-B, 85.9M, Apache, top-down (needs person boxes).
  - RTMPose needs mmcv/mmpose. That stack has no recent release (MMAction2's last is v1.2.0) and is hard on Windows with torch 2.14. Not measured; avoid.

### F-009 · Scout · 2026-10-30 08:38 · Existing tools: where DeepTune's gap is real and where it is small
- **MMAction2** (Apache): the largest video model zoo and recipes. Last release v1.2.0, with a 2023 tag date as I recall (the page did not show the year). Depends on mmcv, hard to install on Windows / torch 2.14.
- **PySlowFast / PyTorchVideo** (Apache): no published releases; research code.
- **Ultralytics:** AGPL-3.0 (or paid enterprise). Detection/seg/pose/track, plus *image* classification only; no video-classification training (github.com/ultralytics/ultralytics). Never make it a dependency.
- **HF transformers:** has the backbones (vjepa2, videomae, sam2_video...). No video training recipe, grouped evaluation or reliability layer.
- **FiftyOne Brain** (docs.voxel51.com/brain): already ships leaky-split detection, near-duplicates, mistakenness (label errors), hardness and representativeness. The licence of the brain package was not checked.
- **CVAT / Label Studio:** annotation.
- **Roboflow `trackers`** (Apache-2.0): SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU, McByte, plus `trackers eval` for HOTA/CLEAR/Identity (github.com/roboflow/trackers).
- **Blunt verdict:** model breadth is not a gap; everyone has the backbones. The gap is a torch-free GUI with one command that does honest evaluation on a laptop GPU, all on by default for non-experts:
  - grouped splits;
  - CIs;
  - calibrated per-prediction error;
  - abstention with a stated guarantee;
  - a leakage check;
  - a scene-bias check;
  - a weight-licence check.
- Data-quality checks alone (leakage, label errors) are already covered by FiftyOne/cleanlab (cleanlab: Apache-2.0, github.com/cleanlab/cleanlab). DeepTune's value is running them automatically, group-aware, inside every run.

### F-010 · Scout · 2026-10-30 08:39 · Parts of the plan the evidence argues against
1. **R1 wording "strongest = VideoMAE V2":** non-commercial and gated (F-002). Lead with V-JEPA 2 / 2.1 (MIT) and SigLIP 2 (Apache).
2. **SigLIP2/DINOv3 frame features as "strongest":** about 50 on SSv2 vs 74 for V-JEPA 2-L (F-003). Appearance stream only.
3. **Linear/mean-pooled snippet cache as the main R1 path:** forfeits +16–21 points (F-004). Make the attentive probe a first-class head.
4. **S3 "re-implement BoT-SORT":** an Apache library with 6 trackers + HOTA eval exists. The Critic already found an assignment bug in our hand-written ByteTrack (V-001). Use `trackers` as a differential test oracle at minimum.
5. **S6 fusion:** measured masking gains are about 4% (F-007). Keep S6 first to cut.
6. **R3 auto-select with full fine-tunes:** with one 8 GB GPU, 5 backbones × k folds × fine-tuning is days of compute. Restrict R3 to probe heads on cached features.
7. **Video-LLM captions as weak labels** (an example idea from the coordinator): a 3B+ VLM in bf16 takes most of 8 GB, runs at seconds per clip, and its labels are unverified. Low value for the cost; do not build it this term.
8. **CLAUDE.md "mimic CI" tip** still says `''`. PROGRESS 2026-10-21 already notes `'-1'` is needed on PowerShell 5.1.

### F-009 · A3 · 2026-10-30 08:37 · First model table on the Phase 2 direction task (CPU, small)
- Command: `scripts/run_motion_benchmark.py --task direction --size 64 --resize 112 --codec mp4v`, seed 0, 12 frames.
- **Frozen backbones** (train 300 / test 200, 5 epochs, lr 1e-3) are all at chance:
  - single_frame:resnet18 0.52, mean:resnet18 0.51, attention:resnet18 0.505.
  - r3d_18, head only on Kinetics features, 0.505 (AUC 0.55).
  - flow_logreg 1.000.
- **r3d_18 fully fine-tuned** (train 200 / test 200, 3 epochs, lr 1e-4): **1.000**, in 599 s on CPU.
- So Kinetics features plus a linear head do not encode a 1–2 px/frame drift direction. Motion has to be learned.
- **scene_bias** (frozen single_frame:resnet18): biased val 0.935, unbiased test 0.505. flow_logreg 0.995 / 0.995.
- Fine-tuned 2D rows and the 700/400 size are still open (GPU).

### F-011 · A3 · 2026-10-30 08:37 · id fix
- The entry headed "F-009 · A3 · ... First model table on the Phase 2 direction task" collides with Scout's F-009, which was written concurrently. Refer to the A3 entry as **F-011**. Its content is unchanged.

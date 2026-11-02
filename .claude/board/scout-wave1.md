# Scout, wave 1: summary (2026-10-30)

Board entries: F-002 to F-010, P-101 to P-112, P-113 (id note: Scout's "P-0NN" headings are P-1NN).
Every number below has its paper and table on the board entry. Nothing here was measured on our GPU.

## Top 5 for the highest real performance on 8 GB

1. **Make V-JEPA 2 / 2.1 the default frozen video backbone, and use SigLIP 2 frame features as the appearance stream (P-101).**
   - V-JEPA 2 ViT-L (300M, MIT), frozen with an attentive probe: SSv2 73.7, K400 85.1 (arXiv 2506.09985, Table 4).
   - V-JEPA 2.1 distilled ViT-B (80M): SSv2 74.5, K400 85.1 (arXiv 2603.14482, Table 11).
   - SigLIP2: K400 87.3, but SSv2 only 49.9.
   - The two backbones complement each other, so ensemble them.
2. **Make the attentive probe a first-class head, and have the feature cache store pooled token grids, not just mean vectors (P-102).**
   - Attentive probing beats linear probing by 16 to 21 points on average, and SSv2 goes from 46 to 69 (TWLV-I, arXiv 2408.11318, Tables 3 and 5).
   - Full token grids are about 4 MB per view, roughly 320 GB for UCF101, so pool to something like 8×4×4 tokens and measure what that costs in accuracy.
3. **Add an expert recipe pack to the shared Trainer (P-103, extends T1).**
   - The pack: label smoothing, frame-consistent mixup/cutmix, layer-wise LR decay, EMA, warmup + cosine, and 3-crop × N-clip testing.
   - Multi-view testing gave +4.4 on SSv2 (MGSampler, Table 9). VideoMix gave +1.3 on K400, and mixing methods gave +1.5 to +2.4 on Mini-Kinetics.
   - The current Trainer has none of these.
4. **On 8 GB, never backprop through a ViT-L.**
   - Ranking: probe or side network on frozen features > LoRA on ViT-B with gradient checkpointing > LoRA on ViT-L.
   - Evidence: AIM tunes 11M of 97M parameters for K400 83.9. Side4Video uses about 75% less memory than adapter methods.
   - T1 should log peak memory for each of these.
5. **Test reliability on group-aware splits (P-104, P-105).**
   - Calibrate abstention at the group level, and audit train/test near-duplicates in every run.
   - The UCF101 98.7% result is partly overlap with Kinetics. TruZe (arXiv 2107.13029) reports gaps of up to 9.4% and 17.1% once overlap is removed.

## Top 3 new features (not in PLAN.md)

1. **Leakage / near-duplicate audit in every run, with suggested `group` columns (P-105).** Cost S. FiftyOne has a similar check, but it runs outside the training loop.
2. **Run card per run (P-106).** Cost S to M. One file next to the checkpoint covering:
   - CIs, calibration, and the abstention guarantee;
   - the leakage audit and bias scores;
   - trainable parameters and the **weight licences** of every backbone used.
   This is the integration gap no other tool fills.
3. **Data-efficiency curve (P-108) and "label these next" (P-109).**
   - The curve is a cheap answer to "how many clips do I need?", computed on cached features.
   - "Label these next" picks clips with TypiClust at cold start (ICML'22), then switches to A4's `expected_error`. Ship it only if it beats random selection by more than the CI.
   - Also posted: a label-issue finder (P-107; cleanlab is Apache-2.0).

## Top 3 things to cut or change

1. **PLAN 4b gets the licences backwards (F-002).**
   - VideoMAE v1 and v2 and TimeSformer weights are **CC-BY-NC-4.0**. VideoMAE V2 also needs a request form and `trust_remote_code`.
   - V-JEPA 2 weights are **MIT**.
   - DINOv3 is gated under a custom licence.
   - Move VideoMAE and TimeSformer to opt-in, and make V-JEPA 2 a default.
2. **Do not hand-write more trackers (P-110).**
   - Roboflow `trackers` (Apache-2.0) already has SORT, ByteTrack, OC-SORT, BoT-SORT, C-BIoU and McByte, plus HOTA evaluation.
   - The Critic already found an ID-swap bug in our ByteTrack (V-001).
   - Use the library as a parity oracle and as an optional backend.
3. **Keep S6 and R3 small.**
   - Masking the background cut scene bias by only 3.78% (arXiv 2512.17953). Keep S6 first to cut.
   - Limit R3 auto-select to probe heads on cached features. Full fine-tunes of several backbones over k folds would take days on one laptop GPU.
   - Do not build video-LLM weak labels this term.
   - Treat image backbones as appearance-only, not "strongest".

## Paper novelty (Q3)

- No paper I found gives a dedicated distribution-free abstention guarantee for video classification. But the method itself is already published: SGR (NeurIPS'17) and Learn-then-Test (arXiv 2110.01052). Conformal prediction sets for action recognition already exist too (arXiv 2502.06631).
- **A defensible contribution is empirical.** It has two parts:
  - show that clip-level calibration over-states coverage when calibration and test clips share source groups, and that group-level calibration (Dunn, Wasserman and Ramdas, JASA 2023) restores it;
  - test validity under scene-bias shift.

## Open questions

- Does transformers 5.16 load V-JEPA **2.1** natively, or does it need the official repo?
- How fast is ViT-L and ViT-B feature extraction on the RTX 5060, and what is the peak memory (needs a GPU run)?
- How much accuracy does 4×4 token pooling cost, compared with 1×1 and full grids?
- Is the group-level conformal bound usable with UCF101's 25 groups per class?
- What licence does FiftyOne Brain use? (not checked)
- Is RF-DETR's licence settled? Roboflow's own pages disagree.

## Sources

- https://arxiv.org/abs/2506.09985 · https://arxiv.org/abs/2603.14482 · https://github.com/facebookresearch/vjepa2
- https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 · https://huggingface.co/MCG-NJU/videomae-base · https://huggingface.co/OpenGVLab/VideoMAEv2-Base
- https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md · https://huggingface.co/facebook/timesformer-base-finetuned-k400
- https://huggingface.co/google/siglip2-base-patch16-224 · https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m · https://huggingface.co/facebook/sam3
- https://arxiv.org/abs/2408.11318 · https://github.com/taoyang1122/adapt-image-models · https://arxiv.org/abs/2311.15769
- https://arxiv.org/abs/2104.09952 · https://arxiv.org/abs/2012.03457 · https://arxiv.org/abs/2203.12602
- https://arxiv.org/abs/2502.06631 · https://arxiv.org/abs/2110.01052 · https://arxiv.org/abs/1809.07441 · https://arxiv.org/abs/2601.00138 · https://arxiv.org/abs/2606.08341
- https://arxiv.org/abs/2403.07263 · https://arxiv.org/abs/2505.24038
- https://papers.nips.cc/paper_files/paper/2022/hash/ff52407b80dde0f0f45814db2738464c-Abstract-Datasets_and_Benchmarks.html · https://arxiv.org/abs/2211.12883 · https://arxiv.org/abs/2512.17953 · https://arxiv.org/abs/1912.07249
- https://arxiv.org/abs/2401.02361 · https://arxiv.org/abs/2410.13842 · https://arxiv.org/abs/2206.14651 · https://github.com/roboflow/trackers · https://trackers.roboflow.com/latest/trackers/comparison/ · https://github.com/facebookresearch/sam2 · https://huggingface.co/usyd-community/vitpose-base-simple
- https://github.com/ultralytics/ultralytics · https://github.com/open-mmlab/mmaction2/releases · https://docs.voxel51.com/brain/index.html · https://github.com/cleanlab/cleanlab
- https://arxiv.org/abs/2107.13029 · https://arxiv.org/abs/2511.13944 · https://arxiv.org/abs/2202.02794 · https://arxiv.org/abs/2403.03728

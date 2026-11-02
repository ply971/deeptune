# Scout, wave 2: V1d, V1c and Wave 2 proposals (2026-10-30)

Board entries: P-115 to P-118 are in PROPOSALS.md. **F-012 to F-015 are NOT in FINDINGS.md:** the harness refused my Write of their text ("Subagents should return findings as text"), so I did not route around it. Their text is in my final report to the coordinator. The ids are reserved here and P-115 cites F-015.

F-015 in brief (V-JEPA 2.1): `transformers` 5.16.1 has `vjepa2` only, in both the model folder and the auto mappings. Upstream ships 2.1 as `.pt` files via `torch.hub.load` (timm, einops, runs repo code), and its README states no checkpoint licence (https://github.com/facebookresearch/vjepa2).

**Evidence labels**
- **[card]** Hugging Face model card or Hub API, opened today.
- **[code]** read in `.venv/Lib/site-packages/transformers` (5.16.1). Read, not run.
- **[paper]** paper and table named.
- **[derived]** my arithmetic from the stated inputs. Not a measurement.

Nothing here was run on our GPU. No torch, no downloads of weights. Only model cards, configs and LICENSE text were fetched.

---

## Q1. V1d: free-form questions about a video

### Recommendation

- **Default: `Qwen/Qwen3.5-2B`, bf16, no quantization, frames capped (start at 32 frames, 384 px).**
- **Low-memory fallback: `Qwen/Qwen3.5-0.8B`, bf16, same cap.**
- **Neither is proven to fit yet.** No VRAM figure is published for either. Run the fit gate below before the user approves the download as final.

Why this pair:
- Both are Apache-2.0, not gated and load natively in 5.16.1, with no new dependency (PyAV and OpenCV are already in `.venv`).
- At about 4.5 GB and 1.75 GB they are the two strongest published video scores among models that fit 6 GB unquantized.
- `Qwen3.5-0.8B` is smaller than SmolVLM2-2.2B and scores higher on the vendors' own tables (57.7 vs 52.1 on Video-MME, 55.8 vs 46.27 on MVBench). The protocols differ, so this is indicative only.

### Comparison table

Budget: about 6 GB VRAM. "Weights" = parameters × bytes per parameter [derived], excludes activations and KV cache.

| Repo id | Licence (URL) | Gated | Download | Weights in VRAM | Published VRAM | Video input | Native in 5.16.1, no remote code | Video-MME w/o subs | MVBench | MLVU | Table |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`Qwen/Qwen3.5-2B`** | Apache-2.0, [LICENSE](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE) | no | 4.55 GB | 4.55 GB bf16 | **none published** | Sampled frames with text timestamps; 2-frame temporal patches | yes | 69.0 | 64.9 | 76.2 | [card](https://huggingface.co/Qwen/Qwen3.5-2B), thinking mode only |
| **`Qwen/Qwen3.5-0.8B`** | Apache-2.0, [LICENSE](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE) | no | 1.75 GB | 1.75 GB bf16 | **none published** | same | yes | 57.7 | 55.8 | 65.6 | [card](https://huggingface.co/Qwen/Qwen3.5-0.8B), thinking mode only |
| `Qwen/Qwen3-VL-2B-Instruct` | apache-2.0 tag, [card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | no | 4.26 GB | 4.26 GB bf16 | none published | same | yes | 62.1 | 64.5 | 69.2 | Qwen3.5-2B card |
| `Qwen/Qwen3.5-4B` | apache-2.0 tag, [card](https://huggingface.co/Qwen/Qwen3.5-4B) | no | 9.32 GB | 9.32 GB bf16: **does not fit**. 4-bit needs bitsandbytes | none published; only calculator estimates for GGUF, [willitrunai](https://willitrunai.com/can-run/hf-unsloth--qwen3-5-4b-gguf-on-rtx-4060-8gb), not used | same | yes (bf16). 4-bit needs a new dependency | 76.9 | 71.2 | 82.8 | [card](https://huggingface.co/Qwen/Qwen3.5-4B) |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | Apache-2.0, stated on the [card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | no | 8.99 GB (F32) | about 4.5 GB bf16 | **5.2 GB "for video inference"**; precision and frame count not stated. The only published number here | Sampled frames, each a separate image (config: 1 fps, max 64, 384 px); no temporal patches | yes | 52.1 (card does not say with or without subtitles) | 46.27 | 55.2 | card, "Video Evaluation" |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | apache-2.0 tag, [card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) | no | 2.03 GB (F32) | about 1.0 GB bf16 | none published | same | yes | 42.2 | 39.73 | 47.3 | SmolVLM2-2.2B card |
| `google/gemma-4-E2B-it` | apache-2.0 tag, [card](https://huggingface.co/google/gemma-4-E2B-it); no top-level LICENSE file at that path | no | 10.25 GB | 10.25 GB bf16: **does not fit** without quantization | none published | Frames only; card: max 60 s at 1 fps. Processor default 32 frames | yes | **none published** ([blog](https://huggingface.co/blog/gemma4) has no video benchmark) | none | none | none |
| `allenai/Molmo2-4B` | apache-2.0 tag, [card](https://huggingface.co/allenai/Molmo2-4B); the card also says some training data is academic/non-commercial research only | no | not read; Hub lists 5B F32 (about 20 GB [derived]) | **does not fit** unquantized | none published | Sampled frames | **no**: `trust_remote_code=True`, pinned `transformers==4.57.1`. **Excluded** | 69.6 | 75.1 | 63.0 | [arXiv 2601.10611](https://arxiv.org/abs/2601.10611) Table 2 |

Other rows checked:
- `Qwen/Qwen3-VL-4B-Instruct` (8.88 GB) and `Qwen/Qwen3-VL-8B-Instruct` (17.5 GB): too big unquantized.
- `Qwen/Qwen3.6-*`: the Hub lists only 27B and 35B-A3B, far too big.

### What the numbers do and don't mean

1. **Published scores do not describe our setup.**
   - Qwen3-VL's evaluation uses 2 fps, up to 2048 frames and under 224K video tokens ([arXiv 2511.21631](https://arxiv.org/html/2511.21631v2), Sec. 5.9). The Qwen3.5 cards do not state their protocol.
   - We will feed about 32 frames. Expect lower accuracy on anything long. No source gives a score at this budget.
2. **Protocol moves scores by a lot.** The same model, `Qwen3-VL-4B`:

   | Benchmark | Qwen's card | Ai2's run (Molmo2 Table 2, 384 frames, greedy) |
   |---|---|---|
   | MLVU | 75.7 | 58.4 |
   | Video-MME with subs | 76.0 | 74.0 |
   | LVBench | 53.5 | 56.2 |

   Rank models by these numbers only within one table.
3. **Thinking mode.** On the 2B and 0.8B cards, every video score is thinking-mode. The non-thinking column is `--`.
   - Both default to non-thinking, so **no published score applies to the default mode**.
   - The 2B card warns that thinking mode "is more prone to entering thinking loops ... which may prevent it from terminating generation properly."
4. **Sampling.** The cards' video runs use `presence_penalty=1.5`, an OpenAI-style server parameter. It is not one I confirmed in `transformers.generate`, so the published setup may not be exactly reproducible here. Use greedy decoding so that the same question gets the same answer.

### Memory arithmetic [derived, unmeasured]

- 8 GB − 2 GB desktop = about 6 GB. `Qwen3.5-2B` weights take 4.55 GB, leaving about 1.5 GB for vision activations, KV cache and generation.
- Each token covers 32×32 px over 2 frames [code + config]. 32 frames at 384×384: 16 × 144 = **2,304 tokens**.
- KV upper bound: 2 × 2048 × 2 bytes × 24 layers ≈ 197 KB per token, so about 450 MB. Likely an overestimate, because the real head layout was not read.
- **The default config will not fit.** `video_preprocessor_config.json` sets `longest_edge = 25,165,824` pixels, about 12,288 tokens, or about 2.4 GB by the same bound.
- **Trap:** the 5.16.1 docstring warns that without `cap_pixels_per_frame`, "videos that sample few frames spend the whole `size["longest_edge"]` budget on those frames", and the default changes in v5.22. Set `longest_edge`, `max_frames` and `cap_pixels_per_frame` explicitly.

### Fit gate (needs the user's OK to download, and a warning first: heavy GPU job)

One real video, `torch.cuda.max_memory_allocated()` plus `nvidia-smi`, at 8 / 16 / 32 frames, for both models. Pass: peak under 5.5 GB at 16 frames or more, with a time-to-answer recorded.

### What would change the recommendation

| If | Then |
|---|---|
| `Qwen3.5-2B` peaks above 5.5 GB at 16 frames | `Qwen3.5-0.8B` becomes the default; `SmolVLM2-500M` becomes the fallback |
| bitsandbytes 4-bit works on our Blackwell and `Qwen3.5-4B` peaks under 5.5 GB | Make 4B the default (76.9 vs 69.0 Video-MME). This needs an install approval. See the risk below |
| Thinking-mode loops hit the token cap on more than a few of the gate questions | Ship non-thinking and say that no published score applies |
| Answers need a guarantee of no remote code but the Qwen template or processor changes in a later `transformers` | Pin the revision hash; retest |
| A vendor changes the licence, or the project wants a non-Alibaba model | `SmolVLM2-2.2B` (published 5.2 GB, lower scores) |

**bitsandbytes risk.** [Issue #1937](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937): RTX 5070 Ti laptop, Windows 11, torch 2.11 cu130, `no kernel image is available`. The maintainer says wheels have included sm_120 since about v0.45.3. The issue closed for inactivity with no confirmed cause. So 4-bit on our GPU is **unknown**, not known bad.

### Failure modes the app must show

| Failure | Evidence | What the page does |
|---|---|---|
| **Wrong counts** | Video counting accuracy: Qwen3-VL-4B 25.3, Qwen3-VL-8B 29.6, Molmo2-8B 35.5 (Molmo2 Table 2; Ai2's own benchmark). No Qwen3.5 figure found | Detect "how many"/"count" questions. Show a note and a link: "Language models count poorly. Use Track & count for a measured count." Never display a VLM count as a metric |
| **Time order and motion** | TOMATO: human-model gap of 57.3% with the best model ([arXiv 2410.23266](https://arxiv.org/abs/2410.23266)). Tomato column 31.8–39.8 (Molmo2 Table 2) | Warn on before/after/direction/speed questions |
| **Confident answers about things not in the video** | "the majority of current models exhibit significant issues with hallucinations" ([VideoHallucer, arXiv 2406.16338](https://arxiv.org/abs/2406.16338)); more data and parameters give "limited benefit" for extrinsic factual hallucination | Permanent label. Add a "Not shown in these frames" hint to the system prompt, but do not claim it fixes anything; it is unevaluated |
| **Events between sampled frames** | Follows from sampling. A 10-min video at 32 frames is 1 frame per 18.75 s [derived] | Show the gap: "1 frame every 18.8 s. Anything shorter may be missed." Offer "ask about a selected time range" |
| **Small objects and text** | Follows from the 384 px cap; no specific source found | Show the resolution used |
| **Generation does not terminate** | 2B and 0.8B card warnings | `max_new_tokens` cap, streaming, a Stop button. On hitting the cap show "No final answer" rather than a truncated answer |
| **Out of memory** | Budget above | Catch the OOM in the subprocess, retry once with half the frames, and say so. Stop if it still fails |
| **Published score implied to apply** | Points 1 to 3 above | Never show benchmark numbers next to answers |

No VLM gives a calibrated confidence. Do not show `expected_error` or any percentage for V1d answers: A4's fit applies to a trained classifier, not to these.

### How the page shows evidence and keeps answers labelled

1. **The app samples the frames itself.** Decode with PyAV, then pass the frames with `do_sample_frames=False` and `video_metadata` (with the real fps). Otherwise the processor warns and defaults to `fps=24`, which gives **wrong timestamps** in the prompt [code: `processing_qwen3_vl.py`]. Pass real indices so the frames shown are exactly the ones the model received.
2. **Answer card, fixed layout:**
   - Header: "Model output, not a measured result."
   - Model id, revision hash, licence, frames used, resolution, thinking on/off.
   - Filmstrip of every frame used, each with its timestamp; click to seek.
   - Sampling density line ("1 frame every N s").
3. **`answer.json` next to the frames:** question, answer, thinking text stored apart from the answer, model id + revision, frame indices, timestamps, SHA-256 of each JPEG, seed, generation settings, and `"kind": "model_output"`. The Results and run card code must treat this `kind` as non-metric.
4. **Do not imply the frames prove the answer.** Word it "Frames the model saw", not "Evidence for this answer". PLAN 4b calls them evidence, but the model gives no per-frame attribution.
5. Torch stays out of the GUI: `ask_video.py` runs as a subprocess, as for the other V1 steps.

---

## Q2. V1c: SAM 2 propagation through `transformers`

### Answer

- **Checkpoint: `facebook/sam2.1-hiera-small`** (184 MB, 46M). Fallback `facebook/sam2.1-hiera-tiny` (156 MB, 38.9M).
- Small costs 7% FPS and gains 1.7 J&F on MOSE, 1.0 on LVOS v2 and 0.1 on SA-V ([repo table](https://github.com/facebookresearch/sam2), A100, torch 2.5.1).

| Checkpoint | Params | FPS (A100) | SA-V | MOSE | LVOS v2 |
|---|---|---|---|---|---|
| tiny | 38.9M | 91.2 | 76.5 | 71.8 | 77.3 |
| small | 46M | 84.8 | 76.6 | 73.5 | 78.3 |
| base+ | 80.8M | 64.1 | 78.2 | 73.7 | 78.2 |
| large | 224.4M | 39.5 | 79.5 | 74.6 | 80.6 |

- **Peak VRAM: no primary source found.** The README and model cards give none. Third-party figures conflict (tiny 1.2 vs 2.15 GB; large 7.89 vs 9.5 GB, from [theneuralbase](https://theneuralbase.com/sam/learn/beginner/gpu-requirements/) and similar pages, no method given). I do not rely on them. Measure before offering base+ or large.
- **Licence: Apache-2.0** for checkpoints and code: [LICENSE](https://github.com/facebookresearch/sam2/blob/main/LICENSE); Hub tag apache-2.0, not gated: [model](https://huggingface.co/facebook/sam2.1-hiera-small). The repo's `LICENSE_cctorch` (BSD-3) and demo fonts belong to the official repo; I did not check whether the port uses them.

### Memory grows linearly, and the docs' setup is the worst case

`init_video_session` defaults `inference_state_device` and `video_storage_device` to **`inference_device`** [code: `processing_sam2_video.py:542-543`], although the session class defaults to CPU. The docs example passes only `inference_device=device`, so everything stays on the GPU.

Growth at 1024 px [derived from the code paths; sizes assume 64 memory channels at 64×64]:

| Item | Size per frame (float32 / bf16) | Per | Where it lives |
|---|---|---|---|
| `processed_frames` | 12.6 / 6.3 MB | frame | `video_storage_device` |
| `high_res_masks` | 4.2 / 2.1 MB | object | `inference_state_device` |
| `pred_masks` | 0.26 / 0.13 MB | object | `inference_state_device` |
| `maskmem_features` | 0.52 / 0.52 MB | object | **GPU, always** |
| `maskmem_pos_enc` | 1.05 / 0.52 MB | object | **GPU, always** |
| `object_pointer`, `object_score_logits` | about 1 KB | object | GPU, always |

**Key point, from reading the code:** `_batch_encode_memories` assigns `maskmem_features` and `maskmem_pos_enc` directly into `output_dict_per_obj`, bypassing `store_output`. So `inference_state_device="cpu"` does not move them. This fits the [forum report](https://discuss.huggingface.co/t/sam2-video-streaming-vram-usage-keeps-increasing-until-oom/168526), where CPU offload allowed "significantly more" frames (no numbers) but is not a fix. Unmeasured.

Nothing frees non-conditioning frames [code: no `pop` on them]. `reset_inference_session` does not clear `processed_frames`.

One object, bf16, about 4 GB of VRAM headroom assumed [derived]:

| Setup | Limit |
|---|---|
| Docs default (all on GPU, eager) | about 9 MB per frame: about 440 frames, **15 s at 30 fps** |
| Video and state on CPU, no pruning | GPU growth 1.05 MB per frame: about 3,800 frames (about 2 min). RAM grows 8.5 MB per frame: 16 GB is full at about 1,900 frames (**about 1 min**), and earlier in practice |
| Streaming + pruning (below) | About 16 MB on the GPU per object, plus cond frames. Constant in video length |

### How to bound it

1. **Use streaming**: `model(inference_session=s, frame=px, frame_idx=t)`. Decode one frame at a time with PyAV. `load_video` and `video=` put the whole clip in RAM.
2. **Always pass `frame_idx`.** `add_new_frame` otherwise uses `len(processed_frames)`, which is wrong once pruned. `propagate_in_video_iterator` also depends on `num_frames = len(processed_frames)`, so do not prune in whole-video mode.
3. **Set `inference_state_device="cpu"`, `video_storage_device="cpu"`, `dtype=torch.bfloat16`.**
4. **Prune after each frame.** The model only looks back `num_maskmem − 1 = 6` frames for memory and `max_object_pointers_in_encoder − 1 = 15` for pointers, both with stride 1 and `.get()` lookups [code + config defaults]:
   ```python
   s.processed_frames.pop(t, None)
   for obj in s.output_dict_per_obj.values():
       old = obj["non_cond_frame_outputs"]
       old.pop(t - 15, None)
       if t in old:
           old[t].pop("high_res_masks", None)
   for tracked in s.frames_tracked_per_obj.values():
       tracked.pop(t - 15, None)
   ```
   Conditioning frames are kept, so keep prompts few (`max_cond_frame_num` can cap selection, not storage).
5. **Write masks to disk as RLE on each frame**; never keep them in a list. The doc's `video_segments[...] =` pattern is itself a leak.
6. **Limit objects and frame rate.** Cost is per object per frame. Offer 5 to 10 fps with a visible note.

These are **private attributes**, so:
- Pin the `transformers` version.
- Add a test on a real short clip: masks from pruned and unpruned sessions must match (IoU ≈ 1.0 over 200 frames, GPU or `scripts/` smoke check).
- Add a test asserting VRAM is flat over length.

**Consequences for the UI:**
- A pruned session can't go backward or accept a correction on an old frame. Corrections restart from that frame in a new session. Reverse tracking is a separate pass.
- [Forum](https://discuss.huggingface.co/t/sam2-video-streaming-vram-usage-keeps-increasing-until-oom/168526) posters report masks developing holes over time and attribute it to memory drift, unconfirmed. Long videos should show a "re-click to refresh" prompt, and mask area over time helps spot this.

Fit gate: `sam2.1-hiera-small` on a 5-min real video in the setup above. Record peak VRAM, RAM and FPS.

---

## Q3. Proposals

P-115 (R1: drop the V-JEPA 2.1 repo route for now), P-116 (T1: BatchNorm at small batch), P-117 (B3: resample groups, paired differences), P-118 (R2: state calibration uncertainty from the group count). All on the board.

Not proposed: B1 and B2. I found nothing beyond what P-102 and F-011 already say that would change what gets built.

---

## Could not verify

- **Any VRAM number for the Qwen models, Gemma 4 or Molmo2, or peak VRAM for SAM 2.** None published by a primary source. SmolVLM2's 5.2 GB is stated without precision or frame count.
- **Video QA through `transformers` for Qwen3.5.** Native class and processor mappings are confirmed in 5.16.1 [code], and the template handles `video`. But the card's transformers sample covers images only and frame-sampling notes refer to vLLM. Not run.
- **Whether `enable_thinking` passes through `apply_chat_template`.** The template reads the variable, so it should; not run.
- **Qwen3.5 video evaluation protocol**, and any Qwen3.5 counting score.
- **Qwen3.5 scores at about 32 frames.** No source.
- **Any non-thinking video score** for 2B and 0.8B.
- **SmolVLM2 Video-MME subtitles mode.** Not stated.
- **Molmo2-4B file size.** Not read; excluded anyway.
- **Gemma 4 licence text.** Tag only, no LICENSE file at the path I tried. The card links a Gemma 4 licence document, not opened. Excluded on size and lack of benchmarks.
- **The SAM 2 byte sizes and the claim that memory tensors stay on the GPU** come from reading code, not measurement.
- **`presence_penalty` in `transformers.generate`.** Not confirmed either way.
- **V-JEPA 2.1 weights licence.** The README states none for checkpoints.

## Sources

Opened:
- https://huggingface.co/Qwen/Qwen3.5-2B · https://huggingface.co/Qwen/Qwen3.5-0.8B · https://huggingface.co/Qwen/Qwen3.5-4B · https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct · https://huggingface.co/blog/smolvlm2
- https://huggingface.co/google/gemma-4-E2B-it · https://huggingface.co/blog/gemma4
- https://huggingface.co/allenai/Molmo2-4B · https://arxiv.org/html/2601.10611v4
- https://arxiv.org/html/2511.21631v2 · https://arxiv.org/abs/2406.16338 · https://arxiv.org/abs/2410.23266
- https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937
- https://github.com/facebookresearch/sam2 · https://huggingface.co/docs/transformers/model_doc/sam2_video · https://discuss.huggingface.co/t/sam2-video-streaming-vram-usage-keeps-increasing-until-oom/168526
- https://github.com/facebookresearch/vjepa2 · https://arxiv.org/html/2107.07511v6 · https://arxiv.org/abs/1803.08494 · https://arxiv.org/abs/2606.03656

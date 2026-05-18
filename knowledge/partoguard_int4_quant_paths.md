# INT4 QAT for Gemma 4 E2B-it Vision: Path A, Path B, and the int4 wall

> **STATUS: EXPLORATORY / POST-HACKATHON (as of 2026-05-17).** This document
> records the failed sub-bf16 quantization experiments (V13 / V14 / V15) that
> motivated keeping V7 at bf16. The shipping deployment is V7 bf16 served as
> Q8_0 GGUF via llama.cpp on the VPS (`knowledge/partoguard_remote_gemma_vps.md`).
> Q8_0 is post-training quantization of bf16 weights — unrelated to the int4
> *training* paths investigated below. Any "blocked", "in flight", or "next
> step" language in this file refers to the int4 research track and is not
> blocking the live deployment.

**Status of the int4 track**: **Path A failed.** Best 4-bit variant ties V13 at 57% vs V7 bf16 baseline at 100%. The 4-bit budget appears insufficient for dense-chart counting on Gemma 4 E2B regardless of quantization scheme. Investigating int8 (W8A16) as the next path per user direction.

**Last updated**: 2026-05-16

## Problem Statement

INT4 quantization-aware fine-tuning of Gemma 4 E2B-it for partograph mark counting. The bf16 baseline (V7, vanilla PEFT, frozen vision, all-linear LoRA) achieves 100% correctness on 350-image eval. Switching to NF4 QLoRA (V13, Unsloth, frozen vision) collapses to ~40% with the model emitting 10–23 dilation points where ground truth is 5–8. The schema is always valid; the visual grounding is broken.

## Root Cause (Documented)

Uniform NF4 quantization applied to the vision encoder MLPs destroys spatial precision needed for discrete mark localization.

- Daniel Han (Unsloth founder): *"The entire vision encoder should not be quantized to 4bit"* — applies to Qwen2-VL, Pixtral, and by extension Gemma 4. ([unsloth.ai/blog/dynamic-4bit](https://unsloth.ai/blog/dynamic-4bit), [github.com/unslothai/unsloth/issues/1347](https://github.com/unslothai/unsloth/issues/1347))
- Visual tokens have activation magnitudes ~40× larger than text tokens (VLMQ paper, [openreview.net/pdf?id=n0pSH3hOJA](https://openreview.net/pdf?id=n0pSH3hOJA)). Uniform NF4 calibration overfits to the visual range and destroys fidelity for both spatial localization and the text output path that emits coordinates.
- AMXFP4 benchmarks: LLaVA1.6 ChartQA drops 54.72 → 46.20 under MXFP4. Counting/OCR tasks are the most sensitive class.

Lower input resolution does **not** mitigate this (FastVLM, arXiv 2412.13303, plus orthogonality result in arXiv 2504.03749) — resolution and quantization stack multiplicatively for chart/OCR tasks.

## Path A — Selected (Unsloth Dynamic 4-bit + unfreeze vision)

```python
# scripts/finetune_e2b_v14_dynamic.py
from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",  # dynamic 4-bit, vision MLPs spared
    load_in_4bit=True,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # ← critical change vs V13
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16, lora_alpha=16,
)
```

Matches Unsloth's documented default for VLM fine-tuning. Adapter loadable in Unsloth runtime; LiteRT export still blocked on upstream litert-torch issue #1005.

## Path B — Documented Fallback (Vanilla bnb + skip vision in quant)

Use if:
- Path A still over-counts after retraining (vision still too damaged even under dynamic quant)
- Need an adapter portable to stock `transformers + peft + bitsandbytes` without Unsloth runtime in deployment
- Want to match the industry-standard PTQ pattern documented by Optimum-Intel and llm-compressor

### Rationale

The HuggingFace Optimum-Intel team validated the configuration "int4 LLM + bf16 (or int8) vision encoder" as the **reliable production setup** for VLM PTQ ([github.com/huggingface/optimum-intel/pull/1394](https://github.com/huggingface/optimum-intel/pull/1394)):

> "A more reliable setup is combining quantization of vision encoder with 4-bit weight only quantization of a language model."

Validated numbers from that PR:

| Model | LM precision | Vision precision | WWB Sim | MME Acc |
|---|---|---|---|---|
| Qwen2-VL-7B | bf16 | bf16 | 100% | 87.25% |
| Qwen2-VL-7B | int4 | int8_sym | 89.78% | 87.00% |
| InternVL2-1B | int4 | bf16 (full) | n/a | n/a |

Same pattern in vllm-project's `llm-compressor` ([issues/1629](https://github.com/vllm-project/llm-compressor/issues/1629)):

```python
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    sequential_targets=["Qwen2_5_VLDecoderLayer"],
    ignore=["lm_head", "re:visual.*"],   # ← skip vision encoder
)
```

### Path B code sketch

```python
# scripts/finetune_e2b_path_b.py (sketch only — not implemented; archived path)
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    # CRITICAL: skip the vision tower from 4-bit quantization.
    # Module name will be "vision_tower" or "vision_model" — verify
    # against actual Gemma 4 E2B module tree before training.
    llm_int8_skip_modules=["vision_tower"],
)

model = AutoModelForVision2Seq.from_pretrained(
    "google/gemma-4-E2B-it",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
))
# Train with vanilla TRL SFTTrainer — no Unsloth runtime dependency.
```

### Trade-offs vs Path A

| Aspect | Path A (Unsloth dynamic) | Path B (vanilla bnb + skip) |
|---|---|---|
| Training speed | Fast (Unsloth optimizations) | Standard (~2× slower) |
| Adapter portability | Best loaded via Unsloth FastVisionModel | Loadable by stock `peft.PeftModel` + bnb |
| Vision encoder precision | Per-module dynamic (Unsloth picks) | Full bf16 (skipped from quant entirely) |
| VRAM at inference | ~3.5GB | ~4.5GB (vision in bf16) |
| Eval runtime | Unsloth FastVisionModel | Standard `transformers.AutoModel` |
| LiteRT-LM exportability | Same upstream block (issue #1005) | Same upstream block (issue #1005) |

### When to switch from A → B

1. Path A V14 still over-counts (>5% incorrect) after training completes
2. Need to deploy adapter on a runtime that doesn't have Unsloth (e.g. plain bnb in a serving stack)
3. Want maximum fidelity in vision encoder during further iterations (e.g. distillation from V7)

## V14 Results — Path A Failed (2026-05-16)

V14 trained successfully on Norway RTX 4090 (15:45 wall time, final loss 0.118 with vision unfrozen — higher than V13's 0.0024 because more params training, healthy convergence). Three eval variants run on 100-image sample (seed 42):

| Variant | Overall | blank | partial | degraded | obstructed | filled |
|---|---|---|---|---|---|---|
| V7 (bf16, frozen vision) — **proven baseline** | **100%** | — | — | — | — | — |
| V13 (Unsloth uniform NF4 + frozen vision) | 57% | 100% | 75% | 68% | 44% | 24% |
| V14 baseline (Dynamic 4-bit + unfrozen vision) | 54% | 100% | 65% | 68% | 38% | 24% |
| V14 + Otsu | 52% | 100% | 50% | 64% | 44% | 28% |
| V14 + Otsu_dilate | **57%** | 100% | 50% | 77% | 38% | 38% |

**Best 4-bit variant ties V13.** Otsu_dilate helped dense/degraded charts (filled +14pp, degraded +9pp) but hurt partial (-25pp). Failure mode flipped: V13 *over*-counted, V14 *under*-counts (collapses to `{"p":[]}` or 1-2 points on dense charts). Filled category stuck at 24-38% across every 4-bit variant tried.

**Conclusion**: 4-bit quantization (any flavor — uniform NF4, dynamic per-module, with or without unfrozen vision, with or without aggressive image preprocessing) is fundamentally insufficient for fine-grained dense-mark counting on Gemma 4 E2B. The research-backed Path A hypothesis was correct in theory but did not survive empirical testing.

## Forward Paths (post-V14 failure, 2026-05-16)

User-prioritized order: **option 4 (int8) → option 3 (more dense data) → option 2 (V7 distillation)**. Path B (option 1) deprioritized. Option 5 documented as final fallback.

### Option 4 — Pivot from int4 to int8 (W8A16) [PRIORITY 1, RESEARCH IN FLIGHT]

Quantize to 8-bit weights with 16-bit activations instead of 4-bit. Half the bf16 size, double the int4 precision. Industry consensus is int8 typically recovers 95%+ of bf16 accuracy on chart/OCR tasks (vs int4's 50-90% recovery).

**Open questions** (research task `bg_79fe2040` in flight):
- Does Unsloth `FastVisionModel` accept `load_in_8bit=True` for Gemma 4?
- Does LiteRT-LM support int8 quantization mode for Gemma 4 vision (vs the dynamic_wi4_afp32 we already used)?
- Estimated Android footprint: ~2.5GB int8 (vs ~1.4GB int4, ~5GB bf16) — still fits Kaggle low-end target?
- Hybrid option: int8 vision encoder + int4 LLM (some papers report sweet spot)

**Estimated cost**: 1 round of training (~2-3h on Norway 4090) if path is supported.

### Option 3 — More dense-chart training data [PRIORITY 2]

Hypothesis: 700 training samples may have insufficient coverage of dense charts (8-12 marks per cervicograph). Generate 1000-2000 additional synthetic samples weighted toward filled/obstructed categories, retrain V14-style.

**Pros**: Cheap (~1h training), uses existing infrastructure.
**Cons**: Won't fix the underlying quantization problem if 4-bit truly is the wall. May only push filled from 38% → 50-60%, not to 100%.

**Estimated cost**: ~1h data generation + ~1h training = 2h total.

### Option 2 — V7 distillation (int4 student from bf16 teacher) [PRIORITY 3]

Train V15 to mimic V7's logits/output distribution instead of training on ground-truth JSON. KL-divergence loss between V7 (bf16, 100% accurate) teacher and V15 (int4) student. The student inherits V7's competence while quantization noise is partially compensated for by the LoRA.

**Pros**: V7 already achieves the target accuracy. Distillation has been shown to recover 90%+ of teacher accuracy in compressed students.
**Cons**: More complex training pipeline (need to run both V7 inference + V15 training simultaneously). Higher VRAM. May still hit the same fundamental int4 wall if the bottleneck is *representation*, not learning.

**Estimated cost**: ~4-5h (script complexity + longer training).

### Option 1 — Path B: vanilla bnb + skip vision in quant [DEPRIORITIZED but documented]

See full Path B section above. Documented earlier as primary fallback to Path A but user has deprioritized in favor of int8 pivot. Still viable if int8 path also fails — would produce an int4 LLM with bf16 vision encoder (true mixed precision).

### Option 5 — Concede int4 is wrong target; ship V7 bf16, defer mobile [LAST RESORT]

If options 4, 3, 2 all fail: accept that Gemma 4 E2B at 4-bit fundamentally cannot do clinical-grade dense-chart counting. Ship V7 bf16 as the desktop/server console, defer Android deployment until either (a) Gemma 5 has better int4 vision behavior, (b) we use a different mobile architecture (e.g. PaliGemma at int8), or (c) LiteRT-LM upstream adds Gemma 4 vision support and we splice on Google's higher-quality pre-built vision sections.

**Pros**: Zero engineering risk. Working V7 already exists.
**Cons**: Loses the Kaggle Gemma 4 hackathon Android target. Deferral not deployment.

## What's Off the Table

- **Lower input resolution** — debunked by FastVLM benchmarks for chart/OCR tasks. Stacks multiplicatively with quantization damage.
- **Frozen vision under uniform NF4** — known foot-gun (V13's failure mode); confirmed by Daniel Han.
- **Path A as currently configured** — empirically shown to tie V13 at 57%. Don't re-run with same params.

## LiteRT-LM Deployment Caveat

Both paths produce a working int4 model on standard runtimes (Unsloth/bnb), but **deployment to LiteRT-LM mixed48 for Android remains blocked** until upstream `litert-torch` adds proper Gemma 4 vision export support ([github.com/google-ai-edge/litert-torch/issues/1005](https://github.com/google-ai-edge/litert-torch/issues/1005)). The current workaround — splicing V8/V14 LLM sections onto Google's pre-built vision sections (`scripts/build_v8_multimodal.py`) — produces a bundle that loads but hangs/segfaults at inference (Conversation API Jinja `.get()` bug).

Per user constraint ("first make 4bit q v7 work good on any standard runtime ... then only we can mess with hacks for litert"), validating Path A on Unsloth runtime is the gating success criterion before resuming LiteRT splice work.

## Source Index

- Unsloth dynamic 4-bit: https://unsloth.ai/blog/dynamic-4bit
- Unsloth issue (vision encoder NF4 breaks): https://github.com/unslothai/unsloth/issues/1347
- Unsloth VLM fine-tuning docs (default `finetune_vision_layers=True`): https://unsloth.ai/docs/basics/vision-fine-tuning
- Optimum-Intel mixed-precision PR: https://github.com/huggingface/optimum-intel/pull/1394
- llm-compressor visual ignore pattern: https://github.com/vllm-project/llm-compressor/issues/1629
- VLMQ paper (visual token activation scale): https://openreview.net/pdf?id=n0pSH3hOJA
- AMXFP4 paper (chart/OCR drop numbers): https://aclanthology.org/2025.findings-acl.776.pdf
- FastVLM (resolution matters for chart tasks): https://arxiv.org/pdf/2412.13303
- Resolution-quant orthogonality: https://arxiv.org/pdf/2504.03749
- LVLM-COUNT (counting hard even at full prec): https://arxiv.org/abs/2412.00686
- LiteRT-LM Gemma 4 export bug: https://github.com/google-ai-edge/litert-torch/issues/1005

---

## V15 INT8 Result (2026-05-16) — Path C: bnb int8 QAT

| Variant | Runtime | 100-img |
|---------|---------|---------|
| V13 (uniform NF4, frozen vision) | Unsloth 4-bit | 57% |
| V14 (Dynamic 4-bit, unfrozen vision) | Unsloth 4-bit | 57% |
| **V15 (bnb int8, unfrozen vision, 1 epoch)** | **Unsloth INT8** | **59%** |
| V15 (same adapter, merged → Q8_0 GGUF) | llama.cpp mtmd-cli | 32% (template bug) |
| V7 reference | bf16 transformers | 100% |

**Findings**:
- bnb int8 QAT marginally beats Dynamic 4-bit (+2pp), confirming int8 ≻ int4 for chart counting (as predicted by AMXFP4 paper).
- The 36pp gap to 95% is **fundamental to single-epoch LoRA under int8 quant** — dense-chart over-counting persists exactly as with V13/V14.
- **GGUF deployment via llama.cpp Q8_0 + mmproj works mechanically** (model loads, image projected, output produced) but loses ~27pp to a chat-template image-marker bug. Engineering-fixable.

**Engineering details** (for resuming GGUF path):
- Merge MUST use Unsloth `FastVisionModel.from_pretrained(adapter, load_in_16bit=True) → merge_and_unload()`. Plain transformers + PEFT fails with `Gemma4ClippableLinear is not supported` because V14/V15 have vision-layer LoRA targets.
- `convert_lora_to_gguf.py` is BROKEN for Gemma 4 (issue #23047) — must merge first.
- llama.cpp Gemma 4 vision support landed in PR #21309 (Apr 2 2026). Requires `-b 2048 -ub 2048`.
- Custom Jinja template via `--jinja --chat-template <inline>` works to suppress Gemma 4 hybrid-thinking CoT, but image marker insertion via mtmd-cli's `--image` flag needs a proper `<start_of_image>` placeholder in the template (currently missing — causes Q8_0 to drop to 32%).
- Mobile footprint: 4.95 GB text + 985 MB mmproj = **5.93 GB** → fits 6 GB Android, marginal on 4 GB.
- bnb int8 direct inference via transformers: 75-95 s/image (deceptive — not a real deployment speed). GGUF Q8_0: 8.5 s/image RTX 4090.

**Open ideas for next iteration** (saved 2026-05-16, awaiting user direction):
1. Train V15 v2: 3 epochs + rank 32 + optional V7-distillation soft targets (plausible 75-85%)
2. Fix GGUF chat-template image marker (recover 27pp on deployment path, ship-ready 4-6 GB artifact at 59%)
3. V14 1120-tokens experiment (orthogonal axis — resolution vs quant)
4. torchao INT8 QAT (Unsloth PR #3859, `qat_scheme="int8"`) — true forward+backward simulated quant vs bnb's weight-only int8
5. Parallel multi-experiment (heavy resource use)

# LoRA & QLoRA Fine-Tuning Reference for Gemma 4 VLMs

This document synthesizes research from multiple sources into a practical reference for our PartoGuard fine-tuning workflow. Focus: Gemma 4 E2B-it, Unsloth, LoRA for vision-language extraction tasks.

---

## 1. LoRA Math

**Core equation**: `W_merged = W_base + (alpha/r) * B @ A`

- **A** in R^(r x k): initialized Kaiming uniform
- **B** in R^(d x r): initialized to **zero** (adapter starts as identity)
- **r**: rank (bottleneck dimension)
- **alpha/r**: scaling factor

Parameter count per layer: `r * (d + k)` vs `d * k` for full FT. For d=k=4096, r=16: 131K vs 16.7M (128x reduction).

---

## 2. Hyperparameter Guide

### Rank (r)

| r | Use case |
|---|----------|
| 4-8 | Style/format changes, tiny datasets (<1K), base model already strong |
| 16 | Universal default (~90% of tasks) |
| 32-64 | Complex reasoning, when r=16 loss plateaus |

**Key finding**: Higher rank = more forgetting. "Intruder dimensions" paper (NeurIPS 2025): high rank LoRA introduces orthogonal vectors that correlate with forgetting (rho=0.971). For our case (base at 87%, teaching output format), r=4-8 is optimal.

### Alpha (alpha)

| Setting | Scale | Use |
|---------|-------|-----|
| alpha = r | 1.0 | Conservative baseline |
| alpha = 2r | 2.0 | Most common production setting |

Alpha and LR interact — higher alpha/r amplifies adapter gradient contribution. If loss unstable early, reduce alpha or LR.

**RSLoRA** (`use_rslora=True`): Changes scaling from alpha/r to alpha/sqrt(r). Stabilizes training at high ranks (r >= 64).

### Learning Rate

- Full fine-tuning: 1e-5 to 5e-5
- LoRA/QLoRA: **1e-4 to 2e-4** (5-20x higher than full FT)
- For strong base model needing gentle nudge: **2e-5** (our V4 plan)

### Epochs

| Dataset size | Epochs |
|-------------|--------|
| <1K samples | 3-5 |
| 1K-10K | 2-3 |
| >10K | 1 |

For chat/VLM on synthetic data: 1-2 epochs max. Beyond that, model memorizes output format.

### Other

- **Batch size**: Target effective batch 16-32 (per_device * grad_accum * gpus)
- **Warmup**: 0.03-0.05 ratio
- **Weight decay**: 0.01 (or 0.0 — LoRA itself regularizes)
- **Scheduler**: cosine (default)
- **Dropout**: 0.0 for VLM (Unsloth default), 0.05-0.1 for small datasets

---

## 3. Target Module Selection

### For VLMs (Gemma 4)

```python
# Attention only (conservative — our V4 plan):
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# All linear (QLoRA paper recommendation — best quality but more forgetting):
target_modules = "all-linear"  # q/k/v/o + gate/up/down
```

**When attention-only**: Small datasets, base already strong, format-only adaptation.
**When all-linear**: Complex tasks, large datasets, need maximum adaptation.

### Vision Tower

**Default: FREEZE.** Vision tower (SigLIP) already excellent. Unfreezing with small dataset causes forgetting + overfitting. Only unfreeze for systematically OOD images (medical, satellite) with large diverse datasets (>50K).

### Silent Failure Mode

PEFT does NOT error if target_modules names don't exist — applies to 0 layers. Always verify:
```python
model.print_trainable_parameters()  # Must show >> 0 trainable params
```

---

## 4. Response-Only Loss (CRITICAL)

Full-sequence loss trains model to "predict" the question — degenerate objective. Response-only loss masks instruction tokens (label=-100).

### Unsloth VLM approach:
```python
UnslothVisionDataCollator(
    model, processor,
    train_on_responses_only=True,
    instruction_part="<|turn>user\n",     # Gemma 4 chat template
    response_part="<|turn>model\n",       # Gemma 4 chat template
)
```

Loss values appear higher with response-only (fewer tokens in denominator) — expected.

---

## 5. QLoRA

4-bit NF4 base (frozen) + bfloat16 LoRA adapters. Dequantizes to bf16 for compute.

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # saves ~0.37 bits/param
)
```

Use QLoRA when VRAM < model size in bf16. Quality gap vs LoRA is minimal for instruction-following.

---

## 6. Merging LoRA Adapters

### Unsloth save_pretrained_merged: BROKEN for Gemma 4

Known bug family (Issues #1352, #1763, #1929, #611). Root cause: adapter keys use `lora_A.weight` but PEFT expects `lora_A.default.weight` — 0 keys match. Produces unmodified base model.

### Working: Manual safetensors merge

Our script at `scripts/merge_lora.py`:
```python
# Key mapping: base_model.model.X.lora_A.weight -> X.weight
# delta = (alpha/r) * B @ A
# W_merged = W_base.float() + delta
```

### Working: PEFT merge_and_unload()

```python
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload(safe_merging=True)  # MUST reassign
```

**Caveat**: Gemma4ClippableLinear crash (Issue #4820) — PEFT can't load adapter if model has ClippableLinear. Use Unsloth's patched loader or manual merge.

### Verify merge correctness

```python
# SHA256 of merged model must differ from base model
# Spot-check: merged_weight ≈ base_weight + (alpha/r) * B @ A
```

---

## 7. Unsloth Gemma 4 Specifics

### Known Bugs (May 2026)

| Bug | Status | Impact |
|-----|--------|--------|
| Gradient accumulation loss explosion (300-400) | Fixed in Unsloth | Verify loss <15 in first 10 steps |
| Gemma4ClippableLinear crash on PEFT load | Fixed in Docker | Use Unsloth Docker or manual merge |
| Vision LoRA zero gradients on 26B | Fixed pending | Verify vision adapter deltas |
| save_pretrained_merged produces base model | Known | Use manual merge script |

### SFTConfig requirements for VLM (all four MUST be set):
```python
SFTConfig(
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    max_length=2048,  # minimum for single-image tasks
)
```

### Inference mode switch:
```python
FastVisionModel.for_inference(model)  # Required before generation
```

### Content order: image BEFORE text in messages (matches pretraining distribution).

---

## 8. Our Training History & Lessons

| Version | Config | Result | Issue |
|---------|--------|--------|-------|
| V1 (PEFT+TRL, XPU) | r=16, alpha=16, all tokens, 2 epochs | **94.57%** | Best result — but used full-sequence loss |
| V2 (Unsloth, L40S) | r=32, alpha=32, 2 epochs | 87% | Merge bug — was actually base model |
| V3 (Unsloth, L40S) | r=16, all-linear, lr=1e-4, 3 epochs | 71% | Catastrophic forgetting — too aggressive |
| Base model | No fine-tuning | ~87% | Zero-shot baseline |

### Why V3 regressed (Oracle analysis):
1. All-linear LoRA (vision + language + MLP) — too broad
2. lr=1e-4 — too high for gentle adaptation
3. 3 epochs — overfitting
4. No response-only loss — long prompt diluted signal

### V4 plan (research-informed):
```python
r = 4-8           # Low rank — base already strong
alpha = r          # Conservative scaling
lr = 2e-5          # 5x lower than V3
epochs = 1         # Prevent overfitting
target = attn-only # q/k/v/o_proj only
vision = frozen    # Don't touch vision tower
loss = response-only  # instruction_part/response_part masking
data = V1 only (400 images)  # Proven quality
```

---

## 9. Practical Checklist Before Training

- [ ] Verify target_modules match actual model layer names
- [ ] `model.print_trainable_parameters()` shows expected count
- [ ] Chat template matches between training and inference
- [ ] Image before text in message content
- [ ] Loss < 15 within first 10 steps (Gemma 4 grad accum bug check)
- [ ] Response-only loss configured (instruction_part / response_part tokens verified)
- [ ] Vision tower frozen unless specifically needed
- [ ] Save LoRA adapter after training (don't rely on merged save)
- [ ] Merge with manual script or PEFT merge_and_unload, NOT Unsloth save_pretrained_merged
- [ ] Verify merge by SHA256 hash comparison with base model

---

## Sources

- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs"
- "LoRA Learns Less and Forgets Less" (arXiv 2405.09673)
- "Intruder dimensions" (NeurIPS 2025) — high rank correlates with forgetting
- Unsloth Gemma 4 training guide: unsloth.ai/docs/models/gemma-4/train
- Unsloth Issues: #1352, #1763, #1929, #3633, #4820, #5039
- PEFT merge docs: huggingface.co/docs/peft/developer_guides/model_merging

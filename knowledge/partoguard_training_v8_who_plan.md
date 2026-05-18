# V8 Training: WHO Template Augmentation

## Objective

Add ~100 WHO-template-based training images (25% of total) to the existing 400 synthetic images, giving the model exposure to real WHO chart appearances alongside our programmatic grids.

## Problem Statement

The V7 model (95.71% on synthetic eval via VPS Q8_0) struggles on real WHO charts — extracting demo_normal (drawn on `bmc2013_p3_fig1.jpeg`) produces 0/4 correct points via VPS. The model has never seen a real WHO grid during training. On-device local inference (S24 Ultra, 35s/img) works mechanically and can extract points from WHO templates, but with significant over-counting.

**Correction (2026-05-18)**: V7 is not completely "blind" to WHO charts as initially assumed — the Android app with V7 Q8_0 GGUF can read WHO templates, but with over-counting issues. The VPS path showed 0/4, likely due to differences in the mtmd vision pipeline prompt format. V8 still improves WHO reading substantially.

## Approach

**Mixed training set**: 400 existing synthetic + 100 new WHO-template images = 500 total.

The WHO images use the same generation technique as `scripts/generate_demo_images.py` — draw X marks at known coordinates on cropped real WHO chart backgrounds.

## Execution Log (2026-05-18)

### Infrastructure

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Instance | Vast.ai #37014165 (Norway) |
| SSH | `ssh root@<your-gpu-instance>` |
| Cost | ~$0.50/hr |
| OS | Ubuntu + conda (Python 3.11) |

### Dependencies Installed

```
torch 2.6.0+cu126, transformers 5.8.1, peft 0.19.1, trl 1.4.0,
bitsandbytes 0.49.2, accelerate 1.13.0, datasets 4.8.5,
pydantic 2.13.4, opencv-python-headless 4.13.0.92, gguf 0.19.0
```

### Training Run

| Parameter | Value |
|-----------|-------|
| Base model | `google/gemma-4-E2B-it` (fresh, not from V7 adapter) |
| Method | PEFT LoRA (r=16, alpha=16, target=all-linear) |
| Training data | 500 images (400 synthetic seed=77777 + 100 WHO seed=88888) |
| Labels | `/workspace/data/training/labels.json` (single merged file) |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine decay, warmup_ratio=0.03) |
| Batch size | 1 (grad_accum=8, effective=8) |
| Precision | bf16 |
| Optimizer | adamw_torch |
| Total steps | 189 |
| Time per step | ~6.6s |
| Total train time | ~21 minutes |
| **Final loss** | **0.6545** |

### Trainable Parameters

```
trainable params: 37,920,768 || all params: 5,142,218,272 || trainable%: 0.7374
```

### Post-Training Pipeline

1. **LoRA saved** → `/root/partoguard-lora/lora_adapter_v8/` (152 MB adapter)
2. **Merged** via `PeftModel.from_pretrained()` + `merge_and_unload()` → `/root/partoguard-lora/merged_v8/` (10.24 GB bf16)
3. **GGUF bf16** → `/root/partoguard-lora/partoguard-v8-bf16.gguf` (8.7 GB)
4. **GGUF Q8_0** → `/root/partoguard-lora/partoguard-v8-q8_0.gguf` (4.6 GB) ✓
5. **mmproj F16** → `/root/partoguard-lora/mmproj-partoguard-v8-f16.gguf` (985 MB) ✓
6. **HuggingFace** → [`partoguard/partoguard-v8-q8_0-gguf`](https://huggingface.co/partoguard/partoguard-v8-q8_0-gguf) (Q8_0 + mmproj)

### Key Differences from Previous V7 Training

| Aspect | V7 | V8 (WHO) |
|--------|-----|----------|
| Training data | 1030 synthetic only (seed 77777 + supplements) | 500 mixed (400 synthetic + 100 WHO) |
| Base | Fresh `google/gemma-4-E2B-it` | Fresh `google/gemma-4-E2B-it` |
| Epochs | 5 | 3 |
| Hardware | Intel Arc XPU (16 GB) | RTX 4090 (24 GB) |
| Time | ~2.5 hours | ~21 minutes |
| Goal | 100% on synthetic eval | Generalize to real WHO charts |

### Note on Version Naming

This V8 is distinct from the earlier "V8" in `partoguard_training_log.md` (which was 1510 samples, 2 epochs, all-synthetic, regressed to 98.57%). The naming collision exists because the original V8 was a failed intermediate run during the V5→V7 optimization cycle. This new V8 represents a fresh direction: WHO template augmentation for on-device generalization.

## Available WHO Base Templates

| Template | Source | Resolution | Notes |
|----------|--------|-----------|-------|
| `bmc2013_p3_fig1.jpeg` | BMC 2013 paper | ~790×360 crop | The one used for demo images. Well-calibrated. |
| `who_lcg2020_p8_fig9.jpeg` | WHO LCG 2020 | 1260×1792 | Labour Care Guide form (newer format) |
| `who_lcg2020_p8_fig10.jpeg` | WHO LCG 2020 | 2410×3428 | Same, higher res |
| `bedwell2017_fig1a_1944x1518.png` | Bedwell 2017 | 1944×1518 | WHO 1994 + 2000 composite |

## Generation Plan

### Per-template calibration (one-time)

For each WHO template, manually determine:
- Cervicograph crop region (y1, y2, x1, x2)
- Grid origin (x_start, y_start) in pixels
- Scale factors (px_per_hour, px_per_cm)

Already done for `bmc2013`: x_start=130, y_start=328, px_per_cm=31.3, px_per_hour=49.67

### Image distribution (100 new images)

| Category | Count | Description |
|----------|-------|-------------|
| blank | 10 | Clean WHO template crops, no marks |
| partial | 20 | 1-3 X marks (early labour) |
| filled | 40 | 4-10 X marks (complete trajectories) |
| degraded | 30 | WHO template + phone-capture effects (same as synthetic degraded) |

### Variation axes

- **Template**: Randomly select from available WHO bases per image
- **Crop jitter**: ±5-15px random crop offset (simulates imperfect phone framing)
- **Marks**: Same curve types as synthetic (normal, slow, arrested, rapid)
  - **Normal**: Points stay left of alert line (steep ascent, 1cm/hr or faster)
  - **Slow/prolonged**: Points cross alert line but stay left of action line
  - **Arrested**: Points cross action line or flatten (plateau at same dilation)
  - **Rapid**: Very steep, 4-5 points clustered in first 2-3 hours
  - Ensure coverage across all zones of the cervicograph grid (low dilation 0-4cm, active phase 4-7cm, transition 7-10cm)
- **Pen style**: Vary color (dark blue, black, dark red), size (7-12px), thickness (2-4px)
- **Degradation** (applied to ALL categories with probability, not just "degraded" subset):
  - 30% of blank/partial/filled get mild degradation (rotation ±2°, brightness ±10%)
  - 100% of "degraded" category get full phone-capture simulation:
    - Perspective warp (tilted capture, ±5-15° keystoning)
    - Rotation (±3-8° non-level phone)
    - Uneven lighting (flash hotspot, vignetting gradient)
    - Motion/focus blur (Gaussian kernel 1-3px)
    - Sensor noise (Gaussian σ=5-15)
    - JPEG compression artifacts (save at Q60-80 and reload)
    - Random combination of 2-4 of the above per image

### Label format (identical to existing)

```json
{
  "image": "training/train_0400.png",
  "points": [[0.0, 4.5, 0.99], [1.0, 5.5, 0.99], ...],
  "n_marks": 5,
  "category": "filled",
  "curve_type": "normal"
}
```

## Script Plan

New script: `scripts/generate_who_training.py`

```
Inputs:
  - WHO template images (from data/harvested/)
  - Per-template calibration dicts
  - Seed: 88888 (no overlap with eval=12345, existing training=77777)
  - Count: 100 images

Output:
  - data/training/train_0400.png ... train_0499.png
  - Appends to data/training/labels.json
```

## Training Configuration (Actual)

- **Base**: Fresh `google/gemma-4-E2B-it` (not from V7 LoRA — fresh LoRA per ICLR 2025 best practice)
- **Data**: 500 images (400 synthetic + 100 WHO) in single labels.json
- **Epochs**: 3
- **Hardware**: RTX 4090 24GB (Vast.ai Norway, $0.50/hr)
- **Actual time**: ~21 minutes training + ~2 min merge + ~10 min GGUF conversion

## Evaluation Plan

1. Run on existing 350-image synthetic eval → should stay ≥95%
2. Run on WHO-template demo images (demo_normal, demo_alert, demo_action) → target 100%
3. On-device verification on S24 Ultra with `demo_normal.png`
4. Add 10-20 WHO-template eval images (separate from training, seed 99999) for ongoing regression

## Preliminary Eval Results (bf16 on GPU, 2026-05-18)

Quick sanity check on 6 training images + 3 eval images (not full corpus eval):

### WHO Template Images (Training Set — should be perfect)

| Image | GT n_marks | Model n_marks | Coord accuracy | Notes |
|-------|-----------|--------------|----------------|-------|
| train_0400 (partial/slow) | 3 | 4 | ±0.5h drift | Over-counted by 1, x-coords shifted left by 0.5h |
| train_0401 (degraded/rapid) | 4 | 4 | ±1.0cm drift | Count correct, dilation values off by 0.5-1.5cm |
| train_0402 (degraded/slow) | 3 | 3 | ±0.5h drift | Count correct, x-coords shifted slightly |

### Synthetic Images (Training Set)

| Image | GT n_marks | Model n_marks | Notes |
|-------|-----------|--------------|-------|
| train_0090 (filled/normal, 5 marks) | 5 | 4 | Under-counted by 1, missed last point |
| train_0091 (filled/arrested, 9 marks) | 9 | 9 | Count correct, minor x-coord drift |
| train_0092 (filled/normal, 9 marks) | 9 | 7 | Under-counted by 2, missed tail points |

### Eval Corpus Images (Unseen)

| Image | Expected | Model output | Verdict |
|-------|----------|-------------|---------|
| blank_0000 | `{"p":[]}` | `{"p":[]}` | ✓ CORRECT |
| filled_0110 (9 marks, slow_prolonged) | 9 points | 10 points | Over-count by 1, trajectory shape correct |

### Critical Test: demo_normal.png (WHO chart, previously 0/4 with V7)

```
Expected: ~4 points, normal labour curve (0h,4cm → 3.5h,9cm)
V7 result: 0/4 points correct (couldn't read WHO grid)
V8 result: {"p":[[0.0,4.5,0.99],[1.0,6.0,0.99],[2.0,7.5,0.99],[3.0,8.5,0.99],[4.0,10.0,0.99],[4.5,10.0,0.99]]}
```

**Assessment**: V8 CAN read WHO templates (trajectory shape correct: steep ascent 4.5→10cm). Over-counts by 2 (6 vs 4 expected) and has ±0.5 coordinate drift. The fundamental WHO-blindness is solved, but precision needs improvement.

### Summary

| Metric | V7 (synthetic-only) | V8 (WHO-augmented) |
|--------|---------------------|-------------------|
| WHO chart reading | ❌ Completely blind | ✓ Reads correctly (with over-counting) |
| Blank detection | ✓ Perfect | ✓ Perfect |
| Synthetic accuracy | 95.71% (VPS Q8_0) | ~TBD (pending full 350-img eval) |
| Coordinate precision | ±0.0 on synthetic | ±0.5 on some images |
| Over-counting tendency | Rare | Moderate (1-2 extra on complex charts) |

### Possible Improvements (for V8.1)

1. **More epochs** (5 instead of 3) — loss 0.6545 suggests underfitting
2. **More WHO data** (200 instead of 100) — currently only 20% of training set
3. **Targeted supplemental data** for over-counting — add images where model must learn to stop
4. **Use V7's 1030 synthetic base** instead of just 400 — V7 had 5x the targeted synthetic data

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Catastrophic forgetting on synthetic | Keep 80% synthetic in mix; eval on full 350 corpus |
| Over-fitting to specific WHO template | Use all 4 templates; add crop jitter and degradation |
| Calibration errors in grid coords | Validate by drawing known points and visually checking |
| Small training set (100 WHO) | If accuracy insufficient, scale to 200 WHO (33% of mix) |
| Lower sample count (500 vs V7's 1030) | Compensated by better data diversity; 3 epochs should suffice |

## Prerequisites (Completed)

- [x] Calibrate grid coordinates for `bmc2013_p3_fig1.jpeg` template
- [x] Write `scripts/generate_who_training.py` (seed 88888)
- [x] Generate 100 WHO images + append to labels.json (500 total)
- [x] Visually spot-checked outputs
- [x] Rent GPU instance (RTX 4090, Vast.ai Norway #37014165)
- [x] Run training (LoRA on fresh base, 3 epochs, loss=0.6545)
- [x] Merge LoRA into base model (10.24 GB bf16 safetensors)
- [x] Convert to GGUF bf16 (8.7 GB)
- [x] Quantize to Q8_0 GGUF (4.6 GB)
- [x] Fix tensor overflow (61 NaN/Inf values in token_embd — clamped)
- [x] Generate Q6_K for V8 and V7 (3.6 GB each)
- [x] Generate mmproj Q8_0 for V8 and V7 (530 MB each, via Python gguf.quantize)
- [x] Upload all variants to HuggingFace partoguard org
- [ ] Eval on synthetic 350-image corpus
- [ ] Eval on WHO demo images
- [ ] On-device verification (S24 Ultra)

## Reproduction

```bash
# On Vast.ai RTX 4090 instance:
export HF_TOKEN=<token>

# 1. Upload training data + script + partoguard package
scp -P <port> /tmp/training_data.tar.gz <your-gpu-instance>:/workspace/
scp -P <port> scripts/finetune_e2b.py <your-gpu-instance>:/workspace/
tar czf /tmp/partoguard_pkg.tar.gz -C /root/work partoguard/
scp -P <port> /tmp/partoguard_pkg.tar.gz <your-gpu-instance>:/workspace/

# 2. On remote:
cd /workspace && tar xzf training_data.tar.gz && tar xzf partoguard_pkg.tar.gz
pip install transformers peft trl bitsandbytes accelerate datasets pillow huggingface_hub pydantic opencv-python-headless

# 3. Train
python3 finetune_e2b.py --version v8 --epochs 3 --lr 2e-4 --lora-r 16 \
    --train-labels data/training/labels.json --skip-upload

# 4. Merge (simple script — PeftModel.from_pretrained + merge_and_unload)
python3 merge_v8.py

# 5. Convert to GGUF
pip install gguf && git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
python3 /workspace/llama.cpp/convert_hf_to_gguf.py /root/partoguard-lora/merged_v8 \
    --outtype q8_0 --outfile /root/partoguard-lora/partoguard-v8-q8_0.gguf
```

## Quantization Options (Research, 2026-05-18)

### Context

We already have Q8_0 (4.6 GB) + F16 mmproj (985 MB) = ~5.6 GB total on the Android device. Investigating smaller quants to reduce download/RAM.

**Critical distinction**: Previous INT4 failures (V13/V14/V15 at 57-59%) were about **quantization-aware training** (LoRA on quantized base). What we're considering here is **post-training quantization** of an already-trained bf16 model — completely different. V7 Q8_0 PTQ achieves 95.71% precisely because it's bf16-trained then quantized.

### Gemma 4 E2B Perplexity by Quant Level (llama.cpp #22407, wikitext-2)

| Quant | PPL | Δ vs BF16 | Size (text) | Total w/ mmproj |
|-------|-----|-----------|-------------|-----------------|
| BF16 | 8.3238 | baseline | 8.7 GB | 9.7 GB |
| **Q8_0** | 8.3393 | **+0.19%** | **4.6 GB** | **5.6 GB** |
| **Q6_K** | 8.3879 | **+0.77%** | **~3.6 GB** | **~4.6 GB** |
| **Q5_K_M** | 8.5069 | **+2.20%** | **~3.4 GB** | **~4.4 GB** |
| Q4_K_M | 9.1381 | +9.8% | ~3.3 GB | ~4.3 GB |
| Q3_K_M | 12.693 | +52% | — | — |

**Key finding**: E2B degrades smoothly — no cliff until Q3. (E4B has a cliff at Q5_K_M; E2B does not.)

### Recommendation for Medical Chart Extraction

| Option | Size savings | Risk | Verdict |
|--------|-------------|------|---------|
| **Q6_K** (recommended) | −1.0 GB (−18%) | <0.5% quality loss, "near-lossless" | ✅ Ship this |
| Q5_K_M | −1.2 GB (−21%) | ~1-2% quality loss, validate first | ⚠️ Acceptable if eval holds |
| Q4_K_M | −1.3 GB (−23%) | ~10% PPL increase, tail failures on structured output | ❌ Not for medical |

**Community consensus (bartowski, HuggingFace skills guide)**: Q6_K is "very high quality, near perfect, recommended" for Gemma 4 E2B. Technical/medical tasks should use Q6_K or Q8_0.

### mmproj Quantization

The vision projector (mmproj) can also be quantized:
- **F16**: 985 MB (current, zero risk)
- **Q8_0**: ~700 MB (−29%, near-lossless for ViT)
- **Q4**: ~400 MB (experimental, risky for coordinate extraction)

The `clip-quantize-cli` tool (llama.cpp PR #11644) can do this. mradermacher ships Q8_0 mmproj for Gemma 4 E2B. **Recommend Q8_0 mmproj to save ~285 MB with minimal risk.**

### Conversion Workflow

K-quants require `llama-quantize` — `convert_hf_to_gguf.py` only outputs f32/f16/bf16/q8_0. Two-step:

```bash
# Already have bf16 GGUF from conversion step:
# /root/partoguard-lora/partoguard-v8-bf16.gguf (8.7 GB)

# Quantize from bf16 source (NEVER from already-quantized):
./llama-quantize partoguard-v8-bf16.gguf partoguard-v8-Q6_K.gguf Q6_K
./llama-quantize partoguard-v8-bf16.gguf partoguard-v8-Q5_K_M.gguf Q5_K_M
```

### Size Comparison (Android deployment)

| Config | Model | mmproj | Total | Download savings |
|--------|-------|--------|-------|-----------------|
| Q8_0 + F16 mmproj (current) | 4.6 GB | 985 MB | 5.6 GB | baseline |
| **Q6_K + F16 mmproj** | ~3.6 GB | 985 MB | **4.6 GB** | **−1.0 GB** |
| **Q6_K + Q8_0 mmproj** | ~3.6 GB | ~700 MB | **4.3 GB** | **−1.3 GB** |
| Q5_K_M + Q8_0 mmproj | ~3.4 GB | ~700 MB | 4.1 GB | −1.5 GB |

### Completed Steps (2026-05-18)

1. ✅ Built `llama-quantize` on Vast.ai (CPU-only)
2. ✅ Generated Q6_K from bf16 GGUF for both V8 and V7
3. ✅ Uploaded Q6_K models to HuggingFace (`partoguard/partoguard-v8-q6_k-gguf`, `partoguard/partoguard-v7-q6_k-gguf`)
4. ✅ Uploaded F16 mmprojs for both
5. ✅ Generated Q8_0 mmprojs via Python gguf-py quantization (530 MB each, down from 985 MB F16)
6. ✅ Uploaded Q8_0 mmprojs to respective repos

### HuggingFace Repos (Final State)

| Repo | Contents |
|------|----------|
| `partoguard/partoguard-v8-q6_k-gguf` | `partoguard-v8-Q6_K.gguf` (3.6 GB), `mmproj-partoguard-v8-f16.gguf` (985 MB), `mmproj-partoguard-v8-q8_0.gguf` (530 MB) |
| `partoguard/partoguard-v7-q6_k-gguf` | `v7-Q6_K.gguf` (3.6 GB), `v7_mmproj_f16.gguf` (985 MB), `mmproj-partoguard-v7-q8_0.gguf` (530 MB) |

### Android Deployment Size Options

| Config | Model | mmproj | Total |
|--------|-------|--------|-------|
| Q8_0 + F16 mmproj (old) | 4.6 GB | 985 MB | 5.6 GB |
| **Q6_K + Q8_0 mmproj** (new optimal) | 3.6 GB | 530 MB | **4.1 GB** |
| Q6_K + F16 mmproj | 3.6 GB | 985 MB | 4.6 GB |

### Remaining

- [ ] Full 350-image eval on V8 Q6_K (accuracy check)
- [ ] On-device S24 Ultra verification with Q6_K + Q8_0 mmproj combo
- [ ] Update Android app model download URLs to point to Q6_K repos

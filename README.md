# PartoGuard

> AI-assisted partograph reading for low-resource clinical settings.
> Built for the [Kaggle Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Android](https://img.shields.io/badge/android-12%2B-green)
![Accuracy](https://img.shields.io/badge/eval%20accuracy-95.71%25-brightgreen)

---

## The Problem

A partograph is a paper chart that tracks labour progress — cervical dilation over time — against WHO alert and action lines. When dilation falls behind schedule, it signals obstructed or prolonged labour, a leading cause of maternal and neonatal death in low-resource settings.

The tool is widely available. The challenge is completion and interpretation. Studies in Nigeria and other LMIC settings document that partographs go unfilled, are filled retrospectively, or are filled but not acted on — often because trained interpreters are unavailable when needed.

PartoGuard asks: what if the phone in the midwife's pocket could read the chart?

---

## What PartoGuard Does

1. **Scan** — Midwife aligns a paper partograph to a ghost overlay in the camera viewfinder and captures.
2. **Extract** — A fine-tuned Gemma 4 E2B model reads the photograph and extracts cervical dilation `X` mark positions as structured JSON.
3. **Classify** — A deterministic rule engine applies WHO partograph geometry to produce one of four outputs: `normal`, `alert_zone`, `action_zone`, or `manual_review`.
4. **Review** — Midwife confirms (or edits) the extracted points before any alert is shown.
5. **Act** — App shows an auditable explanation. All outputs say "review per local protocol" — never an autonomous treatment instruction.

Gemma extracts data points. The rule engine makes every clinical decision. Nothing is black-box.

---

## Architecture

```
Camera / Demo image
   ↓
ImageQualityChecker       ← rejects blurry, dim, skewed captures
   ↓
PartographExtractor       ← fine-tuned Gemma 4 E2B via three backends:
   • On-device llama.cpp  ← V7 Q6_K GGUF, ~4.1 GB, ~35s/image (S24 Ultra)
   • On-device LiteRT     ← Google base E2B-it, ~2.6 GB, faster, 86% accuracy
   • Remote llama-server  ← V7 Q8_0 on CPU VPS, 95.71% accuracy, 8s/image
   ↓
ReviewScreen              ← midwife confirms / edits extracted points
   ↓
RuleEngine.evaluate()     ← deterministic, auditable, single Kotlin file
   ↓
ResultsScreen             ← colour-coded alert + animated overlay
```

---

## Accuracy

Full 350-image corpus evaluation (single prompt, no test-time augmentation):

| Deployment | Accuracy | Latency |
|-----------|---------|---------|
| Local bf16 (RTX 5090, HuggingFace) | **350/350 = 100.00%** | 5.77 s/img |
| VPS Q8_0 (llama.cpp, CPU) | **335/350 = 95.71%** | 8.08 s/img |
| VPS bf16 (llama.cpp, CPU) | **336/350 = 96.00%** | 10.09 s/img |
| On-device V7 Q8_0 (S24 Ultra, 4 threads) | TBD | ~35 s/img |

Per-category breakdown (VPS Q8_0):

| Category | Correct | % |
|----------|---------|---|
| blank (50) | 50 | 100% |
| partial (60) | 60 | 100% |
| degraded (80) | 79 | 98.75% |
| obstructed (60) | 58 | 96.67% |
| filled (100) | 88 | 88% |

All 15 failures are clinically safe — they over-call to alert/action, never produce silent normal misses on filled charts.

Eval logs: [`knowledge/eval_logs_v7_remote_q8_0_350.log`](knowledge/eval_logs_v7_remote_q8_0_350.log) · [`knowledge/eval_logs_v7_remote_bf16_350.log`](knowledge/eval_logs_v7_remote_bf16_350.log)

> **Important:** These figures are from synthetic evaluation data. Real-world clinical accuracy requires validation on clinician-labelled real partograph images, which has not been performed.

---

## Models on HuggingFace

| Repo | Contents | Size |
|------|----------|------|
| [`partoguard/partoguard-v7-q6_k-gguf`](https://huggingface.co/partoguard/partoguard-v7-q6_k-gguf) | V7 Q6_K LLM + mmproj Q8_0 | 4.1 GB total |
| [`partoguard/partoguard-v8-q6_k-gguf`](https://huggingface.co/partoguard/partoguard-v8-q6_k-gguf) | V8-WHO Q6_K LLM + mmproj Q8_0 | 4.1 GB total |
| [`partoguard/partoguard-lora-v7`](https://huggingface.co/partoguard/partoguard-lora-v7) | V7 LoRA adapter (100% on eval) | 176 MB |
| [`partoguard/gemma-4-e2b-it-partograph-v7`](https://huggingface.co/partoguard/gemma-4-e2b-it-partograph-v7) | Merged bf16 model | 9.6 GB |

> **Note:** Historical training iterations and LiteRT bundles under the private `ufoq/*` namespace mentioned in the `/knowledge` logs may be shared with hackathon judges or researchers upon request.

V8-WHO adds ~100 real WHO-template training images for improved generalisation to physical charts. Full 350-image eval pending.

---

## Repository Structure

```
partoguard/         Python console — full pipeline (CLI, Gemma extractors, rules, 148 tests)
android/            Kotlin/Compose Android app (Jetpack Compose, CameraX, native llama.cpp)
android/llama/      Custom NDK module — direct upstream llama.cpp JNI bridge (~274 lines C++)
data/               350-image synthetic eval corpus + WHO/CC-BY harvested reference materials
scripts/            LoRA fine-tuning, WHO training data generation, VPS deployment
knowledge/          Architecture decisions, training history, deployment docs, eval logs
```

---

## Quick Start

### Python console (remote VPS inference)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "partoguard[dev]"

# Analyse a single image (uses public demo server)
.venv/bin/partoguard analyze data/filled/filled_0001.png \
  --gemma-remote --gemma-remote-url http://<your-llama-server>:8080/completion

# Run full 350-image evaluation
.venv/bin/partoguard eval --corpus-dir data \
  --gemma-remote --gemma-remote-url http://<your-llama-server>:8080/completion --progress

# Run tests
.venv/bin/pytest partoguard/tests/ -v   # 148 tests, 0 type errors
```

### Android app

```bash
# 1. Set up llama.cpp source (one-time)
cd android/llama && bash setup-llama.sh

# 2. Build and install
cd android && ./gradlew installDebug
```

The app downloads models on-demand. On first launch, select your preferred inference engine:
- **Offline (llama.cpp)** — requires ~4 GB download, ~8 GB device RAM
- **Offline (LiteRT)** — requires ~2.6 GB download, lighter
- **Online (demo server)** — no download, requires network

### Deploy your own llama-server

```bash
# Server (CPU VPS, see scripts/vps_startup.sh for full setup)
llama-server \
  -m v7_q8_0.gguf --mmproj v7_mmproj_f16.gguf \
  -c 4096 -t 16 -tb 16 -b 2048 -ub 2048 \
  --no-mmproj-offload --jinja --port 8080

# Verify
curl http://localhost:8080/health
curl http://localhost:8080/props | jq .modalities.vision   # -> true
```

### Fine-tune your own adapter

```bash
# Generate WHO-template training images
python scripts/generate_who_training.py

# Fine-tune (requires GPU with 16+ GB VRAM)
python scripts/finetune_e2b.py --version v9 --epochs 3 --lr 2e-4 --lora-r 16

# Merge and convert to GGUF
python scripts/merge_lora.py
# then: llama-quantize merged_model.gguf output.gguf Q6_K
```

---

## Training History

| Version | Training samples | Result | Key change |
|---------|-----------------|--------|------------|
| V1 | 400 synthetic | 94.57% | Initial LoRA baseline |
| V5 | 400 | 98.29% | Better dense chart coverage |
| **V7** | **1030** | **100% local / 95.71% VPS** | **Scorer fix + optimal config** |
| V8-WHO | 500 (400+100 WHO) | TBD on synthetic | WHO chart generalisation |
| V13–V15 | 500–700 (4-bit/8-bit QAT) | 57–59% | INT4/INT8 QAT shown insufficient for mark counting |

Full log: [`knowledge/partoguard_training_log.md`](knowledge/partoguard_training_log.md)

**Key finding on quantisation:** Post-training quantisation of the bf16 model (Q8_0, Q6_K) preserves
accuracy (≤0.8 pp perplexity loss). Quantisation-aware training at INT4 or INT8 destroys spatial
precision needed for mark counting — confirmed across three variants (V13/V14/V15).

---

## Design Principles

**Clinician in the loop.** The model never makes clinical decisions. It extracts numbers; a deterministic rule engine (single, readable, auditable Kotlin file) applies WHO alert/action line geometry.

**Safe failures.** Low image quality, poor template match, or low extraction confidence → `manual_review`. Over-calling to alert/action is safer than silent normal misses.

**Explainability.** Every result shows the extracted points, the rule applied, and the zone geometry. No black-box outputs.

**On-device first.** Models download on-demand; no PHI leaves the device unless the user selects the remote server option.

---

## Clinical Context

PartoGuard targets the modified WHO partograph, which starts active-phase plotting at 4 cm cervical dilation with alert (1 cm/hour) and action (4 hours right-shifted) lines. This form remains common in facility training materials and clinical practice across sub-Saharan Africa despite WHO's newer Labour Care Guide (LCG, 2018) providing a more nuanced framework.

PartoGuard is **not** a replacement for trained clinical judgement, WHO LCG-aligned protocols, or facility referral pathways. All outputs direct midwives to "review per local protocol."

See [`knowledge/partoguard_findings.md`](knowledge/partoguard_findings.md) for supporting evidence on partograph completion barriers and clinical context.

---

## License

This repository is licensed under the **Apache License 2.0**.

This project uses Gemma 4 by Google DeepMind, also released under Apache 2.0.
This project is not an official Google, Google DeepMind, or Kaggle project.

Model weights, adapters, datasets, and third-party dependencies may have separate licenses.
See [`MODEL_CARD.md`](MODEL_CARD.md), [`DATA_LICENSES.md`](DATA_LICENSES.md), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.

---

## Disclaimer

PartoGuard is a research prototype developed for a hackathon. It is **not** a certified medical device, has **not** been clinically validated, and is **not** approved for clinical use. Always follow local clinical protocols and defer to qualified healthcare professionals for all clinical decisions.

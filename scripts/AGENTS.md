# scripts/ — ML Training, Evaluation & Infrastructure

Standalone scripts; NOT part of the `partoguard` package. All require external GPU/VPS environment.

## NAMING CONVENTION

Version suffix = model iteration:

| Suffix | Round | Key change |
|--------|-------|------------|
| (none) | V1 | PEFT LoRA baseline on Intel XPU |
| `_v8` | V8 | Unsloth QLoRA NF4 + unfrozen vision |
| `_v13` | V13 | bnb-QLoRA NF4 — discovered over-counting bug |
| `_v14` | V14 | Unsloth Dynamic 4-bit + unfrozen vision (fixes V13) |
| `_v15` | V15 | Unsloth INT8 W8A16 (higher fidelity than 4-bit) |

## FILE INVENTORY

### Training
| Script | Purpose | Needs |
|--------|---------|-------|
| `finetune_e2b.py` | V1 PEFT LoRA (SFTTrainer) | Intel XPU + HF_TOKEN |
| `finetune_e2b_v8_qlora.py` | V8 QLoRA NF4, unfrozen vision | Unsloth + CUDA |
| `finetune_e2b_v13_bnb_qlora.py` | V13 bnb-QLoRA | bitsandbytes + CUDA |
| `finetune_e2b_v14_dynamic.py` | V14 Dynamic 4-bit | Unsloth + CUDA |
| `finetune_e2b_v15_int8.py` | V15 INT8 LoRA | Unsloth + CUDA |
| `finetune_e2b_v8_loftq.py` | V8 LoRA with LoftQ init | CUDA |
| `finetune_unsloth.py` | Generic Unsloth wrapper | Unsloth + CUDA |
| `generate_training_v2.py` | 700 crop-only training images (seed 99999) | CPU |

### Merge & Export
| Script | Purpose | Notes |
|--------|---------|-------|
| `merge_lora.py` | Manual safetensor-level LoRA merge | Use this — Unsloth merge broken for Gemma4 |
| `merge_lora_v15.py` | V15 transformers merge (avoids Unsloth) | CPU |
| `export_v8_litert.py` | LiteRT mixed48 bundle | needs patched litert-torch |

### Evaluation
| Script | Purpose |
|--------|---------|
| `eval_remote.py` | Full pipeline eval on remote CUDA |
| `eval_unsloth_v13.py` / `v14` / `v15` | Per-version Unsloth evals (JSONL predictions) |
| `eval_v14_local_tokens.py` | int4 visual_token_budget tuning |
| `eval_v15_gguf.py` | GGUF Q8_0 eval via llama-mtmd-cli |
| `score_v13_predictions.py` | Score JSONL predictions offline |

### Diagnostics
| Script | Purpose |
|--------|---------|
| `probe_correctness.py` | Stratified prompt probe + ground-truth scoring |
| `probe_prompt.py` | Fast prompt iteration (~50s per run) |
| `bench_daemon.py` | LiteRT daemon vs subprocess latency |
| `bench_raw_models.py` | E2B vs E4B raw throughput |
| `debug_model.py` | Single-image merged model debug |
| `quant_diagnostic.py` | Quantization level comparison on known-failing images |

### Infrastructure
| Script | Purpose |
|--------|---------|
| `vps_startup.sh` | Idempotent VPS llama-server boot (checks stale PIDs, sets `LD_LIBRARY_PATH` for libgomp) |
| `remote_setup.sh` | Full Vast.ai training pipeline (install → train → eval 100 → full eval) |

### Prompts (`prompts/`)
6 archived prompt variants from iteration (p0_baseline → p5_hybrid). `p5_hybrid.txt` is the production prompt basis used by `_build_remote_extraction_prompt()` in `gemma_adapter.py`.

## CONVENTIONS

- **All safe to re-run** — fine-tune outputs go to `/root/partoguard-lora/` (not committed); never overwrite existing checkpoints.
- **`HF_TOKEN` required for uploads** — set env var or uploads silently skip (`WARNING: HF_TOKEN not set`).
- **Use tmux for long-running processes on VPS** — never `nohup` or `&` (hangs terminal emulators).
- **Model artifacts** land at `/root/partoguard-lora/` (local, not committed) — regenerate from scripts.
- **`vps_startup.sh`** is installed as `/home/admin/.startup` on the VPS — `/home` is persistent across container rebuilds, `/` is not.

## ANTI-PATTERNS

- ❌ Running fine-tune scripts without `HF_TOKEN` — uploads silently skipped
- ❌ Quantizing the full vision encoder to 4-bit — accuracy collapse (see `knowledge/partoguard_int4_quant_paths.md`): use dynamic exclusion or INT8
- ❌ Using `nohup` / `&` for long processes — use `tmux new-session` instead
- ❌ Using Unsloth's built-in LoRA merge for Gemma4 — broken; use `merge_lora.py` (manual safetensor merge)
- ❌ Including `<bos>` in prompt strings (see `partoguard_remote_gemma_vps.md`) — tokenizer auto-adds it

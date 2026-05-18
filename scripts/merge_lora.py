#!/usr/bin/env python3
"""Manual LoRA merge at safetensors level.

Unsloth's save_pretrained_merged is broken for Gemma4 — it ignores LoRA
weights due to key naming mismatch (.default. suffix). This script does
the merge correctly: W_merged = W_base + (alpha/r) * B @ A.

Usage:
    python scripts/merge_lora.py --adapter /path/to/lora_adapter --output /path/to/merged
    python scripts/merge_lora.py  # uses defaults
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

if "HF_TOKEN" not in os.environ:
    raise RuntimeError("HF_TOKEN env var required")

BASE_MODEL = "ufoq/partoguard-gemma4-e2b-ft-v1"
DEFAULT_ADAPTER = "/workspace/partoguard-lora/lora_adapter"
DEFAULT_OUTPUT = "/workspace/partoguard-lora/merged"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)

    # Load adapter config
    with open(adapter_dir / "adapter_config.json") as f:
        adapter_cfg = json.load(f)

    r = adapter_cfg["r"]
    alpha = adapter_cfg["lora_alpha"]
    scaling = alpha / r
    print(f"r={r}, alpha={alpha}, scaling={scaling}")

    # Load base model weights
    print(f"Loading base model weights from {args.base_model}...")
    base_path = hf_hub_download(args.base_model, "model.safetensors")
    base_tensors: dict[str, torch.Tensor] = {}
    with safe_open(base_path, framework="pt") as f:
        for k in f.keys():
            base_tensors[k] = f.get_tensor(k)
    print(f"Base model: {len(base_tensors)} tensors")

    # Load adapter weights
    print("Loading adapter weights...")
    adapter_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            adapter_tensors[k] = f.get_tensor(k)
    print(f"Adapter: {len(adapter_tensors)} tensors")

    # Group LoRA pairs and merge
    lora_a_keys = sorted(k for k in adapter_tensors if "lora_A.weight" in k)
    print(f"Found {len(lora_a_keys)} LoRA A matrices")

    merged_count = 0
    for a_key in lora_a_keys:
        b_key = a_key.replace("lora_A.weight", "lora_B.weight")
        if b_key not in adapter_tensors:
            print(f"WARNING: No B for {a_key}")
            continue

        # Map adapter key to base key
        # Adapter: base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
        # Base:    model.language_model.layers.0.mlp.down_proj.weight
        base_key = a_key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")

        if base_key not in base_tensors:
            print(f"WARNING: Base key not found: {base_key}")
            continue

        A = adapter_tensors[a_key].float()
        B = adapter_tensors[b_key].float()
        W = base_tensors[base_key].float()

        delta = scaling * (B @ A)
        base_tensors[base_key] = (W + delta).to(base_tensors[base_key].dtype)
        merged_count += 1

    print(f"Merged {merged_count} LoRA pairs")

    # Save merged model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir}...")
    save_file(base_tensors, str(output_dir / "model.safetensors"))

    # Copy config and tokenizer from base
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                   "processor_config.json", "chat_template.jinja"]:
        try:
            src = hf_hub_download(args.base_model, fname)
            shutil.copy2(src, str(output_dir / fname))
        except Exception as e:
            print(f"Skip {fname}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()

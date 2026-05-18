#!/usr/bin/env python3
"""Eval fine-tuned model on remote CUDA against the 350-image eval corpus.

Uses the full PartoGuard pipeline (extraction + rule engine + corpus scorer)
with the merged model as the XPU extractor (but on CUDA).

Usage:
    python scripts/eval_remote.py [--limit 100] [--progress]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if "HF_TOKEN" not in os.environ:
    raise RuntimeError("HF_TOKEN env var required")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_PATH = "/workspace/partoguard-lora/merged"
CORPUS_DIR = Path("data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from PIL import Image

    from partoguard.core.extraction.gemma_adapter import (
        _build_litert_extraction_prompt, _json_payload_from_text,
        _normalize_extraction_payload, _phase1_points_from_payload,
    )
    from partoguard.core.corpus_scorer import score_manifest_entry
    from partoguard.core.schemas.contracts import ZoneStatus
    from partoguard.core.rules.engine import classify_zone

    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())

    if args.limit and args.limit < len(manifest):
        import random
        random.seed(args.sample_seed)
        manifest = random.sample(manifest, args.limit)

    print(f"Loading {MODEL_PATH} with Unsloth...")
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(
        MODEL_PATH, load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)

    prompt = _build_litert_extraction_prompt()

    correct = 0
    scored = 0
    failures = []
    t0 = time.time()

    for i, entry in enumerate(manifest):
        img_path = CORPUS_DIR / entry["path"]
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
        ).to("cuda")

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=300, do_sample=False, use_cache=True)

        text = processor.decode(output[0], skip_special_tokens=True)
        parts = text.split("model")
        text = parts[-1].strip() if len(parts) > 1 else text.strip()

        try:
            payload = _json_payload_from_text(text)
            payload = _normalize_extraction_payload(payload)
            points = _phase1_points_from_payload(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            points = []

        if points:
            rule_result = classify_zone(points)
            status = rule_result.status
        else:
            status = ZoneStatus.MANUAL_REVIEW

        verdict = score_manifest_entry(entry, actual_status=status, actual_n_points=len(points))
        scored += 1
        if verdict.correct:
            correct += 1
        else:
            failures.append({
                "path": entry["path"],
                "category": entry["category"],
                "n_marks": entry.get("n_marks", 0),
                "predicted_n": len(points),
                "curve_type": entry.get("curve_type", "?"),
                "paper_style": entry.get("paper_style", "?"),
                "reason": verdict.reason,
            })

        if args.progress and (scored % 10 == 0 or scored == len(manifest)):
            elapsed = time.time() - t0
            rate = scored / elapsed if elapsed > 0 else 0
            print(f"[{scored}/{len(manifest)}] {correct}/{scored} correct ({100*correct/scored:.1f}%) | {rate:.1f} img/s | {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nFinal: {correct}/{scored} ({100*correct/scored:.1f}%) in {elapsed:.0f}s")
    print(f"Rate: {scored/elapsed:.1f} img/s")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        by_cat = {}
        for f in failures:
            by_cat.setdefault(f["category"], []).append(f)
        for cat, items in sorted(by_cat.items()):
            print(f"  {cat} ({len(items)}):")
            for f in items:
                print(f"    {f['path']}: n_marks={f['n_marks']} predicted={f['predicted_n']} curve={f['curve_type']} paper={f['paper_style']} reason={f['reason']}")

    results = {
        "correct": correct, "total": scored,
        "accuracy": correct / scored if scored > 0 else 0,
        "elapsed_s": elapsed,
        "failures": failures,
    }
    Path("/workspace/eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to /workspace/eval_results.json")


if __name__ == "__main__":
    main()

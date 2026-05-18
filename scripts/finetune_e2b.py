"""Fine-tune Gemma 4 E2B-it with PEFT LoRA for partograph X-mark reading.

Uses raw PEFT + transformers (no Unsloth) — V1 recipe that achieved 94.57%.

Usage:
    python scripts/finetune_e2b.py [--dry-run] [--epochs N] [--lr LR] [--version v5]
    python scripts/finetune_e2b.py --train-labels data/training/labels.json --train-labels data/training_v2/labels.json

Requires: peft, trl, datasets, accelerate, torch (CUDA)
Training data: data/training/labels.json + images
Output: {output-dir}/lora_adapter_{version}/
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

if "HF_TOKEN" not in os.environ:
    raise RuntimeError("HF_TOKEN env var required for model download and upload")

MODEL_ID = "google/gemma-4-E2B-it"
DATA_DIR = Path("data")
DEFAULT_TRAIN_LABELS = DATA_DIR / "training" / "labels.json"
DEFAULT_OUTPUT_DIR = Path("/root/partoguard-lora")

HF_WRITE_TOKEN = os.environ["HF_TOKEN"]


# ---------- Preprocessing transforms (applied uniformly to train + eval) ----------

def preprocess_chromatic(img: "Image.Image") -> "Image.Image":
    """Map x-axis position to blue→red color gradient to encode spatial position."""
    arr = np.array(img).astype(np.float32)
    w = arr.shape[1]
    g = np.linspace(0, 1, w).reshape(1, -1)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + g * 40, 0, 255)   # R increases left→right
    arr[:, :, 2] = np.clip(arr[:, :, 2] + (1 - g) * 40, 0, 255)  # B decreases
    from PIL import Image
    return Image.fromarray(arr.astype(np.uint8))


def preprocess_fft_notch(img: "Image.Image") -> "Image.Image":
    """Remove periodic grid lines via FFT notch filter, preserve pencil marks."""
    import cv2
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    dc = fshift[crow, ccol].copy()
    fshift[crow - 2:crow + 3, :] = 0
    fshift[:, ccol - 2:ccol + 3] = 0
    fshift[crow, ccol] = dc
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(img_back, cv2.COLOR_GRAY2RGB))


def preprocess_clahe_lab(img: "Image.Image") -> "Image.Image":
    """Apply CLAHE on LAB L-channel for contrast normalization."""
    import cv2
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


PREPROCESS_REGISTRY: dict[str, Callable] = {
    "chromatic": preprocess_chromatic,
    "fft_notch": preprocess_fft_notch,
    "clahe_lab": preprocess_clahe_lab,
}


def get_preprocess_fn(name: str | None) -> Callable | None:
    if name is None or name == "none":
        return None
    if name not in PREPROCESS_REGISTRY:
        raise ValueError(f"Unknown preprocess: {name}. Options: {list(PREPROCESS_REGISTRY.keys())}")
    return PREPROCESS_REGISTRY[name]


def build_target_json(sample: dict) -> str:
    pts = sample["points"]
    clamped = []
    for p in pts:
        h = max(0.0, min(12.0, round(p[0] * 2) / 2))
        cm = max(0.0, min(10.0, round(p[1] * 2) / 2))
        clamped.append([h, cm, 0.99])
    return json.dumps({"p": clamped}, separators=(",", ":"))


def build_system_prompt() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from partoguard.core.extraction.gemma_adapter import _build_litert_extraction_prompt
    return _build_litert_extraction_prompt()


def load_training_data(
    label_files: list[str],
    data_dir: Path,
    category_filter: str | None = None,
    preprocess_fn: Callable | None = None,
) -> list[dict]:
    from PIL import Image

    prompt = build_system_prompt()
    samples = []

    for lf in label_files:
        labels = json.load(open(lf))
        for entry in labels:
            if category_filter and entry.get("category") != category_filter:
                continue
            img_path = data_dir / entry["image"]
            if not img_path.exists():
                print(f"  SKIP missing: {img_path}")
                continue
            img = Image.open(img_path).convert("RGB")
            if preprocess_fn is not None:
                img = preprocess_fn(img)
            target = build_target_json(entry)
            samples.append({"image": img, "prompt": prompt, "target": target})

    print(f"Loaded {len(samples)} training samples from {len(label_files)} file(s)"
          + (f" (category={category_filter})" if category_filter else ""))
    return samples


def create_dataset(samples: list[dict], processor):
    from datasets import Dataset

    records = []
    for s in samples:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": s["image"]},
                    {"type": "text", "text": s["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": s["target"]}],
            },
        ]
        records.append({"messages": messages, "pil_imgs": [s["image"]]})

    return Dataset.from_list(records)


class VisionChatCollator:
    """Tokenizes vision+text chat samples via the processor's chat template."""

    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        texts = []
        images_list = []

        for sample in batch:
            text = self.processor.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images_list.append(sample["pil_imgs"])

        inputs = self.processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Load model + 1 step only")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--category", type=str, default=None, help="Train only on this category")
    parser.add_argument("--version", type=str, default="v5", help="Version tag for output dir and HF repo")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--train-labels", type=str, action="append", help="Label file(s) to use (can specify multiple)")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF upload after training")
    parser.add_argument("--hf-repo", type=str, default=None, help="HF repo for adapter upload")
    parser.add_argument("--preprocess", type=str, default=None, choices=list(PREPROCESS_REGISTRY.keys()),
                        help="Apply uniform preprocessing to all training images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / f"lora_adapter_{args.version}"

    label_files = args.train_labels or [str(DEFAULT_TRAIN_LABELS)]
    hf_repo = args.hf_repo or f"ufoq/partoguard-lora-{args.version}"
    preprocess_fn = get_preprocess_fn(args.preprocess)

    print(f"=== PartoGuard LoRA Training {args.version} ===")
    print(f"Model: {MODEL_ID}")
    print(f"Labels: {label_files}")
    print(f"Output: {adapter_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, LoRA r: {args.lora_r}")
    if args.preprocess:
        print(f"Preprocessing: {args.preprocess}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model, TaskType
    from trl.trainer.sft_trainer import SFTTrainer
    from trl.trainer.sft_config import SFTConfig

    print(f"Loading {MODEL_ID} in bf16...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )

    print("Attaching LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    samples = load_training_data(label_files, DATA_DIR, category_filter=args.category, preprocess_fn=preprocess_fn)

    if args.dry_run:
        samples = samples[:4]
        args.epochs = 1
        print("DRY RUN: 4 samples, 1 epoch")

    dataset = create_dataset(samples, processor)
    collator = VisionChatCollator(processor, max_length=4096)

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.001,
        logging_steps=5,
        save_strategy="epoch",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=4096,
        bf16=(device != "cpu"),
        fp16=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
    )

    print(f"Starting training: {len(dataset)} samples x {args.epochs} epochs...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
        args=training_args,
    )

    stats = trainer.train()
    print(f"\nTraining complete. Loss: {stats.training_loss:.4f}")

    print(f"Saving LoRA adapter to {adapter_dir}...")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    if not args.skip_upload:
        print(f"Uploading adapter to {hf_repo}...")
        from huggingface_hub import HfApi
        api = HfApi(token=HF_WRITE_TOKEN)
        api.create_repo(hf_repo, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=hf_repo,
            commit_message=f"{args.version} LoRA r={args.lora_r} lr={args.lr} epochs={args.epochs} samples={len(dataset)}",
        )
        print(f"Adapter uploaded to {hf_repo}")

    print(f"\nDone! Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()

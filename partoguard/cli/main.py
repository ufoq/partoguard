from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from partoguard.core.eval import evaluate_corpus_dir, evaluate_synthetic_dir, format_corpus_eval_summary, format_eval_summary
from partoguard.core.extraction.gemma_adapter import build_verifier
from partoguard.core.imaging.synthetic import generate_all
from partoguard.core.pipeline import analyze_image
from partoguard.core.reports.generator import write_json_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="partoguard",
        description="PartoGuard — clinical decision support for partograph review",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze a partograph image")
    analyze.add_argument("image", type=Path, help="Path to a partograph image")
    analyze.add_argument("--json", action="store_true", dest="json_output", help="Print JSON audit to stdout")
    analyze.add_argument("--json-out", type=Path, help="Write JSON audit to this path")
    analyze.add_argument("--gemma-command", help="Trusted local Gemma verifier command. Output must contain accepted_points only.")
    analyze.add_argument("--gemma-litert-e2b", action="store_true", help="Use local Gemma 4 E2B via LiteRT-LM (mobile-friendly, ~5s/image) [default Gemma model]")
    analyze.add_argument("--gemma-litert-e4b", action="store_true", help="Use local Gemma 4 E4B via LiteRT-LM (larger, ~10s/image, slightly higher recall)")
    analyze.add_argument("--gemma-xpu-e2b", action="store_true", help="Use Gemma 4 E2B on Intel XPU via HuggingFace transformers (~3s/image)")
    analyze.add_argument("--gemma-xpu-e2b-ft", action="store_true", help="Use fine-tuned Gemma 4 E2B on Intel XPU (~3s/image)")
    analyze.add_argument("--gemma-cuda-e2b-ft", action="store_true", help="Use fine-tuned Gemma 4 E2B on CUDA (~8s/image)")
    analyze.add_argument("--gemma-xpu-e4b", action="store_true", help="Use Gemma 4 E4B 4-bit on Intel XPU via HuggingFace transformers (~8s/image)")
    analyze.add_argument("--gemma-remote", action="store_true", help="Use a remote llama.cpp server (configure with --gemma-remote-url)")
    analyze.add_argument("--gemma-remote-url", type=str, default="http://vps-box:8080/completion", help="URL of the remote llama.cpp /completion endpoint")
    analyze.add_argument("--litert-bin", type=Path, help="Path to litert-lm binary for --gemma-litert-*")
    analyze.add_argument("--litert-model-path", type=Path, help="Path to a custom .litertlm model file (overrides default HF-cached model)")

    generate = subparsers.add_parser("generate", help="Generate synthetic demo images")
    generate.add_argument("--output-dir", type=Path, default=Path("partoguard/data/synthetic"))

    evaluate = subparsers.add_parser("eval", help="Evaluate against generated synthetic labels")
    evaluate.add_argument("--synthetic-dir", type=Path, default=Path("partoguard/data/synthetic"))
    evaluate.add_argument("--corpus-dir", type=Path, help="Evaluate the full data/manifest.json corpus instead of labels.json")
    evaluate.add_argument("--gemma-litert-e2b", action="store_true", help="Use local Gemma 4 E2B via LiteRT-LM as the extractor (mobile-friendly, ~5s/image) [default Gemma model]")
    evaluate.add_argument("--gemma-litert-e4b", action="store_true", help="Use local Gemma 4 E4B via LiteRT-LM as the extractor (larger, ~10s/image)")
    evaluate.add_argument("--gemma-xpu-e2b", action="store_true", help="Use Gemma 4 E2B on Intel XPU via HuggingFace transformers (~3s/image)")
    evaluate.add_argument("--gemma-xpu-e2b-ft", action="store_true", help="Use fine-tuned Gemma 4 E2B on Intel XPU (~3s/image)")
    evaluate.add_argument("--gemma-cuda-e2b-ft", action="store_true", help="Use fine-tuned Gemma 4 E2B on CUDA (~8s/image)")
    evaluate.add_argument("--adapter-path", type=str, help="Path to LoRA adapter (overrides default for --gemma-cuda-e2b-ft)")
    evaluate.add_argument("--gemma-xpu-e4b", action="store_true", help="Use Gemma 4 E4B 4-bit on Intel XPU via HuggingFace transformers (~8s/image)")
    evaluate.add_argument("--gemma-remote", action="store_true", help="Use a remote llama.cpp server (configure with --gemma-remote-url)")
    evaluate.add_argument("--gemma-remote-url", type=str, default="http://vps-box:8080/completion", help="URL of the remote llama.cpp /completion endpoint")
    evaluate.add_argument("--litert-bin", type=Path, help="Path to litert-lm binary for --gemma-litert-*")
    evaluate.add_argument("--litert-model-path", type=Path, help="Path to a custom .litertlm model file (overrides default HF-cached model)")
    evaluate.add_argument("--limit", type=int, help="Evaluate only N images (sampled if --sample-seed is given, else first N)")
    evaluate.add_argument("--sample-seed", type=int, help="Random seed for --limit sampling")
    evaluate.add_argument("--progress", action="store_true", help="Print per-image progress to stderr")
    evaluate.add_argument("--preprocess", type=str, default=None,
                          choices=["chromatic", "fft_notch", "clahe_lab"],
                          help="Apply uniform preprocessing to eval images (must match training)")

    parser.add_argument("legacy_image", nargs="?", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None and args.legacy_image is not None:
        return _analyze(args.legacy_image, False, None, None, False, False, False, False, False, False, None)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "generate":
        results = generate_all(args.output_dir)
        print(f"Generated {len(results)} synthetic demo images in {args.output_dir}")
        print(f"Labels: {args.output_dir / 'labels.json'}")
        return 0
    if args.command == "analyze":
        return _analyze(args.image, args.json_output, args.json_out, args.gemma_command, args.gemma_litert_e2b, args.gemma_litert_e4b, args.gemma_xpu_e2b, args.gemma_xpu_e4b, args.gemma_xpu_e2b_ft, args.gemma_cuda_e2b_ft, args.litert_bin, args.litert_model_path, args.gemma_remote, args.gemma_remote_url)
    if args.command == "eval":
        if hasattr(args, "adapter_path") and args.adapter_path:
            import os  # noqa: PLC0415
            os.environ["PARTOGUARD_ADAPTER_PATH"] = args.adapter_path
        if hasattr(args, "preprocess") and args.preprocess:
            import os as _os  # noqa: PLC0415
            _os.environ["PARTOGUARD_PREPROCESS"] = args.preprocess
        try:
            factory = _eval_verifier_factory(args.gemma_litert_e2b, args.gemma_litert_e4b, args.gemma_xpu_e2b, args.gemma_xpu_e4b, args.gemma_xpu_e2b_ft, args.gemma_cuda_e2b_ft, args.litert_bin, args.litert_model_path, args.gemma_remote, args.gemma_remote_url)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.corpus_dir is not None:
            summary = evaluate_corpus_dir(
                args.corpus_dir,
                verifier_factory=factory,
                limit=args.limit,
                sample_seed=args.sample_seed,
                progress=args.progress,
            )
            print(format_corpus_eval_summary(summary))
            return 0
        synthetic_summary = evaluate_synthetic_dir(
            args.synthetic_dir,
            verifier_factory=factory,
            limit=args.limit,
            sample_seed=args.sample_seed,
            progress=args.progress,
        )
        print(format_eval_summary(synthetic_summary))
        return 0

    parser.print_help()
    return 1


def _eval_verifier_factory(use_litert_e2b: bool, use_litert_e4b: bool, use_xpu_e2b: bool, use_xpu_e4b: bool, use_xpu_e2b_ft: bool, use_cuda_e2b_ft: bool, litert_bin: Path | None, litert_model_path: Path | None, use_remote: bool = False, remote_url: str | None = None):
    if not use_litert_e2b and not use_litert_e4b and not use_xpu_e2b and not use_xpu_e4b and not use_xpu_e2b_ft and not use_cuda_e2b_ft and not use_remote:
        return None
    return lambda: build_verifier(None, use_litert_e2b=use_litert_e2b, use_litert_e4b=use_litert_e4b, use_xpu_e2b=use_xpu_e2b, use_xpu_e4b=use_xpu_e4b, use_xpu_e2b_ft=use_xpu_e2b_ft, use_cuda_e2b_ft=use_cuda_e2b_ft, litert_bin=litert_bin, litert_model_path=litert_model_path, use_remote=use_remote, remote_url=remote_url)


def _analyze(
    image_path: Path,
    json_output: bool,
    json_out: Path | None,
    gemma_command: str | None,
    gemma_litert_e2b: bool = False,
    gemma_litert_e4b: bool = False,
    gemma_xpu_e2b: bool = False,
    gemma_xpu_e4b: bool = False,
    gemma_xpu_e2b_ft: bool = False,
    gemma_cuda_e2b_ft: bool = False,
    litert_bin: Path | None = None,
    litert_model_path: Path | None = None,
    gemma_remote: bool = False,
    gemma_remote_url: str | None = None,
) -> int:
    if not image_path.exists():
        print("Error: image not found", file=sys.stderr)
        return 1
    try:
        verifier = build_verifier(gemma_command, use_litert_e2b=gemma_litert_e2b, use_litert_e4b=gemma_litert_e4b, use_xpu_e2b=gemma_xpu_e2b, use_xpu_e4b=gemma_xpu_e4b, use_xpu_e2b_ft=gemma_xpu_e2b_ft, use_cuda_e2b_ft=gemma_cuda_e2b_ft, litert_bin=litert_bin, litert_model_path=litert_model_path, use_remote=gemma_remote, remote_url=gemma_remote_url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    result = analyze_image(image_path, verifier=verifier)
    if json_out is not None:
        write_json_audit(json_out, result.json_audit)
    if json_output:
        print(json.dumps(result.json_audit, indent=2))
    else:
        print(result.text_report)
        if json_out is not None:
            print(f"JSON audit written to {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

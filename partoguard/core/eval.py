from __future__ import annotations

import json
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from partoguard.core.corpus_scorer import ScoreVerdict, score_manifest_entry
from partoguard.core.extraction.gemma_adapter import GemmaVerifier, StubVerifier
from partoguard.core.pipeline import analyze_image


VerifierFactory = Callable[[], GemmaVerifier]


def _default_verifier_factory() -> GemmaVerifier:
    return StubVerifier()


@dataclass(frozen=True)
class EvalSummary:
    total: int
    evaluated: int
    correct_zones: int
    incorrect_zones: int
    manual_reviews: int

    @property
    def zone_accuracy(self) -> float:
        return self.correct_zones / self.evaluated if self.evaluated else 0.0

    @property
    def manual_review_rate(self) -> float:
        return self.manual_reviews / self.total if self.total else 0.0

    @property
    def full_set_success_rate(self) -> float:
        return self.correct_zones / self.total if self.total else 0.0


@dataclass(frozen=True)
class CorpusEvalSummary:
    total: int
    manual_reviews: int
    non_manual: int
    blank_total: int
    blank_manual_reviews: int
    by_category: dict[str, dict[str, int]]
    elapsed_seconds: float = 0.0
    correctness_scored: int = 0
    correctness_correct: int = 0
    by_category_correctness: dict[str, dict[str, int]] | None = None

    @property
    def manual_review_rate(self) -> float:
        return self.manual_reviews / self.total if self.total else 0.0

    @property
    def blank_manual_review_rate(self) -> float:
        return self.blank_manual_reviews / self.blank_total if self.blank_total else 0.0

    @property
    def seconds_per_image(self) -> float:
        return self.elapsed_seconds / self.total if self.total else 0.0

    @property
    def correctness_rate(self) -> float:
        return self.correctness_correct / self.correctness_scored if self.correctness_scored else 0.0


def evaluate_synthetic_dir(
    synthetic_dir: Path,
    *,
    verifier_factory: VerifierFactory | None = None,
    limit: int | None = None,
    sample_seed: int | None = None,
    progress: bool = False,
) -> EvalSummary:
    factory = verifier_factory or _default_verifier_factory
    labels_path = synthetic_dir / "labels.json"
    labels = json.loads(labels_path.read_text())
    labels = _sample(labels, limit, sample_seed)
    total = len(labels)
    evaluated = 0
    correct = 0
    incorrect = 0
    manual_reviews = 0
    verifier = factory()
    for index, label in enumerate(labels):
        image_path, expected = _label_image(synthetic_dir, label)
        result = analyze_image(image_path, verifier=verifier)
        status = result.rule_output.status.value
        if status == "manual_review":
            manual_reviews += 1
        else:
            evaluated += 1
            if status == expected:
                correct += 1
            else:
                incorrect += 1
        if progress:
            print(f"  [{index + 1}/{total}] {image_path.name} -> {status}", file=sys.stderr, flush=True)
    return EvalSummary(total, evaluated, correct, incorrect, manual_reviews)


def evaluate_corpus_dir(
    corpus_dir: Path,
    *,
    verifier_factory: VerifierFactory | None = None,
    limit: int | None = None,
    sample_seed: int | None = None,
    progress: bool = False,
) -> CorpusEvalSummary:
    factory = verifier_factory or _default_verifier_factory
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest = _sample(manifest, limit, sample_seed)
    total = len(manifest)
    manual_reviews = 0
    non_manual = 0
    blank_total = 0
    blank_manual_reviews = 0
    by_category: dict[str, dict[str, int]] = {}
    by_category_correctness: dict[str, dict[str, int]] = {}
    correctness_scored = 0
    correctness_correct = 0

    verifier = factory()
    t_start = time.monotonic()
    for index, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            raise ValueError("manifest entry must be an object")
        image_path = _corpus_image_path(corpus_dir, entry)
        category = str(entry.get("category", "unknown"))
        category_counts = by_category.setdefault(category, {"total": 0, "manual_review": 0, "non_manual": 0})
        category_counts["total"] += 1
        if category == "blank":
            blank_total += 1

        result = analyze_image(image_path, verifier=verifier)
        status = result.rule_output.status
        if status.value == "manual_review":
            manual_reviews += 1
            category_counts["manual_review"] += 1
            if category == "blank":
                blank_manual_reviews += 1
        else:
            non_manual += 1
            category_counts["non_manual"] += 1

        verdict: ScoreVerdict | None = None
        if _has_ground_truth(entry):
            verdict = score_manifest_entry(
                entry,
                actual_status=status,
                actual_n_points=len(result.extraction.points),
            )
            correctness_scored += 1
            cat_corr = by_category_correctness.setdefault(
                category, {"total": 0, "correct": 0, "incorrect": 0}
            )
            cat_corr["total"] += 1
            if verdict.correct:
                correctness_correct += 1
                cat_corr["correct"] += 1
            else:
                cat_corr["incorrect"] += 1

        if progress:
            elapsed = time.monotonic() - t_start
            verdict_tag = ""
            if verdict is not None:
                verdict_tag = "  OK" if verdict.correct else f"  BAD ({verdict.reason})"
            print(
                f"  [{index + 1}/{total}] {image_path.name} -> {status.value}{verdict_tag}  "
                f"({elapsed:.1f}s elapsed)",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    return CorpusEvalSummary(
        total=total,
        manual_reviews=manual_reviews,
        non_manual=non_manual,
        blank_total=blank_total,
        blank_manual_reviews=blank_manual_reviews,
        by_category=by_category,
        elapsed_seconds=elapsed,
        correctness_scored=correctness_scored,
        correctness_correct=correctness_correct,
        by_category_correctness=by_category_correctness if correctness_scored else None,
    )


def _has_ground_truth(entry: dict[str, object]) -> bool:
    return (
        "n_marks" in entry
        and "curve_type" in entry
        and "category" in entry
        and isinstance(entry.get("n_marks"), (int, float))
    )


def _sample(items: list[object], limit: int | None, sample_seed: int | None) -> list[object]:
    if limit is None or limit >= len(items):
        return items
    if sample_seed is None:
        return items[:limit]
    rng = random.Random(sample_seed)
    return rng.sample(items, limit)


def _label_image(synthetic_dir: Path, label: object) -> tuple[Path, str]:
    if not isinstance(label, dict):
        raise ValueError("labels.json entry must be an object")
    file_value = label.get("file")
    expected = label.get("expected_zone")
    if not isinstance(file_value, str) or not isinstance(expected, str):
        raise ValueError("labels.json entry missing file or expected_zone")
    image_path = Path(file_value)
    if image_path.is_absolute():
        try:
            image_path.relative_to(synthetic_dir.resolve())
        except ValueError as exc:
            raise ValueError("labels.json contains an image outside the synthetic directory") from exc
    else:
        image_path = synthetic_dir / image_path.name
    return image_path, expected


def _corpus_image_path(corpus_dir: Path, entry: object) -> Path:
    if not isinstance(entry, dict):
        raise ValueError("manifest entry must be an object")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("manifest entry missing path")
    image_path = Path(raw_path)
    if image_path.is_absolute():
        try:
            image_path.relative_to(corpus_dir.resolve())
        except ValueError as exc:
            raise ValueError("manifest.json contains an image outside the corpus directory") from exc
        return image_path
    return corpus_dir / image_path


def format_eval_summary(summary: EvalSummary) -> str:
    return "\n".join(
        [
            "PartoGuard synthetic evaluation",
            f"Total images: {summary.total}",
            f"Evaluated images: {summary.evaluated}",
            f"Correct zone classifications: {summary.correct_zones}",
            f"Incorrect zone classifications: {summary.incorrect_zones}",
            f"Non-manual zone accuracy: {summary.zone_accuracy:.2%}",
            f"Full-set success rate: {summary.full_set_success_rate:.2%}",
            f"Manual review rate: {summary.manual_review_rate:.2%}",
        ]
    )


def format_corpus_eval_summary(summary: CorpusEvalSummary) -> str:
    lines = [
        "PartoGuard corpus evaluation",
        f"Total images: {summary.total}",
        f"Non-manual outputs: {summary.non_manual}",
        f"Manual-review outputs: {summary.manual_reviews}",
        f"Manual review rate: {summary.manual_review_rate:.2%}",
        f"Blank-template manual review rate: {summary.blank_manual_review_rate:.2%}",
    ]
    if summary.elapsed_seconds > 0.0:
        lines.append(f"Elapsed: {summary.elapsed_seconds:.1f}s ({summary.seconds_per_image:.2f}s/image)")
    if summary.correctness_scored:
        lines.append(
            f"Correctness vs manifest ground truth: "
            f"{summary.correctness_correct}/{summary.correctness_scored} "
            f"({summary.correctness_rate:.2%})"
        )
    lines.append("By category:")
    for category in sorted(summary.by_category):
        counts = summary.by_category[category]
        total = counts["total"]
        manual = counts["manual_review"]
        non_manual = counts["non_manual"]
        manual_rate = manual / total if total else 0.0
        line = (
            f"  - {category}: total={total}, non_manual={non_manual}, "
            f"manual_review={manual} ({manual_rate:.2%})"
        )
        if summary.by_category_correctness and category in summary.by_category_correctness:
            corr = summary.by_category_correctness[category]
            corr_total = corr["total"]
            corr_correct = corr["correct"]
            corr_rate = corr_correct / corr_total if corr_total else 0.0
            line += f"  correct={corr_correct}/{corr_total} ({corr_rate:.2%})"
        lines.append(line)
    return "\n".join(lines)

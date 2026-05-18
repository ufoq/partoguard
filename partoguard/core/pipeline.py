from __future__ import annotations

import secrets
import tempfile
from contextlib import contextmanager
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from partoguard.core.extraction.gemma_adapter import GemmaVerifier, StubVerifier
from partoguard.core.extraction.marks import extract_x_marks
from partoguard.core.imaging.preprocess import preprocess_path
from partoguard.core.reports.generator import generate_json_audit, generate_text_report
from partoguard.core.rules.engine import classify_zone
from partoguard.core.schemas.contracts import ExtractionResult, RuleOutput, TemplateID, ZoneStatus


@dataclass(frozen=True)
class AnalysisResult:
    image_path: Path
    extraction: ExtractionResult
    rule_output: RuleOutput
    text_report: str
    json_audit: dict[str, Any]


@contextmanager
def _materialized_chart_crop(chart_crop: np.ndarray) -> Generator[Path | None]:
    tmp_dir = Path(tempfile.gettempdir())
    neutral_name = f"pg_crop_{secrets.token_hex(8)}.png"
    crop_path = tmp_dir / neutral_name
    try:
        ok = cv2.imwrite(str(crop_path), chart_crop)
        if not ok:
            yield None
            return
        yield crop_path
    finally:
        try:
            crop_path.unlink(missing_ok=True)
        except OSError:
            pass


def analyze_image(image_path: Path, *, verifier: GemmaVerifier | None = None) -> AnalysisResult:
    active_verifier: GemmaVerifier = verifier if verifier is not None else StubVerifier()
    use_gemma_extractor = hasattr(active_verifier, "extract_from_image")

    if use_gemma_extractor:
        if not image_path.exists() or not image_path.is_file():
            extraction = ExtractionResult(
                template_id=TemplateID.UNKNOWN,
                chart_present=False,
                registered=False,
                warnings=["image_unreadable", "manual_review"],
            )
            return _result(image_path, extraction, classify_zone([]))
        # Phase 1: no CV preprocessing, no bounded validator. Raw image -> Gemma -> rules.
        # CV preprocess and the bounded validator are preserved in git history at tag
        # pre-raw-model-phase for phase-2 restore.
        extraction = active_verifier.extract_from_image(image_path)  # pyright: ignore[reportAttributeAccessIssue]
        if "manual_review" in extraction.warnings:
            rule = RuleOutput(
                status=ZoneStatus.MANUAL_REVIEW,
                explanation="Gemma extraction failed; manual review required.",
                confidence=extraction.overall_confidence,
                requires_human_review=True,
            )
        elif len(extraction.points) < 2:
            rule = RuleOutput(
                status=ZoneStatus.MANUAL_REVIEW,
                explanation="Fewer than two cervical-dilation points; cannot assess trajectory.",
                confidence=extraction.overall_confidence,
                requires_human_review=True,
            )
        else:
            rule = classify_zone(extraction.points, uncertainty_cm=0.0)
        return _result(image_path, extraction, rule)

    try:
        preprocessed = preprocess_path(image_path)
    except ValueError as exc:
        extraction = ExtractionResult(
            template_id=TemplateID.UNKNOWN,
            chart_present=False,
            registered=False,
            warnings=[str(exc), "manual_review"],
        )
        rule = classify_zone([])
        return _result(image_path, extraction, rule)

    if preprocessed.chart_crop is None or not preprocessed.registered:
        extraction = ExtractionResult(
            template_id=TemplateID.UNKNOWN,
            chart_present=False,
            registered=False,
            warnings=_unique_warnings([*preprocessed.warnings, "manual_review"]),
        )
        rule = RuleOutput(
            status=ZoneStatus.MANUAL_REVIEW,
            explanation="Chart could not be registered reliably; review manually and escalate per protocol.",
            confidence=0.0,
            requires_human_review=True,
        )
        return _result(image_path, extraction, rule)

    with _materialized_chart_crop(preprocessed.chart_crop) as crop_path:
        if crop_path is None and not isinstance(active_verifier, StubVerifier):
            extraction = ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=_unique_warnings([*preprocessed.warnings, "chart_crop_materialization_failed", "manual_review"]),
            )
        else:
            extraction = extract_x_marks(preprocessed.chart_crop)
            if preprocessed.warnings:
                extraction = extraction.model_copy(update={"warnings": _unique_warnings([*extraction.warnings, *preprocessed.warnings])})
            extraction = active_verifier.verify(extraction, chart_crop_path=crop_path)

    if "manual_review" in extraction.warnings or extraction.overall_confidence < 0.42 or len(extraction.points) < 2:
        rule = RuleOutput(
            status=ZoneStatus.MANUAL_REVIEW,
            explanation="At least two reliable cervical dilation X marks are needed to assess labour progress; review manually and escalate per protocol.",
            confidence=extraction.overall_confidence,
            requires_human_review=True,
        )
    else:
        rule = classify_zone(extraction.points, uncertainty_cm=0.0)
    return _result(image_path, extraction, rule)


def _result(image_path: Path, extraction: ExtractionResult, rule: RuleOutput) -> AnalysisResult:
    metadata = {"input_id": _safe_input_id(image_path), "mode": "console"}
    audit = generate_json_audit(rule, extraction, metadata)
    return AnalysisResult(image_path, extraction, rule, generate_text_report(rule, extraction), audit)


def _safe_input_id(image_path: Path) -> str:
    import hashlib

    return hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]


def _unique_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for warning in warnings:
        if warning not in seen:
            seen.add(warning)
            result.append(warning)
    return result

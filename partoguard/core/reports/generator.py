from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from partoguard.core.schemas.contracts import ExtractionResult, RuleOutput


SAFETY_CAVEAT = "PartoGuard is clinical decision support only; review the paper chart and escalate per facility protocol."


def generate_text_report(rule: RuleOutput, extraction: ExtractionResult) -> str:
    points = ", ".join(f"{p.x_hours:g}h/{p.dilation_cm:g}cm ({p.confidence:.2f})" for p in extraction.points)
    if not points:
        points = "no reliable cervical dilation X marks extracted"
    lines = [
        "PartoGuard console report",
        f"Status: {rule.status.value}",
        f"Confidence: {rule.confidence:.2f}",
        f"Extracted cervical dilation points: {points}",
        f"Assessment: {rule.explanation}",
        SAFETY_CAVEAT,
    ]
    if extraction.warnings:
        lines.append(f"Warnings: {', '.join(extraction.warnings)}")
    return "\n".join(lines)


def generate_json_audit(rule: RuleOutput, extraction: ExtractionResult, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "safety_caveat": SAFETY_CAVEAT,
        "extraction": extraction.model_dump(mode="json"),
        "rule_output": rule.model_dump(mode="json"),
    }


def write_json_audit(path: Path, audit: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2))

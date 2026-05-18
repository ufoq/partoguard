from partoguard.core.reports.generator import SAFETY_CAVEAT, generate_json_audit, generate_text_report
from partoguard.core.schemas.contracts import DilationPoint, ExtractionResult, RuleOutput, TemplateID, ZoneStatus


def _objects():
    extraction = ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=[DilationPoint(x_hours=8.0, dilation_cm=8.0, confidence=0.83)],
        overall_confidence=0.83,
    )
    rule = RuleOutput(
        status=ZoneStatus.ACTION_ZONE,
        triggering_point=extraction.points[0],
        explanation="Point at (8h, 8cm) is on or beyond the action threshold; review and escalate per protocol.",
        confidence=0.83,
        requires_human_review=False,
    )
    return extraction, rule


def test_text_report_contains_status_points_and_safety_caveat():
    extraction, rule = _objects()
    report = generate_text_report(rule, extraction)

    assert "action_zone" in report
    assert "8h/8cm" in report
    assert SAFETY_CAVEAT in report
    assert "treat" not in report.lower()


def test_json_audit_has_expected_sections():
    extraction, rule = _objects()
    audit = generate_json_audit(rule, extraction, {"input_id": "synthetic-input"})

    assert audit["metadata"]["input_id"] == "synthetic-input"
    assert audit["extraction"]["points"][0]["x_hours"] == 8.0
    assert audit["rule_output"]["status"] == "action_zone"
    assert "clinical decision support" in audit["safety_caveat"]

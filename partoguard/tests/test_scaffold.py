"""Smoke tests for the PartoGuard scaffold."""

from pathlib import Path

from partoguard.cli.main import build_parser, main
from partoguard.core.schemas.contracts import (
    AnalysisInput,
    DilationPoint,
    ExtractionResult,
    RuleOutput,
    TemplateID,
    ZoneStatus,
)


def test_cli_no_args_returns_zero():
    assert main([]) == 0


def test_cli_missing_image_returns_one():
    assert main(["analyze", "nonexistent_image.jpg"]) == 1


def test_cli_rejects_two_gemma_verifier_modes(tmp_path: Path):
    image = tmp_path / "blank.png"
    image.write_text("not an image")

    assert main(["analyze", str(image), "--gemma-command", "fake", "--gemma-litert-e2b"]) == 1


def test_cli_version(capsys):
    parser = build_parser()
    try:
        parser.parse_args(["--version"])
    except SystemExit as e:
        assert e.code == 0


def test_analysis_input_defaults():
    inp = AnalysisInput(image_path=Path("test.jpg"))
    assert inp.template_id == TemplateID.MODIFIED_WHO_V1


def test_dilation_point_range():
    pt = DilationPoint(x_hours=2.0, dilation_cm=7.0, confidence=0.9)
    assert 0 <= pt.dilation_cm <= 10


def test_extraction_result_empty():
    result = ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=False,
        registered=False,
    )
    assert result.points == []


def test_rule_output_manual_review():
    out = RuleOutput(status=ZoneStatus.MANUAL_REVIEW)
    assert out.requires_human_review is True
    assert out.framework == "modified_who_partograph"

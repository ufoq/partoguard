from pathlib import Path
import sys

import cv2
import numpy as np

from partoguard.core.extraction.gemma_adapter import LocalGemmaVerifier
from partoguard.core.imaging.synthetic import CANONICAL_SCENARIOS, generate_scenario
from partoguard.core.pipeline import analyze_image


def test_pipeline_classifies_clean_action_fullpage(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[2], tmp_path, include_degraded=False, include_fullpage=True)
    image_path = next(item.path for item in generated if item.variant == "clean_fullpage")

    result = analyze_image(image_path)

    assert result.rule_output.status.value == "action_zone"
    assert "review" in result.text_report.lower()
    assert result.json_audit["rule_output"]["status"] == "action_zone"


def test_pipeline_manual_review_for_blank_image(tmp_path: Path):
    import cv2

    image_path = tmp_path / "blank.png"
    cv2.imwrite(str(image_path), np.ones((100, 100, 3), dtype=np.uint8) * 255)

    result = analyze_image(image_path)

    assert result.rule_output.status.value == "manual_review"
    assert result.rule_output.requires_human_review is True


def test_pipeline_manual_review_when_only_one_point_extracted(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[1], tmp_path, include_degraded=True, include_fullpage=False)
    image_path = next(item.path for item in generated if item.variant == "moderate_crop")

    result = analyze_image(image_path)

    assert result.rule_output.status.value == "manual_review"
    assert result.rule_output.requires_human_review is True


def test_pipeline_manual_review_for_non_partograph_grid(tmp_path: Path):
    image_path = tmp_path / "grid.png"
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    for x in range(0, 800, 40):
        cv2.line(image, (x, 0), (x, 599), (0, 0, 0), 1)
    for y in range(0, 600, 40):
        cv2.line(image, (0, y), (799, y), (0, 0, 0), 1)
    cv2.imwrite(str(image_path), image)

    result = analyze_image(image_path)

    assert result.rule_output.status.value == "manual_review"
    assert result.extraction.template_id.value == "unknown"


def test_pipeline_manual_review_for_adversarial_grid_with_diagonals(tmp_path: Path):
    image_path = tmp_path / "non_partograph_with_diagonals.png"
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    for x in range(0, 800, 40):
        cv2.line(image, (x, 0), (x, 599), (0, 0, 0), 1)
    for y in range(0, 600, 40):
        cv2.line(image, (0, y), (799, y), (0, 0, 0), 1)
    cv2.line(image, (80, 340), (420, 40), (0, 0, 0), 2)
    cv2.line(image, (306, 340), (646, 40), (0, 0, 0), 2)
    for cx, cy in [(80, 340), (306, 340), (646, 40)]:
        cv2.line(image, (cx - 12, cy - 12), (cx + 12, cy + 12), (0, 0, 0), 2)
        cv2.line(image, (cx + 12, cy - 12), (cx - 12, cy + 12), (0, 0, 0), 2)
    cv2.imwrite(str(image_path), image)

    result = analyze_image(image_path)

    assert result.rule_output.status.value == "manual_review"
    assert result.extraction.template_id.value == "unknown"


def test_pipeline_unreadable_existing_file_does_not_leak_path(tmp_path: Path):
    image_path = tmp_path / "secret_patient_name.png"
    image_path.write_text("not an image")

    result = analyze_image(image_path)

    joined_warnings = " ".join(result.extraction.warnings)
    audit_text = str(result.json_audit)
    assert result.rule_output.status.value == "manual_review"
    assert "image_unreadable" in result.extraction.warnings
    assert str(image_path) not in joined_warnings
    assert image_path.name not in audit_text


def test_pipeline_gemma_manual_review_warning_forces_manual_review(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[2], tmp_path, include_degraded=False, include_fullpage=True)
    image_path = next(item.path for item in generated if item.variant == "clean_fullpage")
    verifier = LocalGemmaVerifier(command=[sys.executable, "-c", "print('not json')"])

    result = analyze_image(image_path, verifier=verifier)

    assert result.rule_output.status.value == "manual_review"
    assert "manual_review" in result.extraction.warnings
    assert result.extraction.points == []


def test_pipeline_chart_crop_write_failure_forces_manual_review(tmp_path: Path, monkeypatch):
    generated = generate_scenario(CANONICAL_SCENARIOS[2], tmp_path, include_degraded=False, include_fullpage=True)
    image_path = next(item.path for item in generated if item.variant == "clean_fullpage")
    verifier = LocalGemmaVerifier(command=[sys.executable, "-c", "print('{\"accepted_points\":[]}')"])

    from partoguard.core import pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module.cv2, "imwrite", lambda _path, _img: False)

    result = analyze_image(image_path, verifier=verifier)

    assert result.rule_output.status.value == "manual_review"
    assert "chart_crop_materialization_failed" in result.extraction.warnings
    assert "manual_review" in result.extraction.warnings
    assert result.extraction.points == []


def test_json_audit_does_not_expose_input_path(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[2], tmp_path, include_degraded=False, include_fullpage=True)
    image_path = next(item.path for item in generated if item.variant == "clean_fullpage")

    result = analyze_image(image_path)

    assert "image_path" not in result.json_audit["metadata"]
    assert result.json_audit["metadata"]["input_id"]


def test_pipeline_gemma_extractor_path_bypasses_cv_extraction(tmp_path: Path, monkeypatch):
    from pathlib import Path as _Path
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaE2BVerifier
    from partoguard.core import pipeline as pipeline_module
    from partoguard.core.schemas.contracts import DilationPoint, ExtractionResult, TemplateID

    generated = generate_scenario(CANONICAL_SCENARIOS[0], tmp_path, include_degraded=False, include_fullpage=True)
    image_path = next(item.path for item in generated if item.variant == "clean_fullpage")

    def _exploding_extract(*_args, **_kwargs):
        raise AssertionError("CV extract_x_marks must not run when Gemma extractor is active")

    monkeypatch.setattr(pipeline_module, "extract_x_marks", _exploding_extract)

    fake_result = ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=[
            DilationPoint(x_hours=1.0, dilation_cm=4.0, confidence=0.9, source="gemma_e2b_extracted"),
            DilationPoint(x_hours=2.0, dilation_cm=5.0, confidence=0.9, source="gemma_e2b_extracted"),
            DilationPoint(x_hours=3.0, dilation_cm=6.0, confidence=0.9, source="gemma_e2b_extracted"),
        ],
        overall_confidence=0.9,
        warnings=["gemma_e2b_extracted"],
    )

    class _FakeExtractor(LiteRTGemmaE2BVerifier):
        def extract_from_image(self, chart_crop_path: _Path) -> ExtractionResult:  # noqa: ARG002
            return fake_result

    result = analyze_image(image_path, verifier=_FakeExtractor())

    assert result.rule_output.status.value != "manual_review"
    assert all(p.source == "gemma_e2b_extracted" for p in result.extraction.points)
    assert "gemma_e2b_extracted" in result.extraction.warnings


def test_pipeline_gemma_extractor_manual_review_propagates(tmp_path: Path, monkeypatch):
    from pathlib import Path as _Path
    from partoguard.core.extraction.gemma_adapter import LiteRTGemmaE2BVerifier
    from partoguard.core import pipeline as pipeline_module
    from partoguard.core.schemas.contracts import ExtractionResult, TemplateID

    generated = generate_scenario(CANONICAL_SCENARIOS[0], tmp_path, include_degraded=False, include_fullpage=True)
    image_path = next(item.path for item in generated if item.variant == "clean_fullpage")

    monkeypatch.setattr(pipeline_module, "extract_x_marks", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("CV must not run")))

    class _ManualReviewExtractor(LiteRTGemmaE2BVerifier):
        def extract_from_image(self, chart_crop_path: _Path) -> ExtractionResult:  # noqa: ARG002
            return ExtractionResult(
                template_id=TemplateID.MODIFIED_WHO_V1,
                chart_present=True,
                registered=True,
                points=[],
                overall_confidence=0.0,
                warnings=["gemma_e2b_extract_invalid_json", "manual_review"],
            )

    result = analyze_image(image_path, verifier=_ManualReviewExtractor())

    assert result.rule_output.status.value == "manual_review"
    assert "manual_review" in result.extraction.warnings


def test_local_blank_template_returns_manual_review():
    image_path = Path("/data/input/partographs/obgynkey-fig25-3-who-modified-partograph.png")
    if not image_path.exists():
        return

    result = analyze_image(image_path)

    assert result.rule_output.status.value == "manual_review"
    assert result.rule_output.requires_human_review is True

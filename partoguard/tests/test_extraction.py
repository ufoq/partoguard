from pathlib import Path

import cv2

from partoguard.core.extraction.marks import extract_x_marks
from partoguard.core.imaging.preprocess import preprocess_path
from partoguard.core.imaging.synthetic import CANONICAL_SCENARIOS, draw_chart_crop, generate_scenario


def _point_pairs(points):
    return {(round(point.x_hours, 1), round(point.dilation_cm, 1)) for point in points}


def test_extracts_clean_normal_scenario_points():
    chart = draw_chart_crop(CANONICAL_SCENARIOS[0])
    result = extract_x_marks(chart)

    assert result.registered is True
    assert _point_pairs(result.points) == {(0.0, 4.0), (1.0, 5.0), (2.0, 6.0), (3.0, 7.0), (4.0, 8.0), (5.0, 9.0), (6.0, 10.0)}
    assert result.overall_confidence > 0.42


def test_extracts_clean_alert_scenario_points():
    chart = draw_chart_crop(CANONICAL_SCENARIOS[1])
    result = extract_x_marks(chart)

    assert _point_pairs(result.points) == {(0.0, 4.0), (2.0, 5.0), (4.0, 6.0)}


def test_extracts_action_scenario_from_fullpage(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[2], tmp_path, include_degraded=False, include_fullpage=True)
    fullpage = next(item.path for item in generated if item.variant == "clean_fullpage")
    preprocessed = preprocess_path(fullpage)

    assert preprocessed.chart_crop is not None
    result = extract_x_marks(preprocessed.chart_crop)

    assert _point_pairs(result.points) == {(0.0, 4.0), (4.0, 6.0), (8.0, 8.0)}


def test_blank_chart_returns_no_points(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[0], tmp_path, include_degraded=False, include_fullpage=False)
    image = cv2.imread(str(generated[0].path))
    assert image is not None
    image[:] = 255

    result = extract_x_marks(image)

    assert result.points == []
    assert "no_x_marks_detected" in result.warnings

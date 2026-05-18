from pathlib import Path

import cv2
import numpy as np

from partoguard.core.imaging.preprocess import assess_quality, preprocess_image, preprocess_path
from partoguard.core.imaging.synthetic import CHART_H, CHART_W, CANONICAL_SCENARIOS, draw_chart_crop, generate_scenario


def test_preprocess_accepts_clean_chart_crop():
    image = draw_chart_crop(CANONICAL_SCENARIOS[0])
    result = preprocess_image(image)

    assert result.registered is True
    assert result.template_id == "modified_who_partograph_v1"
    assert result.registration_confidence >= 0.55
    assert result.chart_crop is not None
    assert result.chart_crop.shape == (CHART_H, CHART_W, 3)


def test_preprocess_extracts_chart_from_fullpage(tmp_path: Path):
    generated = generate_scenario(CANONICAL_SCENARIOS[1], tmp_path, include_degraded=False, include_fullpage=True)
    fullpage = next(item.path for item in generated if item.variant == "clean_fullpage")

    result = preprocess_path(fullpage)

    assert result.registered is True
    assert result.template_id == "modified_who_partograph_v1"
    assert result.chart_crop is not None
    assert result.chart_crop.shape == (CHART_H, CHART_W, 3)


def test_quality_gate_flags_tiny_blank_image():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    quality = assess_quality(image)

    assert quality.acceptable is False
    assert "image_too_small" in quality.warnings


def test_preprocess_bad_image_returns_manual_review_warning():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = preprocess_image(image)

    assert result.registered is False
    assert result.chart_crop is None
    assert "image_too_small" in result.warnings


def test_preprocess_rejects_non_partograph_grid():
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    for x in range(0, 800, 40):
        cv2.line(image, (x, 0), (x, 599), (0, 0, 0), 1)
    for y in range(0, 600, 40):
        cv2.line(image, (0, y), (799, y), (0, 0, 0), 1)

    result = preprocess_image(image)

    assert result.registered is False
    assert result.template_id == "unknown"
    assert "unknown_template" in result.warnings


def test_preprocess_rejects_adversarial_grid_with_diagonals_and_x_marks():
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

    result = preprocess_image(image)

    assert result.registered is False
    assert result.template_id == "unknown"
    assert "partograph_labels_not_confirmed" in result.warnings

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np

from partoguard.core.imaging.synthetic import CHART_H, CHART_W


@dataclass(frozen=True)
class ImageQuality:
    acceptable: bool
    blur_score: float
    mean_luma: float
    warnings: list[str]


@dataclass(frozen=True)
class PreprocessResult:
    chart_crop: np.ndarray | None
    registered: bool
    template_id: str
    registration_confidence: float
    quality: ImageQuality
    warnings: list[str]
    source_shape: tuple[int, int]


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image_unreadable")
    return image


def assess_quality(image: np.ndarray) -> ImageQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luma = float(gray.mean())
    warnings: list[str] = []
    if blur_score < 20.0:
        warnings.append("image_blurry")
    if mean_luma < 65.0:
        warnings.append("image_underexposed")
    if mean_luma > 252.0 and blur_score < 10.0:
        warnings.append("image_overexposed")
    if image.shape[0] < 300 or image.shape[1] < 300:
        warnings.append("image_too_small")
    return ImageQuality(not warnings, blur_score, mean_luma, warnings)


def preprocess_image(image: np.ndarray) -> PreprocessResult:
    quality = assess_quality(image)
    h, w = image.shape[:2]
    warnings = list(quality.warnings)

    if "image_too_small" in warnings or "image_overexposed" in warnings or "image_underexposed" in warnings:
        return PreprocessResult(None, False, "unknown", 0.0, quality, [*warnings, "manual_review"], (h, w))

    crop = _crop_synthetic_chart_region(image)
    if crop is None:
        warnings.append("chart_region_not_found")
        return PreprocessResult(None, False, "unknown", 0.0, quality, warnings, (h, w))

    chart_crop = cv2.resize(crop, (CHART_W, CHART_H), interpolation=cv2.INTER_AREA)
    confidence, validation_warnings = validate_modified_who_chart(chart_crop)
    warnings.extend(validation_warnings)
    if confidence < 0.55:
        return PreprocessResult(None, False, "unknown", confidence, quality, [*warnings, "unknown_template", "manual_review"], (h, w))
    return PreprocessResult(chart_crop, True, "modified_who_partograph_v1", confidence, quality, warnings, (h, w))


def preprocess_path(path: Path) -> PreprocessResult:
    return preprocess_image(load_image(path))


def _crop_synthetic_chart_region(image: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    if _looks_like_chart_crop(h, w):
        return image.copy()

    top = int(round(h * 0.2042))
    bottom = int(round(h * 0.7917))
    left = int(round(w * 0.0333))
    right = int(round(w * 0.9667))
    if bottom <= top or right <= left:
        return None

    crop = image[max(0, top):min(h, bottom), max(0, left):min(w, right)]
    if crop.size == 0:
        return None
    return crop


def _looks_like_chart_crop(height: int, width: int) -> bool:
    ratio = width / max(height, 1)
    return 1.0 <= ratio <= 1.6 and height <= 800 and width <= 1100


def validate_modified_who_chart(chart_crop: np.ndarray) -> tuple[float, list[str]]:
    """Validate that a crop resembles the supported modified WHO chart.

    This is deliberately conservative: if the grid and Alert/Action line
    structure are not visible, PartoGuard must fall back to manual review.
    """
    chart = cv2.resize(chart_crop, (CHART_W, CHART_H), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    expected_grid_score = _expected_grid_score(edges)
    generic_grid_score = _generic_grid_score(edges)
    grid_score = max(expected_grid_score, generic_grid_score)
    diagonal_score = max(_expected_diagonal_score(edges), _generic_diagonal_score(edges))
    red_score = _red_line_score(chart)
    label_score = _template_label_score(chart)

    score = 0.45 * grid_score + 0.35 * diagonal_score + 0.20 * red_score
    warnings: list[str] = []
    if grid_score < 0.35:
        warnings.append("chart_grid_not_confirmed")
    if diagonal_score < 0.35 and red_score < 0.35:
        warnings.append("alert_action_lines_not_confirmed")
    if expected_grid_score < 0.32 and label_score < 0.45:
        score = min(score, 0.49)
        warnings.append("partograph_labels_not_confirmed")
    return float(min(1.0, score)), warnings


def _generic_grid_score(edges: np.ndarray) -> float:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=90, minLineLength=80, maxLineGap=8)
    if lines is None:
        return 0.0
    vertical = 0
    horizontal = 0
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 80:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle >= 80:
            vertical += 1
        elif angle <= 10:
            horizontal += 1
    return min(1.0, min(vertical / 8, horizontal / 8))


def _generic_diagonal_score(edges: np.ndarray) -> float:
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=90, maxLineGap=12)
    if lines is None:
        return 0.0
    diagonals: list[tuple[float, float, float]] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 90:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if -65 <= angle <= -25 or 25 <= angle <= 65:
            intercept = y1 - math.tan(math.radians(angle)) * x1
            diagonals.append((length, angle, intercept))
    long_lines = [line for line in diagonals if line[0] >= 120]
    if len(long_lines) < 2:
        return 0.0
    long_lines.sort(reverse=True, key=lambda item: item[0])
    for i, first in enumerate(long_lines):
        for second in long_lines[i + 1:]:
            if abs(first[1] - second[1]) <= 8 and abs(first[2] - second[2]) >= 40:
                return 1.0
    return min(0.6, len(long_lines) / 4)


def _expected_grid_score(edges: np.ndarray) -> float:
    vertical_scores: list[float] = []
    for i in range(13):
        x = int(round(80 + i * ((CHART_W - 120) / 12)))
        band = edges[40:540, max(0, x - 2):min(CHART_W, x + 3)]
        vertical_scores.append(float(np.count_nonzero(band)) / max(band.size, 1))

    horizontal_scores: list[float] = []
    for i in range(11):
        y = int(round(40 + i * ((CHART_H - 100) / 10)))
        band = edges[max(0, y - 2):min(CHART_H, y + 3), 80:760]
        horizontal_scores.append(float(np.count_nonzero(band)) / max(band.size, 1))

    present_vertical = sum(score > 0.12 for score in vertical_scores)
    present_horizontal = sum(score > 0.12 for score in horizontal_scores)
    return min(1.0, (present_vertical / 13 + present_horizontal / 11) / 2)


def _expected_diagonal_score(edges: np.ndarray) -> float:
    alert = _line_edge_score(edges, (80, 340), (420, 40))
    action = _line_edge_score(edges, (306, 340), (646, 40))
    return min(1.0, (alert + action) / 2)


def _line_edge_score(edges: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> float:
    mask = np.zeros(edges.shape, dtype=np.uint8)
    cv2.line(mask, start, end, 255, 5)
    expected = np.count_nonzero(mask)
    if expected == 0:
        return 0.0
    return float(min(1.0, np.count_nonzero((edges > 0) & (mask > 0)) / expected * 2.5))


def _red_line_score(chart: np.ndarray) -> float:
    b, g, r = cv2.split(chart)
    red = ((r > 120) & (r > g * 1.35) & (r > b * 1.35)).astype(np.uint8) * 255
    alert = _line_edge_score(red, (80, 340), (420, 40))
    action = _line_edge_score(red, (306, 340), (646, 40))
    return min(1.0, (alert + action) / 2)


def _template_label_score(chart: np.ndarray) -> float:
    """Score label-like ink near expected Alert/Action and axis-label areas.

    Generic grids can be drawn with plausible diagonals, so template support also
    needs evidence of the actual partograph labeling. This intentionally checks
    only coarse regions and ink density; it is not OCR and remains conservative.
    """
    gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    ink = gray < 180
    regions = [
        (20, 330, 5, 75),      # left cm/dilation axis label area
        (550, 790, 20, 80),    # Alert/Action labels near upper right
        (300, 520, 560, 598),  # Hours label / x-axis label area
    ]
    present = 0
    for x1, x2, y1, y2 in regions:
        region = ink[y1:y2, x1:x2].astype(np.uint8)
        if region.size == 0:
            continue
        components = _text_like_component_count(region)
        if components >= 2:
            present += 1
    return present / len(regions)


def _text_like_component_count(binary_region: np.ndarray) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_region, connectivity=8)
    text_like = 0
    for idx in range(1, count):
        x, y, w, h, area = map(int, stats[idx])
        del x, y
        if area < 4 or area > 220:
            continue
        if w < 2 or h < 4 or w > 45 or h > 24:
            continue
        aspect = w / max(h, 1)
        if 0.15 <= aspect <= 6.0:
            text_like += 1
    return text_like

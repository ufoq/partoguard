from __future__ import annotations

import cv2
import numpy as np

from partoguard.core.imaging.synthetic import CHART_H, CHART_W, CM_MAX, CM_MIN, HOURS_MAX, HOURS_MIN, MARGIN_L, MARGIN_T, PLOT_H, PLOT_W
from partoguard.core.schemas.contracts import DilationPoint, ExtractionResult, TemplateID


def extract_x_marks(chart_crop: np.ndarray, *, min_confidence: float = 0.42) -> ExtractionResult:
    if chart_crop.size == 0:
        return ExtractionResult(
            template_id=TemplateID.UNKNOWN,
            chart_present=False,
            registered=False,
            warnings=["empty_chart_crop", "manual_review"],
        )

    chart = cv2.resize(chart_crop, (CHART_W, CHART_H), interpolation=cv2.INTER_AREA)
    dark_mask = _dark_ink_mask(chart)
    candidates: list[DilationPoint] = []

    for hour in range(int(HOURS_MIN), int(HOURS_MAX) + 1):
        for cm in range(int(CM_MIN), int(CM_MAX) + 1):
            cx = _hours_to_px(float(hour))
            cy = _cm_to_px(float(cm))
            score = _x_score(dark_mask, cx, cy)
            if score >= min_confidence:
                candidates.append(
                    DilationPoint(
                        x_hours=float(hour),
                        dilation_cm=float(cm),
                        bbox=_bbox(cx, cy, 24),
                        confidence=min(0.99, score),
                        source="cv",
                    )
                )

    points = _deduplicate(candidates)
    warnings: list[str] = []
    if not points:
        warnings.append("no_x_marks_detected")
    if len(points) > 20:
        return ExtractionResult(
            template_id=TemplateID.UNKNOWN,
            chart_present=True,
            registered=False,
            points=[],
            overall_confidence=0.0,
            warnings=["implausible_mark_density", "manual_review"],
        )
    if len(points) >= 2 and not _points_are_plausible(points):
        return ExtractionResult(
            template_id=TemplateID.UNKNOWN,
            chart_present=True,
            registered=False,
            points=[],
            overall_confidence=0.0,
            warnings=["implausible_dilation_trajectory", "manual_review"],
        )
    confidence = min((point.confidence for point in points), default=0.0)
    return ExtractionResult(
        template_id=TemplateID.MODIFIED_WHO_V1,
        chart_present=True,
        registered=True,
        points=points,
        overall_confidence=confidence,
        warnings=warnings,
    )


def _dark_ink_mask(chart: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(chart)
    red_mask = (r > 120) & (r > g * 1.4) & (r > b * 1.4)
    cleaned = chart.copy()
    cleaned[red_mask] = (255, 255, 255)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    dark = (gray < 135).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)


def _x_score(mask: np.ndarray, cx: int, cy: int, radius: int = 24) -> float:
    h, w = mask.shape[:2]
    x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
    roi = mask[y1:y2, x1:x2]
    if roi.shape[0] < 20 or roi.shape[1] < 20:
        return 0.0

    yy, xx = np.indices(roi.shape)
    n_y, n_x = roi.shape
    diag_width = max(5, int(min(n_x, n_y) * 0.12))
    main_diag = np.abs((xx / max(n_x - 1, 1)) - (yy / max(n_y - 1, 1))) <= diag_width / max(n_x, n_y)
    anti_diag = np.abs((xx / max(n_x - 1, 1)) + (yy / max(n_y - 1, 1)) - 1.0) <= diag_width / max(n_x, n_y)

    ink = roi > 0
    main_count = int(np.count_nonzero(ink & main_diag))
    anti_count = int(np.count_nonzero(ink & anti_diag))
    if main_count < 10 or anti_count < 10:
        return 0.0

    diagonal_area = int(np.count_nonzero(main_diag | anti_diag))
    total_ink = int(np.count_nonzero(ink))
    if diagonal_area == 0 or total_ink == 0:
        return 0.0

    diagonal_density = (main_count + anti_count) / diagonal_area
    balance = min(main_count, anti_count) / max(main_count, anti_count)
    stroke_strength = min(1.0, min(main_count, anti_count) / 70.0)
    clutter_penalty = min(1.0, 450.0 / max(total_ink, 1))
    return float(min(0.99, (0.45 * stroke_strength + 0.55 * min(1.0, diagonal_density * 3.0)) * balance * clutter_penalty))


def _deduplicate(points: list[DilationPoint]) -> list[DilationPoint]:
    best: dict[tuple[int, int], DilationPoint] = {}
    for point in points:
        key = (int(round(point.x_hours)), int(round(point.dilation_cm)))
        if key not in best or point.confidence > best[key].confidence:
            best[key] = point
    return sorted(best.values(), key=lambda p: (p.x_hours, p.dilation_cm))


def _points_are_plausible(points: list[DilationPoint]) -> bool:
    ordered = sorted(points, key=lambda p: p.x_hours)
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.dilation_cm + 0.5 < prev.dilation_cm:
            return False
    distinct_hours = {round(p.x_hours, 1) for p in points}
    distinct_dilations = {round(p.dilation_cm, 1) for p in points}
    if len(distinct_hours) < 2 or len(distinct_dilations) < 2:
        return False
    return True


def _hours_to_px(hours: float) -> int:
    return MARGIN_L + int((hours - HOURS_MIN) / (HOURS_MAX - HOURS_MIN) * PLOT_W)


def _cm_to_px(cm: float) -> int:
    return MARGIN_T + int((CM_MAX - cm) / (CM_MAX - CM_MIN) * PLOT_H)


def _bbox(cx: int, cy: int, radius: int) -> tuple[int, int, int, int]:
    x = max(0, cx - radius)
    y = max(0, cy - radius)
    w = min(CHART_W - x, radius * 2)
    h = min(CHART_H - y, radius * 2)
    return (x, y, w, h)

"""Synthetic partograph chart generator.

Produces labeled chart-crop and full-page images for the three canonical
teaching scenarios (normal, alert-zone, action-zone) plus configurable
custom scenarios.  Every output carries machine-readable labels.

All generated images are clearly synthetic / demo-only material.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHART_W = 800
CHART_H = 600
MARGIN_L = 80
MARGIN_B = 60
MARGIN_T = 40
MARGIN_R = 40

PLOT_W = CHART_W - MARGIN_L - MARGIN_R
PLOT_H = CHART_H - MARGIN_T - MARGIN_B

HOURS_MIN = 0.0
HOURS_MAX = 12.0
CM_MIN = 0.0
CM_MAX = 10.0


@dataclass
class ScenarioPoint:
    hours: float
    dilation_cm: float


@dataclass
class Scenario:
    name: str
    points: list[ScenarioPoint]
    expected_zone: str
    description: str = ""


@dataclass
class GeneratedImage:
    path: Path
    scenario: Scenario
    label: dict[str, Any]
    variant: str  # "clean_crop", "degraded_crop", "clean_fullpage", "degraded_fullpage"


CANONICAL_SCENARIOS = [
    Scenario(
        name="normal_progress",
        points=[
            ScenarioPoint(0.0, 4.0),
            ScenarioPoint(1.0, 5.0),
            ScenarioPoint(2.0, 6.0),
            ScenarioPoint(3.0, 7.0),
            ScenarioPoint(4.0, 8.0),
            ScenarioPoint(5.0, 9.0),
            ScenarioPoint(6.0, 10.0),
        ],
        expected_zone="normal",
        description="4cm at 0h to 10cm at 6h — stays left of Alert line",
    ),
    Scenario(
        name="alert_zone",
        points=[
            ScenarioPoint(0.0, 4.0),
            ScenarioPoint(2.0, 5.0),
            ScenarioPoint(4.0, 6.0),
        ],
        expected_zone="alert_zone",
        description="4cm at 0h, 6cm at 4h — crosses Alert line, stays left of Action",
    ),
    Scenario(
        name="action_zone",
        points=[
            ScenarioPoint(0.0, 4.0),
            ScenarioPoint(4.0, 6.0),
            ScenarioPoint(8.0, 8.0),
        ],
        expected_zone="action_zone",
        description="4cm at 0h, 6cm at 4h, 8cm at 8h — reaches Action line",
    ),
]


def _hours_to_px(h: float) -> int:
    return MARGIN_L + int((h - HOURS_MIN) / (HOURS_MAX - HOURS_MIN) * PLOT_W)


def _cm_to_px(cm: float) -> int:
    return MARGIN_T + int((CM_MAX - cm) / (CM_MAX - CM_MIN) * PLOT_H)


def _px_to_hours(px: int) -> float:
    return HOURS_MIN + (px - MARGIN_L) / PLOT_W * (HOURS_MAX - HOURS_MIN)


def _px_to_cm(px: int) -> float:
    return CM_MAX - (px - MARGIN_T) / PLOT_H * (CM_MAX - CM_MIN)


def draw_chart_crop(
    scenario: Scenario,
    *,
    draw_alert: bool = True,
    draw_action: bool = True,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Render a clean chart-crop image for a scenario. Returns BGR numpy array."""
    img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * 255

    grid_color = (200, 200, 200)
    for h_tick in range(int(HOURS_MAX) + 1):
        x = _hours_to_px(float(h_tick))
        cv2.line(img, (x, MARGIN_T), (x, CHART_H - MARGIN_B), grid_color, 1)
    for cm_tick in range(int(CM_MAX) + 1):
        y = _cm_to_px(float(cm_tick))
        cv2.line(img, (MARGIN_L, y), (CHART_W - MARGIN_R, y), grid_color, 1)

    axis_color = (0, 0, 0)
    cv2.line(img, (MARGIN_L, MARGIN_T), (MARGIN_L, CHART_H - MARGIN_B), axis_color, 2)
    cv2.line(img, (MARGIN_L, CHART_H - MARGIN_B), (CHART_W - MARGIN_R, CHART_H - MARGIN_B), axis_color, 2)

    for h_tick in range(int(HOURS_MAX) + 1):
        x = _hours_to_px(float(h_tick))
        cv2.putText(img, str(h_tick), (x - 5, CHART_H - MARGIN_B + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)
    for cm_tick in range(int(CM_MAX) + 1):
        y = _cm_to_px(float(cm_tick))
        cv2.putText(img, str(cm_tick), (MARGIN_L - 25, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)

    cv2.putText(img, "Hours", (CHART_W // 2 - 20, CHART_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)
    cv2.putText(img, "cm", (10, CHART_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)

    if draw_alert:
        alert_start = (_hours_to_px(0.0), _cm_to_px(4.0))
        alert_end = (_hours_to_px(6.0), _cm_to_px(10.0))
        cv2.line(img, alert_start, alert_end, (0, 0, 200), 2, cv2.LINE_AA)
        cv2.putText(img, "Alert", (alert_end[0] + 5, alert_end[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    if draw_action:
        action_start = (_hours_to_px(4.0), _cm_to_px(4.0))
        action_end = (_hours_to_px(10.0), _cm_to_px(10.0))
        cv2.line(img, action_start, action_end, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "Action", (action_end[0] + 5, action_end[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    _draw_x_marks(img, scenario.points, rng=rng)

    cv2.putText(img, "SYNTHETIC - NOT FOR CLINICAL USE", (MARGIN_L + 10, MARGIN_T - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)

    return img


def _draw_x_marks(
    img: np.ndarray,
    points: list[ScenarioPoint],
    *,
    rng: random.Random | None = None,
) -> None:
    """Draw handwriting-style X marks at each scenario point."""
    if rng is None:
        rng = random.Random(42)

    ink_color = (80, 40, 10)  # dark blue-black ink
    for pt in points:
        cx = _hours_to_px(pt.hours)
        cy = _cm_to_px(pt.dilation_cm)
        sz = rng.randint(8, 14)
        jx = rng.randint(-3, 3)
        jy = rng.randint(-3, 3)
        thickness = rng.randint(2, 3)
        cv2.line(img, (cx - sz + jx, cy - sz + jy), (cx + sz + jx, cy + sz + jy), ink_color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx + sz + jx, cy - sz + jy), (cx - sz + jx, cy + sz + jy), ink_color, thickness, cv2.LINE_AA)


def _make_fullpage(chart_crop: np.ndarray, rng: random.Random) -> np.ndarray:
    """Embed a chart crop into a full-page partograph layout."""
    page_h, page_w = 1200, 900
    page = np.ones((page_h, page_w, 3), dtype=np.uint8) * 245

    header_h = 120
    cv2.putText(page, "PARTOGRAPH", (page_w // 2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(page, "Name: ____________  Date: __/__/__", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(page, "Gravida: ___  Para: ___  Hospital No: _______", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.line(page, (20, header_h), (page_w - 20, header_h), (0, 0, 0), 1)

    fhr_top = header_h + 10
    fhr_h = 100
    cv2.putText(page, "Fetal Heart Rate", (30, fhr_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.rectangle(page, (25, fhr_top + 20), (page_w - 25, fhr_top + fhr_h), (180, 180, 180), 1)

    chart_top = fhr_top + fhr_h + 15
    chart_region_h = page_h - chart_top - 250
    chart_region_w = page_w - 60

    resized = cv2.resize(chart_crop, (chart_region_w, chart_region_h))
    page[chart_top:chart_top + chart_region_h, 30:30 + chart_region_w] = resized

    bottom_start = chart_top + chart_region_h + 10
    sections = ["Contractions per 10 min", "Oxytocin", "Drugs", "Pulse", "BP", "Temp", "Urine"]
    row_h = 25
    for i, label in enumerate(sections):
        y = bottom_start + i * row_h
        cv2.putText(page, label, (30, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.rectangle(page, (200, y), (page_w - 25, y + row_h - 2), (200, 200, 200), 1)

    cv2.putText(page, "SYNTHETIC - NOT FOR CLINICAL USE", (page_w // 2 - 150, page_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    return page


def apply_degradation(img: np.ndarray, rng: random.Random, level: str = "moderate") -> np.ndarray:
    """Apply camera-like degradation to a synthetic image."""
    h, w = img.shape[:2]
    result = img.copy()

    if level in ("moderate", "heavy"):
        angle = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(240, 240, 240))

    if level in ("moderate", "heavy"):
        ksize = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    if level in ("moderate", "heavy"):
        pts1 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        d = rng.randint(10, 30) if level == "moderate" else rng.randint(20, 50)
        pts2 = np.array([
            [rng.randint(0, d), rng.randint(0, d)],
            [w - rng.randint(0, d), rng.randint(0, d)],
            [rng.randint(0, d), h - rng.randint(0, d)],
            [w - rng.randint(0, d), h - rng.randint(0, d)],
        ], dtype=np.float32)
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(result, M_persp, (w, h), borderValue=(235, 235, 235))

    if level == "heavy":
        brightness = rng.randint(-30, -10)
        result = np.clip(result.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    if level == "heavy":
        noise = np.random.RandomState(rng.randint(0, 2**31)).randint(0, 15, result.shape, dtype=np.uint8)
        result = cv2.add(result, noise)

    if level == "heavy":
        shadow_x = rng.randint(0, w // 2)
        shadow_w = rng.randint(w // 4, w // 2)
        overlay = result.copy()
        cv2.rectangle(overlay, (shadow_x, 0), (shadow_x + shadow_w, h), (180, 180, 180), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    return result


def _build_label(scenario: Scenario, variant: str, path: Path) -> dict[str, Any]:
    return {
        "file": str(path),
        "scenario": scenario.name,
        "expected_zone": scenario.expected_zone,
        "variant": variant,
        "synthetic": True,
        "clinical_use": False,
        "points": [
            {"hours": p.hours, "dilation_cm": p.dilation_cm}
            for p in scenario.points
        ],
        "description": scenario.description,
    }


def generate_scenario(
    scenario: Scenario,
    output_dir: Path,
    *,
    seed: int = 42,
    include_degraded: bool = True,
    include_fullpage: bool = True,
) -> list[GeneratedImage]:
    """Generate all variants for a single scenario."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    results: list[GeneratedImage] = []

    chart = draw_chart_crop(scenario, rng=rng)
    clean_crop_path = output_dir / f"{scenario.name}_clean_crop.png"
    cv2.imwrite(str(clean_crop_path), chart)
    label = _build_label(scenario, "clean_crop", clean_crop_path)
    results.append(GeneratedImage(clean_crop_path, scenario, label, "clean_crop"))

    if include_degraded:
        for level in ("moderate", "heavy"):
            degraded = apply_degradation(chart, rng, level=level)
            deg_path = output_dir / f"{scenario.name}_{level}_crop.png"
            cv2.imwrite(str(deg_path), degraded)
            lbl = _build_label(scenario, f"{level}_crop", deg_path)
            results.append(GeneratedImage(deg_path, scenario, lbl, f"{level}_crop"))

    if include_fullpage:
        fullpage = _make_fullpage(chart, rng)
        fp_path = output_dir / f"{scenario.name}_clean_fullpage.png"
        cv2.imwrite(str(fp_path), fullpage)
        lbl = _build_label(scenario, "clean_fullpage", fp_path)
        results.append(GeneratedImage(fp_path, scenario, lbl, "clean_fullpage"))

        if include_degraded:
            for level in ("moderate", "heavy"):
                degraded_fp = apply_degradation(fullpage, rng, level=level)
                dfp_path = output_dir / f"{scenario.name}_{level}_fullpage.png"
                cv2.imwrite(str(dfp_path), degraded_fp)
                lbl = _build_label(scenario, f"{level}_fullpage", dfp_path)
                results.append(GeneratedImage(dfp_path, scenario, lbl, f"{level}_fullpage"))

    return results


def generate_all(
    output_dir: Path,
    *,
    scenarios: list[Scenario] | None = None,
    seed: int = 42,
) -> list[GeneratedImage]:
    """Generate images for all canonical scenarios. Returns list of generated items."""
    if scenarios is None:
        scenarios = CANONICAL_SCENARIOS

    all_results: list[GeneratedImage] = []
    for i, sc in enumerate(scenarios):
        results = generate_scenario(sc, output_dir, seed=seed + i)
        all_results.extend(results)

    labels_path = output_dir / "labels.json"
    labels = [r.label for r in all_results]
    labels_path.write_text(json.dumps(labels, indent=2))

    return all_results

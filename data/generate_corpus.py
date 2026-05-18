#!/usr/bin/env python3
"""Mass synthetic partograph corpus generator.

Produces 300+ images across multiple states:
- blank: unfilled templates (pristine, aged, photocopied, faxed)
- partial: partially filled (1-3 marks, mid-labour)
- filled: fully filled (complete trajectories, 6-10+ marks)
- degraded: camera-phone capture artifacts
- obstructed: partially covered, stained, folded

All images are SYNTHETIC — NOT FOR CLINICAL USE.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# --- Constants (matching existing synthetic.py) ---
CHART_W = 800
CHART_H = 600
MARGIN_L = 80
MARGIN_B = 60
MARGIN_T = 40
MARGIN_R = 40
PLOT_W = CHART_W - MARGIN_L - MARGIN_R
PLOT_H = CHART_H - MARGIN_T - MARGIN_B
HOURS_MAX = 12.0
CM_MAX = 10.0

PAGE_W = 900
PAGE_H = 1200


def _hours_to_px(h: float) -> int:
    return MARGIN_L + int(h / HOURS_MAX * PLOT_W)


def _cm_to_px(cm: float) -> int:
    return MARGIN_T + int((CM_MAX - cm) / CM_MAX * PLOT_H)


# --- Labour curve generators ---

def generate_normal_curve(rng: random.Random, n_points: int = 7, **_: object) -> list[tuple[float, float]]:
    """Normal labour: ~1cm/hr from 4cm, reaching 10cm by 6h."""
    start_cm = rng.uniform(3.5, 4.5)
    rate = rng.uniform(0.8, 1.2)  # cm/hr
    points = []
    for i in range(n_points):
        h = i * rng.uniform(0.8, 1.2)
        cm = min(10.0, start_cm + rate * h + rng.uniform(-0.3, 0.3))
        points.append((h, cm))
        if cm >= 10.0:
            break
    return points


def generate_slow_curve(rng: random.Random, n_points: int = 8, **_: object) -> list[tuple[float, float]]:
    """Slow/prolonged labour: crosses alert line."""
    start_cm = rng.uniform(3.5, 4.5)
    rate = rng.uniform(0.3, 0.6)
    points = []
    for i in range(n_points):
        h = i * rng.uniform(1.0, 1.5)
        cm = min(10.0, start_cm + rate * h + rng.uniform(-0.2, 0.2))
        points.append((h, cm))
    return points


def generate_arrested_curve(rng: random.Random, n_points: int = 9, **_: object) -> list[tuple[float, float]]:
    """Arrested labour: crosses action line, plateaus."""
    start_cm = rng.uniform(3.5, 4.5)
    points = []
    for i in range(n_points):
        h = i * rng.uniform(1.0, 1.8)
        if i < 3:
            cm = start_cm + 0.5 * i + rng.uniform(-0.2, 0.2)
        else:
            cm = start_cm + 1.5 + rng.uniform(-0.3, 0.3)  # plateau
        points.append((h, min(10.0, cm)))
    return points


def generate_rapid_curve(rng: random.Random, n_points: int = 5, **_: object) -> list[tuple[float, float]]:
    """Precipitous labour: very fast dilation."""
    start_cm = rng.uniform(4.0, 6.0)
    rate = rng.uniform(2.0, 3.5)
    points = []
    for i in range(n_points):
        h = i * rng.uniform(0.5, 1.0)
        cm = min(10.0, start_cm + rate * h + rng.uniform(-0.2, 0.2))
        points.append((h, cm))
        if cm >= 10.0:
            break
    return points


CURVE_GENERATORS = {
    "normal": generate_normal_curve,
    "slow_prolonged": generate_slow_curve,
    "arrested": generate_arrested_curve,
    "rapid_precipitous": generate_rapid_curve,
}


# --- Drawing functions ---

def draw_grid(img: np.ndarray, rng: random.Random, style: str = "clean") -> None:
    """Draw the cervicograph grid."""
    if style == "faded":
        grid_color = (220, 220, 220)
    elif style == "bold":
        grid_color = (170, 170, 170)
    else:
        grid_color = (200, 200, 200)

    thickness = 1 if style != "bold" else 2

    for h_tick in range(int(HOURS_MAX) + 1):
        x = _hours_to_px(float(h_tick))
        cv2.line(img, (x, MARGIN_T), (x, CHART_H - MARGIN_B), grid_color, thickness)
    for cm_tick in range(int(CM_MAX) + 1):
        y = _cm_to_px(float(cm_tick))
        cv2.line(img, (MARGIN_L, y), (CHART_W - MARGIN_R, y), grid_color, thickness)

    # Axes
    cv2.line(img, (MARGIN_L, MARGIN_T), (MARGIN_L, CHART_H - MARGIN_B), (0, 0, 0), 2)
    cv2.line(img, (MARGIN_L, CHART_H - MARGIN_B), (CHART_W - MARGIN_R, CHART_H - MARGIN_B), (0, 0, 0), 2)

    # Labels
    for h_tick in range(int(HOURS_MAX) + 1):
        x = _hours_to_px(float(h_tick))
        cv2.putText(img, str(h_tick), (x - 5, CHART_H - MARGIN_B + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for cm_tick in range(int(CM_MAX) + 1):
        y = _cm_to_px(float(cm_tick))
        cv2.putText(img, str(cm_tick), (MARGIN_L - 25, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def draw_alert_action_lines(img: np.ndarray, rng: random.Random, style: str = "normal") -> None:
    """Draw the diagonal alert and action lines."""
    if style == "faded":
        alert_color = (180, 180, 220)
        action_color = (180, 180, 230)
    else:
        alert_color = (0, 0, 200)
        action_color = (0, 0, 255)

    thickness = 2 if style != "bold" else 3

    # Alert: from (0h, 4cm) to (6h, 10cm)
    cv2.line(img, (_hours_to_px(0), _cm_to_px(4)), (_hours_to_px(6), _cm_to_px(10)), alert_color, thickness, cv2.LINE_AA)
    # Action: from (4h, 4cm) to (10h, 10cm)
    cv2.line(img, (_hours_to_px(4), _cm_to_px(4)), (_hours_to_px(10), _cm_to_px(10)), action_color, thickness, cv2.LINE_AA)

    cv2.putText(img, "Alert", (_hours_to_px(6) + 5, _cm_to_px(10) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1)
    cv2.putText(img, "Action", (_hours_to_px(10) + 5, _cm_to_px(10) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, action_color, 1)


def draw_x_mark(img: np.ndarray, cx: int, cy: int, rng: random.Random, pen_style: str = "ballpoint") -> None:
    """Draw a single X mark with realistic variation."""
    if pen_style == "ballpoint":
        color = (rng.randint(60, 100), rng.randint(20, 50), rng.randint(0, 20))
        sz = rng.randint(7, 12)
        thickness = rng.randint(2, 3)
    elif pen_style == "felt_tip":
        color = (rng.randint(0, 30), rng.randint(0, 30), rng.randint(0, 30))
        sz = rng.randint(10, 16)
        thickness = rng.randint(3, 5)
    elif pen_style == "pencil":
        gray = rng.randint(80, 140)
        color = (gray, gray, gray)
        sz = rng.randint(6, 10)
        thickness = rng.randint(1, 2)
    else:  # shaky
        color = (rng.randint(50, 90), rng.randint(20, 50), rng.randint(0, 20))
        sz = rng.randint(9, 15)
        thickness = rng.randint(2, 4)

    jx = rng.randint(-3, 3)
    jy = rng.randint(-3, 3)

    # Add slight curve to lines for realism
    if pen_style == "shaky":
        # Draw as polyline with jitter
        n_seg = 4
        for stroke in range(2):
            pts = []
            if stroke == 0:
                sx, sy = cx - sz + jx, cy - sz + jy
                ex, ey = cx + sz + jx, cy + sz + jy
            else:
                sx, sy = cx + sz + jx, cy - sz + jy
                ex, ey = cx - sz + jx, cy + sz + jy
            for t in range(n_seg + 1):
                frac = t / n_seg
                px = int(sx + (ex - sx) * frac + rng.randint(-2, 2))
                py = int(sy + (ey - sy) * frac + rng.randint(-2, 2))
                pts.append((px, py))
            for k in range(len(pts) - 1):
                cv2.line(img, pts[k], pts[k + 1], color, thickness, cv2.LINE_AA)
    else:
        cv2.line(img, (cx - sz + jx, cy - sz + jy), (cx + sz + jx, cy + sz + jy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx + sz + jx, cy - sz + jy), (cx - sz + jx, cy + sz + jy), color, thickness, cv2.LINE_AA)


def draw_fhr_trace(img: np.ndarray, region_y: int, region_h: int, rng: random.Random, n_hours: int = 8) -> None:
    """Draw a realistic fetal heart rate trace (squiggly line 110-160 bpm range)."""
    baseline = rng.randint(130, 145)
    x_start = MARGIN_L
    x_end = CHART_W - MARGIN_R
    points = []
    for i in range(200):
        x = x_start + int(i / 200 * (x_end - x_start))
        # Baseline variability
        bpm = baseline + rng.uniform(-8, 8) + 5 * math.sin(i * 0.1)
        # Occasional deceleration
        if rng.random() < 0.02:
            bpm -= rng.uniform(15, 30)
        # Map bpm to y (110-180 range to region)
        y = region_y + int((180 - bpm) / 70 * region_h)
        y = max(region_y, min(region_y + region_h, y))
        points.append((x, y))

    color = (rng.randint(0, 50), rng.randint(0, 80), rng.randint(0, 50))
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], color, 1, cv2.LINE_AA)


def draw_contraction_bars(img: np.ndarray, region_y: int, region_h: int, rng: random.Random, n_hours: int = 8) -> None:
    """Draw contraction frequency bars (shaded rectangles)."""
    bar_w = int(PLOT_W / 12)
    for h in range(n_hours):
        x = _hours_to_px(float(h))
        n_contractions = rng.randint(1, 5)
        fill_h = int(n_contractions / 5 * region_h * 0.8)
        shade = rng.randint(40, 120)
        if rng.random() < 0.5:
            # Filled rectangle
            cv2.rectangle(img, (x + 2, region_y + region_h - fill_h),
                          (x + bar_w - 2, region_y + region_h), (shade, shade, shade), -1)
        else:
            # Hatched
            cv2.rectangle(img, (x + 2, region_y + region_h - fill_h),
                          (x + bar_w - 2, region_y + region_h), (shade, shade, shade), 2)


# --- Paper/capture effects ---

def apply_paper_aging(img: np.ndarray, rng: random.Random, level: str = "light") -> np.ndarray:
    """Simulate aged paper: yellowing, foxing spots."""
    result = img.copy()
    h, w = result.shape[:2]

    # Yellowing
    if level in ("light", "heavy"):
        yellow_strength = 15 if level == "light" else 35
        overlay = np.zeros_like(result)
        overlay[:, :, 0] = 0  # no blue tint
        overlay[:, :, 1] = yellow_strength // 2
        overlay[:, :, 2] = yellow_strength
        result = cv2.add(result, overlay)

    # Foxing spots
    if level == "heavy":
        n_spots = rng.randint(10, 40)
        for _ in range(n_spots):
            cx = rng.randint(0, w)
            cy = rng.randint(0, h)
            radius = rng.randint(2, 8)
            color = (rng.randint(160, 200), rng.randint(150, 180), rng.randint(130, 170))
            cv2.circle(result, (cx, cy), radius, color, -1)
            result = cv2.GaussianBlur(result, (3, 3), 0)

    return result


def apply_photocopy_effect(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate photocopied look: high contrast, streaks, slight skew."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    result = cv2.convertScaleAbs(result, alpha=1.4, beta=-30)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    h, w = result.shape[:2]
    # Add horizontal streaks
    n_streaks = rng.randint(2, 6)
    for _ in range(n_streaks):
        y = rng.randint(0, h)
        thickness = rng.randint(1, 3)
        gray = rng.randint(180, 220)
        cv2.line(result, (0, y), (w, y), (gray, gray, gray), thickness)

    return result


def apply_phone_capture(img: np.ndarray, rng: random.Random, severity: str = "mild") -> np.ndarray:
    """Simulate phone camera capture: perspective, blur, uneven lighting."""
    h, w = img.shape[:2]
    result = img.copy()

    # Perspective warp
    d = rng.randint(10, 25) if severity == "mild" else rng.randint(25, 60)
    pts1 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    pts2 = np.array([
        [rng.randint(0, d), rng.randint(0, d)],
        [w - rng.randint(0, d), rng.randint(0, d)],
        [rng.randint(0, d), h - rng.randint(0, d)],
        [w - rng.randint(0, d), h - rng.randint(0, d)],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(result, M, (w, h), borderValue=(200, 200, 200))

    # Slight rotation
    angle = rng.uniform(-4, 4) if severity == "mild" else rng.uniform(-8, 8)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    result = cv2.warpAffine(result, M_rot, (w, h), borderValue=(200, 200, 200))

    # Uneven lighting (vignette or flash hotspot)
    if rng.random() < 0.5:
        # Flash hotspot
        cx, cy = rng.randint(w // 4, 3 * w // 4), rng.randint(h // 4, 3 * h // 4)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        max_dist = max(w, h) * 0.4
        hotspot = np.clip(1.0 - dist / max_dist, 0, 1)
        hotspot = (hotspot * 40).astype(np.uint8)
        hotspot_3ch = np.stack([hotspot] * 3, axis=-1)
        result = cv2.add(result, hotspot_3ch)
    else:
        # Vignette
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - w // 2) ** 2 + (Y - h // 2) ** 2).astype(np.float32)
        max_dist = math.sqrt((w // 2) ** 2 + (h // 2) ** 2)
        vignette = np.clip(dist / max_dist, 0, 1)
        vignette = (vignette * 50).astype(np.uint8)
        vignette_3ch = np.stack([vignette] * 3, axis=-1)
        result = cv2.subtract(result, vignette_3ch)

    # Slight blur
    if severity != "mild" or rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    return result


def apply_obstruction(img: np.ndarray, rng: random.Random, kind: str = "finger") -> np.ndarray:
    """Add obstruction artifacts: finger, coffee stain, fold, tape."""
    h, w = img.shape[:2]
    result = img.copy()

    if kind == "finger":
        # Simulate a finger/thumb at edge
        side = rng.choice(["left", "right", "bottom"])
        if side == "left":
            cx, cy = rng.randint(-30, 30), rng.randint(h // 3, 2 * h // 3)
            axes = (rng.randint(40, 70), rng.randint(80, 150))
        elif side == "right":
            cx, cy = w - rng.randint(-30, 30), rng.randint(h // 3, 2 * h // 3)
            axes = (rng.randint(40, 70), rng.randint(80, 150))
        else:
            cx, cy = rng.randint(w // 3, 2 * w // 3), h - rng.randint(-20, 20)
            axes = (rng.randint(60, 120), rng.randint(40, 70))
        skin_color = (rng.randint(140, 200), rng.randint(120, 170), rng.randint(100, 160))
        cv2.ellipse(result, (cx, cy), axes, rng.uniform(-20, 20), 0, 360, skin_color, -1)
        # Blur the finger region slightly
        result = cv2.GaussianBlur(result, (5, 5), 0)

    elif kind == "coffee_stain":
        cx = rng.randint(w // 4, 3 * w // 4)
        cy = rng.randint(h // 4, 3 * h // 4)
        radius = rng.randint(30, 80)
        # Ring stain
        stain_color = (rng.randint(100, 140), rng.randint(120, 160), rng.randint(140, 190))
        cv2.circle(result, (cx, cy), radius, stain_color, rng.randint(3, 8))
        # Light fill
        overlay = result.copy()
        cv2.circle(overlay, (cx, cy), radius - 5, stain_color, -1)
        result = cv2.addWeighted(result, 0.85, overlay, 0.15, 0)

    elif kind == "fold":
        # Diagonal fold line with shadow
        if rng.random() < 0.5:
            pt1 = (0, rng.randint(h // 3, 2 * h // 3))
            pt2 = (w, rng.randint(h // 3, 2 * h // 3))
        else:
            pt1 = (rng.randint(w // 3, 2 * w // 3), 0)
            pt2 = (rng.randint(w // 3, 2 * w // 3), h)
        cv2.line(result, pt1, pt2, (160, 160, 160), rng.randint(2, 5))
        # Shadow on one side
        offset = rng.randint(3, 8)
        cv2.line(result, (pt1[0] + offset, pt1[1] + offset),
                 (pt2[0] + offset, pt2[1] + offset), (190, 190, 190), rng.randint(4, 10))

    elif kind == "tape":
        # Transparent tape strip
        tx = rng.randint(0, w - 100)
        ty = rng.randint(0, h // 4)
        tw = rng.randint(80, 200)
        th = rng.randint(20, 40)
        overlay = result.copy()
        cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (230, 235, 240), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    return result


# --- Full-page generator ---

def make_fullpage(chart_crop: np.ndarray, rng: random.Random, fill_extras: bool = True) -> np.ndarray:
    """Embed chart crop in full-page partograph with optional FHR/contraction fills."""
    page = np.ones((PAGE_H, PAGE_W, 3), dtype=np.uint8) * rng.randint(240, 255)

    # Header
    cv2.putText(page, "PARTOGRAPH", (PAGE_W // 2 - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(page, "Name: ____________  Date: __/__/__", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(page, "Gravida: ___  Para: ___  Hospital No: _______", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.line(page, (20, 120), (PAGE_W - 20, 120), (0, 0, 0), 1)

    # FHR section
    fhr_top = 130
    fhr_h = 100
    cv2.putText(page, "Fetal Heart Rate", (30, fhr_top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.rectangle(page, (25, fhr_top + 20), (PAGE_W - 25, fhr_top + fhr_h), (180, 180, 180), 1)
    if fill_extras and rng.random() < 0.7:
        draw_fhr_trace(page, fhr_top + 25, fhr_h - 30, rng)

    # Chart region
    chart_top = fhr_top + fhr_h + 15
    chart_h = PAGE_H - chart_top - 250
    chart_w = PAGE_W - 60
    resized = cv2.resize(chart_crop, (chart_w, chart_h))
    page[chart_top:chart_top + chart_h, 30:30 + chart_w] = resized

    # Bottom sections
    bottom_start = chart_top + chart_h + 10
    sections = ["Contractions per 10 min", "Oxytocin", "Drugs", "Pulse", "BP", "Temp", "Urine"]
    row_h = 25
    for i, label in enumerate(sections):
        y = bottom_start + i * row_h
        cv2.putText(page, label, (30, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.rectangle(page, (200, y), (PAGE_W - 25, y + row_h - 2), (200, 200, 200), 1)

    # Contraction bars
    if fill_extras and rng.random() < 0.5:
        contr_y = bottom_start
        draw_contraction_bars(page, contr_y, row_h - 2, rng)

    cv2.putText(page, "SYNTHETIC - NOT FOR CLINICAL USE", (PAGE_W // 2 - 150, PAGE_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    return page


# --- Main generation pipeline ---

@dataclass
class CorpusImage:
    path: str
    category: str  # blank, partial, filled, degraded, obstructed
    subcategory: str
    curve_type: str
    n_marks: int
    degradation: str
    obstruction: str
    is_fullpage: bool
    pen_style: str
    paper_style: str
    seed: int


def generate_corpus(output_dir: Path, target_count: int = 350, seed: int = 12345) -> list[CorpusImage]:
    """Generate the full corpus."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    manifest: list[CorpusImage] = []
    idx = 0

    pen_styles = ["ballpoint", "felt_tip", "pencil", "shaky"]
    paper_styles = ["clean", "aged_light", "aged_heavy", "photocopy"]
    curve_types = list(CURVE_GENERATORS.keys())
    obstruction_types = ["finger", "coffee_stain", "fold", "tape"]
    capture_severities = ["mild", "harsh"]

    # --- BLANK templates (50 images) ---
    blank_dir = output_dir / "blank"
    blank_dir.mkdir(exist_ok=True)
    for i in range(50):
        img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * rng.randint(245, 255)
        grid_style = rng.choice(["clean", "faded", "bold"])
        draw_grid(img, rng, style=grid_style)
        draw_alert_action_lines(img, rng, style=rng.choice(["normal", "faded"]))

        paper = rng.choice(paper_styles)
        if paper == "aged_light":
            img = apply_paper_aging(img, rng, "light")
        elif paper == "aged_heavy":
            img = apply_paper_aging(img, rng, "heavy")
        elif paper == "photocopy":
            img = apply_photocopy_effect(img, rng)

        is_fp = rng.random() < 0.4
        if is_fp:
            img = make_fullpage(img, rng, fill_extras=False)

        fname = f"blank_{idx:04d}.png"
        cv2.imwrite(str(blank_dir / fname), img)
        manifest.append(CorpusImage(
            path=f"blank/{fname}", category="blank", subcategory=paper,
            curve_type="none", n_marks=0, degradation="none",
            obstruction="none", is_fullpage=is_fp, pen_style="none",
            paper_style=paper, seed=seed + idx
        ))
        idx += 1

    # --- PARTIAL (1-3 marks, 60 images) ---
    partial_dir = output_dir / "partial"
    partial_dir.mkdir(exist_ok=True)
    for i in range(60):
        img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * rng.randint(245, 255)
        draw_grid(img, rng)
        draw_alert_action_lines(img, rng)

        curve_type = rng.choice(curve_types)
        points = CURVE_GENERATORS[curve_type](rng)
        n_marks = rng.randint(1, 3)
        points = points[:n_marks]

        pen = rng.choice(pen_styles)
        for h, cm in points:
            draw_x_mark(img, _hours_to_px(h), _cm_to_px(cm), rng, pen_style=pen)

        paper = rng.choice(paper_styles[:2])  # mostly clean/light aged
        if paper == "aged_light":
            img = apply_paper_aging(img, rng, "light")

        is_fp = rng.random() < 0.5
        if is_fp:
            img = make_fullpage(img, rng, fill_extras=rng.random() < 0.3)

        applied_degradation = rng.random() < 0.3
        if applied_degradation:
            img = apply_phone_capture(img, rng, "mild")

        fname = f"partial_{idx:04d}.png"
        cv2.imwrite(str(partial_dir / fname), img)
        manifest.append(CorpusImage(
            path=f"partial/{fname}", category="partial", subcategory=curve_type,
            curve_type=curve_type, n_marks=n_marks, degradation="mild" if applied_degradation else "none",
            obstruction="none", is_fullpage=is_fp, pen_style=pen,
            paper_style=paper, seed=seed + idx
        ))
        idx += 1

    # --- FILLED (6-10+ marks, 100 images) ---
    filled_dir = output_dir / "filled"
    filled_dir.mkdir(exist_ok=True)
    for i in range(100):
        img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * rng.randint(245, 255)
        draw_grid(img, rng)
        draw_alert_action_lines(img, rng)

        curve_type = rng.choice(curve_types)
        points = CURVE_GENERATORS[curve_type](rng, n_points=rng.randint(5, 10))
        if len(points) < 5:
            extra_h = points[-1][0] if points else 0
            while len(points) < 5:
                extra_h += rng.uniform(0.5, 1.5)
                extra_cm = min(10.0, points[-1][1] + rng.uniform(0.3, 1.0))
                points.append((extra_h, extra_cm))

        pen = rng.choice(pen_styles)
        for h, cm in points:
            draw_x_mark(img, _hours_to_px(h), _cm_to_px(cm), rng, pen_style=pen)

        paper = rng.choice(paper_styles)
        if paper == "aged_light":
            img = apply_paper_aging(img, rng, "light")
        elif paper == "aged_heavy":
            img = apply_paper_aging(img, rng, "heavy")
        elif paper == "photocopy":
            img = apply_photocopy_effect(img, rng)

        is_fp = rng.random() < 0.6
        if is_fp:
            img = make_fullpage(img, rng, fill_extras=True)

        fname = f"filled_{idx:04d}.png"
        cv2.imwrite(str(filled_dir / fname), img)
        manifest.append(CorpusImage(
            path=f"filled/{fname}", category="filled", subcategory=curve_type,
            curve_type=curve_type, n_marks=len(points), degradation="none",
            obstruction="none", is_fullpage=is_fp, pen_style=pen,
            paper_style=paper, seed=seed + idx
        ))
        idx += 1

    # --- DEGRADED (phone capture, various severity, 80 images) ---
    degraded_dir = output_dir / "degraded"
    degraded_dir.mkdir(exist_ok=True)
    for i in range(80):
        img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * rng.randint(245, 255)
        draw_grid(img, rng)
        draw_alert_action_lines(img, rng)

        curve_type = rng.choice(curve_types)
        points = CURVE_GENERATORS[curve_type](rng)
        n_marks = rng.randint(0, len(points))
        selected = points[:n_marks]

        pen = rng.choice(pen_styles)
        for h, cm in selected:
            draw_x_mark(img, _hours_to_px(h), _cm_to_px(cm), rng, pen_style=pen)

        is_fp = rng.random() < 0.5
        if is_fp:
            img = make_fullpage(img, rng, fill_extras=n_marks > 0)

        severity = rng.choice(capture_severities)
        img = apply_phone_capture(img, rng, severity)

        # Additional noise for harsh
        if severity == "harsh":
            noise = np.random.RandomState(rng.randint(0, 2**31)).randint(0, 20, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

        fname = f"degraded_{idx:04d}.png"
        cv2.imwrite(str(degraded_dir / fname), img)
        manifest.append(CorpusImage(
            path=f"degraded/{fname}", category="degraded", subcategory=severity,
            curve_type=curve_type, n_marks=n_marks, degradation=severity,
            obstruction="none", is_fullpage=is_fp, pen_style=pen,
            paper_style="clean", seed=seed + idx
        ))
        idx += 1

    # --- OBSTRUCTED (60 images) ---
    obstructed_dir = output_dir / "obstructed"
    obstructed_dir.mkdir(exist_ok=True)
    for i in range(60):
        img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * rng.randint(245, 255)
        draw_grid(img, rng)
        draw_alert_action_lines(img, rng)

        curve_type = rng.choice(curve_types)
        points = CURVE_GENERATORS[curve_type](rng)
        n_marks = rng.randint(2, len(points))
        selected = points[:n_marks]

        pen = rng.choice(pen_styles)
        for h, cm in selected:
            draw_x_mark(img, _hours_to_px(h), _cm_to_px(cm), rng, pen_style=pen)

        is_fp = rng.random() < 0.5
        if is_fp:
            img = make_fullpage(img, rng, fill_extras=True)

        # Apply obstruction
        obs_type = rng.choice(obstruction_types)
        img = apply_obstruction(img, rng, kind=obs_type)

        applied_deg = rng.random() < 0.4
        if applied_deg:
            img = apply_phone_capture(img, rng, "mild")

        fname = f"obstructed_{idx:04d}.png"
        cv2.imwrite(str(obstructed_dir / fname), img)
        manifest.append(CorpusImage(
            path=f"obstructed/{fname}", category="obstructed", subcategory=obs_type,
            curve_type=curve_type, n_marks=n_marks, degradation="mild" if applied_deg else "none",
            obstruction=obs_type, is_fullpage=is_fp, pen_style=pen,
            paper_style="clean", seed=seed + idx
        ))
        idx += 1

    # --- Write manifest ---
    manifest_data = [vars(m) for m in manifest]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    # --- Write summary stats ---
    stats = {
        "total_images": len(manifest),
        "by_category": {},
        "by_curve_type": {},
        "by_pen_style": {},
        "fullpage_count": sum(1 for m in manifest if m.is_fullpage),
        "crop_count": sum(1 for m in manifest if not m.is_fullpage),
    }
    for m in manifest:
        stats["by_category"][m.category] = stats["by_category"].get(m.category, 0) + 1
        if m.curve_type != "none":
            stats["by_curve_type"][m.curve_type] = stats["by_curve_type"].get(m.curve_type, 0) + 1
        if m.pen_style != "none":
            stats["by_pen_style"][m.pen_style] = stats["by_pen_style"].get(m.pen_style, 0) + 1

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    return manifest


if __name__ == "__main__":
    out = Path("/root/work/data")
    print("Generating corpus...")
    results = generate_corpus(out, target_count=350)
    print(f"Generated {len(results)} images in {out}")
    print(f"Manifest: {out / 'manifest.json'}")
    print(f"Stats: {out / 'stats.json'}")

#!/usr/bin/env python3
"""Generate V2 training data using the rich corpus generator.

Produces 700 crop-only training images with full variation (paper_style,
pen_style, degradation, obstruction) using seed 99999 (no overlap with eval
seed 12345 or V1 training seed 77777). All images are chart crops (no
fullpage) to match inference-time input format.

Over-samples failure modes from V1 eval:
- filled with 5-10 marks (especially rapid_precipitous curves)
- aged_heavy paper
- dense marks that the model over-counts

Output: data/training_v2/ with labels.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generate_corpus import (
    CHART_H, CHART_W, CURVE_GENERATORS, HOURS_MAX, CM_MAX,
    _hours_to_px, _cm_to_px,
    draw_grid, draw_alert_action_lines, draw_x_mark,
    apply_paper_aging, apply_photocopy_effect, apply_phone_capture,
    apply_obstruction,
)

import cv2
import numpy as np

SEED = 99999
OUTPUT_DIR = Path("data/training_v2")


def snap_coord(val: float, lo: float, hi: float) -> float:
    """Round to nearest 0.5, clamp to [lo, hi]."""
    return max(lo, min(hi, round(val * 2) / 2))


def generate_training_corpus() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    labels: list[dict] = []
    idx = 0

    pen_styles = ["ballpoint", "felt_tip", "pencil", "shaky"]
    paper_styles = ["clean", "aged_light", "aged_heavy", "photocopy"]
    curve_types = list(CURVE_GENERATORS.keys())
    obstruction_types = ["finger", "coffee_stain", "fold", "tape"]

    def make_image(
        category: str,
        curve_type: str,
        n_marks_override: int | None = None,
        pen: str | None = None,
        paper: str | None = None,
        degradation: str = "none",
        obstruction: str = "none",
    ) -> None:
        nonlocal idx
        img = np.ones((CHART_H, CHART_W, 3), dtype=np.uint8) * rng.randint(245, 255)
        grid_style = rng.choice(["clean", "faded", "bold"])
        draw_grid(img, rng, style=grid_style)
        draw_alert_action_lines(img, rng, style=rng.choice(["normal", "faded"]))

        pen = pen or rng.choice(pen_styles)
        paper = paper or rng.choice(paper_styles)

        if curve_type == "none":
            points: list[tuple[float, float]] = []
        else:
            gen = CURVE_GENERATORS[curve_type]
            raw = gen(rng, n_points=rng.randint(5, 10) if category in ("filled",) else rng.randint(3, 8))
            if n_marks_override is not None:
                raw = raw[:n_marks_override]
            if category == "filled" and len(raw) < 5:
                last_h = raw[-1][0] if raw else 0
                while len(raw) < 5:
                    last_h += rng.uniform(0.5, 1.5)
                    last_cm = min(10.0, raw[-1][1] + rng.uniform(0.3, 1.0)) if raw else rng.uniform(4, 6)
                    raw.append((last_h, last_cm))
            points = raw

        for h, cm in points:
            draw_x_mark(img, _hours_to_px(h), _cm_to_px(cm), rng, pen_style=pen)

        if paper == "aged_light":
            img = apply_paper_aging(img, rng, "light")
        elif paper == "aged_heavy":
            img = apply_paper_aging(img, rng, "heavy")
        elif paper == "photocopy":
            img = apply_photocopy_effect(img, rng)

        if degradation == "mild":
            img = apply_phone_capture(img, rng, "mild")
        elif degradation == "harsh":
            img = apply_phone_capture(img, rng, "harsh")
            noise = np.random.RandomState(rng.randint(0, 2**31)).randint(0, 20, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)

        if obstruction != "none":
            img = apply_obstruction(img, rng, kind=obstruction)

        fname = f"train_v2_{idx:04d}.png"
        cv2.imwrite(str(OUTPUT_DIR / fname), img)

        label_points = [[snap_coord(h, 0, HOURS_MAX), snap_coord(cm, 0, CM_MAX)] for h, cm in points]

        labels.append({
            "image": f"training_v2/{fname}",
            "points": label_points,
            "n_marks": len(points),
            "category": category,
            "curve_type": curve_type,
            "paper_style": paper,
            "pen_style": pen,
            "degradation": degradation,
            "obstruction": obstruction,
        })
        idx += 1

    # ============================================================
    # BLANK (80 images) - model must learn to output {"p":[]}
    # ============================================================
    for _ in range(80):
        make_image("blank", "none",
                   paper=rng.choice(paper_styles),
                   degradation=rng.choice(["none", "mild"]),
                   )

    # ============================================================
    # PARTIAL (80 images) - 1-3 marks
    # ============================================================
    for _ in range(80):
        ct = rng.choice(curve_types)
        make_image("partial", ct,
                   n_marks_override=rng.randint(1, 3),
                   paper=rng.choice(paper_styles),
                   degradation=rng.choice(["none", "none", "mild"]))

    # ============================================================
    # FILLED (300 images) - 5-10 marks, HEAVILY over-sampled
    # Over-sample: rapid_precipitous, aged_heavy, dense (8-10 marks)
    # ============================================================

    for ct in curve_types:
        for _ in range(75):
            paper = rng.choice(["clean", "aged_light", "aged_heavy", "aged_heavy", "photocopy"])
            make_image("filled", ct,
                       paper=paper,
                       degradation=rng.choice(["none", "none", "mild"]))

    # ============================================================
    # DEGRADED (120 images) - phone capture artifacts
    # ============================================================
    for _ in range(120):
        ct = rng.choice(curve_types)
        n = rng.randint(0, 8)
        make_image("degraded", ct if n > 0 else "none",
                   n_marks_override=n if ct != "none" else 0,
                   degradation=rng.choice(["mild", "harsh"]))

    # ============================================================
    # OBSTRUCTED (80 images)
    # ============================================================
    for _ in range(80):
        ct = rng.choice(curve_types)
        obs = rng.choice(obstruction_types)
        make_image("obstructed", ct,
                   n_marks_override=rng.randint(2, 8),
                   obstruction=obs,
                   degradation=rng.choice(["none", "mild"]))

    # ============================================================
    # ADVERSARIAL BLANK (40 images) - grids that could fool model
    # ============================================================
    for _ in range(40):
        make_image("blank", "none",
                   paper=rng.choice(["aged_heavy", "photocopy"]),
                   degradation=rng.choice(["none", "mild", "harsh"]))

    labels_path = OUTPUT_DIR / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2))

    cats = {}
    for l in labels:
        cats[l["category"]] = cats.get(l["category"], 0) + 1
    print(f"Generated {len(labels)} training images in {OUTPUT_DIR}")
    print(f"Categories: {cats}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    generate_training_corpus()

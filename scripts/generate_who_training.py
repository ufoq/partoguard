"""Generate WHO-template training images for V8 mixed training.

Draws X marks on real WHO partograph templates at known coordinates,
with phone-capture degradation effects. Produces 100 images + labels.

Seed: 88888 (no overlap with eval=12345, synthetic training=77777).
Output: data/training/train_0400.png ... train_0499.png + appends to labels.json
"""

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SEED = 88888
N_IMAGES = 100
OUTPUT_DIR = Path("data/training")
TEMPLATES_DIR = Path("data/harvested/open_access_papers/extracted_figures")
TEMPLATES_NEW_DIR = Path("data/harvested/extracted_figures_new")


@dataclass
class TemplateCalibration:
    """Pixel-coordinate calibration for a WHO chart template."""
    path: str
    # Cervicograph crop region (full image coords)
    crop_y1: int
    crop_y2: int
    crop_x1: int
    crop_x2: int
    # Grid origin in CROPPED image coords (0h, 0cm corner)
    x_start: int
    y_start: int  # y pixel for 0cm (bottom of grid)
    # Scale factors
    px_per_hour: float
    px_per_cm: float


# Calibrated templates
TEMPLATES = [
    TemplateCalibration(
        path=str(TEMPLATES_DIR / "bmc2013_p3_fig1.jpeg"),
        crop_y1=320, crop_y2=680, crop_x1=60, crop_x2=850,
        x_start=130, y_start=328,
        px_per_hour=49.67, px_per_cm=31.3,
    ),
    TemplateCalibration(
        path=str(TEMPLATES_DIR / "bmc2013_p3_fig1.jpeg"),
        # Wider crop including some header/footer context
        crop_y1=250, crop_y2=750, crop_x1=30, crop_x2=850,
        x_start=160, y_start=398,
        px_per_hour=49.67, px_per_cm=31.3,
    ),
    TemplateCalibration(
        path=str(TEMPLATES_DIR / "bmc2013_p3_fig1.jpeg"),
        # Full page (no crop) — model should handle this too
        crop_y1=0, crop_y2=1418, crop_x1=0, crop_x2=867,
        x_start=193, y_start=638,
        px_per_hour=47.5, px_per_cm=25.0,
    ),
]


# Curve generators — return list of (hour, dilation_cm) points
def gen_normal(rng: random.Random, n_points: int) -> list[tuple[float, float]]:
    """Normal labour: ~1cm/hr, stays left of alert line."""
    start_h = rng.choice([0.0, 0.5])
    start_d = rng.uniform(3.5, 5.0)
    points = [(start_h, round(start_d * 2) / 2)]
    for i in range(1, n_points):
        h = start_h + i * rng.uniform(0.8, 1.2)
        d = start_d + i * rng.uniform(0.8, 1.5)
        d = min(d, 10.0)
        h = min(h, 12.0)
        points.append((round(h * 2) / 2, round(d * 2) / 2))
    return points


def gen_slow(rng: random.Random, n_points: int) -> list[tuple[float, float]]:
    """Slow labour: crosses alert line (0.5cm/hr)."""
    start_h = rng.choice([0.0, 0.5, 1.0])
    start_d = rng.uniform(3.0, 4.5)
    points = [(start_h, round(start_d * 2) / 2)]
    for i in range(1, n_points):
        h = start_h + i * rng.uniform(1.2, 2.0)
        d = start_d + i * rng.uniform(0.4, 0.7)
        d = min(d, 10.0)
        h = min(h, 12.0)
        points.append((round(h * 2) / 2, round(d * 2) / 2))
    return points


def gen_arrested(rng: random.Random, n_points: int) -> list[tuple[float, float]]:
    """Arrested labour: crosses action line, plateau."""
    start_h = rng.choice([0.0, 0.5])
    start_d = rng.uniform(3.0, 4.5)
    points = [(start_h, round(start_d * 2) / 2)]
    plateau_d = start_d + rng.uniform(1.0, 2.0)
    for i in range(1, n_points):
        h = start_h + i * rng.uniform(1.5, 2.5)
        d = plateau_d + rng.uniform(-0.3, 0.3)
        d = max(start_d, min(d, 10.0))
        h = min(h, 12.0)
        points.append((round(h * 2) / 2, round(d * 2) / 2))
    return points


def gen_rapid(rng: random.Random, n_points: int) -> list[tuple[float, float]]:
    """Rapid labour: very steep, finishes quickly."""
    start_h = 0.0
    start_d = rng.uniform(4.0, 5.5)
    points = [(start_h, round(start_d * 2) / 2)]
    for i in range(1, n_points):
        h = start_h + i * rng.uniform(0.3, 0.7)
        d = start_d + i * rng.uniform(1.2, 2.0)
        d = min(d, 10.0)
        h = min(h, 12.0)
        points.append((round(h * 2) / 2, round(d * 2) / 2))
    return points


CURVE_GENERATORS = {
    "normal": gen_normal,
    "slow": gen_slow,
    "arrested": gen_arrested,
    "rapid": gen_rapid,
}


def draw_x_mark(img: np.ndarray, cx: int, cy: int, rng: random.Random) -> None:
    """Draw a single X mark with randomized pen style."""
    sz = rng.randint(6, 12)
    thickness = rng.randint(2, 4)
    # Pen colors: dark blue, black, dark red, dark green (ballpoint variants)
    colors = [
        (rng.randint(100, 140), rng.randint(15, 40), rng.randint(15, 40)),  # dark red/brown
        (rng.randint(30, 60), rng.randint(30, 60), rng.randint(30, 60)),    # near-black
        (rng.randint(130, 160), rng.randint(50, 80), rng.randint(10, 30)),  # dark blue (BGR)
        (rng.randint(20, 50), rng.randint(80, 120), rng.randint(20, 50)),   # dark green
    ]
    color = rng.choice(colors)

    # Slight rotation/offset for hand-drawn feel
    angle = rng.uniform(-0.2, 0.2)
    dx = int(sz * math.cos(angle + math.pi / 4))
    dy = int(sz * math.sin(angle + math.pi / 4))
    dx2 = int(sz * math.cos(angle - math.pi / 4))
    dy2 = int(sz * math.sin(angle - math.pi / 4))

    # Small positional jitter (hand doesn't land perfectly on grid)
    jx = rng.randint(-2, 2)
    jy = rng.randint(-2, 2)
    cx += jx
    cy += jy

    cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - dx2, cy + dy2), (cx + dx2, cy - dy2), color, thickness, cv2.LINE_AA)


def apply_degradation(img: np.ndarray, rng: random.Random, severity: str = "mild") -> np.ndarray:
    """Apply phone-capture degradation effects."""
    h, w = img.shape[:2]
    effects = []

    if severity == "full":
        # Pick 2-4 effects
        all_effects = ["perspective", "rotation", "lighting", "blur", "noise", "jpeg"]
        n_effects = rng.randint(2, 4)
        effects = rng.sample(all_effects, n_effects)
    else:
        # Mild: 1-2 subtle effects
        all_effects = ["rotation", "lighting", "noise"]
        n_effects = rng.randint(1, 2)
        effects = rng.sample(all_effects, n_effects)

    result = img.copy()

    if "perspective" in effects:
        # Keystoning from tilted phone capture
        margin = rng.randint(10, 30)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Random corner displacement
        dst_pts = np.float32([
            [rng.randint(0, margin), rng.randint(0, margin)],
            [w - rng.randint(0, margin), rng.randint(0, margin)],
            [w - rng.randint(0, margin), h - rng.randint(0, margin)],
            [rng.randint(0, margin), h - rng.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if "rotation" in effects:
        angle = rng.uniform(-5, 5) if severity == "full" else rng.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if "lighting" in effects:
        # Uneven lighting gradient
        gradient = np.zeros((h, w), dtype=np.float32)
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                gradient[y, x] = 1.0 - min(dist / max(w, h), 0.3)
        # This is slow for large images — use vectorized version
        Y, X = np.mgrid[0:h, 0:w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        gradient = 1.0 - np.minimum(dist / max(w, h), 0.3)
        result = (result.astype(np.float32) * gradient[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

    if "blur" in effects:
        k = rng.choice([3, 5]) if severity == "full" else 3
        result = cv2.GaussianBlur(result, (k, k), 0)

    if "noise" in effects:
        sigma = rng.uniform(8, 20) if severity == "full" else rng.uniform(3, 8)
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, result.shape)
        result = (result.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)

    if "jpeg" in effects:
        quality = rng.randint(55, 75)
        _, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return result


def generate_who_training() -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    # Load existing labels
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path) as f:
        labels = json.load(f)

    # Pre-load templates
    template_images = {}
    for tmpl in TEMPLATES:
        if tmpl.path not in template_images:
            img = cv2.imread(tmpl.path)
            if img is None:
                raise ValueError(f"Failed to load template: {tmpl.path}")
            template_images[tmpl.path] = img

    # Distribution: 10 blank, 20 partial, 40 filled, 30 degraded
    categories = (
        [("blank", 0)] * 10 +
        [("partial", rng.randint(1, 3))] * 20 +
        [("filled", rng.randint(4, 8))] * 40 +
        [("degraded", rng.randint(2, 7))] * 30
    )
    # Re-randomize n_marks for partial/filled/degraded after shuffle
    rng.shuffle(categories)

    generated = 0
    for idx, (category, _) in enumerate(categories):
        img_idx = 400 + idx
        filename = f"train_{img_idx:04d}.png"

        # Pick template
        tmpl = rng.choice(TEMPLATES)
        base_img = template_images[tmpl.path].copy()

        # Crop cervicograph region
        cropped = base_img[tmpl.crop_y1:tmpl.crop_y2, tmpl.crop_x1:tmpl.crop_x2].copy()

        # Add crop jitter (±5-10px)
        jitter_x = rng.randint(-8, 8)
        jitter_y = rng.randint(-8, 8)
        h, w = cropped.shape[:2]
        # Pad then re-crop with offset
        padded = cv2.copyMakeBorder(cropped, 15, 15, 15, 15, cv2.BORDER_REPLICATE)
        sy = 15 + jitter_y
        sx = 15 + jitter_x
        cropped = padded[sy:sy + h, sx:sx + w]

        # Determine number of marks
        if category == "blank":
            n_marks = 0
            curve_type = "none"
        elif category == "partial":
            n_marks = rng.randint(1, 3)
            curve_type = rng.choice(list(CURVE_GENERATORS.keys()))
        elif category == "filled":
            n_marks = rng.randint(4, 8)
            curve_type = rng.choice(list(CURVE_GENERATORS.keys()))
        else:  # degraded
            n_marks = rng.randint(0, 7)
            curve_type = rng.choice(list(CURVE_GENERATORS.keys())) if n_marks > 0 else "none"

        # Generate points and draw
        points: list[list[float]] = []
        if n_marks > 0:
            gen_fn = CURVE_GENERATORS[curve_type]
            curve_points = gen_fn(rng, n_marks)

            for h_val, d_val in curve_points:
                # Clamp to valid grid
                h_val = max(0.0, min(12.0, h_val))
                d_val = max(0.0, min(10.0, d_val))

                # Convert to pixel coordinates
                px_x = int(tmpl.x_start + h_val * tmpl.px_per_hour)
                px_y = int(tmpl.y_start - d_val * tmpl.px_per_cm)

                # Draw the X mark
                draw_x_mark(cropped, px_x, px_y, rng)
                points.append([h_val, d_val, 0.99])

        # Apply degradation
        if category == "degraded":
            cropped = apply_degradation(cropped, rng, severity="full")
        elif rng.random() < 0.30:
            # 30% chance of mild degradation for other categories
            cropped = apply_degradation(cropped, rng, severity="mild")

        # Save
        out_path = OUTPUT_DIR / filename
        cv2.imwrite(str(out_path), cropped)

        # Label
        labels.append({
            "image": f"training/{filename}",
            "points": points,
            "n_marks": len(points),
            "category": category,
            "curve_type": curve_type,
            "template": "who",
        })
        generated += 1

    # Write updated labels
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"Generated {generated} WHO-template training images")
    print(f"Total training set: {len(labels)} images")
    print(f"Output: {OUTPUT_DIR}/train_0400.png ... train_0499.png")


if __name__ == "__main__":
    generate_who_training()

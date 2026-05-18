"""Tests for synthetic partograph generation."""

import json
from pathlib import Path

import cv2

from partoguard.core.imaging.synthetic import (
    CANONICAL_SCENARIOS,
    draw_chart_crop,
    generate_all,
    generate_scenario,
    apply_degradation,
    _make_fullpage,
)
import random


def test_draw_chart_crop_returns_valid_image():
    sc = CANONICAL_SCENARIOS[0]
    img = draw_chart_crop(sc)
    assert img.shape == (600, 800, 3)
    assert img.dtype.name == "uint8"


def test_draw_chart_crop_not_all_white():
    sc = CANONICAL_SCENARIOS[0]
    img = draw_chart_crop(sc)
    assert img.mean() < 250


def test_generate_scenario_clean_only(tmp_path):
    sc = CANONICAL_SCENARIOS[0]
    results = generate_scenario(sc, tmp_path, include_degraded=False, include_fullpage=False)
    assert len(results) == 1
    assert results[0].variant == "clean_crop"
    assert results[0].path.exists()


def test_generate_scenario_all_variants(tmp_path):
    sc = CANONICAL_SCENARIOS[0]
    results = generate_scenario(sc, tmp_path)
    variants = {r.variant for r in results}
    assert "clean_crop" in variants
    assert "moderate_crop" in variants
    assert "heavy_crop" in variants
    assert "clean_fullpage" in variants
    assert "moderate_fullpage" in variants
    assert "heavy_fullpage" in variants
    assert len(results) == 6


def test_generate_all_canonical(tmp_path):
    results = generate_all(tmp_path)
    assert len(results) == 18  # 3 scenarios × 6 variants
    labels_path = tmp_path / "labels.json"
    assert labels_path.exists()
    labels = json.loads(labels_path.read_text())
    assert len(labels) == 18
    assert all(lbl["synthetic"] is True for lbl in labels)
    assert all(lbl["clinical_use"] is False for lbl in labels)


def test_labels_contain_points(tmp_path):
    results = generate_all(tmp_path)
    for r in results:
        assert "points" in r.label
        assert len(r.label["points"]) > 0
        assert "expected_zone" in r.label


def test_degradation_changes_image():
    sc = CANONICAL_SCENARIOS[0]
    clean = draw_chart_crop(sc)
    rng = random.Random(99)
    degraded = apply_degradation(clean, rng, level="heavy")
    diff = cv2.absdiff(clean, degraded).mean()
    assert diff > 1.0


def test_fullpage_larger_than_crop():
    sc = CANONICAL_SCENARIOS[1]
    crop = draw_chart_crop(sc)
    rng = random.Random(42)
    fullpage = _make_fullpage(crop, rng)
    assert fullpage.shape[0] > crop.shape[0]
    assert fullpage.shape[1] > crop.shape[1]


def test_all_scenarios_have_expected_zones(tmp_path):
    results = generate_all(tmp_path)
    zones = {r.label["expected_zone"] for r in results}
    assert "normal" in zones
    assert "alert_zone" in zones
    assert "action_zone" in zones

from pathlib import Path
import json

import cv2
import numpy as np

from partoguard.core.eval import evaluate_corpus_dir, evaluate_synthetic_dir, format_corpus_eval_summary, format_eval_summary
from partoguard.core.imaging.synthetic import CANONICAL_SCENARIOS, generate_all


def test_eval_harness_scores_clean_and_degraded_synthetic_set(tmp_path: Path):
    generate_all(tmp_path, scenarios=CANONICAL_SCENARIOS[:3])
    summary = evaluate_synthetic_dir(tmp_path)

    assert summary.total == 18
    assert summary.evaluated > 0
    assert summary.zone_accuracy >= 0.95
    assert summary.full_set_success_rate < summary.zone_accuracy
    assert summary.manual_review_rate > 0.0
    text = format_eval_summary(summary)
    assert "Non-manual zone accuracy" in text
    assert "Full-set success rate" in text


def test_eval_rejects_labels_outside_synthetic_dir(tmp_path: Path):
    labels = tmp_path / "labels.json"
    labels.write_text('[{"file":"/etc/passwd","expected_zone":"normal"}]')

    try:
        evaluate_synthetic_dir(tmp_path)
    except ValueError as exc:
        assert "outside the synthetic directory" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_corpus_eval_reports_category_manual_review_counts(tmp_path: Path):
    generate_all(tmp_path, scenarios=CANONICAL_SCENARIOS[:1])
    blank_image = tmp_path / "blank.png"
    cv2.imwrite(str(blank_image), np.ones((100, 100, 3), dtype=np.uint8) * 255)
    manifest = [
        {"path": "normal_progress_clean_crop.png", "category": "filled"},
        {"path": "blank.png", "category": "blank"},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    summary = evaluate_corpus_dir(tmp_path)
    text = format_corpus_eval_summary(summary)

    assert summary.total == 2
    assert summary.by_category["filled"]["total"] == 1
    assert summary.blank_total == 1
    assert summary.blank_manual_reviews == 1
    assert "PartoGuard corpus evaluation" in text
    assert "Blank-template manual review rate" in text


def test_corpus_eval_rejects_manifest_paths_outside_corpus_dir(tmp_path: Path):
    (tmp_path / "manifest.json").write_text('[{"path":"/etc/passwd","category":"blank"}]')

    try:
        evaluate_corpus_dir(tmp_path)
    except ValueError as exc:
        assert "outside the corpus directory" in str(exc)
    else:
        raise AssertionError("expected ValueError")

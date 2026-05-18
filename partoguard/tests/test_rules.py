"""Unit tests for the deterministic partograph rule engine."""

from __future__ import annotations

from partoguard.core.rules.engine import UNCERTAINTY_CM, classify_zone
from partoguard.core.schemas.contracts import DilationPoint, ZoneStatus


def _point(hours: float, dilation: float, confidence: float = 0.9) -> DilationPoint:
    return DilationPoint(x_hours=hours, dilation_cm=dilation, confidence=confidence)


def test_classify_zone_normal_points_left_of_alert_line():
    result = classify_zone(
        [
            _point(0.0, 5.0, 0.92),
            _point(1.0, 6.3, 0.88),
            _point(2.0, 7.4, 0.91),
        ]
    )

    assert result.status == ZoneStatus.NORMAL
    assert result.requires_human_review is False
    assert result.confidence == 0.88
    assert result.explanation.lower().count("review") >= 1
    assert result.explanation.lower().count("protocol") >= 1


def test_classify_zone_alert_zone_case():
    result = classify_zone([_point(4.0, 6.0, 0.84)])

    assert result.status == ZoneStatus.ALERT_ZONE
    assert result.requires_human_review is False
    assert result.triggering_point == _point(4.0, 6.0, 0.84)
    assert "(4h, 6cm)" in result.explanation
    assert "escalate per protocol" in result.explanation.lower()


def test_classify_zone_action_zone_case():
    result = classify_zone([_point(8.0, 7.0, 0.72)])

    assert result.status == ZoneStatus.ACTION_ZONE
    assert result.requires_human_review is False
    assert result.triggering_point == _point(8.0, 7.0, 0.72)
    assert result.confidence == 0.72
    assert "(8h, 7cm)" in result.explanation


def test_classify_zone_near_alert_line_is_indeterminate():
    result = classify_zone([_point(4.0, 7.6, 0.66)], uncertainty_cm=UNCERTAINTY_CM)

    assert result.status == ZoneStatus.INDETERMINATE
    assert result.requires_human_review is True
    assert result.confidence == 0.66
    assert "(4h, 7.6cm)" in result.explanation
    assert "review manually" in result.explanation.lower()


def test_classify_zone_near_action_line_is_indeterminate():
    result = classify_zone([_point(8.0, 7.6, 0.61)], uncertainty_cm=UNCERTAINTY_CM)

    assert result.status == ZoneStatus.INDETERMINATE
    assert result.requires_human_review is True
    assert result.confidence == 0.61
    assert "(8h, 7.6cm)" in result.explanation


def test_classify_zone_empty_points_returns_manual_review():
    result = classify_zone([])

    assert result.status == ZoneStatus.MANUAL_REVIEW
    assert result.requires_human_review is True
    assert result.confidence == 0.0
    assert "review manually" in result.explanation.lower()


def test_classify_zone_single_point_exactly_on_alert_line_is_normal():
    result = classify_zone([_point(3.0, 7.0, 0.77)])

    assert result.status == ZoneStatus.NORMAL
    assert result.requires_human_review is False
    assert result.triggering_point == _point(3.0, 7.0, 0.77)
    assert "(3h, 7cm)" in result.explanation


def test_classify_zone_multiple_points_worst_status_wins():
    result = classify_zone(
        [
            _point(1.0, 6.4, 0.95),
            _point(4.0, 6.0, 0.82),
            _point(8.0, 7.0, 0.58),
        ]
    )

    assert result.status == ZoneStatus.ACTION_ZONE
    assert result.triggering_point == _point(8.0, 7.0, 0.58)
    assert result.confidence == 0.58


def test_classify_zone_points_before_active_phase_are_ignored():
    result = classify_zone(
        [
            _point(0.0, 3.2, 0.91),
            _point(1.0, 3.8, 0.87),
            _point(4.0, 6.0, 0.79),
        ]
    )

    assert result.status == ZoneStatus.ALERT_ZONE
    assert result.triggering_point == _point(4.0, 6.0, 0.79)
    assert result.confidence == 0.79


def test_classify_zone_canonical_teaching_normal_scenario():
    result = classify_zone([_point(0.0, 4.0, 0.94), _point(6.0, 10.0, 0.83)])

    assert result.status == ZoneStatus.NORMAL
    assert result.requires_human_review is False
    assert result.confidence == 0.83


def test_classify_zone_canonical_teaching_alert_scenario():
    result = classify_zone([_point(0.0, 4.0, 0.96), _point(4.0, 6.0, 0.8)])

    assert result.status == ZoneStatus.ALERT_ZONE
    assert result.requires_human_review is False
    assert result.triggering_point == _point(4.0, 6.0, 0.8)


def test_classify_zone_canonical_teaching_action_scenario():
    result = classify_zone(
        [
            _point(0.0, 4.0, 0.96),
            _point(4.0, 6.0, 0.81),
            _point(8.0, 8.0, 0.63),
        ]
    )

    assert result.status == ZoneStatus.ACTION_ZONE
    assert result.requires_human_review is False
    assert result.triggering_point == _point(8.0, 8.0, 0.63)
    assert result.confidence == 0.63


def test_classify_zone_invalid_dilation_returns_manual_review():
    invalid_point = DilationPoint.model_construct(
        x_hours=2.0,
        dilation_cm=11.0,
        bbox=(0, 0, 0, 0),
        confidence=0.55,
        source="cv",
    )

    result = classify_zone([invalid_point])

    assert result.status == ZoneStatus.MANUAL_REVIEW
    assert result.requires_human_review is True
    assert result.triggering_point == invalid_point
    assert result.confidence == 0.55
    assert "(2h, 11cm)" in result.explanation

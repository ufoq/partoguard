from partoguard.core.corpus_scorer import score_entry
from partoguard.core.schemas.contracts import ZoneStatus


def test_blank_must_manual_review_with_zero_points():
    v = score_entry(
        category="blank", curve_type="none", n_marks=0,
        actual_status=ZoneStatus.MANUAL_REVIEW, actual_n_points=0,
    )
    assert v.correct


def test_blank_fails_when_pipeline_returns_zone():
    v = score_entry(
        category="blank", curve_type="none", n_marks=0,
        actual_status=ZoneStatus.NORMAL, actual_n_points=2,
    )
    assert not v.correct
    assert "manual_review" in v.reason


def test_blank_fails_when_pipeline_hallucinates_points_but_still_manual():
    v = score_entry(
        category="blank", curve_type="none", n_marks=0,
        actual_status=ZoneStatus.MANUAL_REVIEW, actual_n_points=3,
    )
    assert not v.correct


def test_single_mark_accepts_manual_review():
    v = score_entry(
        category="partial", curve_type="normal", n_marks=1,
        actual_status=ZoneStatus.MANUAL_REVIEW, actual_n_points=0,
    )
    assert v.correct


def test_single_mark_accepts_any_zone_with_one_point():
    v = score_entry(
        category="partial", curve_type="normal", n_marks=1,
        actual_status=ZoneStatus.NORMAL, actual_n_points=1,
    )
    assert v.correct


def test_single_mark_rejects_overcount():
    v = score_entry(
        category="partial", curve_type="normal", n_marks=1,
        actual_status=ZoneStatus.NORMAL, actual_n_points=4,
    )
    assert not v.correct
    assert "hallucination" in v.reason


def test_filled_requires_count_and_zone():
    v = score_entry(
        category="filled", curve_type="normal", n_marks=5,
        actual_status=ZoneStatus.ALERT_ZONE, actual_n_points=4,
    )
    assert v.correct


def test_filled_rejects_severe_undercount():
    v = score_entry(
        category="filled", curve_type="normal", n_marks=5,
        actual_status=ZoneStatus.ALERT_ZONE, actual_n_points=1,
    )
    assert not v.correct
    assert "under-count" in v.reason


def test_filled_rejects_severe_overcount():
    v = score_entry(
        category="filled", curve_type="normal", n_marks=5,
        actual_status=ZoneStatus.ACTION_ZONE, actual_n_points=20,
    )
    assert not v.correct
    assert "over-count" in v.reason


def test_filled_rejects_manual_review_on_multi_mark():
    v = score_entry(
        category="filled", curve_type="normal", n_marks=5,
        actual_status=ZoneStatus.MANUAL_REVIEW, actual_n_points=3,
    )
    assert not v.correct
    assert "manual_review" in v.reason


def test_filled_rejects_wrong_zone_for_curve_type():
    v = score_entry(
        category="filled", curve_type="normal", n_marks=5,
        actual_status=ZoneStatus.ACTION_ZONE, actual_n_points=5,
    )
    assert not v.correct
    assert "not in expected" in v.reason


def test_obstructed_allows_undercount_with_valid_zone():
    v = score_entry(
        category="obstructed", curve_type="normal", n_marks=7,
        actual_status=ZoneStatus.NORMAL, actual_n_points=2,
    )
    assert v.correct


def test_obstructed_still_rejects_overcount():
    v = score_entry(
        category="obstructed", curve_type="normal", n_marks=5,
        actual_status=ZoneStatus.ALERT_ZONE, actual_n_points=15,
    )
    assert not v.correct
    assert "over-count" in v.reason


def test_arrested_curve_requires_alert_or_action_zone():
    v = score_entry(
        category="filled", curve_type="arrested", n_marks=5,
        actual_status=ZoneStatus.ACTION_ZONE, actual_n_points=5,
    )
    assert v.correct
    v2 = score_entry(
        category="filled", curve_type="arrested", n_marks=5,
        actual_status=ZoneStatus.NORMAL, actual_n_points=5,
    )
    assert not v2.correct


# --- Sparse observation tolerance (n_marks <= 2) ---


def test_slow_prolonged_n2_accepts_normal():
    """With only 2 marks, 0.5-grid quantisation can push slow_prolonged onto
    the alert line boundary, yielding NORMAL.  The scorer should accept this."""
    v = score_entry(
        category="partial", curve_type="slow_prolonged", n_marks=2,
        actual_status=ZoneStatus.NORMAL, actual_n_points=2,
    )
    assert v.correct


def test_slow_prolonged_n2_still_accepts_alert():
    v = score_entry(
        category="partial", curve_type="slow_prolonged", n_marks=2,
        actual_status=ZoneStatus.ALERT_ZONE, actual_n_points=2,
    )
    assert v.correct


def test_slow_prolonged_n2_still_accepts_action():
    v = score_entry(
        category="partial", curve_type="slow_prolonged", n_marks=2,
        actual_status=ZoneStatus.ACTION_ZONE, actual_n_points=2,
    )
    assert v.correct


def test_slow_prolonged_n3_rejects_normal():
    """With 3+ marks, the relaxation must NOT apply."""
    v = score_entry(
        category="partial", curve_type="slow_prolonged", n_marks=3,
        actual_status=ZoneStatus.NORMAL, actual_n_points=3,
    )
    assert not v.correct


def test_arrested_n2_accepts_normal():
    v = score_entry(
        category="partial", curve_type="arrested", n_marks=2,
        actual_status=ZoneStatus.NORMAL, actual_n_points=2,
    )
    assert v.correct


def test_arrested_n3_rejects_normal():
    v = score_entry(
        category="filled", curve_type="arrested", n_marks=3,
        actual_status=ZoneStatus.NORMAL, actual_n_points=3,
    )
    assert not v.correct


def test_rapid_n2_does_not_get_relaxation():
    """Rapid/precipitous should NOT benefit from sparse observation tolerance."""
    v = score_entry(
        category="partial", curve_type="rapid_precipitous", n_marks=2,
        actual_status=ZoneStatus.ALERT_ZONE, actual_n_points=2,
    )
    assert not v.correct


def test_normal_n2_does_not_get_relaxation():
    """Normal curves should NOT benefit from sparse observation tolerance."""
    v = score_entry(
        category="partial", curve_type="normal", n_marks=2,
        actual_status=ZoneStatus.ACTION_ZONE, actual_n_points=2,
    )
    assert not v.correct
    v = score_entry(
        category="filled", curve_type="rapid_precipitous", n_marks=5,
        actual_status=ZoneStatus.NORMAL, actual_n_points=5,
    )
    assert v.correct
    v2 = score_entry(
        category="filled", curve_type="rapid_precipitous", n_marks=5,
        actual_status=ZoneStatus.ALERT_ZONE, actual_n_points=5,
    )
    assert not v2.correct

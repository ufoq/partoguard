"""Ground-truth correctness scorer for the synthetic partograph corpus.

Given a manifest entry (with `category`, `curve_type`, `n_marks`) and the
pipeline's actual output (`ZoneStatus`, predicted point count), decides whether
the pipeline was correct for that image. Used by `evaluate_corpus_dir` for
production correctness measurement and by `scripts/probe_correctness.py` for
prompt/model iteration.

Strictness rules:

- `n_marks == 0` (blank or empty): the pipeline MUST return MANUAL_REVIEW with
  zero predicted points. Any zone classification is a hallucination failure.
- `n_marks == 1`: the rule engine cannot reliably classify a single point so
  either MANUAL_REVIEW or any single zone is accepted. Predicting more than
  1 point is a hallucination failure.
- `n_marks >= 2`: the pipeline MUST classify a zone within the curve_type's
  acceptable set AND predicted point count must be within tolerance of the
  truth. Tolerance for `filled`/`partial`/`degraded` is symmetric. Tolerance
  for `obstructed` is one-sided (under-counting is allowed because the model
  may correctly refuse to guess marks hidden by obstructions, as long as the
  visible-mark zone is in the acceptable set).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from partoguard.core.schemas.contracts import ZoneStatus

_CURVE_ZONES: Final[dict[str, frozenset[ZoneStatus]]] = {
    "normal": frozenset({ZoneStatus.NORMAL, ZoneStatus.ALERT_ZONE}),
    "slow_prolonged": frozenset({ZoneStatus.ALERT_ZONE, ZoneStatus.ACTION_ZONE}),
    "arrested": frozenset({ZoneStatus.ALERT_ZONE, ZoneStatus.ACTION_ZONE}),
    "rapid_precipitous": frozenset({ZoneStatus.NORMAL}),
    "none": frozenset(),
}

# With only 2 marks near the active-phase start, 0.5-grid quantisation can
# push a borderline slow/arrested trajectory onto the alert line, yielding
# NORMAL.  Accept NORMAL as a valid alternative for these sparse observations.
_SPARSE_OBSERVATION_EXTRA_ZONES: Final[dict[str, frozenset[ZoneStatus]]] = {
    "slow_prolonged": frozenset({ZoneStatus.NORMAL}),
    "arrested": frozenset({ZoneStatus.NORMAL}),
}
_SPARSE_MARK_THRESHOLD: Final[int] = 2

_ANY_ZONE: Final[frozenset[ZoneStatus]] = frozenset(
    {ZoneStatus.NORMAL, ZoneStatus.ALERT_ZONE, ZoneStatus.ACTION_ZONE}
)


@dataclass(frozen=True)
class ScoreVerdict:
    correct: bool
    reason: str
    expected_kind: str
    acceptable_statuses: frozenset[ZoneStatus]
    count_tolerance: int


def _count_tolerance(n_marks: int) -> int:
    return max(2, math.ceil(n_marks * 0.4))


def score_entry(
    *,
    category: str,
    curve_type: str,
    n_marks: int,
    actual_status: ZoneStatus,
    actual_n_points: int,
) -> ScoreVerdict:
    if n_marks == 0:
        ok = actual_status == ZoneStatus.MANUAL_REVIEW and actual_n_points == 0
        reason = (
            "blank or empty: expected manual_review with zero points"
            if not ok
            else "ok: manual_review on empty"
        )
        return ScoreVerdict(
            correct=ok,
            reason=reason,
            expected_kind="must_manual_empty",
            acceptable_statuses=frozenset({ZoneStatus.MANUAL_REVIEW}),
            count_tolerance=0,
        )

    if n_marks == 1:
        if actual_n_points > 1:
            return ScoreVerdict(
                correct=False,
                reason=f"hallucination: {actual_n_points} predicted vs 1 truth",
                expected_kind="single_mark",
                acceptable_statuses=_ANY_ZONE | {ZoneStatus.MANUAL_REVIEW},
                count_tolerance=1,
            )
        return ScoreVerdict(
            correct=True,
            reason="ok: single-mark image, any zone or manual_review accepted",
            expected_kind="single_mark",
            acceptable_statuses=_ANY_ZONE | {ZoneStatus.MANUAL_REVIEW},
            count_tolerance=1,
        )

    tol = _count_tolerance(n_marks)
    zone_set = _CURVE_ZONES.get(curve_type, _ANY_ZONE)
    if n_marks <= _SPARSE_MARK_THRESHOLD:
        zone_set = zone_set | _SPARSE_OBSERVATION_EXTRA_ZONES.get(curve_type, frozenset())

    pred = actual_n_points
    overcount = pred - n_marks
    if overcount > tol:
        return ScoreVerdict(
            correct=False,
            reason=f"over-count: predicted {pred} vs truth {n_marks} (tol +{tol})",
            expected_kind="must_zone_with_count",
            acceptable_statuses=zone_set,
            count_tolerance=tol,
        )
    if category != "obstructed" and overcount < -tol:
        return ScoreVerdict(
            correct=False,
            reason=f"under-count: predicted {pred} vs truth {n_marks} (tol -{tol})",
            expected_kind="must_zone_with_count",
            acceptable_statuses=zone_set,
            count_tolerance=tol,
        )

    if actual_status == ZoneStatus.MANUAL_REVIEW:
        return ScoreVerdict(
            correct=False,
            reason=f"unexpected manual_review: truth has {n_marks} marks",
            expected_kind="must_zone_with_count",
            acceptable_statuses=zone_set,
            count_tolerance=tol,
        )

    if actual_status not in zone_set:
        return ScoreVerdict(
            correct=False,
            reason=(
                f"zone {actual_status.value} not in expected "
                f"{sorted(s.value for s in zone_set)} for curve_type={curve_type}"
            ),
            expected_kind="must_zone_with_count",
            acceptable_statuses=zone_set,
            count_tolerance=tol,
        )

    return ScoreVerdict(
        correct=True,
        reason=f"ok: pred={pred} (truth={n_marks},tol={tol}) zone={actual_status.value}",
        expected_kind="must_zone_with_count",
        acceptable_statuses=zone_set,
        count_tolerance=tol,
    )


def score_manifest_entry(
    entry: dict[str, object],
    *,
    actual_status: ZoneStatus,
    actual_n_points: int,
) -> ScoreVerdict:
    category = str(entry.get("category", ""))
    curve_type = str(entry.get("curve_type", "none"))
    n_marks_raw = entry.get("n_marks", 0)
    n_marks = int(n_marks_raw) if isinstance(n_marks_raw, (int, float)) else 0
    return score_entry(
        category=category,
        curve_type=curve_type,
        n_marks=n_marks,
        actual_status=actual_status,
        actual_n_points=actual_n_points,
    )


__all__ = ["ScoreVerdict", "score_entry", "score_manifest_entry"]

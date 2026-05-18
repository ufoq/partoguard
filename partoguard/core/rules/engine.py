"""Deterministic rule engine for modified WHO partograph zoning."""

from __future__ import annotations

from typing import Final

from partoguard.core.schemas.contracts import DilationPoint, RuleOutput, ZoneStatus

ACTIVE_PHASE_DILATION_CM: Final[float] = 4.0
ACTION_LINE_START_HOURS: Final[float] = 4.0
MAX_DILATION_CM: Final[float] = 10.0
SUPPORTED_FRAMEWORK: Final[str] = "modified_who_partograph"
UNCERTAINTY_CM: Final[float] = 0.5


def classify_zone(
    points: list[DilationPoint],
    framework: str = SUPPORTED_FRAMEWORK,
    uncertainty_cm: float = UNCERTAINTY_CM,
) -> RuleOutput:
    """Classify dilation progress against modified WHO alert/action lines.

    Exact boundary points follow the inclusive geometric definitions of the
    partograph: points on the alert line are treated as normal, and points on
    the action line are treated as action-zone. A non-zero offset within the
    uncertainty band is treated as indeterminate.
    """

    if framework != SUPPORTED_FRAMEWORK:
        return _manual_review(
            framework=framework,
            explanation=(
                f"Framework '{framework}' is not supported for deterministic review; "
                "review manually."
            ),
        )

    if uncertainty_cm < 0:
        return _manual_review(
            framework=framework,
            explanation="Uncertainty band must be non-negative; review manually.",
        )

    if not points:
        return _manual_review(
            framework=framework,
            explanation="No dilation points were provided; review manually.",
        )

    for point in points:
        if point.x_hours < 0 or point.dilation_cm < 0 or point.dilation_cm > MAX_DILATION_CM:
            return _manual_review(
                framework=framework,
                explanation=(
                    f"Point at {_format_point(point)} is outside the supported range; "
                    "review manually."
                ),
                triggering_point=point,
                confidence=point.confidence,
            )

    active_points = [point for point in points if point.dilation_cm >= ACTIVE_PHASE_DILATION_CM]
    if not active_points:
        return _manual_review(
            framework=framework,
            explanation="No active-phase dilation points were available; review manually.",
        )

    statuses: dict[ZoneStatus, list[DilationPoint]] = {
        ZoneStatus.NORMAL: [],
        ZoneStatus.ALERT_ZONE: [],
        ZoneStatus.ACTION_ZONE: [],
        ZoneStatus.INDETERMINATE: [],
    }

    for point in active_points:
        status = _classify_point(point, uncertainty_cm)
        statuses[status].append(point)

    if statuses[ZoneStatus.ACTION_ZONE]:
        return _build_output(
            status=ZoneStatus.ACTION_ZONE,
            framework=framework,
            points=statuses[ZoneStatus.ACTION_ZONE],
            explanation_template=(
                "Point at {point} is on or beyond the action threshold; "
                "review and escalate per protocol."
            ),
        )

    if statuses[ZoneStatus.INDETERMINATE]:
        return _build_output(
            status=ZoneStatus.INDETERMINATE,
            framework=framework,
            points=statuses[ZoneStatus.INDETERMINATE],
            explanation_template=(
                "Point at {point} lies within {uncertainty:g} cm of a decision boundary; "
                "review manually and escalate per protocol."
            ),
            uncertainty_cm=uncertainty_cm,
            requires_human_review=True,
        )

    if statuses[ZoneStatus.ALERT_ZONE]:
        return _build_output(
            status=ZoneStatus.ALERT_ZONE,
            framework=framework,
            points=statuses[ZoneStatus.ALERT_ZONE],
            explanation_template=(
                "Point at {point} falls between the alert and action lines; "
                "review and escalate per protocol."
            ),
        )

    return _build_output(
        status=ZoneStatus.NORMAL,
        framework=framework,
        points=statuses[ZoneStatus.NORMAL],
        explanation_template=(
            "Point at {point} remains on or left of the alert trajectory; "
            "continue routine review per protocol."
        ),
    )


def _classify_point(point: DilationPoint, uncertainty_cm: float) -> ZoneStatus:
    alert_dilation = ACTIVE_PHASE_DILATION_CM + point.x_hours
    alert_gap = point.dilation_cm - alert_dilation

    if _within_uncertainty(alert_gap, uncertainty_cm):
        return ZoneStatus.INDETERMINATE

    if point.x_hours < ACTION_LINE_START_HOURS:
        if point.dilation_cm < alert_dilation:
            return ZoneStatus.ALERT_ZONE
        return ZoneStatus.NORMAL

    action_dilation = point.x_hours
    action_gap = point.dilation_cm - action_dilation

    if _within_uncertainty(action_gap, uncertainty_cm):
        return ZoneStatus.INDETERMINATE

    if point.dilation_cm <= action_dilation:
        return ZoneStatus.ACTION_ZONE
    if point.dilation_cm < alert_dilation:
        return ZoneStatus.ALERT_ZONE
    return ZoneStatus.NORMAL


def _within_uncertainty(distance_cm: float, uncertainty_cm: float) -> bool:
    return 0.0 < abs(distance_cm) <= uncertainty_cm


def _build_output(
    *,
    status: ZoneStatus,
    framework: str,
    points: list[DilationPoint],
    explanation_template: str,
    uncertainty_cm: float | None = None,
    requires_human_review: bool = False,
) -> RuleOutput:
    triggering_point = points[0]
    confidence = min(point.confidence for point in points)
    explanation = explanation_template.format(
        point=_format_point(triggering_point),
        uncertainty=uncertainty_cm if uncertainty_cm is not None else UNCERTAINTY_CM,
    )
    return RuleOutput(
        status=status,
        framework=framework,
        triggering_point=triggering_point,
        explanation=explanation,
        confidence=confidence,
        requires_human_review=requires_human_review,
    )


def _manual_review(
    *,
    framework: str,
    explanation: str,
    triggering_point: DilationPoint | None = None,
    confidence: float = 0.0,
) -> RuleOutput:
    return RuleOutput(
        status=ZoneStatus.MANUAL_REVIEW,
        framework=framework,
        triggering_point=triggering_point,
        explanation=explanation,
        confidence=confidence,
        requires_human_review=True,
    )


def _format_point(point: DilationPoint) -> str:
    return f"({point.x_hours:g}h, {point.dilation_cm:g}cm)"


__all__ = ["UNCERTAINTY_CM", "classify_zone"]

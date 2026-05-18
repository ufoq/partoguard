package com.partoguard.app.rules

import com.partoguard.app.model.ClinicalAlert
import com.partoguard.app.model.ClinicalStatus
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.PartographPoint

/**
 * Deterministic, auditable partograph rule engine. NOT a model.
 *
 * The model (Gemma) extracts points. THIS class decides clinical status.
 * Order of checks matters and is intentionally simple so it can be reviewed
 * by a midwife/obstetrician without reading ML code.
 *
 * Reference rules (simplified for hackathon demo):
 *  - Alert line: from (0h, 4cm) at 1 cm/hr.
 *  - Action line: alert line shifted +4 hours to the right.
 *  - A point ON or to the RIGHT of the alert line is "alert_zone".
 *  - A point ON or to the RIGHT of the action line is "action_zone".
 *  - No points -> "empty".
 *  - needsManualReview from extraction OR <2 points -> "manual_review".
 *
 * In production these thresholds must be reviewed against WHO partograph
 * guidance; current values are demonstration defaults only.
 */
object RuleEngine {

    private const val ALERT_SLOPE_CM_PER_HR = 1.0f
    private const val ALERT_INTERCEPT_CM = 4.0f
    private const val ACTION_SHIFT_HOURS = 4.0f

    fun evaluate(extraction: PartographExtraction): ClinicalAlert {
        if (extraction.needsManualReview) {
            return ClinicalAlert(
                status = ClinicalStatus.MANUAL_REVIEW,
                headline = "Manual review required.",
                reason = extraction.reasonForManualReview ?: "Model is uncertain. A midwife must enter or correct the points before rules can run.",
            )
        }

        if (extraction.points.isEmpty()) {
            return ClinicalAlert(
                status = ClinicalStatus.EMPTY,
                headline = "No marks detected on the cervicograph.",
                reason = "The model did not find any plotted X marks. Confirm the chart is in frame and recapture.",
            )
        }

        val latest = extraction.points.maxBy { it.xHours }
        val actionThresholdCm = ALERT_INTERCEPT_CM + ALERT_SLOPE_CM_PER_HR * (latest.xHours - ACTION_SHIFT_HOURS)
        val alertThresholdCm = ALERT_INTERCEPT_CM + ALERT_SLOPE_CM_PER_HR * latest.xHours

        return when {
            // Below the action line for its hour => action zone.
            latest.dilationCm <= actionThresholdCm && latest.xHours >= ACTION_SHIFT_HOURS ->
                ClinicalAlert(
                    status = ClinicalStatus.ACTION_ZONE,
                    headline = "Action zone — escalate.",
                    reason = "Latest plotted point (${fmt(latest)}) sits to the right of the action line. Local protocol may require obstetric review or transfer.",
                    triggeringPoint = latest,
                )
            // Below the alert line => alert zone.
            latest.dilationCm <= alertThresholdCm ->
                ClinicalAlert(
                    status = ClinicalStatus.ALERT_ZONE,
                    headline = "Alert zone crossed.",
                    reason = "Latest plotted point (${fmt(latest)}) is at or right of the alert line. Increase monitoring and review labour progress.",
                    triggeringPoint = latest,
                )
            else ->
                ClinicalAlert(
                    status = ClinicalStatus.NORMAL,
                    headline = "Normal progression.",
                    reason = "All plotted points stay left of the alert line.",
                )
        }
    }

    private fun fmt(p: PartographPoint): String =
        "${"%.1f".format(p.xHours)} h, ${"%.1f".format(p.dilationCm)} cm"
}

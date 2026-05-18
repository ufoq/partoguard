package com.partoguard.app.model

/** A single cervical dilation point plotted on the partograph cervicograph. */
data class PartographPoint(
    val xHours: Float,
    val dilationCm: Float,
    val confidence: Float,
)

/** Subjective image quality flags produced by the preprocessing stage. */
data class ImageQuality(
    val blurry: Boolean = false,
    val dim: Boolean = false,
    val skewed: Boolean = false,
) {
    val isGood: Boolean get() = !blurry && !dim && !skewed
}

/** Outcome of running the deterministic clinical rule engine. */
enum class ClinicalStatus { NORMAL, ALERT_ZONE, ACTION_ZONE, MANUAL_REVIEW, EMPTY }

/** What the user is told. Severity drives colour and tone. */
data class ClinicalAlert(
    val status: ClinicalStatus,
    val headline: String,
    val reason: String,
    val triggeringPoint: PartographPoint? = null,
)

/**
 * Strict extraction payload — matches the JSON contract the on-device model
 * (currently mocked) is expected to return. Do not let downstream code read
 * raw model output; it always passes through this typed object.
 */
data class PartographExtraction(
    val chartSupported: Boolean,
    val imageQuality: ImageQuality,
    val points: List<PartographPoint>,
    val needsManualReview: Boolean,
    val reasonForManualReview: String? = null,
)

/** Full pipeline result handed to the Review/Results screens. */
data class PartographAnalysis(
    val sourceLabel: String,
    val extraction: PartographExtraction,
    val alert: ClinicalAlert,
)

/** UI-selectable workflow mode (per architecture brief). */
enum class WorkflowMode { AUTO, ASSISTED, MANUAL }

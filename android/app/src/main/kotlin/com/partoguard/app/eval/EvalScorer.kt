package com.partoguard.app.eval

import com.partoguard.app.model.ClinicalStatus
import kotlin.math.ceil
import kotlin.math.max

/**
 * Kotlin port of partoguard/core/corpus_scorer.py — keeps Android eval scoring
 * in lockstep with the Python ground-truth scorer. Read the Python file when
 * changing logic; both must move together.
 *
 * One Android-specific divergence: blank images are accepted as either
 * ClinicalStatus.EMPTY (RuleEngine path) or ClinicalStatus.MANUAL_REVIEW
 * (Python pipeline emits MANUAL_REVIEW; Android RuleEngine emits EMPTY).
 */
data class EvalManifestEntry(
    val path: String,
    val category: String,
    val curveType: String,
    val nMarks: Int,
)

data class ScoreVerdict(
    val correct: Boolean,
    val reason: String,
    val expectedKind: String,
    val acceptableStatuses: Set<ClinicalStatus>,
    val countTolerance: Int,
)

object EvalScorer {

    private val ANY_ZONE: Set<ClinicalStatus> = setOf(
        ClinicalStatus.NORMAL,
        ClinicalStatus.ALERT_ZONE,
        ClinicalStatus.ACTION_ZONE,
    )

    private val CURVE_ZONES: Map<String, Set<ClinicalStatus>> = mapOf(
        "normal"            to setOf(ClinicalStatus.NORMAL, ClinicalStatus.ALERT_ZONE),
        "slow_prolonged"    to setOf(ClinicalStatus.ALERT_ZONE, ClinicalStatus.ACTION_ZONE),
        "arrested"          to setOf(ClinicalStatus.ALERT_ZONE, ClinicalStatus.ACTION_ZONE),
        "rapid_precipitous" to setOf(ClinicalStatus.NORMAL),
        "none"              to emptySet(),
    )

    private val SPARSE_OBSERVATION_EXTRA_ZONES: Map<String, Set<ClinicalStatus>> = mapOf(
        "slow_prolonged" to setOf(ClinicalStatus.NORMAL),
        "arrested"       to setOf(ClinicalStatus.NORMAL),
    )

    private const val SPARSE_MARK_THRESHOLD = 2

    private fun countTolerance(nMarks: Int): Int = max(2, ceil(nMarks * 0.4).toInt())

    fun score(
        entry: EvalManifestEntry,
        actualStatus: ClinicalStatus,
        actualNPoints: Int,
    ): ScoreVerdict {
        if (entry.nMarks == 0) {
            val acceptable = setOf(ClinicalStatus.MANUAL_REVIEW, ClinicalStatus.EMPTY)
            val ok = actualStatus in acceptable && actualNPoints == 0
            return ScoreVerdict(
                correct = ok,
                reason = if (ok) "ok: empty/manual_review on blank"
                         else "blank: expected EMPTY or MANUAL_REVIEW with zero points, got $actualStatus n=$actualNPoints",
                expectedKind = "must_manual_empty",
                acceptableStatuses = acceptable,
                countTolerance = 0,
            )
        }

        if (entry.nMarks == 1) {
            val acceptable = ANY_ZONE + setOf(ClinicalStatus.MANUAL_REVIEW, ClinicalStatus.EMPTY)
            if (actualNPoints > 1) {
                return ScoreVerdict(
                    correct = false,
                    reason = "hallucination: $actualNPoints predicted vs 1 truth",
                    expectedKind = "single_mark",
                    acceptableStatuses = acceptable,
                    countTolerance = 1,
                )
            }
            return ScoreVerdict(
                correct = true,
                reason = "ok: single-mark image, any zone or manual_review accepted",
                expectedKind = "single_mark",
                acceptableStatuses = acceptable,
                countTolerance = 1,
            )
        }

        val tol = countTolerance(entry.nMarks)
        var zoneSet = CURVE_ZONES[entry.curveType] ?: ANY_ZONE
        if (entry.nMarks <= SPARSE_MARK_THRESHOLD) {
            zoneSet = zoneSet + (SPARSE_OBSERVATION_EXTRA_ZONES[entry.curveType] ?: emptySet())
        }

        val overcount = actualNPoints - entry.nMarks
        if (overcount > tol) {
            return ScoreVerdict(
                correct = false,
                reason = "over-count: predicted $actualNPoints vs truth ${entry.nMarks} (tol +$tol)",
                expectedKind = "must_zone_with_count",
                acceptableStatuses = zoneSet,
                countTolerance = tol,
            )
        }
        if (entry.category != "obstructed" && overcount < -tol) {
            return ScoreVerdict(
                correct = false,
                reason = "under-count: predicted $actualNPoints vs truth ${entry.nMarks} (tol -$tol)",
                expectedKind = "must_zone_with_count",
                acceptableStatuses = zoneSet,
                countTolerance = tol,
            )
        }

        if (actualStatus == ClinicalStatus.MANUAL_REVIEW) {
            return ScoreVerdict(
                correct = false,
                reason = "unexpected manual_review: truth has ${entry.nMarks} marks",
                expectedKind = "must_zone_with_count",
                acceptableStatuses = zoneSet,
                countTolerance = tol,
            )
        }

        if (actualStatus !in zoneSet) {
            return ScoreVerdict(
                correct = false,
                reason = "zone $actualStatus not in expected $zoneSet for curve_type=${entry.curveType}",
                expectedKind = "must_zone_with_count",
                acceptableStatuses = zoneSet,
                countTolerance = tol,
            )
        }

        return ScoreVerdict(
            correct = true,
            reason = "ok: pred=$actualNPoints (truth=${entry.nMarks},tol=$tol) zone=$actualStatus",
            expectedKind = "must_zone_with_count",
            acceptableStatuses = zoneSet,
            countTolerance = tol,
        )
    }
}

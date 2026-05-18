package com.partoguard.app.analyzer

import android.graphics.Bitmap
import com.partoguard.app.model.ImageQuality
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.PartographPoint
import com.partoguard.app.session.AnalysisSession
import kotlinx.coroutines.delay

/**
 * Canned-result extractor used while the real Gemma 4 LiteRT model is not yet
 * integrated. Returns one of four hand-picked outcomes selected by source label
 * hash so the same demo image always produces the same result.
 *
 * The optional [session] enables a debug override: if [AnalysisSession.forcedOutcome]
 * is set, the next extraction uses that outcome and the override is cleared.
 *
 * Coordinate conventions: xHours in [0,12], dilationCm in [0,10].
 * The "alert" curve crosses the alert line (1 cm/hr from 4cm) and the "action"
 * curve crosses the action line (alert + 4 hours).
 */
class MockPartographExtractor(
    private val session: AnalysisSession? = null,
) : PartographExtractor {
    override suspend fun extract(bitmap: Bitmap, sourceLabel: String): PartographExtraction {
        delay(MOCK_LATENCY_MS)
        val outcome = consumeForcedOutcome() ?: pickOutcome(sourceLabel)
        return when (outcome) {
            Outcome.NORMAL -> PartographExtraction(
                chartSupported = true,
                imageQuality = ImageQuality(),
                points = listOf(
                    PartographPoint(0f, 4f, 0.94f),
                    PartographPoint(1f, 5.5f, 0.93f),
                    PartographPoint(2f, 7f, 0.92f),
                    PartographPoint(3f, 8.5f, 0.91f),
                    PartographPoint(4f, 10f, 0.90f),
                ),
                needsManualReview = false,
            )
            Outcome.ALERT -> PartographExtraction(
                chartSupported = true,
                imageQuality = ImageQuality(dim = true),
                // Latest point (6h, 8.2cm) lies right of the alert line
                // (alertThresh=10cm at h=6) but left of the action line
                // (actionThresh=6cm at h=6) -> genuine ALERT_ZONE.
                points = listOf(
                    PartographPoint(0f, 4f, 0.92f),
                    PartographPoint(1f, 4.7f, 0.90f),
                    PartographPoint(2f, 5.4f, 0.88f),
                    PartographPoint(3f, 6.1f, 0.87f),
                    PartographPoint(4f, 6.8f, 0.86f),
                    PartographPoint(5f, 7.5f, 0.84f),
                    PartographPoint(6f, 8.2f, 0.82f),
                ),
                needsManualReview = false,
            )
            Outcome.ACTION -> PartographExtraction(
                chartSupported = true,
                imageQuality = ImageQuality(skewed = true),
                points = listOf(
                    PartographPoint(0f, 4f, 0.93f),
                    PartographPoint(2f, 4f, 0.90f),
                    PartographPoint(4f, 4.5f, 0.88f),
                    PartographPoint(6f, 4.5f, 0.85f),
                    PartographPoint(8f, 5f, 0.82f),
                    PartographPoint(10f, 5f, 0.78f),
                    PartographPoint(12f, 5.5f, 0.76f),
                ),
                needsManualReview = false,
            )
            
            Outcome.PT1 -> PartographExtraction(
                chartSupported = true,
                imageQuality = ImageQuality(blurry = true),
                points = listOf(
                    PartographPoint(0f, 4f, 0.95f),
                    PartographPoint(1f, 5f, 0.94f),
                    PartographPoint(3f, 6f, 0.93f),
                    PartographPoint(4f, 7f, 0.92f),
                    PartographPoint(5f, 8f, 0.91f),
                    PartographPoint(7f, 9f, 0.90f),
                    PartographPoint(8f, 10f, 0.89f),
                ),
                needsManualReview = false,
            )
            Outcome.EMPTY -> PartographExtraction(

                chartSupported = true,
                imageQuality = ImageQuality(),
                points = emptyList(),
                needsManualReview = false,
            )
        }
    }

    private fun consumeForcedOutcome(): Outcome? {
        val s = session ?: return null
        val name = s.forcedOutcome ?: return null
        s.forcedOutcome = null
        return Outcome.entries.firstOrNull { it.name == name }
    }

    
    enum class Outcome { NORMAL, ALERT, ACTION, EMPTY, PT1 }


    private fun pickOutcome(sourceLabel: String): Outcome {
        val lower = sourceLabel.lowercase()
        return when {
            
            "pt1" in lower -> Outcome.PT1
            "blank" in lower -> Outcome.EMPTY

            "alert" in lower -> Outcome.ALERT
            "action" in lower || "obstructed" in lower -> Outcome.ACTION
            "normal" in lower || "filled" in lower -> Outcome.NORMAL
            else -> Outcome.entries[(sourceLabel.hashCode() and 0x7fffffff) % Outcome.entries.size]
        }
    }

    companion object {
        private const val MOCK_LATENCY_MS = 1500L
    }
}

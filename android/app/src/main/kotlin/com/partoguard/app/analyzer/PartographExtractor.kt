package com.partoguard.app.analyzer

import android.graphics.Bitmap
import com.partoguard.app.model.PartographExtraction

/**
 * The swap point for real ML.
 *
 * When the fine-tuned Gemma 4 E2B-it model is exported to LiteRT-LM (see
 * knowledge/litert_export_setup.md in the parent repo), drop in a
 * LiteRtPartographExtractor that loads the .litertlm bundle from
 * `assets/model/partoguard-gemma4-e2b-it.litertlm` and returns a
 * [PartographExtraction] parsed from the strict JSON contract.
 *
 * Implementations must:
 *  - Never throw — return a [PartographExtraction] with needsManualReview=true
 *    on any failure (invalid JSON, low confidence, model error).
 *  - Be safe to call from the main thread (do their own dispatching).
 *  - Be deterministic given the same input bitmap.
 */
interface PartographExtractor {
    suspend fun extract(bitmap: Bitmap, sourceLabel: String): PartographExtraction
}

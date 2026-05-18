package com.partoguard.app.session

import android.graphics.Bitmap
import com.partoguard.app.model.ClinicalAlert
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.WorkflowMode

/**
 * Process-singleton holding the active analysis between screens. Compose nav
 * doesn't pass Bitmaps well as args, so the relevant pieces live here keyed by
 * a short string slot. This is intentionally simple — no DI, no Room.
 *
 * When real ML lands, persist a structured audit log here (off-device only
 * after explicit clinician export, per the privacy stance in
 * knowledge/partoguard_corpus_plan.md).
 */
class AnalysisSession {
    var sourceLabel: String = ""
    var bitmap: Bitmap? = null
    var extraction: PartographExtraction? = null
    var alert: ClinicalAlert? = null
    var mode: WorkflowMode = WorkflowMode.AUTO

    /**
     * Debug-only override for the mock extractor. When non-null, the next
     * extraction returns this outcome regardless of source label, then this
     * field is cleared. Set from the hidden debug panel on HomeScreen
     * (7 taps on the Automatic chip within 5s). Values: "NORMAL", "ALERT",
     * "ACTION", "EMPTY". Null = use default keyword/hash routing.
     */
    var forcedOutcome: String? = null

    fun reset() {
        sourceLabel = ""
        bitmap = null
        extraction = null
        alert = null
    }
}

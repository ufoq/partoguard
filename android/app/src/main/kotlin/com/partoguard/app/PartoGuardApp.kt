package com.partoguard.app

import android.app.Application
import com.partoguard.app.session.AnalysisSession

class PartoGuardApp : Application() {

    val session: AnalysisSession by lazy { AnalysisSession() }

    companion object {
        const val GEMMA_SERVER_BASE_URL = "http://163.172.52.31:8080"
    }
}

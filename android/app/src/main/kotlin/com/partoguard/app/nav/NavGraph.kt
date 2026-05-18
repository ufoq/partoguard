package com.partoguard.app.nav

import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.partoguard.app.analyzer.ExtractorSelector
import com.partoguard.app.model.ImageQuality
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.WorkflowMode
import com.partoguard.app.session.AnalysisSession
import com.partoguard.app.ui.analyzing.AnalyzingScreen
import com.partoguard.app.ui.camera.CameraScreen
import com.partoguard.app.ui.home.HomeScreen
import com.partoguard.app.ui.results.ResultsScreen
import com.partoguard.app.ui.review.ReviewScreen
import com.partoguard.app.ui.settings.SettingsScreen

private object Routes {
    const val HOME = "home"
    const val CAMERA = "camera"
    const val ANALYZING = "analyzing"
    const val REVIEW = "review"
    const val RESULTS = "results"
    const val SETTINGS = "settings"
}

@Composable
fun PartoGuardNavGraph(
    session: AnalysisSession,
) {
    val nav: NavHostController = rememberNavController()
    val context = LocalContext.current
    val startRoute = if (ExtractorSelector.isFirstLaunch(context)) Routes.SETTINGS else Routes.HOME

    NavHost(navController = nav, startDestination = startRoute) {
        composable(Routes.HOME) {
            HomeScreen(
                session = session,
                onPickDemo = { nav.navigateAfterCapture(session.mode, session) },
                onLiveCamera = {
                    if (session.mode == WorkflowMode.MANUAL) {
                        nav.navigateAfterCapture(session.mode, session)
                    } else {
                        nav.navigate(Routes.CAMERA)
                    }
                },
                onSettings = { nav.navigate(Routes.SETTINGS) },
            )
        }
        composable(Routes.CAMERA) {
            CameraScreen(
                session = session,
                onCaptured = { nav.navigateAfterCapture(session.mode, session) },
                onBack = { nav.popBackStack() },
            )
        }
        composable(Routes.ANALYZING) {
            val context = LocalContext.current
            AnalyzingScreen(
                session = session,
                extractor = ExtractorSelector.pick(context),
                onReview = {
                    nav.navigate(Routes.REVIEW) {
                        popUpTo(Routes.HOME)
                    }
                },
                onResults = {
                    nav.navigate(Routes.RESULTS) {
                        popUpTo(Routes.HOME)
                    }
                },
            )
        }
        composable(Routes.REVIEW) {
            ReviewScreen(
                session = session,
                onConfirmed = {
                    nav.navigate(Routes.RESULTS) {
                        popUpTo(Routes.HOME)
                    }
                },
                onBack = { nav.popBackStack() },
            )
        }
        composable(Routes.RESULTS) {
            ResultsScreen(
                session = session,
                onAnalyzeAnother = {
                    session.reset()
                    nav.popBackStack(Routes.HOME, inclusive = false)
                },
            )
        }
        composable(Routes.SETTINGS) {
            val ctx = LocalContext.current
            val firstLaunch = ExtractorSelector.isFirstLaunch(ctx)
            SettingsScreen(
                onBack = { nav.popBackStack() },
                isFirstLaunch = firstLaunch,
                onSetupComplete = {
                    nav.navigate(Routes.HOME) {
                        popUpTo(Routes.SETTINGS) { inclusive = true }
                    }
                },
            )
        }
    }
}

// AUTO: extract -> Results (rules in Analyzing).
// ASSISTED: extract -> Review -> Results.
// MANUAL: skip extract; seed empty extraction so ReviewScreen starts blank.
private fun NavHostController.navigateAfterCapture(
    mode: WorkflowMode,
    session: AnalysisSession,
) {
    when (mode) {
        WorkflowMode.AUTO, WorkflowMode.ASSISTED -> navigate(Routes.ANALYZING)
        WorkflowMode.MANUAL -> {
            seedEmptyExtraction(session)
            navigate(Routes.REVIEW)
        }
    }
}

private fun seedEmptyExtraction(session: AnalysisSession) {
    if (session.sourceLabel.isBlank()) {
        session.sourceLabel = "manual_${System.currentTimeMillis()}"
    }
    session.extraction = PartographExtraction(
        chartSupported = true,
        imageQuality = ImageQuality(),
        points = emptyList(),
        needsManualReview = false,
    )
}

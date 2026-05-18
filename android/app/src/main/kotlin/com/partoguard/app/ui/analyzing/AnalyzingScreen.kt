package com.partoguard.app.ui.analyzing

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.partoguard.app.R
import com.partoguard.app.analyzer.PartographExtractor
import com.partoguard.app.model.WorkflowMode
import com.partoguard.app.preprocess.ImageQualityChecker
import com.partoguard.app.rules.RuleEngine
import com.partoguard.app.session.AnalysisSession
import kotlinx.coroutines.delay

@Composable
fun AnalyzingScreen(
    session: AnalysisSession,
    extractor: PartographExtractor,
    onReview: () -> Unit,
    onResults: () -> Unit,
) {
    val bitmap = session.bitmap
    var stepIndex by remember { mutableIntStateOf(0) }
    val steps = listOf(
        stringResource(R.string.analyzing_step_quality),
        stringResource(R.string.analyzing_step_extract),
        stringResource(R.string.analyzing_step_validate),
    )

    LaunchedEffect(bitmap) {
        if (bitmap == null) {
            onReview()
            return@LaunchedEffect
        }
        stepIndex = 0
        ImageQualityChecker.check(bitmap, session.sourceLabel)
        delay(STEP_DELAY_MS)
        stepIndex = 1
        val extraction = extractor.extract(bitmap, session.sourceLabel)
        session.extraction = extraction
        stepIndex = 2
        delay(STEP_DELAY_MS)
        // AUTO: rules run here, jump straight to Results (no manual confirm).
        // ASSISTED: midwife must confirm/edit in ReviewScreen before rules.
        if (session.mode == WorkflowMode.AUTO) {
            session.alert = RuleEngine.evaluate(extraction)
            onResults()
        } else {
            onReview()
        }
    }

    Scaffold(contentWindowInsets = WindowInsets(0, 0, 0, 0)) {
        Column(
            Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp, vertical = 32.dp)
                .windowInsetsPadding(WindowInsets.safeDrawing),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                stringResource(R.string.analyzing_title),
                style = MaterialTheme.typography.headlineSmall,
            )
            Spacer(Modifier.height(24.dp))
            Spinner()
            Spacer(Modifier.height(32.dp))
            steps.forEachIndexed { idx, label ->
                StepRow(label = label, active = idx == stepIndex, done = idx < stepIndex)
                Spacer(Modifier.height(8.dp))
            }
        }
    }
}

@Composable
private fun Spinner() {
    val transition = rememberInfiniteTransition(label = "spin")
    val alpha by transition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 900, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "alpha",
    )
    Box(Modifier.size(72.dp), contentAlignment = Alignment.Center) {
        CircularProgressIndicator(
            color = MaterialTheme.colorScheme.primary.copy(alpha = alpha),
            strokeWidth = 5.dp,
            modifier = Modifier.fillMaxSize(),
        )
    }
}

@Composable
private fun StepRow(label: String, active: Boolean, done: Boolean) {
    val color = when {
        done -> MaterialTheme.colorScheme.primary
        active -> MaterialTheme.colorScheme.onSurface
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(10.dp)
                .padding(end = 0.dp),
        ) {
            val dotColor = if (active || done) MaterialTheme.colorScheme.primary else Color.Gray
            Box(
                Modifier
                    .fillMaxSize()
                    .padding(1.dp),
            ) {
                androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                    drawCircle(color = dotColor)
                }
            }
        }
        Spacer(Modifier.width(12.dp))
        Text(label, color = color, style = MaterialTheme.typography.bodyLarge)
    }
}

private const val STEP_DELAY_MS = 600L

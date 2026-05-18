package com.partoguard.app.ui.results

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.partoguard.app.R
import com.partoguard.app.model.ClinicalAlert
import com.partoguard.app.model.ClinicalStatus
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.PartographPoint
import com.partoguard.app.session.AnalysisSession
import com.partoguard.app.ui.components.ConfidencePill
import com.partoguard.app.ui.components.QualityChips
import com.partoguard.app.ui.components.statusColor

@Composable
fun ResultsScreen(
    session: AnalysisSession,
    onAnalyzeAnother: () -> Unit,
) {
    val extraction = session.extraction
    val alert = session.alert
    if (extraction == null || alert == null) {
        LaunchedEffect(Unit) { onAnalyzeAnother() }
        return
    }

    val statusLabelRes = when (alert.status) {
        ClinicalStatus.NORMAL -> R.string.results_status_normal
        ClinicalStatus.ALERT_ZONE -> R.string.results_status_alert
        ClinicalStatus.ACTION_ZONE -> R.string.results_status_action
        ClinicalStatus.MANUAL_REVIEW -> R.string.results_status_manual
        ClinicalStatus.EMPTY -> R.string.results_status_empty
    }
    val accent = statusColor(alert.status)

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(padding),
            contentPadding = PaddingValues(start = 20.dp, end = 20.dp, top = 20.dp, bottom = 36.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item {
                VerdictHero(
                    accent = accent,
                    statusLabel = stringResource(statusLabelRes),
                    alert = alert,
                    extraction = extraction,
                )
            }
            item {
                SweepCard(
                    accent = accent,
                    bitmap = session.bitmap,
                    extraction = extraction,
                )
            }
            item {
                Text(stringResource(R.string.results_quality_label), style = MaterialTheme.typography.titleSmall)
                Spacer(Modifier.height(6.dp))
                QualityChips(extraction.imageQuality)
            }
            if (extraction.points.isNotEmpty()) {
                item { Text(stringResource(R.string.results_extracted), style = MaterialTheme.typography.titleSmall) }
                items(extraction.points) { p -> PointSummary(p) }
            }
            item {
                Spacer(Modifier.height(8.dp))
                OutlinedButton(
                    onClick = onAnalyzeAnother,
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Text(stringResource(R.string.results_analyze_another))
                }
                Spacer(Modifier.height(8.dp))
                Text(
                    stringResource(R.string.results_disclaimer),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

/**
 * Hero verdict card — first thing the clinician sees. Solid status-color band
 * on the left, large status label, headline, and the triggering measurement
 * pulled out as a numeric callout.
 */
@Composable
private fun VerdictHero(
    accent: Color,
    statusLabel: String,
    alert: ClinicalAlert,
    extraction: PartographExtraction,
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min)) {
            Box(
                Modifier
                    .width(6.dp)
                    .fillMaxHeight()
                    .background(accent),
            )
            Column(Modifier.padding(18.dp).fillMaxWidth()) {
                Text(
                    statusLabel.uppercase(),
                    style = MaterialTheme.typography.labelMedium.copy(
                        fontWeight = FontWeight.SemiBold,
                        letterSpacing = 1.2.sp,
                    ),
                    color = accent,
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    alert.headline,
                    style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.SemiBold),
                    color = MaterialTheme.colorScheme.onSurface,
                )
                val key = alert.triggeringPoint ?: extraction.points.lastOrNull()
                if (key != null) {
                    Spacer(Modifier.height(10.dp))
                    Row(verticalAlignment = Alignment.Bottom) {
                        Text(
                            "%.1f".format(key.dilationCm),
                            style = MaterialTheme.typography.displayMedium.copy(fontWeight = FontWeight.Bold),
                            color = accent,
                        )
                        Spacer(Modifier.width(4.dp))
                        Text(
                            "cm at ${"%.1f".format(key.xHours)} h",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(bottom = 8.dp),
                        )
                    }
                }
                Spacer(Modifier.height(8.dp))
                Text(
                    alert.reason,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun SweepCard(
    accent: Color,
    bitmap: android.graphics.Bitmap?,
    extraction: PartographExtraction,
) {
    val sweep = remember { Animatable(0f) }
    LaunchedEffect(extraction) {
        sweep.snapTo(0f)
        sweep.animateTo(1f, tween(durationMillis = 900, easing = FastOutSlowInEasing))
    }

    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(14.dp),
    ) {
        Box {
            if (bitmap != null && bitmap.height > 0) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = stringResource(R.string.cd_partograph_image),
                    contentScale = ContentScale.Fit,
                    alpha = 0.5f,
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(bitmap.width.toFloat() / bitmap.height.toFloat())
                        .background(Color.White),
                )
            } else {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .aspectRatio(1.4f)
                        .background(Color.White),
                )
            }
            Canvas(
                modifier = Modifier
                    .matchParentSize()
                    .clipToBounds(),
            ) {
                // Plot region inside the chart image (approximate cervicograph bounds).
                val padX = size.width * 0.05f
                val padY = size.height * 0.05f
                val innerW = size.width - 2 * padX
                val innerH = size.height - 2 * padY

                fun xy(hours: Float, cm: Float): Offset {
                    val px = padX + (hours / 12f) * innerW
                    val py = padY + (1f - cm / 10f) * innerH
                    return Offset(px, py)
                }

                val gridColor = Color.Black.copy(alpha = 0.18f)
                for (h in 0..12) {
                    val x = xy(h.toFloat(), 0f).x
                    drawLine(gridColor, Offset(x, padY), Offset(x, padY + innerH), 1f)
                }
                for (cm in 0..10) {
                    val y = xy(0f, cm.toFloat()).y
                    drawLine(gridColor, Offset(padX, y), Offset(padX + innerW, y), 1f)
                }
                drawRect(
                    color = Color.Black.copy(alpha = 0.4f),
                    topLeft = Offset(padX, padY),
                    size = Size(innerW, innerH),
                    style = Stroke(width = 2f),
                )

                // Alert line: 1 cm/h from (0h, 4cm) to (6h, 10cm).
                val alertStart = xy(0f, 4f)
                val alertEnd = xy(6f, 10f)
                drawLine(
                    color = Color(0xFFC97A00).copy(alpha = 0.55f),
                    start = alertStart,
                    end = alertEnd,
                    strokeWidth = 3f,
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 8f)),
                )
                // Action line: alert shifted +4h, drawn from (4h,4cm) to (10h,10cm).
                val actionStart = xy(4f, 4f)
                val actionEnd = xy(10f, 10f)
                drawLine(
                    color = Color(0xFFB31412).copy(alpha = 0.55f),
                    start = actionStart,
                    end = actionEnd,
                    strokeWidth = 3f,
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 8f)),
                )

                // Sweep cursor + clip: only draw the part of the plot revealed so far.
                val sweepX = padX + sweep.value * innerW

                // Vertical "now" cursor while animating.
                if (sweep.value > 0f && sweep.value < 1f) {
                    drawLine(
                        color = accent.copy(alpha = 0.45f),
                        start = Offset(sweepX, padY * 0.6f),
                        end = Offset(sweepX, size.height - padY * 0.6f),
                        strokeWidth = 2f,
                    )
                }

                // Connecting line through points (clipped by sweep).
                if (extraction.points.size >= 2) {
                    val sorted = extraction.points.sortedBy { it.xHours }
                    for (i in 0 until sorted.size - 1) {
                        val a = xy(sorted[i].xHours, sorted[i].dilationCm)
                        val b = xy(sorted[i + 1].xHours, sorted[i + 1].dilationCm)
                        if (a.x <= sweepX) {
                            val end = if (b.x <= sweepX) b else {
                                // partial segment up to cursor
                                val t = ((sweepX - a.x) / (b.x - a.x)).coerceIn(0f, 1f)
                                Offset(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
                            }
                            drawLine(
                                color = accent.copy(alpha = 0.85f),
                                start = a,
                                end = end,
                                strokeWidth = 4f,
                            )
                        }
                    }
                }

                // Points: only those whose x has been swept past.
                extraction.points.forEach { p ->
                    val c = xy(p.xHours, p.dilationCm)
                    if (c.x <= sweepX) {
                        drawCircle(color = Color.White, radius = 13f, center = c)
                        drawCircle(color = accent, radius = 10f, center = c)
                        drawCircle(
                            color = Color.White,
                            radius = 10f,
                            center = c,
                            style = Stroke(width = 2f),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PointSummary(p: PartographPoint) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(10.dp),
    ) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Hour ${"%.1f".format(p.xHours)}")
            Text("${"%.1f".format(p.dilationCm)} cm")
            ConfidencePill(p.confidence)
        }
    }
}

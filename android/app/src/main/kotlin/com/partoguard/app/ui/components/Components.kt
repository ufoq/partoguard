package com.partoguard.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.partoguard.app.model.ClinicalStatus
import com.partoguard.app.model.ImageQuality
import com.partoguard.app.ui.theme.StatusAction
import com.partoguard.app.ui.theme.StatusAlert
import com.partoguard.app.ui.theme.StatusManual
import com.partoguard.app.ui.theme.StatusNormal

fun statusColor(status: ClinicalStatus): Color = when (status) {
    ClinicalStatus.NORMAL -> StatusNormal
    ClinicalStatus.ALERT_ZONE -> StatusAlert
    ClinicalStatus.ACTION_ZONE -> StatusAction
    ClinicalStatus.MANUAL_REVIEW -> StatusManual
    ClinicalStatus.EMPTY -> StatusManual
}

@Composable
fun SeverityBadge(status: ClinicalStatus, label: String, modifier: Modifier = Modifier) {
    val bg = statusColor(status)
    Box(
        modifier
            .clip(RoundedCornerShape(50))
            .background(bg)
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        Text(label, color = Color.White, style = MaterialTheme.typography.labelMedium)
    }
}

@Composable
fun QualityChips(quality: ImageQuality, modifier: Modifier = Modifier) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        if (quality.isGood) Chip("Good", StatusNormal)
        if (quality.blurry) Chip("Blurry", StatusAlert)
        if (quality.dim) Chip("Dim", StatusAlert)
        if (quality.skewed) Chip("Skewed", StatusAlert)
    }
}

@Composable
private fun Chip(text: String, color: Color) {
    Box(
        Modifier
            .clip(RoundedCornerShape(50))
            .background(color.copy(alpha = 0.15f))
            .padding(PaddingValues(horizontal = 10.dp, vertical = 4.dp)),
    ) {
        Text(text, color = color, style = MaterialTheme.typography.labelSmall)
    }
}

/** Categorical confidence band — avoids spurious-precision percentages. */
enum class ConfidenceBand(val label: String) {
    HIGH("High"), MEDIUM("Med"), LOW("Low");

    companion object {
        fun of(confidence: Float): ConfidenceBand = when {
            confidence >= 0.85f -> HIGH
            confidence >= 0.70f -> MEDIUM
            else -> LOW
        }
    }
}

@Composable
fun ConfidencePill(confidence: Float, modifier: Modifier = Modifier) {
    val band = ConfidenceBand.of(confidence)
    val color = when (band) {
        ConfidenceBand.HIGH -> StatusNormal
        ConfidenceBand.MEDIUM -> StatusAlert
        ConfidenceBand.LOW -> StatusManual
    }
    Box(
        modifier
            .clip(RoundedCornerShape(50))
            .background(color.copy(alpha = 0.15f))
            .padding(horizontal = 10.dp, vertical = 4.dp),
    ) {
        Text(band.label, color = color, style = MaterialTheme.typography.labelSmall)
    }
}

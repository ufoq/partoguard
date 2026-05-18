package com.partoguard.app.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.partoguard.app.R
import com.partoguard.app.analyzer.ExtractorSelector
import com.partoguard.app.analyzer.InferenceMode
import com.partoguard.app.model.WorkflowMode
import com.partoguard.app.session.AnalysisSession
import com.partoguard.app.util.AssetLoader

private data class DemoSample(val asset: String, val label: String, val title: String)

private val DEMO_SAMPLES = listOf(
    DemoSample("demo_partographs/demo_normal.png", "demo_normal", "Normal"),
    DemoSample("demo_partographs/demo_alert.png", "demo_alert", "Alert"),
    DemoSample("demo_partographs/demo_action.png", "demo_action", "Action"),
    DemoSample("demo_partographs/demo_blank.png", "demo_blank", "Blank"),
)

private val WHO_SAMPLES = listOf(
    DemoSample("demo_partographs/WHO-normal.png", "WHO_normal", "WHO Normal"),
    DemoSample("demo_partographs/WHO-alert.png", "WHO_alert", "WHO Alert"),
    DemoSample("demo_partographs/WHO-action.png", "WHO_action", "WHO Action"),
    DemoSample("demo_partographs/WHO-blank.png", "WHO_blank", "WHO Blank"),
    DemoSample("demo_partographs/WHO-pt1.png", "WHO_pt1", "WHO PT1"),
)

@Composable
fun HomeScreen(
    session: AnalysisSession,
    onPickDemo: () -> Unit,
    onLiveCamera: () -> Unit,
    onSettings: () -> Unit,
) {
    val context = LocalContext.current
    var mode by remember { mutableStateOf(session.mode) }
    LaunchedEffect(mode) { session.mode = mode }

    Scaffold(contentWindowInsets = WindowInsets(0, 0, 0, 0)) {
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .windowInsetsPadding(WindowInsets.safeDrawing),
            contentPadding = PaddingValues(
                start = 20.dp,
                end = 20.dp,
                top = 20.dp,
                bottom = 40.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(stringResource(R.string.app_name), style = MaterialTheme.typography.displaySmall)
                    IconButton(onClick = onSettings) {
                        Icon(
                            Icons.Filled.Settings,
                            contentDescription = "Settings",
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            item {
                Text(
                    stringResource(R.string.app_tagline),
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
            }
            item { Spacer(Modifier.height(8.dp)) }
            item {
                Text(stringResource(R.string.home_section_modes), style = MaterialTheme.typography.titleSmall)
            }
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    ModeChip(WorkflowMode.AUTO, mode, R.string.mode_auto) { mode = it }
                    ModeChip(WorkflowMode.ASSISTED, mode, R.string.mode_assisted) { mode = it }
                    ModeChip(WorkflowMode.MANUAL, mode, R.string.mode_manual) { mode = it }
                }
            }
            item { Spacer(Modifier.height(4.dp)) }
            item { PrimaryCaptureCard(onLiveCamera) }
            item { Spacer(Modifier.height(12.dp)) }
            item {
                Text(
                    stringResource(R.string.home_section_demo),
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            item {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    DEMO_SAMPLES.forEach { sample ->
                        AssistChip(
                            onClick = {
                                session.bitmap = AssetLoader.loadBitmap(context, sample.asset)
                                session.sourceLabel = sample.label
                                onPickDemo()
                            },
                            label = { Text(sample.title) },
                            colors = AssistChipDefaults.assistChipColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                            ),
                        )
                    }
                }
            }
            item {
                Text(
                    "WHO Templates",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            item {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.horizontalScroll(rememberScrollState()),
                ) {
                    WHO_SAMPLES.forEach { sample ->
                        AssistChip(
                            onClick = {
                                session.bitmap = AssetLoader.loadBitmap(context, sample.asset)
                                session.sourceLabel = sample.label
                                onPickDemo()
                            },
                            label = { Text(sample.title) },
                            colors = AssistChipDefaults.assistChipColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant,
                            ),
                        )
                    }
                }
            }
            item { Spacer(Modifier.height(8.dp)) }
            item {
                val currentMode = ExtractorSelector.getMode(context)
                val footerText = when (currentMode) {
                    InferenceMode.OFFLINE_LITERT -> "Engine: Offline (LiteRT base) \u2022 On-device processing"
                    InferenceMode.OFFLINE_LLAMACPP -> "Engine: Offline (llama.cpp Q8) \u2022 On-device processing"
                    InferenceMode.ONLINE_DEMO -> "Engine: Online (demo server) \u2022 Requires network"
                    InferenceMode.ONLINE_CUSTOM -> "Engine: Online (custom) \u2022 Requires network"
                }
                Text(
                    footerText,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun PrimaryCaptureCard(onLiveCamera: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primary,
        ),
        shape = RoundedCornerShape(18.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Column(Modifier.padding(20.dp).fillMaxWidth()) {
            Text(
                stringResource(R.string.home_capture_title),
                style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.SemiBold),
                color = MaterialTheme.colorScheme.onPrimary,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                stringResource(R.string.home_capture_subtitle),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.85f),
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = onLiveCamera,
                modifier = Modifier.fillMaxWidth().height(52.dp),
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.onPrimary,
                    contentColor = MaterialTheme.colorScheme.primary,
                ),
            ) {
                Text(
                    stringResource(R.string.home_action_live),
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                )
            }
        }
    }
}

@Composable
private fun ModeChip(
    target: WorkflowMode,
    current: WorkflowMode,
    titleRes: Int,
    onSelect: (WorkflowMode) -> Unit,
) {
    FilterChip(
        selected = target == current,
        onClick = { onSelect(target) },
        label = { Text(stringResource(titleRes)) },
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = MaterialTheme.colorScheme.primary,
            selectedLabelColor = MaterialTheme.colorScheme.onPrimary,
        ),
    )
}

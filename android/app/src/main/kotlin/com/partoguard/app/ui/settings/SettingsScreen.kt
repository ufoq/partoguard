package com.partoguard.app.ui.settings

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.partoguard.app.analyzer.DownloadProgress
import com.partoguard.app.analyzer.ExtractorSelector
import com.partoguard.app.analyzer.InferenceMode
import com.partoguard.app.analyzer.ModelDownloadManager
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    isFirstLaunch: Boolean = false,
    onSetupComplete: () -> Unit = {},
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val defaultMode = if (isFirstLaunch) ExtractorSelector.recommendedDefault(context) else ExtractorSelector.getMode(context)
    var selectedMode by remember { mutableStateOf(defaultMode) }
    var customUrl by remember { mutableStateOf(ExtractorSelector.getCustomUrl(context)) }
    val focusManager = LocalFocusManager.current

    if (isFirstLaunch) {
        ExtractorSelector.setMode(context, defaultMode)
    }

    var liteRtDownloaded by remember { mutableStateOf(ExtractorSelector.isLiteRtModelDownloaded(context)) }
    var llamaCppDownloaded by remember { mutableStateOf(ExtractorSelector.isLlamaCppModelDownloaded(context)) }
    var downloadingMode by remember { mutableStateOf<InferenceMode?>(null) }
    var downloadProgress by remember { mutableStateOf(0f) }
    var downloadError by remember { mutableStateOf<String?>(null) }
    val deviceRamGb = remember { ExtractorSelector.totalRamGb(context) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (isFirstLaunch) "Welcome to PartoGuard" else "Settings") },
                navigationIcon = {
                    if (!isFirstLaunch) {
                        IconButton(onClick = onBack) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                        }
                    }
                },
            )
        },
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                "Analysis Engine",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(bottom = 4.dp),
            )
            Text(
                "Choose how partograph images are analyzed. Offline engines run entirely on this device.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 8.dp),
            )

            InferenceMode.entries.forEach { mode ->
                val isDownloaded = when (mode) {
                    InferenceMode.OFFLINE_LITERT -> liteRtDownloaded
                    InferenceMode.OFFLINE_LLAMACPP -> llamaCppDownloaded
                    else -> true
                }
                val isDownloading = downloadingMode == mode

                EngineOptionCard(
                    mode = mode,
                    isSelected = mode == selectedMode,
                    onSelect = {
                        selectedMode = mode
                        ExtractorSelector.setMode(context, mode)
                    },
                    isModelDownloaded = isDownloaded,
                    isDownloading = isDownloading,
                    downloadProgress = if (isDownloading) downloadProgress else 0f,
                    ramInsufficient = mode.minRamGb > 0f && deviceRamGb < mode.minRamGb,
                    deviceRamGb = deviceRamGb,
                    onDownload = {
                        downloadError = null
                        downloadingMode = mode
                        downloadProgress = 0f
                        scope.launch {
                            val flow = when (mode) {
                                InferenceMode.OFFLINE_LITERT -> ModelDownloadManager.downloadLiteRtModel(context)
                                InferenceMode.OFFLINE_LLAMACPP -> ModelDownloadManager.downloadLlamaCppModels(context)
                                else -> return@launch
                            }
                            flow.collect { progress ->
                                when (progress) {
                                    is DownloadProgress.Downloading -> {
                                        downloadProgress = progress.percent
                                    }
                                    is DownloadProgress.Completed -> {
                                        downloadingMode = null
                                        liteRtDownloaded = ExtractorSelector.isLiteRtModelDownloaded(context)
                                        llamaCppDownloaded = ExtractorSelector.isLlamaCppModelDownloaded(context)
                                    }
                                    is DownloadProgress.Failed -> {
                                        downloadingMode = null
                                        downloadError = progress.error
                                    }
                                }
                            }
                        }
                    },
                    onDelete = {
                        when (mode) {
                            InferenceMode.OFFLINE_LITERT -> {
                                ModelDownloadManager.deleteLiteRtModel(context)
                                liteRtDownloaded = false
                            }
                            InferenceMode.OFFLINE_LLAMACPP -> {
                                ModelDownloadManager.deleteLlamaCppModels(context)
                                llamaCppDownloaded = false
                            }
                            else -> {}
                        }
                    },
                )
            }

            AnimatedVisibility(visible = selectedMode == InferenceMode.ONLINE_CUSTOM) {
                Column(modifier = Modifier.padding(top = 8.dp)) {
                    OutlinedTextField(
                        value = customUrl,
                        onValueChange = {
                            customUrl = it
                            ExtractorSelector.setCustomUrl(context, it)
                        },
                        label = { Text("Server URL") },
                        placeholder = { Text("http://your-server:8080") },
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(
                            keyboardType = KeyboardType.Uri,
                            imeAction = ImeAction.Done,
                        ),
                        keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() }),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                    )
                    Text(
                        "Full base URL to a llama-server /completion endpoint",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 4.dp, start = 4.dp),
                    )
                }
            }

            downloadError?.let { error ->
                Text(
                    "Download failed: $error",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            Spacer(Modifier.height(16.dp))

            if (isFirstLaunch) {
                Button(
                    onClick = {
                        ExtractorSelector.markSetupCompleted(context)
                        onSetupComplete()
                    },
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Text("Continue")
                }
            } else {
                Text(
                    "Changes take effect on the next analysis.",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun EngineOptionCard(
    mode: InferenceMode,
    isSelected: Boolean,
    onSelect: () -> Unit,
    isModelDownloaded: Boolean,
    isDownloading: Boolean,
    downloadProgress: Float,
    ramInsufficient: Boolean,
    deviceRamGb: Float,
    onDownload: () -> Unit,
    onDelete: () -> Unit,
) {
    val isOffline = mode == InferenceMode.OFFLINE_LITERT || mode == InferenceMode.OFFLINE_LLAMACPP
    val containerColor = if (isSelected) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    }

    Card(
        colors = CardDefaults.cardColors(containerColor = containerColor),
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onSelect),
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(selected = isSelected, onClick = onSelect)
                Column(modifier = Modifier.padding(start = 8.dp).weight(1f)) {
                    Text(mode.displayName, style = MaterialTheme.typography.bodyLarge)
                    Text(
                        mode.subtitle,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    if (ramInsufficient) {
                        Text(
                            "⚠ Device has %.1f GB RAM — needs %.0f+ GB. May crash.".format(deviceRamGb, mode.minRamGb),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                }
            }

            if (isOffline) {
                Spacer(Modifier.height(6.dp))
                when {
                    isDownloading -> {
                        LinearProgressIndicator(
                            progress = { downloadProgress },
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                        )
                        Text(
                            "${(downloadProgress * 100).toInt()}%",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(start = 12.dp, top = 2.dp),
                        )
                    }
                    isModelDownloaded -> {
                        Row(
                            modifier = Modifier.padding(start = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text(
                                "\u2713 Model ready",
                                style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.primary,
                            )
                            Spacer(Modifier.width(12.dp))
                            OutlinedButton(
                                onClick = onDelete,
                                contentPadding = ButtonDefaults.TextButtonContentPadding,
                            ) {
                                Text("Delete", style = MaterialTheme.typography.labelSmall)
                            }
                        }
                    }
                    else -> {
                        Row(
                            modifier = Modifier.padding(start = 12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Button(
                                onClick = onDownload,
                                contentPadding = ButtonDefaults.TextButtonContentPadding,
                            ) {
                                Text("Download", style = MaterialTheme.typography.labelSmall)
                            }
                            Spacer(Modifier.width(8.dp))
                            val sizeText = when (mode) {
                                InferenceMode.OFFLINE_LITERT -> ModelDownloadManager.liteRtModelSizeMb()
                                InferenceMode.OFFLINE_LLAMACPP -> ModelDownloadManager.llamaCppModelSizeMb()
                                else -> ""
                            }
                            Text(
                                sizeText,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}

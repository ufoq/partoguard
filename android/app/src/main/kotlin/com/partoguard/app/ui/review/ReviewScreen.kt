package com.partoguard.app.ui.review

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.partoguard.app.R
import com.partoguard.app.model.PartographPoint
import com.partoguard.app.rules.RuleEngine
import com.partoguard.app.session.AnalysisSession
import com.partoguard.app.ui.components.ConfidencePill
import com.partoguard.app.ui.components.QualityChips

private const val MAX_HOURS = 14f
private const val MAX_CM = 10f

@Composable
fun ReviewScreen(
    session: AnalysisSession,
    onConfirmed: () -> Unit,
    onBack: () -> Unit,
) {
    val extraction = session.extraction
    if (extraction == null) {
        LaunchedEffect(Unit) { onBack() }
        return
    }

    // Parallel ids list keeps row identity stable across insert/delete so
    // BasicTextField remember-state doesn't leak between rows when one is
    // removed. Without this, `items(count) { index -> remember { ... } }`
    // reuses state by position and shows the deleted row's text in its
    // neighbour.
    val points: SnapshotStateList<PartographPoint> = remember {
        mutableStateListOf<PartographPoint>().apply { addAll(extraction.points) }
    }
    val ids: SnapshotStateList<Int> = remember {
        mutableStateListOf<Int>().apply { addAll(extraction.points.indices.toList()) }
    }
    var nextId by remember { mutableStateOf(extraction.points.size) }

    val density = LocalDensity.current
    val navBarBottom = with(density) { WindowInsets.navigationBars.getBottom(density).toDp() }

    Scaffold(
        contentWindowInsets = WindowInsets(0, 0, 0, 0),
        topBar = {
            Surface(color = MaterialTheme.colorScheme.background) {
                Column(Modifier.fillMaxWidth().windowInsetsPadding(WindowInsets.statusBars)) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(start = 4.dp, end = 16.dp, top = 4.dp)) {
                        IconButton(onClick = onBack) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.cd_back))
                        }
                        Spacer(Modifier.width(4.dp))
                        Text(stringResource(R.string.review_title), style = MaterialTheme.typography.titleLarge)
                    }
                    Text(
                        stringResource(R.string.review_subtitle),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp),
                    )
                    Box(Modifier.padding(horizontal = 20.dp, vertical = 8.dp)) {
                        QualityChips(extraction.imageQuality)
                    }
                }
            }
        },
        bottomBar = {
            Surface(
                color = MaterialTheme.colorScheme.background,
                tonalElevation = 4.dp,
            ) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(start = 20.dp, top = 16.dp, end = 20.dp)
                        .padding(bottom = navBarBottom + 16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    OutlinedButton(
                        onClick = {
                            val lastHour = points.maxOfOrNull { it.xHours } ?: -2f
                            val nextHour = (lastHour + 2f).coerceIn(0f, MAX_HOURS)
                            val lastCm = points.lastOrNull()?.dilationCm ?: 4f
                            val nextCm = lastCm.coerceIn(0f, MAX_CM)
                            points.add(PartographPoint(nextHour, nextCm, 1f))
                            ids.add(nextId++)
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.review_add_point))
                    }
                    Button(
                        onClick = {
                            val finalPoints = points.toList().sortedBy { it.xHours }
                            val confirmed = extraction.copy(
                                points = finalPoints,
                                needsManualReview = false,
                                reasonForManualReview = null,
                            )
                            session.extraction = confirmed
                            session.alert = RuleEngine.evaluate(confirmed)
                            onConfirmed()
                        },
                        modifier = Modifier.fillMaxWidth().height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Text(stringResource(R.string.review_confirm))
                    }
                }
            }
        },
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            item {
                Row(
                    Modifier.fillMaxWidth().padding(top = 4.dp, bottom = 6.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    HeaderCell(stringResource(R.string.review_caption_hours), Modifier.weight(1f))
                    HeaderCell(stringResource(R.string.review_caption_cm), Modifier.weight(1f))
                    HeaderCell(stringResource(R.string.review_caption_conf), Modifier.weight(1f))
                    Spacer(Modifier.width(40.dp))
                }
            }
            if (points.isEmpty()) {
                item {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            stringResource(R.string.review_no_points),
                            modifier = Modifier.padding(16.dp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            } else {
                items(
                    count = points.size,
                    key = { idx -> ids[idx] },
                ) { index ->
                    val p = points[index]
                    PointRow(
                        rowKey = ids[index],
                        p = p,
                        onUpdate = { newP -> points[index] = newP },
                        onDelete = {
                            points.removeAt(index)
                            ids.removeAt(index)
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun HeaderCell(text: String, modifier: Modifier = Modifier) {
    Text(
        text,
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = modifier,
    )
}

private fun formatNumber(v: Float): String =
    if (v.toInt().toFloat() == v) v.toInt().toString() else "%.1f".format(v)

@Composable
private fun CompactNumberField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    BasicTextField(
        value = value,
        onValueChange = onValueChange,
        textStyle = MaterialTheme.typography.bodyMedium.copy(
            color = MaterialTheme.colorScheme.onSurface,
        ),
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        singleLine = true,
        cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
        modifier = modifier
            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(6.dp))
            .border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.outline,
                shape = RoundedCornerShape(6.dp),
            )
            .padding(horizontal = 8.dp, vertical = 8.dp),
    )
}

@Composable
private fun PointRow(
    rowKey: Int,
    p: PartographPoint,
    onUpdate: (PartographPoint) -> Unit,
    onDelete: () -> Unit,
) {
    // Keying remember on rowKey ensures text state is bound to the row's
    // identity, not its position. After deleting a middle row, neighbours
    // keep their own values instead of inheriting the deleted row's text.
    var hoursText by remember(rowKey) { mutableStateOf(formatNumber(p.xHours)) }
    var cmText by remember(rowKey) { mutableStateOf(formatNumber(p.dilationCm)) }

    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            CompactNumberField(
                value = hoursText,
                onValueChange = { raw ->
                    hoursText = raw
                    raw.toFloatOrNull()?.let { v ->
                        if (v in 0f..MAX_HOURS) onUpdate(p.copy(xHours = v))
                    }
                },
                modifier = Modifier.weight(1f),
            )
            CompactNumberField(
                value = cmText,
                onValueChange = { raw ->
                    cmText = raw
                    raw.toFloatOrNull()?.let { v ->
                        if (v in 0f..MAX_CM) onUpdate(p.copy(dilationCm = v))
                    }
                },
                modifier = Modifier.weight(1f),
            )
            Box(Modifier.weight(1f)) { ConfidencePill(p.confidence) }
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = stringResource(R.string.review_delete))
            }
        }
    }
}

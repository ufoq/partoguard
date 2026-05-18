package com.partoguard.app.ui.eval

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
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
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.partoguard.app.analyzer.PartographExtractor
import com.partoguard.app.eval.EvalManifestEntry
import com.partoguard.app.eval.EvalScorer
import com.partoguard.app.eval.ScoreVerdict
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.rules.RuleEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import kotlin.system.measureTimeMillis

private const val TAG = "PARTOGUARD_EVAL"
private const val ASSET_DIR = "eval_samples"
private const val MANIFEST_FILE = "$ASSET_DIR/manifest.json"

private data class EvalRow(
    val entry: EvalManifestEntry,
    val verdict: ScoreVerdict?,
    val extraction: PartographExtraction?,
    val durationMs: Long,
    val state: State,
) {
    enum class State { PENDING, RUNNING, DONE, ERROR }
}

@Composable
fun EvalRunnerScreen(
    extractor: PartographExtractor,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val rows = remember { mutableStateListOf<EvalRow>() }
    var summary by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        runEvalSuite(context, extractor, rows) { summary = it }
    }

    Scaffold(contentWindowInsets = WindowInsets(0, 0, 0, 0)) {
        Column(
            Modifier
                .fillMaxSize()
                .padding(20.dp)
                .windowInsetsPadding(WindowInsets.safeDrawing),
        ) {
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                TextButton(onClick = onBack) { Text("\u2190 Back") }
                Spacer(Modifier.fillMaxWidth(0.05f))
                Text(
                    text = "Eval suite",
                    style = MaterialTheme.typography.headlineSmall,
                )
            }
            Text(
                text = "Logcat tag: $TAG",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            summary?.let {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (it.contains("FAIL"))
                            MaterialTheme.colorScheme.errorContainer
                        else MaterialTheme.colorScheme.primaryContainer,
                    ),
                ) {
                    Text(
                        text = it,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                Spacer(Modifier.height(12.dp))
            } ?: run {
                if (rows.any { it.state == EvalRow.State.RUNNING }) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Spacer(Modifier.height(8.dp))
                }
            }
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(rows) { row ->
                    EvalRowCard(row)
                }
            }
        }
    }
}

@Composable
private fun EvalRowCard(row: EvalRow) {
    val color = when (row.state) {
        EvalRow.State.PENDING -> MaterialTheme.colorScheme.surfaceVariant
        EvalRow.State.RUNNING -> MaterialTheme.colorScheme.tertiaryContainer
        EvalRow.State.DONE -> if (row.verdict?.correct == true)
            Color(0xFFD4F4DD) else Color(0xFFFAD2CF)
        EvalRow.State.ERROR -> Color(0xFFFAD2CF)
    }
    Card(colors = CardDefaults.cardColors(containerColor = color)) {
        Column(Modifier.padding(12.dp).fillMaxWidth()) {
            Text(
                text = row.entry.path,
                style = MaterialTheme.typography.bodyMedium,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                text = "category=${row.entry.category} curve=${row.entry.curveType} truth_n=${row.entry.nMarks}",
                style = MaterialTheme.typography.bodySmall,
            )
            row.extraction?.let { ex ->
                Text(
                    text = "predicted: n=${ex.points.size} manual=${ex.needsManualReview} reason=${ex.reasonForManualReview ?: "-"}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            row.verdict?.let { v ->
                Text(
                    text = if (v.correct) "PASS \u2713 ${v.reason}" else "FAIL \u2717 ${v.reason}",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Bold,
                )
            }
            if (row.state == EvalRow.State.DONE || row.state == EvalRow.State.ERROR) {
                Text(
                    text = "duration: ${row.durationMs} ms",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            if (row.state == EvalRow.State.RUNNING) {
                Text(
                    text = "Running...",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                )
            }
        }
    }
}

private suspend fun runEvalSuite(
    context: Context,
    extractor: PartographExtractor,
    rows: androidx.compose.runtime.snapshots.SnapshotStateList<EvalRow>,
    onDone: (String) -> Unit,
) {
    Log.i(TAG, "START: loading manifest from assets/$MANIFEST_FILE")
    val entries = withContext(Dispatchers.IO) { loadManifest(context) }
    Log.i(TAG, "MANIFEST: ${entries.size} entries")

    entries.forEach { e -> rows.add(EvalRow(e, null, null, 0L, EvalRow.State.PENDING)) }

    var passed = 0
    var failed = 0
    val totalStart = System.currentTimeMillis()

    entries.forEachIndexed { index, entry ->
        rows[index] = rows[index].copy(state = EvalRow.State.RUNNING)
        Log.i(TAG, "RUN[${index + 1}/${entries.size}]: ${entry.path}  truth(category=${entry.category}, curve=${entry.curveType}, n_marks=${entry.nMarks})")

        var extraction: PartographExtraction? = null
        var verdict: ScoreVerdict? = null
        var state = EvalRow.State.DONE
        val durationMs = measureTimeMillis {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    context.assets.open("$ASSET_DIR/${entry.path}").use { BitmapFactory.decodeStream(it) }
                } ?: error("could not decode bitmap")
                extraction = extractor.extract(bitmap, "eval_${entry.path}")
                val alert = RuleEngine.evaluate(extraction!!)
                verdict = EvalScorer.score(entry, alert.status, extraction!!.points.size)
                if (verdict!!.correct) passed++ else failed++
                Log.i(TAG, "RESULT[${index + 1}]: ${if (verdict!!.correct) "PASS" else "FAIL"}  predicted_n=${extraction!!.points.size} status=${alert.status}  reason=${verdict!!.reason}")
            } catch (t: Throwable) {
                state = EvalRow.State.ERROR
                failed++
                Log.e(TAG, "ERROR[${index + 1}]: ${t.javaClass.simpleName}: ${t.message}", t)
            }
        }
        rows[index] = EvalRow(entry, verdict, extraction, durationMs, state)
    }

    val totalSec = (System.currentTimeMillis() - totalStart) / 1000.0
    val verdictMsg = if (failed == 0)
        "PASS: $passed/${entries.size} passed in ${"%.1f".format(totalSec)}s"
    else
        "FAIL: $passed/${entries.size} passed, $failed failed in ${"%.1f".format(totalSec)}s"
    Log.i(TAG, "DONE: $verdictMsg")
    Log.i(TAG, "PARTOGUARD_EVAL_DONE passed=$passed failed=$failed total=${entries.size} seconds=${"%.1f".format(totalSec)}")
    onDone(verdictMsg)
}

private fun loadManifest(context: Context): List<EvalManifestEntry> {
    val text = context.assets.open(MANIFEST_FILE).bufferedReader().use { it.readText() }
    val arr = JSONArray(text)
    return List(arr.length()) { i ->
        val o = arr.getJSONObject(i)
        EvalManifestEntry(
            path = o.getString("path"),
            category = o.getString("category"),
            curveType = o.getString("curve_type"),
            nMarks = o.getInt("n_marks"),
        )
    }
}

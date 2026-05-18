package com.partoguard.app.analyzer

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.partoguard.app.model.ImageQuality
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.PartographPoint
import com.partoguard.llama.LlamaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import kotlin.math.roundToInt

class LlamaCppPartographExtractor(
    private val context: Context,
) : PartographExtractor {

    private val engine = LlamaEngine(context)
    private val initMutex = Mutex()
    private var initialized = false

    override suspend fun extract(bitmap: Bitmap, sourceLabel: String): PartographExtraction =
        withContext(Dispatchers.IO) {
            runCatching { doExtract(bitmap) }.getOrElse { e ->
                Log.e(TAG, "extract failed for $sourceLabel: ${e.javaClass.simpleName}: ${e.message}", e)
                manualReview("llamacpp_error:${e.javaClass.simpleName}")
            }
        }

    private suspend fun doExtract(bitmap: Bitmap): PartographExtraction {
        ensureInitialized()
        val scaled = downscale(bitmap, MAX_IMAGE_DIM)
        val prompt = "<|turn>user\n$MEDIA_MARKER\n$EXTRACTION_PROMPT<turn|>\n<|turn>model\n"
        val response = engine.complete(prompt, scaled, temperature = 0.0f, maxTokens = 400)
        if (scaled !== bitmap) scaled.recycle()
        if (response.isBlank()) return manualReview("llamacpp_empty_response")
        return parseContent(response)
    }

    private suspend fun ensureInitialized() {
        if (initialized) return
        initMutex.withLock {
            if (initialized) return
            val modelFile = File(context.filesDir, GGUF_FILENAME)
            val mmprojFile = File(context.filesDir, MMPROJ_FILENAME)
            check(modelFile.exists()) { "GGUF model not found: ${modelFile.absolutePath}" }
            check(mmprojFile.exists()) { "mmproj not found: ${mmprojFile.absolutePath}" }

            Log.i(TAG, "Initializing LlamaEngine")
            engine.init()

            Log.i(TAG, "Loading model: ${modelFile.absolutePath}")
            Log.i(TAG, "Loading mmproj: ${mmprojFile.absolutePath}")
            val loaded = engine.loadModel(
                modelPath = modelFile.absolutePath,
                mmprojPath = mmprojFile.absolutePath,
                nCtx = CONTEXT_LENGTH,
                nThreads = N_THREADS,
            )
            check(loaded) { "Failed to load model via LlamaEngine" }
            initialized = true
            Log.i(TAG, "LlamaEngine ready")
        }
    }

    fun close() {
        if (initialized) {
            engine.release()
            initialized = false
        }
    }

    private fun parseContent(content: String): PartographExtraction {
        val json = extractJsonObject(content) ?: return manualReview("parse_no_json")
        val pArray = json.optJSONArray("p") ?: return manualReview("parse_missing_p_key")

        val points = mutableListOf<PartographPoint>()
        for (i in 0 until pArray.length()) {
            val item = pArray.optJSONArray(i) ?: continue
            if (item.length() < 2) continue
            val x = item.optDouble(0, Double.NaN).toFloat()
            val d = item.optDouble(1, Double.NaN).toFloat()
            if (x.isNaN() || d.isNaN()) continue
            val c = if (item.length() >= 3) item.optDouble(2, 0.5).toFloat() else 0.5f
            points += PartographPoint(
                xHours = x.coerceIn(0f, 12f).roundToHalf(),
                dilationCm = d.coerceIn(0f, 10f).roundToHalf(),
                confidence = c.coerceIn(0f, 1f),
            )
        }

        return PartographExtraction(
            chartSupported = true,
            imageQuality = ImageQuality(),
            points = points.sortedWith(compareBy({ it.xHours }, { it.dilationCm })),
            needsManualReview = false,
        )
    }

    private fun extractJsonObject(text: String): JSONObject? {
        var s = text.trim()
        if (s.startsWith("```")) {
            s = s.trimStart('`').trim()
            if (s.startsWith("json")) s = s.removePrefix("json").trim()
        }
        return runCatching { JSONObject(s) }.getOrElse {
            val start = s.indexOf('{')
            val end = s.lastIndexOf('}')
            if (start == -1 || end <= start) return null
            runCatching { JSONObject(s.substring(start, end + 1)) }.getOrNull()
        }
    }

    private fun manualReview(reason: String) = PartographExtraction(
        chartSupported = true,
        imageQuality = ImageQuality(),
        points = emptyList(),
        needsManualReview = true,
        reasonForManualReview = reason,
    )

    companion object {
        private const val TAG = "PG_LlamaCppExtractor"
        const val GGUF_FILENAME = "v7_q8_0.gguf"
        const val MMPROJ_FILENAME = "v7_mmproj_f16.gguf"
        private const val CONTEXT_LENGTH = 4096
        private const val N_THREADS = 4
        private const val MEDIA_MARKER = "<__media__>"
        private const val MAX_IMAGE_DIM = 1024

        private const val EXTRACTION_PROMPT =
            "Find every handwritten X mark inside the cervicograph plot of this WHO " +
            "partograph. The cervicograph is the grid where cervical dilation 0-10 cm " +
            "is on the y-axis and hours 0-12 is on the x-axis, with two diagonal " +
            "Alert and Action lines crossing it. An X mark is two short pen strokes " +
            "crossing at one point.\n\n" +
            "Ignore: printed grid lines, the diagonal Alert/Action lines, axis labels, " +
            "contraction shading, fetal-heart-rate dots, and handwriting outside the " +
            "cervicograph.\n\n" +
            "For each X mark return [x_hours, dilation_cm, confidence] where x_hours is " +
            "integer 0-12, dilation_cm is half-integer 0.0-10.0 (in 0.5 increments), " +
            "confidence is 0.0-1.0.\n\n" +
            "If the cervicograph is blank (no X marks), return {\"p\":[]}.\n\n" +
            "Return strictly compact JSON in this exact schema, no markdown fences and " +
            "no commentary:\n" +
            "{\"p\":[[x_hours, dilation_cm, confidence], ...]}"
    }
}

private fun Float.roundToHalf(): Float = (this * 2f).roundToInt().toFloat() / 2f

private fun downscale(bitmap: Bitmap, maxDim: Int): Bitmap {
    val w = bitmap.width
    val h = bitmap.height
    if (w <= maxDim && h <= maxDim) return bitmap
    val scale = maxDim.toFloat() / maxOf(w, h)
    val newW = (w * scale).toInt()
    val newH = (h * scale).toInt()
    return Bitmap.createScaledBitmap(bitmap, newW, newH, true)
}

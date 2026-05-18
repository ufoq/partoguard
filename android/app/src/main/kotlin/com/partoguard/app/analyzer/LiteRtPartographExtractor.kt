package com.partoguard.app.analyzer

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.partoguard.app.model.ImageQuality
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.PartographPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import kotlin.math.roundToInt

/**
 * Tier 1 on-device extractor using Google's base Gemma 4 E2B-it LiteRT-LM bundle.
 *
 * Uses the Conversation API with multimodal content (image + text).
 * Model file: `gemma-4-E2B-it.litertlm` (~2.58 GB), downloaded to app filesDir.
 *
 * The extraction prompt is identical to [RemotePartographExtractor.EXTRACTION_PROMPT]
 * to keep extraction behavior consistent across tiers.
 *
 * Contract:
 *  - Never throws — returns needsManualReview=true on any failure.
 *  - All I/O runs on Dispatchers.IO.
 *  - Engine is lazily initialized on first extraction call.
 */
class LiteRtPartographExtractor(
    private val context: Context,
) : PartographExtractor {

    private var engine: Engine? = null
    private val initMutex = Mutex()

    override suspend fun extract(bitmap: Bitmap, sourceLabel: String): PartographExtraction =
        withContext(Dispatchers.IO) {
            runCatching { doExtract(bitmap) }.getOrElse { e ->
                Log.e(TAG, "extract failed for $sourceLabel: ${e.javaClass.simpleName}: ${e.message}", e)
                manualReview("litert_error:${e.javaClass.simpleName}")
            }
        }

    private suspend fun doExtract(bitmap: Bitmap): PartographExtraction {
        val eng = getOrInitEngine()
        val scaled = downscale(bitmap, MAX_IMAGE_DIM)

        val tempFile = File(context.cacheDir, "litert_input_${System.currentTimeMillis()}.jpg")
        try {
            tempFile.outputStream().use { out ->
                scaled.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
            if (scaled !== bitmap) scaled.recycle()

            val conversationConfig = ConversationConfig(
                samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
            )

            eng.createConversation(conversationConfig).use { conversation ->
                val response = collectResponse(conversation, tempFile.absolutePath)
                return parseContent(response)
            }
        } finally {
            tempFile.delete()
        }
    }

    private suspend fun collectResponse(conversation: Conversation, imagePath: String): String {
        val contents = Contents.of(
            Content.ImageFile(imagePath),
            Content.Text(EXTRACTION_PROMPT),
        )
        val chunks = conversation.sendMessageAsync(contents)
            .catch { e -> throw RuntimeException("LiteRT streaming error: ${e.message}", e) }
            .toList()
        return chunks.joinToString("") { it.toString() }
    }

    private suspend fun getOrInitEngine(): Engine {
        engine?.let { return it }
        return initMutex.withLock {
            engine?.let { return it }
            val modelPath = File(context.filesDir, MODEL_FILENAME).absolutePath
            check(File(modelPath).exists()) {
                "LiteRT model not found at $modelPath — download required"
            }
            val config = EngineConfig(
                modelPath = modelPath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU(),
                cacheDir = context.cacheDir.absolutePath,
            )
            val eng = Engine(config)
            eng.initialize()
            engine = eng
            eng
        }
    }

    /**
     * Release engine resources. Call when the extractor is no longer needed.
     */
    fun close() {
        engine?.close()
        engine = null
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
        private const val TAG = "PG_LiteRtExtractor"
        const val MODEL_FILENAME = "gemma-4-E2B-it.litertlm"
        private const val MAX_IMAGE_DIM = 1024

        /**
         * Extraction prompt — identical to RemotePartographExtractor and
         * partoguard/core/extraction/gemma_adapter.py.
         * DO NOT include <bos>; the tokenizer prepends it automatically.
         */
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

/** Round to nearest 0.5 increment (e.g. 3.7 -> 3.5, 3.8 -> 4.0). */
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

package com.partoguard.app.analyzer

import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import com.partoguard.app.model.ImageQuality
import com.partoguard.app.model.PartographExtraction
import com.partoguard.app.model.PartographPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

/**
 * Production extractor calling the fine-tuned Gemma 4 E2B-it model served by
 * llama.cpp llama-server on a remote host.
 *
 * Protocol:
 *   1. GET {baseUrl}/props          → fetch server-randomised media_marker (cached per instance)
 *   2. POST {baseUrl}/completion    → multimodal extraction request with base64 JPEG + prompt
 *   3. Parse {"p": [[h, d, c], …]} from response "content" field
 *
 * Contract (matches PartographExtractor interface):
 *   - Never throws — returns needsManualReview=true on any failure.
 *   - All I/O runs on Dispatchers.IO.
 *   - media_marker is cached for the lifetime of the server instance and
 *     invalidated automatically on /props errors.
 */
class RemotePartographExtractor(
    private val baseUrl: String,
) : PartographExtractor {

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(180, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    /** Cached per llama-server instance; re-fetched after any /props failure. */
    @Volatile private var cachedMarker: String? = null

    override suspend fun extract(bitmap: Bitmap, sourceLabel: String): PartographExtraction =
        withContext(Dispatchers.IO) {
            runCatching { doExtract(bitmap) }.getOrElse { e ->
                Log.e(TAG, "extract failed for $sourceLabel: ${e.javaClass.simpleName}: ${e.message}", e)
                manualReview("remote_error:${e.javaClass.simpleName}")
            }
        }

    private fun doExtract(bitmap: Bitmap): PartographExtraction {
        val marker = fetchMarker()
        val imageBytes = bitmap.toJpegBytes()
        val content = postCompletion(marker, imageBytes)
        return parseContent(content)
    }

    private fun fetchMarker(): String {
        cachedMarker?.let { return it }
        val url = "${baseUrl.trimEnd('/')}/props"
        val request = Request.Builder().url(url).get().build()
        val response = client.newCall(request).execute()
        check(response.isSuccessful) { "/props failed: HTTP ${response.code}" }
        val body = response.body?.string() ?: error("/props returned empty body")
        val marker = JSONObject(body).optString("media_marker")
            .takeIf { it.isNotBlank() } ?: error("no media_marker in /props response")
        cachedMarker = marker
        return marker
    }

    private fun postCompletion(marker: String, imageBytes: ByteArray): String {
        // Prompt format: Gemma 4 chat template WITHOUT <bos> (tokenizer adds it).
        val promptString = "<|turn>user\n$marker\n$EXTRACTION_PROMPT<turn|>\n<|turn>model\n"
        val payload = JSONObject().apply {
            put("prompt", JSONObject().apply {
                put("prompt_string", promptString)
                put("multimodal_data", JSONArray().apply {
                    put(Base64.encodeToString(imageBytes, Base64.NO_WRAP))
                })
            })
            put("n_predict", 400)
            put("temperature", 0.0)
            put("top_k", 1)
            put("cache_prompt", false)
        }
        val url = "${baseUrl.trimEnd('/')}/completion"
        val requestBody = payload.toString().toRequestBody(JSON_MEDIA_TYPE)
        val request = Request.Builder().url(url).post(requestBody).build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            // Invalidate marker on server errors — server may have restarted.
            if (response.code in 500..599) cachedMarker = null
            error("/completion failed: HTTP ${response.code}")
        }
        val responseBody = response.body?.string() ?: error("/completion returned empty body")
        return JSONObject(responseBody).optString("content", "")
    }

    /**
     * Parses compact extraction payload {"p": [[x_hours, dilation_cm, confidence], …]}.
     * Blank chart → {"p": []} → PartographExtraction with empty points list (not manual review).
     * Any parse error → manualReview with reason tag.
     */
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
                xHours    = x.coerceIn(0f, 12f).roundToHalf(),
                dilationCm = d.coerceIn(0f, 10f).roundToHalf(),
                confidence = c.coerceIn(0f, 1f),
            )
        }

        return PartographExtraction(
            chartSupported   = true,
            imageQuality     = ImageQuality(),
            points           = points.sortedWith(compareBy({ it.xHours }, { it.dilationCm })),
            needsManualReview = false,
        )
    }

    /** Strip markdown fences then parse; fall back to first-{…}-last-} slice. */
    private fun extractJsonObject(text: String): JSONObject? {
        var s = text.trim()
        if (s.startsWith("```")) {
            s = s.trimStart('`').trim()
            if (s.startsWith("json")) s = s.removePrefix("json").trim()
        }
        return runCatching { JSONObject(s) }.getOrElse {
            val start = s.indexOf('{')
            val end   = s.lastIndexOf('}')
            if (start == -1 || end <= start) return null
            runCatching { JSONObject(s.substring(start, end + 1)) }.getOrNull()
        }
    }

    private fun manualReview(reason: String) = PartographExtraction(
        chartSupported    = true,
        imageQuality      = ImageQuality(),
        points            = emptyList(),
        needsManualReview = true,
        reasonForManualReview = reason,
    )

    companion object {
        private const val TAG = "PG_RemoteExtractor"
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()

        /**
         * Extraction prompt — must match _build_remote_extraction_prompt() in
         * partoguard/core/extraction/gemma_adapter.py exactly.
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

/** Round to nearest 0.5 increment (e.g. 3.7 → 3.5, 3.8 → 4.0). */
private fun Float.roundToHalf(): Float = (this * 2f).roundToInt().toFloat() / 2f

/** Compress bitmap to JPEG bytes at the given quality (0–100). */
private fun Bitmap.toJpegBytes(quality: Int = 85): ByteArray {
    val out = ByteArrayOutputStream()
    compress(Bitmap.CompressFormat.JPEG, quality, out)
    return out.toByteArray()
}

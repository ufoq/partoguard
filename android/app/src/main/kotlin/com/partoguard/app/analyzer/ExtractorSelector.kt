package com.partoguard.app.analyzer

import android.app.ActivityManager
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import java.io.File

enum class InferenceMode(val key: String, val displayName: String, val subtitle: String, val minRamGb: Float) {
    OFFLINE_LITERT(
        "offline_litert",
        "Offline (LiteRT base)",
        "Google Gemma 4 E2B-it • ~2.6 GB • 4+ GB RAM",
        4f,
    ),
    OFFLINE_LLAMACPP(
        "offline_llamacpp",
        "Offline (llama.cpp Q8)",
        "Fine-tuned V7 Q8_0 • ~6 GB • 8+ GB RAM",
        8f,
    ),
    ONLINE_DEMO(
        "online_demo",
        "Online (demo server)",
        "Fine-tuned V7 on LAN • requires network",
        0f,
    ),
    ONLINE_CUSTOM(
        "online_custom",
        "Online (custom endpoint)",
        "Your own llama-server instance",
        0f,
    ),
    ;

    companion object {
        fun fromKey(key: String): InferenceMode =
            entries.firstOrNull { it.key == key } ?: ONLINE_DEMO
    }
}

object ExtractorSelector {

    private const val PREFS_NAME = "partoguard_settings"
    private const val PREF_MODE_KEY = "inference_mode"
    private const val PREF_CUSTOM_URL_KEY = "custom_endpoint_url"
    private const val PREF_SETUP_DONE = "setup_completed"
    private const val DEMO_SERVER_URL = "http://163.172.52.31:8080"

    fun isFirstLaunch(context: Context): Boolean {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return !prefs.getBoolean(PREF_SETUP_DONE, false)
    }

    fun markSetupCompleted(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_SETUP_DONE, true)
            .apply()
    }

    fun recommendedDefault(context: Context): InferenceMode {
        return if (totalRamGb(context) >= 7.5f) InferenceMode.OFFLINE_LLAMACPP else InferenceMode.ONLINE_DEMO
    }

    fun getMode(context: Context): InferenceMode {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return InferenceMode.fromKey(prefs.getString(PREF_MODE_KEY, InferenceMode.ONLINE_DEMO.key) ?: InferenceMode.ONLINE_DEMO.key)
    }

    fun setMode(context: Context, mode: InferenceMode) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_MODE_KEY, mode.key)
            .apply()
    }

    fun getCustomUrl(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(PREF_CUSTOM_URL_KEY, "") ?: ""
    }

    fun setCustomUrl(context: Context, url: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_CUSTOM_URL_KEY, url)
            .apply()
    }

    fun pick(context: Context): PartographExtractor {
        return when (getMode(context)) {
            InferenceMode.OFFLINE_LITERT -> liteRtExtractorOrFallback(context)
            InferenceMode.OFFLINE_LLAMACPP -> llamaCppExtractorOrFallback(context)
            InferenceMode.ONLINE_DEMO -> RemotePartographExtractor(DEMO_SERVER_URL)
            InferenceMode.ONLINE_CUSTOM -> {
                val url = getCustomUrl(context).ifBlank { DEMO_SERVER_URL }
                RemotePartographExtractor(url)
            }
        }
    }

    fun getDemoServerUrl(): String = DEMO_SERVER_URL

    private fun liteRtExtractorOrFallback(context: Context): PartographExtractor {
        val modelFile = File(context.filesDir, LiteRtPartographExtractor.MODEL_FILENAME)
        return if (modelFile.exists()) {
            LiteRtPartographExtractor(context)
        } else {
            RemotePartographExtractor(DEMO_SERVER_URL)
        }
    }

    private fun llamaCppExtractorOrFallback(context: Context): PartographExtractor {
        val ggufFile = File(context.filesDir, LlamaCppPartographExtractor.GGUF_FILENAME)
        val mmprojFile = File(context.filesDir, LlamaCppPartographExtractor.MMPROJ_FILENAME)
        return if (ggufFile.exists() && mmprojFile.exists()) {
            LlamaCppPartographExtractor(context)
        } else {
            RemotePartographExtractor(DEMO_SERVER_URL)
        }
    }

    fun isLiteRtModelDownloaded(context: Context): Boolean =
        File(context.filesDir, LiteRtPartographExtractor.MODEL_FILENAME).exists()

    fun isLlamaCppModelDownloaded(context: Context): Boolean {
        return File(context.filesDir, LlamaCppPartographExtractor.GGUF_FILENAME).exists() &&
            File(context.filesDir, LlamaCppPartographExtractor.MMPROJ_FILENAME).exists()
    }

    fun totalRamGb(context: Context): Float {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        return memInfo.totalMem / (1024f * 1024f * 1024f)
    }

    fun hasInternet(context: Context): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(network) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }
}

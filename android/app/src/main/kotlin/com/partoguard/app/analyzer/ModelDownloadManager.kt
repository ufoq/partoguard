package com.partoguard.app.analyzer

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

sealed class DownloadProgress {
    data class Downloading(val bytesRead: Long, val totalBytes: Long) : DownloadProgress() {
        val percent: Float get() = if (totalBytes > 0) bytesRead.toFloat() / totalBytes else 0f
    }
    data object Completed : DownloadProgress()
    data class Failed(val error: String) : DownloadProgress()
}

object ModelDownloadManager {

    private const val TAG = "PG_ModelDownload"

    private const val LITERT_MODEL_URL =
        "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm"

    private const val LLAMACPP_GGUF_URL =
        "https://huggingface.co/partoguard/gemma-4-e2b-it-partograph-v7-gguf/resolve/main/v7_q8_0.gguf"

    private const val LLAMACPP_MMPROJ_URL =
        "https://huggingface.co/partoguard/gemma-4-e2b-it-partograph-v7-gguf/resolve/main/v7_mmproj_f16.gguf"

    const val LLAMACPP_GGUF_FILENAME = "v7_q8_0.gguf"
    const val LLAMACPP_MMPROJ_FILENAME = "v7_mmproj_f16.gguf"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.MINUTES)
        .writeTimeout(10, TimeUnit.MINUTES)
        .followRedirects(true)
        .build()

    fun downloadLiteRtModel(context: Context): Flow<DownloadProgress> =
        downloadFile(
            url = LITERT_MODEL_URL,
            destFile = File(context.filesDir, LiteRtPartographExtractor.MODEL_FILENAME),
        )

    fun downloadLlamaCppModels(context: Context): Flow<DownloadProgress> = flow {
        val ggufFile = File(context.filesDir, LLAMACPP_GGUF_FILENAME)
        val mmprojFile = File(context.filesDir, LLAMACPP_MMPROJ_FILENAME)

        val pending = mutableListOf<Pair<String, File>>()
        if (!ggufFile.exists()) pending.add(LLAMACPP_GGUF_URL to ggufFile)
        if (!mmprojFile.exists()) pending.add(LLAMACPP_MMPROJ_URL to mmprojFile)

        if (pending.isEmpty()) {
            emit(DownloadProgress.Completed)
            return@flow
        }

        val fileSizes = pending.map { (url, _) -> contentLength(url) }
        val totalBytes = fileSizes.sum()
        var completedBytes = 0L

        for ((index, entry) in pending.withIndex()) {
            val (url, destFile) = entry
            val fileSize = fileSizes[index]
            var failed = false

            downloadFile(url, destFile).collect { progress ->
                when (progress) {
                    is DownloadProgress.Downloading -> {
                        val overall = completedBytes + progress.bytesRead
                        val overallTotal = if (totalBytes > 0) totalBytes else progress.totalBytes
                        emit(DownloadProgress.Downloading(overall, overallTotal))
                    }
                    is DownloadProgress.Failed -> {
                        emit(progress)
                        failed = true
                    }
                    is DownloadProgress.Completed -> {
                        completedBytes += fileSize
                    }
                }
            }
            if (failed) return@flow
        }

        emit(DownloadProgress.Completed)
    }.flowOn(Dispatchers.IO)

    private fun contentLength(url: String): Long {
        return try {
            val request = Request.Builder().url(url).head().build()
            val response = client.newCall(request).execute()
            val length = response.header("Content-Length")?.toLongOrNull() ?: -1L
            response.close()
            length
        } catch (e: Exception) {
            Log.w(TAG, "HEAD request failed for $url: ${e.message}")
            -1L
        }
    }

    private fun downloadFile(url: String, destFile: File): Flow<DownloadProgress> = flow {
        val tempFile = File(destFile.parent, "${destFile.name}.tmp")
        try {
            val request = Request.Builder().url(url).build()
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                emit(DownloadProgress.Failed("HTTP ${response.code}"))
                return@flow
            }
            val body = response.body ?: run {
                emit(DownloadProgress.Failed("Empty response body"))
                return@flow
            }
            val totalBytes = body.contentLength()
            var bytesRead = 0L

            tempFile.outputStream().use { out ->
                body.byteStream().use { input ->
                    val buffer = ByteArray(8192)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        out.write(buffer, 0, read)
                        bytesRead += read
                        emit(DownloadProgress.Downloading(bytesRead, totalBytes))
                    }
                }
            }

            tempFile.renameTo(destFile)
            emit(DownloadProgress.Completed)
        } catch (e: Exception) {
            Log.e(TAG, "Download failed: ${e.message}", e)
            tempFile.delete()
            emit(DownloadProgress.Failed(e.message ?: "Unknown error"))
        }
    }.flowOn(Dispatchers.IO)

    fun deleteLiteRtModel(context: Context) {
        File(context.filesDir, LiteRtPartographExtractor.MODEL_FILENAME).delete()
    }

    fun deleteLlamaCppModels(context: Context) {
        File(context.filesDir, LLAMACPP_GGUF_FILENAME).delete()
        File(context.filesDir, LLAMACPP_MMPROJ_FILENAME).delete()
    }

    fun liteRtModelSizeMb(): String = "~2,580 MB"
    fun llamaCppModelSizeMb(): String = "~5,920 MB"
}

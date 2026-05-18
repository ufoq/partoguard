package com.partoguard.llama

import android.content.Context
import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

class LlamaEngine(private val context: Context) {

    companion object {
        private var initialized = false

        init {
            System.loadLibrary("partoguard-llama")
        }
    }

    fun init() {
        if (!initialized) {
            val libDir = context.applicationInfo.nativeLibraryDir
            nativeInit(libDir)
            initialized = true
        }
    }

    fun loadModel(modelPath: String, mmprojPath: String, nCtx: Int = 4096, nThreads: Int = 4): Boolean {
        return nativeLoadModel(modelPath, mmprojPath, nCtx, nThreads)
    }

    fun complete(prompt: String, image: Bitmap?, temperature: Float = 0.1f, maxTokens: Int = 512): String {
        val rgbBytes: ByteArray?
        val width: Int
        val height: Int

        if (image != null) {
            width = image.width
            height = image.height
            rgbBytes = bitmapToRgb(image)
        } else {
            width = 0
            height = 0
            rgbBytes = null
        }

        return nativeComplete(prompt, rgbBytes, width, height, temperature, maxTokens)
    }

    fun stop() = nativeStop()

    fun release() = nativeRelease()

    private fun bitmapToRgb(bitmap: Bitmap): ByteArray {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val rgb = ByteArray(w * h * 3)
        for (i in pixels.indices) {
            val px = pixels[i]
            rgb[i * 3]     = ((px shr 16) and 0xFF).toByte() // R
            rgb[i * 3 + 1] = ((px shr 8) and 0xFF).toByte()  // G
            rgb[i * 3 + 2] = (px and 0xFF).toByte()          // B
        }
        return rgb
    }

    private external fun nativeInit(libDir: String)
    private external fun nativeLoadModel(modelPath: String, mmprojPath: String, nCtx: Int, nThreads: Int): Boolean
    private external fun nativeComplete(prompt: String, imageRgb: ByteArray?, imgWidth: Int, imgHeight: Int, temperature: Float, maxTokens: Int): String
    private external fun nativeStop()
    private external fun nativeRelease()
}

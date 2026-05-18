package com.partoguard.app.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory

/** Tiny helper to load bitmaps from the assets/ directory. */
object AssetLoader {
    fun loadBitmap(context: Context, path: String): Bitmap? {
        return runCatching {
            context.assets.open(path).use { BitmapFactory.decodeStream(it) }
        }.getOrNull()
    }
}

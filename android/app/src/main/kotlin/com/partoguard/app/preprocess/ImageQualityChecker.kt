package com.partoguard.app.preprocess

import android.graphics.Bitmap
import com.partoguard.app.model.ImageQuality

/**
 * Mock image quality preprocessor. The real implementation will use Laplacian
 * variance (blur), mean luminance (dim), and Hough/horizon detection (skew)
 * before the model ever sees the image. Today it only piggy-backs on the
 * source label so demo flows can show the UI affordances.
 */
object ImageQualityChecker {
    fun check(bitmap: Bitmap?, sourceLabel: String): ImageQuality {
        val lower = sourceLabel.lowercase()
        return ImageQuality(
            blurry = "blur" in lower,
            dim = "alert" in lower || "dim" in lower,
            skewed = "action" in lower || "obstructed" in lower || "skew" in lower,
        )
    }
}

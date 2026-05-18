package com.partoguard.app.ui.theme

import androidx.compose.ui.graphics.Color

// Brand — deep clinical navy, no pastels.
val BrandPrimary = Color(0xFF0B4D78)
val BrandPrimaryDark = Color(0xFF06324F)
val BrandSurface = Color(0xFFFBFCFE)
val BrandOnSurface = Color(0xFF0B1F2D)
val BrandSurfaceVariant = Color(0xFFE7EEF4)
val BrandOnSurfaceVariant = Color(0xFF3B515F)

// Clinical status palette — saturated, high-contrast on white. Reviewed for
// WCAG AA on BrandSurface (#FBFCFE).
val StatusNormal = Color(0xFF0E7A47)   // deep green
val StatusAlert = Color(0xFFC97A00)    // deep amber (was washed-out gold)
val StatusAction = Color(0xFFB31412)   // deep red, closer to clinical "STOP"
val StatusManual = Color(0xFF4A5E6B)   // neutral slate

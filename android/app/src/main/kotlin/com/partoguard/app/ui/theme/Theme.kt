package com.partoguard.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColors = lightColorScheme(
    primary = BrandPrimary,
    onPrimary = BrandSurface,
    primaryContainer = BrandPrimaryDark,
    onPrimaryContainer = BrandSurface,
    surface = BrandSurface,
    onSurface = BrandOnSurface,
    surfaceVariant = BrandSurfaceVariant,
    onSurfaceVariant = BrandOnSurfaceVariant,
    background = BrandSurface,
    onBackground = BrandOnSurface,
    error = StatusAction,
    onError = BrandSurface,
)

private val DarkColors = darkColorScheme(
    primary = BrandPrimary,
    onPrimary = BrandSurface,
    primaryContainer = BrandPrimaryDark,
    onPrimaryContainer = BrandSurface,
    surface = BrandPrimaryDark,
    onSurface = BrandSurface,
    background = BrandPrimaryDark,
    onBackground = BrandSurface,
    error = StatusAction,
    onError = BrandSurface,
)

@Composable
fun PartoGuardTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) DarkColors else LightColors
    MaterialTheme(colorScheme = colors, typography = PartoGuardTypography, content = content)
}

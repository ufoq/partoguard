package com.partoguard.app.ui.camera

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.partoguard.app.R
import com.partoguard.app.session.AnalysisSession
import com.partoguard.app.util.AssetLoader

private data class CameraOverride(val outcome: String, val assetPath: String)

private val OVERRIDE_ASSETS: Map<String, CameraOverride> = mapOf(
    "NORMAL" to CameraOverride("NORMAL", "demo_partographs/WHO-normal.png"),
    "ALERT" to CameraOverride("ALERT", "demo_partographs/WHO-alert.png"),
    "ACTION" to CameraOverride("ACTION", "demo_partographs/WHO-action.png"),
    "EMPTY" to CameraOverride("EMPTY", "demo_partographs/WHO-blank.png"),
    "PT1" to CameraOverride("PT1", "demo_partographs/WHO-pt1.png"),
)

@Composable
fun CameraScreen(
    session: AnalysisSession,
    onCaptured: () -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Latched at entry: extractor consumes forcedOutcome later, so re-reading
    // it would risk preview/outcome desync. Null = real camera path.
    val override = remember { session.forcedOutcome?.let { OVERRIDE_ASSETS[it] } }

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA,
            ) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    val controller = remember {
        LifecycleCameraController(context).apply {
            setEnabledUseCases(CameraController.IMAGE_CAPTURE)
            bindToLifecycle(lifecycleOwner)
        }
    }

    var capturing by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var previewSize by remember { mutableStateOf(IntSize.Zero) }
    val captureFailedText = stringResource(R.string.camera_capture_failed)
    val density = LocalDensity.current
    val navBarBottom = with(density) { WindowInsets.navigationBars.getBottom(density).toDp() }

    Scaffold(
        contentWindowInsets = WindowInsets.safeDrawing,
    ) { padding ->
        Box(
            Modifier
                .fillMaxSize()
                .padding(padding)
                .onSizeChanged { previewSize = it }
                .background(Color.Black),
        ) {
            if (hasPermission) {
                AndroidView(
                    modifier = Modifier.fillMaxSize(),
                    factory = { ctx ->
                        PreviewView(ctx).apply {
                            this.controller = controller
                        }
                    },
                )
                ViewfinderOverlay(modifier = Modifier.fillMaxSize())
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.TopCenter)
                        .windowInsetsPadding(WindowInsets.statusBars)
                        .padding(start = 4.dp, end = 16.dp, top = 4.dp, bottom = 4.dp),
                ) {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(R.string.cd_back),
                        )
                    }
                    Spacer(Modifier.width(4.dp))
                    Text(
                        stringResource(R.string.home_action_live),
                        style = MaterialTheme.typography.titleLarge,
                        color = Color.White,
                    )
                }
                Text(
                    text = stringResource(R.string.camera_align_hint),
                    style = MaterialTheme.typography.labelLarge,
                    color = Color.White,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 64.dp, start = 16.dp, end = 16.dp)
                        .background(
                            color = Color.Black.copy(alpha = 0.45f),
                            shape = RoundedCornerShape(50),
                        )
                        .padding(horizontal = 14.dp, vertical = 6.dp),
                )
                Text(
                    text = stringResource(R.string.camera_lighting_hint),
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.White.copy(alpha = 0.85f),
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 96.dp, start = 16.dp, end = 16.dp)
                        .background(
                            color = Color.Black.copy(alpha = 0.45f),
                            shape = RoundedCornerShape(50),
                        )
                        .padding(horizontal = 12.dp, vertical = 5.dp),
                )
                Surface(
                    color = MaterialTheme.colorScheme.background,
                    tonalElevation = 4.dp,
                    modifier = Modifier.align(Alignment.BottomCenter),
                ) {
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .padding(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 32.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        errorMessage?.let { msg ->
                            Text(
                                msg,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.error,
                            )
                        }
                        Button(
                            onClick = {
                                if (capturing) return@Button
                                errorMessage = null
                                capturing = true
                                controller.takePicture(
                                    ContextCompat.getMainExecutor(context),
                                    object : ImageCapture.OnImageCapturedCallback() {
                                        override fun onCaptureSuccess(image: ImageProxy) {
                                            val bmp = if (override != null) {
                                                image.close()
                                                AssetLoader.loadBitmap(context, override.assetPath)
                                            } else {
                                                try {
                                                    val rotationDegrees = image.imageInfo.rotationDegrees
                                                    val raw = image.toBitmap()
                                                    image.close()
                                                    val rotated = applyRotation(raw, rotationDegrees)
                                                    cropToOverlay(rotated, previewSize)
                                                } catch (t: Throwable) {
                                                    try { image.close() } catch (_: Throwable) {}
                                                    null
                                                }
                                            }
                                            capturing = false
                                            if (bmp == null) {
                                                errorMessage = captureFailedText
                                                return
                                            }
                                            session.bitmap = bmp
                                            session.sourceLabel = if (override != null) {
                                                "live_capture_override_${override.outcome}_${System.currentTimeMillis()}"
                                            } else {
                                                "live_capture_${System.currentTimeMillis()}"
                                            }
                                            onCaptured()
                                        }

                                        override fun onError(exception: ImageCaptureException) {
                                            capturing = false
                                            errorMessage = exception.message ?: captureFailedText
                                        }
                                    },
                                )
                            },
                            enabled = hasPermission && !capturing,
                            modifier = Modifier.fillMaxWidth().height(52.dp),
                            shape = RoundedCornerShape(14.dp),
                        ) {
                            Text(stringResource(R.string.camera_capture))
                        }
                    }
                }
            } else {
                Column(
                    Modifier.fillMaxSize().padding(24.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        stringResource(R.string.camera_permission_title),
                        style = MaterialTheme.typography.titleLarge,
                        color = Color.White,
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        stringResource(R.string.camera_permission_rationale),
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color.White,
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                        Text(stringResource(R.string.camera_permission_grant))
                    }
                }
            }
        }
    }
}

@Composable
private fun ViewfinderOverlay(modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val frameW = w * 0.86f
        val frameH = frameW
        val left = (w - frameW) / 2f
        val top = (h - frameH) / 2f
        val cornerLen = frameW * 0.08f
        val stroke = 4f
        val brand = Color(0xFFFFFFFF)

        drawRect(
            color = Color.Black.copy(alpha = 0.35f),
            topLeft = Offset(0f, 0f),
            size = Size(w, top),
        )
        drawRect(
            color = Color.Black.copy(alpha = 0.35f),
            topLeft = Offset(0f, top + frameH),
            size = Size(w, h - (top + frameH)),
        )
        drawRect(
            color = Color.Black.copy(alpha = 0.35f),
            topLeft = Offset(0f, top),
            size = Size(left, frameH),
        )
        drawRect(
            color = Color.Black.copy(alpha = 0.35f),
            topLeft = Offset(left + frameW, top),
            size = Size(w - (left + frameW), frameH),
        )

        drawRect(
            color = brand.copy(alpha = 0.4f),
            topLeft = Offset(left, top),
            size = Size(frameW, frameH),
            style = Stroke(width = 1.5f),
        )

        val tl = Offset(left, top)
        val tr = Offset(left + frameW, top)
        val bl = Offset(left, top + frameH)
        val br = Offset(left + frameW, top + frameH)
        listOf(
            tl to Offset(tl.x + cornerLen, tl.y),
            tl to Offset(tl.x, tl.y + cornerLen),
            tr to Offset(tr.x - cornerLen, tr.y),
            tr to Offset(tr.x, tr.y + cornerLen),
            bl to Offset(bl.x + cornerLen, bl.y),
            bl to Offset(bl.x, bl.y - cornerLen),
            br to Offset(br.x - cornerLen, br.y),
            br to Offset(br.x, br.y - cornerLen),
        ).forEach { (a, b) ->
            drawLine(color = brand, start = a, end = b, strokeWidth = stroke)
        }
    }
}

private fun applyRotation(bitmap: Bitmap, degrees: Int): Bitmap {
    if (degrees == 0) return bitmap
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

private fun cropToOverlay(bitmap: Bitmap, viewSize: IntSize): Bitmap {
    if (viewSize.width == 0 || viewSize.height == 0) return bitmap

    val imgW = bitmap.width.toFloat()
    val imgH = bitmap.height.toFloat()
    val vW = viewSize.width.toFloat()
    val vH = viewSize.height.toFloat()

    // FILL_CENTER: scale image so both dimensions fill the view
    val scale = maxOf(vW / imgW, vH / imgH)
    val scaledW = imgW * scale
    val scaledH = imgH * scale

    val offsetX = (scaledW - vW) / 2f
    val offsetY = (scaledH - vH) / 2f

    // Must match ViewfinderOverlay constants
    val frameW = vW * 0.86f
    val frameH = frameW
    val overlayLeft = (vW - frameW) / 2f
    val overlayTop = (vH - frameH) / 2f

    val cropLeft = ((overlayLeft + offsetX) / scale).toInt().coerceIn(0, bitmap.width - 1)
    val cropTop = ((overlayTop + offsetY) / scale).toInt().coerceIn(0, bitmap.height - 1)
    val cropWidth = (frameW / scale).toInt().coerceAtMost(bitmap.width - cropLeft)
    val cropHeight = (frameH / scale).toInt().coerceAtMost(bitmap.height - cropTop)

    if (cropWidth <= 0 || cropHeight <= 0) return bitmap

    return Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropWidth, cropHeight)
}

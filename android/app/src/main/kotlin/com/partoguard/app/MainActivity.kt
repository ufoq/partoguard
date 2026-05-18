package com.partoguard.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.partoguard.app.nav.PartoGuardNavGraph
import com.partoguard.app.ui.theme.PartoGuardTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as PartoGuardApp
        setContent {
            PartoGuardTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    PartoGuardNavGraph(
                        session = app.session,
                    )
                }
            }
        }
    }
}

package com.partoguard.app.ui.debug

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.selectable
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.partoguard.app.R

@Composable
fun DebugOverrideDialog(
    initial: String?,
    onApply: (String?) -> Unit,
    onDismiss: () -> Unit,
    onRunEval: () -> Unit,
) {
    var selected by remember { mutableStateOf(initial) }
    val options = listOf(
        null to stringResource(R.string.debug_option_default),
        "NORMAL" to stringResource(R.string.debug_option_normal),
        "ALERT" to stringResource(R.string.debug_option_alert),
        "ACTION" to stringResource(R.string.debug_option_action),
        "EMPTY" to stringResource(R.string.debug_option_empty),
        "PT1" to "Action 2 (PT1)",
    )
    AlertDialog(
        onDismissRequest = {
            onApply(selected)
            onDismiss()
        },
        title = { Text(stringResource(R.string.debug_title)) },
        text = {
            Column {
                Text(
                    stringResource(R.string.debug_subtitle),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 12.dp),
                )
                options.forEach { (value, label) ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = value == selected,
                                onClick = { selected = value },
                            )
                            .padding(vertical = 6.dp),
                    ) {
                        RadioButton(
                            selected = value == selected,
                            onClick = { selected = value },
                        )
                        Text(label, modifier = Modifier.padding(start = 8.dp))
                    }
                }
                TextButton(
                    onClick = onRunEval,
                    modifier = Modifier.padding(top = 8.dp),
                ) {
                    Text("Run eval suite \u2192")
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onApply(selected)
                onDismiss()
            }) {
                Text(stringResource(R.string.debug_close))
            }
        },
    )
}

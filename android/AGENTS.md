# android/ — Kotlin/Compose Android App

Compose-based clinical UI. ML is mocked; `PartographExtractor` interface is the sole swap point.

## STRUCTURE

```
android/
├── app/src/main/kotlin/com/partoguard/app/
│   ├── MainActivity.kt                    # Entry; initialises PartoGuardApp + NavGraph
│   ├── PartoGuardApp.kt                   # Application singleton — swap extractor here
│       ├── analyzer/
│   │   ├── PartographExtractor.kt         # INTERFACE: suspend extract(Bitmap, String) → PartographExtraction
│   │   ├── MockPartographExtractor.kt     # Canned outcomes (NORMAL/ALERT/ACTION/EMPTY/PT1); 1.5s latency
│   │   └── RemotePartographExtractor.kt   # HTTP extractor: GET /props → POST /completion → parse {"p":[[h,d,c],…]}
│   ├── model/Models.kt                    # PartographPoint, ImageQuality, ClinicalStatus, ClinicalAlert, WorkflowMode
│   ├── rules/RuleEngine.kt                # Deterministic rules — mirrors Python classify_zone()
│   ├── session/AnalysisSession.kt         # Process-singleton; bitmap, extraction, alert, mode, forcedOutcome
│   ├── nav/NavGraph.kt                    # 5 routes; AUTO/ASSISTED/MANUAL flow branching
│   ├── ui/
│   │   ├── home/HomeScreen.kt             # Mode chips, demo samples, live capture CTA, debug tap trigger
│   │   ├── camera/CameraScreen.kt         # CameraX preview + capture; override injects mock on tap
│   │   ├── analyzing/AnalyzingScreen.kt   # Quality → extract → validate pipeline UI
│   │   ├── review/ReviewScreen.kt         # Editable point table (ASSISTED/MANUAL)
│   │   ├── results/ResultsScreen.kt       # Verdict card + animated sweep overlay
│   │   ├── components/Components.kt       # SeverityBadge, QualityChips, ConfidencePill, statusColor()
│   │   └── debug/DebugOverrideDialog.kt   # Hidden: 7 taps on AUTO chip within 5s
│   ├── preprocess/ImageQualityChecker.kt  # Mock (reads sourceLabel for blurry/dim/skewed flags)
│   └── util/AssetLoader.kt
├── app/src/main/assets/demo_partographs/  # 5 baked-in demo images (normal, alert, action, blank, pt1)
├── app/src/main/AndroidManifest.xml       # CAMERA permission; usesCleartextTraffic=false
├── gradle/libs.versions.toml              # Version catalog (AGP 8.7.3, Kotlin 2.0.21, CameraX 1.4.0)
└── README.md                              # Build, architecture, LiteRT swap path
```

## WHERE TO LOOK

| Task | File |
|------|------|
| Swap in real ML (LiteRT or remote) | `PartoGuardApp.kt` — replace `extractor` lazy val; `GEMMA_SERVER_BASE_URL` constant sets target |
| Add `RemotePartographExtractor` | Done — `analyzer/RemotePartographExtractor.kt`; wire via `PartoGuardApp.GEMMA_SERVER_BASE_URL` |
| Change clinical rules | `rules/RuleEngine.kt::evaluate()` |
| Add a screen | `nav/NavGraph.kt` + new composable in `ui/` |
| Shared screen state | `session/AnalysisSession.kt` |
| Add demo asset | `assets/demo_partographs/` + `DEMO_SAMPLES` list in `HomeScreen.kt` |
| Add debug outcome option | `DebugOverrideDialog.kt::options` + `MockPartographExtractor.Outcome` enum |
| Add eval test sample | `assets/eval_samples/manifest.json` + corresponding PNG; mirror `partoguard/core/corpus_scorer.py` truth |
| Run eval suite | Open debug menu (7 taps on AUTO chip) → "Run eval suite →"; results stream to logcat tag `PARTOGUARD_EVAL` |
| Change navigation flow | `NavGraph.kt::navigateAfterCapture()` |
| Add HTTP networking | `gradle/libs.versions.toml` (add okhttp) + `app/build.gradle.kts` + `AndroidManifest.xml` (INTERNET permission + `network_security_config.xml` for HTTP exemption) |

## CONVENTIONS

- **`PartographExtractor.extract()` must never throw** — return `needsManualReview=true` on any failure; route to `MANUAL_REVIEW`.
- **`AnalysisSession` is the ONLY shared state** — screens communicate via session fields, never through nav args (Bitmaps don't survive serialization).
- **`RuleEngine` stays a single auditable file** — no clinical logic elsewhere in Android code.
- **`session.forcedOutcome` is consumed once** — `MockPartographExtractor.consumeForcedOutcome()` clears it immediately after reading.
- **Camera override behaviour**: real preview always shown; mock Bitmap injected at `onCaptureSuccess`, not at preview. `OVERRIDE_ASSETS` map in `CameraScreen.kt`.
- **`usesCleartextTraffic="true"` in manifest** — HTTP is enabled for now; upgrade to HTTPS + TOFU trust manager when ready.
- **`GEMMA_SERVER_BASE_URL` in `PartoGuardApp.kt`** — single source of truth for the server URL; `http://192.168.0.33:56080` for device on local network, `http://vps-box:8080` for dev machine direct.
- **`RemotePartographExtractor` media_marker cache** — valid for the lifetime of one llama-server instance; auto-invalidated on HTTP 5xx.
- **Release signing** uses `demo.jks` at `/root/keys/demo.jks` (password: `partoguard123`) — demo creds, not production.

## NAVIGATION FLOW

```
AUTO     : HOME → (CAMERA) → ANALYZING → RESULTS
ASSISTED : HOME → (CAMERA) → ANALYZING → REVIEW → RESULTS
MANUAL   : HOME → REVIEW (empty seed) → RESULTS
```
`RuleEngine.evaluate()` runs in `AnalyzingScreen` for AUTO, in `ReviewScreen` for ASSISTED/MANUAL.

## ANTI-PATTERNS

- ❌ Passing `Bitmap` through nav args — always store in `AnalysisSession.bitmap`
- ❌ Running `RuleEngine.evaluate()` before midwife confirmation in ASSISTED mode
- ❌ ML/HTTP calls on the main thread — use `withContext(Dispatchers.IO)` inside the extractor
- ❌ Adding `INTERNET` permission without updating `network_security_config.xml` (cleartext blocked by default)
- ❌ Accessing `session.forcedOutcome` after `CameraScreen` entry — it is latched at entry to prevent preview/outcome desync

## COMMANDS

```bash
cd android
./gradlew assembleDebug          # Debug APK
./gradlew installDebug           # Install on connected device
./gradlew assembleRelease        # Release APK (needs /root/keys/demo.jks)
./gradlew lint
```

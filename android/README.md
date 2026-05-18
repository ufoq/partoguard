# PartoGuard Android (scaffold)

Clinical decision-support prototype for partograph reading. UI scaffold only —
real ML integration is intentionally deferred.

## Status

| Area | Status |
|---|---|
| Compose UI: Home, Camera, Analyzing, Review, Results | Done (mocked flow) |
| CameraX preview + capture | Done |
| Mock `PartographExtractor` returning canned outcomes | Done |
| Deterministic `RuleEngine` for alert/action zone | Done |
| Image quality preprocessing | Mock |
| LiteRT-LM Gemma 4 E2B-it integration | Deferred (swap point: `PartographExtractor`) |
| Persistence / audit log | Deferred |
| Tests | Deferred |

## Build

Open `android/` as a project in Android Studio (Iguana / Koala or newer), let
it sync, then `Run` on a device or emulator (API 26+).

CLI:

```
cd android
./gradlew assembleDebug
```

Note: `gradle/wrapper/gradle-wrapper.jar` is intentionally not committed —
Android Studio will fetch it on first sync, or run `gradle wrapper` once with
a local Gradle 8.10.2 install.

## Project layout

```
app/src/main/
  AndroidManifest.xml
  assets/demo_partographs/      # 4 baked-in sample partographs
  kotlin/com/partoguard/app/
    MainActivity.kt
    PartoGuardApp.kt            # Application; lazy extractor + session
    analyzer/
      PartographExtractor.kt    # interface — swap point for real Gemma
      MockPartographExtractor.kt
    model/Models.kt             # PartographPoint, ImageQuality, ClinicalStatus, etc.
    nav/NavGraph.kt
    preprocess/ImageQualityChecker.kt
    rules/RuleEngine.kt         # deterministic alert/action zone logic
    session/AnalysisSession.kt  # process-singleton between screens
    ui/
      analyzing/AnalyzingScreen.kt
      camera/CameraScreen.kt
      components/Components.kt
      home/HomeScreen.kt
      results/ResultsScreen.kt
      review/ReviewScreen.kt
      theme/{Color,Theme,Type}.kt
    util/AssetLoader.kt
  res/
    drawable/, mipmap-anydpi-v26/, values/, xml/
```

## Architecture (mirrors knowledge/partoguard_implementation_plan.md)

```
Camera or Demo asset
  -> ImageQualityChecker (mock today; Laplacian + luminance later)
  -> PartographExtractor (mock today; Gemma 4 E2B-it via LiteRT-LM later)
  -> ReviewScreen (midwife confirms / edits points; mandatory in ASSISTED)
  -> RuleEngine.evaluate (deterministic, auditable)
  -> ResultsScreen (severity-coloured alert + overlayed dots)
```

The model never decides clinical action. It only extracts points. The rule
engine — readable in a single Kotlin file — owns every clinical decision.

## Workflow modes

- **AUTO**: extract -> brief confirm -> rules
- **ASSISTED**: extract -> edit table -> rules
- **MANUAL**: skip extraction, midwife enters points directly

## Swapping in the real model

Replace `PartoGuardApp.extractor` with a `LiteRtPartographExtractor` that
loads `assets/model/partoguard-gemma4-e2b-it.litertlm` (see
`knowledge/litert_export_setup.md`). The interface contract is the entire
boundary; nothing else in the app needs to change.

## Disclaimer

Decision support only. Not a medical device. Always verify with clinical
judgment.

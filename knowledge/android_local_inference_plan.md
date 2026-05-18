# Android Local Inference Plan

Three-tier on-device + remote inference strategy for the PartoGuard Android app, with verified evidence behind every architectural decision. Drafted 2026-05-18 after a full verification round on Maven artifacts, GitHub issues, eval logs, and source code.

> Canonical references: `partoguard_remote_gemma_vps.md` (VPS), `litert_export_setup.md` (export pipeline). This document covers **Android-side runtime selection only**, not the export pipeline.

---

## 1. Verified Findings (the evidence base)

### 1.1 What works today

| Fact | Status | Evidence |
|---|---|---|
| `com.google.ai.edge.litertlm:litertlm-android:0.11.0` is on Google Maven | **Supported** | Direct POM fetch from `dl.google.com/android/maven2/com/google/ai/edge/litertlm/litertlm-android/0.11.0/litertlm-android-0.11.0.pom` |
| Pre-built `litert-community/gemma-4-E2B-it-litert-lm` runs on Android arm64 | **Supported** | HF model card + Google AI Edge Gallery sample app |
| Google E2B-it LiteRT (base) scores 86/100 on our 350-image corpus | **Supported** | `knowledge/README.md` line 45, `eval_logs_v7_remote_*` peers |
| V7 fine-tuned GGUF Q8_0 on VPS scores **335/350 = 95.71%** at 8.08 s/img | **Supported** | `eval_logs_v7_remote_q8_0_350.log` lines 357-364 |
| V7 fine-tuned GGUF bf16 on VPS scores **336/350 = 96.00%** at 10.09 s/img | **Supported** | `eval_logs_v7_remote_bf16_350.log` lines 357-364 |
| V7 bf16 local HF/CUDA on RTX 5090 scores 350/350 = 100% at 5.77 s/img | **Supported** | `partoguard_remote_gemma_vps.md` line 17 |
| `io.github.ljcamargo:llamacpp-kotlin:0.4.0` is on Maven Central, packaging=aar | **Supported** | `repo1.maven.org/maven2/io/github/ljcamargo/llamacpp-kotlin/maven-metadata.xml` |
| kotlinllamacpp 0.4.0 supports mmproj vision via FD-based loading on Android | **Supported** | `gemma4v.cpp` + `jni.cpp:267-290` + `LlamaHelper.kt` ContentResolver path |

### 1.2 Critical blockers found

| Blocker | Evidence |
|---|---|
| **`ufoq/partoguard-litert-v7-wi4-v3` will likely SIGSEGV on Android** in `RunPrefillAsync` | [`litert-torch#998`](https://github.com/google-ai-edge/litert-torch/issues/998) (OPEN): KV cache dtype mismatch (float32 export vs int8 expected), missing `param_tensor` input, `GenericDataProcessor` instead of `Gemma4DataProcessor`. The JNI is a thin wrapper, the crash lives in the shared C++ runtime → Linux segfault → Android segfault. Confirmed across reporters. |
| **V7 LiteRT has never been benchmarked on the 350-image corpus** | Only V5/V6 partial evals exist (`litert_v5_eval_100.log`, `litert_v6_eval_30.log`, `litert_v6_eval_50_mixed.log`). `litert_export_setup.md` lines 315-321 list "Run 100-image eval, then 350 if 100% passes" as future work. |
| **LiteRT-LM Gallery requires `minSdk=31`** | `github.com/google-ai-edge/gallery/blob/main/Android/src/app/build.gradle.kts` declares `minSdk = 31`. AndroidManifest enforces same. Below 31, the `<uses-native-library>` mechanism for vendor driver loading is unavailable. |
| **kotlinllamacpp `getFormattedChat()` is a JNI stub returning `""`** | `jni.cpp:336-352` always returns `env->NewStringUTF("")`. Means we must format Gemma 4 chat turns manually. |
| **Gemma 4 vision encoder export not supported in litert-torch yet** | Google contributor in `litert-torch#998` (2026-05-13): *"Vision encoder export currently is not supported yet but on our radar."* |

### 1.3 Decisions made (2026-05-18)

| Decision | Rationale |
|---|---|
| **Tier 1 uses Google base E2B-it LiteRT** (not our V7) | V7 LiteRT bundle has structural runtime mismatches that segfault the C++ engine. Binary-patching the pre-built shell is undocumented and fragile. Base E2B-it is officially supported, runs today, well-benchmarked at 86%. |
| **Bump app `minSdk` from 26 to 31** | Required by LiteRT-LM. Aligns with Google's reference Gallery app. Excludes Android 7-11; clinical low-resource device impact accepted as a known tradeoff. |
| **Tier 2 uses `kotlinllamacpp:0.4.0`** + V7 GGUF Q8_0 + V7 mmproj_f16 | Same model artifacts as VPS deployment (`partoguard_remote_gemma_vps.md`). FD-based scoped storage works. mmproj vision wired through `gemma4v.cpp`. Manual chat formatting accepted as a tradeoff. |
| **Tier 3 (remote) keeps the existing `RemotePartographExtractor`** | Already shipped, demo default, no changes needed. |

---

## 2. Three-Tier Architecture

### 2.1 Tier matrix

| Tier | Backend | Model | Disk | RAM | Expected Accuracy | Target Devices | minSdk |
|---|---|---|---|---|---|---|---|
| **1. Local lite** | LiteRT-LM v0.11.0 | `litert-community/gemma-4-E2B-it-litert-lm` (base, no fine-tune) | 2.58 GB | ~1.7 GB CPU / ~0.7 GB CPU+GPU | **86%** (Supported) | 4-6 GB RAM Android 12+ | 31 |
| **2. Local quality** | kotlinllamacpp 0.4.0 | `v7_q8_0.gguf` (4.93 GB) + `v7_mmproj_f16.gguf` (986 MB) | 5.92 GB | ~7-8 GB | **~95.71%** (Prototype assumption — same artifact as VPS Q8_0) | 8+ GB RAM Android 12+ | 31 |
| **3. Remote** | OkHttp → `llama-server` | Whatever the server hosts (currently V7 Q8_0) | 0 | 0 | **95.71%** (Supported) | Any (needs network) | 31 |

> "Prototype assumption" on Tier 2 accuracy: the Q8_0 + mmproj artifacts are byte-identical to the VPS. The runtime path differs (kotlinllamacpp wraps `cui-llama.rn` while VPS uses upstream `llama-server`), so empirical eval is required to confirm the 95.71% figure transfers.

### 2.2 Why these specific tiers

- **Tier 1 covers low-end devices** that would otherwise have to use remote-only. Quality drops to base 86% — a real regression vs VPS 95.71% — but offline use is preserved.
- **Tier 2 matches VPS quality exactly** (same GGUF, same quant) for high-end devices that can afford 6 GB on disk and 8 GB RAM.
- **Tier 3 is the universal fallback**: works on any device with internet, model can be upgraded server-side independently of the app.

### 2.3 Runtime selection logic

Pseudocode for `PartoGuardApp` extractor selection (to live in `analyzer/ExtractorSelector.kt`):

```kotlin
fun pickExtractor(ctx: Context, prefs: SharedPreferences): PartographExtractor {
    val mode = prefs.getString("inference_mode", "auto")
    return when (mode) {
        "remote" -> RemotePartographExtractor(BASE_URL)
        "local_lite" -> LiteRtPartographExtractor(ctx, baseModel)
        "local_quality" -> LlamaCppPartographExtractor(ctx, v7Q8Gguf, mmprojF16)
        else -> autoSelect(ctx) // auto: see below
    }
}

private fun autoSelect(ctx: Context): PartographExtractor {
    val totalRam = ctx.totalRamGb()
    val sdkOk = Build.VERSION.SDK_INT >= 31
    val hasNetwork = ctx.hasInternet()
    return when {
        // High-end + offline-first: prefer local_quality
        totalRam >= 8 && sdkOk && llamaCppModelDownloaded(ctx) ->
            LlamaCppPartographExtractor(ctx, v7Q8Gguf, mmprojF16)
        // Low-end + offline-first: local_lite
        totalRam >= 4 && sdkOk && liteRtModelDownloaded(ctx) ->
            LiteRtPartographExtractor(ctx, baseModel)
        // Otherwise: remote
        hasNetwork -> RemotePartographExtractor(BASE_URL)
        // No network, no model: manual review
        else -> ManualReviewExtractor()
    }
}
```

A user-facing "AI engine" setting in Settings overrides auto-selection.

---

## 3. Implementation Plan (sequenced)

### Phase A — Foundation (small, low-risk)

**Goal: bump minSdk and verify nothing breaks.**

- A1. Bump `minSdk = 31` in `android/app/build.gradle.kts` and `gradle/libs.versions.toml`.
- A2. Bump `targetSdk` to current (36 to match Gallery reference).
- A3. Update `AGENTS.md` Android section to document new minSdk floor.
- A4. Smoke-test existing flows on the SM-M236B device (Android 14, all current paths still work).

**Acceptance**: app builds, installs, all manual-edit/demo/remote paths still work.

### Phase B — Tier 1 (LiteRT base E2B-it)

**Goal: ship Google base E2B-it LiteRT as Tier 1.**

- B1. Add Gradle dep: `implementation("com.google.ai.edge.litertlm:litertlm-android:0.11.0")`
- B2. Add to `AndroidManifest.xml`:
  ```xml
  <uses-native-library android:name="libvndksupport.so" android:required="false"/>
  <uses-native-library android:name="libOpenCL.so" android:required="false"/>
  ```
- B3. Implement `LiteRtPartographExtractor : PartographExtractor`:
  - Engine init with `Backend.GPU()` and CPU fallback.
  - **Use raw `Session` API**, not Conversation API — issue #2078 still open.
  - Manually construct Gemma 4 chat turns (`<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`).
  - Image input: pass via `addImage(Bitmap)` on Session, GPU vision backend only (#2056).
  - Reuse the **exact same extraction prompt** as `RemotePartographExtractor` to keep prompts in lockstep.
- B4. Model delivery: download `gemma-4-E2B-it.litertlm` (2.58 GB) on first-run to `ctx.filesDir`. Show progress UI. Verify SHA256.
- B5. Wire selector: add "Local (lite)" option to Settings.

**Acceptance**:
- Engine initializes within 3s on Pixel 7+.
- Single image extraction returns valid JSON.
- 350-image eval via the Android eval suite (already in project, see `android/AGENTS.md` "Run eval suite") reaches **≥ 80%** (5 percentage points from the reference 86% allowed for Android-runtime variance).

### Phase C — Tier 2 (kotlinllamacpp Q8_0)

**Goal: ship V7 Q8_0 GGUF via kotlinllamacpp for high-RAM devices.**

- C1. Add Gradle dep: `implementation("io.github.ljcamargo:llamacpp-kotlin:0.4.0")`
- C2. Implement `LlamaCppPartographExtractor : PartographExtractor`:
  - Use `LlamaHelper.openFileDescriptor` + `detachFd()` for both GGUF and mmproj_f16.
  - Image input via FD too (`ContentResolver` round-trip — write Bitmap to cache, open as FD).
  - Manually format Gemma 4 chat: `<start_of_turn>user\n<__media__>\n{prompt}<end_of_turn>\n<start_of_turn>model\n`.
  - Note: Tier 2 inputs the prompt as plain text — kotlinllamacpp's `getFormattedChat` JNI is a stub.
- C3. Model delivery: download `v7_q8_0.gguf` (4.93 GB) + `v7_mmproj_f16.gguf` (986 MB) on first-run.
- C4. Wire selector: add "Local (quality)" option, **only** offered when device has ≥8 GB RAM.

**Acceptance**:
- 350-image eval reaches **≥ 90%** (5 pp wiggle from VPS 95.71% allowed for runtime divergence).
- Latency ≤ 20 s/image on flagship (SD 8 Gen 3 / equivalent).
- App APK base size unchanged (models downloaded post-install).

### Phase D — Auto-selector & polish

- D1. `ExtractorSelector` with auto-selection logic + manual override in Settings.
- D2. RAM detection (`ActivityManager.MemoryInfo.totalMem`).
- D3. Network detection.
- D4. Model download UX (progress, pause/resume, retry, integrity check).
- D5. Storage warnings (need ~6 GB free for Tier 2).
- D6. First-run onboarding screen explaining the three options.

---

## 4. Open Questions / Risks

| Risk | Mitigation |
|---|---|
| Tier 1 quality (86%) is a real regression vs Tier 3 (95.71%) | Auto-selector prefers Tier 3 when network available; Tier 1 is offline-only fallback. UX warns user when running on lite tier. |
| Tier 2 GGUF + mmproj on kotlinllamacpp not yet empirically verified at 95.71% | Phase C acceptance gate runs full 350-image corpus on physical device before shipping. |
| `getFormattedChat()` stub in kotlinllamacpp could break if upstream changes | We control prompt format; manual templating insulates us. Add unit test that validates the formatted string byte-for-byte against expected. |
| LiteRT-LM Conversation API still broken (#2078) | We use raw Session API only. No dependency on the broken path. |
| Model storage 6 GB is significant on 32 GB phones | Tier 2 only offered when device free space ≥ 8 GB; otherwise auto-fallback to Tier 1. |
| `minSdk=31` excludes Android 7-11 users | Documented; non-blocking for hackathon scope. Future work: `minSdk=26` build flavour without LiteRT. |
| `ufoq/partoguard-litert-v7-wi4-v3` is now orphaned | Keep on HF for record; document as "unsupported on Android until litert-torch #998/#1005 resolved". Server-side Linux already uses it via the raw Session fallback in `gemma_adapter.py`. |
| Vision encoder export gap (Google: "not supported yet") | Tier 1 uses Google's base vision encoder; Tier 2 uses our V7 vision via mmproj_f16 (unaffected by litert-torch). |

---

## 5. What this plan does NOT change

- **VPS deployment** stays exactly as `partoguard_remote_gemma_vps.md` describes. The `RemotePartographExtractor` HTTP protocol is unchanged.
- **`ufoq/partoguard-litert-v7-wi4-v3`** stays on HF as a research artifact. Marked unsupported on Android in this document. Linux/Python eval can still use it via the existing `_litert_generate_with_image()` raw Session fallback in `gemma_adapter.py`.
- **Console CLI flags** unchanged. `--gemma-litert-e2b` still drives the **Linux** LiteRT path; Android does its own thing.
- **Eval corpus, RuleEngine, ReviewScreen, manual-edit feature** all unchanged.

---

## 6. Sources / Provenance

- LiteRT-LM Maven: <https://dl.google.com/android/maven2/com/google/ai/edge/litertlm/litertlm-android/maven-metadata.xml>
- LiteRT-LM Kotlin getting started: <https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/kotlin/getting_started.md>
- LiteRT-LM issue #2056 (CPU vision crash): <https://github.com/google-ai-edge/LiteRT-LM/issues/2056>
- LiteRT-LM issue #2078 (Conversation API Jinja break): <https://github.com/google-ai-edge/LiteRT-LM/issues/2078>
- litert-torch issue #998 (KV cache mismatch): <https://github.com/google-ai-edge/litert-torch/issues/998>
- litert-torch issue #1005 (case 'gemma4' missing): <https://github.com/google-ai-edge/litert-torch/issues/1005>
- Google AI Edge Gallery (reference app): <https://github.com/google-ai-edge/gallery>
- HF LiteRT base bundle: <https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm>
- kotlinllamacpp repo: <https://github.com/ljcamargo/kotlinllamacpp>
- kotlinllamacpp Maven: <https://repo1.maven.org/maven2/io/github/ljcamargo/llamacpp-kotlin/>
- VPS evidence: `eval_logs_v7_remote_q8_0_350.log`, `eval_logs_v7_remote_bf16_350.log`

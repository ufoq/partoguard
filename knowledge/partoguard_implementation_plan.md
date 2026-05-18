# PartoGuard Implementation Plan

Date created: 2026-05-10.

> **Current state (2026-05-17):** the console runs against the V7 fine-tuned
> Gemma served via llama.cpp `llama-server` on the VPS (`--gemma-remote`);
> see `partoguard_remote_gemma_vps.md`. LiteRT-LM is the on-device Android
> target, not the desktop runner. The "Phase 1 / Phase 2" framing below
> still describes the bounded-validator vs raw-Gemma trade-off and remains
> accurate; only the deployment surface has moved from local LiteRT to a
> remote llama.cpp endpoint for the demo.

## User decisions captured

- **Product target:** mobile app for nurses/midwives.
- **Implementation strategy:** console app first for easier debugging, then mobile UI once the core pipeline is verified.
- **Mobile framework preference:** Flutter or React considered, but first native Android/Kotlin build is recommended because LiteRT-LM has an Android/Kotlin SDK and no mature Flutter/React Native bridge was confirmed.
- **Inference target:** local Gemma 4 E4B/E2B inference, not cloud-first.
- **Language:** English only for MVP.
- **Asset policy:** all local/reference assets may be used during development, but raw medical imagery remains local review-only per `partoguard_corpus_plan.md`.
- **Real-world input:** camera-captured partograph photos.
- **MVP style:** step-by-step, simple app that helps nurses quickly review labour-progress chart state.

## Planning baseline

This plan must follow:

- `partoguard_architecture.md` — Gemma-centered pipeline with deterministic vision guardrails.
- `partoguard_corpus_plan.md` — quarantine-first data handling and no raw medical imagery in repo.
- `partoguard_build_plan.md` — hackathon demo success criteria.
- `partoguard_sources.md` — source claims, unsupported claims to avoid, and local inference caveats.
- `AGENTS.md` — clinician-in-the-loop safety framing.

## Non-negotiable product constraints

- PartoGuard is **clinical decision support**, not diagnosis or autonomous triage.
- Gemma does not decide Alert/Action status; deterministic rules do.
- Gemma is used only for bounded extraction/verification or English explanation from already-computed facts.
- The first MVP focuses on **cervical dilation `X` marks** on the central graph.
- Fetal heart rate, descent, contractions, oxytocin, and maternal vitals remain future modules unless explicitly added later.
- Low confidence, poor image quality, unknown template, or near-line ambiguity must return `manual_review`.
- Outputs say “review/escalate per protocol,” not “perform treatment.”

## Current corpus baseline

The repository now includes a larger development corpus under `data/`:

- 350 synthetic images with manifest metadata: 50 blank, 60 partial, 100 filled, 80 degraded, and 60 obstructed.
- 235 harvested open-access/reference items with provenance and PHI-risk metadata.
- Manifest/stat consistency has been checked: all synthetic and harvested paths exist, and `stats.json` matches `manifest.json` category counts.

The console evaluator supports this corpus via:

```bash
partoguard eval --corpus-dir data
```

Current baseline from the conservative console pipeline:

- Total images: 350
- Non-manual outputs: 18
- Manual-review outputs: 332
- Manual review rate: 94.86%
- Blank-template manual review rate: 100.00%

Interpretation: blank safety behavior is correct, but the app is too conservative on the richer filled/degraded/obstructed synthetic set. The next engineering milestone should reduce unnecessary manual review using Gemma-centered bounded extraction and better crop/mark handling, while preserving safe failure on blanks, unknown templates, and low-confidence cases.

## Recommended technical stack

### Phase 1 — Console app

Recommended language: **Python**.

Why:

- Fastest for CV, synthetic data, command-line debugging, and tests.
- OpenCV/Pillow/NumPy make image normalization and mark extraction practical.
- Console logs make Gemma prompts, JSON outputs, and deterministic rule decisions auditable.

Core libraries:

- `opencv-python` for preprocessing, registration, and candidate mark detection.
- `Pillow` for image I/O.
- `numpy` for geometry and rule math.
- `pydantic` or dataclasses for strict schemas.
- `pytest` for deterministic unit tests.

Gemma local inference path:

- Primary mobile target: **LiteRT-LM** with `litert-community/gemma-4-E4B-it-litert-lm` when device memory allows.
- Current console runner: raw **Gemma 4 E2B** via LiteRT-LM, installed under `/root/partoguard-gemma` and invoked by `partoguard analyze --gemma-litert-e2b`.
- Fallback: `llama.cpp` server with Gemma chat template for local experiments.
- Avoid: Ollama for function-calling until Gemma 4 tool-call parsing is verified fixed.

Raw E2B console smoke test completed on 2026-05-11:

```bash
PYTHONPATH=/root/partoguard-gemma/local/lib/python3.11/dist-packages \
HF_HOME=/root/partoguard-gemma/hf \
/root/partoguard-gemma/local/bin/litert-lm run \
  --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --prompt="Reply with exactly: GEMMA_E2B_OK"
```

Result: `GEMMA_E2B_OK`.

The console integration runs E2B in two modes:

1. **Text-bounded** — Gemma sees only deterministic candidate-point JSON (used when no chart crop is provided).
2. **Image-native (CPU)** — `partoguard analyze --gemma-litert-e2b` automatically attaches the registered chart crop via LiteRT-LM `--attachment ... --vision-backend cpu`. Gemma 4 E2B visually inspects the partograph crop and confirms which candidate X marks are real. Verified on Linux x86_64 CPU on 2026-05-11 against a clean synthetic partograph: extraction produced 7 candidate points, Gemma vision-confirmed them as `cv_gemma_verified`, deterministic rule engine then classified status `normal`.

Native vision runtime dependencies on Debian bookworm: `libvulkan1`, `libegl1`, `libgles2`, `libegl-mesa0`, `libgbm1`, `libdrm2`, and supporting xcb libs. Install via `apt --fix-broken install` after the LiteRT-LM Python package.

### Phase 2 — Mobile app

Recommended first mobile build: **native Android/Kotlin**.

Why:

- LiteRT-LM has an Android/Kotlin SDK.
- Flutter/React Native would require custom native bridges to Kotlin anyway.
- A native first build reduces inference/runtime risk for the hackathon.

Later wrapper option:

- Flutter can be added after the Kotlin inference module stabilizes via platform channels.
- React Native can be added later via a native module, but it is not the fastest route for LiteRT-LM.

Mobile runtime:

- LiteRT-LM Android SDK.
- Model format: `.litertlm` only.
- Model delivery during development: ADB push to device storage.
- Model delivery later: on-demand asset delivery; do not bundle multi-GB model files in APK.
- Primary model: E4B if device memory allows.
- Fallback model: E2B for lower-memory devices.

Before committing to the full mobile feature set, run a target-device feasibility spike on the actual Android phone intended for the demo. Measure LiteRT-LM model load, first-token latency, memory pressure, thermal behavior, and whether E4B is stable enough. If E4B is unstable, lock the MVP to E2B or keep Gemma off the default mobile hot path and rely on CV + deterministic rules for the demo.

## System architecture

```text
Input image
  → deterministic image-quality and alignment guardrails
  → template/chart registration
  → bounded cervical-chart and candidate crops
  → Gemma structured extraction/verification
  → deterministic_rule_engine
  → Gemma explanation/audio/workflow from computed facts
  → audit overlay data
  → console report / mobile UI
```

## Data flow contracts

### Input contract

```json
{
  "image_path": "string",
  "template_id": "modified_who_partograph_v1|unknown",
  "mode": "console|mobile",
  "source": "synthetic|local_reference|camera"
}
```

### Extraction contract

```json
{
  "template_id": "modified_who_partograph_v1",
  "chart_present": true,
  "registered": true,
  "points": [
    {
      "x_hours": 0.0,
      "dilation_cm": 4.0,
      "bbox": [0, 0, 0, 0],
      "confidence": 0.95,
      "source": "cv|gemma_verified|manual"
    }
  ],
  "overall_confidence": 0.91,
  "warnings": []
}
```

### Rule output contract

```json
{
  "status": "normal|alert_zone|action_zone|indeterminate|manual_review",
  "framework": "modified_who_partograph",
  "triggering_point": {
    "x_hours": 8.0,
    "dilation_cm": 8.0
  },
  "explanation": "The plotted dilation point appears in the Action zone. Please review now and follow facility protocol.",
  "confidence": 0.88,
  "requires_human_review": true
}
```

## Step-by-step execution plan

### Step 0 — Repository scaffold

Goal: create a clean console-first project structure.

Proposed layout:

```text
partoguard/
  cli/
  core/
    imaging/
    extraction/
    rules/
    schemas/
    reports/
  data/
    synthetic/
    manifests/
  tests/
  docs/
```

Acceptance criteria:

- Console command exists.
- Tests run.
- No raw downloaded medical images are committed.
- All sample/demo assets have storage-tier labels.

Parallel spike:

- Run a one-device LiteRT-LM feasibility test as early as possible.
- Record model, device, backend, load success/failure, latency, and memory notes.
- Use the result to decide E4B vs E2B for the mobile phase.

### Step 1 — Synthetic scenario generator

Goal: generate controlled, labeled partograph examples for normal/Alert/Action cases.

Use initial scenario seeds:

- Normal: 8:00 a.m. = 4 cm; 12:00 p.m. = 10 cm.
- Alert-zone: 8:00 a.m. = 4 cm; 12:00 p.m. = 6 cm.
- Action-zone: 8:00 a.m. = 4 cm; 12:00 p.m. = 6 cm; 4:00 p.m. = 8 cm.

Acceptance criteria:

- Produces both canonical chart crops and full-page camera-like renders.
- Supports clean and degraded variants.
- Full-page renders include skew, perspective warp, blur, shadows, crop, and optional form noise so registration can be tested.
- Labels include point coordinates and expected zone status.
- Synthetic outputs are clearly marked as synthetic/demo only.

### Step 2 — Deterministic rule engine first

Goal: build and test the rules before any model integration.

Rules:

- Modified WHO partograph mode first.
- Compare extracted points against configured Alert/Action line geometry.
- Include uncertainty margin near lines.
- Return `manual_review` when confidence is low or geometry is ambiguous.

Acceptance criteria:

- Unit tests cover normal, Alert, Action, near-line, missing points, out-of-bounds points, and unknown template.
- Rule engine does not depend on Gemma.

### Step 3 — deterministic preprocessing and registration guardrails

Goal: convert a source image into a canonical chart coordinate system.

Implementation:

- Image quality gate: blur, exposure, dimensions.
- Page/ROI detection for supported templates.
- Perspective correction.
- Template registration.
- Crop cervical dilation chart.

Acceptance criteria:

- Clean synthetic images register reliably.
- Degraded images either register or produce `manual_review`.
- Outputs debug artifacts locally for development, but raw/reference images stay ignored unless approved.

### Step 4 — candidate `X` mark proposal

Goal: propose likely cervical dilation `X` mark regions so Gemma receives bounded, relevant crops rather than an unconstrained full-page image.

Implementation:

- Threshold/edge/stroke detection on the chart crop.
- Connected components or contour filtering.
- Exclude printed grid and line geometry where possible.
- Map candidate centers into `(hours, dilation_cm)`.

Acceptance criteria:

- Detects synthetic `X` marks across clean/degraded examples.
- Reports confidence and uncertain regions.
- Never invents missing points.

### Step 5 — Gemma-centered bounded extraction/verification

Goal: use local Gemma 4 E4B/E2B as the core mark-reading model over bounded chart/candidate crops, with deterministic geometry providing crop coordinates and safety checks.

Implementation path:

1. Start with LiteRT-LM CLI/Python local inference.
2. Use JSON-only prompts against cropped cervical chart images plus CV candidate coordinates.
3. Feed Gemma only chart crops and candidate coordinates, not the full page.
4. Ask Gemma to extract/confirm/reject/flag cervical-dilation marks from bounded crops; full-page clinical interpretation remains forbidden.
5. Parse strict JSON into typed schema.
6. Reject malformed/low-confidence model outputs.

Prompt principle:

> Detect only handwritten cervical dilation `X` marks inside this registered chart crop. Ignore printed grid lines, Alert/Action lines, labels, and handwriting outside the chart. Return JSON only. Do not infer missing points.

Acceptance criteria:

- Model output validates against schema.
- Low-confidence or malformed output becomes `manual_review`.
- Gemma verification is optional per-image and only runs when CV confidence is insufficient or ambiguity is high.
- Rule engine remains the source of zone classification.

### Step 6 — Console report UX

Goal: produce a nurse-readable English report and machine-readable audit file.

Console output includes:

- Input image name.
- Template match status.
- Extracted dilation points.
- Zone status.
- Confidence.
- Explanation.
- Safety caveat.
- Debug artifact paths.

Example safe message:

> “The plotted dilation point appears to be in the Action zone. Please review now and follow your facility escalation protocol.”

Acceptance criteria:

- Report generated for normal, Alert, Action, and safe-failure cases.
- JSON audit file written for tests/demo.
- English only.

### Step 7 — Evaluation harness

Goal: make every pipeline change measurable.

Metrics:

- Registration success rate on synthetic set.
- `X` mark detection precision/recall on synthetic labels.
- Zone classification accuracy on synthetic labels.
- Manual-review rate under degradation.
- Invalid Gemma JSON rate.
- Corpus status distribution by category using `data/manifest.json`.
- Blank-template manual-review rate, expected to remain 100% unless explicit manual annotations are supplied.

Acceptance criteria:

- `pytest` passes.
- Evaluation command emits summary metrics.
- Known limitations are printed and logged.
- `partoguard eval --corpus-dir data` runs against the 350-image corpus and reports category-level manual-review/non-manual counts.

### Step 8 — Android/Kotlin mobile prototype

Goal: convert the verified console pipeline into a nurse-facing Android prototype.

Implementation stages:

1. Native Kotlin app shell.
2. Camera capture/import.
3. Ghost overlay or simple alignment guide.
4. Local preprocessing/registration.
5. LiteRT-LM Gemma E4B/E2B inference module, gated by the target-device spike.
6. Deterministic rules module ported or shared.
7. Results screen: status, extracted points, overlay, safe English message.
8. Manual-review state for bad inputs.

Acceptance criteria:

- Runs on target Android device.
- Can analyze a camera/photo input.
- Shows extracted points and rule explanation.
- Does not require internet for inference.
- Keeps patient photos local.
- Clearly returns `manual_review` for poor-quality or unsupported camera photos.

### Step 9 — Optional Flutter/React wrapper

Goal: build a cross-platform UI only after native inference proves stable.

Recommendation:

- Prefer Flutter over React Native if cross-platform UI is still desired.
- Bridge to the Kotlin LiteRT-LM module via platform channels.
- Keep the native Kotlin module as the inference owner.

Acceptance criteria:

- No regression in local inference.
- Camera image bytes pass correctly to native module.
- UI preserves safety wording and manual-review fallbacks.

## First MVP milestone

The first shippable milestone should be a console app that can run:

```bash
partoguard analyze --image path/to/partograph.jpg --template modified_who_partograph_v1 --json-out report.json
```

And return:

- `normal`, `alert_zone`, `action_zone`, `indeterminate`, or `manual_review`.
- Extracted points and confidence.
- A short English explanation.
- Debug overlay/registered crop path.

## Initial test cases

1. Clean synthetic normal chart.
2. Clean synthetic Alert-zone chart.
3. Clean synthetic Action-zone chart.
4. Blurred chart → `manual_review`.
5. Cropped/missing chart → `manual_review`.
6. Unknown template → `manual_review`.
7. Point near Alert/Action line → `indeterminate` or `manual_review`.
8. Gemma returns invalid JSON → `manual_review`.

Status definitions:

- `normal` — confident extraction; plotted point(s) remain left of configured Alert threshold.
- `alert_zone` — confident extraction; plotted point(s) enter Alert zone but not Action zone.
- `action_zone` — confident extraction; plotted point(s) enter Action zone.
- `indeterminate` — extraction is confident enough to locate point(s), but point(s) are within the configured uncertainty band near a threshold.
- `manual_review` — image quality, template match, extraction confidence, model output, or required fields are insufficient for automated status.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Local E4B too heavy for target phones | Keep E2B fallback; cap context; use LiteRT-LM quantized model. |
| Flutter/React bridge slows timeline | Build first mobile app in native Kotlin. |
| Model hallucinates points | Use CV candidates, JSON schema validation, and manual-review fallback. |
| Synthetic data overfits clean layouts | Add degradation and local-reference testing; do not claim clinical accuracy. |
| Raw assets accidentally committed | Keep repo-wide `.gitignore` raw-asset exclusions; follow corpus plan. |
| App output sounds like treatment order | Use scripted “review/escalate per protocol” language. |

## Definition of done for console phase

- Synthetic data generator works.
- Rule engine is unit-tested.
- CV pipeline extracts `X` marks from synthetic chart crops.
- Gemma local inference can validate/extract structured JSON from a chart crop or is safely stubbed behind the same interface until runtime setup is complete.
- Console report and JSON audit output work.
- Evaluation harness reports metrics.
- Safe-failure cases are demonstrated.

## Definition of done for mobile phase

- Android app captures/imports a partograph photo.
- Image quality gate and registration run locally.
- Local Gemma E4B/E2B feasibility has been tested on the target device; production path uses LiteRT-LM or falls back safely to CV-only + manual-review gates.
- Deterministic rule result appears with overlay and English safety message.
- No internet is required for analysis.
- Patient photos remain local.
- Manual-review fallback is prominent.

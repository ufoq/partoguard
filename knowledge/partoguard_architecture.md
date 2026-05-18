# PartoGuard Architecture

> **Current serving path (2026-05-17):** the console hits the V7 fine-tuned
> Gemma 4 E2B model served as Q8_0 GGUF via llama.cpp `llama-server` on the
> VPS at `vps-box:8080` (95.71% on the 350-image eval, ~8 s/image). bf16 is
> available as a swap-in on the same host. LiteRT-LM remains the on-device
> Android target, not the desktop/server path. See
> `partoguard_remote_gemma_vps.md` for the canonical deployment doc.

## Recommended architecture

Use a **Gemma-centered, deterministic-vision-guarded** pipeline.

Here, “CV” means deterministic computer-vision geometry: image quality checks,
page alignment, chart-grid registration, and candidate crop proposal. It does not
mean PartoGuard should be built as an OpenCV product with Gemma bolted on. The
app experience and intelligence should be built around local Gemma models, while
deterministic vision code acts as a safety scaffold so Gemma receives clean,
bounded chart regions instead of an unconstrained full-page clinical decision.

Gemma 4 should not be asked to interpret an entire photographed partograph and decide risk directly. The safer pipeline is:

```text
Camera capture
  → deterministic image quality + page alignment guardrails
  → template/chart registration and region-of-interest crops
  → Gemma structured extraction from bounded chart/candidate crops
  → deterministic clinical rule engine over extracted coordinates
  → Gemma-generated explanation/audio/referral draft from computed facts
  → audit overlay + human confirmation
```

## Responsibility split (current console implementation)

> **Phase 1 (current).** Deliberately running Gemma E4B without CV preprocessing and without the bounded validator, so raw model behaviour can be measured before phase-2 guardrails are designed. CV preprocessing (`core/imaging/preprocess.py`) and the bounded validator (`_bounded_extracted_points_from_payload`) are preserved in the codebase and exercised by tests, but are bypassed at runtime whenever a Gemma extractor is active (`core/pipeline.py`). To restore the phase-0 CV+validator pipeline, check out git tag `pre-raw-model-phase`.
>
> **Phase-1 measured baseline (Gemma 4 E2B with blank-first prompt, 20-image sample, seed 42):** manual_review 55%, **5.48s/image** on CPU. Blank-chart hallucination eliminated: 5/5 blank templates correctly routed to manual_review by the rule layer (model returns `{"p":[]}`, rule engine declines on `<2` points). Filled charts: 4/5 receive a non-manual zone classification. Headline manual-review rate is higher than the E4B run because the model now correctly refuses to invent marks on blanks instead of hallucinating action_zone calls. Gemma 4 E2B is the mobile-friendly variant for on-device interactive nurse use; E4B remains opt-in via `--gemma-litert-e4b`.

> **Phase-1 best result (Gemma 4 E4B + p5 hybrid cervicograph-focus prompt + LiteRT-LM Python daemon, 25-image stratified probe seed 42, 5 per category):** **100% per-category correctness** (blank 5/5 manual_review; partial 5/5; filled 5/5; degraded 5/5; obstructed 5/5), **8.5–8.7s/image** on CPU (under the 10s interactive-use budget). End-to-end production CLI eval (`--gemma-litert-e4b`, 20-image sample seed 42) measured 9.24s/image with the same 0% blank-hallucination rate. Key changes vs. earlier baseline: (1) LiteRT-LM Python `Engine` daemon eliminates per-call subprocess+model-load overhead and is now the default whenever a `--gemma-litert-*` flag is set; (2) the extraction prompt explicitly disambiguates the cervicograph from FHR/contraction/drug subplots and enforces a count-first discipline (`Do NOT extrapolate`, `Do NOT interpolate`, `list only marks you can visually verify`); (3) the blank-first gate from the earlier prompt is preserved so empty templates remain at 100% manual_review. Implementation lives in `core/extraction/gemma_adapter.py:LiteRTGemmaDaemonExtractor` with a module-level engine cache keyed by `(huggingface_repo, model_reference)`.

The console app implements the recommended pipeline with a strict split. CV is intentionally constrained to image preparation; Gemma is the chart interpreter; the rule engine is the only clinical decision-maker.

| Stage | Owner | Responsibility |
|---|---|---|
| Image quality, page alignment, perspective correction, template registration, canonical chart crop | Deterministic CV (`core/imaging/preprocess.py`) | **[Phase 1: bypassed when a Gemma extractor is active.]** Reject unreadable/non-partograph images; produce a normalized chart crop in canonical coordinates. Never interprets handwriting. |
| Read X marks from the chart image and emit structured `{x_hours, dilation_cm, confidence}` points | Gemma 4 E4B running locally on CPU via LiteRT-LM, image-attached (`LiteRTGemmaE2BVerifier.extract_from_image`; defaults to E4B, E2B opt-in via `--gemma-litert-e2b`) | Visual reading of the chart only. **[Phase 1: permissive parsing only — clamp to chart ranges, round to 0.5, no plausibility/duplicate/max-count rejection.]** Phase 2 will reintroduce bounded validation. Never makes clinical decisions. |
| Classify clinical status (`normal` / `alert_zone` / `action_zone` / `manual_review`) | Deterministic rule engine (`core/rules/engine.py`) | Sole clinical decision-maker. Operates on the extracted point list. Auditable and explainable. |
| Human-facing explanation / referral text | Gemma (optional, from already-computed facts) or deterministic template | Narrates the rule output. Cannot change the classification. |

Fallback paths:

- When no Gemma extractor is configured (stub mode or `LocalGemmaVerifier`), `core/extraction/marks.py::extract_x_marks` runs the deterministic CV mark detector and Gemma acts as a bounded verifier of CV candidates (the older verify path). This path is unchanged.
- In the phase-1 Gemma-extractor path, an unreadable image or a `<2`-points extraction routes to `MANUAL_REVIEW` via the rule layer, not via guardrails on the extractor itself.
- Any extractor failure (subprocess error, timeout, missing crop, schema/range/duplicate/plausibility violation, empty result) routes to `manual_review`.

## Why deterministic vision guardrails

Partographs are structured documents. The grid, labels, axes, and line positions are mostly known once the template is identified. Deterministic geometry should handle alignment, crop boundaries, and coordinate mapping because those steps are auditable and reduce hallucination risk. Gemma should then operate on the clinically relevant bounded regions as the core extraction and user-experience model.

Gemma is best used for:

- OCR / document parsing support.
- Handwritten mark extraction and verification from bounded chart/candidate crops.
- Structured JSON extraction with confidence and uncertainty fields.
- Local-language explanation generation from already-computed facts.
- Conversational guidance for rescanning, manual review, and clinician workflow.

Gemma should not be the source of truth for:

- Whether the Alert/Action line has been crossed.
- Whether the patient needs intervention.
- Whether referral is clinically safe.

## Capture UX

### Ghost overlay scanner

The camera view should display a semi-transparent template outline. The app should auto-capture only when:

- All page corners are visible.
- Rotation/skew is within tolerance.
- The chart grid/template aligns to the overlay.
- Blur is low.
- Exposure is acceptable.
- Glare does not cover key graph regions.

This prevents many downstream failures before inference begins.

### Multi-frame option

For low-end cameras and poor lighting, capture a short 2–3 second scan/video instead of one photo. Fuse the best frames or select the sharpest aligned frame.

Prototype assumption: multi-frame fusion can improve robustness, but it should be demonstrated rather than claimed as validated.

## Image preprocessing

Candidate preprocessing steps:

- White balance and contrast normalization.
- Shadow correction.
- CLAHE / adaptive histogram equalization.
- Light denoising.
- Perspective warp.
- Template registration against a whitelist of supported forms.
- Optional color-channel handling: suppress stain-dominant red regions while preserving blue/black ink.
- Template subtraction where a matching blank template is available.

If registration fails, the app should not continue to clinical logic.

## Extraction strategy

The first supported target should be cervical dilation `X` marks on the central graph.

Suggested staged extraction:

1. Register chart crop to canonical coordinates.
2. Use CV to propose candidate handwritten components.
3. Filter candidates by region, size, stroke pattern, and proximity to expected grid cells.
4. Ask Gemma to extract/verify marks from bounded crops and return structured JSON only.
5. Return structured points with confidence and uncertainty regions.

Suggested model output schema:

```json
{
  "template_id": "modified_who_partograph_v1",
  "chart_present": true,
  "points": [
    {
      "bbox": [0.0, 0.0, 0.0, 0.0],
      "center_norm": [0.0, 0.0],
      "symbol_type": "x",
      "confidence": 0.92
    }
  ],
  "uncertain_regions": [
    {
      "bbox": [0.0, 0.0, 0.0, 0.0],
      "reason": "shadow_or_ambiguous_mark"
    }
  ],
  "overall_confidence": 0.88
}
```

Prompt pattern:

> You are reading a perspective-corrected cervical-dilation chart crop from a WHO-style partograph. Return valid JSON only. Detect only handwritten cervical-dilation marks inside the chart. Ignore printed grid lines, labels, Alert/Action lines, and handwriting outside the chart. Do not infer missing points. If uncertain, mark uncertain.

Verifier pattern:

- Input: same crop plus first-pass candidate points.
- Task: confirm/reject each proposed point, flag likely misses, and avoid full re-interpretation.

## Deterministic rule engine

Clinical logic should run outside the model.

For the modified WHO partograph, once points are mapped into chart coordinates:

- Compare each point to the configured Alert and Action line positions.
- Use an uncertainty margin for marks close to a line.
- Return `normal`, `alert_zone`, `action_zone`, or `indeterminate`.
- Explain the result with the exact plotted point and rule.

If multiple framework modes are supported, make the selected framework explicit:

- `modified_who_partograph`
- `who_labour_care_guide`
- `site_specific_protocol`

## Audio and local-language UX

Audio should be short, scripted, and clinician-reviewed. It should communicate what to review, not issue autonomous treatment decisions.

Example safe Pidgin-style output:

> “Sister, I see labour progress mark wey need review now. The app no sure enough to decide alone. Please check the partograph and call senior midwife or doctor.”

For high-confidence Action-zone prototype demo:

> “Sister, the plotted dilation appears to be in the Action zone on this partograph. Please review now and follow your facility referral protocol.”

## Safety fallbacks

Always prefer safe refusal over silent wrong output.

Return manual-review state if:

- Photo quality is poor.
- Template is unknown.
- Chart crop is missing or occluded.
- Extraction confidence is low.
- Marks are near the Alert/Action line within the uncertainty band.
- Required fields are missing.
- The framework/version of the form cannot be identified.

## Prototype vs production

Prototype can demonstrate:

- Offline capture and image normalization on selected devices.
- Extraction from supported templates.
- Explainable rule-trigger overlays.
- Scripted spoken review prompts.

Production would require:

- Real filled-partograph image dataset.
- Clinician-labeled validation set.
- Agreement studies against midwives/obstetricians.
- Human-factors testing in target facilities.
- Localization review for Pidgin/Hausa/Yoruba.
- Data privacy and medical-device regulatory assessment.

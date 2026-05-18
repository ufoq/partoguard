# PartoGuard Build Plan

This is a hackathon-oriented plan. It prioritizes a credible, explainable prototype over medical-grade claims.

> **Current state (2026-05-17):** Waves 1–5 are landed; the console
> demonstrably ingests an image, registers the chart, extracts marks via
> the V7 fine-tuned Gemma 4 E2B served on the VPS (`--gemma-remote`,
> 95.71% on the full corpus), and applies the deterministic rule engine.
> See `partoguard_remote_gemma_vps.md` for the live deployment and
> `partoguard_training_log.md` for the model timeline. Wave 6 (UX / demo
> story) is the active scope for the May-18 submission.

## Success criteria for the demo

1. Scan or upload a supported partograph image.
2. Show the registered chart overlay.
3. Extract cervical-dilation `X` marks with confidence.
4. Compute whether points fall in normal / alert / action / indeterminate zone using deterministic logic.
5. Speak a short clinician-reviewed alert script.
6. Show at least one safe-failure case where the app refuses to assess because image quality or template match is poor.

## Wave 1 — Knowledge and assets

Deliverables:

- Collect blank WHO-style templates.
- Collect public filled examples from NCBI/PMC/USAID/Ethiopia/OpenLearn.
- Inventory local reference assets under `/data/input/partographs` without copying raw images into the public repo.
- Apply the corpus safety workflow in `partoguard_corpus_plan.md` before any asset is admitted to a training/evaluation set.
- Define supported template IDs.
- Define rules ledger: modified WHO partograph vs WHO Labour Care Guide vs site-specific protocol.
- Draft scripted messages in English + Nigerian Pidgin first; Hausa/Yoruba later only after review.

Acceptance criteria:

- Every template has source link and license/usage note.
- Every clinical rule has source citation and version label.
- Every image/form asset has a storage tier (`open_repo`, `local_review_only`, or `rejected`).
- Unsupported claims are excluded from pitch copy.

## Wave 2 — Synthetic data generator

Deliverables:

- Generate normal, alert-zone, action-zone, and indeterminate plotted curves.
- Render synthetic `X` marks onto blank templates.
- Apply augmentations: rotation, blur, shadows, folds, stains, compression, low light, crop.
- Export labels: mark coordinates, zone classification, template ID.

Acceptance criteria:

- Dataset includes clean and adversarial examples.
- Labels are machine-readable.
- Docs explicitly say synthetic data is for bootstrapping/demo, not clinical validation.

## Wave 3 — deterministic vision guardrails

Deliverables:

- Image quality gate.
- Corner/page detection.
- Perspective correction.
- Template registration.
- Chart ROI cropping.
- Candidate `X` mark detection.

Acceptance criteria:

- Supported clean templates register reliably.
- Bad images produce manual-review state.
- The extracted overlay can be visually audited.

## Wave 4 — Gemma-centered bounded extraction

Deliverables:

- Prompt/schema for bounded chart/candidate-crop extraction.
- Gemma extraction and verification calls for cervical-dilation marks.
- Structured JSON parser.
- Confidence thresholds and uncertainty-region handling.

Acceptance criteria:

- Model returns structured JSON only.
- Missing/uncertain marks are not invented.
- Low confidence does not flow into high-risk alerts.

## Wave 5 — Deterministic clinical logic

Deliverables:

- Rule engine for modified WHO partograph geometry.
- Optional LCG mode stub or documented future-work note.
- Output states: `normal`, `alert_zone`, `action_zone`, `indeterminate`, `manual_review`.
- Audit explanation: extracted point, line/rule, confidence.

Acceptance criteria:

- Unit tests cover normal, alert, action, edge-near-line, missing-points, and bad-template cases.
- Gemma output never directly sets clinical risk; only structured extracted values feed rules.

## Wave 6 — UX and demo story

Deliverables:

- Ghost overlay scanner mock or working camera UI.
- Audio output for normal/alert/action/manual-review cases.
- Referral-alert draft function that creates a message but requires human confirmation.
- Demo video script.

Acceptance criteria:

- User can follow photo → extraction → rule → spoken prompt.
- The demo includes one positive case and one safe-failure case.
- All high-risk wording is “review/escalate per protocol,” not autonomous treatment.

## Recommended demo script

1. Show a busy clinic and paper partograph.
2. Midwife aligns paper to ghost overlay; app auto-captures.
3. App shows detected `X` marks and confidence.
4. App highlights the Action-zone rule geometry.
5. Audio says: “The plotted dilation appears to be in the Action zone. Please review now and follow your facility referral protocol.”
6. Midwife confirms and sends a referral-alert draft.
7. Show a blurred/stained chart; app refuses and requests rescan/manual review.

## Synthetic/demo scenario seeds

Use the downloaded tutorial’s three-case pattern as the first labeled scenario set:

| Scenario | Dilation points | Expected output |
|---|---|---|
| Normal progress | 8:00 a.m. = 4 cm; 12:00 p.m. = 10 cm | Normal/progressing; continue monitoring. |
| Alert-zone warning | 8:00 a.m. = 4 cm; 12:00 p.m. = 6 cm | Alert-zone review; reassess contractions, presentation/position, and local protocol. |
| Action-zone urgent review | 8:00 a.m. = 4 cm; 12:00 p.m. = 6 cm; 4:00 p.m. = 8 cm | Action-zone/manual urgent review; escalate per facility protocol. |

Keep the language clinician-in-the-loop. The tutorial says “act immediately” and mentions possible caesarean section, but PartoGuard should phrase this as “review now and follow facility protocol,” not as a treatment instruction.

## Risks to track

- Lack of real filled-partograph dataset.
- Multiple local form variants.
- Low-end Android memory/thermal constraints.
- Translation safety for Pidgin/Hausa/Yoruba.
- Potential privacy exposure if photos include patient identifiers; raw filled forms remain local review-only by default.
- Medical-device/regulatory implications after prototype stage.

## Open questions

- Which exact Nigerian facility partograph templates should be supported first?
- What phone class is realistic for the demo and target deployment?
- Can we obtain a small clinician-reviewed set of de-identified teaching partographs?
- Should the first MVP support only cervical dilation, or include fetal heart rate danger signs too?

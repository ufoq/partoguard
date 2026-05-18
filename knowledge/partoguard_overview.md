# PartoGuard Overview

## One-line concept

**PartoGuard is a prototype clinical decision-support tool that digitizes selected entries from photographed WHO-style partographs and highlights possible review/referral triggers using deterministic rules.**

It is intended for the Gemma 4 Good Hackathon as an assistant for overworked midwives in Nigerian and similar low-resource maternity settings. The product story stays offline-first — the on-device Android target uses LiteRT-LM. The hackathon **console / server demo** runs against the V7 fine-tuned Gemma served by llama.cpp on a VPS (`partoguard_remote_gemma_vps.md`) so the same fine-tuned weights and rule engine can be exercised end-to-end without an Android device. The remote path is a demo surface, not the production deployment target.

## Problem framing

Partographs are designed to make labour progress visible, but the tool only helps if it is correctly filled, read, and acted on. In many low-resource settings, barriers include staff shortages, incomplete documentation, limited training, paper forms, poor lighting, retrospective charting, and delayed escalation.

PartoGuard’s core story is not “AI replaces the midwife.” The safer and stronger story is:

> Silent partographs are dangerous: the chart may contain the warning, but no one has time or support to interpret it. PartoGuard turns the photographed chart into an explainable, spoken, clinician-in-the-loop review prompt.

## Target user

- Primary user: midwife, nurse, or community health worker monitoring labour.
- Target environment: rural or overloaded facility, intermittent/no internet, Android phone, paper partograph.
- Secondary users: referral hospital staff, supervising midwife, training program instructors.

## Core workflow

1. Midwife opens camera and aligns the paper to a ghost partograph overlay.
2. App auto-captures when corners/grid alignment and image quality are acceptable.
3. Image is processed locally: dewarped, contrast-enhanced, template-registered, and cropped to relevant chart regions.
4. Gemma 4 E2B/E4B assists only with bounded extraction from ambiguous cropped fields.
5. A deterministic rule engine compares extracted values against configured partograph / Labour Care Guide rules.
6. App speaks a short result in clinician-reviewed local-language scripts and shows an audit overlay.
7. If confidence is low, the app refuses automation and requests manual review or rescan.

## Reviewed plan corrections

Use this wording:

- “Prototype clinical decision-support tool.”
- “Digitizes selected entries.”
- “Highlights possible review/referral triggers.”
- “Gemma assists extraction; it does not make clinical decisions.”
- “Clinician judgment remains primary.”
- “If image quality, template matching, or extraction confidence is insufficient, PartoGuard defers to manual review.”

Avoid this wording unless clinically validated later:

- “Diagnoses labour complications.”
- “Autonomous triage.”
- “Safe referral decision.”
- “Clinically validated.”
- “Works on any Android phone.”
- “Saves X lives” as a measured outcome.

## Hackathon strengths

- **High-impact problem:** maternal and perinatal safety during labour monitoring.
- **Gemma-relevant:** multimodal document/image understanding, bounded structured extraction, function calling / structured JSON, possible audio UX.
- **Digital equity fit:** offline-first, local-language, low-literacy-friendly interaction.
- **Explainability:** shows photo, extracted marks, confidence, and the exact rule that triggered the warning.
- **Strong demo arc:** confused midwife → scan paper → visible overlay → spoken warning → referral alert draft.

## Safety position

PartoGuard should be pitched as a workflow and interpretation aid. It should never claim to replace clinical judgment, diagnose obstructed labour, or independently decide intervention. Every high-risk output should be framed as “review now / escalate per local protocol,” not as a final medical decision.

## Why Digital Analysis? (The "Look at the Paper" Fallacy)

A common challenge to PartoGuard's premise is: *If the nurse is already plotting Xs on a paper grid that has the alert/action lines printed on it, why do they need an app? Can't they just look at the lines?*

While true in theory, field realities make this insufficient:

1. **Plotting Errors**: The modified WHO partograph requires the first active-phase X to be snapped to the Alert line, shifting the time axis. Nurses frequently plot at absolute clock columns instead, breaking the printed geometry. PartoGuard evaluates the *actual* slope of labor, correcting for physical grid-placement errors.
2. **Retrospective Plotting**: In understaffed wards, charts are often updated hours after examinations. A point-of-care capture tool digitizes the state and triggers escalation immediately at the bedside.
3. **Boundary Ambiguity**: Hand-drawn Xs are large and imprecise. PartoGuard provides an objective, deterministic ruling on whether a threshold was crossed.
4. **Remote Supervision**: Digitizing the curve allows instant transmission to senior obstetricians at district hospitals for review.

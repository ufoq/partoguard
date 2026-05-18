# Data Licenses

This document describes the licenses for all datasets and data assets used in PartoGuard.

## Synthetic Training and Evaluation Data

| Asset | Generator | License | Notes |
|-------|-----------|---------|-------|
| 350-image synthetic eval corpus (`data/blank/`, `data/partial/`, `data/filled/`, `data/degraded/`, `data/obstructed/`) | `data/generate_corpus.py` (seed 12345) | Apache-2.0 | Programmatically generated; no real patient data |
| 400-image synthetic training set (`data/training/train_0000–0399.png`) | `scripts/generate_training_v2.py` (seed 77777) | Apache-2.0 | Not included in this repo; regenerate with the script |
| 100-image WHO-template training set (`data/training/train_0400–0499.png`) | `scripts/generate_who_training.py` (seed 88888) | Apache-2.0 | Synthetic X marks drawn on open-access WHO backgrounds |

## WHO and Open-Access Reference Materials (`data/harvested/`)

All harvested materials were reviewed for reuse rights and patient data exposure
before inclusion. All have `phi_risk = 0` (no identifiable patient data).

| Source | Type | License | Reference |
|--------|------|---------|-----------|
| WHO Labour Care Guide 2020 | PDF pages, figures | CC BY-NC-SA 3.0 IGO | WHO/RHR/20.01 |
| WHO Safe Motherhood 1996 | PDF pages | WHO publication (free for non-commercial use) | WHO/FHE/MSM/96.24 |
| WHO Multicentre Study 1994 | PDF pages | WHO publication (free for non-commercial use) | WHO/FHE/MSM/94.4 |
| Bedwell et al. 2017, BMC Pregnancy Childbirth | Figures | CC BY 4.0 | DOI: 10.1186/s12884-017-1468-2 |
| Oladapo et al. 2018, BJOG | Figures | CC BY 4.0 | DOI: 10.1111/1471-0528.15010 |
| Wikimedia Commons partograph images | Images | CC BY-SA or public domain | See harvested/manifest.json |
| Additional CC-BY journal figures | Figures | CC BY 4.0 | See harvested/manifest.json for individual citations |

Full source citations for all harvested items are in `data/harvested/manifest.json`.

## Training Data Provenance

No real patient records, identifiable clinical data, or protected health information (PHI)
was used in training or evaluation. All evaluation and training images are either:

1. Programmatically generated from blank WHO templates with synthetic mark placements, or
2. Derived from openly licensed educational and reference materials with no patient data.

The evaluation corpus (350 images) is entirely synthetic and does not represent
any real clinical case.

## Disclaimer

Synthetic data accuracy cannot be extrapolated to clinical performance.
Real-world validation on clinician-labelled de-identified patient data is required
before any clinical deployment.

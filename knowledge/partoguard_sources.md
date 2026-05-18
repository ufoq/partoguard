# PartoGuard Sources and Claim Verification

Date verified: 2026-05-10.

## Gemma 4 capabilities

### Supported

- Gemma 4 E2B and E4B support text, image, and audio modalities.
- Gemma 4 supports OCR/document parsing and image understanding tasks relevant to charts/forms.
- Gemma 4 supports function calling / structured tool-use patterns.
- Gemma 4 edge models are intended for offline/on-device use, but actual phone compatibility depends on memory, thermal behavior, runtime, and quantization.

### Sources

- Google AI for Developers — Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Google AI for Developers — Vision / image understanding: https://ai.google.dev/gemma/docs/capabilities/vision/image
- Google AI for Developers — Function calling with Gemma 4: https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4
- Google DeepMind Gemma model page: https://deepmind.google/models/gemma/gemma-4/
- Google launch blog: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/

## WHO partograph and Labour Care Guide

### Supported

- Modified WHO partograph uses an Alert line beginning at 4 cm with 1 cm/hour slope and an Action line 4 hours to the right.
- WHO 2018 guidance says a fixed 1 cm/hour threshold alone is not recommended as a routine trigger for obstetric intervention.
- WHO Labour Care Guide (2020) is the current WHO direction and uses different labour-progress thresholds and broader care documentation.

### Sources

- WHO Partograph User's Manual, WHO/FHE/MSM/93.9: https://iris.who.int/bitstream/handle/10665/58903/WHO_FHE_MSM_93.9.pdf
- WHO recommendations: Intrapartum care for a positive childbirth experience: https://iris.who.int/server/api/core/bitstreams/ba043cf7-cba4-484d-bf7e-ec79c4102d54/content
- WHO Labour Care Guide manual: https://www.who.int/publications/i/item/9789240017566
- WHO Labour Care Guide PDF: https://iris.who.int/bitstream/handle/10665/337693/9789240017566-eng.pdf
- WHO news item on Labour Care Guide: https://www.who.int/news/item/15-12-2020-monitoring-childbirth-in-a-new-era-for-maternal-health
- FIGO position statement on WHO Labour Care Guide: https://pubmed.ncbi.nlm.nih.gov/40285693/
- Review: Advancement in Partograph: WHO's Labour Care Guide: https://pmc.ncbi.nlm.nih.gov/articles/PMC9652267/

## Public partograph examples

### Useful sources

- NCBI Bookshelf — WHO partograph figures: https://www.ncbi.nlm.nih.gov/books/NBK222105/
- NCBI direct blank form image: https://www.ncbi.nlm.nih.gov/books/NBK222105/bin/p200086fbg67001.jpg
- NCBI direct directions image: https://www.ncbi.nlm.nih.gov/books/NBK222105/bin/p200086fbg68001.jpg
- PMC article with multiple partograph figures: https://pmc.ncbi.nlm.nih.gov/articles/PMC5783902/
- Modified WHO partograph quality-improvement article with completed figure: https://pmc.ncbi.nlm.nih.gov/articles/PMC9622032/
- USAID/Jhpiego partograph fact sheet with prolonged-labour sample: https://pdf.usaid.gov/pdf_docs/Pnact388.pdf
- Ethiopia MOH training material: https://training-covid.moh.gov.et/media/courses/ldc-l3/02_35854_en.html
- OpenLearn labour and delivery module: https://www.open.edu/openlearncreate/mod/oucontent/view.php?id=272&printable=1

### Downloaded tutorial asset

- YouTube — “HOW TO FILL PARTOGRAM - VERY EASY”, OBS & GYNO PROFESSOR, uploaded 2020-05-12: https://www.youtube.com/watch?v=Y-O8LuMRNFo
  - Local analysis assets downloaded to `media/Y-O8LuMRNFo/` on 2026-05-10.
  - Useful for prototype/training inspiration only; do not redistribute video frames without checking YouTube/channel permissions.
  - Analysis note: `knowledge/partoguard_video_analysis.md`.

## Corpus-building and privacy sources

### Confirmed source classes

- HHS HIPAA de-identification guidance: https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
  - Use as a conservative screening baseline for PHI identifiers. It is not a legal determination for this project.
- WHO Labour Care Guide implementation resource package: https://www.who.int/publications/i/item/9789240109346
  - 2025 implementation package; CC BY-NC-SA 3.0 IGO; useful for training/implementation knowledge, not patient data.
- NICHD DASH / Consortium on Safe Labor: https://dash.nichd.nih.gov
  - Structured labour-progress data by request/DUA; useful for synthetic curve generation if access is approved. Not a source of partograph images and not redistributable.
- NICHD overview of Consortium on Safe Labor: https://www.nichd.nih.gov/about/org/dir/dph/officebranch/eb/safe-labor
  - Confirms large electronic-record labour dataset context.
- PartoMa project materials: https://publichealth.ku.dk/about-the-department/global/research/partoma-project/
  - Educational case stories and pocket-guide materials. Reuse rights for AI corpus use are unclear; contact project team before corpus inclusion.
- Frontiers 2025 digital/electronic partograph scoping review: https://www.frontiersin.org/journals/global-womens-health/articles/10.3389/fgwh.2025.1618317/full
  - Open-access review of digital/electronic partograph technologies; bibliography seed and landscape evidence.

### Handling rule

Publicly visible does not mean reusable. Raw filled partographs, screenshots, journal figures, and video frames remain local review-only until both license/reuse and PHI review pass.

## Local Gemma inference and mobile deployment

### Confirmed implementation sources

- Gemma model overview: https://ai.google.dev/gemma/docs/core
  - Confirms Gemma 4 model family context and local/open deployment positioning.
- LiteRT-LM GitHub: https://github.com/google-ai-edge/LiteRT-LM
  - Google edge runtime for local/mobile LLM inference; relevant to console and Android deployment.
- LiteRT-LM Gemma 4 E4B model card: https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm
  - Prepackaged `.litertlm` model path for E4B local/mobile experiments.
- LiteRT-LM Gemma 4 E2B model card: https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm
  - Lower-footprint fallback model path.
- Hugging Face Transformers Gemma 4 docs: https://huggingface.co/docs/transformers/main/model_doc/gemma4
  - Useful for Python prototyping and structured/tool-call experiments.
- Google Gemma llama.cpp integration: https://ai.google.dev/gemma/docs/integrations/llamacpp
  - Fallback local server path for GGUF experiments.
- Keras Gemma 4 multimodal and agentic workflows: https://keras.io/keras_hub/guides/gemma4_multimodal_and_agentic_workflows/
  - Reference for multimodal and tool/structured workflows.

### Implementation caveats

- Prefer LiteRT-LM for the console-to-Android path because it uses the same `.litertlm` model format as mobile.
- Use E4B as the primary mobile model when device memory allows; use E2B as the lower-footprint fallback and current console smoke-test model.
- Avoid relying on Ollama for function-calling until reported Gemma 4 tool-call parsing issues are resolved.
- Do not bundle multi-GB `.litertlm` model files inside a mobile APK; use development push or on-demand asset delivery.
- Raw Gemma 4 E2B was run locally on 2026-05-11 through LiteRT-LM 0.11.0 installed under `/root/partoguard-gemma`; the smoke prompt returned `GEMMA_E2B_OK`.
- Console PartoGuard can invoke that local model with `partoguard analyze IMAGE --gemma-litert-e2b`, currently as a bounded text verifier over existing candidate-point JSON.
- Linux/Python image input for this LiteRT-LM path is not treated as production-ready in this repo yet; Android/Kotlin remains the preferred path for crop/image-native Gemma integration.

## Nigeria / LMIC partograph use evidence

### Nigeria-specific sources

- Opiah et al., Niger Delta midwives and partograph completion: https://pubmed.ncbi.nlm.nih.gov/22783676/
- Oladapo et al., Ogun State peripheral maternity centres: https://www.tandfonline.com/doi/full/10.1080/01443610600811243
- Monjok et al., Calabar partograph knowledge/use barriers: https://www.dovepress.com/getfile.php?fileID=21993
- Anambra State 2024 study: https://journals.unizik.edu.ng/jbi/article/view/4334
- Fawole et al., Nigerian primary healthcare partograph utilization: http://www.ajol.info/index.php/njcp/article/viewFile/53514/42085

### LMIC/systematic sources

- Ollerhead and Osrin, barriers to partograph use: https://ncbi.nlm.nih.gov/pmc/articles/PMC4147181/
- Uganda documentation completeness audit: https://pmc.ncbi.nlm.nih.gov/articles/PMC9912393/

## Mortality and risk claims

### Unsupported / avoid

The claim “For every 30 minutes after Action Line crossed, risk of death goes up 5x” was not verified from WHO, Lancet, PMC, NCBI, or other authoritative sources during research. Do not use it in the pitch/writeup unless a primary source is found.

### Safer supported claims

- Prolonged/obstructed labour is associated with increased maternal risk.
- Partograph non-use/incomplete use is associated with poorer labour monitoring and obstructed-labour risk in several studies.
- WHO reports hundreds of preventable maternal deaths daily worldwide, giving broad context for maternal-health impact.

Useful sources:

- Reproductive Health multi-country obstructed/prolonged labour cohort: https://reproductive-health-journal.biomedcentral.com/article/10.1186/1742-4755-12-S2-S9
- Ethiopia obstructed labour meta-analysis: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0275400
- Ethiopia no-partograph-use and obstructed labour case-control: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0268938
- WHO maternal mortality report: https://www.who.int/publications/i/item/9789240108462

## Claims ledger

| Claim | Status | Notes |
|---|---|---|
| E2B/E4B support image input | Supported | Official Google docs/model card. |
| E2B/E4B support audio input | Supported | Official Google docs/model card. |
| Gemma 4 supports function calling | Supported | Official Google function-calling docs. |
| Gemma can read every messy partograph reliably | Unsupported | Needs real validation. |
| No large open filled-partograph photo dataset identified | Supported by search | State as “not identified,” not impossible. |
| Synthetic images are enough for clinical accuracy | Unsupported / avoid | Demo bootstrap only. |
| PartoGuard diagnoses obstructed labour | Unsupported / avoid | Use decision-support framing. |
| PartoGuard can highlight possible review/referral triggers | Prototype assumption | Valid if rules and confidence gating are shown. |
| Public web images are reusable training data | Unsupported / avoid | Need explicit rights plus PHI review. |
| NICHD CSL can supply labour-curve distributions | Prototype assumption | Requires DASH approval/DUA; no redistribution. |

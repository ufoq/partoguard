# partoguard/core/ — Pipeline Modules

Five-layer pipeline: schemas → imaging → extraction → rules → reports. One-way data flow only.

## STRUCTURE

```
core/
├── schemas/contracts.py     # ALL inter-module types (source of truth)
├── imaging/
│   ├── preprocess.py        # Quality gate → perspective correction → chart registration → crop
│   └── synthetic.py         # 18 reference images for tests/demos
├── extraction/
│   ├── marks.py             # CV X-mark detection (extract_x_marks)
│   └── gemma_adapter.py     # 9 verifier impls + build_verifier() factory
├── rules/engine.py          # Deterministic WHO zoning (classify_zone)
├── reports/generator.py     # Text + JSON audit output
├── pipeline.py              # analyze_image() — assembles the full chain
├── eval.py                  # Corpus + synthetic evaluation harness
└── corpus_scorer.py         # Ground-truth correctness scoring
```

## DATA FLOW

```
Image path
  → imaging/preprocess.py::preprocess_path()   → (quality ok? chart present? registered?)
  → extraction/marks.py::extract_x_marks()     → List[DilationPoint] (CV candidates)
  → extraction/gemma_adapter.py::verify()      → ExtractionResult (filtered/confirmed)
  → rules/engine.py::classify_zone()           → RuleOutput (NORMAL/ALERT/ACTION/MANUAL_REVIEW)
  → reports/generator.py                       → text report + JSON audit
```

Pipeline aborts with `MANUAL_REVIEW` at any stage if: registration fails, confidence < threshold, Gemma unavailable, or points out of range.

## WHERE TO LOOK

| Task | File | Symbol |
|------|------|--------|
| Change WHO clinical rules | `rules/engine.py` | `classify_zone()`, constants at top |
| Add Gemma verifier | `extraction/gemma_adapter.py` | New class + `build_verifier()` case |
| Change extraction prompt | `extraction/gemma_adapter.py` | `_build_remote_extraction_prompt()` |
| Change chart registration | `imaging/preprocess.py` | `preprocess_path()` |
| Add output field | `schemas/contracts.py` | `ExtractionResult`, `RuleOutput` |
| Change scoring logic | `corpus_scorer.py` | `score_manifest_entry()` |
| Add eval metric | `eval.py` | `CorpusEvalSummary` |

## CONVENTIONS

- **Strict one-way flow** — no module calls a layer above it. `pipeline.py` is the only assembler.
- **`schemas/contracts.py` owns all types** — never define `DilationPoint` or similar in other modules.
- **`rules/engine.py` is the sole clinical decision-maker** — Gemma output is data input, never risk output.
- **`pipeline.py` materializes chart crops to `tempfile`** with random names (`pg_crop_{hex}.png`) and cleans up on exit — never leave temp files.
- **Verifier protocol**: `GemmaVerifier` = `verify(extraction, chart_crop_path)`. `GemmaImageExtractor` extends with `extract_from_image(chart_crop_path)`. Pipeline detects the capability with `hasattr(verifier, "extract_from_image")`.

## ANTI-PATTERNS

- ❌ Clinical logic outside `rules/engine.py` — must stay in one auditable file
- ❌ Calling `gemma_adapter` before `preprocess_path()` confirms `chart_present=True`
- ❌ Reading `ExtractionResult.points` without checking `chart_present` and `registered`
- ❌ Changing `ZoneStatus` enum values — breaks `corpus_scorer._CURVE_ZONES` mapping
- ❌ Holding open temp crop files across awaits — always clean up in `finally`

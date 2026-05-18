# partoguard/ — Python Console Package

CLI + full pipeline: image → quality gate → CV extraction → Gemma verification → deterministic rules → audit report.

## STRUCTURE

```
partoguard/
├── cli/main.py              # Entry: 3 subcommands (analyze, generate, eval) + build_verifier() dispatch
├── core/
│   ├── pipeline.py          # analyze_image() — assembles all layers; AnalysisResult dataclass
│   ├── schemas/contracts.py # ALL inter-module types (Pydantic): DilationPoint, ExtractionResult, RuleOutput...
│   ├── extraction/
│   │   ├── gemma_adapter.py # 9 verifier impls + build_verifier() factory (1578 lines — most complex file)
│   │   └── marks.py         # CV X-mark detection: extract_x_marks()
│   ├── rules/engine.py      # classify_zone() — deterministic WHO partograph zoning
│   ├── imaging/
│   │   ├── preprocess.py    # Quality gate → perspective correction → chart registration → crop
│   │   └── synthetic.py     # generate_all() → 18 reference images (3 scenarios × 6 render variants)
│   ├── reports/generator.py # generate_text_report(), generate_json_audit(), write_json_audit()
│   ├── eval.py              # evaluate_corpus_dir(), evaluate_synthetic_dir(); EvalSummary dataclass
│   └── corpus_scorer.py     # score_manifest_entry() — ground-truth correctness against manifest
├── tests/                   # 148 pytest tests (see tests/AGENTS.md)
└── pyproject.toml           # Package config; entry via partoguard.cli.main:main
```

## WHERE TO LOOK

| Task | File | Symbol |
|------|------|--------|
| Add/change CLI flag | `cli/main.py` | `build_parser()` |
| Change full pipeline flow | `core/pipeline.py` | `analyze_image()` |
| Add/change Gemma verifier | `core/extraction/gemma_adapter.py` | New class + `build_verifier()` case |
| Change clinical rules | `core/rules/engine.py` | `classify_zone()` |
| Change extraction prompt | `core/extraction/gemma_adapter.py` | `_build_remote_extraction_prompt()` |
| Add output field | `core/schemas/contracts.py` | `ExtractionResult` or `RuleOutput` |
| Change chart preprocessing | `core/imaging/preprocess.py` | `preprocess_path()` |
| Add synthetic scenario | `core/imaging/synthetic.py` | `CANONICAL_SCENARIOS` |
| Add eval metric | `core/eval.py` + `core/corpus_scorer.py` | `CorpusEvalSummary` |

## CONVENTIONS

- **Never import Gemma/torch at module load** — all heavy imports deferred inside verifier methods (`import torch` inside function body).
- **All verifiers implement `GemmaVerifier` Protocol** — `verify()` + optional `extract_from_image()`. Any failure returns `_manual_review_from()`, never raises.
- **`ExtractionResult` is frozen (Pydantic)** — create new instances; never mutate.
- **`schemas/contracts.py` is the single source of truth** for all inter-module types. No ad-hoc dicts crossing module boundaries.
- **`pipeline.py` is the only assembler** — no module should call another module's orchestration logic directly.
- **Prompt strings must NOT include `<bos>`** — tokenizer adds it automatically; manual inclusion creates duplicate token and causes accuracy regression.
- **JSON response parsing pattern**: strip ` ```json...``` ` fences → find first `{` to last `}` → validate schema → `_manual_review_from()` on parse failure.
- **`pyproject.toml` maps `partoguard = "."`** — import as `from partoguard.core...` from repo root, not from inside `partoguard/`.

## ANTI-PATTERNS

- ❌ Gemma output directly setting `ZoneStatus` — only `classify_zone()` decides risk
- ❌ `import torch` / `import litert_lm` at module top level — deferred imports only
- ❌ Catching bare `Exception` without returning `_manual_review_from()` — every failure must safe-fail
- ❌ Mutating `ExtractionResult` fields — frozen Pydantic model; use `.model_copy(update={...})`
- ❌ Including `<bos>` in prompt strings — regression trigger, breaks eval accuracy

## KEY CONSTANTS

```python
# rules/engine.py
ACTIVE_PHASE_DILATION_CM = 4.0     # Active phase starts ≥ 4 cm
ACTION_LINE_START_HOURS  = 4.0     # Action line starts at 4h from alert
UNCERTAINTY_CM           = 0.5     # Indeterminate band ± 0.5 cm

# imaging/preprocess.py
MIN_CONFIDENCE                   = 0.42
BLUR_SCORE_THRESHOLD             = 20.0
REGISTRATION_CONFIDENCE_THRESHOLD = 0.55
```

## COMMANDS

```bash
.venv/bin/pytest partoguard/tests/ -v                        # 148 tests
.venv/bin/pyright                                             # 0 errors expected
.venv/bin/partoguard analyze <img> --gemma-remote
.venv/bin/partoguard eval --corpus-dir data --gemma-remote --progress
.venv/bin/partoguard generate
```

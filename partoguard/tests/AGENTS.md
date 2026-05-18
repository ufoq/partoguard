# partoguard/tests/ — Test Suite

148 pytest tests. No conftest.py; all fixtures inline. Never call real GPU/LiteRT/VPS.

## FILES

| File | Tests | Covers |
|------|-------|--------|
| `test_gemma_adapter.py` | 42 | All 9 verifier impls; subprocess/HTTP mocked |
| `test_corpus_scorer.py` | 22 | Scoring correctness against all curve_type/category combos |
| `test_remote_gemma_extractor.py` | 18 | `RemoteGemmaExtractor` + `LiteRTGemmaDaemonExtractor` |
| `test_pipeline.py` | 12 | `analyze_image()` integration; `StubVerifier` used throughout |
| `test_rules.py` | 13 | `classify_zone()` edge cases (boundary points, single point, empty) |
| `test_preprocess.py` | 6 | Image quality + registration on synthetic inputs |
| `test_extraction.py` | 4 | CV X-mark detection |
| `test_scaffold.py` | 8 | Utility helpers |
| `test_synthetic.py` | 9 | Corpus generation determinism + manifest correctness |
| `test_eval.py` | 4 | Evaluation harness summary math |
| `test_reports.py` | 2 | Output format correctness |
| `test_integration.py` | 3 | End-to-end CLI via `subprocess.run()` |

## CONVENTIONS

- **No `conftest.py`** — all fixtures defined inline per file with `@pytest.fixture`
- **`tmp_path` for all file I/O** — never write to hard-coded paths
- **Synthetic images on-the-fly** via `imaging/synthetic.py::generate_scenario()` — no pre-baked test assets
- **`StubVerifier` for pipeline tests** — tests the pipeline logic, not the model
- **HTTP mocked for `RemoteGemmaExtractor`** — `requests` is monkey-patched; never hits real VPS
- **CLI tests via `subprocess.run()`** on the installed `partoguard` command (requires `.venv` install)

## COMMANDS

```bash
.venv/bin/pytest partoguard/tests/ -v                         # All 148 tests
.venv/bin/pytest partoguard/tests/test_rules.py -v            # Rules only
.venv/bin/pytest partoguard/tests/ -k "remote" -v             # Remote extractor tests
.venv/bin/pytest partoguard/tests/ --cov=partoguard           # Coverage
```

## ANTI-PATTERNS

- ❌ Tests calling real GPU, LiteRT, or VPS endpoints — always stub/mock
- ❌ Hard-coded file paths outside `tmp_path`
- ❌ Deleting or skipping failing tests — fix the code
- ❌ Asserting on internal `gemma_adapter.py` parsing details — test output contract only

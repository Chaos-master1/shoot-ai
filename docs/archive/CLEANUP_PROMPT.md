# Comprehensive Code Review & Improvement Prompt

You are reviewing the **Kawkab AI** football analytics platform (`src/kawkab/`) — a 120+ module system with ~4800 unit tests, covering everything from YOLO tracking to xG models to tactical analysis. Your job is to inspect every corner, find issues, and fix them.

---

## Project Context

- **Python 3.12+**, build: hatchling, test: pytest 9.x (asyncio_mode=auto, hypothesis, coverage)
- **Frontend**: Vanilla HTML/CSS/JS (34 IIFE files), esbuild bundle, Chart.js, Three.js, matter.js
- **Architecture**: `core/` (analytical models), `services/` (business logic), `ui/bridge.py` (PySide6 desktop bridge), `api/` (FastAPI), `web/` (static SPA)
- **External**: 12 data providers (StatsBomb, Opta, Wyscout, etc.), Ollama LLM, 8 cloud services
- **CLI**: `python -m kawkab <track|batch|evaluate|render|events|possession|...>`
- **Tests**: 232 unit + 6 e2e + 3 integration files; `tests/conftest.py` has `install_kawkab_stubs()` for isolation
- **Test command**: `$env:PYTHONPATH="src"; $env:KAWKAB_TEST_MODE="1"; python -m pytest tests/unit/ --ignore=tests/unit/test_audio_service.py -q`

---

## Areas to Cover (do ALL of these)

### 1. Code Quality & Anti-patterns
- Unused imports, dead code, commented-out blocks, TODO/FIXME/HACK without tracking
- Overly long functions (>50 lines), deeply nested conditionals, duplicated logic
- Inconsistent naming (snake_case vs camelCase, singular vs plural)
- Missing type hints, incorrect type annotations, `Any` overuse
- Mutable default args, bare `except:`, `except Exception:` without narrowing
- Inefficient loops, redundant list/dict comprehensions, CPU-bound work without vectorization

### 2. Error Handling
- Missing try/except around file I/O, network calls, subprocess, DB operations
- Silent `except: pass` swallowing errors
- Functions returning inconsistent types (None vs object vs error code)
- Unhandled edge cases: empty inputs, malformed data, missing files, division by zero

### 3. Security
- SQL injection vectors: raw f-strings in SQL, unvalidated column names
- Path traversal: user input in file paths without `resolve()` or containment check
- Hardcoded secrets/API keys in code (not via env vars)
- XSS in frontend: unescaped user data in innerHTML, dynamic eval, `bridge.trigger` arg unsanitized

### 4. Testing
- Tests that pass alone but fail in full suite (ordering dependencies, shared mutable state)
- Missing edge cases: empty lists, None inputs, boundary values, concurrent access
- Fragile assertions: time-sensitive checks, hardcoded paths, order-dependent comparisons
- Test files that call `install_kawkab_stubs()` or `sys.path.insert` at module level (causes ordering issues)
- Async tests without proper cleanup (event loop leaks, unclosed clients)
- Mock leakage: `sys.modules` pollution, patched objects not restored

### 5. Performance
- Database N+1 queries (loops with per-item DB calls instead of bulk operations)
- Unbounded list growth (appending without limits or pruning)
- Repeated file reads, redundant computations, missing `lru_cache`
- Large objects held in memory longer than needed

### 6. Architecture & Design
- Circular imports, module-level side effects (code that runs on import)
- Tight coupling: services importing other services directly instead of through abstractions
- Missing interfaces/ABCs where multiple implementations exist (storage backends, data providers)
- God classes: `bridge.py` (was 3352 lines), `analysis_service.py`, `storage_service.py`
- Inconsistent patterns: some async, some sync, some files use `load_service_module()`

### 7. Frontend
- Browser console errors, unhandled promise rejections, missing `.catch()`
- DOM leaks: detached elements, unbound event listeners, interval/timeout not cleared
- Performance: layout thrashing, excessive reflows, large DOM operations without DocumentFragment
- i18n gaps: missing `data-i18n` attributes, hardcoded English strings
- Accessibility: missing aria labels, focus management, keyboard navigation gaps

### 8. Configuration & Build
- Hardcoded paths, environment-dependent behavior not documented
- Missing or incorrect dependency versions in pyproject.toml
- Build scripts that assume POSIX paths (broken on Windows)
- Dockerfile inefficiencies, missing `.dockerignore`

### 9. Data Integrity
- Schema drift: migration files that don't match actual DB state
- Missing FK constraints, indexes on foreign key columns
- String columns where enums should be used
- Inconsistent column naming across migrations

### 10. Documentation
- Missing or misleading docstrings
- Stale comments that don't match code
- Public API methods without usage examples
- README features that no longer exist

---

## Fix Protocol

For each issue found:
1. **Fix it directly** — edit the file, don't just report it
2. **Verify**: run the test suite after any change to ensure nothing broke
3. **If fixing many files**, batch related changes to minimize test runs

## Verification After ALL Fixes

Run this to confirm nothing broke:
```powershell
$env:PYTHONPATH="src"; $env:KAWKAB_TEST_MODE="1"; python -m pytest tests/unit/ --ignore=tests/unit/test_audio_service.py -q --tb=line
```
Target: **0 failures, 0 errors** (currently achieved after our last session).

---

## Key Gotchas From Previous Debugging

- Do NOT add `sys.path.insert` or `install_kawkab_stubs()` at module level in test files — `tests/unit/conftest.py` already handles this
- `kawkab.core.paths.knowledge_base` is a `@property` (no setter) on the real `Paths` class; use `_FakePaths` in tests
- YAML stub leak: if you install a yaml mock in `sys.modules["yaml"]`, you MUST restore `yaml.safe_load` after tests
- `time.monotonic()` timing assertions fail after certain test orderings; use structural assertions instead
- `KAWKAB_DB_URL` must be popped before tests (already in `install_kawkab_stubs()`)
- Always use `executemany` for bulk DB operations, never per-row `execute`
- `cli_args` in conftest must pass `--skip 6` for tracking tests to complete in reasonable time

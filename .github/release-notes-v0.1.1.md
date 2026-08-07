## DeafBench v0.1.1

This is the first finalized, review-hardened DeafBench release from the current `main` branch. It packages the accessibility benchmark, CLI, reporting, validation, test coverage, and release automation completed after the original `v0.1.0` tag.

### Highlights

- Accessibility-focused ASR and caption evaluation beyond WER.
- Installable `deafbench` CLI with `compare` and `report` commands.
- Critical-information recall, non-speech information recall, optional speaker attribution accuracy, and median latency metrics.
- JSONL benchmark parsing and deterministic reference/prediction alignment.
- Markdown report generation with failure details.

### Core benchmark and CLI

- Added WER evaluation using JiWER.
- Added critical-information recall with per-sample missed-term reporting.
- Added non-speech information recall for annotations such as `[alarm]`, `[laughter]`, and `[door closes]`.
- Added optional speaker-attribution accuracy.
- Added optional median `latency_ms` reporting.
- Added `deafbench compare <references> <predictions>` for terminal summaries.
- Added `deafbench report <references> <predictions> -o <report.md>` for Markdown reports.
- Added `python -m deafbench` support and the installed `deafbench` console entry point.
- Added example reference and model JSONL files.

### Correctness and data integrity

- WER evaluation failures are surfaced as unavailable/NaN instead of being silently reported as a perfect zero-error score.
- Critical terms now use escaped token-boundary matching so values such as `25` do not match inside `125`.
- Non-speech annotations now use token-boundary matching so `[alarm]` does not match as part of a larger normalized token such as `[alarm]tone`.
- ID alignment no longer falls back positionally after an ID lookup misses, preventing one prediction from being scored against multiple references.
- Null IDs are excluded from ID maps while positional alignment remains supported when inputs do not provide usable IDs.
- Duplicate non-null IDs are rejected independently in references and predictions before alignment.
- JSONL parsing now rejects non-object records.
- Supplied `text` values must be strings.
- Supplied `critical` and `sounds` values must be lists.
- Invalid latency values are rejected when malformed, negative, NaN, or infinite, with the sample ID included in the error.
- CLI evaluation errors are reported cleanly to `stderr` with exit status `1`.
- Report-writing filesystem errors are handled without an uncaught traceback.
- Markdown table values are escaped so pipes, backslashes, and line breaks cannot corrupt generated reports.

### Testing and quality gates

- Added low-level unit tests for pure metric and alignment behavior.
- Added integration tests spanning JSONL parsing, alignment, metrics, and reporting.
- Added functional CLI tests for complete compare/report flows and error handling.
- Added a subprocess smoke test for installed-package health.
- Enforced branch coverage with a 90% minimum coverage gate.
- CI validates Python 3.11, 3.12, 3.13, and 3.14.
- CI runs unit, integration, functional, smoke, full-suite coverage, and installed-CLI checks.

### Packaging and dependency hardening

- Added PEP 517/518 packaging through `pyproject.toml`.
- Added the `deafbench` console script entry point.
- Added a declared `test` extra with `pytest` and `pytest-cov`.
- Constrained JiWER to the tested 3.x range: `jiwer>=3.0.0,<4.0.0`.
- Updated contributor setup to install `.[test]`, matching CI.
- Bumped package and runtime versions together to `0.1.1`.

### CI and release security

- Added GitHub Actions CI and release workflows.
- Pinned every external GitHub Action reference to an immutable 40-character commit SHA.
- Disabled checkout credential persistence where the workflow only requires a read-only working tree.
- Added release distribution build and `twine check` validation.
- Added PyPI Trusted Publishing through GitHub OIDC with `id-token: write` scoped to the publish job.

### Compatibility

- Python: 3.11, 3.12, 3.13, 3.14 tested in CI.
- License: Apache-2.0.

### Full comparison

https://github.com/488315/DeafBench/compare/v0.1.0...v0.1.1

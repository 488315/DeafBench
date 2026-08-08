# Recorder CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an installed `deafbench recorder` command that launches the existing recorder anywhere and bootstraps bundled benchmark references into a local workspace.

**Architecture:** Move recorder runtime code under `deafbench.recorder`, keep `tools.recorder` as a compatibility wrapper, and package Core v1 plus Non-speech v1 references under `deafbench.recorder.data`. The top-level CLI parses the recorder subcommand without importing optional audio dependencies, then lazily hands control to the recorder runtime.

**Tech Stack:** Python 3.11+, argparse, tkinter, numpy, sounddevice, importlib.resources, setuptools package data, pytest.

## Global Constraints

- `deafbench recorder` must work after `pip install "deafbench[recorder]"` outside a source checkout.
- `core-v1` is the default dataset.
- `deafbench recorder --dataset non-speech-v1` must work.
- Existing references must never be overwritten.
- Normal non-recorder CLI commands must not import optional recorder runtime dependencies.
- Keep the legacy `python -m tools.recorder.recorder` workflow working.
- Use AOSP-style commit messages and RED/GREEN behavior commits.

---

### Task 1: Lock the CLI and workspace contract with failing tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_recorder.py`

**Interfaces:**
- Consumes: existing `deafbench.cli.main(args)`.
- Produces: expected `deafbench recorder` dispatch behavior and `ensure_dataset_workspace(repo_root, dataset)` behavior.

- [ ] **Step 1: Add a CLI test that runs `main(["recorder", "--dataset", "non-speech-v1"])` with the recorder launcher monkeypatched and asserts the dataset is forwarded.**
- [ ] **Step 2: Add recorder tests asserting bundled Core v1 references are copied into `<workspace>/benchmarks/core-v1/references.jsonl`, existing references are preserved, and custom missing datasets fail clearly.**
- [ ] **Step 3: Push the test-only commit and verify CI fails for the missing command/module behavior.**
- [ ] **Step 4: Commit with `cli: require installed recorder command` and record the failing CI run in the PR.**

### Task 2: Package the recorder runtime and bundled references

**Files:**
- Create: `deafbench/recorder/__init__.py`
- Create: `deafbench/recorder/core.py`
- Create: `deafbench/recorder/app.py`
- Create: `deafbench/recorder/data/__init__.py`
- Create: `deafbench/recorder/data/core-v1.jsonl`
- Create: `deafbench/recorder/data/non-speech-v1.jsonl`
- Modify: `pyproject.toml`
- Modify: recorder tests to import the packaged implementation.

**Interfaces:**
- Produces: `deafbench.recorder.app.main(argv=None) -> int`, `deafbench.recorder.app.ensure_dataset_workspace(repo_root, dataset) -> tuple[Path, Path]`, and packaged benchmark reference resources.

- [ ] **Step 1: Copy recorder core behavior into `deafbench.recorder.core` without changing audio semantics.**
- [ ] **Step 2: Move the Tk app into `deafbench.recorder.app` and change the default workspace root to `Path.cwd()`.**
- [ ] **Step 3: Implement `ensure_dataset_workspace` using `importlib.resources` so bundled reference JSONL is copied only when the workspace reference is absent.**
- [ ] **Step 4: Add package-data configuration for the two JSONL resources.**
- [ ] **Step 5: Run the focused recorder tests and make them pass.**

### Task 3: Wire the top-level `deafbench recorder` command lazily

**Files:**
- Modify: `deafbench/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `deafbench.recorder.app.main(argv)`.
- Produces: `deafbench recorder [--dataset ...] [--repo-root ...] [--references ...] [--audio-dir ...]`.

- [ ] **Step 1: Add a `recorder` argparse subcommand with the existing recorder options.**
- [ ] **Step 2: Handle the recorder branch before compare/report input loading.**
- [ ] **Step 3: Lazy-import the recorder app only after `recorder` is selected and forward only explicitly parsed recorder arguments.**
- [ ] **Step 4: Run CLI functional tests and the full test suite.**
- [ ] **Step 5: Commit GREEN as `cli: launch installed recorder`.**

### Task 4: Preserve repository compatibility and document usage

**Files:**
- Modify: `tools/recorder/recorder.py`
- Modify: `tools/recorder/core.py`
- Modify: `README.md`

**Interfaces:**
- Produces: old module invocation remains available while the installed command becomes the primary workflow.

- [ ] **Step 1: Replace the old recorder script with a compatibility wrapper that invokes `deafbench.recorder.app.main`.**
- [ ] **Step 2: Re-export recorder core helpers from the packaged implementation for source-tree compatibility.**
- [ ] **Step 3: Document `pip install "deafbench[recorder]"`, `deafbench recorder`, dataset selection, and local workspace paths.**
- [ ] **Step 4: Run the full CI matrix and installed CLI smoke test.**
- [ ] **Step 5: Commit as `docs: document recorder command` if documentation is the only remaining delta.**

### Task 5: PR review gate

**Files:**
- Review all changed files.

- [ ] **Step 1: Open the PR only after GREEN CI.**
- [ ] **Step 2: Run CodeRabbit/Copilot review and inspect every inline thread.**
- [ ] **Step 3: Fix only valid findings with focused commits and regression tests where behavior changes.**
- [ ] **Step 4: Resolve fixed threads, rerun CI, and request a final review.**
- [ ] **Step 5: Mark ready only when CI passes and no actionable/unresolved review findings remain.**

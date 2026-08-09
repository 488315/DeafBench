# Benchmark Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an installed `deafbench benchmark` command that automatically selects a complete human audio set or generates a complete synthetic set with WhisperSpeech, runs Whisper or Whisper-AT, evaluates it, and writes source-aware predictions, report, and run metadata.

**Architecture:** Keep CLI parsing in `deafbench.cli`, orchestration in `deafbench.benchmark.runner`, dataset/source/run path logic in `deafbench.benchmark.workspace`, deterministic scene planning/mixing in `deafbench.benchmark.scenes`, synthetic-set transactions in `deafbench.benchmark.synthetic`, and heavyweight model/TTS imports behind lazy adapters. Existing `tools/transcribe_whisper.py` and `tools/transcribe_whisper_at.py` become compatibility entry points over the packaged adapters only after tests prove equivalent behavior.

**Tech Stack:** Python 3.11-3.14, argparse, pathlib, dataclasses, hashlib, json, tempfile, wave, NumPy, existing DeafBench parser/metrics/report modules, WhisperSpeech `whisperspeech.pipeline.Pipeline`, OpenAI Whisper, Whisper-AT, pytest.

## Global Constraints

- `deafbench benchmark <dataset> --model whisper|whisper-at` is the public command.
- `--audio-source` accepts exactly `auto`, `human`, or `synthetic`; default is `auto`.
- `auto` uses human audio only when every reference has exactly one valid WAV; otherwise it uses one complete synthetic set.
- Never mix human and synthetic WAVs in one benchmark run.
- Human audio remains under `benchmarks/<dataset>/audio/`.
- Synthetic audio remains under `benchmarks/<dataset>/audio-synthetic/` with `manifest.jsonl`.
- New results go only under `benchmarks/<dataset>/runs/<model>/<human|synthetic>/`.
- Keep existing top-level `model-a.jsonl`, `model-b.jsonl`, and reports compatible with existing repository tools.
- Whisper maps to model identity `model-a` and defaults to Whisper `turbo`.
- Whisper-AT maps to model identity `model-b` and defaults to `medium.en`, `at_time_res=10.0`, `top_k=5`, `p_threshold=-1.0`.
- Environmental sound labels remain structured prediction data and must not be appended to ASR `text`.
- Synthetic scene profile is `default-v1`; default seed is `42`; final WAV format is 48 kHz, 16-bit PCM, mono.
- WhisperSpeech generates speech only. DeafBench owns event timing, generated ambience, mixing, resampling, final WAV validation, and manifest data.
- Generated timestamps never modify `references.jsonl`.
- No third-party sound-effect downloads are introduced; reuse DeafBench deterministic event cues.
- Normal `compare`, `report`, `recorder`, and `--help` paths must not import Whisper, Whisper-AT, WhisperSpeech, torch, or torchaudio.
- CI must not download or execute real model weights.
- Every behavior change starts with a test-only RED commit and is followed by a production-only GREEN commit unless production and test separation is impossible for a pure compatibility move.
- Run the existing Python 3.11, 3.12, 3.13, and 3.14 CI matrix after every GREEN checkpoint.
- Do not add Qwen2.5-Omni, the personal Deaf/cochlear speech track, metric changes, arbitrary shell model plugins, or unrelated refactors in this PR.
- Upstream check on 2026-08-09: WhisperSpeech repository reports package version `0.8.9`; `whisperspeech.pipeline.Pipeline` exposes `generate_to_file(fname, text, speaker=None, lang='en', cps=15, ...)`. Use that file-producing API so DeafBench discovers the generated WAV sample rate instead of assuming it.

---

## File Structure

Create or modify these units only:

```text
deafbench/
  cli.py
  benchmark/
    __init__.py
    workspace.py
    scenes.py
    synthetic.py
    runner.py
    models/
      __init__.py
      whisper.py
      whisper_at.py

tools/
  transcribe_whisper.py
  transcribe_whisper_at.py

 tests/
  test_benchmark_cli.py
  test_benchmark_workspace.py
  test_benchmark_scenes.py
  test_benchmark_synthetic.py
  test_benchmark_models.py
  test_benchmark_runner.py
  test_transcribe_whisper.py
  test_transcribe_whisper_at.py

README.md
pyproject.toml
```

`workspace.py` has no model imports. `scenes.py` has NumPy plus the existing recorder event synthesizer. `synthetic.py` receives a speech-generator callable so tests never import WhisperSpeech. Model adapters receive fake modules/backends in tests or lazy-import their real runtime only inside the adapter entry function.

---

### Task 1: Add the benchmark CLI contract and lazy dispatch

**Files:**
- Modify: `deafbench/cli.py`
- Create: `deafbench/benchmark/__init__.py`
- Create: `tests/test_benchmark_cli.py`

**Interfaces:**
- Produces: `deafbench.cli._run_benchmark(benchmark_args: list[str]) -> int`
- Produces CLI arguments: positional `dataset`, required `--model`, optional `--audio-source` defaulting to `auto`, `--repo-root`, `--scene-profile` defaulting to `default-v1`, and `--seed` defaulting to `42`.
- `deafbench.cli` must not import `deafbench.benchmark.runner` until the `benchmark` subcommand is selected.

- [ ] **Step 1: Write the failing functional tests**

Create `tests/test_benchmark_cli.py`:

```python
import sys

import pytest

from deafbench.cli import main


pytestmark = pytest.mark.functional


def test_benchmark_command_forwards_defaults_to_lazy_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr("deafbench.cli._run_benchmark", calls.append, raising=False)

    main(["benchmark", "core-v1", "--model", "whisper"])

    assert calls == [[
        "core-v1",
        "--model", "whisper",
        "--audio-source", "auto",
        "--scene-profile", "default-v1",
        "--seed", "42",
    ]]


def test_benchmark_command_returns_launcher_status(monkeypatch):
    monkeypatch.setattr(
        "deafbench.cli._run_benchmark",
        lambda _args: 9,
        raising=False,
    )

    assert main(["benchmark", "core-v1", "--model", "whisper"]) == 9


def test_compare_does_not_import_benchmark_runner(tmp_path, monkeypatch):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    predictions.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    sys.modules.pop("deafbench.benchmark.runner", None)

    main(["compare", str(references), str(predictions)])

    assert "deafbench.benchmark.runner" not in sys.modules
```

- [ ] **Step 2: Run RED**

Run:

```bash
python -m pytest tests/test_benchmark_cli.py -v
```

Expected: FAIL because `_run_benchmark` and the `benchmark` parser do not exist.

Commit only the test file:

```text
cli: test benchmark command dispatch

Define the installed benchmark command contract before adding runtime code.

Test: python -m pytest tests/test_benchmark_cli.py -v
Bug: N/A
```

- [ ] **Step 3: Implement minimal lazy CLI dispatch**

Add this helper shape to `deafbench/cli.py`:

```python
def _run_benchmark(benchmark_args: list[str]) -> int:
    from .benchmark.runner import main as benchmark_main
    return benchmark_main(benchmark_args)
```

Add an argparse `benchmark` subparser and build the forwarded argument list without importing the runner during parser construction. Keep all existing `compare`, `report`, and `recorder` behavior unchanged.

Create `deafbench/benchmark/__init__.py` containing only:

```python
"""Installed DeafBench benchmark orchestration."""
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
python -m pytest tests/test_benchmark_cli.py tests/test_cli.py -v
```

Expected: all selected tests PASS.

Commit production only:

```text
cli: add benchmark command dispatch

Lazy-load benchmark orchestration so existing lightweight commands keep their
current dependency surface.

Test: python -m pytest tests/test_benchmark_cli.py tests/test_cli.py -v
Bug: N/A
```

- [ ] **Step 5: Verify CI checkpoint**

Push and require the full Python 3.11-3.14 CI matrix to pass before Task 2.

---

### Task 2: Resolve source-aware paths and inspect complete audio sets

**Files:**
- Create: `deafbench/benchmark/workspace.py`
- Create: `tests/test_benchmark_workspace.py`

**Interfaces:**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AudioSource = Literal["auto", "human", "synthetic"]
ResolvedAudioSource = Literal["human", "synthetic"]

@dataclass(frozen=True)
class AudioSetStatus:
    complete: bool
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    invalid: tuple[str, ...]

@dataclass(frozen=True)
class RunPaths:
    dataset_dir: Path
    references: Path
    human_audio: Path
    synthetic_audio: Path
    run_dir: Path
    predictions: Path
    report: Path
    metadata: Path


def validate_dataset_name(dataset: str) -> str: ...
def load_reference_ids(path: Path) -> tuple[str, ...]: ...
def validate_wav_format(path: Path) -> None: ...
def inspect_audio_set(references: Path, audio_dir: Path) -> AudioSetStatus: ...
def resolve_audio_source(requested: AudioSource, human_status: AudioSetStatus) -> ResolvedAudioSource: ...
def resolve_run_paths(repo_root: Path, dataset: str, model: str, source: ResolvedAudioSource) -> RunPaths: ...
```

- [ ] **Step 1: Write RED tests for path resolution and source selection**

Create tests covering the exact rules:

```python
def test_run_paths_are_source_aware(tmp_path):
    paths = resolve_run_paths(tmp_path, "non-speech-v1", "whisper-at", "synthetic")
    root = tmp_path / "benchmarks" / "non-speech-v1"
    assert paths.references == root / "references.jsonl"
    assert paths.human_audio == root / "audio"
    assert paths.synthetic_audio == root / "audio-synthetic"
    assert paths.run_dir == root / "runs" / "whisper-at" / "synthetic"
    assert paths.predictions == paths.run_dir / "predictions.jsonl"
    assert paths.report == paths.run_dir / "report.md"
    assert paths.metadata == paths.run_dir / "run.json"


def test_auto_prefers_complete_human_set():
    status = AudioSetStatus(True, (), (), ())
    assert resolve_audio_source("auto", status) == "human"


def test_auto_uses_synthetic_for_any_incomplete_human_set():
    status = AudioSetStatus(False, ("core-002",), (), ())
    assert resolve_audio_source("auto", status) == "synthetic"


def test_explicit_human_rejects_incomplete_set():
    status = AudioSetStatus(False, ("core-002",), (), ())
    with pytest.raises(ValueError, match="Human audio set is incomplete"):
        resolve_audio_source("human", status)
```

Also create valid/invalid WAV fixtures and prove `inspect_audio_set` reports missing, extra, and invalid IDs without changing either directory.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_workspace.py -v
```

Expected: import/attribute failures because `workspace.py` does not exist.

Commit:

```text
benchmark: test audio source resolution

Pin source-aware paths and the all-or-nothing human audio selection rules.

Test: python -m pytest tests/test_benchmark_workspace.py -v
Bug: N/A
```

- [ ] **Step 3: Implement workspace validation**

Use the recorder-safe dataset-name rule: reject empty, `.`, `..`, or names containing `/`, `\\`, or `:`.

`load_reference_ids` must preserve reference order and reject malformed/duplicate IDs.

`validate_wav_format` must require mono, 16-bit PCM, 48 kHz, uncompressed WAV.

`inspect_audio_set` must compare WAV stems to reference IDs and validate only expected WAV files. Sort returned `missing`, `extra`, and `invalid` tuples for stable output.

`resolve_audio_source` rules:

```python
if requested == "synthetic":
    return "synthetic"
if requested == "human":
    if not human_status.complete:
        raise ValueError("Human audio set is incomplete")
    return "human"
return "human" if human_status.complete else "synthetic"
```

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/test_benchmark_workspace.py -v
```

Expected: PASS.

Commit:

```text
benchmark: resolve complete audio sources

Keep human and synthetic inputs separate and refuse partial human benchmark
runs.

Test: python -m pytest tests/test_benchmark_workspace.py -v
Bug: N/A
```

- [ ] **Step 5: Run the existing transcriber validation tests**

```bash
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
```

Expected: PASS before moving shared logic in later tasks.

---

### Task 3: Add deterministic scene planning and mixing

**Files:**
- Create: `deafbench/benchmark/scenes.py`
- Create: `tests/test_benchmark_scenes.py`
- Reuse: `deafbench/recorder/core.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class TimedEvent:
    label: str
    start_ms: int
    end_ms: int

@dataclass(frozen=True)
class ScenePlan:
    sample_id: str
    scene_profile: str
    seed: int
    sample_rate: int
    speech_start_ms: int
    speech_end_ms: int
    scene_end_ms: int
    background_profile: str
    background_snr_db: float
    events: tuple[TimedEvent, ...]


def resample_mono(samples: np.ndarray, source_rate: int, target_rate: int = 48_000) -> np.ndarray: ...
def plan_scene(sample_id: str, speech_frames: int, sound_labels: list[str], *, seed: int = 42, scene_profile: str = "default-v1", sample_rate: int = 48_000) -> ScenePlan: ...
def mix_scene(speech_pcm: np.ndarray, plan: ScenePlan) -> np.ndarray: ...
```

**Deterministic `default-v1` rules:**
- Speech begins at exactly 500 ms.
- Scene has 500 ms trailing room after the later of speech end or last event end.
- Derive a sample-local RNG seed from the first 8 bytes of `sha256(f"{seed}:{scene_profile}:{sample_id}".encode()).digest()` interpreted unsigned big-endian.
- Generate background ambience with `np.random.default_rng(local_seed).normal(0, 1, frames)`, smooth with a 64-sample moving-average convolution, and scale it to 15.0 dB below speech RMS. If speech RMS is zero, use a fixed RMS target of `0.01` for background scaling.
- For N environmental events, create N evenly spaced target centers from 25% through 75% of the speech duration. Add deterministic jitter in `[-0.08, +0.08] * speech_duration` from the local RNG, clamp each event start between speech start and `max(speech_start, speech_end - event_duration)`, then sort by start time.
- Event PCM comes only from `deafbench.recorder.core.synthesize_sound_event(label)` and is mixed additively.
- Mix in float64, peak-normalize only when absolute peak exceeds 0.98, then convert to int16 mono.

- [ ] **Step 1: Write deterministic RED tests**

```python
def test_scene_plan_is_reproducible_for_fixed_seed():
    first = plan_scene("ns-008", 48_000 * 4, ["[phone rings]", "[knock]"], seed=42)
    second = plan_scene("ns-008", 48_000 * 4, ["[phone rings]", "[knock]"], seed=42)
    assert first == second
    assert first.speech_start_ms == 500
    assert [event.label for event in first.events] == ["[phone rings]", "[knock]"]


def test_scene_plan_changes_when_seed_changes():
    first = plan_scene("ns-008", 48_000 * 4, ["[phone rings]"], seed=42)
    second = plan_scene("ns-008", 48_000 * 4, ["[phone rings]"], seed=43)
    assert first.events != second.events


def test_mix_scene_returns_48khz_int16_mono():
    speech = np.full((48_000, 1), 1000, dtype=np.int16)
    plan = plan_scene("ns-001", len(speech), ["[alarm]"])
    mixed = mix_scene(speech, plan)
    assert mixed.dtype == np.int16
    assert mixed.ndim == 2
    assert mixed.shape[1] == 1
    assert len(mixed) == plan.scene_end_ms * 48
```

Also test `resample_mono` converts a 24 kHz one-second fixture to exactly 48,000 frames.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_scenes.py -v
```

Commit:

```text
benchmark: test deterministic synthetic scenes

Define reproducible timing, resampling, ambience, and event-mixing behavior.

Test: python -m pytest tests/test_benchmark_scenes.py -v
Bug: N/A
```

- [ ] **Step 3: Implement `scenes.py` exactly to the rules above**

Do not import WhisperSpeech or either recognition model in this module. Validate `scene_profile == "default-v1"`; other values raise `ValueError("Unsupported scene profile: ...")`.

- [ ] **Step 4: Run GREEN plus existing event tests**

```bash
python -m pytest tests/test_benchmark_scenes.py tests/test_sound_events.py tests/test_recorder_core_regressions.py -v
```

Expected: PASS.

Commit:

```text
benchmark: build deterministic synthetic scenes

Generate reproducible DeafBench-owned ambience and timed environmental events
around supplied speech PCM.

Test: python -m pytest tests/test_benchmark_scenes.py tests/test_sound_events.py tests/test_recorder_core_regressions.py -v
Bug: N/A
```

- [ ] **Step 5: CI checkpoint**

Require all Python versions green before synthetic generation is layered on top.

---

### Task 4: Generate complete WhisperSpeech synthetic sets transactionally

**Files:**
- Create: `deafbench/benchmark/synthetic.py`
- Create: `tests/test_benchmark_synthetic.py`
- Modify: `pyproject.toml`

**Interfaces:**

```python
@dataclass(frozen=True)
class SpeechAudio:
    samples: np.ndarray
    sample_rate: int

@dataclass(frozen=True)
class TTSInfo:
    engine: str
    version: str

SpeechGenerator = Callable[[str], SpeechAudio]


def create_whisperspeech_generator() -> tuple[SpeechGenerator, TTSInfo]: ...
def generation_fingerprint(references: Path, *, scene_profile: str, seed: int, tts_info: TTSInfo) -> str: ...
def synthetic_set_is_current(audio_dir: Path, references: Path, *, scene_profile: str, seed: int, tts_info: TTSInfo) -> bool: ...
def generate_synthetic_set(references: Path, audio_dir: Path, speech_generator: SpeechGenerator, tts_info: TTSInfo, *, scene_profile: str = "default-v1", seed: int = 42) -> Path: ...
```

**WhisperSpeech runtime adapter:**
- Lazy import `from whisperspeech.pipeline import Pipeline` only inside `create_whisperspeech_generator`.
- Instantiate `Pipeline()` once per generator.
- For each text, call `Pipeline.generate_to_file(temp_wav, text, lang="en")`.
- Read the generated WAV's channel count, sample width, sample rate, and frames. Downmix integer PCM if the generated file contains more than one channel, return `SpeechAudio` with its actual sample rate, and let `scenes.resample_mono` convert to 48 kHz.
- Obtain runtime package version with `importlib.metadata.version("WhisperSpeech")`; fall back to `"unknown"` only if metadata is absent but import succeeded.
- On missing import, raise a purpose-specific error that the CLI can render as: `WhisperSpeech is not installed. Run: python -m pip install "deafbench[benchmark]"`.

**Manifest record fields:**

```json
{
  "id": "ns-008",
  "fingerprint": "sha256...",
  "scene_profile": "default-v1",
  "seed": 42,
  "sample_rate": 48000,
  "tts": {"engine": "whisperspeech", "version": "0.8.9"},
  "speech": {"start_ms": 500, "end_ms": 4210},
  "background": {"profile": "office-v1", "start_ms": 0, "end_ms": 5000, "snr_db": 15.0},
  "events": [{"label": "[phone rings]", "start_ms": 1800, "end_ms": 2700}]
}
```

- [ ] **Step 1: Write RED tests for all-or-nothing generation and cache decisions**

Tests must use a fake speech generator and never import WhisperSpeech:

```python
def fake_speech(text: str) -> SpeechAudio:
    frames = max(4_800, len(text) * 800)
    return SpeechAudio(np.full((frames, 1), 1200, dtype=np.int16), 48_000)


def test_generate_synthetic_set_writes_complete_wavs_and_manifest(tmp_path):
    references = _write_references(tmp_path, [
        {"id": "ns-001", "text": "Stay seated.", "sounds": ["[alarm]"]},
        {"id": "ns-002", "text": "Wait outside.", "sounds": []},
    ])
    audio_dir = tmp_path / "audio-synthetic"
    generate_synthetic_set(
        references,
        audio_dir,
        fake_speech,
        TTSInfo("whisperspeech", "test"),
    )
    assert {path.name for path in audio_dir.glob("*.wav")} == {"ns-001.wav", "ns-002.wav"}
    records = [json.loads(line) for line in (audio_dir / "manifest.jsonl").read_text().splitlines()]
    assert [record["id"] for record in records] == ["ns-001", "ns-002"]
    assert records[0]["events"][0]["label"] == "[alarm]"


def test_failed_regeneration_preserves_previous_complete_set(tmp_path):
    # Seed a valid set, then use a generator that raises on the second sample.
    # Assert every original WAV and manifest byte remains unchanged.
    ...
```

Replace the final comment with full test setup during implementation; do not commit a placeholder. Also test: partial set is stale, fingerprint mismatch is stale, matching fingerprint is reusable, and no human `audio/` file is touched.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_synthetic.py -v
```

Commit:

```text
benchmark: test synthetic set generation

Require complete timestamped synthetic datasets and preservation of the last
valid generated set on failure.

Test: python -m pytest tests/test_benchmark_synthetic.py -v
Bug: N/A
```

- [ ] **Step 3: Implement staging-directory generation and manifest fingerprinting**

Generate into a sibling temporary directory. After every WAV and the manifest validate, swap directories using:

```text
existing audio-synthetic -> temporary backup
new staging              -> audio-synthetic
remove backup only after successful promotion
restore backup if promotion fails
```

Use `atomic_write_wav` from `deafbench.recorder.core` inside the staging directory. Validate the promoted set with `inspect_audio_set` before deleting the backup.

Add this optional extra in `pyproject.toml`:

```toml
benchmark = [
    "numpy>=1.26",
    "WhisperSpeech>=0.8.9",
]
```

Do not add Whisper or Whisper-AT to the base dependencies.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/test_benchmark_synthetic.py tests/test_benchmark_scenes.py tests/test_recorder.py -v
```

Expected: PASS.

Commit:

```text
benchmark: generate WhisperSpeech audio sets

Create complete source-isolated synthetic benchmark audio with reproducible
scene metadata and safe regeneration.

Test: python -m pytest tests/test_benchmark_synthetic.py tests/test_benchmark_scenes.py tests/test_recorder.py -v
Bug: N/A
```

- [ ] **Step 5: Check package metadata without installing model weights**

```bash
python -m pip install -e ".[test]"
python -c "import tomllib, pathlib; data=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); assert 'benchmark' in data['project']['optional-dependencies']"
```

---

### Task 5: Package the Whisper Model A adapter

**Files:**
- Create: `deafbench/benchmark/models/__init__.py`
- Create: `deafbench/benchmark/models/whisper.py`
- Create: `tests/test_benchmark_models.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ModelRunInfo:
    name: str
    model_id: str


def run_whisper(audio_dir: Path, references: Path, output: Path, *, model_id: str = "turbo", backend: Any | None = None) -> ModelRunInfo: ...
```

`backend=None` lazy-imports `whisper`. A supplied fake backend must expose `load_model(model_id)` and returned model `.transcribe(path, language="en", task="transcribe", verbose=False)`.

Use packaged/shared versions of `_load_reference_ids`, `validate_wav_format`, and atomic JSONL writing; do not import from `tools`.

- [ ] **Step 1: Write RED tests proving adapter behavior and lazy dependency errors**

```python
def test_whisper_adapter_writes_model_a_predictions(tmp_path):
    references, audio_dir = _one_sample_dataset(tmp_path, "core-001")
    output = tmp_path / "predictions.jsonl"
    calls = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return {"text": " Synthetic transcript "}

    backend = types.SimpleNamespace(load_model=lambda name: (calls.setdefault("model", name) or FakeModel()))
    info = run_whisper(audio_dir, references, output, backend=backend)

    assert info == ModelRunInfo("whisper", "turbo")
    assert json.loads(output.read_text().splitlines()[0]) == {
        "id": "core-001",
        "text": " Synthetic transcript ",
    }
    assert calls["kwargs"]["language"] == "en"
```

Also prove an ID mismatch prevents `load_model`/transcription and preserves any previous output.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_models.py -k whisper -v
```

Commit:

```text
benchmark: test packaged Whisper adapter

Pin Model A behavior before moving transcription ownership into the package.

Test: python -m pytest tests/test_benchmark_models.py -k whisper -v
Bug: N/A
```

- [ ] **Step 3: Implement the adapter with atomic prediction writes**

Keep transcript whitespace exactly as the model returns it, matching existing `tools.transcribe_whisper` behavior.

If import fails, raise a controlled dependency error containing:

```text
Whisper is not installed. Run: python -m pip install -U openai-whisper
```

- [ ] **Step 4: Run GREEN and existing Model A tests**

```bash
python -m pytest tests/test_benchmark_models.py -k whisper -v
python -m pytest tests/test_transcribe_whisper.py -v
```

Commit:

```text
benchmark: package Whisper model adapter

Run Model A from the installed package while preserving current transcription
validation and output semantics.

Test: python -m pytest tests/test_benchmark_models.py -k whisper -v && python -m pytest tests/test_transcribe_whisper.py -v
Bug: N/A
```

- [ ] **Step 5: CI checkpoint**

Require all matrix jobs green.

---

### Task 6: Package the Whisper-AT Model B adapter

**Files:**
- Create: `deafbench/benchmark/models/whisper_at.py`
- Modify: `tests/test_benchmark_models.py`
- Reuse existing mapping behavior from: `tools/transcribe_whisper_at.py`

**Interfaces:**

```python
AUDIOSET_TO_DEAFBENCH: dict[str, str]

def extract_audio_tags(parsed: Any) -> tuple[list[str], list[str]]: ...
def run_whisper_at(audio_dir: Path, references: Path, output: Path, *, model_id: str = "medium.en", at_time_res: float = 10.0, top_k: int = 5, p_threshold: float = -1.0, backend: Any | None = None) -> ModelRunInfo: ...
```

- [ ] **Step 1: Write RED tests for Model B structured predictions**

Port the current exact broad/specific mapping assertions into packaged-adapter tests. Add a fake backend test that expects:

```json
{
  "id": "ns-001",
  "text": " Please remain seated. ",
  "sounds": ["[alarm]"],
  "audio_tags": ["Speech", "Alarm"]
}
```

Assert `[alarm]` is absent from the `text` field.

Also test invalid `at_time_res` (`0`, negative, or not a 0.4 multiple) is rejected before `backend.load_model` is called.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_models.py -k whisper_at -v
```

Commit:

```text
benchmark: test packaged Whisper-AT adapter

Preserve Model B audio-tag mapping and keep environmental labels outside ASR
text.

Test: python -m pytest tests/test_benchmark_models.py -k whisper_at -v
Bug: N/A
```

- [ ] **Step 3: Implement Model B adapter**

Keep constants exactly aligned with the current runner:

```python
DEFAULT_MODEL = "medium.en"
DEFAULT_AT_TIME_RES = 10.0
DEFAULT_TOP_K = 5
DEFAULT_P_THRESHOLD = -1.0
AUDIOSET_CLASS_COUNT = 527
```

Validate `at_time_res` using `math.isfinite` and an exact 0.4-step check before importing/loading the model.

If import fails, raise a controlled error containing:

```text
Whisper-AT is not installed. See the upstream Whisper-AT installation instructions.
```

- [ ] **Step 4: Run GREEN and legacy tests**

```bash
python -m pytest tests/test_benchmark_models.py -k whisper_at -v
python -m pytest tests/test_transcribe_whisper_at.py -v
```

Commit:

```text
benchmark: package Whisper-AT model adapter

Move Model B inference and AudioSet mapping behind the installed benchmark
interface without changing scoring inputs.

Test: python -m pytest tests/test_benchmark_models.py -k whisper_at -v && python -m pytest tests/test_transcribe_whisper_at.py -v
Bug: N/A
```

- [ ] **Step 5: CI checkpoint**

Require all matrix jobs green.

---

### Task 7: Orchestrate generation, inference, evaluation, and source-aware output

**Files:**
- Create: `deafbench/benchmark/runner.py`
- Create: `tests/test_benchmark_runner.py`
- Reuse: `deafbench/parser.py`, `deafbench/metrics.py`, `deafbench/report.py`, `deafbench/cli.py::format_terminal_output`

**Interfaces:**

```python
@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path
    dataset: str
    model: Literal["whisper", "whisper-at"]
    audio_source: AudioSource = "auto"
    scene_profile: str = "default-v1"
    seed: int = 42

@dataclass(frozen=True)
class BenchmarkResult:
    resolved_source: ResolvedAudioSource
    predictions: Path
    report: Path
    metadata: Path
    metrics: dict[str, Any]


def run_benchmark(config: BenchmarkConfig, *, synthetic_generator: Callable[..., Path] | None = None, model_runner: Callable[..., ModelRunInfo] | None = None) -> BenchmarkResult: ...
def main(argv: list[str] | None = None) -> int: ...
```

- [ ] **Step 1: Write RED orchestration tests with fake generation/model runners**

Cover these complete flows:

```python
def test_auto_uses_human_without_calling_tts_when_human_set_is_complete(...):
    ...


def test_auto_generates_complete_synthetic_set_when_one_human_wav_is_missing(...):
    ...


def test_benchmark_writes_source_aware_outputs_and_run_metadata(...):
    result = run_benchmark(...)
    assert result.predictions == dataset_dir / "runs" / "whisper" / "synthetic" / "predictions.jsonl"
    metadata = json.loads(result.metadata.read_text())
    assert metadata["dataset"] == "core-v1"
    assert metadata["model"] == "whisper"
    assert metadata["audio_source"] == "synthetic"
    assert metadata["scene_profile"] == "default-v1"
    assert metadata["seed"] == 42
```

The fake model runner must write predictions that align exactly with references. Assert real `evaluate_dataset` and `generate_markdown_report` are used by checking expected metric/report content.

Test unsupported model and explicit incomplete human source before generation/inference.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_runner.py -v
```

Commit:

```text
benchmark: test end-to-end runner orchestration

Define the automated source-selection through report-generation contract with
fake heavyweight runtimes.

Test: python -m pytest tests/test_benchmark_runner.py -v
Bug: N/A
```

- [ ] **Step 3: Implement runner and atomic run files**

Order operations exactly:

```text
validate references/dataset
inspect human audio
resolve source
if synthetic: validate fingerprint or generate whole set
validate selected audio set
select model adapter
run inference to predictions.jsonl atomically
parse + align + evaluate
render report
write report.md atomically
write run.json atomically last
print terminal summary
return success
```

`run.json` must include:

```json
{
  "dataset": "non-speech-v1",
  "model": "whisper-at",
  "model_id": "medium.en",
  "audio_source": "synthetic",
  "references": ".../references.jsonl",
  "audio": ".../audio-synthetic",
  "predictions": ".../runs/whisper-at/synthetic/predictions.jsonl",
  "report": ".../runs/whisper-at/synthetic/report.md",
  "samples": 12,
  "benchmark_version": "0.1.1",
  "scene_profile": "default-v1",
  "seed": 42,
  "tts": {"engine": "whisperspeech", "version": "..."}
}
```

For human runs omit `scene_profile`, `seed`, and `tts` instead of writing null synthetic metadata.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/test_benchmark_runner.py tests/test_benchmark_workspace.py tests/test_report.py -v
```

Commit:

```text
benchmark: orchestrate automated benchmark runs

Connect source resolution, synthetic preparation, model inference, evaluation,
and source-aware run artifacts behind one testable runner.

Test: python -m pytest tests/test_benchmark_runner.py tests/test_benchmark_workspace.py tests/test_report.py -v
Bug: N/A
```

- [ ] **Step 5: Verify prior valid outputs survive injected failures**

Add/confirm tests where fake inference raises before output promotion and report serialization raises after an existing valid run exists. Previous valid predictions/report/run metadata must remain readable.

---

### Task 8: Convert legacy transcription scripts to compatibility wrappers

**Files:**
- Modify: `tools/transcribe_whisper.py`
- Modify: `tools/transcribe_whisper_at.py`
- Modify: `tests/test_transcribe_whisper.py`
- Modify: `tests/test_transcribe_whisper_at.py`

**Interfaces:**
- Legacy command flags and default repo-root paths remain unchanged.
- Existing imports used by tests/users remain available through re-exports where practical: `resolve_dataset_paths`, `transcribe_directory`, `extract_audio_tags`.
- Legacy scripts still write top-level `model-a.jsonl` / `model-b.jsonl`; only `deafbench benchmark` writes `runs/` outputs.

- [ ] **Step 1: Add RED compatibility assertions before replacing implementations**

Add tests proving wrapper functions delegate to packaged helpers and direct-script invalid-dataset behavior remains exit code 2. Keep the current tests for sorted output, mismatch rejection, WAV validation, and Whisper-AT mapping.

- [ ] **Step 2: Run RED only for new delegation assertions and commit tests**

```bash
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
```

Commit:

```text
 tools: test packaged transcriber delegation

Protect existing repository commands while moving their runtime ownership into
the installed package.

Test: python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
Bug: N/A
```

Use commit subject without the accidental leading space when committing: `tools: test packaged transcriber delegation`.

- [ ] **Step 3: Replace duplicate runtime logic with thin wrappers**

`tools/transcribe_whisper.py` should keep its parser/defaults and delegate core inference to `deafbench.benchmark.models.whisper`.

`tools/transcribe_whisper_at.py` should keep parser/defaults and re-export `AUDIOSET_TO_DEAFBENCH`, `extract_audio_tags`, and validation helpers from the packaged adapter where existing consumers rely on them.

Do not make the installed package import `tools` in the opposite direction.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py tests/test_benchmark_models.py -v
```

Commit:

```text
 tools: delegate transcribers to packaged adapters

Keep repository transcription commands compatible while removing duplicate
model runtime ownership.

Test: python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py tests/test_benchmark_models.py -v
Bug: N/A
```

Again use commit subjects without leading spaces.

- [ ] **Step 5: CI checkpoint**

Require the full matrix to pass before documentation/finalization.

---

### Task 9: Add installed functional flow and dependency isolation smoke tests

**Files:**
- Modify: `tests/test_benchmark_cli.py`
- Modify: `tests/test_smoke.py`
- Modify: `.github/workflows/ci.yml` only if the existing generic pytest stages do not exercise the new installed command sufficiently.

**Interfaces:**
- No real WhisperSpeech/Whisper/Whisper-AT imports in CI.
- Functional test invokes `deafbench.cli.main` or `python -m deafbench` with fake runner seams.

- [ ] **Step 1: Write RED/coverage tests for installed CLI behavior**

Add a smoke assertion that `python -m deafbench benchmark --help` succeeds after `pip install -e ".[test]"` without model packages installed.

Add a functional test that monkeypatches runner dependency seams, executes:

```text
benchmark non-speech-v1 --model whisper-at --audio-source auto
```

and checks the printed terminal summary contains dataset, model, resolved source, WER line, predictions path, and report path.

- [ ] **Step 2: Run the focused tests**

```bash
python -m pytest tests/test_benchmark_cli.py tests/test_smoke.py tests/test_benchmark_runner.py -v
```

If tests already pass because prior tasks supplied all behavior, do not create a no-op RED commit; commit only genuinely new test coverage.

- [ ] **Step 3: Make only the minimum production adjustment required by failures**

Typical allowed adjustment: expose a dependency-injection seam or ensure parser construction does not import heavyweight modules. Do not change the approved workflow to satisfy a test artifact.

- [ ] **Step 4: Run full local-equivalent validation**

```bash
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m functional
python -m pytest -m smoke
python -m pytest --cov=deafbench --cov-report=term-missing
python -m deafbench benchmark --help
```

Expected: all commands exit 0; coverage remains at least 90%.

- [ ] **Step 5: Commit any real adjustment**

Use either a focused test-only commit or:

```text
benchmark: validate installed command workflow

Keep the benchmark CLI usable without importing heavyweight runtimes until a
real benchmark run selects them.

Test: python -m pytest && python -m deafbench benchmark --help
Bug: N/A
```

---

### Task 10: Document automation, installation, and source-aware outputs

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` only if final dependency instructions reveal a metadata correction.

**Interfaces:**
- Public examples must match the final parser exactly.

- [ ] **Step 1: Update README with the primary workflow**

Document:

```powershell
python -m pip install "deafbench[benchmark]"
deafbench benchmark core-v1 --model whisper
deafbench benchmark non-speech-v1 --model whisper-at
```

State clearly that the benchmark extra installs the synthetic-audio runtime, while Whisper and Whisper-AT still require their model-specific installation instructions. Preserve the existing working Whisper-AT Windows instructions unless the packaged adapter changes those requirements.

Document `auto` behavior in one compact block:

```text
complete audio/ set -> human run
incomplete audio/ set -> complete audio-synthetic/ run
never mix sources
```

Document result paths:

```text
benchmarks/<dataset>/runs/<model>/<audio-source>/predictions.jsonl
benchmarks/<dataset>/runs/<model>/<audio-source>/report.md
benchmarks/<dataset>/runs/<model>/<audio-source>/run.json
```

- [ ] **Step 2: Document synthetic traceability**

Explain `audio-synthetic/manifest.jsonl`, `default-v1`, seed `42`, generated background ambience, timed environmental events, and that WhisperSpeech supplies speech while DeafBench constructs the scene.

Do not claim byte-for-byte TTS reproducibility across WhisperSpeech/runtime/hardware versions.

- [ ] **Step 3: Run documentation-linked CLI checks**

```bash
python -m deafbench benchmark --help
python -m deafbench recorder --help
python -m deafbench compare --help
```

- [ ] **Step 4: Commit documentation**

```text
docs: document automated benchmark workflow

Explain source selection, WhisperSpeech synthetic generation, and traceable
source-aware run outputs.

Test: python -m deafbench benchmark --help
Bug: N/A
```

- [ ] **Step 5: Run the full CI matrix**

Do not mark PR #20 ready until Python 3.11-3.14 all pass on the final documentation head.

---

### Task 11: Manual runtime validation and PR review gate

**Files:**
- No required source edits unless validation discovers a concrete bug.
- Update PR #20 description with exact validation results and commit SHAs.

**Interfaces:**
- Real heavyweight validation is separate from the normal CI matrix.

- [ ] **Step 1: Verify the installed package without real model runtimes**

```powershell
python -m pip install -e ".[test]"
deafbench benchmark --help
deafbench recorder --help
```

- [ ] **Step 2: On a machine with WhisperSpeech installed, validate synthetic generation only**

Use a temporary/small test workspace or a single-purpose developer invocation that exercises `create_whisperspeech_generator()` and one reference. Confirm output can be read and converted to a valid 48 kHz mono WAV. Do not commit generated WAV/model weights.

- [ ] **Step 3: Validate one real model path when dependencies are available**

Run either Whisper or Whisper-AT against a small complete dataset first. Then run the intended dataset only after the smoke run succeeds. Record exact command and result in the PR, not generated model weights.

- [ ] **Step 4: Request automated PR reviews and fix valid findings**

Mark PR #20 ready for review only after the final CI matrix is green. Trigger configured reviewers. For each behavior-changing review finding, add a focused failing regression test first, confirm RED, then fix GREEN. Resolve only threads whose findings are actually addressed.

- [ ] **Step 5: Final verification before declaring ready**

Freshly verify:

```text
PR is open and mergeable
latest head SHA is known
CI latest run passes Python 3.11, 3.12, 3.13, 3.14
all actionable inline review threads are resolved
no requested-change review remains
```

Add `code-reviewed` only after those gates pass. Tell the user PR #20 is ready to merge, but do not merge without an explicit user request.

---

## Plan Self-Review Checklist

- Every approved spec requirement maps to a task: CLI (1), source separation (2), deterministic scenes (3), WhisperSpeech generation/manifest/cache (4), Whisper (5), Whisper-AT (6), orchestration/run metadata/reporting (7), legacy compatibility (8), installed dependency isolation/CI (9), docs (10), review/real integration gate (11).
- No task requires a real model download in CI.
- Source identities cannot overwrite one another because run paths include model and resolved source.
- Human and synthetic samples are never combined.
- Synthetic generation stages an entire set before promotion.
- Reference truth is never rewritten with generated timestamps.
- Whisper-AT sound labels remain separate from transcript text.
- Existing recorder behavior and metric semantics are unchanged.
- Model/TTS imports remain lazy.
- Heavyweight install guidance is explicit without making those dependencies mandatory for `compare`/`report` users.
- The plan contains no implementation `TBD`/`TODO` items and no instruction to invent missing behavior during execution.

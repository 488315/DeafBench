# Benchmark Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an installed `deafbench benchmark` command that automatically uses a complete human audio set or generates a complete synthetic set with WhisperSpeech, runs Whisper or Whisper-AT, evaluates the predictions, and writes source-aware predictions, report, and run metadata.

**Architecture:** Keep CLI parsing in `deafbench.cli`, orchestration in `deafbench.benchmark.runner`, dataset/source/path/file transactions in `deafbench.benchmark.workspace`, deterministic scene planning and mixing in `deafbench.benchmark.scenes`, synthetic-set generation in `deafbench.benchmark.synthetic`, and model-specific code under `deafbench.benchmark.models`. Heavy runtimes are imported only after the chosen command, source, and model require them. Existing `tools/transcribe_whisper.py` and `tools/transcribe_whisper_at.py` remain compatible repository entry points over the packaged model adapters.

**Tech Stack:** Python 3.11-3.14, argparse, pathlib, dataclasses, hashlib, json, tempfile, shutil, wave, NumPy, existing DeafBench parser/metrics/report modules, WhisperSpeech `whisperspeech.pipeline.Pipeline`, OpenAI Whisper, Whisper-AT, pytest.

## Global Constraints

- Public command: `deafbench benchmark <dataset> --model whisper|whisper-at`.
- `--audio-source` accepts exactly `auto`, `human`, or `synthetic`; default is `auto`.
- `auto` uses human audio only when every reference ID has exactly one valid WAV and there are no extra WAVs. Otherwise it uses one complete synthetic set.
- Never combine partial human audio with generated replacements in one benchmark run.
- Human audio stays under `benchmarks/<dataset>/audio/`.
- Synthetic audio stays under `benchmarks/<dataset>/audio-synthetic/` with `manifest.jsonl`.
- New benchmark results go under `benchmarks/<dataset>/runs/<model>/<human|synthetic>/`.
- Existing top-level `model-a.jsonl`, `model-b.jsonl`, and report workflows remain supported through repository compatibility tools.
- Whisper maps to `model-a` and defaults to `turbo`.
- Whisper-AT maps to `model-b` and defaults to `medium.en`, `at_time_res=10.0`, `top_k=5`, `p_threshold=-1.0`.
- Whisper-AT sound labels stay in structured `sounds`; never append them to ASR `text`.
- Synthetic scene profile is `default-v1`; default seed is `42`; final WAVs are 48 kHz, 16-bit PCM, mono.
- WhisperSpeech generates speech only. DeafBench owns resampling, generated ambience, event timing, mixing, final WAV validation, and manifest metadata.
- Generated timestamps never modify `references.jsonl`.
- Do not download or bundle third-party sound effects. Reuse `deafbench.recorder.core.synthesize_sound_event`.
- `compare`, `report`, `recorder`, and root help must not import Whisper, Whisper-AT, WhisperSpeech, torch, or torchaudio through the new benchmark package.
- CI must not download or execute real model weights.
- Each behavior change starts with a test-only RED commit, followed by the minimal GREEN production commit. Do not create no-op commits.
- After each GREEN checkpoint, require Python 3.11, 3.12, 3.13, and 3.14 CI to pass.
- Do not add Qwen2.5-Omni, the personal Deaf/cochlear speech track, metric changes, arbitrary shell-model plugins, or unrelated refactors in this PR.
- Upstream verification on 2026-08-09: WhisperSpeech `settings.ini` reports version `0.8.9`, and `whisperspeech.pipeline.Pipeline` exposes `generate_to_file(fname, text, speaker=None, lang='en', cps=15, step_callback=None)`. The adapter uses that file-producing API and reads the produced WAV instead of assuming an output sample rate.

---

## File Structure

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

`workspace.py` is stdlib-only. `scenes.py` imports NumPy and recorder sound-event helpers. `synthetic.py` receives a speech-generator callable in tests and imports WhisperSpeech only inside the real generator factory. Model adapters import their real runtime only inside the public run function when no fake backend is supplied.

---

### Task 1: Add the benchmark CLI contract and lazy dispatch

**Files:**
- Modify: `deafbench/cli.py`
- Create: `deafbench/benchmark/__init__.py`
- Create: `tests/test_benchmark_cli.py`

**Interfaces:**
- `deafbench.cli._run_benchmark(benchmark_args: list[str]) -> int`
- Positional `dataset`
- Required `--model` with choices `whisper`, `whisper-at`
- Optional `--audio-source` with choices `auto`, `human`, `synthetic`, default `auto`
- Optional `--repo-root`
- Optional `--scene-profile`, default `default-v1`
- Optional `--seed`, integer, default `42`

- [ ] **Step 1: Write the failing functional tests**

Create `tests/test_benchmark_cli.py`:

```python
import sys

import pytest

from deafbench import cli


pytestmark = pytest.mark.functional


def test_benchmark_command_forwards_defaults_to_lazy_launcher(monkeypatch):
    calls = []
    monkeypatch.setitem(cli.__dict__, "_run_benchmark", calls.append)

    cli.main(["benchmark", "core-v1", "--model", "whisper"])

    assert calls == [[
        "core-v1",
        "--model", "whisper",
        "--audio-source", "auto",
        "--scene-profile", "default-v1",
        "--seed", "42",
    ]]


def test_benchmark_command_returns_launcher_status(monkeypatch):
    monkeypatch.setitem(cli.__dict__, "_run_benchmark", lambda _args: 9)

    assert cli.main(["benchmark", "core-v1", "--model", "whisper"]) == 9


def test_compare_does_not_import_benchmark_runner(tmp_path):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    references.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    predictions.write_text('{"id":"s1","text":"hello"}\n', encoding="utf-8")
    sys.modules.pop("deafbench.benchmark.runner", None)

    cli.main(["compare", str(references), str(predictions)])

    assert "deafbench.benchmark.runner" not in sys.modules
```

- [ ] **Step 2: Run RED and commit only the test file**

```bash
python -m pytest tests/test_benchmark_cli.py -v
```

Expected: the benchmark-dispatch tests fail because argparse does not recognize `benchmark`.

Commit:

```text
cli: test benchmark command dispatch

Define the installed benchmark command contract before adding runtime code.

Test: python -m pytest tests/test_benchmark_cli.py -v
Bug: N/A
```

- [ ] **Step 3: Add the parser and lazy helper**

Add:

```python
def _run_benchmark(benchmark_args: list[str]) -> int:
    from .benchmark.runner import main as benchmark_main
    return benchmark_main(benchmark_args)
```

Forward arguments in this order:

```python
benchmark_args = [
    parsed.dataset,
    "--model", parsed.model,
    "--audio-source", parsed.audio_source,
    "--scene-profile", parsed.scene_profile,
    "--seed", str(parsed.seed),
]
if parsed.benchmark_repo_root is not None:
    benchmark_args.extend(["--repo-root", parsed.benchmark_repo_root])
return _run_benchmark(benchmark_args)
```

Use `dest="benchmark_repo_root"` for the benchmark root argument.

Create `deafbench/benchmark/__init__.py`:

```python
"""Installed DeafBench benchmark orchestration."""
```

- [ ] **Step 4: Run GREEN and existing CLI tests**

```bash
python -m pytest tests/test_benchmark_cli.py tests/test_cli.py -v
```

Expected: PASS.

Commit:

```text
cli: add benchmark command dispatch

Lazy-load benchmark orchestration so existing lightweight commands keep their
current dependency surface.

Test: python -m pytest tests/test_benchmark_cli.py tests/test_cli.py -v
Bug: N/A
```

- [ ] **Step 5: Require green Python 3.11-3.14 CI before Task 2**

---

### Task 2: Resolve paths, validate audio sets, and provide atomic file helpers

**Files:**
- Create: `deafbench/benchmark/workspace.py`
- Create: `tests/test_benchmark_workspace.py`

**Interfaces:**

```python
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
def atomic_write_text(path: Path, text: str) -> None: ...
def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None: ...
def atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None: ...
```

The ellipses above are Python signature stubs only. The behavior for every interface is specified in the implementation steps.

- [ ] **Step 1: Write RED tests with concrete WAV/reference helpers**

Create:

```python
import json
import wave

import pytest

from deafbench.benchmark.workspace import (
    AudioSetStatus,
    inspect_audio_set,
    resolve_audio_source,
    resolve_run_paths,
    validate_dataset_name,
)


def _write_wav(path, *, channels=1, width=2, rate=48_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * channels * width * 32)


def _write_references(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": sample_id, "text": sample_id}) + "\n" for sample_id in ids),
        encoding="utf-8",
    )
```

Add:

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


def test_inspect_audio_set_reports_missing_extra_and_invalid(tmp_path):
    references = tmp_path / "references.jsonl"
    audio = tmp_path / "audio"
    _write_references(references, ["s1", "s2", "s3"])
    _write_wav(audio / "s1.wav")
    _write_wav(audio / "s2.wav", channels=2)
    _write_wav(audio / "extra.wav")

    status = inspect_audio_set(references, audio)

    assert status.complete is False
    assert status.missing == ("s3",)
    assert status.extra == ("extra",)
    assert status.invalid == ("s2",)


def test_auto_prefers_only_a_complete_human_set():
    assert resolve_audio_source("auto", AudioSetStatus(True, (), (), ())) == "human"
    assert resolve_audio_source("auto", AudioSetStatus(False, ("s2",), (), ())) == "synthetic"


def test_explicit_human_rejects_incomplete_set():
    with pytest.raises(ValueError, match="Human audio set is incomplete"):
        resolve_audio_source("human", AudioSetStatus(False, ("s2",), (), ()))


@pytest.mark.parametrize("dataset", ["", ".", "..", "a/b", "a\\b", "C:temp"])
def test_dataset_name_rejects_unsafe_values(dataset):
    with pytest.raises(ValueError, match="Invalid dataset name"):
        validate_dataset_name(dataset)
```

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_workspace.py -v
```

Expected: import failure because `deafbench.benchmark.workspace` does not exist.

Commit:

```text
benchmark: test audio source resolution

Pin source-aware paths and all-or-nothing human audio selection.

Test: python -m pytest tests/test_benchmark_workspace.py -v
Bug: N/A
```

- [ ] **Step 3: Implement workspace and transaction helpers**

Dataset validation matches recorder safety rules: reject empty, `.`, `..`, or any name containing `/`, `\\`, or `:`.

`load_reference_ids` preserves JSONL order and raises on malformed JSON, a non-object record, missing/empty string ID, duplicate ID, or empty reference file.

`validate_wav_format` requires:

```python
channels == 1
sample_width == 2
sample_rate == 48_000
compression == "NONE"
```

`inspect_audio_set` compares `*.wav` stems to reference IDs and returns sorted `missing`, `extra`, and `invalid`. `complete` is true only when all three tuples are empty.

Use:

```python
def resolve_audio_source(requested, human_status):
    if requested == "synthetic":
        return "synthetic"
    if requested == "human":
        if not human_status.complete:
            raise ValueError("Human audio set is incomplete")
        return "human"
    if requested != "auto":
        raise ValueError(f"Unsupported audio source: {requested}")
    return "human" if human_status.complete else "synthetic"
```

Atomic text/JSON/JSONL writes create a sibling temporary file with `NamedTemporaryFile(delete=False, dir=destination.parent)`, close it, then call `os.replace(temp_path, destination)`. Delete an unpromoted temp file in `finally`.

- [ ] **Step 4: Run GREEN and existing transcriber tests**

```bash
python -m pytest tests/test_benchmark_workspace.py -v
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
```

Expected: PASS.

Commit:

```text
benchmark: resolve complete audio sources

Keep source selection strict and provide atomic file operations for later run
artifacts.

Test: python -m pytest tests/test_benchmark_workspace.py tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
Bug: N/A
```

- [ ] **Step 5: Require green Python 3.11-3.14 CI**

---

### Task 3: Add deterministic scene planning, resampling, ambience, and event mixing

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
def plan_scene(sample_id: str, speech_frames: int, sound_labels: Sequence[str], seed: int = 42, scene_profile: str = "default-v1", sample_rate: int = 48_000) -> ScenePlan: ...
def mix_scene(speech_pcm: np.ndarray, plan: ScenePlan) -> np.ndarray: ...
```

**Exact `default-v1` algorithm:**
- Speech starts at 500 ms.
- Speech end is `500 + round(1000 * speech_frames / 48000)` ms.
- Derive `event_seed` from the first 8 bytes of SHA-256 over `f"{seed}:events:{scene_profile}:{sample_id}"`, interpreted unsigned big-endian.
- Derive `background_seed` the same way with namespace `background`.
- No events: empty event tuple.
- One event: target center is 50% of speech duration.
- Multiple events: target centers are `np.linspace(0.25, 0.75, count)` across speech duration.
- Each target gets one deterministic jitter draw in `[-0.08, 0.08] * speech_duration`.
- Event duration comes from `synthesize_sound_event(label)` frame length at 48 kHz.
- Clamp event start to `[speech_start_ms, max(speech_start_ms, speech_end_ms - event_duration_ms)]`.
- Sort events by `(start_ms, label)`.
- Scene end is 500 ms after the later of speech end or last event end.
- Background profile is `office-v1`; SNR is `15.0` dB.
- Generate background with `default_rng(background_seed).normal(0.0, 1.0, scene_frames)`, smooth using a 64-sample averaging kernel with `np.convolve(..., mode="same")`, normalize RMS, then scale to `speech_rms / 10 ** (15.0 / 20.0)`. If speech RMS is zero, target noise RMS is `0.01`.
- Convert int16 event cues to float by dividing by `32768.0` before mixing.
- Mix speech, ambience, and events in float64. If absolute peak exceeds `0.98`, multiply the whole scene by `0.98 / peak`. Convert to int16 with `np.clip(np.rint(scene * 32767.0), -32768, 32767).astype(np.int16)` and return shape `(frames, 1)`.
- `resample_mono` averages channels when needed and uses `np.interp` on normalized source/target positions; it returns float64 mono shape `(frames, 1)`.

- [ ] **Step 1: Write RED tests**

```python
def test_scene_plan_is_reproducible_and_seed_sensitive():
    first = plan_scene("ns-008", 48_000 * 4, ["[phone rings]", "[knock]"], seed=42)
    same = plan_scene("ns-008", 48_000 * 4, ["[phone rings]", "[knock]"], seed=42)
    different = plan_scene("ns-008", 48_000 * 4, ["[phone rings]", "[knock]"], seed=43)

    assert first == same
    assert first.events != different.events
    assert first.speech_start_ms == 500
    assert {event.label for event in first.events} == {"[phone rings]", "[knock]"}


def test_resample_mono_doubles_24khz_frame_count():
    source = np.linspace(-0.25, 0.25, 24_000, dtype=np.float64).reshape(-1, 1)
    result = resample_mono(source, 24_000)
    assert result.shape == (48_000, 1)


def test_mix_scene_returns_int16_mono_with_planned_length():
    speech = np.full((48_000, 1), 0.1, dtype=np.float64)
    plan = plan_scene("ns-001", len(speech), ["[alarm]"], seed=42)
    mixed = mix_scene(speech, plan)

    assert mixed.dtype == np.int16
    assert mixed.shape == (plan.scene_end_ms * 48, 1)
```

Also test an unsupported scene profile raises `ValueError` before mixing.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_scenes.py -v
```

Commit:

```text
benchmark: test deterministic synthetic scenes

Define reproducible resampling, timing, ambience, and event mixing.

Test: python -m pytest tests/test_benchmark_scenes.py -v
Bug: N/A
```

- [ ] **Step 3: Implement `scenes.py` to the exact algorithm above**

Do not import WhisperSpeech or recognition models in this module.

- [ ] **Step 4: Run GREEN plus existing sound-event regressions**

```bash
python -m pytest tests/test_benchmark_scenes.py tests/test_sound_events.py tests/test_recorder_core_regressions.py -v
```

Commit:

```text
benchmark: build deterministic synthetic scenes

Generate DeafBench-owned background ambience and timestamped environmental
events around supplied speech.

Test: python -m pytest tests/test_benchmark_scenes.py tests/test_sound_events.py tests/test_recorder_core_regressions.py -v
Bug: N/A
```

- [ ] **Step 5: Require green Python 3.11-3.14 CI**

---

### Task 4: Generate and safely replace complete WhisperSpeech synthetic sets

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
def generation_fingerprint(references: Path, scene_profile: str, seed: int, tts_info: TTSInfo) -> str: ...
def synthetic_set_is_current(audio_dir: Path, references: Path, scene_profile: str, seed: int, tts_info: TTSInfo) -> bool: ...
def generate_synthetic_set(references: Path, audio_dir: Path, speech_generator: SpeechGenerator, tts_info: TTSInfo, scene_profile: str = "default-v1", seed: int = 42) -> Path: ...
```

**WhisperSpeech adapter behavior:**
- Import `Pipeline` only inside `create_whisperspeech_generator`.
- Construct one `Pipeline()` and reuse it for every generated sample.
- For each text, create a temporary `.wav`, call `pipeline.generate_to_file(str(path), text, lang="en")`, then lazy-import `soundfile` and call `soundfile.read(path, dtype="float32", always_2d=True)`.
- Return the actual sample rate from `soundfile.read`; scene code performs the 48 kHz conversion.
- Get package version with `importlib.metadata.version("WhisperSpeech")`; if metadata lookup alone fails after a successful import, record `unknown`.
- Missing WhisperSpeech raises exactly `RuntimeError('WhisperSpeech is not installed. Run: python -m pip install "deafbench[benchmark]"')`.

**Fingerprint input:** canonical JSON containing SHA-256 of raw `references.jsonl` bytes, scene profile, integer seed, TTS engine, and TTS version. Serialize with `json.dumps(value, sort_keys=True, separators=(",", ":"))`, encode UTF-8, then SHA-256 that byte string.

- [ ] **Step 1: Write RED generation, cache, and transaction tests**

Use:

```python
def _write_references(path):
    records = [
        {"id": "ns-001", "text": "Stay seated.", "sounds": ["[alarm]"]},
        {"id": "ns-002", "text": "Wait outside.", "sounds": []},
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def _fake_speech(text):
    frames = 24_000 + len(text) * 100
    return SpeechAudio(np.full((frames, 1), 0.1, dtype=np.float64), 24_000)
```

Tests:

```python
def test_generate_synthetic_set_writes_complete_wavs_and_timestamp_manifest(tmp_path):
    references = _write_references(tmp_path / "references.jsonl")
    audio_dir = tmp_path / "audio-synthetic"

    manifest = generate_synthetic_set(
        references,
        audio_dir,
        _fake_speech,
        TTSInfo("whisperspeech", "test"),
        seed=42,
    )

    assert manifest == audio_dir / "manifest.jsonl"
    assert {path.name for path in audio_dir.glob("*.wav")} == {"ns-001.wav", "ns-002.wav"}
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == ["ns-001", "ns-002"]
    assert records[0]["speech"]["start_ms"] == 500
    assert records[0]["events"][0]["label"] == "[alarm]"
    assert records[0]["sample_rate"] == 48_000


def test_failed_regeneration_preserves_previous_complete_set(tmp_path):
    references = _write_references(tmp_path / "references.jsonl")
    audio_dir = tmp_path / "audio-synthetic"
    info = TTSInfo("whisperspeech", "test")
    generate_synthetic_set(references, audio_dir, _fake_speech, info, seed=42)
    before = {path.name: path.read_bytes() for path in audio_dir.iterdir() if path.is_file()}
    calls = 0

    def failing_speech(text):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("tts failed")
        return _fake_speech(text)

    with pytest.raises(RuntimeError, match="tts failed"):
        generate_synthetic_set(references, audio_dir, failing_speech, info, seed=43)

    after = {path.name: path.read_bytes() for path in audio_dir.iterdir() if path.is_file()}
    assert after == before
```

Also test `synthetic_set_is_current` returns false for a missing WAV, missing manifest, changed seed, changed reference bytes, or changed TTS version, and true for an untouched matching set.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_synthetic.py -v
```

Commit:

```text
benchmark: test synthetic set generation

Require complete timestamped synthetic sets and preservation of the last valid
set when regeneration fails.

Test: python -m pytest tests/test_benchmark_synthetic.py -v
Bug: N/A
```

- [ ] **Step 3: Implement generation and directory promotion**

Generate in a sibling staging directory from `tempfile.mkdtemp(prefix=".audio-synthetic-", dir=audio_dir.parent)`.

For every reference record:
1. Call the supplied speech generator.
2. Resample with `resample_mono`.
3. Build a `ScenePlan` from record ID and `sounds`.
4. Mix with `mix_scene`.
5. Write `<id>.wav` inside staging with `atomic_write_wav`.
6. Append a manifest record containing ID, set fingerprint, scene profile, seed, sample rate, TTS engine/version, speech timestamps, background metadata, and event timestamps.

Write `manifest.jsonl` last inside staging. Validate staging with `inspect_audio_set`. Promote using:

```python
backup = audio_dir.with_name(f".{audio_dir.name}-backup")
if backup.exists():
    shutil.rmtree(backup)
if audio_dir.exists():
    os.replace(audio_dir, backup)
try:
    os.replace(staging, audio_dir)
except Exception:
    if backup.exists() and not audio_dir.exists():
        os.replace(backup, audio_dir)
    raise
else:
    if backup.exists():
        shutil.rmtree(backup)
```

Clean an unpromoted staging directory in `finally`.

Add to `[project.optional-dependencies]`:

```toml
benchmark = [
    "numpy>=1.26",
    "WhisperSpeech>=0.8.9",
]
```

Do not add Whisper or Whisper-AT to base dependencies.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/test_benchmark_synthetic.py tests/test_benchmark_scenes.py tests/test_recorder.py -v
```

Commit:

```text
benchmark: generate WhisperSpeech audio sets

Create complete source-isolated synthetic audio with reproducible scene
metadata and safe full-set regeneration.

Test: python -m pytest tests/test_benchmark_synthetic.py tests/test_benchmark_scenes.py tests/test_recorder.py -v
Bug: N/A
```

- [ ] **Step 5: Require green Python 3.11-3.14 CI**

CI installs `[test]`, not `[benchmark]`; it must continue without WhisperSpeech or real weights.

---

### Task 5: Package Whisper and Whisper-AT adapters

**Files:**
- Create: `deafbench/benchmark/models/__init__.py`
- Create: `deafbench/benchmark/models/whisper.py`
- Create: `deafbench/benchmark/models/whisper_at.py`
- Create: `tests/test_benchmark_models.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ModelRunInfo:
    name: str
    model_id: str


def run_whisper(audio_dir: Path, references: Path, output: Path, model_id: str = "turbo", backend: Any | None = None) -> ModelRunInfo: ...
def extract_audio_tags(parsed: Any) -> tuple[list[str], list[str]]: ...
def run_whisper_at(audio_dir: Path, references: Path, output: Path, model_id: str = "medium.en", at_time_res: float = 10.0, top_k: int = 5, p_threshold: float = -1.0, backend: Any | None = None) -> ModelRunInfo: ...
```

- [ ] **Step 1: Write RED Model A tests**

```python
def test_whisper_adapter_writes_model_a_prediction(tmp_path):
    references, audio_dir = _one_sample_dataset(tmp_path, "core-001")
    output = tmp_path / "predictions.jsonl"
    calls = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return {"text": " Synthetic transcript "}

    class FakeBackend:
        def load_model(self, name):
            calls["model"] = name
            return FakeModel()

    info = run_whisper(audio_dir, references, output, backend=FakeBackend())

    assert info == ModelRunInfo("whisper", "turbo")
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0]) == {
        "id": "core-001",
        "text": " Synthetic transcript ",
    }
    assert calls["model"] == "turbo"
    assert calls["kwargs"] == {
        "language": "en",
        "task": "transcribe",
        "verbose": False,
    }
```

`_one_sample_dataset` writes one matching JSONL reference and one valid 48 kHz mono WAV. Add an ID-mismatch test where backend `load_model` raises `AssertionError` if called; validation must fail first.

- [ ] **Step 2: Write RED Model B tests in the same test-only commit**

Use the existing exact AudioSet mapping. Broad labels `Door`, `Sliding door`, and `Telephone` remain raw-only. Specific labels continue mapping as in the current tool.

```python
def test_whisper_at_adapter_keeps_sounds_out_of_text(tmp_path):
    references, audio_dir = _one_sample_dataset(tmp_path, "ns-001", sounds=["[alarm]"])
    output = tmp_path / "predictions.jsonl"

    class FakeModel:
        def transcribe(self, path, **kwargs):
            return {"text": " Please remain seated. "}

    class FakeBackend:
        def load_model(self, name):
            assert name == "medium.en"
            return FakeModel()

        def parse_at_label(self, result, **kwargs):
            assert result["text"] == " Please remain seated. "
            return [{"audio tags": [("Speech", 2.0), ("Alarm", 1.5)]}]

    info = run_whisper_at(audio_dir, references, output, backend=FakeBackend())
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert info == ModelRunInfo("whisper-at", "medium.en")
    assert record["text"] == " Please remain seated. "
    assert record["sounds"] == ["[alarm]"]
    assert record["audio_tags"] == ["Speech", "Alarm"]
    assert "[alarm]" not in record["text"]
```

Also test `at_time_res` values `0`, `-0.4`, and `0.5` fail before `load_model`.

- [ ] **Step 3: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_models.py -v
```

Commit:

```text
benchmark: test packaged model adapters

Pin Model A transcript behavior and Model B structured audio-tag behavior before
moving runtime ownership into the package.

Test: python -m pytest tests/test_benchmark_models.py -v
Bug: N/A
```

- [ ] **Step 4: Implement both adapters**

Both adapters call `inspect_audio_set` first and require `complete is True`; they sort WAVs by filename and collect all records before `atomic_write_jsonl` promotes output.

Whisper real backend:

```python
if backend is None:
    try:
        import whisper as backend
    except ImportError as exc:
        raise RuntimeError(
            "Whisper is not installed. Run: python -m pip install -U openai-whisper"
        ) from exc
model = backend.load_model(model_id)
```

Whisper-AT constants remain:

```python
DEFAULT_MODEL = "medium.en"
DEFAULT_AT_TIME_RES = 10.0
DEFAULT_TOP_K = 5
DEFAULT_P_THRESHOLD = -1.0
AUDIOSET_CLASS_COUNT = 527
```

Validate `at_time_res` with `math.isfinite`, positivity, and `math.isclose(value / 0.4, round(value / 0.4), rel_tol=0.0, abs_tol=1e-9)` before backend import/loading.

Whisper-AT missing dependency message:

```text
Whisper-AT is not installed. See the upstream Whisper-AT installation instructions.
```

- [ ] **Step 5: Run GREEN and legacy tests**

```bash
python -m pytest tests/test_benchmark_models.py -v
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
```

Commit:

```text
benchmark: package model adapters

Run Whisper and Whisper-AT from installed DeafBench while preserving current
validation and structured prediction semantics.

Test: python -m pytest tests/test_benchmark_models.py tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
Bug: N/A
```

Require green Python 3.11-3.14 CI.

---

### Task 6: Orchestrate source selection, synthetic preparation, inference, evaluation, and transactional run output

**Files:**
- Create: `deafbench/benchmark/runner.py`
- Create: `tests/test_benchmark_runner.py`
- Reuse: `deafbench/recorder/workspace.py`, `deafbench/parser.py`, `deafbench/metrics.py`, `deafbench/report.py`, `deafbench/__init__.py`

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


def run_benchmark(
    config: BenchmarkConfig,
    synthetic_factory: Callable[[], tuple[SpeechGenerator, TTSInfo]] | None = None,
    synthetic_generator: Callable[..., Path] | None = None,
    whisper_runner: Callable[..., ModelRunInfo] | None = None,
    whisper_at_runner: Callable[..., ModelRunInfo] | None = None,
) -> BenchmarkResult: ...

def main(argv: list[str] | None = None) -> int: ...
```

**Runtime order:**
1. Use `ensure_dataset_workspace(repo_root, dataset)` so installed Core v1 and Non-speech v1 references are seeded exactly like recorder behavior without overwriting existing references.
2. Resolve run paths.
3. Inspect human audio and resolve source.
4. Human: use human audio after complete validation.
5. Synthetic: inspect whether the existing synthetic set is reusable. If it is stale or incomplete, lazy-create the WhisperSpeech generator and rebuild the full set through `synthetic_generator`.
6. Re-inspect selected audio and require a complete set before inference.
7. Create a sibling staging run directory.
8. Invoke selected model adapter with staging `predictions.jsonl`.
9. Parse references/predictions, align, evaluate, and generate the Markdown report into staging.
10. Write `run.json` into staging.
11. Promote the whole staging run directory with a backup/swap transaction so a failed rerun cannot pair new predictions with an old report or metadata file.
12. Print dataset, model, resolved source, existing terminal metric summary, predictions path, and report path.

- [ ] **Step 1: Write RED runner tests using fake heavyweight seams**

Human-selection test:

```python
def test_auto_uses_complete_human_set_without_synthetic_factory(tmp_path):
    _write_complete_dataset(tmp_path, human_complete=True)
    synthetic_called = False

    def fail_synthetic_factory():
        nonlocal synthetic_called
        synthetic_called = True
        raise AssertionError("synthetic factory must not be used")

    result = run_benchmark(
        BenchmarkConfig(tmp_path, "core-v1", "whisper"),
        synthetic_factory=fail_synthetic_factory,
        whisper_runner=_fake_whisper_runner,
    )

    assert result.resolved_source == "human"
    assert synthetic_called is False
```

Synthetic-selection test:

```python
def test_auto_generates_synthetic_set_when_human_is_missing(tmp_path):
    _write_complete_dataset(tmp_path, human_complete=False)
    generated = []

    def fake_factory():
        return _fake_speech, TTSInfo("whisperspeech", "test")

    def recording_generator(
        references,
        audio_dir,
        speech_generator,
        tts_info,
        scene_profile="default-v1",
        seed=42,
    ):
        generated.append((scene_profile, seed))
        return generate_synthetic_set(
            references,
            audio_dir,
            speech_generator,
            tts_info,
            scene_profile=scene_profile,
            seed=seed,
        )

    result = run_benchmark(
        BenchmarkConfig(tmp_path, "core-v1", "whisper", "auto", "default-v1", 42),
        synthetic_factory=fake_factory,
        synthetic_generator=recording_generator,
        whisper_runner=_fake_whisper_runner,
    )

    assert result.resolved_source == "synthetic"
    assert generated == [("default-v1", 42)]
```

Run-output test must assert:

```python
run_root = tmp_path / "benchmarks" / "core-v1" / "runs" / "whisper" / "synthetic"
assert result.predictions == run_root / "predictions.jsonl"
assert result.report == run_root / "report.md"
assert result.metadata == run_root / "run.json"
metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
assert metadata["dataset"] == "core-v1"
assert metadata["model"] == "whisper"
assert metadata["audio_source"] == "synthetic"
assert metadata["scene_profile"] == "default-v1"
assert metadata["seed"] == 42
assert metadata["benchmark_version"] == "0.1.1"
```

Add a failed-rerun test: create a valid run directory, then invoke with a model runner that raises `RuntimeError("inference failed")`; assert every file byte in the previous run directory is unchanged.

- [ ] **Step 2: Run RED and commit tests only**

```bash
python -m pytest tests/test_benchmark_runner.py -v
```

Commit:

```text
benchmark: test end-to-end runner orchestration

Define automated source selection through transactional report generation with
fake heavyweight runtimes.

Test: python -m pytest tests/test_benchmark_runner.py -v
Bug: N/A
```

- [ ] **Step 3: Implement runner**

Synthetic `run.json` contains actual values for these fields:

```json
{
  "dataset": "core-v1",
  "model": "whisper",
  "model_id": "turbo",
  "audio_source": "synthetic",
  "references": "resolved reference path",
  "audio": "resolved audio path",
  "predictions": "final predictions path",
  "report": "final report path",
  "samples": 25,
  "benchmark_version": "0.1.1",
  "scene_profile": "default-v1",
  "seed": 42,
  "tts": {"engine": "whisperspeech", "version": "runtime version"}
}
```

The example values above show field shape; code writes the actual dataset/model/model ID/path/sample count/TTS version. Human metadata omits `scene_profile`, `seed`, and `tts` entirely.

Runner parser defaults `--repo-root` to `Path.cwd()`. It validates model and source choices before any heavyweight import.

- [ ] **Step 4: Run GREEN plus evaluator/report tests**

```bash
python -m pytest tests/test_benchmark_runner.py tests/test_benchmark_workspace.py tests/test_report.py tests/test_metrics.py -v
```

Commit:

```text
benchmark: orchestrate automated benchmark runs

Connect strict source resolution, synthetic preparation, model inference,
evaluation, and transactional source-aware run artifacts.

Test: python -m pytest tests/test_benchmark_runner.py tests/test_benchmark_workspace.py tests/test_report.py tests/test_metrics.py -v
Bug: N/A
```

- [ ] **Step 5: Require green Python 3.11-3.14 CI**

---

### Task 7: Convert repository transcribers to thin compatibility wrappers

**Files:**
- Modify: `tools/transcribe_whisper.py`
- Modify: `tools/transcribe_whisper_at.py`
- Modify: `tests/test_transcribe_whisper.py`
- Modify: `tests/test_transcribe_whisper_at.py`

**Compatibility contract:**
- Existing parser flags and repository-root defaults remain.
- Whisper tool still resolves to top-level `model-a.jsonl`.
- Whisper-AT tool still resolves to top-level `model-b.jsonl`.
- Existing callable `transcribe_directory` behavior remains available.
- Existing `extract_audio_tags` import remains available from the Whisper-AT tool.
- Tools may import packaged adapters; the installed package never imports `tools`.

- [ ] **Step 1: Add test-only delegation assertions**

For Whisper, monkeypatch packaged `run_whisper`, invoke tool `main` with a valid one-sample dataset, and assert it forwards resolved references/audio/output plus model ID `turbo`.

For Whisper-AT, monkeypatch packaged `run_whisper_at` and assert it forwards `medium.en`, `10.0`, `5`, and `-1.0`.

Keep all current tests for sorted predictions, mismatch rejection, invalid WAV format, direct-script execution, invalid dataset parsing, and AudioSet mappings.

- [ ] **Step 2: Run and commit only new failing delegation coverage**

```bash
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
```

Commit:

```text
tools: test packaged transcriber delegation

Protect repository commands while moving model runtime ownership into the
installed package.

Test: python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py -v
Bug: N/A
```

- [ ] **Step 3: Replace duplicate model runtime logic with wrappers**

Keep local path parser helpers where they preserve old top-level output names. Re-export packaged mapping/helper functions where behavior is identical. Tool `main` functions delegate model execution to packaged adapters.

If the existing `transcribe_directory(audio_dir, output, transcribe, references=None)` injection API needs a small compatibility function to stay stable, keep and test that compatibility function locally. Do not expose repository-root assumptions from the installed package.

- [ ] **Step 4: Run GREEN**

```bash
python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py tests/test_benchmark_models.py -v
```

Commit:

```text
tools: delegate transcribers to packaged adapters

Keep existing repository transcription commands compatible while removing
duplicate model runtime ownership.

Test: python -m pytest tests/test_transcribe_whisper.py tests/test_transcribe_whisper_at.py tests/test_benchmark_models.py -v
Bug: N/A
```

- [ ] **Step 5: Require green Python 3.11-3.14 CI**

---

### Task 8: Installed workflow tests, documentation, manual runtime validation, and review gate

**Files:**
- Modify: `tests/test_benchmark_cli.py`
- Modify: `tests/test_smoke.py`
- Modify: `README.md`
- Modify: `pyproject.toml` only if validation finds a concrete metadata error

- [ ] **Step 1: Add dependency-isolation and terminal-summary coverage**

Add:

```python
def test_benchmark_help_does_not_require_heavy_dependencies():
    result = subprocess.run(
        [sys.executable, "-m", "deafbench", "benchmark", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--audio-source" in result.stdout
```

Add a runner/CLI output test with fake seams and assert output contains:

```text
Dataset: non-speech-v1
Model: whisper-at
Audio source: synthetic
WER
Critical Information
Predictions:
Report:
```

If these tests already pass with no production adjustment, commit test coverage only. Do not manufacture a RED failure or no-op production commit.

- [ ] **Step 2: Run full local-equivalent validation**

```bash
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m functional
python -m pytest -m smoke
python -m pytest --cov=deafbench --cov-report=term-missing
python -m deafbench benchmark --help
python -m deafbench recorder --help
python -m deafbench compare --help
```

Coverage must remain at least 90%.

- [ ] **Step 3: Document the public workflow**

README examples:

```powershell
python -m pip install "deafbench[benchmark]"
deafbench benchmark core-v1 --model whisper
deafbench benchmark non-speech-v1 --model whisper-at
```

State that `[benchmark]` installs the default synthetic-audio runtime, while Whisper and Whisper-AT still require their model-specific installation instructions. Preserve current tested Whisper-AT Windows install guidance.

Document source policy:

```text
complete audio/ set -> human run
incomplete audio/ set -> complete audio-synthetic/ run
never mix sources
```

Document outputs:

```text
benchmarks/<dataset>/audio-synthetic/manifest.jsonl
benchmarks/<dataset>/runs/<model>/<audio-source>/predictions.jsonl
benchmarks/<dataset>/runs/<model>/<audio-source>/report.md
benchmarks/<dataset>/runs/<model>/<audio-source>/run.json
```

Explain that WhisperSpeech supplies speech; DeafBench supplies ambience, event timing, and mixing; `default-v1` uses seed 42 by default; and DeafBench does not promise byte-identical TTS across runtime or hardware versions.

Commit:

```text
docs: document automated benchmark workflow

Explain automatic source selection, WhisperSpeech synthetic scenes, and
traceable source-aware run artifacts.

Test: python -m deafbench benchmark --help
Bug: N/A
```

- [ ] **Step 4: Perform real runtime smoke validation outside normal CI when dependencies are available**

On a developer machine with `[benchmark]` installed, create the WhisperSpeech generator, generate one short reference, and verify the promoted WAV passes `validate_wav_format`. Do not commit generated WAVs, downloaded weights, caches, or temporary manifests.

Then validate one small real Whisper or Whisper-AT run before a full benchmark dataset. Record the exact command and result in PR #20.

- [ ] **Step 5: Final PR review gate**

Mark PR #20 ready only after final Python 3.11-3.14 CI is green. Trigger configured automated reviewers. For every valid behavior-changing finding, add a focused regression test, confirm RED, apply the minimum GREEN fix, rerun CI, and resolve the thread. Do not merge without explicit user instruction.

Before telling the user it is ready, freshly verify:

```text
PR #20 is open and mergeable
latest head SHA is known
latest CI passes Python 3.11, 3.12, 3.13, 3.14
no actionable inline review thread is unresolved
no request-changes review remains
```

Add `code-reviewed` only after those gates pass.

---

## Plan Self-Review

- Spec coverage: Task 1 CLI; Task 2 source separation/path safety/atomic files; Task 3 deterministic scenes; Task 4 WhisperSpeech generation/manifest/fingerprints/transactional set replacement; Task 5 Whisper and Whisper-AT adapters; Task 6 orchestration/evaluation/source-aware transactional output; Task 7 legacy compatibility; Task 8 installed smoke/docs/manual integration/review.
- No CI task imports or downloads real model weights.
- Human and synthetic samples cannot mix because source resolution selects one complete directory.
- Human and synthetic results cannot overwrite each other because both model and resolved source are part of the run path.
- Failed synthetic regeneration preserves the previous complete synthetic directory.
- Failed benchmark reruns preserve the previous complete run bundle.
- Generated timing lives only in synthetic manifest data, not reference truth.
- Whisper-AT sounds remain separate from transcript text, preserving WER semantics.
- Existing recorder and metric semantics are unchanged.
- Heavy runtime imports remain lazy.
- Test snippets and runner dependency-injection seams are internally consistent and executable as written.
- Placeholder scan found no unresolved implementation markers.

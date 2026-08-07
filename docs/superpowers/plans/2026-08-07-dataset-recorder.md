# Dataset Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows Tkinter recorder for DeafBench benchmark prompts that captures Voicemeeter Out B3 audio and writes standardized 48 kHz 16-bit mono WAV files.

**Architecture:** Keep dataset/audio logic in `tools/recorder/core.py` so it is testable without Tkinter or an audio device. Keep the desktop UI and `sounddevice` stream handling in `tools/recorder/recorder.py`. Tests exercise prompt loading, navigation, device selection, downmixing, and atomic WAV replacement without requiring Voicemeeter.

**Tech Stack:** Python 3.11+, Tkinter, sounddevice, NumPy, wave, pytest.

## Global Constraints

- WAV, 48,000 Hz, signed 16-bit PCM, mono.
- Prefer an input device whose name contains `Voicemeeter Out B3`, case-insensitively.
- Never silently fall back to an unrelated microphone.
- Load prompts from `benchmarks/core-v1/references.jsonl`.
- Save audio to `benchmarks/core-v1/audio/<sample-id>.wav`.
- Stop saves and advances automatically.
- Retry may replace any selected earlier sample while preserving the old WAV until the replacement is fully written.
- Do not add waveform editing, trimming, denoising, normalization, ASR, cloud upload, or installer packaging.

---

### Task 1: Testable recorder core

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/recorder/__init__.py`
- Create: `tools/recorder/core.py`
- Create: `tests/test_recorder.py`

**Interfaces:**
- `load_prompts(path: Path) -> list[dict]`
- `output_path(audio_dir: Path, sample_id: str) -> Path`
- `is_recorded(audio_dir: Path, sample_id: str) -> bool`
- `next_unrecorded_index(prompts, audio_dir, current_index) -> int | None`
- `find_preferred_input_device(devices, needle='Voicemeeter Out B3') -> int | None`
- `downmix_to_mono(samples) -> numpy.ndarray`
- `atomic_write_wav(path, samples, sample_rate=48000) -> None`

- [ ] Write failing tests for prompt validation, duplicate IDs, filenames, recorded state, next-unrecorded selection, B3 selection, stereo downmix, int16 clipping, and atomic replacement.
- [ ] Run CI and confirm recorder tests fail because core implementation is missing.
- [ ] Implement minimal core functions.
- [ ] Run CI and confirm tests pass.

### Task 2: Desktop recorder GUI

**Files:**
- Create: `tools/recorder/recorder.py`
- Create: `tools/recorder/requirements.txt`
- Modify: `pyproject.toml`

**Interfaces:**
- `RecorderApp(root, references_path, audio_dir)` owns Tkinter state.
- `AudioRecorder` wraps `sounddevice.InputStream`, records int16 blocks, exposes duration/input level, and returns captured frames on stop.

- [ ] Add recorder dependencies and GUI implementation using the tested core helpers.
- [ ] Validate import/CLI smoke behavior without requiring an audio device.
- [ ] Ensure Record/Stop/Retry/Previous/Next state transitions are explicit and Stop auto-advances.
- [ ] Ensure device dropdown lists only input-capable devices and B3 is selected when present.

### Task 3: Final verification and cleanup

**Files:**
- Delete: `docs/superpowers/specs/2026-08-07-dataset-recorder-design.md` if present on the working branch.
- Delete: `docs/superpowers/plans/2026-08-07-dataset-recorder.md`

- [ ] Run the full CI matrix.
- [ ] Verify changed files contain only recorder implementation/tests/dependency metadata.
- [ ] Remove temporary planning artifacts.
- [ ] Open a focused feature PR with launch instructions and manual Windows validation checklist.

# Benchmark runner design

## Goal

Add an installed `deafbench benchmark` workflow that can prepare a complete
audio set, run a selected captioning model, evaluate the predictions, generate a
Markdown report, and print the final metrics without requiring a DeafBench source
checkout.

The default workflow is automated. When a complete human recording set is not
available, DeafBench generates a complete synthetic set with WhisperSpeech and a
DeafBench-owned scene builder instead of mixing human and synthetic samples in
one run.

## User interface

Primary commands:

```powershell
deafbench benchmark core-v1 --model whisper
deafbench benchmark non-speech-v1 --model whisper-at
```

Audio source defaults to `auto`:

```powershell
deafbench benchmark core-v1 --model whisper --audio-source auto
deafbench benchmark core-v1 --model whisper --audio-source human
deafbench benchmark core-v1 --model whisper --audio-source synthetic
```

`auto` uses the complete human set only when every required WAV is present and
valid. Otherwise it generates or reuses a complete synthetic set. It never fills
individual missing human recordings with synthetic files.

## High-level flow

```text
validate dataset/workspace
        |
select audio source
        |
        +-- complete human set --------------------------+
        |                                                |
        +-- incomplete human set + auto                  |
             |                                           |
             v                                           |
      generate/reuse synthetic set                       |
      with WhisperSpeech + scene builder                 |
             |                                           |
             +-------------------------------------------+
                             |
                     model adapter
                     |          |
                  whisper   whisper-at
                     |          |
                     +-----+----+
                           |
                    predictions JSONL
                           |
                    DeafBench metrics
                           |
                    Markdown report
                           |
                    terminal summary
```

## Package architecture

The installed package owns the workflow. It must not subprocess repository-only
scripts from `tools/`.

Proposed package responsibilities:

```text
deafbench/
  benchmark/
    __init__.py
    runner.py          # orchestration and run metadata
    workspace.py       # dataset/audio/run path resolution and validation
    synthetic.py       # synthetic set generation and scene construction
    scenes.py          # deterministic timing/noise/event scene logic
    models/
      __init__.py
      whisper.py       # Model A adapter
      whisper_at.py    # Model B adapter
```

Keep `deafbench.cli` focused on argument parsing and dispatch. Existing
`tools/transcribe_whisper.py` and `tools/transcribe_whisper_at.py` become thin
compatibility entry points over the packaged model adapters after equivalent
behavior is covered by tests.

Do not perform unrelated refactors of evaluation, reporting, or recorder code.

## Model identities

The first benchmark runner supports exactly two model names:

```text
whisper     -> OpenAI Whisper turbo -> model-a identity
whisper-at  -> Whisper-AT medium.en -> model-b identity
```

Model-specific loading, transcription, audio tagging, and dependency errors stay
inside their adapters. The orchestrator consumes a common adapter result instead
of knowing model internals.

Whisper-AT keeps the existing explicit AudioSet-to-DeafBench sound-label mapping.
Sound labels are structured prediction data and must not be inserted into speech
`text`, so environmental-sound scoring cannot change WER.

## Audio-source selection

### Human

Human recordings live at:

```text
benchmarks/<dataset>/audio/
```

A human set is complete only when its WAV IDs exactly match the reference IDs
and every WAV satisfies DeafBench's expected format validation.

`--audio-source human` fails clearly when the set is incomplete. It does not
open the recorder automatically in the first benchmark-runner release. Manual
recording remains available through `deafbench recorder`.

### Synthetic

Synthetic recordings live separately at:

```text
benchmarks/<dataset>/audio-synthetic/
```

WhisperSpeech generates the spoken reference text. DeafBench owns all scene
construction around that speech, including background ambience, event placement,
timing, mixing, sample-rate conversion, and final WAV validation.

`--audio-source synthetic` generates or reuses a complete synthetic set.

### Auto

`--audio-source auto` is the default:

1. Validate the human set without modifying it.
2. If the human set is complete, use it.
3. Otherwise generate or reuse a complete synthetic set.
4. Never combine human and synthetic WAVs in a single benchmark run.

This makes the one-command workflow automated while keeping human results and
synthetic results scientifically distinguishable.

## Synthetic scene generation

WhisperSpeech is responsible only for TTS speech generation. DeafBench's scene
builder is responsible for the benchmark audio scene.

For every synthetic sample the scene builder produces a 48 kHz, 16-bit PCM,
mono WAV and a structured manifest record.

The default `default-v1` scene profile supports:

- generated reference speech
- deterministic speech placement
- deterministic background ambience placement
- deterministic environmental-event placement
- event overlap with speech when the profile schedules it
- fixed scene seed, default `42`
- explicit background-noise level metadata
- exact millisecond start/end times for speech, ambience, and events

The existing supported DeafBench events remain:

```text
[alarm]
[door closes]
[phone rings]
[knock]
[error notification]
[siren]
```

The first release should reuse DeafBench's deterministic generated event cues.
It should not download or bundle third-party sound-effect libraries.

Background ambience must be generated or synthesized by DeafBench for v1 so the
benchmark does not add an external audio-asset dependency. The profile and level
must be recorded in the manifest.

## Synthetic manifest

Generated timing does not modify `references.jsonl`. Reference files remain the
benchmark truth for expected text, critical information, and sound labels.

Synthetic generation writes:

```text
benchmarks/<dataset>/audio-synthetic/manifest.jsonl
```

A record has the following shape:

```json
{
  "id": "ns-008",
  "scene_profile": "default-v1",
  "seed": 42,
  "sample_rate": 48000,
  "tts": {
    "engine": "whisperspeech",
    "version": "recorded-at-runtime"
  },
  "speech": {
    "start_ms": 500,
    "end_ms": 4210
  },
  "background": {
    "profile": "office-v1",
    "start_ms": 0,
    "end_ms": 5000,
    "snr_db": 15.0
  },
  "events": [
    {
      "label": "[phone rings]",
      "start_ms": 1800,
      "end_ms": 2700
    },
    {
      "label": "[knock]",
      "start_ms": 3400,
      "end_ms": 3650
    }
  ]
}
```

The manifest records actual generated values, not placeholders. The TTS package
and model/version information available at runtime must be captured so a result
can be traced back to its generator environment.

## Reproducibility

Scene scheduling and DeafBench-owned mixing are seed-driven. For the same
reference set, scene profile, and seed, DeafBench must produce the same scene
timing decisions and event/background mix parameters.

Do not claim cross-version or cross-hardware byte-for-byte TTS reproducibility
unless WhisperSpeech actually provides it. Instead:

- reuse an existing valid synthetic WAV when its manifest fingerprint matches
  the requested generation settings
- record the WhisperSpeech/runtime version in the manifest
- regenerate the full synthetic set when the requested scene fingerprint no
  longer matches

The synthetic set is all-or-nothing. A stale or partial generated set is rebuilt
as one coherent source set rather than mixing generation configurations.

## Source-aware run outputs

Benchmark runs must not overwrite human results with synthetic results.

Write results under:

```text
benchmarks/<dataset>/runs/<model>/<audio-source>/
  predictions.jsonl
  report.md
  run.json
```

Examples:

```text
benchmarks/non-speech-v1/runs/whisper/human/
benchmarks/non-speech-v1/runs/whisper/synthetic/
benchmarks/non-speech-v1/runs/whisper-at/synthetic/
```

`run.json` records at least:

- dataset
- model name and configured model identifier
- audio source: `human` or `synthetic`
- reference path
- audio path
- prediction path
- report path
- sample count
- scene profile and seed for synthetic runs
- TTS engine/version for synthetic runs
- benchmark package version

Prediction and report writes must use atomic replacement so a failed rerun does
not destroy the previous valid output.

The old top-level `model-a.jsonl`, `model-b.jsonl`, and report files remain
supported for existing repository workflows during this PR. The new benchmark
command writes only to `runs/` so source identity is explicit.

## Evaluation and reporting

After the model adapter completes:

1. Parse references and predictions with the installed DeafBench parser.
2. Align IDs using the existing alignment rules.
3. Evaluate with the existing metric implementation.
4. Generate the existing Markdown report format.
5. Save the report under the source-aware run directory.
6. Print the same terminal metric summary used by `deafbench compare` plus the
   dataset, model, resolved audio source, prediction path, and report path.

No metric semantics change in this PR.

## Dependency behavior

Normal `compare`, `report`, `recorder`, and `--help` usage must not import model
or TTS runtimes.

Benchmark dependencies are lazy-loaded after `benchmark` is selected and after
the resolved audio source shows which components are needed.

Packaging should use optional extras instead of making heavyweight model
runtimes mandatory for all DeafBench users. The implementation plan must verify
the supported install shape against the upstream packages before locking exact
extra requirements.

Missing dependencies must produce direct install/upstream guidance rather than a
Python traceback.

## Error handling

Fail before starting model inference when any of these are true:

- unsafe dataset name
- missing or invalid references
- explicit human source is incomplete
- synthetic generation cannot create a complete set
- WAV/reference IDs do not match
- generated manifest and synthetic WAV set disagree
- unsupported model name
- required model/TTS dependency is missing

Do not silently score partial datasets.

A failed generation or inference run must leave the previous valid prediction,
report, and synthetic set intact wherever atomic replacement can preserve them.

## CLI status and automation

`deafbench benchmark` returns zero only when generation/source resolution,
inference, evaluation, and report generation all complete successfully.

The workflow must be usable without GUI interaction when `auto` resolves to
synthetic audio. This is the default automation path for clean machines and CI
experiments that have the required model/TTS dependencies installed.

## Testing

Use TDD with test-only RED commits before production GREEN commits for behavior
changes.

Coverage must include:

- CLI parsing and model/audio-source validation
- `auto` choosing a complete human set
- `auto` choosing synthetic when any human WAV is missing
- no mixed human/synthetic source set
- explicit incomplete `human` failure
- source-aware run-path resolution
- deterministic scene timing for a fixed seed
- timestamp/manifest structure
- stale/partial synthetic-set regeneration decision
- prediction/report atomic replacement behavior
- adapter dispatch without importing unused heavyweight dependencies
- clean missing-dependency messages
- Whisper adapter behavior with a fake backend
- Whisper-AT adapter behavior and existing sound-label mapping
- compatibility behavior for the existing transcription tools
- full installed CLI functional flow with fake TTS/model adapters

CI continues to pass on Python 3.11, 3.12, 3.13, and 3.14 without downloading
or running real model weights. Real WhisperSpeech/Whisper/Whisper-AT execution is
manual integration validation outside the normal CI matrix.

## Non-goals for this PR

Do not add Qwen2.5-Omni yet.

Do not add the personal Deaf/cochlear speech track yet.

Do not change DeafBench metric semantics.

Do not add third-party environmental sound-effect downloads.

Do not mix incomplete human recordings with generated replacements.

Do not add a general model-plugin marketplace or arbitrary shell-command runner.

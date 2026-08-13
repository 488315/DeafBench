# DeafBench

I wrote DeafBench because I am Deaf, I use cochlear implants, and the normal
ASR score does not always describe whether captions are useful to me. A system
can have a low word error rate and still get the time, medication amount,
username, Wi-Fi name, confirmation code, speaker, or sound event wrong. Those
are not small mistakes when the caption is the information I have to act on.

My background is in IT, so I built this like an audit instead of a demo.
DeafBench keeps the references, model revisions, decoding settings, evaluator
revision, and artifact hashes with the result. It reports WER, but it also
reports strict lexical and typed canonical recall for critical information,
non-speech information, speaker attribution, latency, and the actual
substitution, insertion, and deletion counts. The typed evaluator only accepts
the harmless representation changes allowed for that entity type; it does not
turn a different username, code, time, or Wi-Fi name into a pass.

The project now has two separate jobs. The synthetic track measures whether an
ASR system preserves accessibility-critical information. The Hugging Face
compatibility track uses the pinned Open ASR Leaderboard datasets, normalizer,
preprocessing, WER calculation, and seven-dataset macro-average. I do not mix
those scores because they answer different questions.

## The goal

My goal is to build an ASR system that beats the Hugging Face Open ASR
Leaderboard while still doing better on the information that matters to Deaf
and hard-of-hearing users. The current reproduced Zipformer baseline scored
**5.23% public seven-set macro WER** with the pinned official-compatible local
workflow. That result is useful evidence that the runner matches the public
contract, but it is not a verified leaderboard win, it does not include the
private sets, and the Zipformer checkpoint is CC-BY-NC-4.0, so it cannot be the
commercial foundation without separate permission.

I will only say DeafBench beat the leaderboard after a separate candidate is
evaluated at a declared milestone and Hugging Face verifies the result. Until
then, 5.23% is the local public compatibility baseline to beat, not a product
claim. The exact upstream revisions, commands, and evidence are in
[`experiments/open-asr/README.md`](experiments/open-asr/README.md).

## Models that work with DeafBench

The models below have working adapters in this repository. The seven newer
adapters have completed a 25-sample synthetic-v2 run and a two-row public
real-speech smoke run, with byte-stable local result manifests under
`experiments/model-results`. A smoke run proves that the pinned adapter executes
and produces a valid result; it does not prove model quality or a leaderboard
score.

| DeafBench model name | Pinned model | Current evidence | License lane |
| --- | --- | --- | --- |
| `whisper` | OpenAI Whisper `turbo` | Core v1 and non-speech v1 reports | Runtime model; review upstream terms |
| `whisper-at` | Whisper-AT `medium.en` | Adapter and benchmark workflow | Research integration; review upstream terms |
| `faster-whisper` | `Systran/faster-whisper-small.en` | Frozen Core v1 baseline | Runtime model; review upstream terms |
| `distil-whisper` | `distil-whisper/distil-large-v3` | Adapter and local runner | Runtime model; review upstream terms |
| `qwen3-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B-hf` | Synthetic-v2 plus real-speech smoke | Commercial candidate, Apache-2.0 |
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B-hf` | Synthetic-v2 plus real-speech smoke | Commercial candidate, Apache-2.0 |
| `parakeet-tdt-0.6b-v2` | `nvidia/parakeet-tdt-0.6b-v2` | Synthetic-v2 plus real-speech smoke | Commercial candidate, CC-BY-4.0 attribution required |
| `granite-speech-4.1-2b` | `ibm-granite/granite-speech-4.1-2b` | Synthetic-v2 plus real-speech smoke | Commercial candidate, Apache-2.0 |
| `granite-speech-4.1-2b-nar` | `ibm-granite/granite-speech-4.1-2b-nar` | Synthetic-v2 plus real-speech smoke | Commercial candidate, Apache-2.0; audited remote code |
| `ark-asr-0.6b` | `AutoArk-AI/ARK-ASR-0.6B` | Synthetic-v2 plus real-speech smoke | Commercial candidate, Apache-2.0; audited isolated remote code |
| `ark-asr-0.6b-int8-onnx` | `AutoArk-AI/ark-asr-0.6b-int8-onnx` | Synthetic-v2 plus real-speech smoke | Commercial candidate, Apache-2.0; audited isolated remote code |

The machine-readable registry at
[`deafbench/model-registry.json`](deafbench/model-registry.json) pins revisions,
runtimes, license classifications, attribution requirements, expected download
sizes, and measured peak VRAM. That registry is operational metadata, not legal
advice. Model weights are third-party software and are not owned by DeafBench.

## OpenAI Whisper turbo results

`Model A` uses OpenAI Whisper `turbo` through `tools/transcribe_whisper.py`.

| Benchmark | Samples | WER | Critical Information Recall | Non-Speech Information Recall |
| --- | ---: | ---: | ---: | ---: |
| **Core v1** | 25 | **23.4%** | **88.7% (55/62)** | **N/A** |
| **Non-speech v1** | 12 | **2.0%** | **95.0% (19/20)** | **0.0% (0/19)** |

This is why DeafBench exists: on **Non-speech v1**, Whisper got **2.0% WER** and **95.0% critical information recall**, but captioned **0 of 19** environmental sound events.

Full reports:

- `benchmarks/core-v1/model-a-report.md`
- `benchmarks/non-speech-v1/model-a-report.md`

WER does not tell the full accessibility story. DeafBench also measures critical information loss and non-speech events that WER misses.

---

## Quickstart

### Installation

```bash
pip install deafbench
```

Install the recorder extra if you want to capture benchmark audio:

```powershell
python -m pip install "deafbench[recorder]"
deafbench recorder
```

Or install locally for development:

```bash
git clone https://github.com/488315/DeafBench.git
cd DeafBench
pip install -e .
```

### Automated benchmark workflow

Install the default synthetic-audio runtime, then run a complete benchmark:

```powershell
python -m pip install "deafbench[benchmark]"

# Both model backends require ffmpeg on PATH.
# Core v1 with OpenAI Whisper
python -m pip install -U openai-whisper
deafbench benchmark core-v1 --model whisper

# Non-speech v1 with Whisper-AT on Windows
python -m pip install numba numpy torch tqdm more-itertools tiktoken==0.3.3
python -m pip install --no-deps whisper-at
deafbench benchmark non-speech-v1 --model whisper-at
```

The `benchmark` extra installs WhisperSpeech and the audio dependencies used to
build synthetic scenes. OpenAI Whisper and Whisper-AT are separate inference
backends, so install the backend shown immediately before its benchmark command.
The extra does not install either inference backend.

Two additional local models use the Faster-Whisper runtime. For existing human
audio, install the local-model extra. Synthetic runs also need the benchmark
extra that supplies WhisperSpeech and its runtime dependencies:

```powershell
# Existing human audio
python -m pip install "deafbench[local-models]"

# Synthetic audio from an editable repository checkout
python -m pip install -e ".[benchmark,local-models]"

# CPU-friendly INT8 baseline; downloads small.en on first use.
python -m deafbench benchmark core-v1 --model faster-whisper --audio-source synthetic --repo-root .

# Distilled comparison; downloads distil-large-v3 on first use.
deafbench benchmark core-v1 --model distil-whisper
```

Both default to CPU INT8 so they work without an NVIDIA GPU. Faster-Whisper
uses `small.en`; Distil-Whisper uses the upstream-documented
`distil-large-v3` checkpoint with previous-text conditioning disabled. The
Faster-Whisper runtime decodes audio through PyAV, so these two models do not
need a separate system FFmpeg installation.

With the default `--audio-source auto` policy, DeafBench selects one complete
source for the whole run:

```text
complete audio/ set   -> human run
incomplete audio/ set -> complete audio-synthetic/ run
never mix sources
```

Use `--audio-source human` or `--audio-source synthetic` to require a specific
source. Human mode fails if `audio/` is incomplete. Synthetic mode generates or
reuses a complete synthetic set before inference begins.

WhisperSpeech supplies the speech signal. DeafBench supplies ambience,
environmental-event timing, and final mixing. The `default-v1` scene profile
uses seed `42` unless overridden. Scene planning is reproducible for the same
inputs, but DeafBench does not promise byte-identical TTS output across runtime,
model, or hardware versions.

Each successful run writes traceable, source-aware run artifacts. Synthetic
generation additionally writes its reusable manifest:

```text
benchmarks/<dataset>/runs/<model>/<audio-source>/predictions.jsonl
benchmarks/<dataset>/runs/<model>/<audio-source>/report.md
benchmarks/<dataset>/runs/<model>/<audio-source>/run.json
benchmarks/<dataset>/audio-synthetic/manifest.jsonl  # synthetic only
```

`run.json` records the resolved source, model identity, paths, sample count, and
benchmark version. Synthetic runs also record the scene profile, seed, and TTS
engine/version. Run directories include both model and source so human and
synthetic results cannot overwrite one another.

Reports keep critical-information scoring in two separate views. Strict lexical
recall measures the expected surface form; canonical semantic recall applies
only the normalization allowed by an entity's explicit type, such as TIME or
DIGIT_SEQUENCE. Reports also include per-sample WER and aggregate substitution,
insertion, and deletion counts.

### Usage

**1. Compare predictions against reference captions:**

```bash
deafbench compare examples/references.jsonl examples/model-a.jsonl
```

Output:
```text
DeafBench v0.1

Samples: 3

WER                         8.4%
Critical Information       91.2%
Non-Speech Information     62.5%
Speaker Attribution        87.0%
Median Latency             1.4s

⚠ 2 critical-information failures detected
```

**2. Generate a Markdown evaluation report:**

```bash
deafbench report examples/references.jsonl examples/model-a.jsonl --output report.md
```

### OpenAI Whisper transcription

Install OpenAI Whisper and generate `model-a.jsonl` for Core v1:

```powershell
python -m pip install -U openai-whisper
python tools\transcribe_whisper.py --dataset core-v1
```

Run the same helper for non-speech v1:

```powershell
python tools\transcribe_whisper.py --dataset non-speech-v1
```

The helper uses Whisper `turbo` in English and writes predictions into the selected benchmark directory.

### Whisper-AT Model B

Model B uses [Whisper-AT](https://github.com/YuanGongND/whisper-at) to keep speech recognition and audio-event tagging in one run. DeafBench stores the ASR transcript in `text`, mapped benchmark sound events in `sounds`, and the original Whisper-AT AudioSet labels in `audio_tags`. Keeping sound labels out of `text` means environmental-sound scoring does not change the speech WER.

Whisper-AT documents this Windows installation workaround:

```powershell
python -m pip install numba numpy torch tqdm more-itertools tiktoken==0.3.3
python -m pip install --no-deps whisper-at
```

Whisper-AT also requires `ffmpeg`. Check the upstream Whisper-AT README if its installation requirements change.

Generate Model B predictions for both current benchmarks:

```powershell
python tools\transcribe_whisper_at.py --dataset core-v1
python tools\transcribe_whisper_at.py --dataset non-speech-v1
```

The runner defaults to Whisper-AT `medium.en`. Model A uses Whisper `turbo`, so this comparison measures the complete captioning systems rather than isolating only the audio-tagging layer.

Generate the reports after Model B finishes:

```powershell
deafbench report benchmarks\core-v1\references.jsonl benchmarks\core-v1\model-b.jsonl --output benchmarks\core-v1\model-b-report.md
deafbench report benchmarks\non-speech-v1\references.jsonl benchmarks\non-speech-v1\model-b.jsonl --output benchmarks\non-speech-v1\model-b-report.md
```

Measured Model B numbers should only be added to this README after those runs are completed.

### Non-speech v1 recording workflow

`non-speech-v1` stays separate from Core v1. Each reference has one or more `sounds` labels. The GUI shows those labels before recording. When you press **Stop**, the recorder synthesizes each sound and appends it after the speech in label order.

```powershell
python -m pip install "deafbench[recorder]"
deafbench recorder --dataset non-speech-v1
```

`deafbench recorder` can run outside a source checkout. It defaults to Core v1 and seeds `benchmarks\<dataset>\references.jsonl` in the current directory when the bundled benchmark is not already there. Existing references are left alone, and recordings go to `benchmarks\<dataset>\audio`.

For example:

```json
"sounds": ["[phone rings]", "[knock]"]
```

produces:

```text
recorded speech → short gap → phone ring → short gap → knock
```

Transcribe and score it with:

```powershell
python tools\transcribe_whisper.py --dataset non-speech-v1
deafbench report benchmarks\non-speech-v1\references.jsonl benchmarks\non-speech-v1\model-a.jsonl --output benchmarks\non-speech-v1\model-a-report.md
```

Supported generated events are `[alarm]`, `[door closes]`, `[phone rings]`, `[knock]`, `[error notification]`, and `[siren]`.

---

## Input JSONL Schema

`references.jsonl`
```json
{
  "id": "sample-001",
  "text": "John Doe needs 25 milligrams on Friday.",
  "critical": ["John Doe", "25 milligrams", "Friday"],
  "sounds": ["[alarm]"],
  "speaker": "Speaker 1"
}
```

`predictions.jsonl`
```json
{
  "id": "sample-001",
  "text": "Guy needs 20 milligrams on Friday.",
  "latency_ms": 820,
  "speaker": "Speaker 1"
}
```

---

## License

[Apache License 2.0](LICENSE)

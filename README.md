# DeafBench

Evaluate what ASR metrics miss.

DeafBench is an open-source benchmark for measuring AI caption failures that matter to Deaf and hard-of-hearing users.

## Why DeafBench?

I'm Deaf, I use cochlear implants, and I have an IT background, so I look at captioning the same way I'd troubleshoot a system: if it drops the part that matters, a good-looking metric does not mean much.

The captioning systems I personally struggle with most are Google Chrome Live Caption and Android Live Caption through Android System Intelligence. Google Gemini can also have a hard time understanding me when I speak. My speech isn't always clear, but I still try my best to speak, and I want speech systems to be tested for that real-world accessibility gap instead of only clean audio.

DeafBench is an **ASR benchmark for Deaf and hard-of-hearing captions** built for **caption evaluation beyond word error rate (WER)**. It checks what speech-to-text and automatic speech recognition systems preserve, including critical information, environmental sound captions, speaker attribution, and latency.

The goal is simple: compare ASR and audio-captioning systems based on what a Deaf or hard-of-hearing user actually gets, not just how close the transcript is word for word.

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
deafbench benchmark core-v1 --model whisper
deafbench benchmark non-speech-v1 --model whisper-at
```

The `benchmark` extra installs WhisperSpeech and the audio dependencies used to
build synthetic scenes. OpenAI Whisper and Whisper-AT still require their own
model-specific installation steps below; the extra does not install either
inference backend.

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

Each successful run writes traceable, source-aware artifacts:

```text
benchmarks/<dataset>/audio-synthetic/manifest.jsonl
benchmarks/<dataset>/runs/<model>/<audio-source>/predictions.jsonl
benchmarks/<dataset>/runs/<model>/<audio-source>/report.md
benchmarks/<dataset>/runs/<model>/<audio-source>/run.json
```

`run.json` records the resolved source, model identity, paths, sample count, and
benchmark version. Synthetic runs also record the scene profile, seed, and TTS
engine/version. Run directories include both model and source so human and
synthetic results cannot overwrite one another.

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

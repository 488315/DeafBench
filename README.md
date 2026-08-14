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

The models below have working adapters in this repository. Nine newer adapters
have recorded local observations for a 25-sample synthetic-v2 run and a two-row
public real-speech smoke run. Their byte-stable metadata manifests are under
`experiments/model-results`, but the sample-level predictions and run artifacts
are not published in this checkout, so these values cannot be independently
recomputed from the repository alone. A smoke observation shows that the pinned
adapter executed in the recorded environment; it does not prove model quality
or a leaderboard score. OpenAI Whisper `turbo` remains legacy report evidence
only and does not have one of these newer manifests.

| DeafBench model name | Pinned model | Current evidence | License lane |
| --- | --- | --- | --- |
| `whisper` | OpenAI Whisper `turbo` | Core v1 and non-speech v1 reports | Runtime model; review upstream terms |
| `whisper-at` | Whisper-AT `medium.en` | Synthetic-v2, real-speech smoke, and non-speech-v1 | Commercial candidate, BSD-2-Clause |
| `faster-whisper` | `Systran/faster-whisper-small.en` | Frozen Core v1 baseline | Runtime model; review upstream terms |
| `distil-whisper` | `Systran/faster-distil-whisper-large-v3` | Synthetic-v2 plus real-speech smoke | Commercial candidate, MIT |
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

## Results for all integrated models

These numbers do not belong in one leaderboard. Synthetic-v2 measures
accessibility-critical information on 25 generated samples, while the public
real-speech smoke set has only two rows and proves execution rather than model
quality. Core v1 and Non-speech v1 are earlier, separately frozen benchmarks.

### Synthetic-v2 accessibility results

| Model | WER | Strict lexical recall | Canonical semantic recall | Local RTFx | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distil-Whisper large-v3 | 23.8% | 66.1% | 91.9% | 0.80 | CPU |
| Whisper-AT `medium.en` | 26.2% | 67.7% | 96.8% | 7.34 | 4.46 GiB |
| Qwen3-ASR 0.6B | 26.6% | 67.7% | 90.3% | 10.47 | 1.52 GiB |
| Qwen3-ASR 1.7B | 21.0% | 67.7% | 91.9% | 9.89 | 3.86 GiB |
| Parakeet TDT 0.6B v2 | 22.7% | 64.5% | 91.9% | 71.23 | 4.67 GiB |
| Granite Speech 4.1 2B | 18.9% | 69.4% | 91.9% | 12.91 | 4.35 GiB |
| Granite Speech 4.1 2B NAR | 40.6% | 64.5% | 87.1% | 45.07 | 4.26 GiB |
| ARK-ASR 0.6B | 30.1% | 66.1% | 90.3% | 14.85 | 2.20 GiB |
| ARK-ASR 0.6B INT8 ONNX | 26.6% | 66.1% | 90.3% | 2.40 | CPU |

### Two-row public real-speech smoke results

| Model | WER | Local RTFx | Peak VRAM |
| --- | ---: | ---: | ---: |
| Distil-Whisper large-v3 | 1.73% | 3.27 | CPU |
| Whisper-AT `medium.en` | 1.73% | 11.19 | 4.46 GiB |
| Qwen3-ASR 0.6B | 2.31% | 12.27 | 1.72 GiB |
| Qwen3-ASR 1.7B | 1.16% | 12.43 | 4.05 GiB |
| Parakeet TDT 0.6B v2 | 2.31% | 91.85 | 4.67 GiB |
| Granite Speech 4.1 2B | 3.47% | 7.43 | 4.42 GiB |
| Granite Speech 4.1 2B NAR | 2.89% | 25.15 | 4.46 GiB |
| ARK-ASR 0.6B | 2.89% | 7.49 | 2.29 GiB |
| ARK-ASR 0.6B INT8 ONNX | 2.89% | 3.01 | CPU |

The byte-stable metadata records for both tables are in
[`experiments/model-results`](experiments/model-results). They are recorded
local observations, not independently recomputable evidence, because their
sample-level artifacts are not in this checkout. GPU rows are local RTX 4070
measurements, and CPU rows are labeled separately. The two-row smoke results
are not the seven-dataset macro-average and are not Hugging Face verified.

### Earlier model evidence

| Model | Benchmark evidence | Result |
| --- | --- | --- |
| OpenAI Whisper `turbo` | Core v1, 25 samples | 23.4% WER; 88.7% legacy critical-information recall |
| OpenAI Whisper `turbo` | Non-speech v1, 12 samples | 2.0% WER; 95.0% legacy critical-information recall; 0.0% non-speech recall |
| Faster-Whisper `small.en` | Frozen Core v1 synthetic baseline, 25 samples | 26.2% WER; 69.4% strict lexical recall; 90.3% canonical semantic recall |
| Whisper-AT `medium.en` | Non-speech v1, 12 samples | 2.0% WER; 95.0% strict and canonical critical recall; 0 of 19 expected sound events matched |

The OpenAI Whisper reports are
[`benchmarks/core-v1/model-a-report.md`](benchmarks/core-v1/model-a-report.md)
and
[`benchmarks/non-speech-v1/model-a-report.md`](benchmarks/non-speech-v1/model-a-report.md).
The Faster-Whisper classification and scoring evidence is in
[`benchmarks/core-v1/faster-whisper-synthetic-analysis.md`](benchmarks/core-v1/faster-whisper-synthetic-analysis.md).
The older Whisper recall value uses the evaluator that produced those frozen
reports, so I do not label it as strict or canonical scoring.

WER does not tell the full accessibility story. DeafBench also measures
critical information loss and non-speech events that WER misses.

## Accessibility stress testing

[`accessibility-stress-v1`](benchmarks/accessibility-stress-v1/README.md) adds a
byte-frozen 24-utterance reference set for paired clean and degraded runs. It
predeclares fixed-SNR street, office, wind, breathing, keyboard, and rustling
noise;
noise-only interstitials; 8 kHz telephony; reverberation; long pauses; rate
variation; overlap; and codec degradation. The evaluator keeps WER edits,
deletion share, typed critical failures, interstitial hallucinations, caption
timing drift, and observed local load metrics separate instead of reducing the
stress run to one number.

This is synthetic stress coverage, not a Deaf or dysarthric speech dataset. I
will not use rate changes, pauses, or noise to claim demographic performance.
That evidence requires a separate authorized and consented human-speech lane
with subgroup reporting and a corpus that is appropriate for that purpose.

The executable local lane requires clean WAV files named for the selected
reference IDs. `--implemented-only` runs the six transformation families that
DeafBench can currently materialize and labels the result with the exact sample
count. It does not count the declared overlap or codec cases as completed.

```powershell
python -m deafbench stress `
  --references benchmarks/accessibility-stress-v1/references.jsonl `
  --clean-audio benchmarks/accessibility-stress-v1/audio-clean `
  --output benchmarks/accessibility-stress-v1/runs/faster-whisper/local `
  --model faster-whisper `
  --implemented-only
```

The output directory contains hash-bound preparation evidence, clean and
stressed predictions, and a local result. Generated audio, predictions, and
runs stay untracked.

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

### Customer-run accessibility audit

The founding-pilot workflow runs on the customer's authorized computer. Raw
audio, transcripts, filenames, paths, and critical-information values are not
customer-export artifacts. Install the signing dependency and inspect the
supported local actions:

```powershell
python -m pip install "deafbench[zero-custody-pilot]"
deafbench audit --help
```

Run the synthetic rehearsal before evaluating authorized, non-sensitive
audio:

```powershell
deafbench audit rehearse `
  --repo-root . `
  --output-dir .\rehearsal-export `
  --signing-key C:\secure-local-path\deafbench-signing-key.pem
```

The customer-local evaluation and aggregate export are one command:

```powershell
deafbench audit run `
  --repo-root . `
  --case-root C:\customer-controlled\deafbench-case `
  --attestation C:\customer-controlled\execution-attestation.json `
  --output-dir C:\customer-controlled\deafbench-export `
  --signing-key C:\secure-local-path\deafbench-signing-key.pem
```

The signing key and all customer artifacts stay on the customer's computer.
The export is labelled customer-executed and environment-dependent; it is not
a certification or a Hugging Face leaderboard result.

### Automated benchmark workflow

Install the default synthetic-audio runtime, then run a complete benchmark:

```powershell
python -m pip install "deafbench[benchmark]"

# Both model backends require ffmpeg on PATH.
# Core v1 with OpenAI Whisper
python -m pip install -U openai-whisper
deafbench benchmark core-v1 --model whisper
```

Whisper-AT uses a pinned upstream commit whose installer imports the removed
`pkg_resources` module. DeafBench keeps that source and its runtime requirements
unchanged, verifies their hashes, and applies a packaged build-only patch that
uses `pathlib` to read `requirements.txt` and requires setuptools 83 or newer.
Install it from a DeafBench checkout with Python 3.11:

```powershell
python -m pip install ".[test]"
python -m deafbench.whisper_at_compat
python -c "import whisper_at"
```

Python 3.11 is required for this pinned Whisper-AT runtime because its exact
`tiktoken==0.3.3` dependency does not publish wheels for every newer supported
DeafBench interpreter. This restriction preserves the upstream dependency pin
instead of silently changing model behavior. The patch manifest records the
upstream commit and every before/after source hash in
`deafbench/whisper_at_compat/manifest.json`.

The `benchmark` extra installs WhisperSpeech and the audio dependencies used to
build synthetic scenes. OpenAI Whisper is a separate inference backend, so
install it before its benchmark command. The extra does not install inference
backends.

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

# Distilled comparison; downloads the pinned CTranslate2 repository on first use.
deafbench benchmark synthetic-v2 --model distil-whisper --audio-source synthetic --repo-root .
```

Both default to CPU INT8 so they work without an NVIDIA GPU. Faster-Whisper
uses `small.en`; Distil-Whisper pins
`Systran/faster-distil-whisper-large-v3` with previous-text conditioning
disabled. The Faster-Whisper runtime decodes audio through PyAV, so these two
models do not need a separate system FFmpeg installation.

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
insertion, and deletion counts. Conventional transcription output now names
orthographic and normalized WER and CER separately and records the normalization
policy. The exact aggregation, normalization, RTFx, and leaderboard boundaries
are documented in
[`docs/asr-evaluation-methodology.md`](docs/asr-evaluation-methodology.md).

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

Install Whisper-AT with the exact revision and dependency versions in the
[automated benchmark workflow](#automated-benchmark-workflow) above. Do not
replace that command with an unpinned PyPI installation when reproducing the
recorded evidence. Whisper-AT also requires `ffmpeg` on `PATH`.

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

# DeafBench

Evaluate what ASR metrics miss.

DeafBench is an open-source benchmark for measuring AI caption failures that matter to Deaf and hard-of-hearing users.

## OpenAI Whisper turbo results

`Model A` uses OpenAI Whisper `turbo` through `tools/transcribe_whisper.py`.

| Benchmark | Samples | WER | Critical Information Recall | Non-Speech Information Recall |
| --- | ---: | ---: | ---: | ---: |
| **Core v1** | 25 | **23.4%** | **88.7% (55/62)** | **N/A** |
| **Non-speech v1** | 12 | **2.0%** | **95.0% (19/20)** | **0.0% (0/19)** |

This is why DeafBench exists: Whisper got **2.0% WER** and **95.0% critical information recall**, but captioned **0 of 19** environmental sound events.

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

Or install locally for development:

```bash
git clone https://github.com/488315/DeafBench.git
cd DeafBench
pip install -e .
```

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

### Non-speech v1 recording workflow

`non-speech-v1` stays separate from Core v1. Each reference has one or more `sounds` labels. The GUI shows those labels before recording. When you press **Stop**, the recorder synthesizes each sound and appends it after the speech in label order.

```powershell
python -m pip install -r tools\recorder\requirements.txt
python -m tools.recorder.recorder --dataset non-speech-v1
```

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

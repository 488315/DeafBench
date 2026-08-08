# DeafBench

Evaluate what ASR metrics miss.

DeafBench is an open-source benchmark for measuring AI caption failures that matter to Deaf and hard-of-hearing users.

```text
Traditional ASR benchmark:

Model A: 7.8% WER
Model B: 8.1% WER

DeafBench:

Model A: 72% critical information preserved
Model B: 96% critical information preserved
```

Two models can have similar Word Error Rate (WER) while producing very different accessibility outcomes. DeafBench highlights critical information loss (names, numbers, negations, dates, technical terms) and non-speech annotations that traditional WER obscures.

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

### Non-speech v1 recording workflow

`non-speech-v1` keeps its acoustic-event benchmark separate from the Core v1 baseline. Each reference record contains one or more `sounds` labels. The recorder shows those labels in the GUI and, when you press **Stop**, appends the matching deterministic synthetic sound events after the recorded speech in the same order as the labels.

```powershell
python -m pip install -r tools\recorder\requirements.txt
python -m tools.recorder.recorder --dataset non-speech-v1
```

For example, a prompt with:

```json
"sounds": ["[phone rings]", "[knock]"]
```

is saved as recorded speech, a short gap, the synthetic phone ring, another gap, then the synthetic knock.

Transcribe and score the completed set with:

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

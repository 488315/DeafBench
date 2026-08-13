# Faster-Whisper Synthetic Baseline Analysis

This analysis covers the original seven critical-information failures from the
25-sample `small.en` faster-whisper run. Generated WAV files and run bundles
remain local, untracked evidence and are not part of this document.

## Verification method

- The WhisperSpeech 0.8.9 generator was called with each reference record's
  exact `text` field. The manifest binds the 25 WAV files to those synthesis
  inputs, `default-v1`, seed 42, and the TTS version.
- The manifest proves the requested input and generation settings, but not that
  the synthesizer pronounced every character correctly.
- OpenAI Whisper `turbo` independently transcribed the same cached 25 WAV files.
  It does not share the faster-whisper runtime or `small.en` model.
- No classification relies on listening alone.

## Original failure classification

| Sample | Faster-whisper evidence | Independent evidence | Classification | Canonical decision and reason |
| --- | --- | --- | --- | --- |
| `core-001` | `2 clear 15 p.m.` | `2 clear 15 p.m.` | Synthetic-audio generation defect | Fail. Two independent paths agree on the unintended word `clear`; TIME normalization must not erase it. |
| `core-006` | `8 30 p.m.` | `8.30 pm` | Formatting-only mismatch | Pass. Both represent the complete intended time, 8:30 PM. |
| `core-009` | `4 core 5 p.m.` | `4.05 p.m.` | Ambiguous | Fail. Neither path recovers 4:45 PM, and the evidence cannot distinguish unclear synthesis from two recognition errors. |
| `core-011` | `devcusser23` | `devkassar23` | Synthetic-audio generation defect | Fail. Both paths corrupt the same username region; USERNAME requires the exact canonical characters `dev_user23`. |
| `core-012` | `11 o'clock 45pm` | `11 o'clock 45 pm` | Semantic equivalent | Pass. Both contain the complete spoken form of 11:45 PM. |
| `core-016` | `seven` | `7` | Synthetic-audio generation defect | Fail. Both paths stop after the first of seven intended digits; an incomplete sequence cannot match. |
| `core-019` | `Alpha's Guest` | `Office Guest` | True recognition error | Fail. The independent path recovers the intended SSID, while faster-whisper substitutes a different name. |

All seven remain failures under strict lexical scoring. Only `core-006` and
`core-012` pass canonical semantic scoring. Typed CODE scoring also exposes a
separate pre-existing error in `core-019`: `Alpha-79` is not the exact code
`alpha79`, so punctuation normalization must not hide it.

## Scope boundary

These results characterize only DeafBench's local synthetic `core-v1` runner.
They are not comparable with the Hugging Face Open ASR Leaderboard because this
runner does not reproduce its official datasets, normalization, or
macro-average calculation.

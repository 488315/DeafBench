# Real speech development v1

This lane is a deterministic 100-sample cohort from the public LibriSpeech
`clean` validation split. It is for model selection and smoke evaluation only.
The pinned source revision is recorded in `manifest.json`, and the source FLAC
hash for every selected sample is recorded in `references.jsonl`.

The cohort is selected by sorting all 2,703 validation sample IDs by SHA-256 and
taking the lowest 100. Selection does not inspect transcripts or official test
labels. The official Open ASR Leaderboard LibriSpeech test splits are explicitly
excluded.

LibriSpeech is distributed under CC-BY-4.0. See the
[dataset card](https://huggingface.co/datasets/openslr/librispeech_asr) for
source and attribution details.

This public read-speech cohort is not evidence of performance on Deaf or Hard of
Hearing speech, dysarthric speech, accents, spontaneous conversation, or noisy
captioning environments. Those claims require separate authorized datasets and
subgroup reporting.

# Core v1 benchmark

## Frozen synthetic baseline

The original 25-sample synthetic corpus and faster-whisper baseline are
immutable. `freeze-manifest.json` records the reference, audio, generation,
model, evaluator, dependency, prediction, report, and score identities. Do not
regenerate or overwrite files below `audio-synthetic/` or
`runs/faster-whisper/synthetic/`.

Run `python -m pytest tests/test_core_v1_freeze.py -q` to verify tracked inputs
and any locally available generated evidence against the permanent manifest.
Replacement audio belongs to a separately named corpus and must never update
this freeze record.

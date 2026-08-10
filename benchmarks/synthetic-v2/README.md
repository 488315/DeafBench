# DeafBench synthetic-v2

This corpus is a separately versioned successor to the immutable `core-v1`
synthetic evidence. It measures preservation of accessibility-critical
information; it is not an Open ASR Leaderboard result.

`quality-policy.json` was committed before replacement audio was generated.
Every admitted sample must pass every listed model-independent gate. A
reference-conditioned forced aligner supplies coverage evidence. An independent
ASR transcript is supporting evidence only: it cannot admit a sample, and a
disagreement with the aligner quarantines the sample.

Only `core-001`, `core-009`, `core-011`, and `core-016` may be regenerated.
All other audio must remain byte-identical to its `core-v1` parent. References,
critical values, and typed critical-entity labels must remain unchanged for all
25 samples.

Generated audio, quarantine files, and benchmark run artifacts are evidence,
not source files, and must not be committed. Their cryptographic hashes and
generation/validation metadata belong in the frozen corpus manifest.

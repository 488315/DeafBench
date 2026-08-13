# Changelog

## 0.2.0 - Unreleased

DeafBench 0.2.0 separates accessibility-critical synthetic evaluation from
the pinned Open ASR Leaderboard compatibility lane. This release does not
claim a verified leaderboard result.

### Added

- Typed strict and canonical critical-information recall with per-entity
  failure reporting.
- Frozen Core v1 and synthetic-v2 manifests with integrity verification.
- Pinned adapters and license metadata for Qwen3-ASR, Parakeet TDT, Granite
  Speech, and isolated ARK-ASR runtimes.
- A customer-executed, zero-custody audit and aggregate-only signed export.
- Public real-speech compatibility scoring and error analysis using the pinned
  upstream normalization contract.

### Changed

- Benchmark reports preserve substitutions, insertions, deletions, per-sample
  WER, local latency, RTFx, model revision, decoding configuration, and corpus
  identity.
- PyPI publication fails closed unless a published GitHub release and the
  explicit repository approval flag are both present.

### Boundaries

- Model weights remain third-party artifacts governed by their own licenses.
- Customer audio stays on the customer's authorized computer.
- Local public-set results are not Hugging Face verification or certification.

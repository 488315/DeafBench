# Changelog

## 2026-08-13 / v0.2.0

DeafBench 0.2.0 turns the repository into a reproducible ASR evaluation
system with separate accessibility-critical, public development, non-speech,
and Open ASR compatible lanes. Results from different lanes remain separate.
This release does not claim a Hugging Face verified leaderboard result,
certification, or measured performance where a model has not been run.

### Evaluation integrity and frozen corpora

- Froze Core v1 and synthetic-v2 with reference hashes, per-audio SHA-256
  values, synthesis and model metadata, decoding configuration, evaluator
  identity, dependency versions, and immutable result evidence.
- Added strict lexical and canonical semantic critical-information recall.
  Typed comparison covers times, digit sequences, usernames, codes and
  passwords, SSIDs, and proper names without fuzzy matching values that must
  remain exact.
- Preserved per-sample WER plus aggregate substitutions, insertions, and
  deletions. Reports include critical failures by entity type and corpus,
  model, evaluator, and decoding identities.
- Added synthetic-v2 quality gates for container validity, sample rate,
  channels, waveform content, truncation, silence, clipping, duration,
  reference hashes, alignment coverage, and typed-entity fidelity. Validator
  disagreements quarantine a sample instead of silently accepting it.
- Pinned the Open ASR Leaderboard compatibility implementation and kept its
  datasets, English normalization, preprocessing, WER, macro-average, and
  RTFx contracts separate from DeafBench's typed accessibility metrics.
- Added byte-stable result-manifest validation and explicit tests for Core v1,
  synthetic-v2, accessibility-stress-v1, real-speech-dev-v1, Open ASR
  evidence, and non-speech-v1 integrity.

### Public development and accessibility stress lanes

- Added `real-speech-dev-v1`, a deterministic 100-sample cohort selected from
  the pinned LibriSpeech clean validation split by the lowest SHA-256 sample
  IDs. It is declared for model selection and excludes official Open ASR test
  labels.
- Added `deafbench dev-corpus materialize`. The command validates the full
  2,703-row source population, identities, source text, audio hashes, and
  duplicate IDs before transactionally promoting verified 48 kHz mono WAVs.
  Generated audio, temporary materialization directories, backups, and runs
  remain ignored.
- Recorded a local faster-whisper `small.en` development baseline using the
  official-compatible English normalizer: 3.5587% WER, 51 substitutions,
  11 insertions, and 8 deletions across 1,967 reference words. This is local
  development evidence, not an official leaderboard result.
- Added deterministic interstitial tests for street noise, office chatter,
  keyboard clicks, breathing, and rustling at parameterized SNR levels.
  Scoring distinguishes ignored intervals, declared non-speech annotations,
  and hallucinated lexical output.
- Added the byte-frozen `accessibility-stress-v1` contract with 24 paired
  clean/degraded cases across eleven high-impact risk categories.
- Added model-independent transforms and metrics for noise, 8 kHz telephony,
  reverberation, extended pauses, speech-rate changes, deletion-heavy
  failures, critical-risk groups, caption timing drift, latency, TTFB, RTFx,
  throughput, CPU use, and peak VRAM.
- Kept synthetic acoustic proxies separate from claims about Deaf,
  hard-of-hearing, dysarthric, accented, or other demographic speech.

### Models, runtimes, and license evidence

- Added a fail-closed machine-readable model registry with pinned revisions,
  upstream URLs, SPDX licenses, commercial-use classifications, attribution
  and redistribution requirements, remote-code status, languages, expected
  size, runtime support, and intended evaluation lane.
- Added optional, isolated adapters and dependency extras for Qwen3-ASR 0.6B
  and 1.7B, NVIDIA Parakeet TDT 0.6B v2, IBM Granite Speech 4.1 2B and NAR,
  ARK-ASR 0.6B, faster-whisper, Distil-Whisper, and Whisper-AT.
- Kept Zipformer XL in the `research_only` lane because CC-BY-NC-4.0 does not
  permit using it as the paid product foundation without separate permission.
- Added pinned Distil-Whisper and Whisper-AT local evidence. Whisper-AT model
  names and checkpoint hashes fail closed, audio-event tags stay separate
  from ASR transcript text, and only declared mappings affect event recall.
- Froze non-speech-v1 and preserved speech WER and environmental-event recall
  as separate metrics.
- Preserved exact third-party license and notice material where required.
  The registry is an engineering control, not legal advice, and DeafBench
  does not redistribute or claim ownership of third-party model weights.

### Customer-run accessibility audit

- Added the customer-executed, zero-custody Accessibility-Critical ASR Audit.
  Audio stays on the customer's authorized computer; the workflow does not
  require upload, remote shell access, unattended access, or credentials.
- Added `deafbench audit rehearse`, `deafbench audit run`, and
  `deafbench audit export`. The earlier nested `audit` action remains as a
  compatibility alias, and optional audit dependencies are lazy-loaded.
- Added fail-closed machine-readable authorization, founding-pilot exclusions,
  isolated case IDs, event-ledger integrity, retention and logical-deletion
  controls, incident stops, deletion certificates, and a synthetic rehearsal.
- Added aggregate-only exports that reject raw audio, transcripts, filenames,
  sample-level results, speaker identities, paths, secrets, and critical-value
  leakage, including malicious filenames.
- Added signed reproducibility manifests containing evaluator and model
  identities, dataset count, configuration, aggregate metrics, and artifact
  hashes. Self-signatures are described as integrity evidence only unless the
  signing-key fingerprint is independently trusted.
- Added source-control scanning and a fail-closed pre-commit hook for customer
  artifacts, transcripts, predictions, reports, secrets, case identifiers,
  and renamed private keys, including nested repositories and worktrees.
- Kept payment activation closed. DeafBench 0.2.0 does not claim HIPAA
  compliance, formal certification, hosted-customer-data controls, or
  readiness for regulated or high-risk founding-pilot data.

### CLI, reporting, and packaging

- Added the installed `benchmark`, `dev-corpus`, and customer-audit command
  dispatchers while keeping unrelated commands independent of optional model,
  dataset, and signing dependencies.
- Added deterministic synthetic scene generation, automated model-adapter
  orchestration, dataset recording helpers, generic benchmark transcription,
  and the Whisper-AT Model B workflow.
- Improved terminal and Markdown reports with aligned failure IDs, escaped
  sample IDs, non-speech failure details, trimmed output, and explicit
  unavailable-metric handling.
- Added canonical handling for spoken identifiers, repeated underscores,
  numeric captions, conjunctions, times, years, versions, IPv4 addresses, and
  zero-cent currency while preserving exactness for security-sensitive values.
- Bundled the benchmark reference data required by installed recorder and
  evaluation workflows; generated audio, predictions, caches, and runs are
  not packaged or tracked.
- Added canonical package links for the project page, repository, and issue
  tracker and set the package version to 0.2.0.

### Validation and release safeguards

- Raised the enforced repository coverage floor to 90.6% with two-decimal
  reporting and added direct malformed-authorization and adapter failure-path
  coverage.
- CI now runs on Python 3.11 through 3.14 and directly enforces Ruff, bytecode
  compilation, frozen-manifest integrity, the complete test suite, branch-aware
  coverage, customer-artifact scanning, and a clean built-wheel installation.
- Hardened real-speech materialization after review: unexpected transitive
  import failures are preserved, only selected audio payloads remain buffered,
  the hash-selected cohort is recomputed, existing destinations are replaced
  transactionally, and post-promotion backup cleanup cannot misreport a valid
  destination as failed.
- Separated the recorded Python 3.14 decoding environment from corpus
  materialization. The exact materialization Python minor was not captured;
  the enforced supported range is recorded as Python 3.11 through 3.13 rather
  than inventing evidence.
- GitHub release builds create and check both wheel and source distribution.
  PyPI publication remains disabled unless the release is published and the
  repository variable `DEAFBENCH_PYPI_PUBLISH` is explicitly set to `true`.

### Compatibility and boundaries

- Base package and CI support Python 3.11 through 3.14. The pinned public
  development materializer currently supports Python 3.11 through 3.13 due to
  its optional data stack.
- Model weights and public datasets remain third-party artifacts governed by
  their own licenses and terms.
- Local measurements depend on the declared hardware, runtime, corpus, and
  decoding configuration and are not directly comparable with leaderboard
  hardware measurements unless the full official contract matches.
- Do not train on official test sets, tune repeatedly against test labels, or
  claim that DeafBench beat 5.37% Average WER until Hugging Face verifies a
  submission under the official contract.

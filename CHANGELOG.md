# Changelog

## 2026-08-15 / v0.2.2

### Customer audit usability

- Added the supported `deafbench[audit]` install extra and the top-level
  `deafbench audit <case-folder>` workflow. First run requires a customer case
  name, records the local authorization boundary, validates `references.csv`
  and WAV inputs, and stores reusable state under `.deafbench/`; later runs
  reuse that setup automatically.
- Added category-first customer reports in accessible HTML plus a companion PDF.
  Findings preserve REF/HYP alignment, one primary failure category, related
  factors, consequence-based severity, sentence-case customer labels, and
  deterministic investigation guidance. Opening the HTML report is best-effort
  and never changes audit success.
- Added `deafbench review <case-folder>` for one-finding-at-a-time customer
  review. DeafBench preserves its automatic severity, records customer context
  or a severity adjustment with a required reason, signs review state when the
  case signing key exists, and regenerates HTML/PDF once when the review ends.
- Made CPU audit evidence valid by allowing `peak_vram_bytes` to be null across
  model-run validation, canonical result manifests, signed aggregate exports,
  and customer report data while retaining numeric validation when VRAM is
  measured.
- Hardened report generation and review handling: JiWER 3.x empty-text cases are
  handled explicitly, PDF output embeds Unicode-capable fonts, negation matching
  uses token boundaries, saved findings are validated before review, and report
  replacement rolls back safely if promotion fails.
- Limited sample-level customer work to the latest successful run. Successful
  reruns remove older `.deafbench/runs/` work directories while the local
  authorization record keeps its planned case deletion date.
- Added a neutral English accessibility benchmark table to the README and kept
  the synthetic accessibility track separate from the two-row real-speech smoke
  evidence and Hugging Face compatibility results.

### Validation

- Release validation covers Python 3.11 through 3.14, clean-wheel installation,
  ARK and Whisper-AT compatibility, customer-artifact source-control scanning,
  and the existing 90.6% branch-coverage gate.

## 2026-08-14 / v0.2.1

### Security

- Upgraded the pinned Open ASR runtime and the DeafBench build backend to
  setuptools 83.0.0 or newer. This closes CVE-2025-47273
  (`GHSA-5rjg-fvgr-3xxf`) and CVE-2026-59890
  (`GHSA-h35f-9h28-mq5c`) while retaining Python 3.11 through 3.14 support.
- Replaced the deprecated table-form project license with the PEP 639 SPDX
  expression and explicit Apache-2.0 license-file declaration.
- Replaced Whisper-AT's build-only `pkg_resources` requirements parser with a
  hash-verified `pathlib` parser while preserving its pinned source, runtime
  requirements, adapter, license, and frozen evidence. Fresh Python 3.11
  environments now install it with setuptools 83 or newer.
- Raised the ARK native and ONNX Transformers floor to 5.5.0 and validated the
  exact floor without changing their pinned model revision. This fixes
  CVE-2026-1839 (`GHSA-69w3-r845-3855`), CVE-2026-4372
  (`GHSA-29pf-2h5f-8g72`), and CVE-2026-5241
  (`GHSA-fgcw-684q-jj6r`) for DeafBench runtime dependencies.
- Removed the unused Transformers package from the isolated Open ASR lock,
  aligned Granite's declared runtime with its tested Transformers 5.13 range,
  and made ARK snapshot resolution validate symlink targets and staged bytes.
  These changes preserve the supported adapters instead of hiding an
  incompatible model to satisfy dependency scanning.
- Patched the installable Open ASR/Zipformer experiment environment from
  PyTorch 2.4.0 to 2.6.0, the first release containing the upstream fix for
  CVE-2025-32434 (`GHSA-53q9-r3pm-6pq6`). The vulnerable runtime could execute
  attacker-controlled code while loading an untrusted PyTorch checkpoint even
  when `torch.load(..., weights_only=True)` was requested.
- Updated the compiled `k2` wheel, TorchAudio, Triton, SymPy, and CUDA runtime
  pins as one ABI-compatible CUDA 12.4 set. The Python 3.12 resolver now checks
  the complete environment instead of allowing a patched Torch package to be
  paired with binaries compiled for Torch 2.4.0.
- Replaced the global PyTorch extra index with immutable, SHA-256-pinned
  official Torch and TorchAudio wheel URLs. PyPI remains the normal source for
  unrelated packages, avoiding the prior cross-index resolution ambiguity.
- Added a regression test that fails when the Open ASR lock returns to a Torch
  release below 2.6.0, uses mismatched Torch/TorchAudio versions, selects a
  `k2` wheel for a different Torch ABI, restores the global extra index, or
  removes the expected wheel hashes.
- Recorded time-bounded dispositions for the low-severity development alerts
  CVE-2025-3001 (`GHSA-qfhq-4f3w-5fph`) and CVE-2025-3000
  (`GHSA-rrmf-rvhw-rf47`). The reviewed Granite NAR and Open ASR paths fail
  closed after 2026-11-13 or when their source hashes, affected-API inventory,
  upstream revisions, or Torch/TorchAudio/k2 ABI lock changes.
- The two development Torch alerts remain open: the official Granite NAR stack
  is pinned to Torch 2.9.1, while the Open ASR experiment uses the matched
  Torch 2.6.0 CUDA 12.4 stack. DeafBench does not describe those alerts as
  fixed and does not substitute an unvalidated binary-stack upgrade.

This patch changes the reproducible runtime used for future Open ASR baseline
executions. It does not alter or relabel the frozen v0.2.0 result artifacts,
does not claim a Hugging Face verified leaderboard result, and keeps the
runtime Critical, High, and Medium Dependabot alert counts at zero.

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

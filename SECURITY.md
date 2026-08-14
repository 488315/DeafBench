# Security

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's security
advisory interface. Do not include customer audio, transcripts, credentials,
or other sensitive content in a report.

## Granite NAR Torch advisories

DeafBench's optional `granite-nar-asr` extra pins the dependency versions used
by the upstream Granite Speech 4.1 2B NAR example: Torch 2.9.1, Torchaudio
2.9.1, and TorchCodec 0.9.1. Dependabot alerts #15 and #16 identify two
low-severity, local-attack memory-corruption advisories in that optional model
extra, which GitHub classifies as development scope:

- `GHSA-qfhq-4f3w-5fph` / `CVE-2025-3001` affects `torch.lstm_cell`.
- `GHSA-rrmf-rvhw-rf47` / `CVE-2025-3000` affects `torch.jit.script`.

Neither affected API appears in DeafBench production code or in the exact
hash-audited remote source for the pinned Granite NAR revision. Granite NAR
also runs outside the main process, with network access disabled during
inference. Before execution, both the parent and isolated worker require exact
reviewed-file hashes and scan those files for the affected APIs.

This is a temporary tolerable-risk disposition, not a claim that the
vulnerabilities are fixed. The second advisory requires Torch 2.13.0, while
the official Torchaudio packages and compatibility table do not yet provide a
matching Torchaudio 2.13 release. DeafBench therefore retains the matched
upstream stack instead of installing an unsupported audio ABI combination.

An import-only trial of the matched Torch 2.10, Torchaudio 2.10, and TorchCodec
0.10 stack was not sufficient to validate real Granite NAR inference. On the
available Windows host, TorchCodec could not load because the required shared
FFmpeg libraries were absent, and the upstream model requires CUDA with
FlashAttention 2. DeafBench therefore does not claim that candidate upgrade is
compatible merely because its Python imports succeeded.

The separate Open-ASR Zipformer lane remains ABI-locked to Torch 2.6,
Torchaudio 2.6, and its Torch-2.6-specific k2 wheel. The exact pinned evaluation
entrypoint and imported model path do not call either affected API. The Icefall
checkout does contain `torch.jit.script` calls in export and conversion tools,
but those tools are not imported or executed by the evaluation runner. This is
recorded as a separate reachability assessment, not represented as a patched
dependency.

The machine-readable record in
`deafbench/dependency-risk-dispositions.json` expires on 2026-11-13. Tests fail
if the advisory identity, affected API, dependency versions, model and external
source revisions, audited source hashes, isolation assumptions, or review
deadline drift. Granite NAR and Open-ASR execution both fail closed after that
deadline. The disposition must be reviewed sooner if upstream publishes a
supported matched stack or any reachability assumption changes.

References:

- [Granite Speech 4.1 2B NAR model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
- [TorchCodec compatibility table](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)
- [Torchaudio compatibility guidance](https://docs.pytorch.org/audio/stable/installation.html)

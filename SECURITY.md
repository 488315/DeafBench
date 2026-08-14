# Security

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's security
advisory interface. Do not include customer audio, transcripts, credentials,
or other sensitive content in a report.

## Granite NAR Torch advisories

DeafBench's optional `granite-nar-asr` extra pins the dependency versions used
by the upstream Granite Speech 4.1 2B NAR example: Torch 2.9.1, Torchaudio
2.9.1, and TorchCodec 0.9.1. Dependabot alerts #15 and #16 identify two
low-severity, local-attack memory-corruption advisories in that optional
development dependency:

- `GHSA-qfhq-4f3w-5fph` / `CVE-2025-3001` affects `torch.lstm_cell`.
- `GHSA-rrmf-rvhw-rf47` / `CVE-2025-3000` affects `torch.jit.script`.

Neither affected API appears in DeafBench production code or in the exact
hash-audited remote source for the pinned Granite NAR revision. Granite NAR
also runs outside the main process, with network access disabled during
inference and exact reviewed-file hashes required before execution.

This is a temporary tolerable-risk disposition, not a claim that the
vulnerabilities are fixed. The second advisory requires Torch 2.13.0, while
the official Torchaudio packages and compatibility table do not yet provide a
matching Torchaudio 2.13 release. DeafBench therefore retains the matched
upstream stack instead of installing an unsupported audio ABI combination.

The machine-readable record in
`deafbench/dependency-risk-dispositions.json` expires on 2026-11-13. Tests fail
if the advisory identity, affected API, dependency versions, model revision,
audited source hashes, isolation assumptions, or review deadline drift. The
disposition must be reviewed sooner if upstream publishes a supported matched
stack or any reachability assumption changes.

References:

- [Granite Speech 4.1 2B NAR model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar)
- [TorchCodec compatibility table](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)
- [Torchaudio compatibility guidance](https://docs.pytorch.org/audio/stable/installation.html)

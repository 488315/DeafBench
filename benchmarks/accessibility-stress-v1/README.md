# Accessibility Stress v1

This corpus predeclares paired clean and stressed cases for testing whether an
ASR model preserves information that a caption user may need to act on. The
reference set is byte-frozen and contains 24 synthetic utterances covering
times, dates, dosages, names, usernames, codes, digit sequences, SSIDs,
addresses, money, and negation. It does not contain generated audio, model
predictions, or benchmark scores.

Each case starts with a clean control and declares one acoustic or behavioral
stress condition. The predeclared families are additive noise at fixed SNR,
noise-only interstitials, 8 kHz mu-law telephony, deterministic reverberation,
long intra-phrase pauses, rate variation, overlapping speech, and low-bitrate
codec round trips. Generated audio and run artifacts must remain untracked.

The current in-process helpers implement additive noise, interstitial noise,
telephony, reverberation, long pauses, and rate variation. Overlap requires a
separately authorized speech signal, and codec cases require a recorded codec
round trip. A runner must fail closed rather than score either case without
that evidence.

## What to report

Report the clean and stressed lanes separately, then report the change between
them. The minimum result is:

- WER plus substitutions, insertions, and deletions;
- deletion share of all word edits;
- strict lexical and canonical critical-information recall;
- critical failures grouped by the declared risk category;
- lexical hallucinations in known noise-only intervals;
- median and maximum token timestamp drift, including the count over 500 ms;
- local RTFx, median and p95 latency, TTFB, throughput, peak CPU, and peak VRAM
  when a declared load trial was actually run.

Thresholds such as 10% WER, 500 ms TTFB, or RTF below 1.0 are application
requirements, not universal evidence that a model is accessible. A report must
predeclare its acceptance thresholds and retain the raw aggregate counts.

## Scope boundary

These transformations are synthetic acoustic and behavioral proxies. They do
not represent Deaf, hard-of-hearing, dysarthric, accented, age-based, or any
other demographic speech, and a passing result is not evidence of equitable
performance for those speakers. Rate changes, long pauses, and reverberation
must not be labelled as simulated Deaf speech.

A future human-speech lane needs authorized and appropriately licensed audio,
informed consent for the intended evaluation, disjoint development and final
evaluation cohorts, documented subgroup definitions, minimum sample sizes, and
subgroup uncertainty reporting. DeafBench must not publish or compare subgroup
results until those conditions are met. No real customer or participant audio
belongs in this repository.

## Reproducibility

The frozen reference SHA-256 is
`b47adac789092a1b8094c450340b178d64a06cccaf124893e8c4c7622ae73b61`.
The loader in `deafbench.benchmark.stress_contract` rejects unsupported fields,
stressors, SNR levels, and risk categories. Implemented acoustic transforms
live in small modules under `deafbench.benchmark`; the existing interstitial
evaluator remains the authority for noise-only hallucination scoring.

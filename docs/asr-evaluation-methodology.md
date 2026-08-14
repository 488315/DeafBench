# ASR evaluation methodology

DeafBench reports conventional transcription accuracy and accessibility
quality as separate metric families. A lower WER or CER does not imply that a
caption preserved a time, dosage, negation, name, username, code, SSID, speaker,
or environmental event.

## Conventional transcription metrics

The default DeafBench evaluator reports these corpus-level measurements:

| Field | Definition |
| --- | --- |
| Orthographic WER | Case- and punctuation-sensitive word error rate after whitespace collapse. |
| Orthographic CER | Case- and punctuation-sensitive character error rate after whitespace collapse. Spaces count as characters. |
| Normalized WER | Word error rate after `deafbench-asr-normalization-v1`. |
| Normalized CER | Character error rate after `deafbench-asr-normalization-v1`. Spaces count as characters. |
| Substitutions, insertions, deletions | Orthographic word-alignment counts retained by the compatibility fields. Normalized counts are separately named. |
| Local RTFx | Total audio seconds divided by inference wall seconds. Higher is faster. |

WER, CER, and edit counts are accumulated across the corpus rather than
averaging per-sample percentages. Per-sample orthographic WER and edit counts
remain available for error analysis.

Non-lexical records are excluded from WER and CER so sound-only accessibility
samples remain evaluable. If a corpus contains no lexical reference records,
conventional transcription metrics are reported as unavailable; non-speech and
other accessibility metrics continue to be evaluated.

Local RTFx is environment-dependent. A local RTX 4070 or CPU measurement is not
directly comparable with an official leaderboard hardware result unless the
hardware, runtime, batching, preprocessing, warm-up, and timing boundaries are
the same.

## `deafbench-asr-normalization-v1`

The normalized view applies these operations in order:

1. Unicode NFKC normalization.
2. Unicode case folding.
3. Replacement of Unicode punctuation and symbol characters with spaces.
4. Whitespace collapse and trimming.

The policy does not expand abbreviations, remove fillers, convert spoken numbers
to digits, rewrite dates or times, merge words, or apply the typed
critical-entity normalizer. For example, `eight` and `8` remain different.
References that become empty under this policy fail closed instead of silently
leaving the scoring denominator.

Changing any operation requires a new policy identifier. Historical outputs
retain their recorded policy so a normalization change cannot be presented as
a model improvement.

## Accessibility metrics

Strict lexical and typed canonical critical-information recall remain separate
from WER and CER. Typed canonical scoring permits only the representation
changes declared for each entity type. Non-speech recall, speaker attribution,
caption timing, interstitial hallucinations, and accessibility stress failures
remain separate outputs when the selected benchmark provides those labels.

## Hugging Face compatibility boundary

The Open ASR Leaderboard compatibility lane uses its pinned upstream English
normalizer, preprocessing, alignment, dataset aggregation, and RTFx contract.
It does not substitute `deafbench-asr-normalization-v1` or DeafBench's typed
critical-information evaluator. Results from the two lanes must stay labeled
separately.

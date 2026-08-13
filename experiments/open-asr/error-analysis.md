# Zipformer baseline error analysis

Status: all seven public test sets are complete and analyzed. Private scripted
and conversational sets remain unavailable without an authorized submission.

## Evidence

The pinned official scorer measured 1.73% WER on the first two
`librispeech/test.clean` rows: 0 deletions, 0 insertions, and 3 substitutions.
The two rows contain 68.865 seconds of audio. This tiny sample is suitable for
pipeline validation only and must not be used to claim model quality.

Observed substitutions:

- `guests` -> `guest`: singular/plural inflection.
- `Creighton` -> `Crichton` twice: a name or rare-word error.

`tonight` -> `to night` does not count after the official compound-boundary
normalization. This is why the runner's convenience WER (2.89%) differs from
the official scorer (1.73%).

## Requested categories

| Category | Diagnostic evidence | Next validation priority |
|---|---|---|
| Meetings and overlapping speech | AMI-Cleaned is the hardest completed set at 10.30 WER | High; classify utterance errors before choosing a change |
| Conversational speech | AMI and GigaSpeech contain conversational segments, but the private conversational set is unavailable | Develop only on matching train/validation data |
| Accents | VoxPopuli scored 4.31 WER; retained errors include short function-word and inflection differences | Validate any accent intervention off the public test set |
| Financial terminology | Earnings22 scored 7.68 WER and retained rows contain domain wording and numbers | High; classify utterance errors before choosing a change |
| Names and rare words | `Creighton` -> `Crichton` twice | High diagnostic signal |
| Numbers and abbreviations | GigaSpeech and SPGISpeech retained errors include numeric and domain-term confusions | Build train/validation slices before changing decoding |
| Disfluencies | AMI-Cleaned aggregate errors do not isolate disfluencies | Await utterance-level alignment |
| Long-form segmentation | Rows are about 34 seconds; no segmentation failure | Reassess across AMI/Earnings22 |
| Substitutions | 3 | Current observed error type |
| Insertions | 0 | No diagnostic evidence |
| Deletions | 0 | No diagnostic evidence |

No improvement experiment should be selected from two rows. Rank opportunities
only after the seven public manifests are complete and analyzed, then develop
against training/validation data rather than repeatedly tuning on test labels.

## Five-set utterance diagnostics

`results/zipformer-public-5set-errors.json` applies the pinned official
normalizer and compound-aware alignment to every completed row. Its aggregate
deletion, insertion, and substitution counts exactly reproduce the official
five-set scorer, while retaining only the 20 highest-error rows per dataset.

The highest-error AMI rows dominate current diagnostic error mass. Several
have hypotheses that are semantically unrelated to very short references and
WER above 100%, which is consistent with competing speech, reference/channel
alignment, or segmentation effects but does not distinguish among them without
audio review. Earnings22 rows 67, 69, and 942 contain fluent hypotheses that
materially disagree with the supplied reference wording, so label quality and
segment alignment must be checked before treating every difference as a model
error. LibriSpeech-other's largest errors visibly concentrate dialect spelling
and pronunciation; LibriSpeech-clean includes archaic wording, names, and
number renderings. VoxPopuli's largest rows are mostly shorter function-word,
inflection, and title differences.

AMI, GigaSpeech, and Earnings22 are the first diagnostic targets because they
have the three highest public-set WERs. That is a prioritization signal, not
permission to tune on public test labels: candidate changes must be selected
and measured on corresponding training or validation data.

## Seven-set utterance diagnostics

`results/zipformer-public-7set-errors.json` analyzes every officially
evaluable row and exactly reproduces the deletion, insertion, and substitution
totals in the complete scorer artifact. SPGISpeech adds 39,341 rows with 3,216
deletions, 3,620 insertions, and 9,083 substitutions for **1.64% official
WER**. Its largest retained errors include fluent near-matches, reference and
hypothesis boundary differences, domain terms, and numeric confusions such as
`8000` versus `800`; these observations do not establish a single cause.

Across the complete public suite, AMI-Cleaned (10.30), GigaSpeech-Cleaned
(8.33), and Earnings22 (7.68) are the highest-WER domains. The next legitimate
improvement work should therefore construct meeting, broad conversational,
and financial validation slices from permitted non-test data, then compare one
bounded intervention at a time. Public test labels must remain evaluation-only.

## Six-set utterance diagnostics

GigaSpeech-Cleaned adds 18,757 analyzed rows. Its aggregate 7,332 deletions,
7,212 insertions, and 18,526 substitutions exactly reproduce the official
scorer's **8.33% WER**. The largest retained row pairs reinforce the need for
data review before model changes: row 18,143 has 1.338 seconds of audio but a
50-word reference, and row 505 has a fluent hypothesis substantially longer
than its supplied reference. Other high-error rows contain noisy reference
wording, conversational profanity, financial language, esports names, and
numbers.

These observations are diagnostic evidence, not causal labels. Audio review
and corresponding train/validation splits are required to distinguish model
errors from truncation, segmentation, or transcription noise. AMI remains the
highest-WER completed set, followed by GigaSpeech-Cleaned and Earnings22.

## Full LibriSpeech test-clean baseline

The pinned official scorer measured **1.31% WER** over all 2,620 rows and
19,452.481 seconds of audio: 60 deletions, 59 insertions, and 575
substitutions. The same manifest's runner convenience metric was 1.67%, which
confirms that only the official normalized score should be used for leaderboard
comparisons. This clean-read-speech result does not address meetings, accents,
financial speech, or conversational private tests.

## Full VoxPopuli baseline

The official scorer measured **4.31% WER** over all 628 rows and 7,122.378
seconds of audio: 295 deletions, 182 insertions, and 289 substitutions. This
accented parliamentary-speech set is materially harder than LibriSpeech clean
for the baseline. Detailed error categories still require normalized
utterance-level alignment rather than inference from aggregate counts.

## Full Earnings22 baseline

The official loader retained **2,737 of 2,741 raw rows** after removing four
references that normalized to empty or its ignore sentinel. The official
scorer measured **7.68% WER** over 19,531.592 seconds of audio: 1,015
deletions, 986 insertions, and 1,736 substitutions. At 86.5395 RTFx, 464.446
seconds wall time, and 4,220,642,304 peak VRAM bytes, batch 16 remained safely
within the RTX 4070 constraint. Financial/long-form speech is now the largest
measured opportunity, but aggregate counts alone do not establish whether
terminology, names, numbers, segmentation, or acoustic conditions are causal.

## Full LibriSpeech test-other baseline

The official scorer measured **3.01% WER** over all 2,939 rows and 19,229.570
seconds of audio: 103 deletions, 144 insertions, and 1,343 substitutions. This
is 1.70 WER points worse than test-clean under the same pinned model and
decoder. Substitutions account for most of the gap; utterance-level alignment
is still required before attributing it to acoustics, names, or rare words.

## Full AMI-Cleaned baseline

The official loader retained **7,715 of 7,805 raw rows** after removing 90
references that normalized to empty or its ignore sentinel. The official
scorer measured **10.30% WER** over 28,727.590 seconds of audio: 2,506
deletions, 2,117 insertions, and 3,856 substitutions. At 89.4465 RTFx,
566.743 seconds wall time, and 2,829,196,800 peak VRAM bytes, batch 16 stayed
within the RTX 4070 constraint. Meeting speech is now the largest measured
opportunity, but overlap, disfluency, names, segmentation, and acoustic
conditions require utterance-level evidence before selecting an intervention.

## Full GigaSpeech-Cleaned baseline

The official loader retained **18,757 of 18,768 raw rows** after removing 11
references that normalized to empty or its ignore sentinel. The official
scorer measured **8.33% WER** over 126,515.196 seconds of audio: 7,332
deletions, 7,212 insertions, and 18,526 substitutions. At 99.2171 RTFx,
2,025.625 seconds wall time, and 2,567,505,408 peak VRAM bytes, batch 16 stayed
within the RTX 4070 constraint. Two identical official scoring runs produced
SHA-256 `10FE1A88D5AEBAA6B968D4B937CEBBE4BF9896C562D72A9063FEC1856686FFB6`.

## Full SPGISpeech baseline

The official scorer measured **1.64% WER** over all 39,341 rows and
360,007.500 seconds of audio: 3,216 deletions, 3,620 insertions, and 9,083
substitutions. At 100.6914 RTFx, 5,451.722 seconds wall time, and
3,062,188,544 peak VRAM bytes, batch 32 stayed within the RTX 4070 constraint.
The runner convenience metric was 1.81% and is not the leaderboard score. Two
identical complete seven-set scoring runs produced SHA-256
`634398F9F58415214F82720933B648AE135AFA28727B00FFC3AB401B50DDCA0E`.

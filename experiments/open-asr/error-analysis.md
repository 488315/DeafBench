# Zipformer baseline error analysis

Status: three of seven public test sets are complete. Cross-domain analysis
remains incomplete pending the other four public manifests.

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

| Category | Diagnostic evidence | Priority before full baseline |
|---|---|---|
| Meetings and overlapping speech | Not represented | Await AMI-Cleaned results |
| Conversational speech | Not represented | Await public and private confirmation |
| Accents | Not identified | Await per-dataset errors |
| Financial terminology | Earnings22 is the hardest completed set at 7.68 WER | High; classify utterance errors before choosing a change |
| Names and rare words | `Creighton` -> `Crichton` twice | High diagnostic signal |
| Numbers and abbreviations | Not represented | Await full results |
| Disfluencies | Not represented | Await AMI-Cleaned results |
| Long-form segmentation | Rows are about 34 seconds; no segmentation failure | Reassess across AMI/Earnings22 |
| Substitutions | 3 | Current observed error type |
| Insertions | 0 | No diagnostic evidence |
| Deletions | 0 | No diagnostic evidence |

No improvement experiment should be selected from two rows. Rank opportunities
only after the seven public manifests are complete and analyzed, then develop
against training/validation data rather than repeatedly tuning on test labels.

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

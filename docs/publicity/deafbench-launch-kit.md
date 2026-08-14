# DeafBench launch kit

This file contains reusable, evidence-based copy for sharing DeafBench. It is
not a record of posts already published, community approval, independent
validation, adoption, or a Hugging Face verified leaderboard result.

## The point in one paragraph

I wrote DeafBench because I am Deaf, I use cochlear implants, and a transcript
can have a low word error rate while still failing as an accessible caption.
DeafBench keeps WER, but it also measures whether a model preserves times,
names, usernames, codes, digit sequences, Wi-Fi names, negation, and expected
sound events. It keeps synthetic accessibility evidence, small real-speech
smoke tests, local performance, and Open ASR compatible results in separate
lanes instead of combining unlike numbers.

## The strongest public demonstration

On DeafBench's 12-sample synthetic demonstration, OpenAI Whisper `turbo`
recorded **2.0% speech WER** while matching **0 of 19 expected sound events**.
The speech transcript looked accurate, but the captions omitted every declared
alarm, siren, knock, door close, phone ring, and error-notification event in
that lane. The frozen source is
[`benchmarks/non-speech-v1/model-a-report.md`](../../benchmarks/non-speech-v1/model-a-report.md).

The correct conclusion is narrow: WER alone did not describe caption
completeness in this 12-sample synthetic demonstration. This is not a
demographic fairness study, a general ranking of Whisper, or evidence about all
audio and captioning systems.

## Facts that are ready to share

- The current stable package is `deafbench==0.2.1` on
  [PyPI](https://pypi.org/project/deafbench/).
- The source is Apache-2.0 and supports Python 3.11 through 3.14.
- Eleven adapters are integrated, but their results remain separated by
  benchmark lane and missing evidence is not filled in.
- Synthetic-v2 is a frozen 25-sample accessibility-critical benchmark with
  strict lexical and canonical semantic critical-information recall.
- The public Open ASR compatibility result is 5.23% seven-set macro WER for a
  pinned Zipformer reproduction. It is local public-contract evidence, not a
  Hugging Face verified leaderboard result, and the CC-BY-NC-4.0 checkpoint is
  research-only in DeafBench's license registry.
- The [metadata-only Hugging Face repository](https://huggingface.co/datasets/kvjones0243/deafbench-synthetic-v2)
  publishes authorized references and reproducibility manifests. It does not
  redistribute audio, predictions, run artifacts, or model weights.
- The [public project page](https://488315.github.io/products/deafbench/)
  explains the evidence lanes and links to the code and PyPI package.

The current `main` branch contains separately named orthographic WER,
normalized WER, CER, edit counts, timing, and accessibility metrics that merged
after v0.2.1. Do not imply those next-release fields are already present in the
PyPI wheel.

## Short technical post

> Low WER does not necessarily mean complete accessible captions.
>
> I built DeafBench to measure the information a conventional ASR score can
> hide. In its 12-sample synthetic non-speech demonstration, Whisper `turbo`
> recorded 2.0% speech WER while matching 0 of 19 expected sound events. That
> does not rank the model generally; it shows why caption evaluation needs
> speech accuracy and accessibility information reported separately.
>
> DeafBench is Apache-2.0, supports Python 3.11-3.14, and has a stable v0.2.1
> package on PyPI. The code, frozen evidence, methodology, project page, and
> metadata-only Hugging Face card are public.
>
> Code: https://github.com/488315/DeafBench  
> Demo: https://488315.github.io/products/deafbench/  
> Install: `python -m pip install "deafbench==0.2.1"`

## Accessibility-community version

> I am Deaf and use cochlear implants. I wrote DeafBench because captions can
> look mostly correct while losing the part I needed: a time, a name, a code,
> the word “not,” or an alarm in the room. The project scores those failures
> separately from ordinary word error rate and keeps the evidence boundaries
> visible. The public data is still small and synthetic, so I am sharing this
> as an open evaluation tool and a reproducible demonstration, not as a claim
> that it represents every Deaf or hard-of-hearing person.

## ASR and ML community version

> DeafBench is an ASR evaluation harness that keeps orthographic/normalized
> WER, CER, edit counts, critical-entity recall, non-speech event recall,
> timing, and local performance as separately named metrics. Typed canonical
> scoring accepts harmless time and digit representations without fuzzy
> matching usernames, codes, SSIDs, or different proper names. The repository
> also pins the public Open ASR normalization and macro-average contract, but
> does not call a local reproduction a leaderboard win.

## Open-source community version

> DeafBench 0.2.1 is available on PyPI under Apache-2.0. The repository has a
> Python 3.11-3.14 CI matrix, clean-wheel installation, frozen-manifest checks,
> source-control scanning, model license metadata, and isolated optional model
> dependencies. Contributions that add reproducible evidence, improve
> accessibility failure analysis, or strengthen benchmark integrity are more
> useful than adding unsupported scores.

## Professional-network version

> I built DeafBench to answer a practical captioning question that ordinary
> WER does not answer: did the transcript preserve the information a Deaf or
> hard-of-hearing person needed? The project now evaluates eleven ASR adapters,
> separates accessibility, real-speech smoke, performance, and Open ASR lanes,
> and publishes its evidence boundaries instead of turning every number into a
> ranking. The stable package is on PyPI, the source and methodology are on
> GitHub, and synthetic-v2 metadata is on Hugging Face.

## Research-community note

> DeafBench should currently be treated as an engineering benchmark and public
> research artifact, not a validated demographic study. Its strongest current
> use is demonstrating metric disagreement and providing reproducible typed
> critical-information scoring. A future study would need authorized natural
> speech, demographic design, consent, redistribution terms, and independent
> review before making population-level claims.

## Shareable links

- Repository: https://github.com/488315/DeafBench
- PyPI: https://pypi.org/project/deafbench/
- Website: https://488315.github.io/products/deafbench/
- Hugging Face: https://huggingface.co/datasets/kvjones0243/deafbench-synthetic-v2
- Methodology: https://github.com/488315/DeafBench/blob/main/docs/asr-evaluation-methodology.md
- Frozen non-speech report: https://github.com/488315/DeafBench/blob/main/benchmarks/non-speech-v1/model-a-report.md
- Citation metadata: https://github.com/488315/DeafBench/blob/main/CITATION.cff

## Visual and alt text

Use the existing DeafBench social card from
`https://488315.github.io/assets/images/deafbench-social-card.png`.

Suggested alt text:

> DeafBench, accessibility-critical ASR evaluation, with WER, critical recall,
> and latency beside a transcript-alignment waveform.

Do not turn the 2.0%/0-of-19 demonstration into an unlabeled model ranking.
Keep the sample count, synthetic scope, and separate metric names in the image
caption or adjacent text.

## Publication checks

Before reusing any draft:

1. Verify the linked report and package version are still current.
2. Keep the dataset, sample count, model, and metric name beside every number.
3. State whether evidence is synthetic, smoke-test, local, or official.
4. Keep local hardware results separate from official leaderboard hardware.
5. Do not claim a leaderboard win until Hugging Face verifies one.
6. Do not claim certification, clinical validity, demographic fairness,
   customer adoption, endorsements, or research acceptance.
7. Do not imply model weights belong to DeafBench or that every integrated
   model has complete results.
8. Share one relevant post per community and follow that community's rules;
   do not cross-post the same message repeatedly.

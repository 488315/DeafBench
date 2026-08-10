# Hugging Face Open ASR Leaderboard: public contract snapshot

**Status:** researched public contract, not an executed reproduction.
**Observed:** 2026-08-10.
**Scope:** the current public English short-form contract for soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M. This supports DeafBench comparison design; it is not a claim that the leaderboard measures caption accessibility.

## Source pins and evidence quality

All links are first-party Hugging Face or Hugging Face-maintained source repositories. “Verified” means directly present at the linked revision; “inference” is a constrained conclusion; “unknown” is not established by public sources.

| Artifact | Pinned public revision observed | Establishes |
| --- | --- | --- |
| Leaderboard code | [9585fc39bff55697a2ec1c5f13921b18812bfde8](https://github.com/huggingface/open_asr_leaderboard/tree/9585fc39bff55697a2ec1c5f13921b18812bfde8) | Runner, normalizer, job script, contributor contract. |
| Official Zipformer Space | [64c698c42932a54bc7a40a7f172d03c8c4838fe6](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-zipformer/tree/64c698c42932a54bc7a40a7f172d03c8c4838fe6) | Docker environment and runner used by the job script. |
| Main short-form dataset | [b6bdcd0beb34f8975dc659796176d88f43aff502](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/tree/b6bdcd0beb34f8975dc659796176d88f43aff502) | Parquet configs, splits, fields, declared tasks. |
| Model repository | [d410fb15a71cbf87ec5e0a860356563deb9d8f01](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M/tree/d410fb15a71cbf87ec5e0a860356563deb9d8f01) | Model card, configuration, published result metadata. |

Dataset and model SHAs came from the public Hub API at observation time; the linked revisions preserve the substantive source material.

## Verified public evaluation set

The model-specific [launcher](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/soundsgoodai/run_zipformer.sh) and [Jobs script](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/soundsgoodai/submit_jobs.sh) select seven configurations from hf-audio/open-asr-leaderboard, all with max_eval_samples=-1:

| Current scorer label | Dataset config | Split | Examples at dataset pin |
| --- | --- | --- | ---: |
| AMI-Cleaned WER | ami_cleaned | test | 7,805 |
| Earnings22 WER | earnings22 | test | 2,741 |
| GigaSpeech-Cleaned WER | gigaspeech_cleaned | test | 18,768 |
| LS Clean WER | librispeech | test.clean | 2,620 |
| LS Other WER | librispeech | test.other | 2,939 |
| SPGISpeech WER | spgispeech | test | 39,341 |
| Voxpopuli-Cleaned-AA WER | voxpopuli_cleaned_aa | test | 628 |

**Verified:** every listed config has audio (declared 16 kHz), dataset, text, id, and audio_length_s in the [pinned dataset card](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/blob/b6bdcd0beb34f8975dc659796176d88f43aff502/README.md). The [published evaluation YAML](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M/blob/d410fb15a71cbf87ec5e0a860356563deb9d8f01/.eval_results/open_asr_leaderboard.yaml) reports 5.56 mean WER and the seven per-set WERs dated 2026-07-19.

## Score-to-beat correction

At the pinned evaluator revision, `scripts/data/en_shortform.csv` lists the
lowest public seven-set average as **5.42** for
`CohereLabs/cohere-transcribe-03-2026`. The requested **5.37** is the
VoxPopuli column for `zoom/scribe_v1`, whose public average is 5.47; it is not
an Average WER in this snapshot. DeafBench therefore records 5.42 as the
pinned public comparison and retains 5.37 only as an unverified external
target. Neither value proves success on the unavailable private scripted and
conversational evaluations.

**Important distinction:** the dataset [eval.yaml](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/blob/b6bdcd0beb34f8975dc659796176d88f43aff502/eval.yaml) also lists older/un-cleaned task identifiers and Common Voice/TED-LIUM. For this model’s published result, the executable model-specific scripts are stronger operational authority for the seven-set aggregate.

## Verified runner, preprocessing, and decoding

1. [run_eval.py](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/soundsgoodai/run_eval.py) loads streaming by default, filters rows whose normalized reference is empty or exactly "ignore time segment in scoring", and retains raw reference/prediction text. Loader/filter are in [data_utils.py](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/normalizer/data_utils.py).
2. It snapshots only model .pt, .model, and .yaml files, loads the checkpoint strictly, and uses the configuration’s SentencePiece tokenizer and modified_beam_search. The [pinned model configuration](https://huggingface.co/soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M/blob/d410fb15a71cbf87ec5e0a860356563deb9d8f01/config.yaml) specifies beam **6**, 16 kHz fbank, 25 ms frames, 10 ms shift, 80 mel bins, 20–7600 Hz, no dithering, snip_edges false, and a non-causal transducer + CTC + CR-CTC model.
3. The runner requires 1-D mono float32 audio; if needed, it clips to [-1,1], converts to signed 16-bit PCM, resamples via audioop.ratecv, converts back, and extracts kaldi_native_fbank features. It pads sequences shorter than nine frames with log(1e-10) before batched encoder inference.
4. The launcher uses **batch size 64**, device 0, seven invocations, and one default warm-up batch per invocation. Warm-up is not written to results. Each item gets its batch wall time divided equally; RTFx is total audio seconds / total recorded inference seconds.

The official Space [Dockerfile](https://huggingface.co/spaces/hf-audio/open-asr-leaderboard-zipformer/blob/64c698c42932a54bc7a40a7f172d03c8c4838fe6/Dockerfile) pins CUDA 12.4.1, Python 3.12, PyTorch/torchaudio 2.4.0+cu124, datasets 3.6.0, and a K2 wheel, but clones Icefall shallowly without a commit SHA.

## Verified scoring

The runner first calculates WER through evaluate.load("wer"); final multi-manifest scoring is [score_results](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/normalizer/eval_utils.py). For English, raw reference and prediction pass through [EnglishTextNormalizer](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/normalizer/normalizer.py): lowercasing; removal of bracketed/parenthetical text and listed fillers; contraction/title expansion; punctuation/diacritic removal except numeric symbols; number, selected spelling/name, compound, and acronym normalization; whitespace collapse.

The final scorer whitespace-tokenizes and calls kaldialign.batch_error_rate with merge_compounds=true, rounds each WER to two percentage decimals, and reports the public average as the **unweighted arithmetic mean of the seven already-rounded dataset WERs**. It does not enable its optional multilingual compound-pair preprocessing for this English run. This is lexical ASR WER: it does not separately preserve casing, punctuation, speaker attribution, timing, segmentation quality, or accessibility-critical errors.

## Submission/reproduction contract

**Verified public path:** the job script launches one Hugging Face Job per model/dataset pair using hf-audio/open-asr-leaderboard-zipformer, H200 flavor, eight-hour timeout, batch size 64, and a bucket mounted at /results. It copies each JSONL manifest, syncs results locally, warns when fewer than seven JSONLs appear, then scores whatever manifests are present. The [contributor template](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/.github/PULL_REQUEST_TEMPLATE.md) requires raw un-normalized transcripts in manifests, common decoding parameters across datasets, batch support, data_utils use, maximum feasible H200 batch size, reported WER/RTFx, and model metadata.

**Verified manifest schema:** one JSON object per line: audio_filepath, duration, time, text, and pred_text, written by [write_manifest](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/normalizer/eval_utils.py).

## Licenses and access

- **Verified:** the model card declares **CC-BY-NC-4.0**; commercial use needs separate permission analysis.
- **Verified:** the aggregate dataset has no single declared repository license. Its [card](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/blob/b6bdcd0beb34f8975dc659796176d88f43aff502/README.md) identifies source-level terms: LibriSpeech and AMI CC-BY-4.0, Earnings-22 CC-BY-SA-4.0, GigaSpeech Apache-2.0, VoxPopuli CC0, and SPGISpeech a Kensho user agreement. It says GigaSpeech and SPGISpeech require accepting upstream terms. Independently verify terms before use or redistribution.

## Inferences, unknowns, and safeguards

- **Inference:** the 5.56 result is reproducible in method from public code, but is not fully bit-reproducible from the submission command alone: it does not pin Space, model, dataset, Icefall, or every transitive dependency revision.
- **Verified public gap:** no inspected source provides the seven submitted JSONLs, Job IDs/logs, exact H200 driver/runtime build, random seed, or checkpoint digest. Do not represent these as known.
- **Verified public gap:** audio is pre-segmented. The runner does not evaluate diarization, live streaming, turn segmentation, VAD, latency beyond aggregate RTFx, or a caption display.
- **Verified public gap:** the executable seven-set script contains no private data. The README says maintainers also use private sets, but their contents, protocol, scoring, and results were not public in sources inspected.
- **DeafBench safeguard:** retain raw references/outputs; pin every source and normalizer revision; report every corpus/split separately; and add semantic, timing, speaker, and critical-information measures rather than treating normalized average WER as an accessibility score.

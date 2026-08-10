<!-- markdownlint-disable MD013 -->

# Commercial model candidate review

Observed 2026-08-10. This is a licensing and integration screen, not a claim that any candidate is fit for the Open ASR Leaderboard or a production accessibility workflow. Recheck the pinned revision and obtain legal review before distribution or a commercial launch.

## Decision matrix

| Candidate | Pinned Hub revision | Model/card scope | Approximate artifact size | Runtime and remote code | Commercial-license implication |
| --- | --- | --- | ---: | --- | --- |
| [Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/tree/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c) | `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` | 782M BF16 parameters; 30 languages and 22 Chinese dialects on the card | 1.46 GiB `model.safetensors` | Native Transformers 5.13+ (`AutoProcessor`, `AutoModelForMultimodalLM`); the official example does not use `trust_remote_code` and this revision has no custom auto-map source | Apache-2.0 permits commercial use and redistribution, subject to its license, notices, and attribution requirements. No `LICENSE` or `NOTICE` file is listed in this revision; retain the Apache text and provenance when distributing. |
| [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/bcd2b5b7f32b480ab5790554cfa8347f246a14f3) | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` | 2.04B BF16 parameters; same 30-language/22-dialect family claim | 3.80 GiB `model.safetensors` | Native Transformers 5.13+; no remote-code invocation in the official example or custom auto-map source in this revision | Apache-2.0; preserve required licensing and any applicable notices. No repository `LICENSE` or `NOTICE` file is listed at this revision. |
| [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/tree/ae9ad07059c7c739ffaf932226a8fe64ae2620b0) | `ae9ad07059c7c739ffaf932226a8fe64ae2620b0` | 600M parameters; English, 16 kHz mono WAV/FLAC | 2.30 GiB `.nemo` checkpoint | NeMo (`nemo_toolkit[asr]`, `ASRModel.from_pretrained`); Transformers remote code is not applicable | CC-BY-4.0 allows commercial use, adaptation, and redistribution with appropriate credit, a license link, and change indication. No `NOTICE` file is listed. |
| [Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/tree/de575db64086f84fdc79da4932d1076e965bc546) | `de575db64086f84fdc79da4932d1076e965bc546` | 2.31B safetensors parameters; ASR for English, French, German, Spanish, Portuguese, and Japanese | 4.31 GiB model shard total | Native Transformers 4.52.1+ (`AutoProcessor`, `AutoModelForSpeechSeq2Seq`); official path has no `trust_remote_code` or custom auto-map source | Apache-2.0; preserve required licensing and any applicable notices. No repository `LICENSE` or `NOTICE` file is listed at this revision. |
| [Granite Speech 4.1 2B NAR](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/tree/a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2) | `a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2` | 2.25B safetensors parameters; ASR for English, French, German, Spanish, and Portuguese | 4.20 GiB `model.safetensors` | Requires Transformers 5.5.3+, PyTorch 2.9.1+, FlashAttention 2, and explicitly `trust_remote_code=True` for model and processor | Apache-2.0. The repository supplies custom Python, so pin this exact revision and review/allowlist that code before execution; preserve required licensing and notices. |
| [ARK-ASR-0.6B](https://huggingface.co/AutoArk-AI/ARK-ASR-0.6B/tree/45776b56d58cdfb2e2eb632f7e110f38684633e0) | `45776b56d58cdfb2e2eb632f7e110f38684633e0` | Card describes a 0.6B decoder plus a separate 0.6B-scale Whisper-style audio encoder/adapter; ASR in 19 listed languages | 2.42 GiB `model.safetensors` | Transformers/PyTorch; card explicitly requires `trust_remote_code=True` for model, processor, and tokenizer, and points to its upstream inference code | Apache-2.0. Treat the custom Python as a supply-chain boundary: pin, review, and allowlist before execution; preserve required licensing and notices. |

Artifact figures use the model file(s) at the pinned revision, not installed runtime size. Parameter counts are those exposed by the Hub metadata or card; they are not independently recomputed.

## Integration recommendation

Begin with **Qwen3-ASR-0.6B-hf** as the lowest-risk multilingual integration candidate: its card documents a native Transformers path without remote code, it has an Apache-2.0 license tag, and its artifact is materially smaller than the larger alternatives. Run it only on the predeclared real-speech development track first.

Use **Parakeet TDT 0.6B v2** as a separate English-only speed/accuracy comparison if its CC-BY attribution obligations are acceptable. Evaluate **Qwen3-ASR-1.7B-hf** only after the 0.6B integration is stable so a capacity increase is measured rather than conflated with integration changes.

Defer Granite NAR and ARK-ASR until a deliberate, pinned custom-code review has approved `trust_remote_code=True`. Granite Speech 4.1 2B is a viable native-Transformers multilingual candidate, but its substantially larger artifact makes it a later controlled comparison rather than the first smoke test.

None of these licensing statements establishes trademark, dataset, patent, privacy, export-control, hosting, or downstream-distribution clearance.

## First-party evidence

- Qwen 0.6B: [pinned card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/blob/7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c/README.md), [Hub metadata](https://huggingface.co/api/models/Qwen/Qwen3-ASR-0.6B-hf), and [Transformers Qwen3-ASR documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_asr).
- Qwen 1.7B: [pinned card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/blob/bcd2b5b7f32b480ab5790554cfa8347f246a14f3/README.md) and [Hub metadata](https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B-hf).
- Parakeet: [pinned card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/ae9ad07059c7c739ffaf932226a8fe64ae2620b0/README.md), [Hub metadata](https://huggingface.co/api/models/nvidia/parakeet-tdt-0.6b-v2), and [NeMo ASR model documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html).
- Granite 2B: [pinned card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/blob/de575db64086f84fdc79da4932d1076e965bc546/README.md) and [Hub metadata](https://huggingface.co/api/models/ibm-granite/granite-speech-4.1-2b).
- Granite NAR: [pinned card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/blob/a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2/README.md), [custom model source](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar/blob/a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2/modeling_granite_speech_nar.py), and [Hub metadata](https://huggingface.co/api/models/ibm-granite/granite-speech-4.1-2b-nar).
- ARK-ASR: [pinned card](https://huggingface.co/AutoArk-AI/ARK-ASR-0.6B/blob/45776b56d58cdfb2e2eb632f7e110f38684633e0/README.md), [custom model source](https://huggingface.co/AutoArk-AI/ARK-ASR-0.6B/blob/45776b56d58cdfb2e2eb632f7e110f38684633e0/modeling_arkasr.py), [official inference script](https://github.com/AutoArk/open-audio-opd/blob/main/scripts/infer/ark_asr_transformers.py), and [Hub metadata](https://huggingface.co/api/models/AutoArk-AI/ARK-ASR-0.6B).
- License texts: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) and [CC-BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en).

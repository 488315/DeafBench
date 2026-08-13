"""Pinned native-Transformers adapter for Granite Speech 4.1 2B."""

from __future__ import annotations

from collections.abc import Sequence
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl
from deafbench.model_registry import ModelRegistryError, get_model_license


MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
MODEL_NAME = "granite-speech-4.1-2b"
MAX_NEW_TOKENS = 200
SAMPLE_RATE = 16_000
DEFAULT_PROMPT = (
    "<|audio|>transcribe the speech with proper punctuation and capitalization."
)


@dataclass(frozen=True)
class _TransformersBackend:
    AutoModelForSpeechSeq2Seq: Any
    AutoProcessor: Any
    clock: Any
    numpy: Any
    resample_poly: Any
    torch: Any


def _load_backend() -> _TransformersBackend:
    try:
        import numpy
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "scipy", "torch", "transformers"}:
            raise
        raise RuntimeError(
            "Granite Speech is not installed. Run: "
            'python -m pip install "deafbench[granite-asr]"'
        ) from exc
    return _TransformersBackend(
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        perf_counter,
        numpy,
        resample_poly,
        torch,
    )


def _licensed_revision() -> str:
    model_license = get_model_license(MODEL_ID)
    if model_license.remote_code_required:
        raise ModelRegistryError("Granite adapter rejects remote code")
    if not any(
        runtime.startswith("transformers")
        for runtime in model_license.supported_runtimes
    ):
        raise ModelRegistryError(
            "Granite adapter requires a registered Transformers runtime"
        )
    return model_license.revision


def _transcription_prompt(keywords: Sequence[str]) -> str:
    if not keywords:
        return DEFAULT_PROMPT
    normalized: list[str] = []
    for keyword in keywords:
        value = keyword.strip()
        if (
            not value
            or len(value) > 80
            or any(character in value for character in ",\r\n")
        ):
            raise ValueError("Granite keywords must be plain nonempty text")
        normalized.append(value)
    if len(normalized) > 64 or len(set(normalized)) != len(normalized):
        raise ValueError("Granite keywords must be unique and limited to 64")
    return f"<|audio|>transcribe the speech to text. Keywords: {', '.join(normalized)}"


def _read_pcm16_mono(
    wav_path: Path,
    numpy: Any,
    resample_poly: Any,
) -> tuple[Any, float]:
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Granite requires mono PCM16 WAV: {wav_path}")
        source_rate = handle.getframerate()
        duration_seconds = handle.getnframes() / source_rate
        frames = handle.readframes(handle.getnframes())
    samples = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32)
    samples /= 32_768.0
    if source_rate == SAMPLE_RATE:
        return samples, duration_seconds
    divisor = gcd(source_rate, SAMPLE_RATE)
    return (
        resample_poly(
            samples,
            SAMPLE_RATE // divisor,
            source_rate // divisor,
        ).astype(numpy.float32, copy=False),
        duration_seconds,
    )


def _decode_transcription(tokenizer: Any, generated_ids: Any) -> str:
    decoded = tokenizer.batch_decode(
        generated_ids,
        add_special_tokens=False,
        skip_special_tokens=True,
    )
    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
        or not isinstance(decoded[0], str)
    ):
        raise ValueError("Invalid Granite transcription output")
    return decoded[0]


def run_granite_speech(
    audio_dir: Path,
    references: Path,
    output: Path,
    keywords: Sequence[str] = (),
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe a complete WAV set with pinned Granite Speech."""
    revision = _licensed_revision()
    prompt_text = _transcription_prompt(keywords)
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    use_cuda = runtime.torch.cuda.is_available()
    if use_cuda:
        runtime.torch.cuda.reset_peak_memory_stats()
    device = runtime.torch.device("cuda" if use_cuda else "cpu")
    dtype = runtime.torch.bfloat16 if use_cuda else runtime.torch.float32
    load_options = {"revision": revision, "trust_remote_code": False}
    processor = runtime.AutoProcessor.from_pretrained(MODEL_ID, **load_options)
    model = runtime.AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        **load_options,
    )
    model.to(device)
    model.eval()
    chat = [{"role": "user", "content": prompt_text}]
    prompt = processor.tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    total_audio_seconds = 0.0
    with runtime.torch.inference_mode():
        for wav_path in wav_paths:
            started = runtime.clock()
            audio, audio_seconds = _read_pcm16_mono(
                wav_path,
                runtime.numpy,
                runtime.resample_poly,
            )
            waveform = runtime.torch.from_numpy(audio).unsqueeze(0)
            inputs = processor(
                prompt,
                waveform,
                device=device,
                return_tensors="pt",
            ).to(device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[:, prompt_length:]
            text = _decode_transcription(processor.tokenizer, generated_ids)
            latency_ms = round((runtime.clock() - started) * 1_000.0, 6)
            latencies_ms.append(latency_ms)
            total_audio_seconds += audio_seconds
            records.append(
                {"id": wav_path.stem, "latency_ms": latency_ms, "text": text}
            )

    total_latency_seconds = sum(latencies_ms) / 1_000.0
    if total_latency_seconds <= 0:
        raise ValueError("Granite Speech timing must be positive")
    atomic_write_jsonl(output, records)
    return ModelRunInfo(
        name=MODEL_NAME,
        model_id=MODEL_ID,
        revision=revision,
        decoding={
            "device": str(device),
            "dtype": str(dtype),
            "keyword_biasing": bool(keywords),
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_beams": 1,
            "trust_remote_code": False,
        },
        performance={
            "local_rtfx": total_audio_seconds / total_latency_seconds,
            "median_latency_ms": median(latencies_ms),
            "peak_vram_bytes": (
                runtime.torch.cuda.max_memory_allocated() if use_cuda else None
            ),
            "timing_scope": "decode_only_excludes_model_load",
        },
    )

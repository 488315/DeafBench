"""Pinned native-Transformers adapter for the Qwen3-ASR family."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

from deafbench.benchmark.models import ModelRunInfo, _validated_wavs
from deafbench.benchmark.workspace import atomic_write_jsonl
from deafbench.model_registry import ModelRegistryError, get_model_license


DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
MAX_NEW_TOKENS = 256


@dataclass(frozen=True)
class _TransformersBackend:
    AutoProcessor: Any
    AutoModelForMultimodalLM: Any
    numpy: Any
    resample_poly: Any
    torch: Any


def _load_backend() -> _TransformersBackend:
    try:
        import numpy
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "scipy", "torch", "transformers"}:
            raise
        raise RuntimeError(
            "Qwen3-ASR is not installed. Run: "
            'python -m pip install "deafbench[qwen-asr]"'
        ) from exc
    return _TransformersBackend(
        AutoProcessor,
        AutoModelForMultimodalLM,
        numpy,
        resample_poly,
        torch,
    )


def _licensed_revision(model_id: str) -> str:
    model_license = get_model_license(model_id)
    if model_license.remote_code_required:
        raise ModelRegistryError(
            f"Qwen adapter rejects remote-code model: {model_id}"
        )
    if not any(
        runtime.startswith("transformers")
        for runtime in model_license.supported_runtimes
    ):
        raise ModelRegistryError(
            f"Qwen adapter requires a registered Transformers runtime: {model_id}"
        )
    return model_license.revision


def _decode_transcription(processor: Any, generated_ids: Any) -> str:
    decoded = processor.decode(
        generated_ids,
        return_format="transcription_only",
    )
    if (
        not isinstance(decoded, list)
        or len(decoded) != 1
        or not isinstance(decoded[0], str)
    ):
        raise ValueError("Invalid Qwen3-ASR transcription output")
    return decoded[0]


def _read_pcm16_mono(
    wav_path: Path,
    target_rate: int,
    numpy: Any,
    resample_poly: Any,
) -> Any:
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Qwen3-ASR requires mono PCM16 WAV: {wav_path}")
        source_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = numpy.frombuffer(frames, dtype="<i2").astype(numpy.float32)
    samples /= 32_768.0
    if source_rate == target_rate:
        return samples
    divisor = gcd(source_rate, target_rate)
    return resample_poly(
        samples,
        target_rate // divisor,
        source_rate // divisor,
    ).astype(numpy.float32, copy=False)


def run_qwen3_asr(
    audio_dir: Path,
    references: Path,
    output: Path,
    model_id: str = DEFAULT_MODEL,
    backend: Any | None = None,
) -> ModelRunInfo:
    """Transcribe a complete WAV set with a pinned native Qwen3-ASR model."""
    revision = _licensed_revision(model_id)
    wav_paths = _validated_wavs(audio_dir, references)
    runtime = _load_backend() if backend is None else backend
    load_options = {"revision": revision, "trust_remote_code": False}
    processor = runtime.AutoProcessor.from_pretrained(model_id, **load_options)
    sampling_rate = processor.feature_extractor.sampling_rate
    model = runtime.AutoModelForMultimodalLM.from_pretrained(
        model_id,
        **load_options,
    )
    target_device = runtime.torch.device(
        "cuda" if runtime.torch.cuda.is_available() else "cpu"
    )
    model.to(target_device)
    model.eval()

    records: list[dict[str, str]] = []
    with runtime.torch.inference_mode():
        for wav_path in wav_paths:
            inputs = processor.apply_transcription_request(
                audio=_read_pcm16_mono(
                    wav_path,
                    sampling_rate,
                    runtime.numpy,
                    runtime.resample_poly,
                ),
                language="English",
            ).to(model.device, model.dtype)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_length:]
            records.append(
                {
                    "id": wav_path.stem,
                    "text": _decode_transcription(processor, generated_ids),
                }
            )

    atomic_write_jsonl(output, records)
    return ModelRunInfo(
        name="qwen3-asr-0.6b",
        model_id=model_id,
        revision=revision,
        decoding={
            "device": str(model.device),
            "dtype": str(model.dtype),
            "language": "English",
            "max_new_tokens": MAX_NEW_TOKENS,
            "trust_remote_code": False,
        },
    )

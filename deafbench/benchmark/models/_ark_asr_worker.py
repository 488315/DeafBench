"""Child-process inference worker for audited ARK-ASR source."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

from deafbench.benchmark.models._audio_chunks import contiguous_audio_chunks
from deafbench.benchmark.models._isolated import RESULT_MARKER
from deafbench.remote_code_audit import load_remote_code_audit, verify_audited_files


MODEL_ID = "AutoArk-AI/ARK-ASR-0.6B"
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 30.0
PROMPT = "Please transcribe this audio."


@dataclass(frozen=True)
class _Backend:
    AutoModelForCausalLM: Any
    AutoProcessor: Any
    AutoTokenizer: Any
    clock: Any
    soundfile: Any
    torch: Any


def _load_backend() -> _Backend:
    try:
        import soundfile
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ARK-ASR is not installed. Run: "
            'python -m pip install "deafbench[ark-asr]"'
        ) from exc
    return _Backend(
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        perf_counter,
        soundfile,
        torch,
    )


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ARK-ASR request requires {field}")
    return value


def _validated_wav_paths(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    values = payload.get("wav_paths")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("ARK-ASR request requires WAV paths")
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("ARK-ASR request contains an invalid WAV path")
        path = Path(value)
        if not path.is_absolute() or path.suffix.lower() != ".wav":
            raise ValueError("ARK-ASR request contains an unsafe WAV path")
        paths.append(path.resolve(strict=True))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("ARK-ASR request requires unique WAV paths")
    return tuple(paths)


def _bad_words_ids(tokenizer: Any) -> list[list[int]]:
    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
    )
    return [[token_id] for token_id in sorted(bad_ids)]


def _conversation(wav_path: Path) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(wav_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def run_request(
    payload: Mapping[str, Any],
    backend: _Backend | None = None,
) -> dict[str, Any]:
    """Verify audited source and transcribe the requested local WAV files."""
    model_id = _required_text(payload, "model_id")
    revision = _required_text(payload, "revision")
    if model_id != MODEL_ID:
        raise ValueError(f"unexpected ARK-ASR model: {model_id}")
    audit = load_remote_code_audit(model_id)
    if revision != audit.revision:
        raise ValueError("ARK-ASR request revision differs from audit")
    snapshot_root = Path(_required_text(payload, "snapshot_root"))
    if not snapshot_root.is_absolute():
        raise ValueError("ARK-ASR snapshot must be absolute")
    snapshot_root = snapshot_root.resolve(strict=True)
    verify_audited_files(audit, snapshot_root)
    wav_paths = _validated_wav_paths(payload)

    runtime = _load_backend() if backend is None else backend
    if not runtime.torch.cuda.is_available():
        raise RuntimeError("ARK-ASR requires CUDA")
    runtime.torch.cuda.reset_peak_memory_stats()
    device = runtime.torch.device("cuda")
    dtype = runtime.torch.float16
    load_options = {"local_files_only": True, "trust_remote_code": True}
    processor = runtime.AutoProcessor.from_pretrained(
        str(snapshot_root),
        **load_options,
    )
    tokenizer = runtime.AutoTokenizer.from_pretrained(
        str(snapshot_root),
        **load_options,
    )
    model = runtime.AutoModelForCausalLM.from_pretrained(
        str(snapshot_root),
        attn_implementation="sdpa",
        dtype=dtype,
        **load_options,
    ).to(device)
    model.eval()
    bad_words_ids = _bad_words_ids(tokenizer)

    records: list[dict[str, object]] = []
    latencies_ms: list[float] = []
    total_audio_seconds = 0.0
    with runtime.torch.inference_mode():
        for wav_path in wav_paths:
            with contiguous_audio_chunks(
                wav_path,
                runtime.soundfile,
                max_audio_seconds=MAX_AUDIO_SECONDS,
                runtime_name="ARK-ASR",
            ) as (
                audio_seconds,
                chunk_paths,
            ):
                runtime.torch.cuda.synchronize()
                started = runtime.clock()
                chunk_texts: list[str] = []
                for chunk_path in chunk_paths:
                    inputs = processor.apply_chat_template(
                        _conversation(chunk_path),
                        add_generation_prompt=True,
                        return_tensors="pt",
                        sampling_rate=SAMPLE_RATE,
                        audio_padding="longest",
                        text_kwargs={"padding": "longest"},
                        audio_max_length=int(MAX_AUDIO_SECONDS * SAMPLE_RATE),
                    ).to(device)
                    if "audios" in inputs:
                        inputs["audios"] = inputs["audios"].to(dtype=dtype)
                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=256,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bad_words_ids=bad_words_ids,
                    )
                    decoded = tokenizer.batch_decode(
                        outputs[:, inputs.input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                    if (
                        not isinstance(decoded, list)
                        or len(decoded) != 1
                        or not isinstance(decoded[0], str)
                    ):
                        raise ValueError("ARK-ASR returned an invalid transcription")
                    chunk_texts.append(decoded[0].strip())
                runtime.torch.cuda.synchronize()
                latency_ms = round((runtime.clock() - started) * 1_000.0, 6)
            latencies_ms.append(latency_ms)
            total_audio_seconds += audio_seconds
            records.append(
                {
                    "id": wav_path.stem,
                    "latency_ms": latency_ms,
                    "text": " ".join(text for text in chunk_texts if text),
                }
            )

    total_latency_seconds = sum(latencies_ms) / 1_000.0
    if total_latency_seconds <= 0:
        raise ValueError("ARK-ASR timing must be positive")
    return {
        "decoding": {
            "attn_implementation": "sdpa",
            "audio_max_seconds": MAX_AUDIO_SECONDS,
            "device": str(device),
            "dtype": str(dtype),
            "do_sample": False,
            "max_new_tokens": 256,
            "long_audio_strategy": "contiguous_30_second_chunks_without_overlap",
            "prompt": PROMPT,
            "trust_remote_code": True,
        },
        "performance": {
            "local_rtfx": total_audio_seconds / total_latency_seconds,
            "median_latency_ms": median(latencies_ms),
            "peak_vram_bytes": runtime.torch.cuda.max_memory_allocated(),
            "timing_scope": "decode_only_excludes_model_load",
        },
        "records": records,
    }


def main() -> int:
    """Read one JSON request from stdin and emit one machine-readable result."""
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("ARK-ASR worker request must be an object")
    result = run_request(payload)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

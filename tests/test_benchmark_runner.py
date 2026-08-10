import json
import wave
from pathlib import Path

import numpy as np
import pytest

from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.runner import BenchmarkConfig, run_benchmark
from deafbench.benchmark.synthetic import (
    SpeechAudio,
    TTSInfo,
    generate_synthetic_set,
)
from deafbench.benchmark.workspace import atomic_write_jsonl


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00" * 32)


def _write_dataset(root: Path, *, human_complete: bool) -> Path:
    dataset_dir = root / "benchmarks" / "core-v1"
    dataset_dir.mkdir(parents=True)
    references = dataset_dir / "references.jsonl"
    records = [
        {
            "id": "core-001",
            "text": "Meet at platform four.",
            "critical": ["platform four"],
            "sounds": [],
        },
        {
            "id": "core-002",
            "text": "The alarm is active.",
            "critical": ["alarm"],
            "sounds": ["[alarm]"],
        },
    ]
    references.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    if human_complete:
        for record in records:
            _write_wav(dataset_dir / "audio" / f"{record['id']}.wav")
    return references


def _fake_speech(text: str) -> SpeechAudio:
    return SpeechAudio(
        np.full((24_000 + len(text),), 0.1, dtype=np.float64),
        24_000,
    )


def _fake_model_runner(
    _audio_dir: Path,
    references: Path,
    output: Path,
) -> ModelRunInfo:
    records = [
        {
            "id": record["id"],
            "text": record["text"],
            "sounds": record.get("sounds", []),
        }
        for record in (
            json.loads(line)
            for line in references.read_text(encoding="utf-8").splitlines()
        )
    ]
    atomic_write_jsonl(output, records)
    return ModelRunInfo("whisper", "test-model")


def _fake_synthetic_generator(
    references: Path,
    audio_dir: Path,
    _speech_generator: object,
    _tts_info: TTSInfo,
    scene_profile: str = "default-v1",
    seed: int = 42,
) -> Path:
    for line in references.read_text(encoding="utf-8").splitlines():
        _write_wav(audio_dir / f"{json.loads(line)['id']}.wav")
    manifest = audio_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"scene_profile": scene_profile, "seed": seed}) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_auto_uses_complete_human_set_without_synthetic_factory(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, human_complete=True)

    def fail_synthetic_factory() -> tuple[object, TTSInfo]:
        raise AssertionError("synthetic factory must not be used")

    result = run_benchmark(
        BenchmarkConfig(tmp_path, "core-v1", "whisper"),
        synthetic_factory=fail_synthetic_factory,
        whisper_runner=_fake_model_runner,
    )

    assert result.resolved_source == "human"
    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert "scene_profile" not in metadata
    assert "seed" not in metadata
    assert "tts" not in metadata


def test_auto_generates_synthetic_transactional_run(tmp_path: Path) -> None:
    _write_dataset(tmp_path, human_complete=False)
    generated: list[tuple[str, int]] = []

    def recording_generator(*args: object, **kwargs: object) -> Path:
        generated.append((str(kwargs["scene_profile"]), int(kwargs["seed"])))
        return _fake_synthetic_generator(*args, **kwargs)  # type: ignore[arg-type]

    result = run_benchmark(
        BenchmarkConfig(tmp_path, "core-v1", "whisper"),
        synthetic_factory=lambda: (
            _fake_speech,
            TTSInfo("whisperspeech", "test-version"),
        ),
        synthetic_generator=recording_generator,
        whisper_runner=_fake_model_runner,
    )

    run_root = (
        tmp_path
        / "benchmarks"
        / "core-v1"
        / "runs"
        / "whisper"
        / "synthetic"
    )
    assert result.resolved_source == "synthetic"
    assert generated == [("default-v1", 42)]
    assert result.predictions == run_root / "predictions.jsonl"
    assert result.report == run_root / "report.md"
    assert result.metadata == run_root / "run.json"
    assert result.metrics["samples"] == 2
    assert result.report.read_text(encoding="utf-8").startswith(
        "# DeafBench Evaluation Report"
    )
    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata == {
        "audio": str(tmp_path / "benchmarks/core-v1/audio-synthetic"),
        "audio_source": "synthetic",
        "benchmark_version": "0.1.1",
        "dataset": "core-v1",
        "model": "whisper",
        "model_id": "test-model",
        "predictions": str(result.predictions),
        "references": str(tmp_path / "benchmarks/core-v1/references.jsonl"),
        "report": str(result.report),
        "samples": 2,
        "scene_profile": "default-v1",
        "seed": 42,
        "tts": {"engine": "whisperspeech", "version": "test-version"},
    }


def test_current_synthetic_set_does_not_construct_whisperspeech(
    tmp_path: Path,
) -> None:
    references = _write_dataset(tmp_path, human_complete=False)
    audio_dir = references.parent / "audio-synthetic"
    generate_synthetic_set(
        references,
        audio_dir,
        _fake_speech,
        TTSInfo("whisperspeech", "persisted-test-version"),
    )

    def fail_synthetic_factory() -> tuple[object, TTSInfo]:
        raise AssertionError("current cache must not construct TTS")

    result = run_benchmark(
        BenchmarkConfig(tmp_path, "core-v1", "whisper"),
        synthetic_factory=fail_synthetic_factory,
        whisper_runner=_fake_model_runner,
    )

    assert result.resolved_source == "synthetic"
    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["tts"] == {
        "engine": "whisperspeech",
        "version": "persisted-test-version",
    }


def test_failed_rerun_preserves_every_previous_run_byte(tmp_path: Path) -> None:
    _write_dataset(tmp_path, human_complete=True)
    config = BenchmarkConfig(tmp_path, "core-v1", "whisper")
    first = run_benchmark(config, whisper_runner=_fake_model_runner)
    before = {
        path.relative_to(first.predictions.parent): path.read_bytes()
        for path in first.predictions.parent.rglob("*")
        if path.is_file()
    }

    def fail_model(*_args: object, **_kwargs: object) -> ModelRunInfo:
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        run_benchmark(config, whisper_runner=fail_model)

    after = {
        path.relative_to(first.predictions.parent): path.read_bytes()
        for path in first.predictions.parent.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_invalid_references_fail_before_audio_or_model_work(tmp_path: Path) -> None:
    references = _write_dataset(tmp_path, human_complete=False)
    references.write_text(
        '{"id":"../escape","text":"unsafe"}\n',
        encoding="utf-8",
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime work must not begin")

    with pytest.raises(ValueError, match="Invalid reference ID"):
        run_benchmark(
            BenchmarkConfig(tmp_path, "core-v1", "whisper"),
            synthetic_factory=fail,  # type: ignore[arg-type]
            whisper_runner=fail,  # type: ignore[arg-type]
        )

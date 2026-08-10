import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import deafbench.benchmark.runner as runner_module
from deafbench.benchmark.models import ModelRunInfo
from deafbench.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)
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


def _write_dataset(
    root: Path,
    *,
    human_complete: bool,
    dataset: str = "core-v1",
) -> Path:
    dataset_dir = root / "benchmarks" / dataset
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


def test_validated_v2_set_does_not_regenerate_synthetic_audio(tmp_path: Path) -> None:
    references = _write_dataset(
        tmp_path,
        human_complete=False,
        dataset="synthetic-v2",
    )
    audio_dir = references.parent / "audio-synthetic"
    generation = []
    accepted = []
    for line in references.read_text(encoding="utf-8").splitlines():
        sample_id = json.loads(line)["id"]
        wav = audio_dir / f"{sample_id}.wav"
        _write_wav(wav)
        generation.append(
            {
                "id": sample_id,
                "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                "replacement_reason": "test replacement",
            }
        )
        accepted.append({"id": sample_id, "status": "accepted"})
    atomic_write_jsonl(references.parent / "generation-manifest.jsonl", generation)
    (references.parent / "quality-report.json").write_text(
        json.dumps({"samples": accepted}),
        encoding="utf-8",
    )

    def fail_synthetic_factory() -> tuple[object, TTSInfo]:
        raise AssertionError("validated v2 audio must not be regenerated")

    result = run_benchmark(
        BenchmarkConfig(tmp_path, "synthetic-v2", "whisper"),
        synthetic_factory=fail_synthetic_factory,
        whisper_runner=_fake_model_runner,
    )

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["tts"]["engine"] == "validated-synthetic-v2"


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


def test_failed_run_promotion_restores_previous_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_dataset(tmp_path, human_complete=True)
    config = BenchmarkConfig(tmp_path, "core-v1", "whisper")
    first = run_benchmark(config, whisper_runner=_fake_model_runner)
    run_dir = first.predictions.parent
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    real_replace = os.replace

    def fail_staging_promotion(source: object, destination: object) -> None:
        source_path = Path(source)  # type: ignore[arg-type]
        if source_path.name.startswith(".human-run-"):
            raise OSError("promotion failed")
        real_replace(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(runner_module.os, "replace", fail_staging_promotion)

    with pytest.raises(OSError, match="promotion failed"):
        run_benchmark(config, whisper_runner=_fake_model_runner)

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("model", "function_name"),
    [
        ("whisper", "run_whisper"),
        ("whisper-at", "run_whisper_at"),
        ("faster-whisper", "run_faster_whisper"),
        ("distil-whisper", "run_distil_whisper"),
        ("qwen3-asr-0.6b", "run_qwen3_asr"),
    ],
)
def test_default_model_runner_is_selected_lazily(
    model: str,
    function_name: str,
) -> None:
    runner = runner_module._default_model_runner(model)  # type: ignore[arg-type]

    assert runner.__name__ == function_name


@pytest.mark.parametrize(
    "model", ["faster-whisper", "distil-whisper", "qwen3-asr-0.6b"]
)
def test_main_accepts_additional_local_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    result = BenchmarkResult(
        "human",
        tmp_path / "predictions.jsonl",
        tmp_path / "report.md",
        tmp_path / "run.json",
        {
            "samples": 1,
            "wer": 0.0,
            "critical_recall": 100.0,
            "non_speech_recall": None,
            "speaker_accuracy": None,
            "median_latency_ms": None,
            "critical_failures": [],
        },
    )
    calls: list[BenchmarkConfig] = []
    monkeypatch.setattr(
        runner_module,
        "run_benchmark",
        lambda config: calls.append(config) or result,
    )

    status = runner_module.main(
        ["core-v1", "--model", model, "--repo-root", str(tmp_path)]
    )

    assert status == 0
    assert calls == [BenchmarkConfig(tmp_path, "core-v1", model)]


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "unknown", "Unsupported benchmark model"),
        ("audio_source", "mixed", "Unsupported audio source"),
    ],
)
def test_direct_config_rejects_invalid_choices_before_workspace(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    values = {
        "repo_root": tmp_path,
        "dataset": "core-v1",
        "model": "whisper",
        "audio_source": "auto",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        run_benchmark(BenchmarkConfig(**values))  # type: ignore[arg-type]

    assert not (tmp_path / "benchmarks").exists()


def test_main_prints_run_identity_summary_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    result = BenchmarkResult(
        "human",
        run_dir / "predictions.jsonl",
        run_dir / "report.md",
        run_dir / "run.json",
        {
            "samples": 1,
            "wer": 0.0,
            "critical_recall": 100.0,
            "non_speech_recall": None,
            "speaker_accuracy": None,
            "median_latency_ms": None,
            "critical_failures": [],
        },
    )
    calls: list[BenchmarkConfig] = []
    monkeypatch.setattr(
        runner_module,
        "run_benchmark",
        lambda config: calls.append(config) or result,
    )

    status = runner_module.main(
        [
            "core-v1",
            "--model",
            "whisper",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert status == 0
    assert calls == [BenchmarkConfig(tmp_path, "core-v1", "whisper")]
    output = capsys.readouterr().out
    assert "Dataset: core-v1" in output
    assert "Model: whisper" in output
    assert "Audio source: human" in output
    assert f"Predictions: {result.predictions}" in output
    assert f"Report: {result.report}" in output


def test_help_does_not_import_optional_numpy_dependency() -> None:
    code = """
import builtins
original_import = builtins.__import__
def block_numpy(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError(name=name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_numpy
from deafbench.benchmark.runner import main
main(["--help"])
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Run a complete DeafBench model benchmark" in result.stdout


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_prediction_id_mismatch_preserves_previous_bundle(
    tmp_path: Path,
    mode: str,
) -> None:
    _write_dataset(tmp_path, human_complete=True)
    config = BenchmarkConfig(tmp_path, "core-v1", "whisper")
    first = run_benchmark(config, whisper_runner=_fake_model_runner)
    run_dir = first.predictions.parent
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    def mismatched_runner(
        _audio_dir: Path,
        _references: Path,
        output: Path,
    ) -> ModelRunInfo:
        records = [{"id": "core-001", "text": "partial"}]
        if mode == "extra":
            records.extend(
                [
                    {"id": "core-002", "text": "second"},
                    {"id": "extra", "text": "unexpected"},
                ]
            )
        atomic_write_jsonl(output, records)
        return ModelRunInfo("whisper", "broken-model")

    with pytest.raises(ValueError, match="Prediction IDs do not match"):
        run_benchmark(config, whisper_runner=mismatched_runner)

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before

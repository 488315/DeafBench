import builtins
import json
import os
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

import deafbench.benchmark.synthetic as synthetic_module
from deafbench.benchmark.scenes import plan_scene
from deafbench.benchmark.synthetic import (
    SpeechAudio,
    TTSInfo,
    create_whisperspeech_generator,
    generate_synthetic_set,
    generation_fingerprint,
    synthetic_set_is_current,
)


def _write_references(path: Path) -> Path:
    records = [
        {"id": "ns-001", "text": "Stay seated.", "sounds": ["[alarm]"]},
        {"id": "ns-002", "text": "Wait outside.", "sounds": []},
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    return path


def _fake_speech(text: str) -> SpeechAudio:
    frames = 24_000 + len(text) * 100
    return SpeechAudio(
        np.full((frames, 1), 0.1, dtype=np.float64),
        24_000,
    )


def _generate(
    tmp_path: Path,
    *,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    references = _write_references(tmp_path / "references.jsonl")
    audio_dir = tmp_path / "audio-synthetic"
    manifest = generate_synthetic_set(
        references,
        audio_dir,
        _fake_speech,
        TTSInfo("whisperspeech", "test"),
        seed=seed,
    )
    return references, audio_dir, manifest


def _manifest_records(manifest: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]


def _write_manifest(manifest: Path, records: list[dict[str, object]]) -> None:
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_generate_synthetic_set_writes_complete_wavs_and_timestamp_manifest(
    tmp_path: Path,
) -> None:
    references, audio_dir, manifest = _generate(tmp_path)

    assert manifest == audio_dir / "manifest.jsonl"
    assert {path.name for path in audio_dir.glob("*.wav")} == {
        "ns-001.wav",
        "ns-002.wav",
    }
    records = _manifest_records(manifest)
    assert [record["id"] for record in records] == ["ns-001", "ns-002"]
    fingerprint = generation_fingerprint(
        references,
        "default-v1",
        42,
        TTSInfo("whisperspeech", "test"),
    )
    expected_inputs = [
        ("ns-001", "Stay seated.", ["[alarm]"]),
        ("ns-002", "Wait outside.", []),
    ]
    for record, (sample_id, text, sounds) in zip(
        records,
        expected_inputs,
        strict=True,
    ):
        speech_frames = 2 * (24_000 + len(text) * 100)
        plan = plan_scene(sample_id, speech_frames, sounds, seed=42)
        assert set(record) == {
            "id",
            "wav",
            "fingerprint",
            "scene_profile",
            "seed",
            "sample_rate",
            "tts",
            "speech",
            "background",
            "events",
        }
        assert record["id"] == sample_id
        assert record["wav"] == f"{sample_id}.wav"
        assert record["fingerprint"] == fingerprint
        assert record["scene_profile"] == "default-v1"
        assert record["seed"] == 42
        assert record["sample_rate"] == 48_000
        assert record["tts"] == {
            "engine": "whisperspeech",
            "version": "test",
        }
        assert record["speech"] == {
            "start_ms": plan.speech_start_ms,
            "end_ms": plan.speech_end_ms,
        }
        assert record["background"] == {
            "profile": plan.background_profile,
            "start_ms": plan.background_start_ms,
            "end_ms": plan.background_end_ms,
            "snr_db": plan.background_snr_db,
        }
        assert record["events"] == [
            {
                "label": event.label,
                "start_ms": event.start_ms,
                "end_ms": event.end_ms,
            }
            for event in plan.events
        ]
        with wave.open(str(audio_dir / str(record["wav"])), "rb") as handle:
            background = record["background"]
            assert isinstance(background, dict)
            expected_frames = (
                int(background["end_ms"]) * int(record["sample_rate"]) // 1000
            )
            assert handle.getnframes() == expected_frames
            samples = np.frombuffer(
                handle.readframes(expected_frames),
                dtype="<i2",
            )
        assert np.any(samples != 0)


def test_failed_regeneration_preserves_previous_complete_set(
    tmp_path: Path,
) -> None:
    references, audio_dir, _ = _generate(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in audio_dir.iterdir()
        if path.is_file()
    }
    calls = 0

    def failing_speech(text: str) -> SpeechAudio:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("tts failed")
        return _fake_speech(text)

    with pytest.raises(RuntimeError, match="tts failed"):
        generate_synthetic_set(
            references,
            audio_dir,
            failing_speech,
            TTSInfo("whisperspeech", "test"),
            seed=43,
        )

    after = {
        path.name: path.read_bytes()
        for path in audio_dir.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".audio-synthetic-*"))


def test_untouched_matching_set_is_current(tmp_path: Path) -> None:
    references, audio_dir, _ = _generate(tmp_path)

    assert synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        42,
    )


def test_scoring_metadata_does_not_invalidate_synthetic_audio(
    tmp_path: Path,
) -> None:
    references, audio_dir, _ = _generate(tmp_path)
    records = [
        json.loads(line)
        for line in references.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["critical"] = ["seated"]
    records[0]["critical_types"] = {"seated": "PROPER_NAME"}
    references.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        42,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_wav",
        "extra_wav",
        "invalid_wav",
        "truncated_wav_payload",
        "missing_manifest",
        "malformed_manifest",
        "changed_reference",
        "nonuniform_fingerprint",
        "nonuniform_seed",
        "missing_timing",
        "missing_provenance",
        "unsafe_wav",
        "mismatched_wav",
        "unrecomputed_fingerprint",
    ],
)
def test_cache_rejects_incomplete_or_inconsistent_sets(
    tmp_path: Path,
    mutation: str,
) -> None:
    references, audio_dir, manifest = _generate(tmp_path)
    records = _manifest_records(manifest)

    if mutation == "missing_wav":
        (audio_dir / "ns-002.wav").unlink()
    elif mutation == "extra_wav":
        (audio_dir / "extra.wav").write_bytes(
            (audio_dir / "ns-001.wav").read_bytes()
        )
    elif mutation == "invalid_wav":
        (audio_dir / "ns-001.wav").write_bytes(b"not a wav")
    elif mutation == "truncated_wav_payload":
        wav = audio_dir / "ns-001.wav"
        wav.write_bytes(wav.read_bytes()[:-2])
    elif mutation == "missing_manifest":
        manifest.unlink()
    elif mutation == "malformed_manifest":
        manifest.write_text("not json\n", encoding="utf-8")
    elif mutation == "changed_reference":
        references.write_text(
            references.read_text(encoding="utf-8").replace(
                "Stay seated.", "Stand up."
            ),
            encoding="utf-8",
        )
    elif mutation == "nonuniform_fingerprint":
        records[1]["fingerprint"] = "different"
        _write_manifest(manifest, records)
    elif mutation == "nonuniform_seed":
        records[1]["seed"] = 43
        _write_manifest(manifest, records)
    elif mutation == "missing_timing":
        records[0].pop("speech")
        _write_manifest(manifest, records)
    elif mutation == "missing_provenance":
        records[0].pop("tts")
        _write_manifest(manifest, records)
    elif mutation == "unsafe_wav":
        records[0]["wav"] = "../ns-001.wav"
        _write_manifest(manifest, records)
    elif mutation == "mismatched_wav":
        records[0]["wav"] = "ns-002.wav"
        _write_manifest(manifest, records)
    elif mutation == "unrecomputed_fingerprint":
        records[0]["tts"] = {
            "engine": "whisperspeech",
            "version": "changed",
        }
        records[1]["tts"] = records[0]["tts"]
        _write_manifest(manifest, records)

    assert not synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        42,
    )


def test_cache_rejects_requested_generation_changes(tmp_path: Path) -> None:
    references, audio_dir, _ = _generate(tmp_path)

    assert not synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        43,
    )
    assert not synthetic_set_is_current(
        audio_dir,
        references,
        "future-v2",
        42,
    )


@pytest.mark.parametrize(
    ("persisted_seed", "requested_seed"),
    [(True, 1), (42.0, 42)],
)
def test_cache_rejects_non_integer_persisted_seed(
    tmp_path: Path,
    persisted_seed: object,
    requested_seed: int,
) -> None:
    references, audio_dir, manifest = _generate(
        tmp_path,
        seed=requested_seed,
    )
    records = _manifest_records(manifest)
    for record in records:
        record["seed"] = persisted_seed
    _write_manifest(manifest, records)

    assert not synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        requested_seed,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_event",
        "wrong_event",
        "event_timing",
        "background_profile",
        "background_snr",
        "speech_timing",
    ],
)
def test_cache_rejects_semantically_changed_scene_metadata(
    tmp_path: Path,
    mutation: str,
) -> None:
    references, audio_dir, manifest = _generate(tmp_path)
    records = _manifest_records(manifest)
    first = records[0]
    events = first["events"]
    assert isinstance(events, list) and events
    background = first["background"]
    speech = first["speech"]
    assert isinstance(background, dict)
    assert isinstance(speech, dict)

    if mutation == "missing_event":
        first["events"] = []
    elif mutation == "wrong_event":
        events[0]["label"] = "[knock]"
    elif mutation == "event_timing":
        events[0]["start_ms"] += 1
        events[0]["end_ms"] += 1
    elif mutation == "background_profile":
        background["profile"] = "street-v1"
    elif mutation == "background_snr":
        background["snr_db"] = 12.0
    elif mutation == "speech_timing":
        speech["start_ms"] += 1

    _write_manifest(manifest, records)
    assert not synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        42,
    )


def test_cache_validation_never_imports_whisperspeech(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, audio_dir, _ = _generate(tmp_path)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("whisperspeech"):
            raise AssertionError("cache validation must not import WhisperSpeech")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        42,
    )


@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
def test_generator_preserves_unrelated_import_error(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[ImportError],
) -> None:
    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "whisperspeech.pipeline":
            raise error_type("No module named 'torch'", name="torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ImportError) as caught:
        create_whisperspeech_generator()
    assert caught.value.name == "torch"


def test_generator_preserves_incompatible_pipeline_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "whisperspeech.pipeline":
            raise ImportError(
                "cannot import name 'Pipeline'",
                name="whisperspeech.pipeline",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(ImportError, match="cannot import name 'Pipeline'"):
        create_whisperspeech_generator()


def test_generator_explains_missing_whisperspeech_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "whisperspeech.pipeline":
            raise ModuleNotFoundError(
                "No module named 'whisperspeech'",
                name="whisperspeech",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(
        RuntimeError,
        match=r'python -m pip install "deafbench\[benchmark\]"',
    ):
        create_whisperspeech_generator()


def test_generator_reuses_pipeline_and_reports_actual_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    instances = 0

    class FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.detached = False
            self.on_cpu = False

        def detach(self) -> "FakeTensor":
            self.detached = True
            return self

        def cpu(self) -> "FakeTensor":
            assert self.detached
            self.on_cpu = True
            return self

        def __array__(
            self,
            dtype: np.dtype[np.float32] | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            assert self.on_cpu
            return np.asarray(self.values, dtype=dtype)

    generated_audio: list[FakeTensor] = []

    class FakePipeline:
        def __init__(self) -> None:
            nonlocal instances
            instances += 1

        def generate(self, text: str, *, lang: str) -> "FakeTensor":
            calls.append((text, lang))
            audio = FakeTensor(np.ones((1, 3), dtype=np.float32))
            generated_audio.append(audio)
            return audio

    whisperspeech = types.ModuleType("whisperspeech")
    whisperspeech.__path__ = []  # type: ignore[attr-defined]
    pipeline = types.ModuleType("whisperspeech.pipeline")
    pipeline.Pipeline = FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisperspeech", whisperspeech)
    monkeypatch.setitem(sys.modules, "whisperspeech.pipeline", pipeline)
    monkeypatch.setattr(synthetic_module.metadata, "version", lambda _: "0.8.9")

    generate, info = create_whisperspeech_generator()
    first = generate("Stay seated.")
    second = generate("Wait outside.")

    assert info == TTSInfo("whisperspeech", "0.8.9")
    assert instances == 1
    assert calls == [
        ("Stay seated.", "en"),
        ("Wait outside.", "en"),
    ]
    assert all(audio.detached and audio.on_cpu for audio in generated_audio)
    assert first.sample_rate == second.sample_rate == 24_000
    np.testing.assert_array_equal(first.samples, np.ones((3, 1)))


def test_whisperspeech_audio_adds_channel_to_mono_samples() -> None:
    samples = synthetic_module._normalize_whisperspeech_audio(
        np.ones(3, dtype=np.float32)
    )

    assert samples.shape == (3, 1)


def test_whisperspeech_audio_rejects_invalid_shape() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"WhisperSpeech returned invalid audio shape \(1, 2, 3\)",
    ):
        synthetic_module._normalize_whisperspeech_audio(
            np.ones((1, 2, 3), dtype=np.float32)
        )


def test_interrupted_promotion_restores_last_complete_set_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, audio_dir, _ = _generate(tmp_path)
    before = {
        path.name: path.read_bytes()
        for path in audio_dir.iterdir()
        if path.is_file()
    }
    backup = audio_dir.with_name(f".{audio_dir.name}-backup")
    os.replace(audio_dir, backup)
    original_replace = synthetic_module.os.replace

    def failing_replace(source: object, destination: object) -> None:
        if Path(source) != backup and Path(destination) == audio_dir:
            raise OSError("promotion failed")
        original_replace(source, destination)

    monkeypatch.setattr(synthetic_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="promotion failed"):
        generate_synthetic_set(
            references,
            audio_dir,
            _fake_speech,
            TTSInfo("whisperspeech", "test"),
            seed=43,
        )

    after = {
        path.name: path.read_bytes()
        for path in audio_dir.iterdir()
        if path.is_file()
    }
    assert after == before


def test_backup_cleanup_failure_does_not_fail_committed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, audio_dir, _ = _generate(tmp_path)
    original_rmtree = synthetic_module.shutil.rmtree

    def failing_cleanup(path: object, *args: object, **kwargs: object) -> None:
        if Path(path).name == ".audio-synthetic-backup":
            raise OSError("cleanup failed")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(synthetic_module.shutil, "rmtree", failing_cleanup)
    manifest = generate_synthetic_set(
        references,
        audio_dir,
        _fake_speech,
        TTSInfo("whisperspeech", "test"),
        seed=43,
    )

    assert manifest.exists()
    assert synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        43,
    )


@pytest.mark.parametrize("frame_delta", [-1, 1])
def test_cache_rejects_wav_with_wrong_frame_count(
    tmp_path: Path,
    frame_delta: int,
) -> None:
    references, audio_dir, manifest = _generate(tmp_path)
    wav_path = audio_dir / "ns-001.wav"
    with wave.open(str(wav_path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())

    with wave.open(str(wav_path), "wb") as destination:
        destination.setparams(params)
        if frame_delta < 0:
            frames = frames[: 2 * frame_delta]
        else:
            frames += b"\x00\x00" * frame_delta
        destination.writeframes(frames)

    assert manifest.exists()
    assert not synthetic_set_is_current(
        audio_dir,
        references,
        "default-v1",
        42,
    )

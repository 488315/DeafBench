import json
import wave
from pathlib import Path

import numpy as np
import pytest

from deafbench.benchmark.scenes import plan_scene
from deafbench.benchmark.synthetic import (
    SpeechAudio,
    TTSInfo,
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


def _generate(tmp_path: Path) -> tuple[Path, Path, Path]:
    references = _write_references(tmp_path / "references.jsonl")
    audio_dir = tmp_path / "audio-synthetic"
    manifest = generate_synthetic_set(
        references,
        audio_dir,
        _fake_speech,
        TTSInfo("whisperspeech", "test"),
        seed=42,
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


def test_untouched_matching_set_is_current(tmp_path: Path) -> None:
    references, audio_dir, _ = _generate(tmp_path)

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
        "missing_manifest",
        "malformed_manifest",
        "changed_reference",
        "nonuniform_fingerprint",
        "nonuniform_seed",
        "missing_timing",
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
    elif mutation == "missing_manifest":
        manifest.unlink()
    elif mutation == "malformed_manifest":
        manifest.write_text("not json\n", encoding="utf-8")
    elif mutation == "changed_reference":
        references.write_bytes(references.read_bytes() + b"\n")
    elif mutation == "nonuniform_fingerprint":
        records[1]["fingerprint"] = "different"
        _write_manifest(manifest, records)
    elif mutation == "nonuniform_seed":
        records[1]["seed"] = 43
        _write_manifest(manifest, records)
    elif mutation == "missing_timing":
        records[0].pop("speech")
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

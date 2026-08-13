import hashlib
import json
import math
import os
import wave
from pathlib import Path

import numpy as np

from deafbench.benchmark.synthetic_v2_corpus import (
    REPLACEMENT_REASONS,
    GeneratedSpeech,
    build_synthetic_v2_candidates,
)


def _wav(path: Path, frequency: float) -> None:
    rate = 48_000
    time = np.arange(rate) / rate
    samples = 0.15 * np.sin(2 * math.pi * frequency * time)
    pcm = np.round(samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, sample_id, prepared):
        self.calls.append(sample_id)
        rate = 24_000
        time = np.arange(rate) / rate
        samples = 0.2 * np.sin(2 * math.pi * 330 * time)
        return GeneratedSpeech(
            samples=samples[:, None],
            sample_rate=rate,
            metadata={
                "engine": "fake-tts",
                "version": "test",
                "spoken_aliases": dict(prepared.spoken_aliases),
            },
        )


def test_builder_regenerates_only_predeclared_samples(tmp_path: Path):
    core = tmp_path / "core-v1"
    core_audio = core / "audio-synthetic"
    core_audio.mkdir(parents=True)
    records = [
        {
            "id": sample_id,
            "text": f"Say {index}.",
            "critical": [str(index)],
            "critical_types": {str(index): "DIGIT_SEQUENCE"},
            "sounds": [],
        }
        for index, sample_id in enumerate(REPLACEMENT_REASONS, start=1)
    ]
    records.append(
        {
            "id": "core-019",
            "text": "Keep this sample.",
            "critical": ["Keep"],
            "sounds": [],
        }
    )
    references = "".join(
        json.dumps(record, separators=(",", ":")) + "\n" for record in records
    )
    (core / "references.jsonl").write_text(references, encoding="utf-8")
    parent_hashes = {}
    for index, record in enumerate(records, start=1):
        path = core_audio / f"{record['id']}.wav"
        _wav(path, 180 + index)
        parent_hashes[record["id"]] = hashlib.sha256(path.read_bytes()).hexdigest()

    generator = _FakeGenerator()
    destination = tmp_path / "synthetic-v2"

    manifest_path = build_synthetic_v2_candidates(
        core,
        destination,
        generator,
        seed=42,
    )

    assert set(generator.calls) == set(REPLACEMENT_REASONS)
    assert (destination / "references.jsonl").read_bytes() == (
        core / "references.jsonl"
    ).read_bytes()
    manifest = {
        row["id"]: row
        for row in map(json.loads, manifest_path.read_text(encoding="utf-8").splitlines())
    }
    for sample_id, reason in REPLACEMENT_REASONS.items():
        assert manifest[sample_id]["replacement_reason"] == reason
        assert manifest[sample_id]["parent_audio_sha256"] == parent_hashes[sample_id]
        assert manifest[sample_id]["audio_sha256"] != parent_hashes[sample_id]
        assert manifest[sample_id]["generation"]["engine"] == "fake-tts"
        speech_path = destination / "validation-speech" / f"{sample_id}.wav"
        assert manifest[sample_id]["validation_speech_sha256"] == hashlib.sha256(
            speech_path.read_bytes()
        ).hexdigest()
        with wave.open(str(destination / "audio-synthetic" / f"{sample_id}.wav")) as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        assert not np.any(pcm[: 48_000 // 10])
        assert not np.any(pcm[-48_000 // 10 :])
        assert manifest[sample_id]["scene"]["speech"]["start_ms"] == 100
        assert manifest[sample_id]["scene"]["edge_silence_ms"] == 100
    assert manifest["core-019"]["replacement_reason"] is None
    assert manifest["core-019"]["audio_sha256"] == parent_hashes["core-019"]
    assert manifest["core-019"]["validation_speech_sha256"] is None
    assert not (destination / "validation-speech" / "core-019.wav").exists()


def test_builder_refuses_to_overwrite_a_candidate_corpus(tmp_path: Path):
    destination = tmp_path / "synthetic-v2"
    destination.mkdir()

    try:
        build_synthetic_v2_candidates(tmp_path / "core", destination, _FakeGenerator())
    except FileExistsError as error:
        assert "synthetic-v2" in str(error)
    else:
        raise AssertionError("existing corpus was overwritten")


def test_builder_makes_only_promoted_corpus_readable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    core = tmp_path / "core-v1"
    core_audio = core / "audio-synthetic"
    core_audio.mkdir(parents=True)
    records = [
        {
            "id": sample_id,
            "text": f"Say {index}.",
            "critical": [str(index)],
            "critical_types": {str(index): "DIGIT_SEQUENCE"},
            "sounds": [],
        }
        for index, sample_id in enumerate(REPLACEMENT_REASONS, start=1)
    ]
    (core / "references.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    for index, record in enumerate(records, start=1):
        _wav(core_audio / f"{record['id']}.wav", 180 + index)

    destination = tmp_path / "synthetic-v2"
    events: list[tuple[str, Path, Path | int]] = []
    real_replace = os.replace

    def record_replace(source, target) -> None:
        events.append(("replace", Path(source), Path(target)))
        real_replace(source, target)

    def record_chmod(path: Path, mode: int) -> None:
        events.append(("chmod", path, mode))

    monkeypatch.setattr("deafbench.benchmark.synthetic_v2_corpus.os.replace", record_replace)
    monkeypatch.setattr(Path, "chmod", record_chmod)

    build_synthetic_v2_candidates(core, destination, _FakeGenerator())

    promotion = next(
        index
        for index, event in enumerate(events)
        if event[0] == "replace" and event[2] == destination
    )
    chmod_events = [event for event in events if event[0] == "chmod"]
    assert chmod_events
    assert all(events.index(event) > promotion for event in chmod_events)
    assert all(
        path == destination or destination in path.parents
        for _, path, _ in chmod_events
    )
    assert {mode for _, _, mode in chmod_events} == {0o644, 0o755}

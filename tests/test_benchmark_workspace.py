import json
import wave
from pathlib import Path

import pytest

from deafbench.benchmark.workspace import (
    AudioSetStatus,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    atomic_write_wav,
    inspect_audio_set,
    load_reference_ids,
    load_reference_records,
    resolve_audio_source,
    resolve_run_paths,
    validate_dataset_name,
    validate_wav_format,
)


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    width: int = 2,
    rate: int = 48_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * channels * width * 32)


def _write_references(path: Path, sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": sample_id, "text": sample_id}) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )


def test_run_paths_are_source_aware(tmp_path: Path) -> None:
    paths = resolve_run_paths(
        tmp_path,
        "non-speech-v1",
        "whisper-at",
        "synthetic",
    )
    root = tmp_path / "benchmarks" / "non-speech-v1"

    assert paths.dataset_dir == root
    assert paths.references == root / "references.jsonl"
    assert paths.human_audio == root / "audio"
    assert paths.synthetic_audio == root / "audio-synthetic"
    assert paths.run_dir == root / "runs" / "whisper-at" / "synthetic"
    assert paths.predictions == paths.run_dir / "predictions.jsonl"
    assert paths.report == paths.run_dir / "report.md"
    assert paths.metadata == paths.run_dir / "run.json"


def test_inspect_audio_set_reports_missing_extra_and_invalid(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references.jsonl"
    audio = tmp_path / "audio"
    _write_references(references, ["s1", "s2", "s3"])
    _write_wav(audio / "s1.wav")
    _write_wav(audio / "s2.wav", channels=2)
    _write_wav(audio / "extra.wav")

    status = inspect_audio_set(references, audio)

    assert status.complete is False
    assert status.missing == ("s3",)
    assert status.extra == ("extra",)
    assert status.invalid == ("s2",)


def test_missing_audio_directory_is_an_incomplete_set(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    _write_references(references, ["s2", "s1"])

    status = inspect_audio_set(references, tmp_path / "missing")

    assert status == AudioSetStatus(False, ("s1", "s2"), (), ())


def test_auto_prefers_only_a_complete_human_set() -> None:
    assert resolve_audio_source(
        "auto",
        AudioSetStatus(True, (), (), ()),
    ) == "human"
    assert resolve_audio_source(
        "auto",
        AudioSetStatus(False, ("s2",), (), ()),
    ) == "synthetic"


def test_explicit_human_rejects_incomplete_set() -> None:
    with pytest.raises(ValueError, match="Human audio set is incomplete"):
        resolve_audio_source(
            "human",
            AudioSetStatus(False, ("s2",), (), ()),
        )


def test_explicit_synthetic_does_not_require_human_audio() -> None:
    status = AudioSetStatus(False, ("s1",), (), ())
    assert resolve_audio_source("synthetic", status) == "synthetic"


def test_unknown_audio_source_is_rejected() -> None:
    status = AudioSetStatus(True, (), (), ())
    with pytest.raises(ValueError, match="Unsupported audio source: recorded"):
        resolve_audio_source("recorded", status)  # type: ignore[arg-type]


@pytest.mark.parametrize("dataset", ["", ".", "..", "a/b", "a\\b", "C:temp"])
def test_dataset_name_rejects_unsafe_values(dataset: str) -> None:
    with pytest.raises(ValueError, match="Invalid dataset name"):
        validate_dataset_name(dataset)


def test_dataset_name_returns_safe_value() -> None:
    assert validate_dataset_name("non-speech-v1") == "non-speech-v1"


@pytest.mark.parametrize(
    "sample_id",
    [
        "",
        "   ",
        " s1",
        "s1 ",
        ".",
        "..",
        "../escape",
        "a/b",
        "a\\b",
        "C:escape",
        "/absolute",
        "\u0000",
    ],
)
def test_reference_id_rejects_unsafe_wav_names(
    tmp_path: Path,
    sample_id: str,
) -> None:
    references = tmp_path / "references.jsonl"
    references.write_text(
        json.dumps(
            {
                "id": sample_id,
                "text": "hello",
                "critical": [],
                "sounds": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid reference ID"):
        load_reference_records(references)


@pytest.mark.parametrize(
    "record",
    [
        {"id": "s1", "critical": [], "sounds": []},
        {"id": "s1", "text": 42, "critical": [], "sounds": []},
        {"id": "s1", "text": "hello", "critical": "hello", "sounds": []},
        {"id": "s1", "text": "hello", "critical": [], "sounds": "[alarm]"},
        {"id": "s1", "text": "hello", "critical": [1], "sounds": []},
        {"id": "s1", "text": "hello", "critical": [], "sounds": [1]},
    ],
)
def test_reference_schema_is_validated_before_audio_or_inference(
    tmp_path: Path,
    record: object,
) -> None:
    references = tmp_path / "references.jsonl"
    references.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid reference record"):
        load_reference_records(references)


@pytest.mark.parametrize("contents", ["not json\n", "[]\n", "\n"])
def test_reference_file_rejects_invalid_or_empty_content(
    tmp_path: Path,
    contents: str,
) -> None:
    references = tmp_path / "references.jsonl"
    references.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_reference_records(references)


def test_reference_parser_preserves_order_defaults_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references.jsonl"
    references.write_text(
        '{"id":"s2","text":"two"}\n'
        '{"id":"s1","text":"one","critical":["one"],"sounds":[]}\n',
        encoding="utf-8",
    )

    records = load_reference_records(references)

    assert tuple(record["id"] for record in records) == ("s2", "s1")
    assert records[0]["critical"] == []
    assert records[0]["sounds"] == []
    assert load_reference_ids(references) == ("s2", "s1")

    references.write_text(
        '{"id":"s1","text":"one"}\n{"id":"s1","text":"again"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate reference ID"):
        load_reference_records(references)


@pytest.mark.parametrize(
    ("channels", "width", "rate"),
    [(2, 2, 48_000), (1, 1, 48_000), (1, 2, 44_100)],
)
def test_wav_format_rejects_nonstandard_audio(
    tmp_path: Path,
    channels: int,
    width: int,
    rate: int,
) -> None:
    path = tmp_path / "sample.wav"
    _write_wav(path, channels=channels, width=width, rate=rate)

    with pytest.raises(ValueError, match="Invalid WAV format"):
        validate_wav_format(path)


def test_atomic_text_json_and_jsonl_writers_promote_complete_files(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "nested" / "report.md"
    json_path = tmp_path / "run.json"
    jsonl_path = tmp_path / "predictions.jsonl"

    atomic_write_text(text_path, "complete\n")
    atomic_write_json(json_path, {"model": "whisper"})
    atomic_write_jsonl(jsonl_path, ({"id": "s1"}, {"id": "s2"}))

    assert text_path.read_text(encoding="utf-8") == "complete\n"
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "model": "whisper"
    }
    parsed_records = [
        json.loads(line) for line in jsonl_path.read_text().splitlines()
    ]
    assert parsed_records == [
        {"id": "s1"},
        {"id": "s2"},
    ]
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_atomic_wav_writer_promotes_exact_standard_pcm(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "sample.wav"
    frames = b"\x01\x00\xff\x7f\x00\x80"

    atomic_write_wav(destination, frames)

    validate_wav_format(destination)
    with wave.open(str(destination), "rb") as handle:
        assert handle.readframes(handle.getnframes()) == frames


def test_atomic_wav_writer_rejects_nonstandard_sample_rate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="sample_rate must be 48000"):
        atomic_write_wav(tmp_path / "sample.wav", b"\x00\x00", 44_100)

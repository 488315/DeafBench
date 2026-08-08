from __future__ import annotations

import json
import os
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = REPO_ROOT / "benchmarks" / "core-v1" / "references.jsonl"
AUDIO_DIR = REPO_ROOT / "benchmarks" / "core-v1" / "audio"
OUTPUT = REPO_ROOT / "benchmarks" / "core-v1" / "model-a.jsonl"


def _load_reference_ids(path: Path) -> set[str]:
    reference_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid reference JSON on line {line_number}: {exc.msg}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Invalid reference record on line {line_number}: expected an object")

            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid reference id on line {line_number}: expected a string")
            if sample_id in reference_ids:
                raise ValueError(f"Duplicate reference id: {sample_id}")

            reference_ids.add(sample_id)

    if not reference_ids:
        raise ValueError(f"No reference records found in {path}")

    return reference_ids


def _validate_wav_format(path: Path) -> None:
    expected = "48 kHz, 16-bit PCM, mono"

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid WAV format for {path.name}: expected {expected}") from exc

    if (
        channels != 1
        or sample_width != 2
        or sample_rate != 48_000
        or compression != "NONE"
    ):
        raise ValueError(f"Invalid WAV format for {path.name}: expected {expected}")


def _atomic_write_jsonl(output: Path, records: list[dict[str, str]]) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def transcribe_directory(
    audio_dir: Path,
    output: Path,
    transcribe: Callable[[Path], str],
    *,
    references: Path | None = None,
) -> list[dict[str, str]]:
    wav_paths = sorted(audio_dir.glob("core-*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No core WAV files found in {audio_dir}")

    if references is not None:
        reference_ids = _load_reference_ids(references)
        audio_ids = {wav.stem for wav in wav_paths}
        if reference_ids != audio_ids:
            missing_wavs = sorted(reference_ids - audio_ids)
            extra_wavs = sorted(audio_ids - reference_ids)
            raise ValueError(
                "Reference/audio ID mismatch: "
                f"missing WAVs={missing_wavs}; extra WAVs={extra_wavs}"
            )

    for wav in wav_paths:
        _validate_wav_format(wav)

    records: list[dict[str, str]] = []

    for wav in wav_paths:
        text = transcribe(wav)
        record = {"id": wav.stem, "text": text}
        records.append(record)

    _atomic_write_jsonl(output, records)
    return records


def main() -> None:
    try:
        import whisper
    except ImportError as exc:
        raise SystemExit(
            "Whisper is not installed. Run: python -m pip install -U openai-whisper"
        ) from exc

    model = whisper.load_model("turbo")

    def transcribe(wav: Path) -> str:
        print(f"Transcribing {wav.stem}...")
        result = model.transcribe(
            str(wav),
            language="en",
            task="transcribe",
            verbose=False,
        )
        text = result["text"]
        print(f"  {text}")
        return text

    records = transcribe_directory(
        AUDIO_DIR,
        OUTPUT,
        transcribe,
        references=REFERENCES,
    )
    print(f"\nSaved {len(records)} predictions to {OUTPUT}")


if __name__ == "__main__":
    main()

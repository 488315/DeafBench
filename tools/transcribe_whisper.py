from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "benchmarks" / "core-v1" / "audio"
OUTPUT = REPO_ROOT / "benchmarks" / "core-v1" / "model-a.jsonl"


def transcribe_directory(
    audio_dir: Path,
    output: Path,
    transcribe: Callable[[Path], str],
) -> list[dict[str, str]]:
    wav_paths = sorted(audio_dir.glob("core-*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No core WAV files found in {audio_dir}")

    records: list[dict[str, str]] = []

    for wav in wav_paths:
        text = transcribe(wav)
        record = {"id": wav.stem, "text": text}
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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

    records = transcribe_directory(AUDIO_DIR, OUTPUT, transcribe)
    print(f"\nSaved {len(records)} predictions to {OUTPUT}")


if __name__ == "__main__":
    main()

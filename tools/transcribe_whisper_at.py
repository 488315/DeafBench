from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

try:
    from tools.transcribe_whisper import (
        _atomic_write_jsonl,
        _load_reference_ids,
        _validate_wav_format,
    )
except ModuleNotFoundError:
    from transcribe_whisper import (
        _atomic_write_jsonl,
        _load_reference_ids,
        _validate_wav_format,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "medium.en"
DEFAULT_AT_TIME_RES = 10.0
DEFAULT_TOP_K = 5
DEFAULT_P_THRESHOLD = -1.0

AUDIOSET_TO_DEAFBENCH = {
    "Alarm": "[alarm]",
    "Alarm clock": "[alarm]",
    "Smoke detector, smoke alarm": "[alarm]",
    "Fire alarm": "[alarm]",
    "Car alarm": "[alarm]",
    "Slam": "[door closes]",
    "Telephone bell ringing": "[phone rings]",
    "Ringtone": "[phone rings]",
    "Knock": "[knock]",
    "Beep, bleep": "[error notification]",
    "Ping": "[error notification]",
    "Ding": "[error notification]",
    "Siren": "[siren]",
    "Civil defense siren": "[siren]",
    "Police car (siren)": "[siren]",
    "Ambulance (siren)": "[siren]",
    "Fire engine, fire truck (siren)": "[siren]",
}


def resolve_dataset_paths(
    repo_root: Path,
    dataset: str = "core-v1",
) -> tuple[Path, Path, Path]:
    """Return references, audio, and Model B prediction paths."""
    if not dataset or dataset in {".", ".."} or any(
        separator in dataset for separator in ("/", "\\", ":")
    ):
        raise ValueError("Invalid dataset name")

    dataset_dir = Path(repo_root) / "benchmarks" / dataset
    return (
        dataset_dir / "references.jsonl",
        dataset_dir / "audio",
        dataset_dir / "model-b.jsonl",
    )


def _iter_parsed_segments(parsed: Any) -> Iterable[dict[str, Any]]:
    if isinstance(parsed, dict):
        yield parsed
        return

    if isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes)):
        for segment in parsed:
            if isinstance(segment, dict):
                yield segment


def extract_audio_tags(parsed: Any) -> tuple[list[str], list[str]]:
    """Return unique raw AudioSet tags and mapped DeafBench sound labels."""
    raw_tags: list[str] = []
    sounds: list[str] = []
    seen_raw: set[str] = set()
    seen_sounds: set[str] = set()

    for segment in _iter_parsed_segments(parsed):
        tags = segment.get("audio tags", [])
        if not isinstance(tags, Iterable) or isinstance(tags, (str, bytes)):
            continue

        for tag_entry in tags:
            if not isinstance(tag_entry, (list, tuple)) or not tag_entry:
                continue

            label = tag_entry[0]
            if not isinstance(label, str) or not label:
                continue

            if label not in seen_raw:
                raw_tags.append(label)
                seen_raw.add(label)

            mapped = AUDIOSET_TO_DEAFBENCH.get(label)
            if mapped is not None and mapped not in seen_sounds:
                sounds.append(mapped)
                seen_sounds.add(mapped)

    return raw_tags, sounds


def transcribe_directory(
    audio_dir: Path,
    output: Path,
    transcribe: Callable[[Path], dict[str, Any]],
    *,
    references: Path | None = None,
) -> list[dict[str, Any]]:
    """Transcribe and tag WAV files into structured Model B predictions."""
    wav_paths = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

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

    records: list[dict[str, Any]] = []
    for wav in wav_paths:
        prediction = transcribe(wav)
        text = prediction.get("text", "")
        sounds = prediction.get("sounds", [])
        audio_tags = prediction.get("audio_tags", [])

        if not isinstance(text, str):
            raise ValueError(f"Invalid transcript for {wav.name}: expected a string")
        if not isinstance(sounds, list) or not all(isinstance(sound, str) for sound in sounds):
            raise ValueError(f"Invalid sound labels for {wav.name}: expected a list of strings")
        if not isinstance(audio_tags, list) or not all(
            isinstance(tag, str) for tag in audio_tags
        ):
            raise ValueError(f"Invalid audio tags for {wav.name}: expected a list of strings")

        records.append(
            {
                "id": wav.stem,
                "text": text,
                "sounds": sounds,
                "audio_tags": audio_tags,
            }
        )

    _atomic_write_jsonl(output, records)
    return records


def _parse_at_time_res(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive multiple of 0.4"
        ) from exc

    units = parsed / 0.4
    if (
        not math.isfinite(parsed)
        or parsed <= 0
        or not math.isclose(units, round(units), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise argparse.ArgumentTypeError("must be a positive multiple of 0.4")

    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe DeafBench benchmark WAV files with Whisper-AT"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset", default="core-v1", help="Benchmark directory under benchmarks/")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--at-time-res",
        type=_parse_at_time_res,
        default=DEFAULT_AT_TIME_RES,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--p-threshold", type=float, default=DEFAULT_P_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        default_references, default_audio_dir, default_output = resolve_dataset_paths(
            args.repo_root,
            args.dataset,
        )
    except ValueError as exc:
        parser.error(str(exc))

    references = args.references or default_references
    audio_dir = args.audio_dir or default_audio_dir
    output = args.output or default_output

    try:
        import whisper_at as whisper
    except ImportError as exc:
        raise SystemExit(
            "Whisper-AT is not installed. See the upstream Whisper-AT installation instructions."
        ) from exc

    model = whisper.load_model(args.model)

    def transcribe(wav: Path) -> dict[str, Any]:
        print(f"Transcribing {wav.stem}...")
        result = model.transcribe(
            str(wav),
            language="en",
            task="transcribe",
            verbose=False,
            at_time_res=args.at_time_res,
        )
        parsed = whisper.parse_at_label(
            result,
            language="follow_asr",
            top_k=args.top_k,
            p_threshold=args.p_threshold,
            include_class_list=list(range(527)),
        )
        raw_tags, sounds = extract_audio_tags(parsed)
        text = result["text"]
        print(f"  {text}")
        if sounds:
            print(f"  sounds: {', '.join(sounds)}")
        if raw_tags:
            print(f"  audio tags: {', '.join(raw_tags)}")
        return {
            "text": text,
            "sounds": sounds,
            "audio_tags": raw_tags,
        }

    records = transcribe_directory(
        audio_dir,
        output,
        transcribe,
        references=references,
    )
    print(f"\nSaved {len(records)} predictions to {output}")


if __name__ == "__main__":
    main()

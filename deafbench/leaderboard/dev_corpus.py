"""Pinned, disjoint public development corpus contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, cast

from deafbench.benchmark.workspace import (
    atomic_write_json,
    atomic_write_wav,
    load_reference_records,
)


DEV_DATASET_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
_DATASET_ID = "openslr/librispeech_asr"
_CONFIG = "clean"
_SPLIT = "validation"
_LICENSE = "CC-BY-4.0"
_OFFICIAL_TEST_EXCLUSIONS = {
    "hf-audio/open_asr_leaderboard:librispeech:test.clean",
    "hf-audio/open_asr_leaderboard:librispeech:test.other",
}
_SHA256_LENGTH = 64
_MISSING_DEPENDENCIES = (
    'real-speech dependencies are missing; install "deafbench[real-speech-dev]"'
)


class DevCorpusError(ValueError):
    """Raised when the development corpus contract cannot be trusted."""


@dataclass(frozen=True)
class DevSample:
    """One immutable public development sample identity."""

    sample_id: str
    text: str
    source_audio_sha256: str


@dataclass(frozen=True)
class DevCorpusContract:
    """Validated source identity and ordered development samples."""

    dataset_id: str
    revision: str
    config: str
    split: str
    population_count: int
    samples: tuple[DevSample, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return sample IDs in the declared source order."""
        return tuple(sample.sample_id for sample in self.samples)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DevCorpusError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _require_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise DevCorpusError(f"{label} fields do not match the schema")


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    except (OSError, json.JSONDecodeError) as exc:
        raise DevCorpusError("development corpus manifest is unreadable") from exc


def load_dev_contract(
    manifest_path: Path | str,
    references_path: Path | str,
    *,
    expected_count: int = 100,
) -> DevCorpusContract:
    """Validate the pinned development lane and return its ordered identity."""
    manifest_file = Path(manifest_path)
    references_file = Path(references_path)
    manifest = _load_manifest(manifest_file)
    _require_keys(
        manifest,
        {
            "schema_version",
            "name",
            "purpose",
            "source",
            "selection",
            "official_evaluation_exclusions",
            "references_sha256",
        },
        "manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["name"] != "real-speech-dev-v1"
        or manifest["purpose"] != "model_selection_only"
    ):
        raise DevCorpusError("development corpus identity is unsupported")

    source = _mapping(manifest["source"], "source")
    _require_keys(
        source,
        {"dataset_id", "revision", "config", "split", "license"},
        "source",
    )
    if source["dataset_id"] != _DATASET_ID:
        raise DevCorpusError("development dataset ID is unsupported")
    if source["revision"] != DEV_DATASET_REVISION:
        raise DevCorpusError("development dataset revision is not pinned")
    if source["config"] != _CONFIG:
        raise DevCorpusError("development dataset config is unsupported")
    if source["split"] != _SPLIT:
        raise DevCorpusError("development data must use the validation split")
    if source["license"] != _LICENSE:
        raise DevCorpusError("development dataset license is unsupported")

    selection = _mapping(manifest["selection"], "selection")
    _require_keys(
        selection, {"strategy", "count", "population_count"}, "selection"
    )
    if selection["strategy"] != "sha256_id_lowest":
        raise DevCorpusError("development cohort selection is unsupported")
    if (
        isinstance(selection["count"], bool)
        or selection["count"] != expected_count
    ):
        raise DevCorpusError("development cohort sample count is invalid")
    population_count = selection["population_count"]
    if (
        isinstance(population_count, bool)
        or not isinstance(population_count, int)
        or population_count < expected_count
    ):
        raise DevCorpusError("development source population count is invalid")

    exclusions = manifest["official_evaluation_exclusions"]
    if not isinstance(exclusions, list) or not _OFFICIAL_TEST_EXCLUSIONS <= set(
        exclusions
    ):
        raise DevCorpusError("required official test exclusions are missing")

    try:
        actual_hash = hashlib.sha256(references_file.read_bytes()).hexdigest()
    except OSError as exc:
        raise DevCorpusError("development references are unreadable") from exc
    if manifest["references_sha256"] != actual_hash:
        raise DevCorpusError("development reference hash mismatch")

    try:
        records = load_reference_records(references_file)
    except (OSError, ValueError) as exc:
        raise DevCorpusError("development references are invalid") from exc
    if len(records) != expected_count:
        raise DevCorpusError("development cohort sample count is invalid")

    samples: list[DevSample] = []
    for row in records:
        source_hash = row.get("source_audio_sha256")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise DevCorpusError("development source audio hash is invalid")
        samples.append(
            DevSample(
                sample_id=cast(str, row["id"]),
                text=cast(str, row["text"]),
                source_audio_sha256=source_hash,
            )
        )

    return DevCorpusContract(
        dataset_id=_DATASET_ID,
        revision=DEV_DATASET_REVISION,
        config=_CONFIG,
        split=_SPLIT,
        population_count=population_count,
        samples=tuple(samples),
    )


def _pinned_source_rows(contract: DevCorpusContract) -> Iterable[Mapping[str, Any]]:
    if sys.version_info >= (3, 14):
        raise DevCorpusError(
            "real-speech development materialization requires Python 3.11-3.13"
        )
    try:
        from datasets import Audio, load_dataset
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
        raise DevCorpusError(_MISSING_DEPENDENCIES) from exc

    dataset = load_dataset(
        contract.dataset_id,
        contract.config,
        split=contract.split,
        revision=contract.revision,
        streaming=True,
    )
    return cast(
        Iterable[Mapping[str, Any]],
        dataset.cast_column("audio", Audio(decode=False)),
    )


def _decode_audio(encoded: bytes) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ModuleNotFoundError as exc:
        if exc.name not in {"numpy", "soundfile", "scipy", "scipy.signal"}:
            raise
        raise DevCorpusError(_MISSING_DEPENDENCIES) from exc

    try:
        audio, source_rate = sf.read(
            io.BytesIO(encoded), dtype="float32", always_2d=True
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DevCorpusError("development source audio is unreadable") from exc
    if audio.size == 0 or source_rate <= 0 or not np.isfinite(audio).all():
        raise DevCorpusError("development source audio is invalid")
    mono = audio.mean(axis=1)
    divisor = math.gcd(source_rate, 48_000)
    resampled = resample_poly(mono, 48_000 // divisor, source_rate // divisor)
    pcm = np.rint(np.clip(resampled, -1.0, 1.0) * 32767.0).astype("<i2")
    return cast(bytes, pcm.tobytes())


def _promote_materialization(staging: Path, destination: Path) -> None:
    backup = staging.with_name(f"{staging.name}-previous")
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if had_destination and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _select_source_rows(
    rows: Iterable[Mapping[str, Any]], contract: DevCorpusContract
) -> list[Mapping[str, Any]]:
    population = list(rows)
    if len(population) != contract.population_count:
        raise DevCorpusError("development source population count changed")
    if any(not isinstance(row.get("id"), str) for row in population):
        raise DevCorpusError("development source sample ID is invalid")
    if len({cast(str, row["id"]) for row in population}) != len(population):
        raise DevCorpusError("development source contains duplicate sample IDs")
    return sorted(
        population,
        key=lambda row: (
            hashlib.sha256(cast(str, row["id"]).encode("utf-8")).hexdigest(),
            cast(str, row["id"]),
        ),
    )[: len(contract.samples)]


def materialize_dev_corpus(
    manifest_path: Path | str,
    references_path: Path | str,
    destination: Path | str,
    *,
    source_rows: Iterable[Mapping[str, Any]] | None = None,
    expected_count: int = 100,
) -> Mapping[str, Any]:
    """Materialize the pinned cohort only after every source row is verified."""
    contract = load_dev_contract(
        manifest_path, references_path, expected_count=expected_count
    )
    rows = _select_source_rows(
        source_rows if source_rows is not None else _pinned_source_rows(contract),
        contract,
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}-materialize-",
            dir=destination_path.parent,
        )
    )
    promoted = False
    audio_manifest: list[Mapping[str, Any]] = []
    try:
        for expected, row in zip(contract.samples, rows, strict=True):
            if row.get("id") != expected.sample_id:
                raise DevCorpusError("development source sample order changed")
            if row.get("text") != expected.text:
                raise DevCorpusError(
                    f"development source text changed: {expected.sample_id}"
                )
            audio = row.get("audio")
            if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
                raise DevCorpusError("development source audio payload is invalid")
            encoded = cast(bytes, audio["bytes"])
            if hashlib.sha256(encoded).hexdigest() != expected.source_audio_sha256:
                raise DevCorpusError(
                    f"development source audio hash changed: {expected.sample_id}"
                )
            output = staging / f"{expected.sample_id}.wav"
            atomic_write_wav(output, _decode_audio(encoded))
            audio_manifest.append(
                {
                    "id": expected.sample_id,
                    "source_audio_sha256": expected.source_audio_sha256,
                    "wav_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }
            )

        result: Mapping[str, Any] = {
            "schema_version": 1,
            "source": {
                "dataset_id": contract.dataset_id,
                "revision": contract.revision,
                "config": contract.config,
                "split": contract.split,
            },
            "sample_count": len(contract.samples),
            "sample_rate": 48_000,
            "references_sha256": hashlib.sha256(
                Path(references_path).read_bytes()
            ).hexdigest(),
            "audio": audio_manifest,
        }
        atomic_write_json(staging / "materialization-manifest.json", result)
        _promote_materialization(staging, destination_path)
        promoted = True
        return result
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

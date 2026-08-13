"""Pinned, disjoint public development corpus contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, cast

from deafbench.benchmark.workspace import load_reference_records


DEV_DATASET_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
_DATASET_ID = "openslr/librispeech_asr"
_CONFIG = "clean"
_SPLIT = "validation"
_LICENSE = "CC-BY-4.0"
_OFFICIAL_TEST_EXCLUSIONS = {
    "hf-audio/open_asr_leaderboard:librispeech:test.clean",
    "hf-audio/open_asr_leaderboard:librispeech:test.other",
}


class DevCorpusError(ValueError):
    """Raised when the development corpus contract cannot be trusted."""


@dataclass(frozen=True)
class DevCorpusContract:
    """Validated source identity and ordered development sample IDs."""

    dataset_id: str
    revision: str
    config: str
    split: str
    sample_ids: tuple[str, ...]


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
    _require_keys(selection, {"strategy", "count"}, "selection")
    if selection["strategy"] != "ordered_prefix":
        raise DevCorpusError("development cohort selection is unsupported")
    if (
        isinstance(selection["count"], bool)
        or selection["count"] != expected_count
    ):
        raise DevCorpusError("development cohort sample count is invalid")

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

    return DevCorpusContract(
        dataset_id=_DATASET_ID,
        revision=DEV_DATASET_REVISION,
        config=_CONFIG,
        split=_SPLIT,
        sample_ids=tuple(cast(str, row["id"]) for row in records),
    )

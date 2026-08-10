"""Fail-closed metadata registry for ASR model licensing and runtime scope."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence, cast


_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_LANES = frozenset(
    {
        "commercial_candidate",
        "research_only",
        "proprietary_api",
        "blocked_pending_review",
    }
)
_COMMERCIAL_CLASSIFICATIONS = frozenset(
    {
        "commercial_permitted",
        "commercial_with_attribution",
        "noncommercial",
        "proprietary_terms",
        "pending_review",
    }
)
_REQUIRED_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "upstream_url",
        "spdx_license",
        "commercial_use",
        "attribution_requirements",
        "redistribution_restrictions",
        "remote_code_required",
        "supported_languages",
        "parameter_count",
        "expected_weight_size_bytes",
        "tested_peak_vram_bytes",
        "supported_runtimes",
        "intended_lane",
        "license_files",
        "notice_files",
    }
)


class ModelRegistryError(ValueError):
    """Raised when model licensing metadata is absent or invalid."""


@dataclass(frozen=True)
class ModelLicense:
    """Validated licensing and execution scope for one pinned model revision."""

    model_id: str
    revision: str
    upstream_url: str
    spdx_license: str
    commercial_use: str
    attribution_requirements: tuple[str, ...]
    redistribution_restrictions: tuple[str, ...]
    remote_code_required: bool
    supported_languages: tuple[str, ...]
    parameter_count: int
    expected_weight_size_bytes: int
    tested_peak_vram_bytes: int | None
    supported_runtimes: tuple[str, ...]
    intended_lane: str
    license_files: tuple[str, ...]
    notice_files: tuple[str, ...]


def _required_text(record: Mapping[str, Any], field: str, model_id: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ModelRegistryError(f"invalid {field} for model: {model_id}")
    return value


def _text_sequence(
    record: Mapping[str, Any],
    field: str,
    model_id: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    value = record[field]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or (not allow_empty and not value)
    ):
        raise ModelRegistryError(f"invalid {field} for model: {model_id}")
    return tuple(value)


def _positive_integer(record: Mapping[str, Any], field: str, model_id: str) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelRegistryError(f"invalid {field} for model: {model_id}")
    return value


def _validate_model(record: object) -> ModelLicense:
    if not isinstance(record, Mapping):
        raise ModelRegistryError("model registry entries must be objects")
    missing = _REQUIRED_FIELDS - record.keys()
    model_id = record.get("model_id", "<unknown>")
    if missing:
        raise ModelRegistryError(
            f"missing model metadata for {model_id}: {', '.join(sorted(missing))}"
        )

    model_id = _required_text(record, "model_id", "<unknown>")
    revision = _required_text(record, "revision", model_id)
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ModelRegistryError(f"invalid revision for model: {model_id}")
    upstream_url = _required_text(record, "upstream_url", model_id)
    if upstream_url != f"https://huggingface.co/{model_id}":
        raise ModelRegistryError(f"invalid upstream_url for model: {model_id}")

    commercial_use = _required_text(record, "commercial_use", model_id)
    if commercial_use not in _COMMERCIAL_CLASSIFICATIONS:
        raise ModelRegistryError(f"invalid commercial_use for model: {model_id}")
    intended_lane = _required_text(record, "intended_lane", model_id)
    if intended_lane not in _LANES:
        raise ModelRegistryError(f"invalid intended_lane for model: {model_id}")
    if intended_lane == "commercial_candidate" and commercial_use not in {
        "commercial_permitted",
        "commercial_with_attribution",
    }:
        raise ModelRegistryError(
            f"commercial_candidate lacks commercial permission: {model_id}"
        )

    remote_code_required = record["remote_code_required"]
    if not isinstance(remote_code_required, bool):
        raise ModelRegistryError(
            f"invalid remote_code_required for model: {model_id}"
        )
    tested_peak_vram = record["tested_peak_vram_bytes"]
    if tested_peak_vram is not None and (
        isinstance(tested_peak_vram, bool)
        or not isinstance(tested_peak_vram, int)
        or tested_peak_vram <= 0
    ):
        raise ModelRegistryError(
            f"invalid tested_peak_vram_bytes for model: {model_id}"
        )

    return ModelLicense(
        model_id=model_id,
        revision=revision,
        upstream_url=upstream_url,
        spdx_license=_required_text(record, "spdx_license", model_id),
        commercial_use=commercial_use,
        attribution_requirements=_text_sequence(
            record, "attribution_requirements", model_id, allow_empty=True
        ),
        redistribution_restrictions=_text_sequence(
            record, "redistribution_restrictions", model_id, allow_empty=True
        ),
        remote_code_required=remote_code_required,
        supported_languages=_text_sequence(
            record, "supported_languages", model_id, allow_empty=False
        ),
        parameter_count=_positive_integer(record, "parameter_count", model_id),
        expected_weight_size_bytes=_positive_integer(
            record, "expected_weight_size_bytes", model_id
        ),
        tested_peak_vram_bytes=cast(int | None, tested_peak_vram),
        supported_runtimes=_text_sequence(
            record, "supported_runtimes", model_id, allow_empty=False
        ),
        intended_lane=intended_lane,
        license_files=_text_sequence(
            record, "license_files", model_id, allow_empty=False
        ),
        notice_files=_text_sequence(
            record, "notice_files", model_id, allow_empty=True
        ),
    )


def validate_model_registry(payload: object) -> tuple[ModelLicense, ...]:
    """Validate a decoded registry and reject any incomplete model entry."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ModelRegistryError("unsupported model registry schema")
    if payload.get("legal_advice") is not False:
        raise ModelRegistryError("model registry must disclaim legal advice")
    records = payload.get("models")
    if not isinstance(records, list):
        raise ModelRegistryError("model registry models must be a list")
    models = tuple(_validate_model(record) for record in records)
    ids = [model.model_id for model in models]
    if len(ids) != len(set(ids)):
        raise ModelRegistryError("model registry contains duplicate model IDs")
    return models


def load_model_registry() -> tuple[ModelLicense, ...]:
    """Load and validate the registry bundled with the DeafBench package."""
    resource = files("deafbench").joinpath("model-registry.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError("model registry is unavailable or invalid") from exc
    models = validate_model_registry(payload)
    verify_model_license_files(models, files("deafbench"))
    return models


def verify_model_license_files(
    models: Sequence[ModelLicense], package_root: Traversable
) -> None:
    """Reject unsafe or absent license evidence referenced by model metadata."""
    for model in models:
        for relative in (*model.license_files, *model.notice_files):
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ModelRegistryError(
                    f"unsafe license file for model {model.model_id}: {relative}"
                )
            resource = package_root.joinpath(*path.parts)
            if not resource.is_file():
                raise ModelRegistryError(
                    f"missing license file for model {model.model_id}: {relative}"
                )


def get_model_license(model_id: str) -> ModelLicense:
    """Return pinned license metadata or fail closed for an unknown model."""
    for model in load_model_registry():
        if model.model_id == model_id:
            return model
    raise ModelRegistryError(f"missing license metadata for model: {model_id}")

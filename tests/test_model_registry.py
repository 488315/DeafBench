import hashlib
from copy import deepcopy
from importlib.resources import files

import pytest

from deafbench.model_registry import (
    ModelRegistryError,
    get_model_license,
    load_model_registry,
    validate_model_registry,
    verify_model_license_files,
)
import deafbench.model_registry as model_registry


_MODEL = {
    "model_id": "example/model",
    "revision": "a" * 40,
    "upstream_url": "https://huggingface.co/example/model",
    "spdx_license": "Apache-2.0",
    "commercial_use": "commercial_permitted",
    "attribution_requirements": ["Preserve the Apache-2.0 license."],
    "redistribution_restrictions": ["Provide required license notices."],
    "remote_code_required": False,
    "supported_languages": ["en"],
    "parameter_count": 600_000_000,
    "expected_weight_size_bytes": 1_200_000_000,
    "tested_peak_vram_bytes": None,
    "supported_runtimes": ["transformers"],
    "intended_lane": "commercial_candidate",
    "license_files": ["licenses/example-model/LICENSE"],
    "notice_files": [],
}


def _registry(model: dict[str, object]) -> dict[str, object]:
    return {"schema_version": 1, "legal_advice": False, "models": [model]}


def test_packaged_model_registry_is_valid() -> None:
    models = load_model_registry()
    models_by_id = {model.model_id: model for model in models}
    model = models_by_id["Qwen/Qwen3-ASR-0.6B-hf"]

    assert model.revision == "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
    assert model.spdx_license == "Apache-2.0"
    assert model.intended_lane == "commercial_candidate"
    assert model.remote_code_required is False

    distil = models_by_id["Systran/faster-distil-whisper-large-v3"]
    assert distil.revision == "c3058b475261292e64a0412df1d2681c06260fab"
    assert distil.spdx_license == "MIT"
    assert distil.intended_lane == "commercial_candidate"
    assert distil.remote_code_required is False

    whisper_at = models_by_id["YuanGongND/whisper-at"]
    assert whisper_at.revision == "17d94d6acd53866390ce70f95afa13507dcb8aef"
    assert whisper_at.spdx_license == "BSD-2-Clause"
    assert whisper_at.intended_lane == "commercial_candidate"
    assert whisper_at.remote_code_required is False


def test_packaged_apache_license_matches_canonical_bytes() -> None:
    license_bytes = files("deafbench").joinpath(
        "licenses", "Apache-2.0.txt"
    ).read_bytes()

    assert hashlib.sha256(license_bytes).hexdigest() == (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    )


def test_packaged_cc_by_license_matches_canonical_bytes() -> None:
    license_bytes = files("deafbench").joinpath(
        "licenses", "CC-BY-4.0.txt"
    ).read_bytes()

    assert hashlib.sha256(license_bytes).hexdigest() == (
        "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"
    )


def test_packaged_whisper_at_license_matches_upstream_blob() -> None:
    license_bytes = files("deafbench").joinpath(
        "licenses", "BSD-2-Clause-Whisper-AT.txt"
    ).read_bytes()

    assert hashlib.sha256(license_bytes).hexdigest() == (
        "033fade57eb7bbd6a1266ca63d55101be6892268b5756483afc5239456c30ae0"
    )


def test_registry_rejects_missing_license_metadata() -> None:
    model = deepcopy(_MODEL)
    del model["spdx_license"]

    with pytest.raises(ModelRegistryError, match="missing model metadata"):
        validate_model_registry(_registry(model))


def test_registry_rejects_noncommercial_product_candidate() -> None:
    model = deepcopy(_MODEL)
    model["commercial_use"] = "noncommercial"

    with pytest.raises(ModelRegistryError, match="lacks commercial permission"):
        validate_model_registry(_registry(model))


def test_registry_requires_exact_upstream_model_url() -> None:
    model = deepcopy(_MODEL)
    model["upstream_url"] = "https://example.com/model"

    with pytest.raises(ModelRegistryError, match="invalid upstream_url"):
        validate_model_registry(_registry(model))


def test_registry_accepts_exact_github_repository_url() -> None:
    model = deepcopy(_MODEL)
    model["upstream_url"] = "https://github.com/example/model"

    validated = validate_model_registry(_registry(model))

    assert validated[0].upstream_url == "https://github.com/example/model"


@pytest.mark.parametrize(
    "upstream_url",
    [
        "http://github.com/example/model",
        "https://github.com/example/model/",
        "https://github.com/example/other",
        "https://evil.example/example/model",
    ],
)
def test_registry_rejects_inexact_github_repository_url(upstream_url) -> None:
    model = deepcopy(_MODEL)
    model["upstream_url"] = upstream_url

    with pytest.raises(ModelRegistryError, match="invalid upstream_url"):
        validate_model_registry(_registry(model))


def test_unknown_model_fails_closed() -> None:
    with pytest.raises(ModelRegistryError, match="missing license metadata"):
        get_model_license("unknown/model")


def test_registry_rejects_missing_license_file(tmp_path) -> None:
    model = validate_model_registry(_registry(deepcopy(_MODEL)))[0]

    with pytest.raises(ModelRegistryError, match="missing license file"):
        verify_model_license_files((model,), tmp_path)


def test_registry_accepts_declared_license_file(tmp_path) -> None:
    model = validate_model_registry(_registry(deepcopy(_MODEL)))[0]
    license_path = tmp_path / "licenses" / "example-model" / "LICENSE"
    license_path.parent.mkdir(parents=True)
    license_path.write_text("license evidence", encoding="utf-8")

    verify_model_license_files((model,), tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "", "invalid model_id"),
        ("revision", "main", "invalid revision"),
        ("commercial_use", "unknown", "invalid commercial_use"),
        ("intended_lane", "unknown", "invalid intended_lane"),
        ("remote_code_required", "false", "invalid remote_code_required"),
        ("tested_peak_vram_bytes", True, "invalid tested_peak_vram_bytes"),
        ("supported_languages", [], "invalid supported_languages"),
        ("parameter_count", 0, "invalid parameter_count"),
        ("supported_runtimes", "transformers", "invalid supported_runtimes"),
    ],
)
def test_registry_rejects_invalid_typed_metadata(field, value, message) -> None:
    model = deepcopy(_MODEL)
    model[field] = value
    if field == "model_id":
        model["upstream_url"] = "https://huggingface.co/"

    with pytest.raises(ModelRegistryError, match=message):
        validate_model_registry(_registry(model))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 2, "legal_advice": False, "models": []},
        {"schema_version": 1, "legal_advice": True, "models": []},
        {"schema_version": 1, "legal_advice": False, "models": {}},
    ],
)
def test_registry_rejects_invalid_top_level_contract(payload) -> None:
    with pytest.raises(ModelRegistryError):
        validate_model_registry(payload)


def test_registry_rejects_duplicate_model_ids() -> None:
    payload = _registry(deepcopy(_MODEL))
    payload["models"].append(deepcopy(_MODEL))

    with pytest.raises(ModelRegistryError, match="duplicate model IDs"):
        validate_model_registry(payload)


@pytest.mark.parametrize("relative", ["../LICENSE", "/LICENSE"])
def test_registry_rejects_unsafe_license_paths(tmp_path, relative) -> None:
    model = deepcopy(_MODEL)
    model["license_files"] = [relative]
    validated = validate_model_registry(_registry(model))[0]

    with pytest.raises(ModelRegistryError, match="unsafe license file"):
        verify_model_license_files((validated,), tmp_path)


def test_registry_rejects_non_object_model_entry() -> None:
    payload = {"schema_version": 1, "legal_advice": False, "models": ["model"]}

    with pytest.raises(ModelRegistryError, match="entries must be objects"):
        validate_model_registry(payload)


def test_registry_wraps_unreadable_packaged_json(monkeypatch) -> None:
    class UnreadableResource:
        def joinpath(self, *parts):
            return self

        def read_text(self, **kwargs):
            raise OSError("unavailable")

    monkeypatch.setattr(model_registry, "files", lambda package: UnreadableResource())

    with pytest.raises(ModelRegistryError, match="unavailable or invalid"):
        load_model_registry()

from copy import deepcopy

import pytest

from deafbench.model_registry import (
    ModelRegistryError,
    get_model_license,
    load_model_registry,
    validate_model_registry,
    verify_model_license_files,
)


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
    assert load_model_registry() == ()


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

import json
from pathlib import Path

import pytest

from deafbench.pilot.zero_custody import load_execution_attestation


def _attestation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_mode": "customer_run",
        "customer_authorized_computer": True,
        "customer_audio_uploaded": False,
        "customer_audio_transferred_to_deafbench": False,
        "remote_shell_enabled": False,
        "unattended_access_enabled": False,
        "credentials_shared": False,
        "aggregate_only_export": True,
    }


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_attestation_accepts_customer_run_zero_custody(tmp_path: Path) -> None:
    record = load_execution_attestation(_write(tmp_path / "attestation.json", _attestation()))

    assert record.execution_mode == "customer_run"
    assert record.aggregate_only_export is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_mode", "vendor_run"),
        ("customer_authorized_computer", False),
        ("customer_audio_uploaded", True),
        ("customer_audio_transferred_to_deafbench", True),
        ("remote_shell_enabled", True),
        ("unattended_access_enabled", True),
        ("credentials_shared", True),
        ("aggregate_only_export", False),
    ],
)
def test_attestation_rejects_non_zero_custody_conditions(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _attestation()
    payload[field] = value

    with pytest.raises(ValueError, match="zero-custody"):
        load_execution_attestation(_write(tmp_path / "attestation.json", payload))


def test_attestation_fails_closed_on_missing_or_extra_fields(tmp_path: Path) -> None:
    missing = _attestation()
    missing.pop("credentials_shared")
    extra = {**_attestation(), "customer_name": "not allowed"}

    with pytest.raises(ValueError, match="fields"):
        load_execution_attestation(_write(tmp_path / "missing.json", missing))
    with pytest.raises(ValueError, match="fields"):
        load_execution_attestation(_write(tmp_path / "extra.json", extra))


@pytest.mark.parametrize("content", ["not json", "["])
def test_attestation_rejects_unreadable_content(tmp_path: Path, content: str) -> None:
    path = tmp_path / "attestation.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="unreadable"):
        load_execution_attestation(path)


def test_attestation_rejects_unsupported_schema(tmp_path: Path) -> None:
    payload = _attestation()
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema"):
        load_execution_attestation(_write(tmp_path / "attestation.json", payload))


def test_attestation_requires_literal_booleans(tmp_path: Path) -> None:
    payload = _attestation()
    payload["credentials_shared"] = "false"

    with pytest.raises(ValueError, match="Boolean"):
        load_execution_attestation(_write(tmp_path / "attestation.json", payload))

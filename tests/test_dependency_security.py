import json
import tomllib
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from deafbench.dependency_security import (
    DependencyDispositionError,
    load_dependency_dispositions,
    validate_dependency_disposition,
)


_ROOT = Path(__file__).parents[1]
_DISPOSITION = {
    "schema_version": 1,
    "alert_number": 15,
    "package": "torch",
    "manifest": "pyproject.toml",
    "dependency_scope": "development",
    "installed_version": "2.9.1",
    "advisory": {
        "ghsa": "GHSA-qfhq-4f3w-5fph",
        "cve": "CVE-2025-3001",
        "affected_apis": ["torch.lstm_cell"],
    },
    "status": "tolerable_risk",
    "reviewed_utc": "2026-08-13",
    "review_by": "2026-11-13",
    "model": {
        "id": "ibm-granite/granite-speech-4.1-2b-nar",
        "revision": "a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2",
        "audit_resource": "granite-speech-4.1-2b-nar.json",
    },
    "compatible_stack": {
        "torch": "2.9.1",
        "torchaudio": "2.9.1",
        "torchcodec": "0.9.1",
    },
    "reachability": {
        "first_party": False,
        "audited_remote_code": False,
    },
}


def test_packaged_torch_dispositions_bind_both_alerts() -> None:
    dispositions = load_dependency_dispositions()

    assert {item.alert_number for item in dispositions} == {15, 16}
    assert {item.status for item in dispositions} == {"tolerable_risk"}
    assert {item.package for item in dispositions} == {"torch"}
    assert {item.review_by for item in dispositions} == {date(2026, 11, 13)}


def test_disposition_rejects_expired_review() -> None:
    with pytest.raises(DependencyDispositionError, match="expired"):
        validate_dependency_disposition(_DISPOSITION, today=date(2026, 11, 14))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("status",), "fixed", "status"),
        (("dependency_scope",), "runtime", "scope"),
        (("reachability", "first_party"), True, "reachability"),
        (("model", "revision"), "0" * 40, "revision"),
        (("review_by",), "2026-11-14", "deadline"),
    ],
)
def test_disposition_rejects_changed_security_boundary(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = deepcopy(_DISPOSITION)
    target = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(DependencyDispositionError, match=message):
        validate_dependency_disposition(payload, today=date(2026, 8, 13))


def test_granite_nar_stack_matches_disposition() -> None:
    metadata = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["optional-dependencies"]["granite-nar-asr"]
    expected = {
        item.split("==", 1)[0]: item.split("==", 1)[1]
        for item in dependencies
        if item.split("==", 1)[0] in {"torch", "torchaudio", "torchcodec"}
    }
    dispositions = load_dependency_dispositions(today=date(2026, 8, 13))

    assert all(item.compatible_stack == expected for item in dispositions)


def test_first_party_code_does_not_call_affected_torch_apis() -> None:
    affected_apis = {
        api
        for item in load_dependency_dispositions(today=date(2026, 8, 13))
        for api in item.affected_apis
    }
    production_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_ROOT / "deafbench").rglob("*.py")
        if path.name != "dependency_security.py"
    )

    assert all(api not in production_source for api in affected_apis)


def test_disposition_audit_hashes_match_remote_code_audit() -> None:
    dispositions_path = _ROOT / "deafbench" / "dependency-risk-dispositions.json"
    dispositions = json.loads(dispositions_path.read_text(encoding="utf-8"))
    audit_path = (
        _ROOT
        / "deafbench"
        / "remote-code-audits"
        / dispositions["dispositions"][0]["model"]["audit_resource"]
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert dispositions["reviewed_source_sha256"] == {
        item["path"]: item["sha256"] for item in audit["files"]
    }

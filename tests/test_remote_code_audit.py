import hashlib
from copy import deepcopy

import pytest

from deafbench.remote_code_audit import (
    RemoteCodeAuditError,
    load_remote_code_audit,
    validate_remote_code_audit,
    verify_audited_files,
)


_SOURCE = b"reviewed source\n"
_AUDIT = {
    "schema_version": 1,
    "model_id": "ibm-granite/granite-speech-4.1-2b-nar",
    "revision": "a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2",
    "execution_policy": {
        "allow_network_during_inference": False,
        "isolate_from_main_process": True,
        "require_exact_file_hashes": True,
        "trust_remote_code": True,
    },
    "files": [
        {
            "path": "modeling.py",
            "sha256": hashlib.sha256(_SOURCE).hexdigest(),
        }
    ],
}


def test_packaged_granite_nar_audit_matches_registry() -> None:
    audit = load_remote_code_audit("ibm-granite/granite-speech-4.1-2b-nar")

    assert audit.revision == "a1e3416e25ce29ab3852778e54fa8b3bd59c4bf2"
    assert {record.path for record in audit.audited_files} == {
        "__init__.py",
        "config.json",
        "configuration_granite_speech_nar.py",
        "feature_extraction_granite_speech_nar.py",
        "modeling_granite_speech_nar.py",
        "processing_granite_speech_nar.py",
    }


def test_audit_rejects_network_access() -> None:
    payload = deepcopy(_AUDIT)
    payload["execution_policy"]["allow_network_during_inference"] = True

    with pytest.raises(RemoteCodeAuditError, match="unsafe remote-code policy"):
        validate_remote_code_audit(payload)


def test_audit_rejects_registry_revision_mismatch() -> None:
    payload = deepcopy(_AUDIT)
    payload["revision"] = "0" * 40

    with pytest.raises(RemoteCodeAuditError, match="differs from registry"):
        validate_remote_code_audit(payload)


def test_audit_rejects_unsafe_file_path() -> None:
    payload = deepcopy(_AUDIT)
    payload["files"][0]["path"] = "../modeling.py"

    with pytest.raises(RemoteCodeAuditError, match="unsafe audited file"):
        validate_remote_code_audit(payload)


def test_file_verification_accepts_exact_bytes(tmp_path) -> None:
    audit = validate_remote_code_audit(_AUDIT)
    (tmp_path / "modeling.py").write_bytes(_SOURCE)

    verify_audited_files(audit, tmp_path)


def test_file_verification_accepts_hugging_face_blob_symlink(tmp_path) -> None:
    audit = validate_remote_code_audit(_AUDIT)
    blob = tmp_path / "blobs" / "reviewed"
    blob.parent.mkdir()
    blob.write_bytes(_SOURCE)
    snapshot = tmp_path / "snapshots" / audit.revision
    snapshot.mkdir(parents=True)
    (snapshot / "modeling.py").symlink_to(blob)

    verify_audited_files(audit, snapshot)


def test_file_verification_rejects_changed_bytes(tmp_path) -> None:
    audit = validate_remote_code_audit(_AUDIT)
    (tmp_path / "modeling.py").write_bytes(b"changed source\n")

    with pytest.raises(RemoteCodeAuditError, match="hash mismatch"):
        verify_audited_files(audit, tmp_path)

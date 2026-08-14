import json
from types import SimpleNamespace
import wave

import pytest

from deafbench.benchmark.models import granite_nar


def _workspace(tmp_path):
    references = tmp_path / "references.jsonl"
    references.write_text(
        json.dumps({"id": "sample-1", "text": "reference"}) + "\n",
        encoding="utf-8",
    )
    audio = tmp_path / "audio"
    audio.mkdir()
    with wave.open(str(audio / "sample-1.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\0\0" * 4_800)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    return references, audio, snapshot


def test_adapter_verifies_snapshot_and_writes_worker_records(
    monkeypatch,
    tmp_path,
) -> None:
    references, audio, snapshot = _workspace(tmp_path)
    output = tmp_path / "predictions.jsonl"
    revision = "a" * 40
    audit = SimpleNamespace(revision=revision)
    verified: list[object] = []
    requests: list[tuple[str, object]] = []
    monkeypatch.setattr(granite_nar, "_licensed_revision", lambda: revision)
    monkeypatch.setattr(
        granite_nar,
        "load_remote_code_audit",
        lambda model_id: audit,
    )
    monkeypatch.setattr(
        granite_nar,
        "verify_dependency_disposition_snapshot",
        lambda received, root: verified.append((received, root)),
    )

    def invoke(module, payload):
        requests.append((module, payload))
        return {
            "decoding": {"trust_remote_code": True},
            "performance": {"peak_vram_bytes": 1234},
            "records": [{"id": "sample-1", "latency_ms": 12.5, "text": "prediction"}],
        }

    info = granite_nar.run_granite_nar(
        audio,
        references,
        output,
        snapshot_root=snapshot,
        worker_invoker=invoke,
    )

    assert verified == [(audit, snapshot.resolve())]
    assert requests[0][0] == granite_nar.WORKER_MODULE
    assert requests[0][1]["revision"] == revision
    assert requests[0][1]["wav_paths"] == [str((audio / "sample-1.wav").resolve())]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "id": "sample-1",
        "latency_ms": 12.5,
        "text": "prediction",
    }
    assert info.revision == revision
    assert info.performance == {"peak_vram_bytes": 1234}


def test_adapter_downloads_exact_registered_revision(monkeypatch, tmp_path) -> None:
    references, audio, snapshot = _workspace(tmp_path)
    revision = "a" * 40
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(granite_nar, "_licensed_revision", lambda: revision)
    monkeypatch.setattr(
        granite_nar,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision=revision),
    )
    monkeypatch.setattr(
        granite_nar, "verify_dependency_disposition_snapshot", lambda *args: None
    )

    def download(**options):
        calls.append(options)
        return str(snapshot)

    granite_nar.run_granite_nar(
        audio,
        references,
        tmp_path / "predictions.jsonl",
        snapshot_download=download,
        worker_invoker=lambda module, payload: {
            "decoding": {},
            "performance": {},
            "records": [{"id": "sample-1", "latency_ms": 1.0, "text": "ok"}],
        },
    )

    assert calls == [{"repo_id": granite_nar.MODEL_ID, "revision": revision}]


def test_adapter_rejects_incomplete_worker_output(monkeypatch, tmp_path) -> None:
    references, audio, snapshot = _workspace(tmp_path)
    revision = "a" * 40
    monkeypatch.setattr(granite_nar, "_licensed_revision", lambda: revision)
    monkeypatch.setattr(
        granite_nar,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision=revision),
    )
    monkeypatch.setattr(
        granite_nar, "verify_dependency_disposition_snapshot", lambda *args: None
    )

    with pytest.raises(ValueError, match="incomplete predictions"):
        granite_nar.run_granite_nar(
            audio,
            references,
            tmp_path / "predictions.jsonl",
            snapshot_root=snapshot,
            worker_invoker=lambda module, payload: {
                "decoding": {},
                "performance": {},
                "records": [],
            },
        )


def test_registered_revision_requires_audited_flash_runtime(monkeypatch) -> None:
    license_entry = SimpleNamespace(
        revision="a" * 40,
        remote_code_required=False,
        supported_runtimes=(),
    )
    monkeypatch.setattr(granite_nar, "get_model_license", lambda model_id: license_entry)

    with pytest.raises(granite_nar.ModelRegistryError, match="remote-code"):
        granite_nar._licensed_revision()

    license_entry.remote_code_required = True
    with pytest.raises(granite_nar.ModelRegistryError, match="FlashAttention"):
        granite_nar._licensed_revision()

    license_entry.supported_runtimes = ("flash-attn==2.8.3",)
    monkeypatch.setattr(
        granite_nar,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision="b" * 40),
    )
    with pytest.raises(granite_nar.ModelRegistryError, match="differs"):
        granite_nar._licensed_revision()


def test_registered_revision_accepts_matching_audit(monkeypatch) -> None:
    license_entry = SimpleNamespace(
        revision="a" * 40,
        remote_code_required=True,
        supported_runtimes=("flash-attn==2.8.3",),
    )
    monkeypatch.setattr(granite_nar, "get_model_license", lambda model_id: license_entry)
    monkeypatch.setattr(
        granite_nar,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision="a" * 40),
    )
    validated: list[bool] = []
    monkeypatch.setattr(
        granite_nar,
        "load_dependency_dispositions",
        lambda: validated.append(True),
    )

    assert granite_nar._licensed_revision() == "a" * 40
    assert validated == [True]


@pytest.mark.parametrize(
    "record",
    [
        1,
        {"id": "wrong", "latency_ms": 1.0, "text": "ok"},
        {"id": "sample-1", "latency_ms": 1.0, "text": 1},
        {"id": "sample-1", "latency_ms": True, "text": "ok"},
        {"id": "sample-1", "latency_ms": -1.0, "text": "ok"},
    ],
)
def test_worker_record_validation_rejects_invalid_records(record) -> None:
    with pytest.raises(ValueError, match="invalid prediction"):
        granite_nar._validated_records([record], ["sample-1"])


@pytest.mark.parametrize("field", ["decoding", "performance"])
def test_adapter_requires_worker_metadata(field) -> None:
    with pytest.raises(ValueError, match=f"omitted {field}"):
        granite_nar._required_mapping({}, field)

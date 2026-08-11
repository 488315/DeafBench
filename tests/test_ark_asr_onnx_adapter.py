import json
from types import SimpleNamespace
import wave

import pytest

from deafbench.benchmark.models import ark_asr_onnx


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


def _stub_audit(monkeypatch, revision, verified):
    audit = SimpleNamespace(revision=revision)
    monkeypatch.setattr(ark_asr_onnx, "_licensed_revision", lambda: revision)
    monkeypatch.setattr(
        ark_asr_onnx,
        "load_remote_code_audit",
        lambda model_id: audit,
    )
    monkeypatch.setattr(
        ark_asr_onnx,
        "verify_audited_files",
        lambda received, root: verified.append((received, root)),
    )
    return audit


def _worker_result(records):
    return {
        "decoding": {"precision": "int8"},
        "performance": {"peak_vram_bytes": 0},
        "records": records,
    }


def test_adapter_verifies_snapshot_and_writes_worker_records(
    monkeypatch,
    tmp_path,
) -> None:
    references, audio, snapshot = _workspace(tmp_path)
    output = tmp_path / "predictions.jsonl"
    revision = "a" * 40
    verified: list[object] = []
    audit = _stub_audit(monkeypatch, revision, verified)
    requests: list[tuple[str, object]] = []

    def invoke(module, payload):
        requests.append((module, payload))
        return _worker_result([{"id": "sample-1", "latency_ms": 12.5, "text": "text"}])

    info = ark_asr_onnx.run_ark_asr_onnx(
        audio,
        references,
        output,
        snapshot_root=snapshot,
        worker_invoker=invoke,
    )

    assert verified == [(audit, snapshot.resolve())]
    assert requests[0][0] == ark_asr_onnx.WORKER_MODULE
    assert requests[0][1]["revision"] == revision
    assert requests[0][1]["wav_paths"] == [str((audio / "sample-1.wav").resolve())]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "id": "sample-1",
        "latency_ms": 12.5,
        "text": "text",
    }
    assert info.revision == revision
    assert info.performance == {"peak_vram_bytes": 0}


def test_adapter_downloads_exact_registered_revision(monkeypatch, tmp_path) -> None:
    references, audio, snapshot = _workspace(tmp_path)
    revision = "a" * 40
    _stub_audit(monkeypatch, revision, [])
    calls: list[dict[str, str]] = []

    def download(**options):
        calls.append(options)
        return str(snapshot)

    ark_asr_onnx.run_ark_asr_onnx(
        audio,
        references,
        tmp_path / "predictions.jsonl",
        snapshot_download=download,
        worker_invoker=lambda module, payload: _worker_result(
            [{"id": "sample-1", "latency_ms": 1.0, "text": "ok"}]
        ),
    )

    assert calls == [{"repo_id": ark_asr_onnx.MODEL_ID, "revision": revision}]


def test_adapter_rejects_incomplete_worker_output(monkeypatch, tmp_path) -> None:
    references, audio, snapshot = _workspace(tmp_path)
    _stub_audit(monkeypatch, "a" * 40, [])

    with pytest.raises(ValueError, match="incomplete predictions"):
        ark_asr_onnx.run_ark_asr_onnx(
            audio,
            references,
            tmp_path / "predictions.jsonl",
            snapshot_root=snapshot,
            worker_invoker=lambda module, payload: _worker_result([]),
        )


def test_registered_revision_requires_audited_remote_code(monkeypatch) -> None:
    license_entry = SimpleNamespace(revision="a" * 40, remote_code_required=False)
    monkeypatch.setattr(ark_asr_onnx, "get_model_license", lambda model_id: license_entry)

    with pytest.raises(ark_asr_onnx.ModelRegistryError, match="remote-code"):
        ark_asr_onnx._licensed_revision()

    license_entry.remote_code_required = True
    monkeypatch.setattr(
        ark_asr_onnx,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision="b" * 40),
    )
    with pytest.raises(ark_asr_onnx.ModelRegistryError, match="differs"):
        ark_asr_onnx._licensed_revision()


def test_registered_revision_accepts_matching_audit(monkeypatch) -> None:
    license_entry = SimpleNamespace(revision="a" * 40, remote_code_required=True)
    monkeypatch.setattr(ark_asr_onnx, "get_model_license", lambda model_id: license_entry)
    monkeypatch.setattr(
        ark_asr_onnx,
        "load_remote_code_audit",
        lambda model_id: SimpleNamespace(revision="a" * 40),
    )

    assert ark_asr_onnx._licensed_revision() == "a" * 40


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
        ark_asr_onnx._validated_records([record], ["sample-1"])


@pytest.mark.parametrize("field", ["decoding", "performance"])
def test_adapter_requires_worker_metadata(field) -> None:
    with pytest.raises(ValueError, match=f"omitted {field}"):
        ark_asr_onnx._required_mapping({}, field)

import json
import wave
from pathlib import Path

import pytest

from deafbench.benchmark.models import ModelRunInfo
from deafbench.pilot.audit import (
    PILOT_MODEL_RUNNERS,
    _default_runners,
    run_customer_audit,
)
from deafbench.pilot.export import create_customer_export
from deafbench.pilot.zero_custody import ExecutionAttestation
from deafbench.pilot.export_scan import assert_export_safe
from deafbench.result_manifest import validate_result_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _case(tmp_path: Path) -> Path:
    root = tmp_path / ("case-" + "0" * 32)
    audio = root / "input" / "audio"
    audio.mkdir(parents=True)
    (root / "work").mkdir()
    (root / "authorization.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": root.name,
                "authorization_reference": "agreement-sha256:" + "a" * 64,
                "authorization_date": "2026-08-01",
                "ownership_confirmed": True,
                "scope": "authorized non-sensitive ASR audit",
                "permitted_models": list(PILOT_MODEL_RUNNERS),
                "planned_delivery_date": "2026-08-02",
                "planned_deletion_date": "2026-08-16",
                "sensitivity_classification": "non-sensitive",
                "deletion_agreement": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "input" / "references.jsonl").write_text(
        json.dumps(
            {
                "id": "sample-001",
                "text": "Join Wi-Fi Aurora Guest.",
                "critical": ["Aurora Guest"],
                "critical_types": {"Aurora Guest": "SSID"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with wave.open(str(audio / "sample-001.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * 4_800)
    return root


def _runners(*, mutate_input: bool = False):
    revisions = {
        "Qwen/Qwen3-ASR-1.7B-hf": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        "nvidia/parakeet-tdt-0.6b-v2": "ae9ad07059c7c739ffaf932226a8fe64ae2620b0",
        "ibm-granite/granite-speech-4.1-2b": (
            "de575db64086f84fdc79da4932d1076e965bc546"
        ),
    }

    def build(model_id: str):
        def run(audio_dir: Path, references: Path, output: Path) -> ModelRunInfo:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "id": "sample-001",
                        "text": "Join Wi-Fi Aurora.",
                        "latency_ms": 10.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            if mutate_input and model_id.startswith("Qwen/"):
                references.write_text("{}\n", encoding="utf-8")
            return ModelRunInfo(
                name="mock",
                model_id=model_id,
                revision=revisions[model_id],
                decoding={"device": "cuda", "language": "English"},
                performance={
                    "local_rtfx": 10.0,
                    "median_latency_ms": 10.0,
                    "peak_vram_bytes": 1024,
                },
            )

        return run

    return {model_id: build(model_id) for model_id in PILOT_MODEL_RUNNERS}


def test_customer_audit_writes_three_valid_local_result_manifests(
    tmp_path: Path,
) -> None:
    case_root = _case(tmp_path)

    result = run_customer_audit(
        repo_root=REPO_ROOT,
        case_root=case_root,
        runners=_runners(),
    )

    assert result.sample_count == 1
    assert len(result.result_paths) == 3
    for path in result.result_paths:
        manifest = validate_result_manifest(
            json.loads(path.read_text(encoding="utf-8"))
        )
        evaluation = manifest["evaluations"][0]
        assert manifest["status"] == "customer_audit_complete"
        assert manifest["verification"] == {
            "status": "recorded_local_observation",
            "sample_artifacts_in_repository": False,
            "independently_recomputable_from_checkout": False,
        }
        assert evaluation["lane"] == "customer-audit"
        assert evaluation["critical_failures"] == [
            {
                "entity_type": "SSID",
                "id": "sample-001",
                "term": "Aurora Guest",
            }
        ]


def test_customer_audit_rejects_input_changed_during_execution(tmp_path: Path) -> None:
    case_root = _case(tmp_path)

    with pytest.raises(ValueError, match="changed during evaluation"):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=case_root,
            runners=_runners(mutate_input=True),
        )


def test_customer_audit_rejects_unpermitted_pilot_model(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    authorization_path = case_root / "authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["permitted_models"] = list(PILOT_MODEL_RUNNERS[:-1])
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")

    called = False

    def runner(*args: object, **kwargs: object) -> ModelRunInfo:
        nonlocal called
        called = True
        raise AssertionError("unauthorized model must not run")

    with pytest.raises(ValueError, match="does not permit"):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=case_root,
            runners={model_id: runner for model_id in PILOT_MODEL_RUNNERS},
        )
    assert called is False
    assert not (case_root / "work" / "results").exists()


def test_customer_audit_discovers_exact_default_model_set() -> None:
    assert set(_default_runners()) == set(PILOT_MODEL_RUNNERS)


def test_customer_audit_rejects_incomplete_audio_set(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    (case_root / "input" / "audio" / "sample-001.wav").unlink()

    with pytest.raises(ValueError, match="complete valid audio set"):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=case_root,
            runners=_runners(),
        )


def test_customer_audit_rejects_existing_results(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    (case_root / "work" / "results").mkdir()

    with pytest.raises(FileExistsError, match="already exist"):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=case_root,
            runners=_runners(),
        )


def test_customer_audit_rejects_incomplete_runner_set(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    runners = _runners()
    runners.pop(PILOT_MODEL_RUNNERS[-1])

    with pytest.raises(ValueError, match="exact three-model"):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=case_root,
            runners=runners,
        )


@pytest.mark.parametrize(
    ("revision", "performance", "message"),
    [
        (
            "f" * 40,
            {"local_rtfx": 1, "median_latency_ms": 1, "peak_vram_bytes": 1},
            "revision differs",
        ),
        (
            "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
            None,
            "complete local performance",
        ),
    ],
)
def test_customer_audit_rejects_invalid_model_evidence(
    tmp_path: Path,
    revision: str,
    performance: dict[str, int] | None,
    message: str,
) -> None:
    case_root = _case(tmp_path)
    runners = _runners()
    model_id = PILOT_MODEL_RUNNERS[0]
    original = runners[model_id]

    def invalid_runner(audio_dir: Path, references: Path, output: Path) -> ModelRunInfo:
        info = original(audio_dir, references, output)
        return ModelRunInfo(
            name=info.name,
            model_id=info.model_id,
            revision=revision,
            decoding=info.decoding,
            performance=performance,
        )

    runners[model_id] = invalid_runner
    with pytest.raises(ValueError, match=message):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=case_root,
            runners=runners,
        )


def test_customer_audit_rejects_case_inside_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlaps a Git worktree"):
        run_customer_audit(
            repo_root=REPO_ROOT,
            case_root=REPO_ROOT / "case-test",
            runners=_runners(),
        )


def test_customer_audit_results_create_aggregate_only_export(tmp_path: Path) -> None:
    case_root = _case(tmp_path)
    audit = run_customer_audit(
        repo_root=REPO_ROOT,
        case_root=case_root,
        runners=_runners(),
    )

    output = tmp_path / "export"
    exported = create_customer_export(
        repo_root=REPO_ROOT,
        result_paths=list(audit.result_paths),
        output_dir=output,
        signing_key=tmp_path / "signing-key.pem",
        execution_attestation=ExecutionAttestation("customer_run", True, "d" * 64),
    )

    exported_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output.rglob("*"))
        if path.is_file()
    )
    assert exported.sample_count == 1
    assert "sample-001" not in exported_text
    assert "Aurora Guest" not in exported_text
    assert_export_safe(output)

import json
from pathlib import Path

import pytest

from deafbench.pilot.export import (
    PILOT_MODEL_IDS,
    create_customer_export,
)
from deafbench.pilot.export_scan import assert_export_safe
from deafbench.pilot.manifest import (
    EXECUTION_NOTICE,
    SELF_SIGNED_NOTICE,
    verify_signed_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _result_paths() -> list[Path]:
    root = REPO_ROOT / "experiments" / "model-results"
    return [
        root / "qwen3-asr-1.7b.json",
        root / "parakeet-tdt-0.6b-v2.json",
        root / "granite-speech-4.1-2b.json",
    ]


def _customer_result_paths(tmp_path: Path) -> list[Path]:
    tmp_path.mkdir(parents=True)
    paths = []
    for source in _result_paths():
        value = json.loads(source.read_text(encoding="utf-8"))
        value["status"] = "customer_audit_complete"
        value["corpora"] = [
            {
                **value["corpora"][0],
                "name": "customer-authorized-audio",
            }
        ]
        value["evaluations"] = [value["evaluations"][0]]
        value["evaluations"][0]["lane"] = "customer-audit"
        destination = tmp_path / source.name
        destination.write_text(json.dumps(value), encoding="utf-8")
        paths.append(destination)
    return paths


def test_customer_export_contains_only_three_model_aggregates(tmp_path: Path) -> None:
    output = tmp_path / "export"

    result = create_customer_export(
        repo_root=REPO_ROOT,
        result_paths=_result_paths(),
        output_dir=output,
        signing_key=tmp_path / "private" / "signing-key.pem",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    report = (output / "report.md").read_text(encoding="utf-8")
    assert result.model_count == 3
    assert result.dataset_count == 25
    assert [model["model_id"] for model in manifest["models"]] == list(
        PILOT_MODEL_IDS
    )
    assert manifest["execution_notice"] == EXECUTION_NOTICE
    assert report.startswith(f"# Accessibility-Critical ASR Audit\n\n{EXECUTION_NOTICE}")
    assert SELF_SIGNED_NOTICE in report
    assert "core-" not in report
    assert "Dr. Martinez" not in json.dumps(manifest)
    assert "alpha seven nine" not in json.dumps(manifest)
    assert all(
        "archive" not in model["configuration"] for model in manifest["models"]
    )
    assert_export_safe(output)
    assert verify_signed_manifest(output / "manifest.json") is True


def test_customer_export_accepts_customer_audit_result_manifests(
    tmp_path: Path,
) -> None:
    result = create_customer_export(
        repo_root=REPO_ROOT,
        result_paths=_customer_result_paths(tmp_path / "results"),
        output_dir=tmp_path / "export",
        signing_key=tmp_path / "signing-key.pem",
    )

    assert result.model_count == 3
    assert result.dataset_count == 25


def test_customer_export_rejects_mixed_evaluation_tracks(tmp_path: Path) -> None:
    customer = _customer_result_paths(tmp_path / "results")

    with pytest.raises(ValueError, match="same evaluation track"):
        create_customer_export(
            repo_root=REPO_ROOT,
            result_paths=[_result_paths()[0], *customer[1:]],
            output_dir=tmp_path / "export",
            signing_key=tmp_path / "signing-key.pem",
        )


def test_customer_export_aggregates_critical_failures_by_type(tmp_path: Path) -> None:
    output = tmp_path / "export"

    create_customer_export(
        repo_root=REPO_ROOT,
        result_paths=_result_paths(),
        output_dir=output,
        signing_key=tmp_path / "signing-key.pem",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    qwen = manifest["models"][0]["aggregate_metrics"]
    parakeet = manifest["models"][1]["aggregate_metrics"]
    assert qwen["critical_failures_by_entity_type"] == {
        "CODE": 2,
        "UNCLASSIFIED": 3,
    }
    assert parakeet["critical_failures_by_entity_type"] == {
        "SSID": 1,
        "TIME": 2,
        "UNCLASSIFIED": 2,
    }


def test_customer_export_is_stable_when_inputs_are_reordered(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    key = tmp_path / "signing-key.pem"

    create_customer_export(
        repo_root=REPO_ROOT,
        result_paths=_result_paths(),
        output_dir=first,
        signing_key=key,
    )
    create_customer_export(
        repo_root=REPO_ROOT,
        result_paths=list(reversed(_result_paths())),
        output_dir=second,
        signing_key=key,
    )

    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / "report.md").read_bytes() == (second / "report.md").read_bytes()


def test_customer_export_rejects_incomplete_model_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact three-model"):
        create_customer_export(
            repo_root=REPO_ROOT,
            result_paths=_result_paths()[:2],
            output_dir=tmp_path / "export",
            signing_key=tmp_path / "signing-key.pem",
        )


def test_customer_export_rejects_revision_not_in_registry(tmp_path: Path) -> None:
    value = json.loads(_result_paths()[0].read_text(encoding="utf-8"))
    value["model"]["revision"] = "0" * 40
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="registry"):
        create_customer_export(
            repo_root=REPO_ROOT,
            result_paths=[altered, *_result_paths()[1:]],
            output_dir=tmp_path / "export",
            signing_key=tmp_path / "signing-key.pem",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(status="draft"), "unsupported result manifest"),
        (
            lambda value: value["corpora"][0].update(frozen=False),
            "requires frozen corpora",
        ),
    ],
)
def test_customer_export_requires_validated_result_manifests(
    tmp_path: Path, mutation, message: str
) -> None:
    value = json.loads(_result_paths()[0].read_text(encoding="utf-8"))
    mutation(value)
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        create_customer_export(
            repo_root=REPO_ROOT,
            result_paths=[altered, *_result_paths()[1:]],
            output_dir=tmp_path / "export",
            signing_key=tmp_path / "signing-key.pem",
        )


def test_customer_export_does_not_overwrite_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "export"
    output.mkdir()

    with pytest.raises(FileExistsError, match="must not already exist"):
        create_customer_export(
            repo_root=REPO_ROOT,
            result_paths=_result_paths(),
            output_dir=output,
            signing_key=tmp_path / "signing-key.pem",
        )

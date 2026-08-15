import json
from pathlib import Path

import pytest

from deafbench.pilot.customer_report import (
    _alignment,
    build_report_data,
    render_html,
    write_pdf,
    write_reports,
)


pytestmark = pytest.mark.unit


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _result(path: Path, model_id: str = "Qwen/Qwen3-ASR-1.7B-hf") -> Path:
    path.write_text(
        json.dumps(
            {
                "model": {"model_id": model_id},
                "evaluations": [
                    {
                        "metrics": {
                            "wer_percent": 20.0,
                            "strict_lexical_recall_percent": 50.0,
                            "canonical_semantic_recall_percent": 50.0,
                            "local_rtfx": 10.0,
                            "median_latency_ms": 120.0,
                            "peak_vram_bytes": 1024.0,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _report_data(tmp_path: Path, *, reviews=None):
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    result = tmp_path / "result.json"
    _write_jsonl(
        references,
        [
            {
                "id": "sample-001",
                "text": "Your confirmation code is 83927.",
                "critical": ["83927"],
                "critical_types": {"83927": "CODE"},
            },
            {
                "id": "sample-002",
                "text": "Your confirmation code is 481926.",
                "critical": ["481926"],
                "critical_types": {"481926": "CODE"},
            },
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "id": "sample-001",
                "text": "Your confirmation code is 83972.",
                "latency_ms": 100,
            },
            {
                "id": "sample-002",
                "text": "Your confirmation code is 481962.",
                "latency_ms": 140,
            },
        ],
    )
    return build_report_data(
        case_name="Acme support caption audit",
        case_id="case-" + "a" * 32,
        references_path=references,
        prediction_paths=[predictions],
        result_paths=[_result(result)],
        reviews=reviews,
    )


def test_report_data_groups_each_failed_sample_once(tmp_path: Path) -> None:
    data = _report_data(tmp_path)

    assert data["sample_count"] == 2
    assert len(data["findings"]) == 2
    assert data["category_counts"] == {"codes_passwords_login_information": 2}
    assert all(
        finding["primary_category"] == "codes_passwords_login_information"
        for finding in data["findings"]
    )
    assert all(finding["severity"] == "major" for finding in data["findings"])


def test_html_is_category_first_and_visualizes_wer_alignment(tmp_path: Path) -> None:
    report = render_html(_report_data(tmp_path))

    assert "Codes, passwords &amp; login information" in report
    assert report.count('class="finding"') == 2
    assert "REF:" in report
    assert "HYP:" in report
    assert 'class="token substitute"' in report
    assert 'class="marker substitute">S<' in report
    assert "Correct words" in report
    assert "Deletions" in report
    assert "Substitutions" in report
    assert "Insertions" in report
    assert "Word error rate" in report
    assert "Primary category" in report
    assert "Related factors" in report
    assert "Recommended investigation" in report
    assert "PRIMARY CATEGORY" not in report
    assert "data-category=" in report
    assert "category-filter" in report
    assert "model-filter" in report
    assert "finding-search" in report


def test_html_keeps_text_markers_in_addition_to_color(tmp_path: Path) -> None:
    report = render_html(_report_data(tmp_path))

    assert ".alignment .substitute" in report
    assert ".alignment .delete" in report
    assert ".alignment .insert" in report
    assert '>S<' in report
    assert "Error type:" in report


def test_review_context_and_severity_are_visible_without_erasing_original(tmp_path: Path) -> None:
    first = _report_data(tmp_path)
    finding_id = first["findings"][0]["finding_id"]
    data = _report_data(
        tmp_path,
        reviews={
            finding_id: {
                "reviewed": True,
                "customer_severity": "critical",
                "reason": "This code controls emergency access.",
                "context": "Used during the night shift.",
            }
        },
    )
    finding = next(item for item in data["findings"] if item["finding_id"] == finding_id)
    report = render_html(data)

    assert finding["severity"] == "major"
    assert finding["effective_severity"] == "critical"
    assert "Reviewed" in report
    assert "Customer context added" in report
    assert "Severity adjusted" in report
    assert "DeafBench severity" in report
    assert "Customer severity" in report
    assert "This code controls emergency access." in report
    assert "Used during the night shift." in report


def test_sound_and_speaker_failures_are_combined_into_one_sample_finding(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references-sound.jsonl"
    predictions = tmp_path / "predictions-sound.jsonl"
    result = tmp_path / "result-sound.json"
    _write_jsonl(
        references,
        [
            {
                "id": "sample-001",
                "text": "Leave through the nearest safe exit.",
                "critical": [],
                "sounds": ["[alarm]"],
                "speaker": "Alex",
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "id": "sample-001",
                "text": "Leave through the nearest safe exit.",
                "sounds": [],
                "speaker": "Sam",
                "latency_ms": 10,
            }
        ],
    )

    data = build_report_data(
        case_name="Safety audit",
        case_id="case-" + "b" * 32,
        references_path=references,
        prediction_paths=[predictions],
        result_paths=[_result(result)],
    )

    assert len(data["findings"]) == 1
    finding = data["findings"][0]
    assert finding["primary_category"] == "important_sounds"
    assert {problem["kind"] for problem in finding["problems"]} == {"sound", "speaker"}
    assert "sound_event" in finding["related_factors"]
    assert "speaker_attribution" in finding["related_factors"]


def test_report_renders_clean_empty_state_when_no_findings_exist(tmp_path: Path) -> None:
    references = tmp_path / "references-clean.jsonl"
    predictions = tmp_path / "predictions-clean.jsonl"
    result = tmp_path / "result-clean.json"
    _write_jsonl(
        references,
        [{"id": "sample-001", "text": "Hello world.", "critical": []}],
    )
    _write_jsonl(
        predictions,
        [{"id": "sample-001", "text": "Hello world.", "latency_ms": 10}],
    )
    data = build_report_data(
        case_name="Clean audit",
        case_id="case-" + "c" * 32,
        references_path=references,
        prediction_paths=[predictions],
        result_paths=[_result(result)],
    )

    assert data["findings"] == []
    report = render_html(data)
    assert "No accessibility-critical failures detected" in report


def test_report_data_rejects_mismatched_or_incomplete_model_artifacts(tmp_path: Path) -> None:
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(references, [{"id": "sample-001", "text": "Hello", "critical": []}])
    _write_jsonl(predictions, [{"id": "sample-001", "text": "Hello"}])

    with pytest.raises(ValueError, match="counts do not match"):
        build_report_data(
            case_name="Audit",
            case_id="case-" + "d" * 32,
            references_path=references,
            prediction_paths=[predictions],
            result_paths=[],
        )

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="result manifest is incomplete"):
        build_report_data(
            case_name="Audit",
            case_id="case-" + "d" * 32,
            references_path=references,
            prediction_paths=[predictions],
            result_paths=[incomplete],
        )

    bad_metrics = tmp_path / "bad-metrics.json"
    bad_metrics.write_text(
        json.dumps({"model": {"model_id": "model"}, "evaluations": [{}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metrics are incomplete"):
        build_report_data(
            case_name="Audit",
            case_id="case-" + "d" * 32,
            references_path=references,
            prediction_paths=[predictions],
            result_paths=[bad_metrics],
        )


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("", "", {"insertions": 0, "deletions": 0, "wer": 0.0}),
        ("", "unexpected words", {"insertions": 2, "deletions": 0, "wer": 2.0}),
        ("expected words", "", {"insertions": 0, "deletions": 2, "wer": 1.0}),
    ],
)
def test_alignment_handles_empty_text_under_jiwer_three(
    reference: str, prediction: str, expected: dict[str, float | int]
) -> None:
    alignment = _alignment(reference, prediction)

    assert alignment["insertions"] == expected["insertions"]
    assert alignment["deletions"] == expected["deletions"]
    assert alignment["wer"] == expected["wer"]


def test_pdf_is_real_selectable_text_document(tmp_path: Path) -> None:
    pytest.importorskip("fpdf")
    destination = tmp_path / "report.pdf"

    write_pdf(_report_data(tmp_path), destination)

    assert destination.read_bytes().startswith(b"%PDF-")
    assert destination.stat().st_size > 1000


def test_pdf_renders_unicode_customer_text(tmp_path: Path) -> None:
    pytest.importorskip("fpdf")
    pytest.importorskip("matplotlib")
    data = _report_data(tmp_path)
    data["case_name"] = "Café “Málaga” — accessibility audit"
    data["findings"][0]["reference_text"] = "Call José at 3:15 PM — Γειά σου."
    data["findings"][0]["predicted_text"] = "Call Jose at 3:50 PM — Γεια σου."
    destination = tmp_path / "unicode.pdf"

    write_pdf(data, destination)

    assert destination.read_bytes().startswith(b"%PDF-")
    assert destination.stat().st_size > 1000


def test_pdf_renders_customer_review_fields(tmp_path: Path) -> None:
    pytest.importorskip("fpdf")
    first = _report_data(tmp_path)
    finding_id = first["findings"][0]["finding_id"]
    data = _report_data(
        tmp_path,
        reviews={
            finding_id: {
                "reviewed": True,
                "customer_severity": "critical",
                "reason": "Emergency access context.",
                "context": "Used by night staff.",
            }
        },
    )
    destination = tmp_path / "reviewed.pdf"

    write_pdf(data, destination)

    assert destination.read_bytes().startswith(b"%PDF-")
    assert destination.stat().st_size > 1000


def test_write_reports_creates_matching_html_and_pdf_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("fpdf")
    output = tmp_path / "reports"

    html_path, pdf_path = write_reports(_report_data(tmp_path), output)

    assert html_path == output / "index.html"
    assert pdf_path == output / "report.pdf"
    assert "Acme support caption audit" in html_path.read_text(encoding="utf-8")
    assert pdf_path.read_bytes().startswith(b"%PDF-")

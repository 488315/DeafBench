import csv
import io
import json
import wave
from pathlib import Path

import pytest

from deafbench.pilot.audit import CustomerAuditResult
from deafbench.pilot import customer


pytestmark = pytest.mark.functional


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\x00\x00" * 4_800)


def _customer_case(tmp_path: Path) -> Path:
    root = tmp_path / "customer-case"
    root.mkdir()
    with (root / "references.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "critical", "critical_types", "sounds", "speaker"])
        writer.writerow(
            [
                "sample-001",
                "Your confirmation code is 83927.",
                "83927",
                json.dumps({"83927": "CODE"}),
                "",
                "",
            ]
        )
    _wav(root / "audio" / "sample-001.wav")
    return root


def _inputs(*answers: str):
    iterator = iter(answers)

    def read(_prompt: str) -> str:
        return next(iterator)

    return read


def _fake_audit_runner(**kwargs: object) -> CustomerAuditResult:
    work_dir = Path(kwargs["work_dir"])
    prediction = work_dir / "qwen3-asr-1.7b" / "predictions.jsonl"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text(
        json.dumps(
            {
                "id": "sample-001",
                "text": "Your confirmation code is 83972.",
                "latency_ms": 10.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = work_dir / "results" / "qwen3-asr-1.7b.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(
        json.dumps(
            {
                "model": {"model_id": "Qwen/Qwen3-ASR-1.7B-hf"},
                "evaluations": [
                    {
                        "metrics": {
                            "wer_percent": 20.0,
                            "strict_lexical_recall_percent": 0.0,
                            "canonical_semantic_recall_percent": 0.0,
                            "local_rtfx": 10.0,
                            "median_latency_ms": 10.0,
                            "peak_vram_bytes": 1024.0,
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    start = kwargs.get("on_model_start")
    sample = kwargs.get("on_sample_complete")
    complete = kwargs.get("on_model_complete")
    if callable(start):
        start("Qwen/Qwen3-ASR-1.7B-hf", 1)
    if callable(sample):
        sample("Qwen/Qwen3-ASR-1.7B-hf", Path("sample-001.wav"))
    if callable(complete):
        complete("Qwen/Qwen3-ASR-1.7B-hf")
    return CustomerAuditResult((result,), 1, (prediction,))


def _fake_exporter(**kwargs: object) -> object:
    destination = Path(kwargs["output_dir"])
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text("{}\n", encoding="utf-8")
    return object()


def _fake_reports(_data: object, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    html = destination / "index.html"
    pdf = destination / "report.pdf"
    html.write_text("<html lang='en'><title>Audit</title></html>", encoding="utf-8")
    pdf.write_bytes(b"%PDF-test")
    return html, pdf


def test_first_run_requires_explicit_case_name_and_keeps_state_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _customer_case(tmp_path)
    monkeypatch.setattr(customer, "_require_audit_runtime", lambda: None)
    monkeypatch.setattr(customer, "write_reports", _fake_reports)
    stream = io.StringIO()

    customer.run_audit(
        root,
        input_fn=_inputs(
            "",
            "Acme support caption audit",
            "yes",
            "",
            "no",
            "",
        ),
        browser_opener=lambda _url: False,
        stream=stream,
        repo_root=tmp_path / "not-a-repo",
        audit_runner=_fake_audit_runner,
        exporter=_fake_exporter,
    )

    state = json.loads((root / ".deafbench" / "case.json").read_text(encoding="utf-8"))
    assert state["case_name"] == "Acme support caption audit"
    assert state["case_id"].startswith("case-")
    assert root.name != state["case_name"]
    assert (root / ".deafbench" / "authorization.json").exists()
    assert (root / ".deafbench" / "attestation.json").exists()
    assert (root / ".deafbench" / "references.jsonl").exists()
    assert (root / "audit-report" / "index.html").exists()
    assert (root / "audit-report" / "report.pdf").exists()
    assert (root / "audit-report" / "evidence" / "manifest.json").exists()
    output = stream.getvalue()
    assert "[ok] Case configuration" in output
    assert "[ok] 1 audio samples validated" in output
    assert "Audit complete." in output
    assert "Could not open a graphical browser; the audit still completed." in output


def test_future_run_reuses_saved_setup_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _customer_case(tmp_path)
    monkeypatch.setattr(customer, "_require_audit_runtime", lambda: None)
    monkeypatch.setattr(customer, "write_reports", _fake_reports)
    customer.run_audit(
        root,
        input_fn=_inputs(
            "Acme support caption audit",
            "yes",
            "",
            "no",
            "",
        ),
        no_open=True,
        stream=io.StringIO(),
        repo_root=tmp_path / "not-a-repo",
        audit_runner=_fake_audit_runner,
        exporter=_fake_exporter,
    )

    def unexpected_prompt(prompt: str) -> str:
        raise AssertionError(f"Unexpected prompt: {prompt}")

    customer.run_audit(
        root,
        input_fn=unexpected_prompt,
        no_open=True,
        stream=io.StringIO(),
        repo_root=tmp_path / "not-a-repo",
        audit_runner=_fake_audit_runner,
        exporter=_fake_exporter,
    )

    runs = [path for path in (root / ".deafbench" / "runs").iterdir() if path.is_dir()]
    assert len(runs) == 2


def test_missing_references_creates_csv_template(tmp_path: Path) -> None:
    root = tmp_path / "customer-case"
    root.mkdir()

    with pytest.raises(ValueError, match="created a template"):
        customer._materialize_references(root)

    assert (root / "references.csv").read_text(encoding="utf-8") == (
        "id,text,critical,critical_types,sounds,speaker\n"
    )


def test_csv_is_converted_to_valid_internal_reference(tmp_path: Path) -> None:
    root = _customer_case(tmp_path)
    (root / ".deafbench").mkdir()

    destination = customer._materialize_references(root)
    record = json.loads(destination.read_text(encoding="utf-8"))

    assert record == {
        "critical": ["83927"],
        "critical_types": {"83927": "CODE"},
        "id": "sample-001",
        "sounds": [],
        "text": "Your confirmation code is 83927.",
    }


def test_browser_open_failure_never_changes_audit_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _customer_case(tmp_path)
    monkeypatch.setattr(customer, "_require_audit_runtime", lambda: None)
    monkeypatch.setattr(customer, "write_reports", _fake_reports)
    stream = io.StringIO()

    result = customer.run_audit(
        root,
        case_name="Acme support caption audit",
        input_fn=_inputs("yes", "", "no", ""),
        browser_opener=lambda _url: (_ for _ in ()).throw(OSError("no browser")),
        stream=stream,
        repo_root=tmp_path / "not-a-repo",
        audit_runner=_fake_audit_runner,
        exporter=_fake_exporter,
    )

    assert result == 0
    assert "the audit still completed" in stream.getvalue()


def test_review_preserves_automatic_severity_and_records_customer_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "customer-case"
    state_root = root / ".deafbench"
    output = root / "audit-report"
    state_root.mkdir(parents=True)
    output.mkdir()
    (output / "index.html").write_text("old", encoding="utf-8")
    (output / "report.pdf").write_bytes(b"old")
    (state_root / "case.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "case-" + "a" * 32,
                "case_name": "Acme support caption audit",
            }
        ),
        encoding="utf-8",
    )
    report_data = {
        "schema_version": 1,
        "case_name": "Acme support caption audit",
        "case_id": "case-" + "a" * 32,
        "generated_at": "2026-08-15T00:00:00+00:00",
        "sample_count": 1,
        "models": [],
        "category_counts": {"directions_location_instructions": 1},
        "findings": [
            {
                "finding_id": "finding-1",
                "model_id": "model",
                "model_label": "Model",
                "sample_id": "sample-001",
                "reference_text": "Turn left at the emergency exit.",
                "predicted_text": "Turn right at the emergency exit.",
                "primary_category": "directions_location_instructions",
                "related_factors": ["direction"],
                "severity": "major",
                "effective_severity": "major",
                "impact": "Wrong direction.",
                "recommendation": "Review directions.",
                "alignment": {
                    "columns": [],
                    "correct": 5,
                    "deletions": 0,
                    "substitutions": 1,
                    "insertions": 0,
                    "wer": 0.166,
                },
                "problems": [],
                "review": {},
            }
        ],
    }
    (state_root / "report-data.json").write_text(
        json.dumps(report_data), encoding="utf-8"
    )
    monkeypatch.setattr(customer, "write_reports", _fake_reports)
    stream = io.StringIO()

    result = customer.run_review(
        root,
        input_fn=_inputs(
            "2",
            "critical",
            "This direction is used during an emergency evacuation.",
        ),
        no_open=True,
        stream=stream,
    )

    assert result == 0
    reviews = json.loads((state_root / "reviews.json").read_text(encoding="utf-8"))
    assert reviews["finding-1"]["customer_severity"] == "critical"
    assert "emergency evacuation" in reviews["finding-1"]["reason"]
    updated = json.loads((state_root / "report-data.json").read_text(encoding="utf-8"))
    finding = updated["findings"][0]
    assert finding["severity"] == "major"
    assert finding["effective_severity"] == "critical"
    assert finding["review"]["customer_severity"] == "critical"
    assert "HTML report updated" in stream.getvalue()
    assert "PDF report updated" in stream.getvalue()


@pytest.mark.parametrize(
    ("answers", "message"),
    [
        (("Audit", "no"), "permission"),
        (("Audit", "yes", "", "yes"), "does not accept"),
        (("Audit", "yes", "", "no", "no"), "local-only"),
    ],
)
def test_first_run_setup_fails_closed(
    tmp_path: Path, answers: tuple[str, ...], message: str
) -> None:
    root = tmp_path / "customer-case"
    root.mkdir()

    with pytest.raises(ValueError, match=message):
        customer._first_run_setup(root, case_name=None, input_fn=_inputs(*answers))


def test_input_helpers_reprompt_or_reject_bad_values() -> None:
    assert customer._yes_no("Continue?", _inputs("maybe", "yes")) is True
    assert customer._yes_no("Continue?", _inputs(""), default=True) is True
    assert customer._classification(_inputs("unknown", "public-domain")) == "public_domain"
    with pytest.raises(ValueError, match="plain text"):
        customer._required_text("Case name: ", _inputs("bad\x01name"))


def test_critical_type_csv_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        customer._parse_critical_types("not-json", ["83927"])
    with pytest.raises(ValueError, match="JSON object"):
        customer._parse_critical_types("[]", ["83927"])
    with pytest.raises(ValueError, match="unsupported"):
        customer._parse_critical_types('{"other":"CODE"}', ["83927"])
    with pytest.raises(ValueError, match="unsupported"):
        customer._parse_critical_types('{"83927":"UNKNOWN"}', ["83927"])
    assert customer._parse_critical_types("", ["83927"]) == {}


def test_references_csv_schema_and_rows_are_validated(tmp_path: Path) -> None:
    root = tmp_path / "customer-case"
    state = root / ".deafbench"
    state.mkdir(parents=True)
    (root / "references.csv").write_text("id,text\nsample-001,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain id, text, and critical"):
        customer._materialize_references(root)

    (root / "references.csv").write_text(
        "id,text,critical\n,hello,hello\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing id or text"):
        customer._materialize_references(root)

    (root / "references.csv").write_text(
        "id,text,critical\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="at least one data row"):
        customer._materialize_references(root)


def test_audio_validation_reports_missing_extra_and_invalid_files(tmp_path: Path) -> None:
    root = tmp_path / "customer-case"
    state = root / ".deafbench"
    state.mkdir(parents=True)
    references = state / "references.jsonl"
    references.write_text(
        json.dumps({"id": "sample-001", "text": "hello", "critical": []}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Audio folder was missing"):
        customer._validate_inputs(root, references)
    assert (root / "audio").is_dir()

    (root / "audio" / "sample-001.wav").write_bytes(b"not a wav")
    _wav(root / "audio" / "extra.wav")
    with pytest.raises(ValueError) as exc_info:
        customer._validate_inputs(root, references)
    message = str(exc_info.value)
    assert "extra: extra" in message
    assert "invalid WAV: sample-001" in message


def test_runtime_check_prints_exact_audit_install_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        customer.importlib.util,
        "find_spec",
        lambda name: None if name == "nemo" else object(),
    )

    with pytest.raises(RuntimeError, match=r'deafbench\[audit\]'):
        customer._require_audit_runtime()


def test_report_open_success_is_silent(tmp_path: Path) -> None:
    stream = io.StringIO()
    customer._open_report(tmp_path / "index.html", lambda _url: True, stream)
    assert stream.getvalue() == ""


def test_replace_report_output_replaces_old_evidence(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    (staging / "evidence").mkdir(parents=True)
    (staging / "index.html").write_text("new html", encoding="utf-8")
    (staging / "report.pdf").write_bytes(b"new pdf")
    (staging / "evidence" / "manifest.json").write_text("new", encoding="utf-8")
    (output / "evidence").mkdir(parents=True)
    (output / "evidence" / "old.txt").write_text("old", encoding="utf-8")

    customer._replace_report_output(staging, output)

    assert (output / "index.html").read_text(encoding="utf-8") == "new html"
    assert (output / "report.pdf").read_bytes() == b"new pdf"
    assert not (output / "evidence" / "old.txt").exists()
    assert (output / "evidence" / "manifest.json").read_text(encoding="utf-8") == "new"


def test_review_keep_context_skip_and_quit_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "customer-case"
    state_root = root / ".deafbench"
    output = root / "audit-report"
    state_root.mkdir(parents=True)
    output.mkdir()
    (output / "index.html").write_text("old", encoding="utf-8")
    (output / "report.pdf").write_bytes(b"old")
    (state_root / "case.json").write_text(
        json.dumps({"schema_version": 1, "case_id": "case-" + "a" * 32, "case_name": "Audit"}),
        encoding="utf-8",
    )
    findings = []
    for index in range(4):
        findings.append(
            {
                "finding_id": f"finding-{index}",
                "model_id": "model",
                "model_label": "Model",
                "sample_id": f"sample-{index}",
                "reference_text": "Turn left.",
                "predicted_text": "Turn right.",
                "primary_category": "directions_location_instructions",
                "related_factors": ["direction"],
                "severity": "major",
                "effective_severity": "major",
                "impact": "Wrong direction.",
                "recommendation": "Review directions.",
                "alignment": {"columns": [], "correct": 1, "deletions": 0, "substitutions": 1, "insertions": 0, "wer": 0.5},
                "problems": [],
                "review": {},
            }
        )
    (state_root / "report-data.json").write_text(
        json.dumps({"findings": findings, "case_name": "Audit", "generated_at": "old"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(customer, "write_reports", _fake_reports)

    customer.run_review(
        root,
        input_fn=_inputs("1", "3", "Used by field technicians.", "4", "5"),
        no_open=True,
        stream=io.StringIO(),
    )

    reviews = json.loads((state_root / "reviews.json").read_text(encoding="utf-8"))
    assert reviews["finding-0"]["reviewed"] is True
    assert reviews["finding-1"]["context"] == "Used by field technicians."
    assert "finding-2" not in reviews
    assert "finding-3" not in reviews


def test_cli_wrappers_return_status_and_render_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(customer, "run_audit", lambda *args, **kwargs: 7)
    assert customer.audit_main([str(tmp_path), "--no-open"]) == 7
    monkeypatch.setattr(customer, "run_review", lambda *args, **kwargs: 8)
    assert customer.review_main([str(tmp_path), "--no-open"]) == 8

    def fail(*_args: object, **_kwargs: object) -> int:
        raise ValueError("bad customer input")

    monkeypatch.setattr(customer, "run_audit", fail)
    with pytest.raises(SystemExit) as exc_info:
        customer.audit_main([str(tmp_path)])
    assert exc_info.value.code == 1
    assert "bad customer input" in capsys.readouterr().err

    monkeypatch.setattr(customer, "run_review", fail)
    with pytest.raises(SystemExit) as exc_info:
        customer.review_main([str(tmp_path)])
    assert exc_info.value.code == 1
    assert "bad customer input" in capsys.readouterr().err


def test_existing_review_file_loads_and_signs_with_case_key(tmp_path: Path) -> None:
    from base64 import b64decode

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from deafbench.pilot.manifest import _load_or_create_key

    state_root = tmp_path / ".deafbench"
    state_root.mkdir()
    reviews = {
        "finding-1": {
            "reviewed": True,
            "context": "Used by field technicians.",
        }
    }
    (state_root / "reviews.json").write_text(
        json.dumps({**reviews, "ignored": "not a review object"}),
        encoding="utf-8",
    )

    assert customer._load_reviews(state_root) == reviews

    _load_or_create_key(state_root / "signing-key.pem")
    customer._sign_reviews(state_root, reviews)

    signature = json.loads(
        (state_root / "reviews-signature.json").read_text(encoding="utf-8")
    )
    assert signature["algorithm"] == "Ed25519"
    public_key = Ed25519PublicKey.from_public_bytes(
        b64decode(signature["public_key_base64"])
    )
    body = json.dumps(reviews, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key.verify(b64decode(signature["signature_base64"]), body)


def test_sign_reviews_is_noop_without_case_key(tmp_path: Path) -> None:
    state_root = tmp_path / ".deafbench"
    state_root.mkdir()

    customer._sign_reviews(state_root, {"finding-1": {"reviewed": True}})

    assert not (state_root / "reviews-signature.json").exists()


def test_no_review_changes_leave_reports_untouched(tmp_path: Path) -> None:
    root = tmp_path / "customer-case"
    state_root = root / ".deafbench"
    state_root.mkdir(parents=True)
    (state_root / "case.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "case-" + "a" * 32,
                "case_name": "Audit",
            }
        ),
        encoding="utf-8",
    )
    (state_root / "report-data.json").write_text(
        json.dumps(
            {
                "findings": [],
                "case_name": "Audit",
                "generated_at": "old",
            }
        ),
        encoding="utf-8",
    )
    stream = io.StringIO()

    assert customer.run_review(root, input_fn=_inputs(), no_open=True, stream=stream) == 0
    assert "No review changes saved." in stream.getvalue()

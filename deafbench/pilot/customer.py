"""Friendly customer-local audit and review workflows."""

from __future__ import annotations

import argparse
import base64
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
import webbrowser
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TextIO

from deafbench.benchmark.workspace import (
    atomic_write_text,
    inspect_audio_set,
    load_reference_records,
)
from deafbench.critical_entities import ENTITY_TYPES
from deafbench.pilot.audit import PILOT_MODEL_RUNNERS, run_customer_audit
from deafbench.pilot.intake import (
    ALLOWED_CLASSIFICATIONS,
    PROHIBITED_CATEGORIES,
    evaluate_intake,
)
from deafbench.pilot.customer_report import build_report_data, write_reports
from deafbench.pilot.taxonomy import category_label, severity_label
from deafbench.pilot.workspace import validate_case_root


InputFunction = Callable[[str], str]
BrowserOpener = Callable[[str], bool]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _state_dir(case_root: Path) -> Path:
    return case_root / ".deafbench"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
    )


def _required_text(prompt: str, input_fn: InputFunction) -> str:
    while True:
        value = input_fn(prompt).strip()
        if value:
            if any(ord(character) < 32 for character in value) or len(value) > 120:
                raise ValueError("Case name must be plain text no longer than 120 characters")
            return value
        print("A case name is required.")


def _yes_no(prompt: str, input_fn: InputFunction, *, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        value = input_fn(prompt + suffix).strip().casefold()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _classification(input_fn: InputFunction) -> str:
    choices = (
        "non_sensitive",
        "synthetic",
        "public_domain",
        "explicitly_consented_test",
    )
    display = ", ".join(value.replace("_", "-") for value in choices)
    while True:
        value = input_fn(
            f"Content classification ({display}) [non-sensitive]: "
        ).strip()
        if not value:
            return "non_sensitive"
        normalized = value.casefold().replace("-", "_").replace(" ", "_")
        if normalized in ALLOWED_CLASSIFICATIONS:
            return normalized
        print("Choose one of the listed content classifications.")


def _first_run_setup(
    case_root: Path,
    *,
    case_name: str | None,
    input_fn: InputFunction,
) -> dict[str, Any]:
    print("DeafBench audit setup")
    print()
    name = case_name.strip() if case_name and case_name.strip() else _required_text(
        "Case name: ", input_fn
    )
    if not _yes_no(
        "Do you own this audio or have permission to evaluate it?",
        input_fn,
    ):
        raise ValueError("Audit setup requires permission to evaluate the audio")
    classification = _classification(input_fn)
    prohibited_prompt = (
        "Does this case contain medical records, consumer health data, minors' audio, "
        "payment information, authentication secrets, legal recordings, or other "
        "regulated/high-risk material?"
    )
    has_prohibited = _yes_no(prohibited_prompt, input_fn)
    prohibited = {key: has_prohibited for key in PROHIBITED_CATEGORIES}
    decision = evaluate_intake(
        sensitivity_classification=classification,
        prohibited_categories=prohibited,
    )
    if not decision.accepted:
        raise ValueError(
            "This customer audit does not accept the declared sensitive or regulated material"
        )
    if not _yes_no(
        "Run this audit locally without uploading customer audio or enabling remote access?",
        input_fn,
        default=True,
    ):
        raise ValueError("Customer audit requires the local-only execution boundary")

    state_root = _state_dir(case_root)
    state_root.mkdir(parents=True, exist_ok=True)
    case_id = f"case-{uuid.uuid4().hex}"
    today = date.today()
    state = {
        "schema_version": 1,
        "case_id": case_id,
        "case_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sensitivity_classification": classification,
    }
    _write_json(state_root / "case.json", state)
    _write_json(
        state_root / "authorization.json",
        {
            "schema_version": 1,
            "case_id": case_id,
            "authorization_reference": f"customer-wizard:{case_id}",
            "authorization_date": today.isoformat(),
            "ownership_confirmed": True,
            "scope": "customer-local accessibility-critical caption audit",
            "permitted_models": list(PILOT_MODEL_RUNNERS),
            "planned_delivery_date": today.isoformat(),
            "planned_deletion_date": (today + timedelta(days=14)).isoformat(),
            "sensitivity_classification": classification,
            "deletion_agreement": True,
        },
    )
    _write_json(
        state_root / "attestation.json",
        {
            "schema_version": 1,
            "execution_mode": "customer_run",
            "customer_authorized_computer": True,
            "customer_audio_uploaded": False,
            "customer_audio_transferred_to_deafbench": False,
            "remote_shell_enabled": False,
            "unattended_access_enabled": False,
            "credentials_shared": False,
            "aggregate_only_export": True,
        },
    )
    return state


def _load_or_setup_case(
    case_root: Path,
    *,
    case_name: str | None,
    input_fn: InputFunction,
) -> dict[str, Any]:
    state_path = _state_dir(case_root) / "case.json"
    if not state_path.exists():
        return _first_run_setup(
            case_root,
            case_name=case_name,
            input_fn=input_fn,
        )
    state = _read_json(state_path)
    if state.get("schema_version") != 1 or not str(state.get("case_id", "")).startswith("case-"):
        raise ValueError("DeafBench case configuration is invalid")
    if not isinstance(state.get("case_name"), str) or not state["case_name"].strip():
        raise ValueError("DeafBench case configuration is missing a case name")
    if case_name and case_name.strip() and case_name.strip() != state["case_name"]:
        state["case_name"] = case_name.strip()
        _write_json(state_path, state)
    return state


def _split_pipe(value: str | None) -> list[str]:
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _parse_critical_types(value: str | None, critical: Sequence[str]) -> dict[str, str]:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("references.csv critical_types must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("references.csv critical_types must be a JSON object")
    result = {str(key): str(entity_type) for key, entity_type in parsed.items()}
    if set(result) - set(critical) or set(result.values()) - ENTITY_TYPES:
        raise ValueError("references.csv critical_types contains an unsupported term or type")
    return result


def _ensure_reference_template(case_root: Path) -> Path:
    path = case_root / "references.csv"
    if path.exists():
        return path
    path.write_text(
        "id,text,critical,critical_types,sounds,speaker\n",
        encoding="utf-8",
        newline="\n",
    )
    raise ValueError(
        f"references.csv was missing. DeafBench created a template at {path}; add one row per WAV file and run the audit again."
    )


def _materialize_references(case_root: Path) -> Path:
    source = _ensure_reference_template(case_root)
    destination = _state_dir(case_root) / "references.jsonl"
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "text", "critical"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("references.csv must contain id, text, and critical columns")
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("id") or "").strip()
            text = row.get("text")
            if not sample_id or text is None:
                raise ValueError(f"references.csv row {row_number} is missing id or text")
            critical = _split_pipe(row.get("critical"))
            record: dict[str, Any] = {
                "id": sample_id,
                "text": text.strip(),
                "critical": critical,
                "critical_types": _parse_critical_types(
                    row.get("critical_types"), critical
                ),
                "sounds": _split_pipe(row.get("sounds")),
            }
            if row.get("speaker") and row["speaker"].strip():
                record["speaker"] = row["speaker"].strip()
            records.append(record)
    if not records:
        raise ValueError("references.csv must contain at least one data row")
    atomic_write_text(
        destination,
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    )
    load_reference_records(destination)
    return destination


def _validate_inputs(case_root: Path, references: Path) -> int:
    audio_dir = case_root / "audio"
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)
        raise ValueError(
            f"Audio folder was missing. DeafBench created {audio_dir}; add 48 kHz mono PCM16 WAV files matching references.csv IDs and run again."
        )
    status = inspect_audio_set(references, audio_dir)
    if not status.complete:
        details = []
        if status.missing:
            details.append("missing: " + ", ".join(status.missing))
        if status.extra:
            details.append("extra: " + ", ".join(status.extra))
        if status.invalid:
            details.append("invalid WAV: " + ", ".join(status.invalid))
        raise ValueError("Audio validation failed (" + "; ".join(details) + ")")
    return len(load_reference_records(references))


class _MoonSpinner:
    def __init__(self, message: str, stream: TextIO, enabled: bool) -> None:
        self.message = message
        self.stream = stream
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_MoonSpinner":
        if not self.enabled:
            print(f"{self.message}...", file=self.stream)
            return self

        def run() -> None:
            frames = ("◐", "◓", "◑", "◒")
            index = 0
            while not self._stop.wait(0.12):
                self.stream.write(f"\r{frames[index % len(frames)]} {self.message}...")
                self.stream.flush()
                index += 1

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.enabled:
            self.stream.write("\r" + " " * (len(self.message) + 8) + "\r")
            self.stream.flush()


class AuditProgress:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream
        encoding = (getattr(stream, "encoding", None) or "").casefold()
        self.unicode = bool(getattr(stream, "isatty", lambda: False)()) and "utf" in encoding
        self._bars: dict[str, Any] = {}

    def check(self, message: str) -> None:
        prefix = "✓" if self.unicode else "[ok]"
        print(f"{prefix} {message}", file=self.stream)

    @contextmanager
    def spinner(self, message: str) -> Iterator[None]:
        with _MoonSpinner(message, self.stream, self.unicode):
            yield

    def model_start(self, model_id: str, total: int) -> None:
        label = {
            "Qwen/Qwen3-ASR-1.7B-hf": "Qwen3-ASR 1.7B",
            "nvidia/parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B v2",
            "ibm-granite/granite-speech-4.1-2b": "Granite Speech 4.1 2B",
        }.get(model_id, model_id)
        print(f"\nEvaluating {label}", file=self.stream)
        if not self.unicode:
            return
        try:
            from tqdm import tqdm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'Audit progress support is not installed. Run: python -m pip install "deafbench[audit]"'
            ) from exc
        self._bars[model_id] = tqdm(
            total=total,
            unit="sample",
            leave=True,
            file=self.stream,
            bar_format="  {percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt}",
        )

    def sample_complete(self, model_id: str, _path: Path) -> None:
        bar = self._bars.get(model_id)
        if bar is not None:
            bar.update(1)

    def model_complete(self, model_id: str) -> None:
        bar = self._bars.pop(model_id, None)
        if bar is not None:
            bar.close()
        self.check("Complete")


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _load_reviews(state_root: Path) -> dict[str, dict[str, Any]]:
    path = state_root / "reviews.json"
    if not path.exists():
        return {}
    value = _read_json(path)
    return {
        str(key): dict(item)
        for key, item in value.items()
        if isinstance(item, dict)
    }


def _sign_reviews(state_root: Path, reviews: Mapping[str, Mapping[str, Any]]) -> None:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'Audit signing support is not installed. Run: python -m pip install "deafbench[audit]"'
        ) from exc

    key_path = state_root / "signing-key.pem"
    if not key_path.exists():
        return
    key = serialization.load_pem_private_key(key_path.read_bytes(), None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("Audit signing key is not Ed25519")
    body = json.dumps(reviews, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    _write_json(
        state_root / "reviews-signature.json",
        {
            "algorithm": "Ed25519",
            "public_key_base64": base64.b64encode(public_key).decode("ascii"),
            "signature_base64": base64.b64encode(key.sign(body)).decode("ascii"),
        },
    )


def _require_audit_runtime() -> None:
    required = {
        "cryptography": "cryptography",
        "fpdf": "fpdf2",
        "nemo": "nemo_toolkit",
        "numba": "numba",
        "scipy": "scipy",
        "tqdm": "tqdm",
        "torchaudio": "torchaudio",
        "transformers": "transformers",
    }
    missing = [label for module, label in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            "Audit dependencies are not installed (missing: "
            + ", ".join(missing)
            + '). Run: python -m pip install "deafbench[audit]"'
        )


def _replace_report_output(staging: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    os.replace(staging / "index.html", output / "index.html")
    os.replace(staging / "report.pdf", output / "report.pdf")
    old_evidence = output / "evidence"
    if old_evidence.exists():
        shutil.rmtree(old_evidence)
    shutil.move(str(staging / "evidence"), str(old_evidence))


def _open_report(path: Path, opener: BrowserOpener, stream: TextIO) -> None:
    try:
        opened = opener(path.resolve().as_uri())
    except Exception:
        opened = False
    if not opened:
        print(f"Report: {path}", file=stream)
        print("Could not open a graphical browser; the audit still completed.", file=stream)


def run_audit(
    case_root: Path,
    *,
    case_name: str | None = None,
    no_open: bool = False,
    input_fn: InputFunction = input,
    browser_opener: BrowserOpener = webbrowser.open,
    stream: TextIO = sys.stdout,
    repo_root: Path | None = None,
    audit_runner: Callable[..., Any] = run_customer_audit,
    exporter: Callable[..., Any] | None = None,
) -> int:
    _require_audit_runtime()
    if exporter is None:
        try:
            from deafbench.pilot.export import create_customer_export
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'Audit export support is not installed. Run: python -m pip install "deafbench[audit]"'
            ) from exc
        exporter = create_customer_export
    root = validate_case_root(Path(case_root))
    state = _load_or_setup_case(root, case_name=case_name, input_fn=input_fn)
    progress = AuditProgress(stream)
    print("\nDeafBench audit", file=stream)
    print(str(state["case_name"]), file=stream)
    print(file=stream)
    progress.check("Case configuration")

    references = _materialize_references(root)
    sample_count = _validate_inputs(root, references)
    progress.check(f"{sample_count} audio samples validated")

    state_root = _state_dir(root)
    run_root = state_root / "runs" / _run_id()
    work_root = run_root / "work"
    package_root = Path(repo_root) if repo_root is not None else _repo_root()
    audit_result = audit_runner(
        repo_root=package_root,
        case_root=root,
        authorization_path=state_root / "authorization.json",
        expected_case_id=str(state["case_id"]),
        references_path=references,
        audio_dir=root / "audio",
        work_dir=work_root,
        on_model_start=progress.model_start,
        on_sample_complete=progress.sample_complete,
        on_model_complete=progress.model_complete,
    )

    reviews = _load_reviews(state_root)
    with progress.spinner("Analyzing accessibility-critical failures"):
        report_data = build_report_data(
            case_name=str(state["case_name"]),
            case_id=str(state["case_id"]),
            references_path=references,
            prediction_paths=audit_result.prediction_paths,
            result_paths=audit_result.result_paths,
            reviews=reviews,
        )
    progress.check(f"{len(report_data['findings'])} failures identified")

    output = root / "audit-report"
    with tempfile.TemporaryDirectory(dir=root, prefix=".deafbench-report-") as temporary:
        staging = Path(temporary)
        with progress.spinner("Building customer reports"):
            write_reports(report_data, staging)
            from deafbench.pilot.zero_custody import load_execution_attestation

            execution_attestation = load_execution_attestation(
                state_root / "attestation.json"
            )
            exporter(
                repo_root=package_root,
                result_paths=list(audit_result.result_paths),
                output_dir=staging / "evidence",
                signing_key=state_root / "signing-key.pem",
                execution_attestation=execution_attestation,
            )
        progress.check("HTML report")
        progress.check("PDF report")
        progress.check("Evidence manifest verified")
        _replace_report_output(staging, output)

    _write_json(state_root / "report-data.json", report_data)
    _write_json(
        state_root / "latest-run.json",
        {
            "schema_version": 1,
            "run_root": str(run_root),
            "report_generated_at": report_data["generated_at"],
        },
    )
    print("\nAudit complete.", file=stream)
    if not no_open:
        print("Opening report...", file=stream)
        _open_report(output / "index.html", browser_opener, stream)
    else:
        print(f"Report: {output / 'index.html'}", file=stream)
    return 0


def _apply_reviews(
    report_data: dict[str, Any], reviews: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    for finding in report_data.get("findings", []):
        finding_id = str(finding.get("finding_id", ""))
        review = dict(reviews.get(finding_id, {}))
        finding["review"] = review
        finding["effective_severity"] = str(
            review.get("customer_severity", finding["severity"])
        )
    report_data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report_data


def run_review(
    case_root: Path,
    *,
    input_fn: InputFunction = input,
    browser_opener: BrowserOpener = webbrowser.open,
    stream: TextIO = sys.stdout,
    no_open: bool = False,
) -> int:
    root = validate_case_root(Path(case_root))
    state_root = _state_dir(root)
    state = _read_json(state_root / "case.json")
    report_data = _read_json(state_root / "report-data.json")
    findings = report_data.get("findings")
    if not isinstance(findings, list):
        raise ValueError("No completed DeafBench audit is available for review")
    reviews = _load_reviews(state_root)
    changed = False
    print("DeafBench review", file=stream)
    print(str(state["case_name"]), file=stream)
    for index, finding in enumerate(findings, start=1):
        print(file=stream)
        print(f"Finding {index} of {len(findings)}", file=stream)
        print(category_label(str(finding["primary_category"])), file=stream)
        print(file=stream)
        print("Reference", file=stream)
        print(str(finding["reference_text"]), file=stream)
        print(file=stream)
        print("Caption", file=stream)
        print(str(finding["predicted_text"]), file=stream)
        print(file=stream)
        print("DeafBench severity", file=stream)
        print(severity_label(str(finding["severity"])), file=stream)
        print(file=stream)
        print("1. Keep", file=stream)
        print("2. Change severity", file=stream)
        print("3. Add context", file=stream)
        print("4. Skip", file=stream)
        print("5. Quit", file=stream)
        action = input_fn("Choose an action: ").strip()
        finding_id = str(finding["finding_id"])
        current = dict(reviews.get(finding_id, {}))
        if action == "1":
            current["reviewed"] = True
            current["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            reviews[finding_id] = current
            changed = True
            print("Review saved", file=stream)
        elif action == "2":
            allowed = ("no_real_impact", "minor", "moderate", "major", "critical")
            while True:
                raw = input_fn(
                    "New severity (no real impact, minor, moderate, major, critical): "
                ).strip().casefold().replace(" ", "_")
                if raw in allowed:
                    break
                print("Choose one of the listed severities.", file=stream)
            reason = _required_text("Reason: ", input_fn)
            current.update(
                {
                    "reviewed": True,
                    "customer_severity": raw,
                    "reason": reason,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            reviews[finding_id] = current
            changed = True
            print("Review saved", file=stream)
        elif action == "3":
            context = _required_text("Customer context: ", input_fn)
            current.update(
                {
                    "reviewed": True,
                    "context": context,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            reviews[finding_id] = current
            changed = True
            print("Review saved", file=stream)
        elif action == "4":
            continue
        elif action == "5":
            break
        else:
            print("Skipped; choose 1 through 5 next time.", file=stream)

    if not changed:
        print("\nNo review changes saved.", file=stream)
        return 0
    _write_json(state_root / "reviews.json", reviews)
    _sign_reviews(state_root, reviews)
    updated = _apply_reviews(report_data, reviews)
    _write_json(state_root / "report-data.json", updated)
    output = root / "audit-report"
    with tempfile.TemporaryDirectory(dir=root, prefix=".deafbench-review-") as temporary:
        staging = Path(temporary)
        write_reports(updated, staging)
        os.replace(staging / "index.html", output / "index.html")
        os.replace(staging / "report.pdf", output / "report.pdf")
    print("\nReview complete.", file=stream)
    print("HTML report updated", file=stream)
    print("PDF report updated", file=stream)
    if not no_open:
        print("Opening updated report...", file=stream)
        _open_report(output / "index.html", browser_opener, stream)
    return 0


def audit_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deafbench audit",
        description="Run a customer-local accessibility-critical caption audit.",
    )
    parser.add_argument("case", type=Path, help="Folder containing audio/ and references.csv")
    parser.add_argument("--name", help="Required case name on first run")
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML report")
    args = parser.parse_args(argv)
    try:
        return run_audit(args.case, case_name=args.name, no_open=args.no_open)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


def review_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deafbench review",
        description="Review customer context for findings from a completed audit.",
    )
    parser.add_argument("case", type=Path, help="Folder containing the completed audit")
    parser.add_argument("--no-open", action="store_true", help="Do not open the updated HTML report")
    args = parser.parse_args(argv)
    try:
        return run_review(args.case, no_open=args.no_open)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ISSUE_TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE"


def test_required_community_files_are_present() -> None:
    required = (
        ROOT / "CODE_OF_CONDUCT.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "LICENSE",
        ROOT / "SECURITY.md",
        ROOT / ".github" / "pull_request_template.md",
        ISSUE_TEMPLATE / "bug_report.yml",
        ISSUE_TEMPLATE / "feature_request.yml",
        ISSUE_TEMPLATE / "config.yml",
    )

    for path in required:
        assert path.is_file(), f"missing community standard: {path.relative_to(ROOT)}"
        assert path.read_text(encoding="utf-8").strip()


def test_issue_forms_fail_closed_for_sensitive_reports() -> None:
    config = yaml.safe_load((ISSUE_TEMPLATE / "config.yml").read_text(encoding="utf-8"))
    assert config["blank_issues_enabled"] is False
    links = {link["name"]: link["url"] for link in config["contact_links"]}
    assert links == {
        "Security vulnerability": (
            "https://github.com/488315/DeafBench/security/advisories/new"
        ),
        "Usage question": "https://github.com/488315/DeafBench/discussions",
    }

    expected_confirmations = {
        "bug_report.yml": {
            "safety": (
                "I removed customer data, raw transcripts, secrets, and private "
                "identifiers.",
                "This is not a confidential security vulnerability.",
            )
        },
        "feature_request.yml": {
            "scope": (
                "The request is about DeafBench's evaluation or audit workflow.",
                "The request does not require publishing customer or regulated data.",
            )
        },
    }

    for name, expected_by_id in expected_confirmations.items():
        form = yaml.safe_load((ISSUE_TEMPLATE / name).read_text(encoding="utf-8"))
        assert form["name"]
        assert form["description"]
        checkboxes = {
            item["id"]: item["attributes"]["options"]
            for item in form["body"]
            if item["type"] == "checkboxes"
        }
        assert checkboxes.keys() == expected_by_id.keys()
        for checkbox_id, expected_labels in expected_by_id.items():
            options = checkboxes[checkbox_id]
            assert tuple(option["label"] for option in options) == expected_labels
            assert all(option.get("required") is True for option in options)


def test_pull_request_template_preserves_review_gates() -> None:
    template = (ROOT / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "## Summary",
        "## Problem",
        "## Solution",
        "## Validation",
        "## Risk and rollback",
        "## Review focus",
    ):
        assert heading in template
    assert "customer data" in template.lower()
    assert "git diff --check" in template

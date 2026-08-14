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
    assert any("security/advisories/new" in link["url"] for link in config["contact_links"])

    for name in ("bug_report.yml", "feature_request.yml"):
        form = yaml.safe_load((ISSUE_TEMPLATE / name).read_text(encoding="utf-8"))
        assert form["name"]
        assert form["description"]
        rendered = str(form).lower()
        assert "customer" in rendered
        assert "sensitive" in rendered or "secret" in rendered
        required_checks = [
            option
            for item in form["body"]
            if item["type"] == "checkboxes"
            for option in item["attributes"]["options"]
        ]
        assert required_checks
        assert all(option.get("required") is True for option in required_checks)


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

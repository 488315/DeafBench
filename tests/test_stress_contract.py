import json
from pathlib import Path

import pytest

from deafbench.benchmark.stress_contract import load_stress_cases


def _write_case(path: Path, **changes: object) -> None:
    record: dict[str, object] = {
        "id": "stress-001",
        "text": "Meet Priya Shah at 8:30 PM.",
        "critical": ["Priya Shah", "8:30 PM"],
        "critical_types": {
            "Priya Shah": "PROPER_NAME",
            "8:30 PM": "TIME",
        },
        "risk_categories": {
            "Priya Shah": "PROPER_NAME",
            "8:30 PM": "TIME",
        },
        "sounds": [],
        "stressors": [
            {
                "kind": "additive_noise",
                "profile": "office-chatter",
                "snr_db": 0.0,
            }
        ],
    }
    record.update(changes)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_stress_contract_loads_a_typed_case(tmp_path: Path) -> None:
    path = tmp_path / "references.jsonl"
    _write_case(path)

    cases = load_stress_cases(path)

    assert cases[0]["id"] == "stress-001"
    assert cases[0]["stressors"][0]["snr_db"] == 0.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"critical": [], "critical_types": {}, "risk_categories": {}},
            "at least one critical term",
        ),
        ({"risk_categories": {}}, "exactly cover critical terms"),
        (
            {
                "risk_categories": {
                    "Priya Shah": "DEMOGRAPHIC_IDENTITY",
                    "8:30 PM": "TIME",
                }
            },
            "unsupported risk category",
        ),
        ({"stressors": []}, "at least one stressor"),
        ({"stressors": [{"kind": "unknown"}]}, "unsupported stressor"),
        (
            {
                "stressors": [
                    {
                        "kind": "additive_noise",
                        "profile": "office-chatter",
                        "snr_db": -4.0,
                    }
                ]
            },
            "unsupported SNR",
        ),
        (
            {
                "stressors": [
                    {
                        "kind": "interstitial_noise",
                        "profile": "keyboard-clicks",
                        "snr_db": 10.0,
                        "duration_seconds": 0.0,
                    }
                ]
            },
            "positive duration",
        ),
    ],
)
def test_stress_contract_rejects_invalid_cases(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "references.jsonl"
    _write_case(path, **changes)

    with pytest.raises(ValueError, match=message):
        load_stress_cases(path)


def test_stress_contract_rejects_unexpected_stressor_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "references.jsonl"
    _write_case(
        path,
        stressors=[
            {
                "kind": "telephony",
                "codec": "g711-mulaw",
                "sample_rate_hz": 8_000,
                "trust_me": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="unexpected fields"):
        load_stress_cases(path)

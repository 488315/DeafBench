"""Fail-closed schema for accessibility stress benchmark cases."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from deafbench.benchmark.noise import NOISE_PROFILES
from deafbench.benchmark.workspace import load_reference_records


RISK_CATEGORIES = frozenset(
    {
        "ADDRESS",
        "CODE",
        "DATE",
        "DIGIT_SEQUENCE",
        "DOSAGE",
        "MONEY",
        "NEGATION",
        "PROPER_NAME",
        "SSID",
        "TIME",
        "USERNAME",
    }
)
SUPPORTED_SNR_DB = frozenset({-5.0, 0.0, 10.0, 20.0})

_STRESSOR_FIELDS = {
    "clean": frozenset({"kind"}),
    "additive_noise": frozenset({"kind", "profile", "snr_db"}),
    "interstitial_noise": frozenset(
        {"kind", "profile", "snr_db", "duration_seconds"}
    ),
    "telephony": frozenset({"kind", "codec", "sample_rate_hz"}),
    "reverberation": frozenset({"kind", "rt60_seconds"}),
    "long_pause": frozenset({"kind", "duration_seconds"}),
    "rate": frozenset({"kind", "factor"}),
    "overlap": frozenset({"kind", "snr_db"}),
    "compression": frozenset({"kind", "codec", "bit_rate_kbps"}),
}


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Stress case {label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Stress case {label} must be a finite number")
    return number


def _validate_stressor(stressor: object) -> dict[str, Any]:
    if not isinstance(stressor, dict) or not isinstance(stressor.get("kind"), str):
        raise ValueError("Stress case stressors must be objects with a kind")
    kind = stressor["kind"]
    expected_fields = _STRESSOR_FIELDS.get(kind)
    if expected_fields is None:
        raise ValueError(f"Stress case has unsupported stressor: {kind}")
    actual_fields = set(stressor)
    if actual_fields != expected_fields:
        raise ValueError(f"Stress case {kind} has unexpected fields")

    if kind in {"additive_noise", "interstitial_noise"}:
        if stressor["profile"] not in NOISE_PROFILES:
            raise ValueError("Stress case has unsupported noise profile")
        snr_db = _finite_number(stressor["snr_db"], "SNR")
        if snr_db not in SUPPORTED_SNR_DB:
            raise ValueError("Stress case has unsupported SNR")
    if kind == "interstitial_noise":
        duration = _finite_number(stressor["duration_seconds"], "duration")
        if duration <= 0.0:
            raise ValueError("Stress case requires a positive duration")
    if kind == "telephony" and (
        stressor["codec"] != "g711-mulaw" or stressor["sample_rate_hz"] != 8_000
    ):
        raise ValueError("Stress case telephony profile must use 8 kHz G.711 mu-law")
    if kind == "reverberation" and not 0.1 <= _finite_number(
        stressor["rt60_seconds"], "RT60"
    ) <= 2.0:
        raise ValueError("Stress case RT60 must be between 0.1 and 2.0 seconds")
    if kind == "long_pause" and _finite_number(
        stressor["duration_seconds"], "duration"
    ) <= 0.0:
        raise ValueError("Stress case requires a positive duration")
    if kind == "rate" and not 0.5 <= _finite_number(
        stressor["factor"], "rate factor"
    ) <= 2.0:
        raise ValueError("Stress case rate factor must be between 0.5 and 2.0")
    if kind == "overlap" and _finite_number(
        stressor["snr_db"], "overlap SNR"
    ) not in SUPPORTED_SNR_DB:
        raise ValueError("Stress case has unsupported SNR")
    if kind == "compression" and (
        stressor["codec"] not in {"mp3", "opus"}
        or stressor["bit_rate_kbps"] not in {16, 24, 32, 64}
    ):
        raise ValueError("Stress case has unsupported compression profile")
    return dict(stressor)


def load_stress_cases(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Load reference records plus the model-independent stress contract."""
    validated: list[Mapping[str, Any]] = []
    for record in load_reference_records(path):
        critical = record["critical"]
        if not critical:
            raise ValueError("Stress case requires at least one critical term")
        risks = record.get("risk_categories")
        if not isinstance(risks, dict) or set(risks) != set(critical):
            raise ValueError("Stress case risk categories must exactly cover critical terms")
        if not all(
            isinstance(value, str) and value in RISK_CATEGORIES
            for value in risks.values()
        ):
            raise ValueError("Stress case has unsupported risk category")
        stressors = record.get("stressors")
        if not isinstance(stressors, list) or not stressors:
            raise ValueError("Stress case requires at least one stressor")
        normalized = dict(record)
        normalized["risk_categories"] = dict(risks)
        normalized["stressors"] = [_validate_stressor(item) for item in stressors]
        validated.append(normalized)
    return tuple(validated)

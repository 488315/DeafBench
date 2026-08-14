import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from deafbench.benchmark.stress_runner import prepare_stress_audio


def _write_fixture(tmp_path, stressor):
    references = tmp_path / "references.jsonl"
    row = {
        "id": "stress-001",
        "text": "Meet Priya at 8:30 PM.",
        "critical": ["Priya", "8:30 PM"],
        "critical_types": {"Priya": "PROPER_NAME", "8:30 PM": "TIME"},
        "risk_categories": {"Priya": "PROPER_NAME", "8:30 PM": "TIME"},
        "sounds": [],
        "stressors": [{"kind": "clean"}, stressor],
    }
    references.write_text(json.dumps(row) + "\n", encoding="utf-8")
    audio = tmp_path / "audio-clean"
    audio.mkdir()
    time = np.arange(8_000, dtype=np.float32) / 16_000
    sf.write(
        audio / "stress-001.wav",
        0.1 * np.sin(2 * np.pi * 220 * time),
        16_000,
        subtype="PCM_16",
    )
    return references, audio


@pytest.mark.parametrize(
    "stressor",
    [
        {"kind": "additive_noise", "profile": "street-noise", "snr_db": 0.0},
        {
            "kind": "interstitial_noise",
            "profile": "keyboard-clicks",
            "snr_db": 10.0,
            "duration_seconds": 0.1,
        },
        {"kind": "telephony", "codec": "g711-mulaw", "sample_rate_hz": 8_000},
        {"kind": "reverberation", "rt60_seconds": 0.2},
        {"kind": "long_pause", "duration_seconds": 0.1},
        {"kind": "rate", "factor": 1.5},
    ],
)
def test_prepare_stress_audio_materializes_supported_pair(tmp_path, stressor):
    references, audio = _write_fixture(tmp_path, stressor)
    destination = tmp_path / "prepared"

    manifest = prepare_stress_audio(references, audio, destination)

    clean = destination / "clean" / "stress-001.wav"
    stressed = destination / "stressed" / "stress-001.wav"
    assert clean.is_file()
    assert stressed.is_file()
    assert manifest["sample_count"] == 1
    assert manifest["samples"][0]["clean_sha256"] == hashlib.sha256(
        clean.read_bytes()
    ).hexdigest()
    assert manifest["samples"][0]["stressed_sha256"] == hashlib.sha256(
        stressed.read_bytes()
    ).hexdigest()
    assert json.loads(
        (destination / "preparation-manifest.json").read_text(encoding="utf-8")
    ) == manifest
    if stressor["kind"] == "interstitial_noise":
        assert manifest["samples"][0]["noise_only_interval"]["end_frame"] > 0


@pytest.mark.parametrize(
    "stressor",
    [
        {"kind": "overlap", "snr_db": 0.0},
        {"kind": "compression", "codec": "opus", "bit_rate_kbps": 16},
    ],
)
def test_prepare_stress_audio_rejects_unimplemented_stressor(tmp_path, stressor):
    references, audio = _write_fixture(tmp_path, stressor)

    with pytest.raises(ValueError, match="unsupported stressors"):
        prepare_stress_audio(references, audio, tmp_path / "prepared")

    assert not (tmp_path / "prepared").exists()


def test_prepare_stress_audio_preserves_existing_destination(tmp_path):
    references, audio = _write_fixture(
        tmp_path,
        {"kind": "additive_noise", "profile": "wind", "snr_db": 0.0},
    )
    destination = tmp_path / "prepared"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        prepare_stress_audio(references, audio, destination)

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("case_ids", [[], ["stress-001", "stress-001"]])
def test_prepare_stress_audio_rejects_empty_or_duplicate_selection(
    tmp_path, case_ids
):
    references, audio = _write_fixture(
        tmp_path,
        {"kind": "additive_noise", "profile": "wind", "snr_db": 0.0},
    )

    with pytest.raises(ValueError, match="unique selected"):
        prepare_stress_audio(
            references,
            audio,
            tmp_path / "prepared",
            case_ids=case_ids,
        )


def test_prepare_stress_audio_rejects_unknown_selection(tmp_path):
    references, audio = _write_fixture(
        tmp_path,
        {"kind": "additive_noise", "profile": "wind", "snr_db": 0.0},
    )

    with pytest.raises(ValueError, match="unknown case IDs"):
        prepare_stress_audio(
            references,
            audio,
            tmp_path / "prepared",
            case_ids=["stress-999"],
        )


def test_prepare_stress_audio_rejects_more_than_two_lanes(tmp_path):
    references, audio = _write_fixture(
        tmp_path,
        {"kind": "additive_noise", "profile": "wind", "snr_db": 0.0},
    )
    record = json.loads(references.read_text(encoding="utf-8"))
    record["stressors"].append({"kind": "rate", "factor": 1.5})
    references.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="one clean and one stressed lane"):
        prepare_stress_audio(references, audio, tmp_path / "prepared")


def test_prepare_stress_audio_rejects_missing_audio_and_cleans_staging(tmp_path):
    references, audio = _write_fixture(
        tmp_path,
        {"kind": "additive_noise", "profile": "wind", "snr_db": 0.0},
    )
    (audio / "stress-001.wav").unlink()

    with pytest.raises(ValueError, match="missing clean audio"):
        prepare_stress_audio(references, audio, tmp_path / "prepared")

    assert not list(tmp_path.glob(".prepared-prepare-*"))


def test_prepare_stress_audio_rejects_unreadable_audio(tmp_path):
    references, audio = _write_fixture(
        tmp_path,
        {"kind": "additive_noise", "profile": "wind", "snr_db": 0.0},
    )
    (audio / "stress-001.wav").write_bytes(b"not-wave-audio")

    with pytest.raises(ValueError, match="cannot read clean audio"):
        prepare_stress_audio(references, audio, tmp_path / "prepared")

import json
from pathlib import Path

import numpy as np
import pytest

from tools.recorder.core import (
    DEFAULT_SAMPLE_RATE,
    SOUND_EVENT_GAP_MS,
    SUPPORTED_SOUND_EVENTS,
    append_sound_events,
    synthesize_sound_event,
)
from tools.recorder.recorder import resolve_dataset_paths


EXPECTED_SOUND_EVENTS = {
    "[alarm]",
    "[door closes]",
    "[phone rings]",
    "[knock]",
    "[error notification]",
    "[siren]",
}


def test_supported_sound_event_labels_are_stable():
    assert set(SUPPORTED_SOUND_EVENTS) == EXPECTED_SOUND_EVENTS


@pytest.mark.parametrize("label", sorted(EXPECTED_SOUND_EVENTS))
def test_synthetic_sound_event_is_nonempty_int16_mono(label):
    event = synthesize_sound_event(label)

    assert event.dtype == np.int16
    assert event.ndim == 2
    assert event.shape[1] == 1
    assert event.shape[0] > 0
    assert np.max(np.abs(event.astype(np.int32))) > 0


def test_append_sound_events_follows_label_order_after_recording():
    speech = np.array([[100], [-100], [50]], dtype=np.int16)
    labels = ["[knock]", "[phone rings]"]
    gap_frames = DEFAULT_SAMPLE_RATE * SOUND_EVENT_GAP_MS // 1000
    gap = np.zeros((gap_frames, 1), dtype=np.int16)

    result = append_sound_events(speech, labels)

    expected = np.concatenate(
        [
            speech,
            gap,
            synthesize_sound_event("[knock]"),
            gap,
            synthesize_sound_event("[phone rings]"),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(result, expected)


def test_append_sound_events_leaves_recording_unchanged_without_labels():
    speech = np.array([[100, 300], [-200, 200]], dtype=np.int16)

    result = append_sound_events(speech, [])

    assert result.tolist() == [[200], [0]]


def test_append_sound_events_rejects_unknown_label():
    with pytest.raises(ValueError, match="Unsupported sound event"):
        append_sound_events(np.array([[0]], dtype=np.int16), ["[unknown]"])


def test_resolve_dataset_paths_supports_non_speech_v1(tmp_path):
    references, audio_dir = resolve_dataset_paths(tmp_path, "non-speech-v1")

    assert references == tmp_path / "benchmarks" / "non-speech-v1" / "references.jsonl"
    assert audio_dir == tmp_path / "benchmarks" / "non-speech-v1" / "audio"


def test_non_speech_v1_references_use_supported_sound_labels():
    references = Path(__file__).resolve().parents[1] / "benchmarks" / "non-speech-v1" / "references.jsonl"
    records = [json.loads(line) for line in references.read_text(encoding="utf-8").splitlines() if line]

    assert len(records) >= 10
    assert all(record["id"].startswith("ns-") for record in records)
    assert all(record.get("sounds") for record in records)

    labels = {label for record in records for label in record["sounds"]}
    assert labels == EXPECTED_SOUND_EVENTS

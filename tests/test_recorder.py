import json
import wave

import numpy as np
import pytest

from tools.recorder import recorder as recorder_module
from tools.recorder.core import (
    atomic_write_wav,
    downmix_to_mono,
    find_preferred_input_device,
    is_recorded,
    load_prompts,
    next_unrecorded_index,
    output_path,
)
from tools.recorder.recorder import AudioRecorder, resolve_dataset_paths


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_load_prompts_accepts_valid_records(tmp_path):
    path = tmp_path / "references.jsonl"
    _write_jsonl(path, [
        {"id": "core-001", "text": "Read this sentence.", "critical": [], "sounds": []},
        {"id": "core-002", "text": "Read the next sentence.", "critical": ["next"]},
    ])

    prompts = load_prompts(path)

    assert [item["id"] for item in prompts] == ["core-001", "core-002"]
    assert prompts[1]["text"] == "Read the next sentence."


@pytest.mark.parametrize(
    "record, message",
    [
        ({"text": "Missing ID"}, "id"),
        ({"id": "", "text": "Empty ID"}, "id"),
        ({"id": "core-001"}, "text"),
        ({"id": "core-001", "text": 123}, "text"),
    ],
)
def test_load_prompts_rejects_invalid_records(tmp_path, record, message):
    path = tmp_path / "references.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match=message):
        load_prompts(path)


def test_load_prompts_reports_json_line_number(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text('{"id":"core-001","text":"ok"}\n{bad json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_prompts(path)


def test_load_prompts_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "references.jsonl"
    _write_jsonl(path, [
        {"id": "core-001", "text": "First"},
        {"id": "core-001", "text": "Second"},
    ])

    with pytest.raises(ValueError, match="Duplicate sample ID: core-001"):
        load_prompts(path)


def test_load_prompts_rejects_unsupported_sound_labels(tmp_path):
    path = tmp_path / "references.jsonl"
    _write_jsonl(path, [
        {
            "id": "ns-001",
            "text": "Read this sentence.",
            "sounds": ["[unknown]"],
        }
    ])

    with pytest.raises(ValueError, match=r"Unsupported sound event.*\[unknown\]"):
        load_prompts(path)


def test_output_path_and_recorded_status(tmp_path):
    audio_dir = tmp_path / "audio"
    expected = audio_dir / "core-006.wav"

    assert output_path(audio_dir, "core-006") == expected
    assert is_recorded(audio_dir, "core-006") is False

    audio_dir.mkdir()
    expected.write_bytes(b"wav")
    assert is_recorded(audio_dir, "core-006") is True


def test_next_unrecorded_index_prefers_later_missing_sample(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    prompts = [{"id": f"core-{index:03d}", "text": "x"} for index in range(1, 5)]
    (audio_dir / "core-002.wav").write_bytes(b"done")
    (audio_dir / "core-003.wav").write_bytes(b"done")

    assert next_unrecorded_index(prompts, audio_dir, 1) == 3


def test_next_unrecorded_index_does_not_wrap(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    prompts = [{"id": "core-001", "text": "x"}, {"id": "core-002", "text": "y"}]
    (audio_dir / "core-002.wav").write_bytes(b"done")

    assert next_unrecorded_index(prompts, audio_dir, 1) is None


def test_find_preferred_input_device_ignores_output_only_devices():
    devices = [
        {"name": "Voicemeeter Out B3", "max_input_channels": 0},
        {"name": "Microphone", "max_input_channels": 2},
        {"name": "Voicemeeter Out B3 (VB-Audio)", "max_input_channels": 8},
    ]

    assert find_preferred_input_device(devices) == 2


def test_find_preferred_input_device_returns_none_without_b3():
    devices = [{"name": "Microphone", "max_input_channels": 2}]
    assert find_preferred_input_device(devices) is None


def test_downmix_to_mono_averages_stereo_and_returns_int16():
    stereo = np.array([[1000, 3000], [-2000, 2000], [32767, 32767]], dtype=np.int16)

    mono = downmix_to_mono(stereo)

    assert mono.dtype == np.int16
    assert mono.shape == (3, 1)
    assert mono[:, 0].tolist() == [2000, 0, 32767]


def test_downmix_to_mono_preserves_mono_shape():
    mono = np.array([[100], [-100]], dtype=np.int16)
    result = downmix_to_mono(mono)
    assert result.dtype == np.int16
    assert result.tolist() == [[100], [-100]]


def test_atomic_write_wav_writes_48khz_16bit_mono(tmp_path):
    path = tmp_path / "audio" / "core-001.wav"
    samples = np.array([[100], [-100], [0]], dtype=np.int16)

    atomic_write_wav(path, samples)

    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48000
        assert wav_file.getnframes() == 3


def test_atomic_write_wav_replaces_existing_file_only_after_success(tmp_path, monkeypatch):
    path = tmp_path / "core-001.wav"
    old = np.array([[1], [2]], dtype=np.int16)
    new = np.array([[3], [4]], dtype=np.int16)
    atomic_write_wav(path, old)
    original_bytes = path.read_bytes()

    real_replace = __import__("os").replace

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("tools.recorder.core.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_wav(path, new)

    assert path.read_bytes() == original_bytes

    monkeypatch.setattr("tools.recorder.core.os.replace", real_replace)
    atomic_write_wav(path, new)
    assert path.read_bytes() != original_bytes


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakeAudioBackend:
    def __init__(self):
        self.checked = None
        self.stream = None

    def check_input_settings(self, **kwargs):
        self.checked = kwargs

    def InputStream(self, **kwargs):
        self.stream = _FakeStream(**kwargs)
        return self.stream


def test_audio_recorder_configures_48khz_int16_stream():
    backend = _FakeAudioBackend()
    recorder = AudioRecorder(backend=backend)

    recorder.start(device_index=4, channels=2)

    assert backend.checked == {
        "device": 4,
        "channels": 2,
        "dtype": "int16",
        "samplerate": 48000,
    }
    assert backend.stream.started is True
    assert backend.stream.kwargs["device"] == 4
    assert backend.stream.kwargs["channels"] == 2
    assert backend.stream.kwargs["dtype"] == "int16"
    assert backend.stream.kwargs["samplerate"] == 48000

    backend.stream.kwargs["callback"](
        np.array([[100, 300], [200, 400]], dtype=np.int16),
        2,
        None,
        None,
    )
    captured = recorder.stop()

    assert backend.stream.stopped is True
    assert backend.stream.closed is True
    assert captured.tolist() == [[100, 300], [200, 400]]


def test_resolve_dataset_paths_uses_core_v1_layout(tmp_path):
    references, audio_dir = resolve_dataset_paths(tmp_path)
    assert references == tmp_path / "benchmarks" / "core-v1" / "references.jsonl"
    assert audio_dir == tmp_path / "benchmarks" / "core-v1" / "audio"


@pytest.mark.parametrize("dataset", ["C:temp", "bad:name"])
def test_resolve_dataset_paths_rejects_drive_qualified_names(tmp_path, dataset):
    with pytest.raises(ValueError, match="Invalid dataset name"):
        resolve_dataset_paths(tmp_path, dataset)


def test_main_reports_invalid_dataset_as_parser_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(recorder_module, "_sounddevice", object())

    with pytest.raises(SystemExit) as exc_info:
        recorder_module.main([
            "--repo-root",
            str(tmp_path),
            "--dataset",
            "C:temp",
        ])

    assert exc_info.value.code == 2
    assert "Invalid dataset name" in capsys.readouterr().err

"""Windows GUI recorder for DeafBench benchmark audio."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    DEFAULT_SAMPLE_RATE,
    atomic_write_wav,
    find_preferred_input_device,
    is_recorded,
    load_prompts,
    next_unrecorded_index,
    output_path,
)


try:
    import sounddevice as _sounddevice
except ImportError:  # pragma: no cover - exercised on user machines
    _sounddevice = None


FORMAT_TEXT = "48 kHz · 16-bit PCM · mono"


def resolve_dataset_paths(repo_root: Path) -> tuple[Path, Path]:
    """Return the default Core v1 references and audio paths."""
    dataset_dir = Path(repo_root) / "benchmarks" / "core-v1"
    return dataset_dir / "references.jsonl", dataset_dir / "audio"


class AudioRecorder:
    """Small sounddevice wrapper with no GUI dependencies."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend if backend is not None else _sounddevice
        self._stream = None
        self._blocks: list[np.ndarray] = []
        self._channels = 1
        self._started_at: float | None = None
        self._peak_level = 0.0
        self._callback_statuses: list[str] = []
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def duration(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def peak_level(self) -> float:
        with self._lock:
            return self._peak_level

    def start(self, device_index: int, channels: int) -> None:
        if self.backend is None:
            raise RuntimeError("sounddevice is not installed")
        if self.is_recording:
            raise RuntimeError("A recording is already in progress")
        if channels not in (1, 2):
            raise ValueError("Recorder supports one or two capture channels")

        self.backend.check_input_settings(
            device=device_index,
            channels=channels,
            dtype="int16",
            samplerate=DEFAULT_SAMPLE_RATE,
        )

        self._channels = channels
        self._blocks = []
        self._peak_level = 0.0
        self._callback_statuses = []

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            block = np.asarray(indata, dtype=np.int16).copy()
            with self._lock:
                if status:
                    self._callback_statuses.append(str(status))
                self._blocks.append(block)
                if block.size:
                    self._peak_level = min(
                        1.0,
                        float(np.max(np.abs(block.astype(np.int32)))) / 32767.0,
                    )

        self._stream = self.backend.InputStream(
            device=device_index,
            channels=channels,
            dtype="int16",
            samplerate=DEFAULT_SAMPLE_RATE,
            callback=callback,
        )
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise
        self._started_at = time.monotonic()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("No recording is in progress")

        stream = self._stream
        self._stream = None
        try:
            stream.stop()
        finally:
            stream.close()

        with self._lock:
            blocks = list(self._blocks)
            statuses = list(self._callback_statuses)
            self._blocks.clear()
            self._callback_statuses.clear()
            self._peak_level = 0.0
        self._started_at = None

        if statuses:
            raise RuntimeError("Audio capture reported status: " + "; ".join(statuses))
        if not blocks:
            return np.empty((0, self._channels), dtype=np.int16)
        return np.concatenate(blocks, axis=0)


class RecorderApp:
    """Tkinter application for recording a DeafBench JSONL prompt set."""

    def __init__(
        self,
        root: Any,
        references_path: Path,
        audio_dir: Path,
        backend: Any = None,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = root
        self.references_path = Path(references_path)
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = load_prompts(self.references_path)
        self.current_index = 0
        self.retry_mode = False
        self._closing = False
        self.backend = backend if backend is not None else _sounddevice
        self.recorder = AudioRecorder(self.backend)
        self.devices: list[dict[str, Any]] = []
        self.device_indices: list[int] = []

        self.root.title("DeafBench Dataset Recorder")
        self.root.geometry("1040x650")
        self.root.minsize(860, 540)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._load_devices()
        self._refresh_sample_list()
        self._select_sample(0)
        self._tick()

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 18))
        ttk.Label(left, text="Samples", font=("Segoe UI", 11, "bold")).pack(anchor="w")

        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True, pady=(8, 0))
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.sample_list = tk.Listbox(
            list_frame,
            width=25,
            height=24,
            exportselection=False,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
        )
        scrollbar.config(command=self.sample_list.yview)
        self.sample_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.sample_list.bind("<<ListboxSelect>>", self._on_sample_selected)

        main = ttk.Frame(outer)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)

        self.sample_id_var = tk.StringVar()
        ttk.Label(main, textvariable=self.sample_id_var, font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.prompt_var = tk.StringVar()
        ttk.Label(
            main,
            textvariable=self.prompt_var,
            font=("Segoe UI", 22),
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(18, 26))

        device_frame = ttk.LabelFrame(main, text="Recording input", padding=12)
        device_frame.grid(row=2, column=0, sticky="ew")
        device_frame.columnconfigure(0, weight=1)

        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.device_var,
            state="readonly",
        )
        self.device_combo.grid(row=0, column=0, sticky="ew")
        self.device_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_controls())

        ttk.Label(device_frame, text=FORMAT_TEXT).grid(row=1, column=0, sticky="w", pady=(8, 0))

        meter_frame = ttk.Frame(main)
        meter_frame.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        meter_frame.columnconfigure(0, weight=1)
        self.level = ttk.Progressbar(meter_frame, maximum=100, mode="determinate")
        self.level.grid(row=0, column=0, sticky="ew")
        self.duration_var = tk.StringVar(value="0.0 s")
        ttk.Label(meter_frame, textvariable=self.duration_var, width=8).grid(
            row=0, column=1, padx=(12, 0)
        )

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).grid(row=4, column=0, sticky="w", pady=(10, 0))

        controls = ttk.Frame(main)
        controls.grid(row=5, column=0, sticky="ew", pady=(28, 0))
        self.previous_button = ttk.Button(controls, text="Previous", command=self.previous_sample)
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(controls, text="Next", command=self.next_sample)
        self.next_button.pack(side="left", padx=(8, 0))
        self.retry_button = ttk.Button(controls, text="Retry", command=self.retry_selected)
        self.retry_button.pack(side="right")
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_recording)
        self.stop_button.pack(side="right", padx=(0, 8))
        self.record_button = ttk.Button(controls, text="Record", command=self.start_recording)
        self.record_button.pack(side="right", padx=(0, 8))

    def _load_devices(self) -> None:
        if self.backend is None:
            self.status_var.set("Install recorder dependencies to access audio devices")
            self.device_combo["values"] = []
            self._update_controls()
            return

        try:
            all_devices = list(self.backend.query_devices())
        except Exception as exc:
            self.status_var.set(f"Could not enumerate audio devices: {exc}")
            self._update_controls()
            return

        self.devices = [dict(device) for device in all_devices]
        labels: list[str] = []
        self.device_indices = []
        for index, device in enumerate(self.devices):
            if int(device.get("max_input_channels", 0) or 0) > 0:
                self.device_indices.append(index)
                labels.append(f"{index}: {device.get('name', 'Unknown input')}")

        self.device_combo["values"] = labels
        preferred = find_preferred_input_device(self.devices)
        if preferred is not None and preferred in self.device_indices:
            position = self.device_indices.index(preferred)
            self.device_combo.current(position)
            self.status_var.set(f"Selected {self.devices[preferred]['name']}")
        elif labels:
            self.device_combo.set("")
            self.status_var.set("Voicemeeter Out B3 not found. Select an input device.")
        else:
            self.status_var.set("No input-capable audio devices found")
        self._update_controls()

    def _selected_device_index(self) -> int | None:
        position = self.device_combo.current()
        if position < 0 or position >= len(self.device_indices):
            return None
        return self.device_indices[position]

    def _selected_capture_channels(self, device_index: int) -> int:
        max_channels = int(self.devices[device_index].get("max_input_channels", 0) or 0)
        if max_channels < 1:
            raise RuntimeError("Selected device has no input channels")
        return 2 if max_channels >= 2 else 1

    def _refresh_sample_list(self) -> None:
        selected = self.current_index
        self.sample_list.delete(0, self.tk.END)
        for prompt in self.prompts:
            sample_id = str(prompt["id"])
            marker = "✓" if is_recorded(self.audio_dir, sample_id) else "○"
            self.sample_list.insert(self.tk.END, f"{marker} {sample_id}")
        self.sample_list.selection_clear(0, self.tk.END)
        self.sample_list.selection_set(selected)
        self.sample_list.see(selected)

    def _select_sample(self, index: int) -> None:
        if not 0 <= index < len(self.prompts):
            return
        if self.recorder.is_recording:
            return
        self.current_index = index
        prompt = self.prompts[index]
        self.sample_id_var.set(str(prompt["id"]))
        self.prompt_var.set(str(prompt["text"]))
        self.retry_mode = False
        self._refresh_sample_list()
        self._update_controls()

    def _on_sample_selected(self, _event: Any) -> None:
        selected = self.sample_list.curselection()
        if selected:
            self._select_sample(int(selected[0]))

    def previous_sample(self) -> None:
        self._select_sample(self.current_index - 1)

    def next_sample(self) -> None:
        self._select_sample(self.current_index + 1)

    def retry_selected(self) -> None:
        sample_id = str(self.prompts[self.current_index]["id"])
        if not is_recorded(self.audio_dir, sample_id):
            self.status_var.set("This sample has no recording yet. Use Record.")
            return
        self.retry_mode = True
        self.status_var.set(f"Retrying {sample_id}. Existing WAV stays safe until Stop.")
        self._start_capture()

    def start_recording(self) -> None:
        sample_id = str(self.prompts[self.current_index]["id"])
        if is_recorded(self.audio_dir, sample_id):
            self.status_var.set("Recording already exists. Use Retry to replace it.")
            return
        self.retry_mode = False
        self._start_capture()

    def _start_capture(self) -> None:
        device_index = self._selected_device_index()
        if device_index is None:
            self.messagebox.showerror("Recording input", "Select an input device before recording.")
            return

        try:
            channels = self._selected_capture_channels(device_index)
            self.recorder.start(device_index=device_index, channels=channels)
        except Exception as exc:
            self.retry_mode = False
            self.messagebox.showerror("Could not start recording", str(exc))
            self.status_var.set("Recording did not start")
            self._update_controls()
            return

        sample_id = str(self.prompts[self.current_index]["id"])
        mode = "Retrying" if self.retry_mode else "Recording"
        self.status_var.set(f"{mode} {sample_id}…")
        self._update_controls()

    def stop_recording(self) -> None:
        if not self.recorder.is_recording:
            return

        sample_id = str(self.prompts[self.current_index]["id"])
        try:
            captured = self.recorder.stop()
            if captured.shape[0] == 0:
                raise RuntimeError("No audio was captured. Please try again.")
            destination = output_path(self.audio_dir, sample_id)
            atomic_write_wav(destination, captured)
        except Exception as exc:
            self.retry_mode = False
            self.messagebox.showerror("Could not save recording", str(exc))
            self.status_var.set(f"{sample_id} was not replaced or advanced")
            self._update_controls()
            return

        was_retry = self.retry_mode
        self.retry_mode = False
        self._refresh_sample_list()

        next_index = next_unrecorded_index(self.prompts, self.audio_dir, self.current_index)
        if next_index is None and self.current_index + 1 < len(self.prompts):
            next_index = self.current_index + 1

        if next_index is not None:
            self._select_sample(next_index)
            action = "Replaced" if was_retry else "Saved"
            self.status_var.set(f"{action} {sample_id}. Ready for {self.prompts[next_index]['id']}.")
        else:
            self.status_var.set(f"Saved {sample_id}. No later samples remain.")
        self._update_controls()

    def _update_controls(self) -> None:
        recording = self.recorder.is_recording
        sample_id = str(self.prompts[self.current_index]["id"])
        recorded = is_recorded(self.audio_dir, sample_id)
        has_device = self._selected_device_index() is not None

        self.record_button.config(state="disabled" if recording or recorded or not has_device else "normal")
        self.stop_button.config(state="normal" if recording else "disabled")
        self.retry_button.config(state="disabled" if recording or not recorded or not has_device else "normal")
        self.previous_button.config(state="disabled" if recording or self.current_index == 0 else "normal")
        self.next_button.config(
            state="disabled" if recording or self.current_index >= len(self.prompts) - 1 else "normal"
        )
        self.device_combo.config(state="disabled" if recording else "readonly")

    def _tick(self) -> None:
        if self._closing:
            return

        try:
            if self.recorder.is_recording:
                self.duration_var.set(f"{self.recorder.duration:.1f} s")
                self.level["value"] = self.recorder.peak_level * 100.0
            else:
                self.level["value"] = 0
                if self.duration_var.get() != "0.0 s":
                    self.duration_var.set("0.0 s")
            self.root.after(100, self._tick)
        except self.tk.TclError:
            self._closing = True

    def _on_close(self) -> None:
        if self.recorder.is_recording:
            self.messagebox.showwarning("Recording in progress", "Stop the current recording before closing.")
            return
        self._closing = True
        self.root.destroy()


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record DeafBench benchmark prompts")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--references", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _sounddevice is None:
        raise SystemExit(
            "sounddevice is not installed. Run: python -m pip install -r tools/recorder/requirements.txt"
        )

    default_references, default_audio_dir = resolve_dataset_paths(args.repo_root)
    references_path = args.references or default_references
    audio_dir = args.audio_dir or default_audio_dir

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    try:
        RecorderApp(root, references_path, audio_dir)
    except Exception as exc:
        root.withdraw()
        messagebox.showerror("DeafBench Dataset Recorder", str(exc))
        root.destroy()
        return 1

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
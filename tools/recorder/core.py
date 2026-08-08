"""Compatibility exports for the packaged DeafBench recorder core."""

from deafbench.recorder.core import (
    DEFAULT_DEVICE_NEEDLE,
    DEFAULT_SAMPLE_RATE,
    SOUND_EVENT_GAP_MS,
    SUPPORTED_SOUND_EVENTS,
    append_sound_events,
    atomic_write_wav,
    downmix_to_mono,
    find_preferred_input_device,
    is_recorded,
    load_prompts,
    next_unrecorded_index,
    output_path,
    synthesize_sound_event,
)


__all__ = [
    "DEFAULT_DEVICE_NEEDLE",
    "DEFAULT_SAMPLE_RATE",
    "SOUND_EVENT_GAP_MS",
    "SUPPORTED_SOUND_EVENTS",
    "append_sound_events",
    "atomic_write_wav",
    "downmix_to_mono",
    "find_preferred_input_device",
    "is_recorded",
    "load_prompts",
    "next_unrecorded_index",
    "output_path",
    "synthesize_sound_event",
]

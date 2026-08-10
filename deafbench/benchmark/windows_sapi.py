"""Windows SAPI replacement speech backend for synthetic-v2 construction."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

import soundfile as sf

from .spoken_reference import SpokenReference
from .synthetic_v2_corpus import GeneratedSpeech


_SCRIPT = r"""
param(
    [Parameter(Mandatory=$true)][string]$SsmlPath,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [Parameter(Mandatory=$true)][string]$VoiceName,
    [Parameter(Mandatory=$true)][int]$VoiceRate
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$speaker = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $speaker.SelectVoice($VoiceName)
    $speaker.Rate = $VoiceRate
    $speaker.Volume = 100
    $voice = $speaker.Voice
    $speaker.SetOutputToWaveFile($OutputPath)
    $speaker.SpeakSsml([System.IO.File]::ReadAllText($SsmlPath))
    $speaker.SetOutputToNull()
    [ordered]@{
        assembly = [System.Speech.Synthesis.SpeechSynthesizer].Assembly.FullName
        voice_name = $voice.Name
        voice_id = $voice.Id
        culture = $voice.Culture.Name
        gender = $voice.Gender.ToString()
        age = $voice.Age.ToString()
        rate = $VoiceRate
        volume = 100
    } | ConvertTo-Json -Compress
}
finally {
    $speaker.Dispose()
}
"""


class WindowsSapiGenerator:
    """Render prepared SSML with a named installed Windows voice."""

    def __init__(
        self,
        *,
        voice_name: str = "Microsoft Zira Desktop",
        powershell: str = "powershell.exe",
        rate: int = 4,
        sample_rates: Mapping[str, int] | None = None,
    ) -> None:
        if not -10 <= rate <= 10:
            raise ValueError("Windows SAPI rate must be between -10 and 10")
        resolved_sample_rates = dict(sample_rates or {})
        if any(not -10 <= value <= 10 for value in resolved_sample_rates.values()):
            raise ValueError("Windows SAPI sample rates must be between -10 and 10")
        executable = shutil.which(powershell)
        if executable is None:
            raise RuntimeError(f"Windows PowerShell is unavailable: {powershell}")
        self._powershell = executable
        self._voice_name = voice_name
        self._rate = rate
        self._sample_rates = resolved_sample_rates

    def generate(
        self,
        sample_id: str,
        prepared: SpokenReference,
    ) -> GeneratedSpeech:
        """Generate one waveform while retaining exact voice and alias metadata."""
        rate = self._sample_rates.get(sample_id, self._rate)
        with tempfile.TemporaryDirectory(prefix=f"deafbench-{sample_id}-") as raw_dir:
            work = Path(raw_dir)
            script = work / "synthesize.ps1"
            ssml = work / "reference.ssml"
            output = work / "speech.wav"
            script.write_text(_SCRIPT, encoding="utf-8")
            ssml.write_text(prepared.ssml, encoding="utf-8")
            completed = subprocess.run(
                [
                    self._powershell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-SsmlPath",
                    str(ssml),
                    "-OutputPath",
                    str(output),
                    "-VoiceName",
                    self._voice_name,
                    "-VoiceRate",
                    str(rate),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            voice = json.loads(completed.stdout.strip())
            samples, sample_rate = sf.read(output, dtype="float32", always_2d=True)
        return GeneratedSpeech(
            samples=samples,
            sample_rate=sample_rate,
            metadata={
                "engine": "windows-system-speech",
                "version": voice["assembly"],
                "voice": {key: value for key, value in voice.items() if key != "assembly"},
                "operating_system": platform.platform(),
                "synthesis_profile": "typed-ssml-v2",
                "tts_seed": None,
                "reference_sha256": prepared.reference_sha256,
                "spoken_aliases": dict(prepared.spoken_aliases),
            },
        )

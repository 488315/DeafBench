"""Transactional preparation for the accessibility stress lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, cast

import numpy as np
import soundfile as sf

from deafbench.benchmark.interstitial import build_interstitial_scene
from deafbench.benchmark.stress_audio import (
    add_noise_at_snr,
    apply_reverberation,
    insert_silence,
    simulate_telephony,
    vary_rate,
)
from deafbench.benchmark.stress_contract import load_stress_cases
from deafbench.benchmark.workspace import atomic_write_json


IMPLEMENTED_STRESSORS = frozenset(
    {
        "additive_noise",
        "interstitial_noise",
        "telephony",
        "reverberation",
        "long_pause",
        "rate",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_stressor(
    samples: np.ndarray,
    sample_rate: int,
    stressor: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, Mapping[str, int | float | str] | None]:
    kind = cast(str, stressor["kind"])
    if kind == "additive_noise":
        return (
            add_noise_at_snr(
                samples,
                cast(str, stressor["profile"]),
                cast(float, stressor["snr_db"]),
                sample_rate,
                seed,
            ),
            None,
        )
    if kind == "interstitial_noise":
        split = len(samples) // 2
        scene = build_interstitial_scene(
            samples[:split],
            samples[split:],
            profile=cast(str, stressor["profile"]),
            snr_db=cast(float, stressor["snr_db"]),
            duration_seconds=cast(float, stressor["duration_seconds"]),
            sample_rate=sample_rate,
            seed=seed,
        )
        return scene.samples, {
            "start_frame": scene.interval.start_frame,
            "end_frame": scene.interval.end_frame,
            "profile": scene.interval.profile,
            "snr_db": scene.interval.snr_db,
        }
    if kind == "telephony":
        return simulate_telephony(samples, sample_rate), None
    if kind == "reverberation":
        return (
            apply_reverberation(
                samples,
                sample_rate,
                cast(float, stressor["rt60_seconds"]),
            ),
            None,
        )
    if kind == "long_pause":
        return (
            insert_silence(
                samples,
                sample_rate,
                cast(float, stressor["duration_seconds"]),
            ),
            None,
        )
    if kind == "rate":
        return vary_rate(samples, cast(float, stressor["factor"])), None
    raise ValueError(f"Stress runner does not implement {kind}")


def _selected_cases(
    cases: Sequence[Mapping[str, Any]], case_ids: Sequence[str] | None
) -> tuple[Mapping[str, Any], ...]:
    by_id = {cast(str, case["id"]): case for case in cases}
    selected_ids = tuple(case_ids) if case_ids is not None else tuple(by_id)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Stress runner requires unique selected case IDs")
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"Stress runner received unknown case IDs: {missing}")
    return tuple(by_id[sample_id] for sample_id in selected_ids)


def prepare_stress_audio(
    references: Path,
    clean_audio: Path,
    destination: Path,
    *,
    case_ids: Sequence[str] | None = None,
    seed: int = 42,
) -> Mapping[str, Any]:
    """Prepare paired audio for implemented cases and promote it atomically."""
    cases = _selected_cases(load_stress_cases(references), case_ids)
    unsupported: set[str] = set()
    for case in cases:
        stressors = cast(Sequence[Mapping[str, Any]], case["stressors"])
        if len(stressors) != 2 or stressors[0] != {"kind": "clean"}:
            raise ValueError("Stress runner requires one clean and one stressed lane")
        kind = cast(str, stressors[1]["kind"])
        if kind not in IMPLEMENTED_STRESSORS:
            unsupported.add(kind)
    if unsupported:
        raise ValueError(
            "Stress runner cannot score unsupported stressors: "
            f"{sorted(unsupported)}"
        )

    if destination.exists():
        raise ValueError("Stress destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-prepare-", dir=destination.parent)
    )
    promoted = False
    records: list[Mapping[str, Any]] = []
    try:
        clean_output = staging / "clean"
        stressed_output = staging / "stressed"
        clean_output.mkdir()
        stressed_output.mkdir()
        for index, case in enumerate(cases):
            sample_id = cast(str, case["id"])
            source = clean_audio / f"{sample_id}.wav"
            if not source.is_file():
                raise ValueError(f"Stress runner is missing clean audio: {sample_id}")
            try:
                samples, sample_rate = sf.read(
                    source, dtype="float32", always_2d=True
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Stress runner cannot read clean audio: {sample_id}"
                ) from exc
            if not len(samples) or sample_rate <= 0 or not np.isfinite(samples).all():
                raise ValueError(f"Stress runner received invalid audio: {sample_id}")

            clean_path = clean_output / source.name
            shutil.copyfile(source, clean_path)
            stressor = cast(Mapping[str, Any], case["stressors"][1])
            stressed, interval = _apply_stressor(
                samples, sample_rate, stressor, seed=seed + index
            )
            stressed_path = stressed_output / source.name
            sf.write(stressed_path, stressed, sample_rate, subtype="PCM_16")
            record: dict[str, Any] = {
                "id": sample_id,
                "stressor": dict(stressor),
                "clean_sha256": _sha256(clean_path),
                "stressed_sha256": _sha256(stressed_path),
            }
            if interval is not None:
                record["noise_only_interval"] = dict(interval)
            records.append(record)

        manifest: Mapping[str, Any] = {
            "schema_version": 1,
            "lane": "accessibility-stress-v1",
            "references_sha256": _sha256(references),
            "seed": seed,
            "sample_count": len(records),
            "samples": records,
        }
        atomic_write_json(staging / "preparation-manifest.json", manifest)
        os.replace(staging, destination)
        promoted = True
        return manifest
    finally:
        if not promoted and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

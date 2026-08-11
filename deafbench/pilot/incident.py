"""Fail-closed processing stop for pilot control failures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar


FAILURE_CATEGORIES = frozenset(
    {"authorization", "integrity", "access", "retention", "deletion"}
)
T = TypeVar("T")


class ProcessingBlocked(RuntimeError):
    pass


class IncidentStop:
    """Persist processing state and require explicit approval to restore it."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def _state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {"blocked": False, "incidents": [], "restorations": []}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def run_gate(self, category: str, operation: Callable[[], T]) -> T:
        state = self._state()
        if state["blocked"]:
            raise ProcessingBlocked("pilot processing is stopped pending approval")
        if category not in FAILURE_CATEGORIES:
            raise ValueError("unsupported incident category")
        try:
            return operation()
        except Exception as error:
            state["blocked"] = True
            incidents = list(state["incidents"])
            incidents.append(
                {
                    "category": category,
                    "reason_type": type(error).__name__,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            state["incidents"] = incidents
            self._write(state)
            raise ProcessingBlocked(f"pilot processing stopped: {category}") from error

    def restore(self, *, approval_reference: str, operator: str) -> None:
        state = self._state()
        if not state["blocked"]:
            raise ValueError("processing is not blocked")
        if not approval_reference.strip() or not operator.strip():
            raise ValueError("restoration requires explicit approval and operator")
        restorations = list(state["restorations"])
        restorations.append(
            {
                "approval_reference": approval_reference,
                "operator": operator,
                "restored_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        state["restorations"] = restorations
        state["blocked"] = False
        self._write(state)

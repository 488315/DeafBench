"""Finite retention policy for founding-pilot case artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class RetentionSchedule:
    delivered_at: datetime
    review_deadline: datetime
    deletion_deadline: datetime
    extension_approval_reference: str | None = None


def schedule_after_delivery(delivered_at: datetime) -> RetentionSchedule:
    """Set the seven-day review and default fourteen-day deletion deadlines."""

    if delivered_at.tzinfo is None:
        raise ValueError("delivery time must include a timezone")
    delivered = delivered_at.astimezone(timezone.utc)
    return RetentionSchedule(
        delivered_at=delivered,
        review_deadline=delivered + timedelta(days=7),
        deletion_deadline=delivered + timedelta(days=14),
    )


def request_earlier_deletion(
    schedule: RetentionSchedule, requested_at: datetime
) -> RetentionSchedule:
    if requested_at.tzinfo is None:
        raise ValueError("deletion request must include a timezone")
    requested = requested_at.astimezone(timezone.utc)
    if requested < schedule.delivered_at:
        raise ValueError("deletion request predates delivery")
    return replace(schedule, deletion_deadline=min(requested, schedule.deletion_deadline))


def extend_retention(
    schedule: RetentionSchedule,
    *,
    deletion_deadline: datetime,
    written_approval_reference: str,
) -> RetentionSchedule:
    """Extend only to a defined date backed by a written customer approval."""

    if not written_approval_reference.strip():
        raise ValueError("retention extension requires written approval")
    if deletion_deadline.tzinfo is None:
        raise ValueError("extension deadline must include a timezone")
    deadline = deletion_deadline.astimezone(timezone.utc)
    if deadline <= schedule.deletion_deadline:
        raise ValueError("extension deadline must be later than the current deadline")
    return replace(
        schedule,
        deletion_deadline=deadline,
        extension_approval_reference=written_approval_reference,
    )

from datetime import datetime, timedelta, timezone

import pytest

from deafbench.pilot.retention import (
    extend_retention,
    request_earlier_deletion,
    schedule_after_delivery,
)


DELIVERY = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)


def test_delivery_sets_finite_review_and_deletion_deadlines() -> None:
    schedule = schedule_after_delivery(DELIVERY)

    assert schedule.review_deadline == DELIVERY + timedelta(days=7)
    assert schedule.deletion_deadline == DELIVERY + timedelta(days=14)


def test_customer_can_request_earlier_deletion() -> None:
    schedule = request_earlier_deletion(
        schedule_after_delivery(DELIVERY), DELIVERY + timedelta(days=2)
    )

    assert schedule.deletion_deadline == DELIVERY + timedelta(days=2)


def test_extension_requires_written_approval_and_deadline() -> None:
    schedule = schedule_after_delivery(DELIVERY)
    with pytest.raises(ValueError, match="written approval"):
        extend_retention(
            schedule,
            deletion_deadline=DELIVERY + timedelta(days=30),
            written_approval_reference=" ",
        )
    with pytest.raises(ValueError, match="timezone"):
        extend_retention(
            schedule,
            deletion_deadline=datetime(2026, 9, 1),
            written_approval_reference="approval-1",
        )


def test_approved_extension_records_reference_and_date() -> None:
    deadline = DELIVERY + timedelta(days=30)
    schedule = extend_retention(
        schedule_after_delivery(DELIVERY),
        deletion_deadline=deadline,
        written_approval_reference="approval-1",
    )

    assert schedule.deletion_deadline == deadline
    assert schedule.extension_approval_reference == "approval-1"

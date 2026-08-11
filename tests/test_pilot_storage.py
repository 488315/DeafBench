from pathlib import Path

import pytest

from deafbench.pilot.storage import ProtectionState, protect_case_storage


def test_storage_requires_measured_active_volume_protection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not verified"):
        protect_case_storage(
            tmp_path,
            protection_probe=lambda _: ProtectionState(False, "access denied"),
            acl_restrictor=lambda _: True,
        )


def test_storage_requires_verified_account_only_acl(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ACL"):
        protect_case_storage(
            tmp_path,
            protection_probe=lambda _: ProtectionState(True, "BitLocker Protection On"),
            acl_restrictor=lambda _: False,
        )


def test_storage_records_only_measured_protections(tmp_path: Path) -> None:
    result = protect_case_storage(
        tmp_path,
        protection_probe=lambda _: ProtectionState(True, "BitLocker Protection On"),
        acl_restrictor=lambda _: True,
    )

    assert result.volume_protection == "BitLocker Protection On"
    assert result.account_acl_restricted is True

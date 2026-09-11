# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

import pytest

from moe_infinity import _store


@pytest.mark.parametrize(
    ("phase_enabled", "adaptive_enabled", "manager_enabled"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_manager_enablement_composes_feature_flags(
    phase_enabled, adaptive_enabled, manager_enabled
):
    assert hasattr(_store.expert_dispatcher, "configure_residency_manager")
    assert hasattr(_store.prefetch_handle, "configure_residency_manager")
    assert manager_enabled is (phase_enabled or adaptive_enabled)


def test_phase_and_adaptive_clients_report_one_manager():
    assert hasattr(_store.expert_dispatcher, "get_residency_manager_id")
    assert hasattr(_store.prefetch_handle, "get_residency_manager_id")

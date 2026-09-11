from moe_infinity.memory.expert_policy import ExpertPhase


class RecordingManager:
    def __init__(self):
        self.calls = []
        self._resident = {}

    def begin_admission(self, key, gpu_id, phase, source):
        self.calls.append((key, gpu_id, int(phase), source))
        self._resident[key] = (gpu_id, int(phase), source)
        return True

    def snapshot(self):
        return {
            "resident_experts": len(self._resident),
            "demand_admissions": sum(
                1 for v in self._resident.values() if v[2] == "demand"
            ),
            "prefetch_completed": sum(
                1 for v in self._resident.values() if v[2] == "prefetch"
            ),
        }


class FakeClient:
    def __init__(self, manager, source):
        self._manager = manager
        self._source = source

    def admit(self, key, gpu_id, phase):
        return self._manager.begin_admission(key, gpu_id, phase, self._source)

    @property
    def manager(self):
        return self._manager


def test_demand_and_prefetch_clients_share_one_manager_snapshot():
    manager = RecordingManager()
    dispatcher_client = FakeClient(manager, "demand")
    prefetch_client = FakeClient(manager, "prefetch")

    assert dispatcher_client.manager is prefetch_client.manager

    dispatcher_client.admit((0, 1), 0, ExpertPhase.DECODE)
    prefetch_client.admit((0, 2), 0, ExpertPhase.DECODE)

    assert manager.calls == [
        ((0, 1), 0, int(ExpertPhase.DECODE), "demand"),
        ((0, 2), 0, int(ExpertPhase.DECODE), "prefetch"),
    ]

    snapshot = manager.snapshot()
    assert snapshot["resident_experts"] == 2
    assert snapshot["demand_admissions"] == 1
    assert snapshot["prefetch_completed"] == 1


def test_duplicate_key_does_not_double_count_snapshot():
    manager = RecordingManager()
    client = FakeClient(manager, "demand")
    client.admit((0, 1), 0, ExpertPhase.DECODE)
    client.admit((0, 1), 0, ExpertPhase.DECODE)
    assert manager.snapshot()["resident_experts"] == 1

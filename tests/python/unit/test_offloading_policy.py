import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_lru_basic_get_put():
    from moe_infinity.memory.offloading_policy import LRUPolicy

    lru: LRUPolicy[str, int] = LRUPolicy(capacity=3)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)

    assert lru.get("a") == 1
    assert lru.get("b") == 2
    assert lru.get("c") == 3


def test_lru_eviction_order():
    from moe_infinity.memory.offloading_policy import LRUPolicy

    lru: LRUPolicy[str, int] = LRUPolicy(capacity=3)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    lru.put("d", 4)

    assert lru.get("a") is None
    assert lru.get("b") == 2
    assert lru.get("c") == 3
    assert lru.get("d") == 4


def test_lru_get_refreshes_order():
    from moe_infinity.memory.offloading_policy import LRUPolicy

    lru: LRUPolicy[str, int] = LRUPolicy(capacity=3)
    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    assert lru.get("a") == 1
    lru.put("d", 4)

    assert lru.get("b") is None
    assert lru.get("a") == 1


def test_lru_capacity_property():
    from moe_infinity.memory.offloading_policy import LRUPolicy

    lru: LRUPolicy[str, int] = LRUPolicy(capacity=7)
    assert lru.capacity == 7


def test_arc_basic_put_get():
    from moe_infinity.memory.offloading_policy import ARCPolicy

    arc: ARCPolicy[str, int] = ARCPolicy(capacity=4)
    arc.put("a", 1)
    arc.put("b", 2)
    arc.put("c", 3)

    assert arc.get("a") == 1
    assert arc.get("b") == 2
    assert arc.get("c") == 3


def test_arc_eviction_when_full():
    from moe_infinity.memory.offloading_policy import ARCPolicy

    arc: ARCPolicy[str, int] = ARCPolicy(capacity=2)
    arc.put("a", 1)
    arc.put("b", 2)
    arc.put("c", 3)

    assert len(arc) == 2
    assert arc.get("a") is None
    assert arc.get("b") == 2
    assert arc.get("c") == 3


def test_arc_adaptation():
    from moe_infinity.memory.offloading_policy import ARCPolicy

    arc: ARCPolicy[str, int] = ARCPolicy(capacity=2)
    arc.put("a", 1)
    arc.put("b", 2)
    assert arc.get("a") == 1
    arc.put("c", 3)
    arc.put("b", 20)

    assert arc.p > 0
    assert arc.t2_size >= 1


def test_arc_len():
    from moe_infinity.memory.offloading_policy import ARCPolicy

    arc: ARCPolicy[int, int] = ARCPolicy(capacity=3)
    arc.put(1, 10)
    arc.put(2, 20)
    arc.put(3, 30)
    assert len(arc) == 3

    arc.put(4, 40)
    assert len(arc) == 3

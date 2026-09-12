from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, cast

from typing_extensions import override


class CPAwareKVManager(ABC):
    @abstractmethod
    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float:
        """Returns reuse score 0.0-1.0 based on CP index prediction"""

    @abstractmethod
    def get_cp_cached_blocks(self, request_id: str) -> list[int]:
        """Returns list of block hashes from CP index for this request"""

    @abstractmethod
    def notify_blocks_allocated(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        """Inform CP index that blocks were allocated for request"""

    @abstractmethod
    def notify_blocks_freed(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        """Inform CP index that blocks were freed"""

    @abstractmethod
    def get_allocation_priority(self, request_ids: list[str]) -> list[str]:
        """Returns request_ids sorted by CP overlap (highest first)"""

    def remove_request(self, request_id: str) -> None:
        """Drop all per-request tracking state on terminal removal"""
        _ = request_id


class NullCPAwareKVManager(CPAwareKVManager):
    @override
    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float:
        _ = request_id
        _ = token_ids
        return 0.0

    @override
    def get_cp_cached_blocks(self, request_id: str) -> list[int]:
        _ = request_id
        return []

    @override
    def notify_blocks_allocated(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        _ = request_id
        _ = block_hashes

    @override
    def notify_blocks_freed(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        _ = request_id
        _ = block_hashes

    @override
    def get_allocation_priority(self, request_ids: list[str]) -> list[str]:
        return list(request_ids)


class ContextPilotKVManager(CPAwareKVManager):
    _middleware: object

    def __init__(self, middleware: object):
        self._middleware = middleware
        self._request_to_blocks: dict[str, list[int]] = {}

    @override
    def predict_prefix_reuse(
        self, request_id: str, token_ids: list[int]
    ) -> float:
        predictor = self._get_middleware_callable("predict_prefix_reuse")
        if not callable(predictor):
            return 0.0
        try:
            score_obj = predictor(request_id, list(token_ids))
        except Exception:
            return 0.0
        if not isinstance(score_obj, (int, float)):
            return 0.0
        return self._clamp01(float(score_obj))

    @override
    def get_cp_cached_blocks(self, request_id: str) -> list[int]:
        getter = self._get_middleware_callable("get_cp_cached_blocks")
        if callable(getter):
            try:
                result = getter(request_id)
                if isinstance(result, list):
                    result_list = cast(list[object], result)
                    return self._normalize_int_list(result_list)
            except Exception:
                pass
        return list(self._request_to_blocks.get(request_id, []))

    @override
    def notify_blocks_allocated(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        normalized = [int(v) for v in block_hashes]
        self._request_to_blocks[request_id] = list(normalized)

        notifier = self._get_middleware_callable("notify_blocks_allocated")
        if callable(notifier):
            try:
                _ = notifier(request_id, list(normalized))
            except Exception:
                return

    @override
    def notify_blocks_freed(
        self, request_id: str, block_hashes: list[int]
    ) -> None:
        freed = {int(v) for v in block_hashes}
        current = self._request_to_blocks.get(request_id, [])
        remaining = [
            block_hash for block_hash in current if block_hash not in freed
        ]
        if remaining:
            self._request_to_blocks[request_id] = remaining
        elif request_id in self._request_to_blocks:
            del self._request_to_blocks[request_id]

        notifier = self._get_middleware_callable("notify_blocks_freed")
        if callable(notifier):
            try:
                _ = notifier(request_id, list(freed))
            except Exception:
                return

    @override
    def get_allocation_priority(self, request_ids: list[str]) -> list[str]:
        ranker = self._get_middleware_callable("get_allocation_priority")
        if callable(ranker):
            try:
                ranked = ranker(list(request_ids))
                if isinstance(ranked, list):
                    input_set = set(request_ids)
                    out: list[str] = []
                    seen: set[str] = set()
                    ranked_obj = cast(list[object], ranked)
                    for rid_obj in ranked_obj:
                        rid_str = str(rid_obj)
                        if rid_str in input_set and rid_str not in seen:
                            out.append(rid_str)
                            seen.add(rid_str)
                    for rid in request_ids:
                        if rid not in seen:
                            out.append(rid)
                    return out
            except Exception:
                pass

        return sorted(
            request_ids,
            key=lambda rid: self.predict_prefix_reuse(rid, []),
            reverse=True,
        )

    @override
    def remove_request(self, request_id: str) -> None:
        self._request_to_blocks.pop(request_id, None)

    def _get_middleware_callable(self, name: str) -> Optional[object]:
        return cast(Optional[object], getattr(self._middleware, name, None))

    @staticmethod
    def _normalize_int_list(values: list[object]) -> list[int]:
        normalized: list[int] = []
        for value in values:
            if isinstance(value, bool):
                normalized.append(int(value))
                continue
            if isinstance(value, int):
                normalized.append(value)
                continue
            if isinstance(value, float):
                normalized.append(int(value))
                continue
            if isinstance(value, str):
                try:
                    normalized.append(int(value))
                except ValueError:
                    continue
        return normalized

    @staticmethod
    def _clamp01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value


__all__ = [
    "CPAwareKVManager",
    "NullCPAwareKVManager",
    "ContextPilotKVManager",
]

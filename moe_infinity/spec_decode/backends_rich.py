"""Physical DFlash cohorts for explicitly row-aware rich target wrappers."""

from __future__ import annotations

from typing import Any

from moe_infinity.spec_decode.backends_bare_hf import BatchedBareHFBackend
from moe_infinity.spec_decode.protocols import BackendCapabilities


class BatchedRichBackend(BatchedBareHFBackend):
    """Reuse dense lockstep mechanics while injecting rich target forwards."""

    name = "dflash-batched-rich"
    allows_rich_forward = True

    def __init__(self, speculator: Any) -> None:
        super().__init__(speculator)
        moe = speculator.moe
        explicit_declaration = getattr(moe, "__dict__", {}).get(
            "_supports_native_rich_batch"
        )
        if callable(explicit_declaration):
            try:
                declared = bool(explicit_declaration())
            except Exception:
                declared = False
        else:
            declared = bool(getattr(moe, "_native_rich_batch_capable", False))
        self.wrapper_supported = declared
        base = self.capabilities
        self.capabilities = BackendCapabilities(
            supports_batch=self.wrapper_supported,
            supports_sampling=base.supports_sampling,
            supports_ragged_rows=base.supports_ragged_rows,
            cache_kind=base.cache_kind,
            supports_route_ahead=(
                self.wrapper_supported
                and bool(
                    getattr(
                        speculator.executor_evidence, "wiring_reachable", False
                    )
                )
            ),
            supports_rich_forward=self.wrapper_supported,
            pairing_evidence=base.pairing_evidence,
            executor_evidence=speculator.executor_evidence,
        )

    def supports(self, request: Any) -> bool:
        return self.wrapper_supported and super().supports(request)


__all__ = ["BatchedRichBackend"]

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TransferType(Enum):
    EXPERT_FETCH = "EXPERT_FETCH"
    EXPERT_EVICT = "EXPERT_EVICT"
    KV_SWAP_IN = "KV_SWAP_IN"
    KV_SWAP_OUT = "KV_SWAP_OUT"


class TransferPriority(Enum):
    URGENT = 0
    HIGH = 5
    NORMAL = 10
    LOW = 15
    BACKGROUND = 19


@dataclass
class TransferRequest:
    transfer_id: str
    transfer_type: TransferType
    priority: TransferPriority
    source_device: str
    target_device: str
    tensor_id: Optional[str] = None
    block_ids: list[int] = field(default_factory=list)


@dataclass
class TransferResult:
    transfer_id: str
    status: str
    duration_ms: float
    bytes_transferred: int = 0
    error: Optional[str] = None

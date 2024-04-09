from dataclasses import dataclass
import numpy as np
import hashlib

@dataclass
class ExpertTraceEntry:
    seq_id: str = None
    matrix: np.ndarray = None
    access: int = 0
    num_new_tokens: int = 0

    def __hash__(self):
        return hash(self.seq_id)
    

@dataclass
class ExpertCacheEntry:
    expert_idx: int = None
    layer_idx: int = None
    r: float = 0.0
    visit: int = 0
    timestamp: int = 0

    def __hash__(self):
        return hash((self.layer_idx, self.expert_idx))
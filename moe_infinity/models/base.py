import torch

class MoELayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
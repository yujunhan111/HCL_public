import torch
import torch.nn as nn


class StructureEncoder(nn.Module):
    """
    Generic feedforward encoder shared across all fusion methods.
    Maps input of shape [B, input_dim] to [B, output_dim].

    Used as the per-modality, per-level encoder inside any fusion module
    that needs a learnable FFN projection.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
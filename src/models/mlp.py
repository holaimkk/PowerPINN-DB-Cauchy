from __future__ import annotations

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Small fully-connected network used by the toy PINN baseline."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(build_activation(activation))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

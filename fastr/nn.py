# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Network layers used by FaStR."""

from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """A feed-forward network with no activation after its last layer."""

    def __init__(self, input_dim, output_dim, hidden_dims=(), norm_layer=None, activation=nn.ReLU):
        super().__init__()
        layers = []
        for width in hidden_dims:
            layers.append(nn.Linear(input_dim, width))
            if norm_layer is not None:
                layers.append(norm_layer(width))
            if activation is not None:
                layers.append(activation())
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Actor(nn.Module):
    """Map a state to an action in [-1, 1]."""

    def __init__(self, input_dim, output_dim, hidden_dims, norm_layer=nn.LayerNorm):
        super().__init__()
        self.output_layer = MLP(input_dim, output_dim, hidden_dims, norm_layer)

    def forward(self, obs):
        return torch.tanh(self.output_layer(obs))


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = activation
        if input_dim != hidden_dim:
            self.residual = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual = nn.Identity()
        if dropout and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dropout(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x + self.residual(residual)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_dims: Sequence[int] = [],
        activation: Callable = F.relu,
        dropout: float = None,
        device: str = "cpu",
    ):
        super().__init__()

        assert len(hidden_dims) > 0, "hidden_dims must be a list of at least one integer"
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        hidden_dims = [hidden_dims[0]] + list(hidden_dims)
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout,
                    activation,
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class PositionalFeature(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.dim = dim

    def forward(self, x):
        freqs = torch.arange(0, self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.to(freqs.dtype) @ freqs.unsqueeze(0)
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        return x


class RffLayer(nn.Module):
    """Map x to Fourier features: [sin(Wx + b), cos(Wx + b)]."""

    def __init__(self, input_dim: int, rff_dim: int) -> None:
        super().__init__()
        assert rff_dim % 2 == 0, "rff_dim must be even"
        self.layer = nn.Linear(input_dim, rff_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class RffReward(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list[int],
        linear: bool = False,
        rff_dim: int | None = None,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(feature_dim)
        if linear:
            self.rff_layer = nn.Identity()
            self.net = MLP(
                input_dim=feature_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                norm_layer=nn.LayerNorm,
                activation=nn.ELU,
            )
        else:
            self.rff_layer = RffLayer(feature_dim, rff_dim)
            self.net = MLP(
                input_dim=rff_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                norm_layer=nn.LayerNorm,
                activation=nn.ELU,
            )

    def forward(self, x):
        return self.net(self.rff_layer(self.ln(x)))

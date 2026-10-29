# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Value networks that use separate state and action features."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import MLP, RffLayer

#  Building blocks


class BilinearQ(nn.Module):
    """Compute Q(z_s, z_a) = z_s^T W z_a + b, with a learned matrix W."""

    def __init__(self, d_s: int, d_a: int, use_bias: bool = True) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_a = d_a
        self.bilinear = nn.Bilinear(d_s, d_a, 1, bias=use_bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.bilinear.weight, gain=0.01)
        if self.bilinear.bias is not None:
            nn.init.zeros_(self.bilinear.bias)

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (B, 1)."""
        return self.bilinear(z_s, z_a)

    @property
    def W(self) -> torch.Tensor:
        """Return the bilinear weight matrix, with shape (d_s, d_a)."""
        return self.bilinear.weight.squeeze(0)


# Hybrid critic: a bilinear value plus a neural network value.


class HybridQ(nn.Module):

    def __init__(
        self,
        d_s: int,
        d_a: int,
        rff_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 256),
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.bilinear = BilinearQ(d_s, d_a, use_bias=use_bias)

        self.ln = nn.LayerNorm(d_s + d_a)
        self.rff = RffLayer(d_s + d_a, rff_dim)
        self.mlp = MLP(
            input_dim=rff_dim,
            output_dim=1,
            hidden_dims=list(hidden_dims),
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        )

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (B, 1)."""
        z_s_n = F.normalize(z_s, dim=-1)
        z_a_n = F.normalize(z_a, dim=-1)

        # Compute the bilinear value.
        q_bilinear = self.bilinear(z_s_n, z_a_n)

        # Add the neural network value.
        q_residual = self.mlp(self.rff(self.ln(torch.cat([z_s_n, z_a_n], dim=-1))))

        return q_bilinear + q_residual

    def forward_thompson(
        self,
        z_s: torch.Tensor,
        z_a: torch.Tensor,
        noise_scale: float,
        inv_std_s: torch.Tensor,
        inv_std_a: torch.Tensor,
    ) -> torch.Tensor:
        """Add random noise to W and keep the other value term fixed.

        Use more noise for features with smaller running squared values.
        noise_scale sets the noise size.
        inv_std_s and inv_std_a scale the noise for each state and action feature.
        """
        z_s_n = F.normalize(z_s, dim=-1)
        z_a_n = F.normalize(z_a, dim=-1)

        # Add more noise for features with smaller running squared values.
        W = self.bilinear.W  # (d_s, d_a)
        noise = torch.randn_like(W)
        # Scale noise using both state and action statistics.
        scaled_noise = noise * inv_std_s.unsqueeze(1) * inv_std_a.unsqueeze(0)
        W_tilde = W + noise_scale * scaled_noise

        # Compute the value with the changed weights.
        q_bilinear = (z_s_n @ W_tilde * z_a_n).sum(-1, keepdim=True)
        bias = self.bilinear.bilinear.bias
        if bias is not None:
            q_bilinear = q_bilinear + bias

        # Keep the neural network term fixed.
        q_residual = self.mlp(self.rff(self.ln(torch.cat([z_s_n, z_a_n], dim=-1))))

        return q_bilinear + q_residual

    @property
    def W(self) -> torch.Tensor:
        """Return the bilinear weight matrix, with shape (d_s, d_a)."""
        return self.bilinear.W


# Use two value networks for the critic.


class BilinearDoubleQ(nn.Module):
    """Use two separate bilinear value networks."""

    def __init__(self, d_s: int, d_a: int, use_bias: bool = True) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_a = d_a
        self.q1 = BilinearQ(d_s, d_a, use_bias=use_bias)
        self.q2 = BilinearQ(d_s, d_a, use_bias=use_bias)

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (2, B, 1)."""
        return torch.stack([self.q1(z_s, z_a), self.q2(z_s, z_a)], dim=0)


class HybridDoubleQ(nn.Module):

    def __init__(
        self,
        d_s: int,
        d_a: int,
        rff_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 256),
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_a = d_a
        self.q1 = HybridQ(d_s, d_a, rff_dim, hidden_dims, use_bias)
        self.q2 = HybridQ(d_s, d_a, rff_dim, hidden_dims, use_bias)

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (2, B, 1)."""
        return torch.stack([self.q1(z_s, z_a), self.q2(z_s, z_a)], dim=0)

    def forward_thompson(
        self,
        z_s: torch.Tensor,
        z_a: torch.Tensor,
        noise_scale: float,
        inv_std_s: torch.Tensor,
        inv_std_a: torch.Tensor,
    ) -> torch.Tensor:
        """Use a separate random change to W for each value network."""
        return torch.stack(
            [
                self.q1.forward_thompson(z_s, z_a, noise_scale, inv_std_s, inv_std_a),
                self.q2.forward_thompson(z_s, z_a, noise_scale, inv_std_s, inv_std_a),
            ],
            dim=0,
        )


# Multiply state and action Fourier features before predicting a value.


class ProductKernelQ(nn.Module):
    """Multiply state and action Fourier features, then predict a value."""

    def __init__(
        self,
        d_s: int,
        d_a: int,
        rff_dim: int = 1024,
        hidden_dims: Sequence[int] = (512,),
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.ln_s = nn.LayerNorm(d_s)
        self.ln_a = nn.LayerNorm(d_a)
        self.rff_s = RffLayer(d_s, rff_dim)
        self.rff_a = RffLayer(d_a, rff_dim)
        self.ln_joint = nn.LayerNorm(rff_dim)
        self.mlp = MLP(
            input_dim=rff_dim,
            output_dim=1,
            hidden_dims=list(hidden_dims),
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        )

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        h_s = self.rff_s(self.ln_s(z_s))
        h_a = self.rff_a(self.ln_a(z_a))
        return self.mlp(self.ln_joint(h_s * h_a))


class ProductKernelDoubleQ(nn.Module):
    """Use two separate product-kernel value networks."""

    def __init__(
        self,
        d_s: int,
        d_a: int,
        rff_dim: int = 1024,
        hidden_dims: Sequence[int] = (512,),
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.q1 = ProductKernelQ(d_s, d_a, rff_dim, hidden_dims, use_bias)
        self.q2 = ProductKernelQ(d_s, d_a, rff_dim, hidden_dims, use_bias)

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.q1(z_s, z_a), self.q2(z_s, z_a)], dim=0)


# Add a linear value term and a network term with fixed encoder inputs.


class SpectralResidualQ(nn.Module):

    def __init__(
        self,
        d: int,
        rff_dim: int = 1024,
        hidden_dims: Sequence[int] = (512,),
    ) -> None:
        super().__init__()
        self.d = d

        # This linear term also updates the encoder.
        self.linear_head = nn.Linear(d, 1, bias=True)
        nn.init.orthogonal_(self.linear_head.weight, gain=0.01)
        nn.init.zeros_(self.linear_head.bias)

        # This neural network term does not update the encoder.
        self.ln = nn.LayerNorm(d)
        self.rff = RffLayer(d, rff_dim)
        self.mlp = MLP(
            input_dim=rff_dim,
            output_dim=1,
            hidden_dims=list(hidden_dims),
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        )

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (B, 1)."""
        psi = z_s * z_a  # Element-wise product, with shape (B, d).

        # Pass gradients from the linear term to both encoders.
        q_struct = self.linear_head(psi)

        # Stop encoder gradients for the neural network term.
        q_resid = self.mlp(self.rff(self.ln(psi.detach())))

        return q_struct + q_resid

    def forward_struct_only(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return only the linear value term."""
        psi = z_s * z_a
        return self.linear_head(psi)

    @property
    def theta(self) -> torch.Tensor:
        """Return the weights of the linear value term."""
        return self.linear_head.weight.squeeze(0)


class SpectralResidualDoubleQ(nn.Module):

    def __init__(
        self,
        d: int,
        rff_dim: int = 1024,
        hidden_dims: Sequence[int] = (512,),
    ) -> None:
        super().__init__()
        self.d = d
        self.q1 = SpectralResidualQ(d, rff_dim, hidden_dims)
        self.q2 = SpectralResidualQ(d, rff_dim, hidden_dims)

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (2, B, 1)."""
        return torch.stack([self.q1(z_s, z_a), self.q2(z_s, z_a)], dim=0)

    def forward_struct_only(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return the two linear value terms, with shape (2, B, 1)."""
        return torch.stack(
            [
                self.q1.forward_struct_only(z_s, z_a),
                self.q2.forward_struct_only(z_s, z_a),
            ],
            dim=0,
        )


# Apply two value networks to the product of state and action features.


class RffPsiDoubleQ(nn.Module):

    def __init__(
        self,
        d: int,
        rff_dim: int = 1024,
        hidden_dims: Sequence[int] = (512,),
    ) -> None:
        super().__init__()
        self.d = d

        self.ln1 = nn.LayerNorm(d)
        self.rff1 = RffLayer(d, rff_dim)
        self.mlp1 = MLP(
            input_dim=rff_dim,
            output_dim=1,
            hidden_dims=list(hidden_dims),
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        )

        self.ln2 = nn.LayerNorm(d)
        self.rff2 = RffLayer(d, rff_dim)
        self.mlp2 = MLP(
            input_dim=rff_dim,
            output_dim=1,
            hidden_dims=list(hidden_dims),
            norm_layer=nn.LayerNorm,
            activation=nn.ELU,
        )

    def forward(self, z_s: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """Return values with shape (2, B, 1)."""
        psi = z_s * z_a  # Element-wise product.
        q1 = self.mlp1(self.rff1(self.ln1(psi)))
        q2 = self.mlp2(self.rff2(self.ln2(psi)))
        return torch.stack([q1, q2], dim=0)

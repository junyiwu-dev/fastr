# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""State and action encoders and the FaStR model loss."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import PositionalFeature, ResidualMLP, RffReward
from .utils import noise_schedule


class FaStRNCE(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_s: int,
        d_a: int,
        phi_s_hidden_dims: list[int],
        phi_a_hidden_dims: list[int],
        mu_hidden_dims: list[int],
        reward_hidden_dims: list[int],
        rff_dim: int | None = None,
        num_noises: int = 0,
        linear: bool = False,
        ranking: bool = True,
        diagonal_m: bool = False,
        d_s_full: int | None = None,
        d_a_full: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_a = d_a
        self.d_s_full = d_s_full if d_s_full is not None else d_s
        self.d_a_full = d_a_full if d_a_full is not None else d_a
        self.diagonal_m = diagonal_m

        if diagonal_m:
            assert d_s == d_a, f"diagonal_m requires d_s == d_a, got {d_s} vs {d_a}"
            self.joint_dim = d_s  # Diagonal transition model
        else:
            self.joint_dim = d_s * d_a  # NCE effective dim = d_s * d_a
        self.linear = linear
        self.ranking = ranking
        self.device = device

        if num_noises > 0:
            self.use_noise_perturbation = True
            self.N = num_noises
            self.register_buffer(
                "alphabars", noise_schedule(num_noises)[..., None].to(device), persistent=False
            )
        else:
            self.use_noise_perturbation = False
            self.N = 1

        self.mlp_t = nn.Sequential(
            PositionalFeature(128),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
        )

        self.mlp_phi_s = ResidualMLP(
            input_dim=obs_dim,
            output_dim=self.d_s_full,
            hidden_dims=phi_s_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )

        self.mlp_phi_a = ResidualMLP(
            input_dim=action_dim,
            output_dim=self.d_a_full,
            hidden_dims=phi_a_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )

        if self.d_s_full != d_s:
            self.proj_s = nn.Linear(self.d_s_full, d_s)
        else:
            self.proj_s = nn.Identity()
        if self.d_a_full != d_a:
            self.proj_a = nn.Linear(self.d_a_full, d_a)
        else:
            self.proj_a = nn.Identity()

        # The full model returns one value for each state/action feature pair.
        # The diagonal model returns one value for each matching feature pair.
        self.mlp_M = ResidualMLP(
            input_dim=obs_dim + (128 if self.use_noise_perturbation else 0),
            output_dim=self.joint_dim,
            hidden_dims=mu_hidden_dims,
            activation=nn.Mish(),
            dropout=None,
            device=device,
        )

        self.reward = RffReward(
            feature_dim=self.d_s_full + self.d_a_full,
            hidden_dims=reward_hidden_dims,
            linear=linear,
            rff_dim=rff_dim,
        )

        if not self.ranking:
            self.normalizer = nn.Parameter(
                torch.zeros([self.N], dtype=torch.float32, device=self.device)
            )

    #  Forward helpers

    def forward_phi_s(self, s: torch.Tensor) -> torch.Tensor:
        """Map states to features with shape (*, d_s_full)."""
        return self.mlp_phi_s(s)

    def forward_phi_a(self, a: torch.Tensor) -> torch.Tensor:
        """Map actions to features with shape (*, d_a_full)."""
        return self.mlp_phi_a(a)

    def forward_M(self, sp: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Return the transition model output.

        The full model returns shape (*, d_s, d_a).
        The diagonal model returns shape (*, d).
        """
        if t is not None:
            t_ff = self.mlp_t(t)
            sp = torch.concat([sp, t_ff], dim=-1)
        M = self.mlp_M(sp)
        M = torch.tanh(M)
        if self.diagonal_m:
            return M  # (*, d)
        else:
            return M.unflatten(-1, (self.d_s, self.d_a))  # (*, d_s, d_a)

    #  Logit computation  (bilinear NCE)

    def compute_logits(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sp: torch.Tensor,
        z_s: torch.Tensor | None = None,
        z_a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score each state-action pair against each next state.

        Return shape (N, B, B), where N is the number of noise levels
        and B is the batch size. The score is phi_s(s)^T M(next_s) phi_a(a).
        """
        B = sp.shape[0]

        if z_s is None:
            z_s = self.proj_s(self.forward_phi_s(s))  # (B, d_s)
        if z_a is None:
            z_a = self.proj_a(self.forward_phi_a(a))  # (B, d_a)

        if self.diagonal_m:
            z_phi = z_s * z_a  # (B, d)
        else:
            # Form all pairs of state and action features, then flatten them.
            z_phi = torch.bmm(
                z_s.unsqueeze(2),  # (B, d_s, 1)
                z_a.unsqueeze(1),  # (B, 1, d_a)
            )  # (B, d_s, d_a)
            z_phi = z_phi.reshape(B, -1)  # (B, d_s * d_a)

        if self.use_noise_perturbation:
            sp = sp.unsqueeze(0).repeat(self.N, 1, 1)  # (N, B, obs)
            t = torch.arange(0, self.N).to(self.device)
            t = t.repeat_interleave(B).reshape(self.N, B)
            alphabars = self.alphabars[t]
            eps = torch.randn_like(sp)
            xt = alphabars.sqrt() * sp + (1 - alphabars).sqrt() * eps
            t = t.unsqueeze(-1)
        else:
            xt = sp.unsqueeze(0)  # (1, B, obs)
            t = None

        # Transition model output: (N, B, joint_dim).
        z_mu = self.forward_M(xt, t)
        z_mu = z_mu.reshape(z_mu.shape[0], B, -1)  # (N, B, joint_dim)

        z_phi = z_phi.unsqueeze(0).repeat(z_mu.shape[0], 1, 1)  # (N, B, joint_dim)
        logits = torch.bmm(z_phi, z_mu.transpose(-1, -2))  # (N, B, B)
        return logits

    #  Loss computation

    def compute_loss(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sp: torch.Tensor,
        r: torch.Tensor,
    ):
        """Return logged values, the model loss, and the reward prediction loss."""
        B = s.shape[0]

        z_s_full = self.forward_phi_s(s)  # (B, d_s_full)
        z_a_full = self.forward_phi_a(a)  # (B, d_a_full)

        z_s_nce = self.proj_s(z_s_full)  # (B, d_s)
        z_a_nce = self.proj_a(z_a_full)  # (B, d_a)

        logits = self.compute_logits(s, a, sp, z_s=z_s_nce, z_a=z_a_nce)  # (N, B, B)

        if self.linear:
            eff_logits = (F.softplus(logits, beta=3.0) + 1e-6).log()
            if self.ranking:
                labels = torch.arange(B, device=s.device).unsqueeze(0).expand(self.N, -1)
                model_loss = F.cross_entropy(eff_logits, labels, reduction="none").mean(-1)
            else:
                labels = torch.eye(B, device=s.device).unsqueeze(0).expand(self.N, -1, -1)
                eff_logits = eff_logits * self.normalizer.exp()[:, None, None] / B
                model_loss = F.binary_cross_entropy_with_logits(
                    eff_logits, labels, reduction="none"
                ).mean([-2, -1])
        else:
            eff_logits = logits
            if self.ranking:
                labels = torch.arange(B, device=s.device).unsqueeze(0).expand(self.N, -1)
                model_loss = F.cross_entropy(eff_logits, labels, reduction="none").mean(-1)
            else:
                labels = torch.eye(B, device=s.device).unsqueeze(0).expand(self.N, -1, -1)
                eff_logits = eff_logits + self.normalizer[:, None, None] - math.log(B)
                model_loss = F.binary_cross_entropy_with_logits(
                    eff_logits, labels, reduction="none"
                ).mean([-2, -1])

        z_concat = torch.cat([z_s_full, z_a_full], dim=-1)  # (B, d_s_full + d_a_full)
        reward_loss = F.mse_loss(self.reward(z_concat), r)

        pos_logits = logits[
            torch.arange(self.N).unsqueeze(1),
            torch.arange(B),
            torch.arange(B).unsqueeze(0).repeat(self.N, 1),
        ]  # (N, B)
        pos_logits_per_noise = pos_logits.mean(dim=1)  # (N,)
        neg_logits = (logits.sum(dim=-1) - pos_logits) / (logits.shape[-1] - 1)
        neg_logits_per_noise = neg_logits.mean(dim=-1)  # (N,)

        metrics = {
            "loss/model_loss": model_loss.mean().item(),
            "loss/reward_loss": reward_loss.item(),
            "misc/phi_s_norm": z_s_full.abs().mean().item(),
            "misc/phi_s_std": z_s_full.std(0).mean().item(),
            "misc/phi_a_norm": z_a_full.abs().mean().item(),
            "misc/phi_a_std": z_a_full.std(0).mean().item(),
        }
        checkpoints = list(range(0, self.N, 5))
        metrics.update(
            {f"detail/pos_logits_{i}": pos_logits_per_noise[i].item() for i in checkpoints}
        )
        metrics.update(
            {f"detail/neg_logits_{i}": neg_logits_per_noise[i].item() for i in checkpoints}
        )
        metrics.update(
            {
                f"detail/logit_gap_{i}": (pos_logits_per_noise[i] - neg_logits_per_noise[i]).item()
                for i in checkpoints
            }
        )
        metrics.update({f"detail/model_loss_{i}": model_loss[i].item() for i in checkpoints})
        return metrics, model_loss.mean(), reward_loss

    # Get features for the critic and exploration bonus.

    def compute_feature(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return state and action features for the critic.

        Their shapes are (B, d_s_full) and (B, d_a_full).
        """
        z_s = self.forward_phi_s(s)
        z_a = self.forward_phi_a(a)
        return z_s, z_a

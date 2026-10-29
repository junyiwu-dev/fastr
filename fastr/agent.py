# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Learn FaStR features, update the actor and critic, and choose actions."""

from operator import itemgetter

import torch
import torch.nn as nn

from .critic import (
    BilinearDoubleQ,
    HybridDoubleQ,
    ProductKernelDoubleQ,
    RffPsiDoubleQ,
    SpectralResidualDoubleQ,
)
from .network import FaStRNCE
from .nn import Actor
from .utils import convert_to_tensor, make_target, sync_target


class FaStR:

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg,
        device,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = cfg.discount
        self.tau = cfg.tau
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_freq = cfg.target_update_freq
        self.target_policy_noise = cfg.target_policy_noise
        self.noise_clip = cfg.noise_clip
        self.exploration_noise = cfg.exploration_noise
        self._step = 0

        self.d_s = cfg.d_s
        self.d_a = cfg.d_a
        self.d_s_full = getattr(cfg, "d_s_full", cfg.d_s)
        self.d_a_full = getattr(cfg, "d_a_full", cfg.d_a)
        self.feature_update_ratio = cfg.feature_update_ratio
        self.reward_coef = cfg.reward_coef
        self.back_critic_grad = float(cfg.back_critic_grad)  # 0.0=off, 1.0=full, 0.1=soft
        self.ctrl_coef = cfg.ctrl_coef
        self.critic_coef = cfg.critic_coef
        self.feature_tau = cfg.feature_tau

        self.bonus_coef = getattr(cfg, "bonus_coef", 0.0)
        self.cov_ema_alpha = getattr(cfg, "cov_ema_alpha", 0.1)
        self.lambda_s = getattr(cfg, "lambda_s", 1.0)
        self.lambda_a = getattr(cfg, "lambda_a", 1.0)
        self._eps = 1e-8  # Avoid division by zero and zero square roots.
        self._cov_metrics = {}  # Set by update_covariance().
        self.adaptive_beta = getattr(cfg, "adaptive_beta", False)
        self._q_ema = 1.0  # Running mean of abs(Q) for the bonus scale.

        # bonus_type: "factored" | "action_only" | "joint"
        # factored: multiply state and action uncertainty.
        # action_only: use action uncertainty.
        # joint: use the product of state and action features.
        self.bonus_type = getattr(cfg, "bonus_type", "factored")
        if self.bonus_type not in {"factored", "action_only", "joint"}:
            raise ValueError(f"Unknown bonus_type: {self.bonus_type}")
        if self.bonus_type == "joint" and self.d_s_full != self.d_a_full:
            raise ValueError("joint bonus requires equal state/action feature dimensions")

        self.nce = FaStRNCE(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_s=cfg.d_s,
            d_a=cfg.d_a,
            phi_s_hidden_dims=cfg.phi_s_hidden_dims,
            phi_a_hidden_dims=cfg.phi_a_hidden_dims,
            mu_hidden_dims=cfg.mu_hidden_dims,
            reward_hidden_dims=cfg.reward_hidden_dims,
            rff_dim=cfg.rff_dim,
            num_noises=cfg.num_noises,
            linear=cfg.linear,
            ranking=cfg.ranking,
            diagonal_m=getattr(cfg, "diagonal_m", False),
            d_s_full=self.d_s_full,
            d_a_full=self.d_a_full,
            device=device,
        ).to(device)
        self.nce_target = make_target(self.nce)
        self.feature_optim = torch.optim.Adam(
            self.nce.parameters(),
            lr=cfg.feature_lr,
        )

        self.actor = Actor(
            input_dim=self.obs_dim,
            output_dim=self.action_dim,
            hidden_dims=cfg.actor_hidden_dims,
            norm_layer=nn.LayerNorm,
        ).to(self.device)

        use_bias = getattr(cfg, "use_bias", True)
        critic_type = getattr(cfg, "critic_type", "bilinear")
        critic_rff_dim = getattr(cfg, "critic_rff_dim", 1024)
        critic_hidden = getattr(cfg, "critic_hidden_dims", [512])
        self.critic_type = critic_type
        if critic_type == "hybrid":
            self.critic = HybridDoubleQ(
                d_s=self.d_s_full,
                d_a=self.d_a_full,
                rff_dim=critic_rff_dim,
                hidden_dims=critic_hidden,
                use_bias=use_bias,
            ).to(self.device)
        elif critic_type == "product_kernel":
            self.critic = ProductKernelDoubleQ(
                d_s=self.d_s_full,
                d_a=self.d_a_full,
                rff_dim=critic_rff_dim,
                hidden_dims=critic_hidden,
                use_bias=use_bias,
            ).to(self.device)
        elif critic_type == "spectral_residual":
            # Add a linear value term and a neural network term.
            # State and action features must have the same size.
            assert self.d_s_full == self.d_a_full, (
                f"spectral_residual critic requires d_s == d_a (diagM), "
                f"got d_s={self.d_s_full}, d_a={self.d_a_full}"
            )
            self.critic = SpectralResidualDoubleQ(
                d=self.d_s_full,
                rff_dim=critic_rff_dim,
                hidden_dims=critic_hidden,
            ).to(self.device)
        elif critic_type == "rff_psi":
            # Apply the value network to the product of the two features.

            # State and action features must have the same size.
            assert self.d_s_full == self.d_a_full, (
                f"rff_psi critic requires d_s == d_a (diagM), "
                f"got d_s={self.d_s_full}, d_a={self.d_a_full}"
            )
            self.critic = RffPsiDoubleQ(
                d=self.d_s_full,
                rff_dim=critic_rff_dim,
                hidden_dims=critic_hidden,
            ).to(self.device)
        elif critic_type == "bilinear":
            self.critic = BilinearDoubleQ(
                d_s=self.d_s_full,
                d_a=self.d_a_full,
                use_bias=use_bias,
            ).to(self.device)

        else:
            raise ValueError(f"Unknown critic_type: {critic_type}")

        self.actor_target = make_target(self.actor)
        self.critic_target = make_target(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # Track squared state and action features separately.
        # In joint mode, track their product instead.
        self.thompson_sampling = getattr(cfg, "thompson_sampling", True)
        self._var_s = self.lambda_s * torch.ones(self.d_s_full, device=self.device)
        self._var_a = self.lambda_a * torch.ones(self.d_a_full, device=self.device)
        self._var_psi = torch.ones(self.d_s_full, device=self.device)  # for joint mode

    #  Training mode

    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)
        self.nce.train(training)

    @torch.no_grad()
    def select_action(self, obs, step=None, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, ...]
        action = self.actor(obs)
        if not deterministic:
            action = (action + self.exploration_noise * torch.randn_like(action)).clamp(-1, 1)
        return action.squeeze(0).cpu().numpy()

    # Update running feature statistics.

    @torch.no_grad()
    def update_covariance(self, obs: torch.Tensor, action: torch.Tensor):
        """Update running means of squared features.

        Use separate state and action statistics, or statistics of their product.
        """
        alpha = self.cov_ema_alpha

        z_s = self.nce.forward_phi_s(obs)  # (B, d_s_full)
        z_a = self.nce.forward_phi_a(action)  # (B, d_a_full)

        if self.bonus_type in ("factored", "action_only"):
            # Update the mean square for each state and action feature.
            var_s_batch = (z_s**2).mean(dim=0)  # (d_s_full,)
            var_a_batch = (z_a**2).mean(dim=0)  # (d_a_full,)
            self._var_s = alpha * var_s_batch + (1 - alpha) * self._var_s
            self._var_a = alpha * var_a_batch + (1 - alpha) * self._var_a

            self._cov_metrics = {
                "bonus/var_s_mean": self._var_s.mean().item(),
                "bonus/var_s_max": self._var_s.max().item(),
                "bonus/var_s_min": self._var_s.min().item(),
                "bonus/var_a_mean": self._var_a.mean().item(),
                "bonus/var_a_max": self._var_a.max().item(),
                "bonus/var_a_min": self._var_a.min().item(),
            }
        else:
            # Update the mean square of the feature product.
            psi = z_s * z_a  # (B, d)
            var_psi_batch = (psi**2).mean(dim=0)  # (d,)
            self._var_psi = alpha * var_psi_batch + (1 - alpha) * self._var_psi

            self._cov_metrics = {
                "bonus/var_psi_mean": self._var_psi.mean().item(),
                "bonus/var_psi_max": self._var_psi.max().item(),
                "bonus/var_psi_min": self._var_psi.min().item(),
            }

    #  BONUS COMPUTATION

    def compute_bonus(
        self,
        z_s: torch.Tensor,
        z_a: torch.Tensor,
    ) -> torch.Tensor:
        """Return an exploration bonus from the running feature statistics.

        The factored mode multiplies state and action uncertainty.
        The action_only mode uses action uncertainty.
        The joint mode uses the product of state and action features.
        With adaptive_beta, divide the bonus scale by the running mean of abs(Q).
        """
        if self.bonus_coef == 0.0:
            return torch.zeros(z_s.shape[0], 1, device=z_s.device)

        # Use a fixed scale or adjust it using abs(Q).
        if self.adaptive_beta:
            effective_beta = self.bonus_coef / (self._q_ema + self._eps)
        else:
            effective_beta = self.bonus_coef

        if self.bonus_type == "action_only":
            # Use only action uncertainty.

            norm_a = torch.sqrt(
                (z_a**2 / (self._var_a + self.lambda_a)).sum(dim=-1, keepdim=True) + self._eps
            )  # (B, 1)
            bonus = effective_beta * norm_a

        elif self.bonus_type == "factored":
            # Use both state and action uncertainty.
            # The state features are fixed during this actor update.
            norm_s = torch.sqrt(
                (z_s**2 / (self._var_s + self.lambda_s)).sum(dim=-1, keepdim=True) + self._eps
            )  # (B, 1)
            # The actor can change the bonus through its action.
            norm_a = torch.sqrt(
                (z_a**2 / (self._var_a + self.lambda_a)).sum(dim=-1, keepdim=True) + self._eps
            )  # (B, 1)
            bonus = effective_beta * norm_s * norm_a

        else:  # "joint"
            # Use uncertainty in the product features.
            psi = z_s * z_a  # Only the action features pass gradients to the actor.
            bonus = effective_beta * torch.sqrt(
                (psi**2 / (self._var_psi + self.lambda_s)).sum(dim=-1, keepdim=True) + self._eps
            )

        return bonus

    #  TRAINING LOOP

    def train_step(self, buffer, batch_size):
        tot_metrics = {}

        for _ in range(self.feature_update_ratio):
            batch = buffer.sample(batch_size)
            obs, action, next_obs, reward, terminal = [
                convert_to_tensor(b, self.device)[:batch_size]
                for b in itemgetter("obs", "action", "next_obs", "reward", "terminal")(batch)
            ]

            feature_metrics, ctrl_loss, reward_loss = self.feature_step(
                obs, action, next_obs, reward
            )
            tot_metrics.update(feature_metrics)

            critic_metrics, critic_loss = self.critic_step(obs, action, next_obs, reward, terminal)
            tot_metrics.update(critic_metrics)

            self.feature_optim.zero_grad()
            self.critic_optim.zero_grad()
            (
                ctrl_loss * self.ctrl_coef
                + reward_loss * self.reward_coef
                + critic_loss * self.critic_coef
            ).backward()
            self.critic_optim.step()
            self.feature_optim.step()

            if self.bonus_coef > 0.0:
                self.update_covariance(obs, action)
                tot_metrics.update(self._cov_metrics)

        if self._step % self.actor_update_freq == 0:
            actor_metrics, actor_loss = self.actor_step(obs)
            tot_metrics.update(actor_metrics)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        if self._step % self.target_update_freq == 0:
            self.sync_target()

        self._step += 1
        return tot_metrics

    #  Feature step  (contrastive + reward)

    def feature_step(self, obs, action, next_obs, reward):
        metrics, ctrl_loss, reward_loss = self.nce.compute_loss(obs, action, next_obs, reward)
        return metrics, ctrl_loss, reward_loss

    # Update the critic. The target value does not include the bonus.

    def critic_step(self, obs, action, next_obs, reward, terminal):
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.target_policy_noise).clip(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_obs) + noise).clip(-1.0, 1.0)

            next_z_s, next_z_a = self.get_feature(next_obs, next_action, use_target=True)
            q_target = self.critic_target(next_z_s, next_z_a).min(0)[0]
            q_target = reward + self.discount * (1 - terminal) * q_target

        if self.critic_type == "spectral_residual":
            # The linear value term updates the encoder.
            # The neural network term does not update the encoder.

            # Use the current features for this critic type.
            z_s, z_a = self.get_feature(obs, action, use_target=False)
        elif self.back_critic_grad > 0:
            z_s_live, z_a_live = self.get_feature(obs, action, use_target=False)
            # Scale encoder gradients by back_critic_grad.
            alpha = self.back_critic_grad
            z_s = z_s_live * alpha + z_s_live.detach() * (1 - alpha)
            z_a = z_a_live * alpha + z_a_live.detach() * (1 - alpha)
        else:
            z_s, z_a = self.get_feature(obs, action, use_target=True)
            z_s = z_s.detach()
            z_a = z_a.detach()

        q_pred = self.critic(z_s, z_a)  # (2, B, 1)
        critic_loss = (q_target - q_pred).pow(2).sum(0).mean()

        # Track abs(Q) to adjust the bonus scale.
        if self.adaptive_beta and self.bonus_coef > 0:
            with torch.no_grad():
                q_abs_mean = q_pred.detach().abs().mean().item()
                self._q_ema = 0.99 * self._q_ema + 0.01 * max(q_abs_mean, 1.0)

        metrics = {
            "loss/critic_loss": critic_loss.item(),
        }
        # Log the linear value term and its share of the total value.
        if self.critic_type == "spectral_residual":
            with torch.no_grad():
                q_struct = self.critic.forward_struct_only(z_s.detach(), z_a.detach())
                metrics["misc/q_struct_mean"] = q_struct.mean().item()
                metrics["misc/q_struct_ratio"] = (
                    q_struct.abs().mean() / (q_pred.abs().mean() + 1e-8)
                ).item()

        return metrics, critic_loss

    #  Actor step  (maximize Q + bonus)

    def actor_step(self, obs):
        new_action = self.actor(obs)

        # This critic always uses current features.
        # Other critics follow back_critic_grad.
        if self.critic_type == "spectral_residual" or self.back_critic_grad > 0:
            z_s, z_a = self.get_feature(obs, new_action, use_target=False)
        else:
            z_s, z_a = self.get_feature(obs, new_action, use_target=True)

        if (
            self.bonus_coef > 0.0
            and self.thompson_sampling
            and hasattr(self.critic, "forward_thompson")
        ):
            # Add noise to W instead of adding a bonus to Q.
            inv_std_s = 1.0 / torch.sqrt(self._var_s + self.lambda_s)
            inv_std_a = 1.0 / torch.sqrt(self._var_a + self.lambda_a)
            q_value = self.critic.forward_thompson(
                z_s, z_a, self.bonus_coef, inv_std_s, inv_std_a
            )  # (2, B, 1)
            bonus = torch.zeros(z_s.shape[0], 1, device=z_s.device)
        else:
            q_value = self.critic(z_s, z_a)  # (2, B, 1)
            # Factored UCB bonus (if enabled)
            bonus = self.compute_bonus(z_s.detach(), z_a)  # (B, 1)

        actor_loss = -(q_value.mean(0) + bonus).mean()

        metrics = {
            "misc/q_value_mean": q_value.mean().item(),
            "misc/q_value_std": q_value.std(0).mean().item(),
            "misc/q_value_min": q_value.min(0)[0].mean().item(),
            "loss/actor_loss": actor_loss.item(),
        }
        if self.bonus_coef > 0.0:
            with torch.no_grad():
                q_base = self.critic(z_s.detach(), z_a.detach())
                z_s_d = z_s.detach()
                z_a_d = z_a.detach()
                if self.bonus_type in ("factored", "action_only"):
                    state_unc = torch.sqrt(
                        (z_s_d**2 / (self._var_s + self.lambda_s)).sum(-1) + self._eps
                    )
                    action_unc = torch.sqrt(
                        (z_a_d**2 / (self._var_a + self.lambda_a)).sum(-1) + self._eps
                    )
                else:
                    psi_d = z_s_d * z_a_d
                    state_unc = torch.sqrt(
                        (psi_d**2 / (self._var_psi + self.lambda_s)).sum(-1) + self._eps
                    )
                    action_unc = state_unc

                effective_beta = (
                    self.bonus_coef / (self._q_ema + self._eps)
                    if self.adaptive_beta
                    else self.bonus_coef
                )
            metrics.update(
                {
                    "bonus/bonus_mean": bonus.mean().item(),
                    "bonus/bonus_type": {"factored": 0, "action_only": 1, "joint": 2}[
                        self.bonus_type
                    ],
                    "bonus/q_base_mean": q_base.mean().item(),
                    "bonus/state_unc_mean": state_unc.mean().item(),
                    "bonus/action_unc_mean": action_unc.mean().item(),
                    "bonus/effective_beta": effective_beta,
                }
            )
            if self.adaptive_beta:
                metrics["bonus/q_ema"] = self._q_ema

        return metrics, actor_loss

    # Get state and action features.

    def get_feature(
        self, obs: torch.Tensor, action: torch.Tensor, use_target: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.nce_target if use_target else self.nce
        return model.compute_feature(obs, action)

    # Update target networks.

    def sync_target(self):
        sync_target(self.critic, self.critic_target, self.tau)
        sync_target(self.actor, self.actor_target, self.tau)
        sync_target(self.nce, self.nce_target, self.feature_tau)

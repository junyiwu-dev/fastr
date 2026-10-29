# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Train FaStR: python -m fastr.train env=dmc."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from .agent import FaStR
from .envs import make_env
from .logger import Logger
from .replay import ReplayBuffer
from .utils import set_device, set_seed


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._validate_config()
        seed = set_seed(cfg.seed)
        self.device = set_device(cfg.device)
        self.train_env = make_env(cfg, seed)
        self.eval_env = make_env(cfg, seed + 100)
        self.buffer = ReplayBuffer(
            self.train_env.observation_space.shape[0],
            self.train_env.action_space.shape[0],
            cfg.buffer_size,
            cfg.norm_obs,
            cfg.norm_reward,
        )
        self.agent = FaStR(
            self.train_env.observation_space.shape[0],
            self.train_env.action_space.shape[0],
            cfg.algo,
            self.device,
        )
        self.logger = Logger(
            Path(cfg.log_dir) / "fastr" / cfg.name / cfg.task, f"seed{seed}", active=not cfg.debug
        )
        if not cfg.debug:
            OmegaConf.save(cfg, self.logger.log_dir / "config.yaml", resolve=True)
            print(f"Run directory: {self.logger.log_dir}")
        self.global_step = 0

    def _validate_config(self):
        cfg = self.cfg
        for key in (
            "frame_skip",
            "buffer_size",
            "batch_size",
            "eval_frames",
            "eval_episodes",
            "log_frames",
        ):
            if cfg[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if cfg.backend == "gym" and cfg.frame_skip != 1:
            raise ValueError("Gymnasium runs require frame_skip=1")
        if cfg.buffer_size < cfg.batch_size:
            raise ValueError("buffer_size must be at least batch_size")
        if cfg.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning")
        for key in ("train_frames", "warmup_frames", "random_frames"):
            if cfg[key] < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("feature_update_ratio", "actor_update_freq", "target_update_freq"):
            if cfg.algo[key] <= 0:
                raise ValueError(f"algo.{key} must be positive")

    @property
    def global_frame(self):
        return self.global_step * self.cfg.frame_skip

    def train(self):
        cfg = self.cfg
        ep_length = ep_return = 0
        ep_success = False
        obs, _ = self.train_env.reset()
        try:
            for _ in trange(int(cfg.train_frames // cfg.frame_skip), desc="fastr"):
                if self.global_frame < cfg.random_frames:
                    action = self.train_env.action_space.sample()
                else:
                    action = self.agent.select_action(self.buffer.normalize_obs(obs))
                next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                self.buffer.add(obs, action, next_obs, reward, terminated)
                obs = next_obs
                self.global_step += 1
                ep_length += 1
                ep_return += reward
                ep_success |= bool(info.get("success", False))
                if terminated or truncated:
                    metrics = {"episode_return": ep_return, "episode_length": ep_length}
                    if self.train_env.is_robotics:
                        metrics["episode_success"] = float(ep_success)
                    self.logger.log_scalars("rollout", metrics, self.global_frame)
                    obs, _ = self.train_env.reset()
                    ep_length = ep_return = 0
                    ep_success = False

                train_metrics = {}
                if self.global_frame >= cfg.warmup_frames and self.buffer.size >= cfg.batch_size:
                    train_metrics = self.agent.train_step(self.buffer, cfg.batch_size)
                previous_frame = self.global_frame - cfg.frame_skip
                if self.global_frame // cfg.log_frames > previous_frame // cfg.log_frames:
                    self.logger.log_scalars("", train_metrics, self.global_frame)
                if self.global_frame // cfg.eval_frames > previous_frame // cfg.eval_frames:
                    self.logger.log_scalars("eval", self.evaluate(), self.global_frame)
        finally:
            self.close()

    def evaluate(self):
        self.agent.train(False)
        returns, lengths, successes = [], [], []
        try:
            for _ in range(self.cfg.eval_episodes):
                obs, _ = self.eval_env.reset()
                length = episode_return = 0
                success = False
                terminated = truncated = False
                while not (terminated or truncated):
                    action = self.agent.select_action(
                        self.buffer.normalize_obs(obs), deterministic=True
                    )
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    length += 1
                    episode_return += reward
                    success |= bool(info.get("success", False))
                returns.append(episode_return)
                lengths.append(length)
                successes.append(success)
        finally:
            self.agent.train(True)
        metrics = {
            "return_mean": np.mean(returns),
            "return_std": np.std(returns),
            "length_mean": np.mean(lengths),
        }
        if self.eval_env.is_robotics:
            metrics["success_rate"] = np.mean(successes)
        return metrics

    def close(self):
        self.train_env.close()
        self.eval_env.close()
        self.logger.close()


@hydra.main(
    version_base=None, config_path=str(Path(__file__).parent / "config"), config_name="config"
)
def main(cfg: DictConfig):
    Trainer(cfg).train()


if __name__ == "__main__":
    main()

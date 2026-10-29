# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Load the task package needed for this run."""


def make_env(cfg, seed):
    if cfg.backend == "dmc":
        from .dmc import DMControlEnv

        return DMControlEnv(cfg.task, seed, cfg.frame_skip)
    if cfg.backend == "gym":
        from .gym import GymEnv

        return GymEnv(cfg.task, seed, cfg.max_episode_steps)
    raise ValueError(f"Unknown environment backend: {cfg.backend}")

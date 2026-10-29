# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Gymnasium tasks with flat state inputs and actions in [-1, 1]."""

import gymnasium as gym
import numpy as np


class GymEnv:
    def __init__(self, task, seed, max_episode_steps=None):
        self.is_robotics = task.startswith(("AdroitHand", "HandManipulate", "HandReach"))
        if self.is_robotics:
            import gymnasium_robotics

            gym.register_envs(gymnasium_robotics)
        kwargs = {} if max_episode_steps is None else {"max_episode_steps": max_episode_steps}
        self.env = gym.make(task, **kwargs)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self._goal_obs = isinstance(self.env.observation_space, gym.spaces.Dict)
        if self._goal_obs:
            # Keep the state and the desired goal as inputs.
            spaces = self.env.observation_space
            size = int(np.prod(spaces["observation"].shape) + np.prod(spaces["desired_goal"].shape))
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        else:
            self.observation_space = self.env.observation_space
        self._low = self.env.action_space.low
        self._high = self.env.action_space.high
        if not (np.isfinite(self._low).all() and np.isfinite(self._high).all()):
            raise ValueError("FaStR requires bounded continuous actions")
        self.action_space = gym.spaces.Box(-1.0, 1.0, self._low.shape, np.float32)
        self.action_space.seed(seed)

    def _observation(self, obs):
        if self._goal_obs:
            obs = np.concatenate([obs["observation"].ravel(), obs["desired_goal"].ravel()])
        return np.asarray(obs, dtype=np.float32).ravel()

    def reset(self):
        obs, info = self.env.reset()
        return self._observation(obs), info

    def step(self, action):
        action = self._low + (np.clip(action, -1, 1) + 1) * 0.5 * (self._high - self._low)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.is_robotics:
            info["success"] = bool(info.get("success", info.get("is_success", False)))
        return self._observation(obs), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()

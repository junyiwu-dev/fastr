# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""DeepMind Control Suite tasks with state inputs."""

import gymnasium as gym
import numpy as np
from dm_control import suite
from dm_control.suite import pendulum
from dm_control.suite.wrappers import action_scale


class SwingUpDense(pendulum.SwingUp):
    """Pendulum swingup with a reward at each step."""

    def get_reward(self, physics):
        return pendulum.rewards.tolerance(
            physics.pole_vertical(), (pendulum._COSINE_BOUND, 1), margin=1
        )


class DMControlEnv:
    is_robotics = False

    def __init__(self, task, seed, frame_skip=2, horizon=1000):
        domain, task_name = task.split("-", 1)
        if domain == "pendulum" and task_name == "swingup_dense":
            physics = pendulum.Physics.from_xml_string(*pendulum.get_model_and_assets())
            env = pendulum.control.Environment(
                physics, SwingUpDense(random=seed), time_limit=pendulum._DEFAULT_TIME_LIMIT
            )
        else:
            env = suite.load(
                domain, task_name, task_kwargs={"random": seed}, visualize_reward=False
            )
        self.env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
        self.frame_skip = frame_skip
        self.horizon = horizon
        size = sum(int(np.prod(spec.shape)) for spec in self.env.observation_spec().values())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, self.env.action_spec().shape, np.float32)
        self.action_space.seed(seed)

    @staticmethod
    def _observation(time_step):
        return np.concatenate(
            [np.asarray(v).ravel() for v in time_step.observation.values()]
        ).astype(np.float32)

    def reset(self):
        self.frames = 0
        return self._observation(self.env.reset()), {}

    def step(self, action):
        reward = 0.0
        for _ in range(self.frame_skip):
            time_step = self.env.step(np.asarray(action, dtype=np.float32))
            reward += time_step.reward or 0.0
            self.frames += 1
            if time_step.last() or self.frames >= self.horizon:
                break
        terminated = time_step.last() and time_step.discount == 0
        truncated = (time_step.last() or self.frames >= self.horizon) and not terminated
        return self._observation(time_step), reward, bool(terminated), bool(truncated), {}

    def close(self):
        self.env.close()

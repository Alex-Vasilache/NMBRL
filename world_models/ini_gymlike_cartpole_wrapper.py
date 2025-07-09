import os
import sys

# This is a hacky way to ensure the imports from the submodule work
submodule_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "environments", "CartPoleSimulation")
)
gymlike_root = os.path.join(submodule_root, "GymlikeCartPole")

if submodule_root not in sys.path:
    sys.path.insert(0, submodule_root)
if gymlike_root not in sys.path:
    sys.path.insert(0, gymlike_root)

from EnvGym.CartpoleEnv import CartPoleEnv
from EnvGym.state_utils import (
    ANGLE_IDX,
    ANGLED_IDX,
    ANGLE_COS_IDX,
    ANGLE_SIN_IDX,
    POSITION_IDX,
)


from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


#  - "stabilization" → balance only from near-upright starts
#  - "swing_up"      → random starts + swing-up reward shaping
TASK = "swingup"

CARTPOLE_TYPE = "custom_sim"  # "openai", "custom_sim", "physical"

SEED = 42
N_ENVS = 16

import numpy as np


def make_env(render_mode="none", max_episode_steps=1000):
    """
    Instantiate the CartPoleEnv, then wrap with Monitor.
    Monitor records episode reward/length for logging callbacks.
    """

    def init_state(rng: np.random.Generator) -> np.ndarray:
        """
        Provide an initial 6-D state vector starting with pole down.
        """
        low, high = -0.05, 0.05
        s = rng.uniform(low=low, high=high, size=(6,))
        # Start with pole down (angle = π)
        s[ANGLE_IDX] = np.pi + rng.uniform(-0.05, 0.05)
        s[ANGLE_COS_IDX] = np.cos(s[ANGLE_IDX])
        s[ANGLE_SIN_IDX] = np.sin(s[ANGLE_IDX])
        return s.astype(np.float32)

    def _init():
        env = CartPoleEnv(
            render_mode=render_mode,
            task=TASK,
            cartpole_type=CARTPOLE_TYPE,
            max_episode_steps=max_episode_steps,
        )

        env.task.init_state = init_state
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return Monitor(env)

    return _init


class GymlikeCartpoleWrapper(VecNormalize):
    def __init__(
        self,
        seed: int = 42,
        n_envs: int = 16,
        render_mode: str = "none",
        max_episode_steps: int = 1000,
    ):
        self.task = TASK
        self._seed = seed
        self.n_envs = n_envs
        self.cartpole_type = CARTPOLE_TYPE
        self.max_episode_steps = max_episode_steps

        env_fns = [make_env(render_mode, max_episode_steps) for _ in range(self.n_envs)]

        if n_envs > 1:
            # Create the vectorized environment first
            vec_env = SubprocVecEnv(env_fns)

        else:
            vec_env = DummyVecEnv(env_fns)

        print(vec_env.observation_space)
        print(vec_env.action_space)

        # Initialize the parent VecNormalize class
        super().__init__(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

        # Set the seed
        self.seed(self._seed)

    def step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()

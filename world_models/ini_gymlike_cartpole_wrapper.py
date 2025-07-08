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

from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


#  - "stabilization" → balance only from near-upright starts
#  - "swing_up"      → random starts + swing-up reward shaping
TASK = "swingup"

CARTPOLE_TYPE = "custom_sim"  # "openai", "custom_sim", "physical"

SEED = 42
N_ENVS = 16


def make_env():
    """
    Instantiate the CartPoleEnv, then wrap with Monitor.
    Monitor records episode reward/length for logging callbacks.
    """
    env = CartPoleEnv(render_mode=None, task=TASK, cartpole_type=CARTPOLE_TYPE)
    env = TimeLimit(env, max_episode_steps=env.max_episode_steps)
    return Monitor(env)


class GymlikeCartpoleWrapper(VecNormalize):
    def __init__(self, seed: int = 42, n_envs: int = 16):
        self.task = TASK
        self._seed = seed
        self.n_envs = n_envs
        self.cartpole_type = CARTPOLE_TYPE

        # Create the vectorized environment first
        vec_env = SubprocVecEnv([make_env for _ in range(self.n_envs)])

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

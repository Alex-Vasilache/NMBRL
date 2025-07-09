# This file will contain a wrapper for the world model.
# The world model will be a neural network that predicts the next state of the environment.

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch import nn


def make_env():
    """
    Instantiate the CartPoleEnv, then wrap with Monitor.
    Monitor records episode reward/length for logging callbacks.
    """
    env = WorldModelNeuralNetwork()
    return Monitor(env)


class WorldModelWrapper(VecNormalize):
    def __init__(self, seed: int = 42, n_envs: int = 16):
        self._seed = seed
        self.n_envs = n_envs
        self.env = make_env()

        # Initialize the parent VecNormalize class
        super().__init__(
            self.env,
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

class WorldModelNeuralNetwork(nn.Module):
    

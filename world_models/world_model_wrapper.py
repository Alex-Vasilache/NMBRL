# This file will contain a wrapper for the world model.
# The world model will be a neural network that predicts the next state of the environment.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from world_models.world_model_v1 import load_model
import os


class WorldModelWrapper(DummyVecEnv):
    def __init__(
        self,
        simulated_env: gym.Env,
        batch_size: int = 1,
        trained_folder: str = "world_models/trained/v1",
    ):

        model_path = os.path.join(trained_folder, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.nn_model = load_model(model_path)

        # Initialize from the simulated environment
        self.observation_space = simulated_env.observation_space
        self.action_space = simulated_env.action_space

        assert isinstance(self.observation_space, spaces.Box)
        self.state_size = self.observation_space.shape[0]

        assert isinstance(self.action_space, spaces.Box)
        self.action_size = self.action_space.shape[0]

        # A hack to create a valid gym env for DummyVecEnv to wrap
        class DummyGymEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space

            def reset(self, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0, False, False, {}

        # We don't use the DummyVecEnv's env list, but this sets self.num_envs
        super().__init__(
            [
                lambda: DummyGymEnv(self.observation_space, self.action_space)
                for _ in range(batch_size)
            ]
        )

        self.state = None

        # Override buffers from DummyVecEnv with correct shapes
        self.buf_obs = np.zeros(
            (self.num_envs, self.state_size), dtype=self.observation_space.dtype
        )
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def step(self, action_np):  # Overrides DummyVecEnv.step
        action_tensor = (
            torch.from_numpy(action_np).float().reshape(self.num_envs, self.action_size)
        )

        with torch.no_grad():
            if self.state is None:
                raise RuntimeError("Must call reset() or set_state() before step().")

            nn_input = torch.cat([self.state, action_tensor], dim=1)
            outputs = self.nn_model(nn_input)

            next_state_tensor = outputs[:, : self.state_size]
            reward_tensor = outputs[:, self.state_size]

            self.state = next_state_tensor

        terminated = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        return next_state_tensor.numpy(), reward_tensor.numpy(), terminated, infos

    def reset(self):  # Overrides DummyVecEnv.reset
        self.state = torch.zeros((self.num_envs, self.state_size), dtype=torch.float32)
        return self.state.numpy()

    def set_state(self, state_np: np.ndarray):
        """Sets the internal state of the world model."""
        if state_np.ndim == 1:
            state_np = np.expand_dims(state_np, axis=0)

        expected_shape = (self.num_envs, self.state_size)
        if state_np.shape != expected_shape:
            raise ValueError(
                f"Expected state shape {expected_shape}, but got {state_np.shape}"
            )

        self.state = torch.from_numpy(state_np).float()

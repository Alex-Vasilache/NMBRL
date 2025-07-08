import numpy as np
import math
import sys
import os
import time
import torch
from typing import Optional

from .base_world_model import BaseWorldModel
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit


class INIGymlikeCartPoleWrapper(BaseWorldModel):
    """
    A wrapper for the Gym-like CartPole environment to make it compatible with the BaseWorldModel interface.
    This allows the RL agent to interact with the "real" environment in the same way it would
    interact with a learned world model. Supports batching for parallel environment execution.
    """

    def __init__(
        self,
        max_steps=500,
        task="swingup",
        cartpole_type="custom_sim",
        visualize=False,
        **kwargs,
    ):
        """
        Initializes the INIGymlikeCartPoleWrapper.

        :param max_steps: Maximum number of steps per episode (default: 500)
        :param task: The task for the environment, e.g., "swingup" or "stabilization" (default: "swingup")
        :param cartpole_type: The type of cartpole to use, e.g., "custom_sim" or "openai" (default: "custom_sim")
        :param visualize: If True, a window with the CartPole visualization will be opened (default: False)
        :param kwargs: Additional arguments.
        """
        self.batch_size = 0  # Will be set by reset()

        # This is a hacky way to ensure the imports from the submodule work
        submodule_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "environments", "CartPoleSimulation"
            )
        )
        gymlike_root = os.path.join(submodule_root, "GymlikeCartPole")

        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)
        if gymlike_root not in sys.path:
            sys.path.insert(0, gymlike_root)

        from EnvGym.CartpoleEnv import CartPoleEnv

        self.CartPoleEnv = CartPoleEnv
        self.task = task
        self.cartpole_type = cartpole_type

        # Manual termination parameters
        self.max_steps = max_steps

        # Create a temporary env to get space information
        temp_env = self._make_env()
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()

        self.visualize = visualize

        self.vec_env: Optional[DummyVecEnv] = None

        # Initialize with a default batch size of 1
        self.reset(batch_size=1)

    @property
    def envs(self):
        if self.vec_env:
            return self.vec_env.envs
        return []

    def _make_env(self, render_mode=None):
        env = self.CartPoleEnv(
            task=self.task, cartpole_type=self.cartpole_type, render_mode=render_mode
        )
        env = TimeLimit(env, max_episode_steps=self.max_steps)
        return env

    def step(self, actions: np.ndarray):
        """
        Steps through each environment in the batch.

        :param actions: A numpy array of actions, shape (batch_size, action_dim)
        :return: A tuple of (next_states, rewards, dones, info)
        """
        if not self.vec_env:
            # This should not happen if reset() is called in __init__, but it satisfies the linter
            empty_obs = np.zeros((self.batch_size, *self.observation_space.shape))
            return (
                empty_obs,
                np.zeros(self.batch_size),
                np.ones(self.batch_size, dtype=bool),
                [{}] * self.batch_size,
            )

        # Ensure actions are float32
        actions = actions.astype(np.float32)
        next_states, rewards, dones, infos = self.vec_env.step(actions)

        if self.visualize:
            self._render()

        return next_states, rewards, dones, infos

    def reset(self, batch_size=None, initial_state=None):
        """
        Resets the environments.

        :param batch_size: The number of parallel environments to create.
        :param initial_state: A specific state or batch of states to reset the environments to.
        :return: The initial states.
        """
        if batch_size is not None:
            self.batch_size = batch_size

        if self.vec_env:
            self.vec_env.close()

        env_fns = []
        for i in range(self.batch_size):
            render_mode = "human" if self.visualize and i == 0 else None
            env_fns.append(
                lambda render_mode=render_mode: self._make_env(render_mode=render_mode)
            )

        self.vec_env = DummyVecEnv(env_fns)

        initial_states = self.vec_env.reset()

        if initial_state is not None:
            if torch.is_tensor(initial_state):
                initial_state = initial_state.detach().cpu().numpy()

            if initial_state.ndim == 1:
                # If a single state is provided, repeat it for the whole batch
                initial_state_batch = np.tile(initial_state, (self.batch_size, 1))
            else:
                initial_state_batch = initial_state

            # This is a bit of a hack, as we are accessing the underlying envs directly
            # And assuming the 'state' attribute can be set.
            for i in range(self.batch_size):
                self.vec_env.envs[i].state = initial_state_batch[i]  # type: ignore

            # If we set the state manually, we should return that state
            initial_states = initial_state_batch

        return initial_states

    def _render(self):
        """Renders the first environment."""
        if self.visualize and self.vec_env:
            self.vec_env.render()

    def close(self):
        """Closes all environments."""
        if self.vec_env:
            self.vec_env.close()
        self.vec_env = None

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

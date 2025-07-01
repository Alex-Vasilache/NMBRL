# This file will contain a wrapper for the existing CartPole environment (and others in the future).
# This wrapper will implement the `BaseWorldModel` interface, allowing the agent to interact
# with the real environment as if it were a learned world model.

import numpy as np
import math

from Neuromorphic_MBRL.world_models.base_world_model import BaseWorldModel
from CartPole import CartPole as RealCartPole
from CartPole.state_utilities import create_cartpole_state


class EnvironmentWrapper(BaseWorldModel):
    """
    A wrapper for the existing CartPole environment to make it compatible with the BaseWorldModel interface.
    This allows the RL agent to interact with the "real" environment in the same way it would
    interact with a learned world model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the EnvironmentWrapper.

        :param kwargs: Arguments to be passed to the underlying CartPole environment.
        """
        # Define thresholds for termination
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Initialize the "real" CartPole environment
        self.env = RealCartPole()

        # Set a default simulation timestep if not provided
        self.env.dt_simulation = kwargs.get("dt_simulation", 0.02)

        self.reset()

    def step(self, action: np.ndarray):
        """
        Takes a single step in the environment.

        :param action: A numpy array representing the action to take, corresponding to the dimensionless motor power Q.
        :return: A tuple containing (next_state, reward, terminated, info).
        """
        # Set the action (motor power)
        self.env.Q = float(action[0])

        # Update the environment state
        self.env.update_state()

        # Get the new state
        state = self.env.s_with_noise_and_latency

        # Determine if the episode is terminated
        terminated = bool(
            state[0] < -self.x_threshold
            or state[0] > self.x_threshold
            or state[2] < -self.theta_threshold_radians
            or state[2] > self.theta_threshold_radians
        )

        # Assign reward
        if not terminated:
            reward = 1.0
        else:
            reward = 0.0

        # The 'info' dictionary is not used for now
        info = {}

        return state, reward, terminated, info

    def reset(self):
        """
        Resets the environment to a new initial state.

        :return: The initial state of the environment.
        """
        # `reset_mode=1` initializes the cart at a random state.
        self.env.set_cartpole_state_at_t0(reset_mode=1)
        return self.env.s

# This file will contain a wrapper for the existing CartPole environment (and others in the future).
# This wrapper will implement the `BaseWorldModel` interface, allowing the agent to interact
# with the real environment as if it were a learned world model.

import numpy as np
import math
import sys
import os

from .base_world_model import BaseWorldModel


class INICartPoleWrapper(BaseWorldModel):
    """
    A wrapper for the existing INI CartPole environment to make it compatible with the BaseWorldModel interface.
    This allows the RL agent to interact with the "real" environment in the same way it would
    interact with a learned world model.
    """

    def __init__(self, **kwargs):
        """
        Initializes the INICartPoleWrapper.

        :param kwargs: Arguments to be passed to the underlying CartPole environment.
        """
        # This is a hacky way to ensure the imports from the submodule work,
        # and that the submodule can find its own config files.
        submodule_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "environments", "CartPoleSimulation"
            )
        )
        si_toolkit_src_path = os.path.join(submodule_root, "SI_Toolkit", "src")

        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)
        if si_toolkit_src_path not in sys.path:
            sys.path.insert(0, si_toolkit_src_path)

        from CartPole import CartPole as RealCartPole
        from CartPole.state_utilities import ANGLE_COS_IDX, POSITION_IDX

        self.POSITION_IDX = POSITION_IDX
        self.ANGLE_COS_IDX = ANGLE_COS_IDX

        # We need to change the CWD so that the environment can find its config files.
        # This is not ideal, but necessary given the structure of the submodule.
        original_cwd = os.getcwd()
        os.chdir(submodule_root)

        try:
            # Initialize the "real" CartPole environment
            self.env = RealCartPole()
        finally:
            os.chdir(original_cwd)

        # Define thresholds for termination
        self.x_threshold = 2.4

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

        # For a swingup-and-balance task, termination should only occur if the cart
        # goes off the track. The pole falling over is part of the task to be learned.
        terminated = bool(
            state[self.POSITION_IDX] < -self.x_threshold
            or state[self.POSITION_IDX] > self.x_threshold
        )

        # Assign reward
        # The reward is based on the pole's angle. We use the cosine of the angle,
        # which is at index ANGLE_COS_IDX (==0), to reward the agent for keeping the pole upright.
        # The reward is scaled to be in the [0, 1] range.
        reward = (state[self.ANGLE_COS_IDX] + 1.0) / 2.0

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

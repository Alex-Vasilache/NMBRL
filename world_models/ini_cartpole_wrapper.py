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

    def __init__(
        self, max_steps=1000, target_position=0.0, target_equilibrium=1.0, **kwargs
    ):
        """
        Initializes the INICartPoleWrapper.

        :param max_steps: Maximum number of steps before manual termination (default: 1000)
        :param target_position: Target position for the cart (default: 0.0)
        :param target_equilibrium: Target equilibrium scaling factor (default: 1.0)
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
        from CartPole.state_utilities import ANGLE_COS_IDX, POSITION_IDX, ANGLE_IDX
        from CartPole.cartpole_parameters import TrackHalfLength

        self.POSITION_IDX = POSITION_IDX
        self.ANGLE_COS_IDX = ANGLE_COS_IDX
        self.ANGLE_IDX = ANGLE_IDX
        self.TrackHalfLength = TrackHalfLength

        # We need to change the CWD so that the environment can find its config files.
        # This is not ideal, but necessary given the structure of the submodule.
        original_cwd = os.getcwd()
        os.chdir(submodule_root)

        try:
            # Initialize the "real" CartPole environment
            self.env = RealCartPole()

            # Import and initialize the cost function
            from Control_Toolkit_ASF.Cost_Functions.CartPole.quadratic_boundary import (
                quadratic_boundary,
            )
            from SI_Toolkit.computation_library import NumpyLibrary
            from types import SimpleNamespace

            # Create variable parameters object to hold target values
            variable_parameters = SimpleNamespace()
            variable_parameters.target_position = target_position
            variable_parameters.target_equilibrium = target_equilibrium

            # Initialize the cost function with required parameters
            self.cost_function = quadratic_boundary(variable_parameters, NumpyLibrary())
            self.max_cost = self.cost_function.MAX_COST

        finally:
            os.chdir(original_cwd)

        # Manual termination parameters
        self.max_steps = max_steps
        self.step_count = 0

        # Store target parameters for access
        self.target_position = target_position
        self.target_equilibrium = target_equilibrium

        # Previous control input for jerk penalty
        self.previous_action = 0.0

        # Set a default simulation timestep if not provided
        self.env.dt_simulation = kwargs.get("dt_simulation", 0.02)

        self.reset()

    def _compute_reward_from_cost(self, state, action):
        """
        Convert the quadratic boundary cost to a reward using the original cost function.
        Lower cost = higher reward.
        """
        # Reshape state and action for the cost function (expects batch dimensions)
        state_batch = np.expand_dims(state, axis=0)  # Shape: (1, state_dim)
        state_batch = np.expand_dims(state_batch, axis=0)  # Shape: (1, 1, state_dim)

        action_batch = np.array([[[action]]])  # Shape: (1, 1, 1)

        # Compute the stage cost using the original cost function
        stage_cost = self.cost_function._get_stage_cost(
            state_batch, action_batch, self.previous_action
        )

        # Extract scalar cost value
        total_cost = float(stage_cost[0, 0])

        # Convert cost to reward (negative cost); normalize the cost to be between 0 and 1
        reward = -total_cost / self.max_cost

        # For debugging, compute individual components with correct weights
        try:
            # Import the weight constants from the module
            from Control_Toolkit_ASF.Cost_Functions.CartPole.quadratic_boundary import (
                dd_weight,
                ep_weight,
                cc_weight,
                ccrc_weight,
                R,
            )

            # Get raw cost components (unweighted)
            raw_dd_cost = float(
                self.cost_function._distance_difference_cost(state[self.POSITION_IDX])
            )
            raw_ep_cost = float(self.cost_function._E_pot_cost(state[self.ANGLE_IDX]))
            raw_cc_cost = float(self.cost_function._CC_cost(action_batch)[0, 0])

            # Control change rate cost (if we have previous action)
            raw_ccrc_cost = 0.0
            if self.previous_action is not None:
                raw_ccrc_cost = float(
                    self.cost_function._control_change_rate_cost(
                        action_batch, self.previous_action
                    )[0, 0]
                )

            # Apply the weights from the config file to get weighted costs
            weighted_dd_cost = dd_weight * raw_dd_cost
            weighted_ep_cost = ep_weight * raw_ep_cost
            weighted_cc_cost = cc_weight * raw_cc_cost
            weighted_ccrc_cost = ccrc_weight * raw_ccrc_cost

        except Exception as e:
            # Fallback if individual cost computation fails
            weighted_dd_cost = weighted_ep_cost = weighted_cc_cost = (
                weighted_ccrc_cost
            ) = 0.0

        return reward, {
            "distance_cost": weighted_dd_cost,
            "angle_cost": weighted_ep_cost,
            "control_cost": weighted_cc_cost,
            "jerk_cost": weighted_ccrc_cost,
            "total_cost": total_cost,
        }

    def step(self, action: np.ndarray):
        """
        Takes a single step in the environment.

        :param action: A numpy array representing the action to take, corresponding to the dimensionless motor power Q.
        :return: A tuple containing (next_state, reward, terminated, info).
        """
        # Set the action (motor power)
        action_value = float(action[0])
        self.env.Q = action_value

        # Update the environment state
        self.env.update_state()

        # Get the new state
        state = self.env.s_with_noise_and_latency

        # Compute reward using the original quadratic boundary cost function
        reward, cost_info = self._compute_reward_from_cost(state, action_value)

        # Manual termination after max_steps (no out-of-bounds termination)
        self.step_count += 1
        terminated = self.step_count >= self.max_steps

        # Update previous action for next step's jerk calculation
        self.previous_action = action_value

        # Info dictionary with cost breakdown
        info = {"step_count": self.step_count, "max_steps": self.max_steps, **cost_info}

        return state, reward, terminated, info

    def reset(self):
        """
        Resets the environment to a new initial state.

        :return: The initial state of the environment.
        """
        # `reset_mode=1` initializes the cart at a random state.
        self.env.set_cartpole_state_at_t0(reset_mode=1)

        # Reset step counter and previous action
        self.step_count = 0
        self.previous_action = 0.0

        return self.env.s

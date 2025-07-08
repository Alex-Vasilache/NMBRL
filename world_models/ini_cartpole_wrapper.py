# This file will contain a wrapper for the existing CartPole environment (and others in the future).
# This wrapper will implement the `BaseWorldModel` interface, allowing the agent to interact
# with the real environment as if it were a learned world model.

import numpy as np
import math
import sys
import os
import time

from .base_world_model import BaseWorldModel


class INICartPoleWrapper(BaseWorldModel):
    """
    A wrapper for the existing INI CartPole environment to make it compatible with the BaseWorldModel interface.
    This allows the RL agent to interact with the "real" environment in the same way it would
    interact with a learned world model. Supports batching for parallel environment execution.
    """

    def __init__(
        self,
        max_steps=1000,
        target_position=0.0,
        target_equilibrium=1.0,
        dt_simulation=0.02,
        visualize=False,
        **kwargs,
    ):
        """
        Initializes the INICartPoleWrapper.

        :param max_steps: Maximum number of steps before manual termination (default: 1000)
        :param target_position: Target position for the cart (default: 0.0)
        :param target_equilibrium: Target equilibrium scaling factor (default: 1.0)
        :param dt_simulation: Simulation timestep (default: 0.02)
        :param visualize: If True, a window with the CartPole visualization will be opened (default: False)
        :param kwargs: Arguments to be passed to the underlying CartPole environment.
        """
        self.batch_size = 0  # Will be set by reset()
        self.dt_simulation = dt_simulation

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
        from CartPole.cartpole_target_slider import TargetSlider
        from CartPole.state_utilities import (
            ANGLE_COS_IDX,
            POSITION_IDX,
            ANGLE_IDX,
            ANGLED_IDX,
            POSITIOND_IDX,
            ANGLE_SIN_IDX,
        )
        from CartPole.cartpole_parameters import TrackHalfLength, v_max

        self.POSITION_IDX = POSITION_IDX
        self.ANGLE_COS_IDX = ANGLE_COS_IDX
        self.ANGLE_IDX = ANGLE_IDX
        self.ANGLED_IDX = ANGLED_IDX
        self.POSITIOND_IDX = POSITIOND_IDX
        self.ANGLE_SIN_IDX = ANGLE_SIN_IDX
        self.TrackHalfLength = TrackHalfLength
        self.v_max = v_max

        # We need to change the CWD so that the environment can find its config files.
        # This is not ideal, but necessary given the structure of the submodule.
        original_cwd = os.getcwd()
        os.chdir(submodule_root)

        try:
            # Store classes for later environment creation
            self.RealCartPole = RealCartPole
            self.TargetSlider = TargetSlider

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

        # Store target parameters for access
        self.target_position = target_position
        self.target_equilibrium = target_equilibrium

        # This is an assumed limit for angular velocity, used for normalization
        self.max_angular_velocity = 15.0  # rad/s

        self.visualize = visualize

        # Initialize empty containers that will be filled by reset()
        self.envs = []
        self.target_sliders = []
        self.step_counts = np.array([])
        self.previous_actions = np.array([])

        # Initialize with default batch size of 1
        self.reset(batch_size=1)

    def _init_visualization(self):
        """Initializes the visualization elements for the first environment only."""
        import matplotlib.pyplot as plt
        from CartPole.cartpole_drawer import CartPoleDrawer

        # Only visualize the first environment to avoid multiple windows
        self.fig, self.axes = plt.subplots(2, 1, figsize=(16, 10))
        self.drawer = CartPoleDrawer(self.envs[0], self.target_sliders[0])
        self.drawer.draw_constant_elements(self.fig, self.axes[0], self.axes[1])

        # Manually add the patches for the cart, which is normally done in the animation init
        self.axes[0].add_patch(self.drawer.Mast)
        self.drawer.Mast.set_transform(self.drawer.t2 + self.axes[0].transData)
        self.axes[0].add_patch(self.drawer.ZeroAngleTick)
        self.drawer.ZeroAngleTick.set_transform(
            self.drawer.t_zero_angle + self.axes[0].transData
        )
        self.axes[0].add_patch(self.drawer.Chassis)
        self.axes[0].add_patch(self.drawer.WheelLeft)
        self.axes[0].add_patch(self.drawer.WheelRight)
        self.axes[0].add_patch(self.drawer.Acceleration_Arrow)

        self.fig.show()

    def _render(self):
        """Renders the current state of the first environment only."""
        if self.visualize:
            self.drawer.update_drawing()
            self.drawer.Mast.set_transform(self.drawer.t2 + self.axes[0].transData)
            self.drawer.ZeroAngleTick.set_transform(
                self.drawer.t_zero_angle + self.axes[0].transData
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _compute_reward_from_cost(self, states, actions):
        """
        Convert the quadratic boundary cost to a reward using the original cost function.
        Lower cost = higher reward.

        :param states: Batch of states with shape (batch_size, state_dim)
        :param actions: Batch of actions with shape (batch_size,)
        :return: Tuple of (rewards, cost_info_dict)
        """
        # Reshape for cost function (expects shape: (1, batch_size, state_dim))
        state_batch = np.expand_dims(
            states, axis=0
        )  # Shape: (1, batch_size, state_dim)

        # Reshape actions (expects shape: (1, batch_size, 1))
        action_batch = np.expand_dims(actions, axis=-1)  # Shape: (batch_size, 1)
        action_batch = np.expand_dims(action_batch, axis=0)  # Shape: (1, batch_size, 1)

        # Compute the stage cost using the original cost function
        stage_costs = self.cost_function._get_stage_cost(
            state_batch,
            action_batch,
            self.previous_actions[0],  # Use first env's previous action for now
        )

        # Extract costs and normalize
        total_costs = stage_costs[0, :] / self.max_cost  # Shape: (batch_size,)
        total_costs = 1 / (1 + np.exp(-total_costs))

        # Convert cost to reward (negative cost)
        rewards = -total_costs

        # Compute individual cost components for debugging
        cost_info = {}
        try:
            # Import the weight constants from the module
            from Control_Toolkit_ASF.Cost_Functions.CartPole.quadratic_boundary import (
                dd_weight,
                ep_weight,
                cc_weight,
                ccrc_weight,
                R,
            )

            # Get raw cost components for all environments
            positions = states[:, self.POSITION_IDX]
            angles = states[:, self.ANGLE_IDX]

            # Distance difference costs
            raw_dd_costs = np.array(
                [
                    float(self.cost_function._distance_difference_cost(pos))
                    for pos in positions
                ]
            )

            # Potential energy costs
            raw_ep_costs = np.array(
                [float(self.cost_function._E_pot_cost(angle)) for angle in angles]
            )

            # Control costs
            raw_cc_costs = self.cost_function._CC_cost(action_batch)[0, :]

            # Control change rate costs
            raw_ccrc_costs = np.zeros(self.batch_size)
            for i in range(self.batch_size):
                if self.previous_actions[i] is not None:
                    action_single = np.array([[[actions[i]]]])
                    raw_ccrc_costs[i] = float(
                        self.cost_function._control_change_rate_cost(
                            action_single, self.previous_actions[i]
                        )[0, 0]
                    )

            # Apply weights
            cost_info = {
                "distance_cost": dd_weight * raw_dd_costs,
                "angle_cost": ep_weight * raw_ep_costs,
                "control_cost": cc_weight * raw_cc_costs,
                "jerk_cost": ccrc_weight * raw_ccrc_costs,
                "total_cost": total_costs,
            }

        except Exception as e:
            # Fallback if individual cost computation fails
            cost_info = {
                "distance_cost": np.zeros(self.batch_size),
                "angle_cost": np.zeros(self.batch_size),
                "control_cost": np.zeros(self.batch_size),
                "jerk_cost": np.zeros(self.batch_size),
                "total_cost": total_costs,
            }

        return rewards, cost_info

    def _normalize_state(self, states):
        """
        Normalizes the state variables to be within the [0, 1] range.

        :param states: States to normalize, can be single state or batch of states
        :return: Normalized states with same shape as input
        """
        # Handle both single states and batches
        if states.ndim == 1:
            states = states.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        normalized_states = np.zeros_like(states)

        # Normalize position: [-TrackHalfLength, TrackHalfLength] -> [0, 1]
        pos_clamped = np.clip(
            states[:, self.POSITION_IDX], -self.TrackHalfLength, self.TrackHalfLength
        )
        normalized_states[:, self.POSITION_IDX] = (
            pos_clamped + self.TrackHalfLength
        ) / (2 * self.TrackHalfLength)

        # Normalize positionD: [-v_max, v_max] -> [0, 1]
        posD_clamped = np.clip(states[:, self.POSITIOND_IDX], -self.v_max, self.v_max)
        normalized_states[:, self.POSITIOND_IDX] = (posD_clamped + self.v_max) / (
            2 * self.v_max
        )

        # Normalize angle: [-pi, pi] -> [0, 1]
        normalized_states[:, self.ANGLE_IDX] = (states[:, self.ANGLE_IDX] + np.pi) / (
            2 * np.pi
        )

        # Normalize angleD: [-max_angular_velocity, max_angular_velocity] -> [0, 1]
        angleD_clamped = np.clip(
            states[:, self.ANGLED_IDX],
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )
        normalized_states[:, self.ANGLED_IDX] = (
            angleD_clamped + self.max_angular_velocity
        ) / (2 * self.max_angular_velocity)

        # Normalize angle_cos and angle_sin: [-1, 1] -> [0, 1]
        normalized_states[:, self.ANGLE_COS_IDX] = (
            states[:, self.ANGLE_COS_IDX] + 1
        ) / 2
        normalized_states[:, self.ANGLE_SIN_IDX] = (
            states[:, self.ANGLE_SIN_IDX] + 1
        ) / 2

        if squeeze_output:
            return normalized_states.squeeze(0)
        return normalized_states

    def _create_environments(self, batch_size):
        """Create or resize environments to match the target batch size."""
        current_size = len(self.envs)

        # We need to change the CWD for environment creation
        submodule_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "environments", "CartPoleSimulation"
            )
        )
        original_cwd = os.getcwd()

        try:
            os.chdir(submodule_root)

            if batch_size > current_size:
                # Create additional environments sequentially to avoid global variable conflicts
                for i in range(current_size, batch_size):
                    env = self.RealCartPole()
                    target_slider = self.TargetSlider()
                    env.target_slider = target_slider
                    env.dt_simulation = self.dt_simulation

                    # Initialize with a safe reset to ensure parameters are set correctly
                    env.set_cartpole_state_at_t0(
                        reset_mode=0
                    )  # Use mode 0 for safe initialization

                    self.envs.append(env)
                    self.target_sliders.append(target_slider)

            elif batch_size < current_size:
                # Remove excess environments
                self.envs = self.envs[:batch_size]
                self.target_sliders = self.target_sliders[:batch_size]

        finally:
            os.chdir(original_cwd)

        # Update batch size and related arrays
        self.batch_size = batch_size
        self.step_counts = np.zeros(self.batch_size, dtype=int)
        self.previous_actions = np.zeros(self.batch_size)

    def _ensure_global_parameters(self):
        """Ensure global CartPole parameters are properly set to prevent division by zero."""
        try:
            # Import the global variables and default parameters
            from CartPole.cartpole_parameters import (
                k,
                m_cart,
                m_pole,
                g,
                J_fric,
                M_fric,
                L,
                v_max,
                u_max,
                controlNoiseScale,
                controlNoiseBias,
                controlNoiseCorrelation,
                TrackHalfLength,
                controlNoise_mode,
                CP_PARAMETERS_DEFAULT,
            )

            # Check if any critical parameters are zero or invalid
            critical_params = [
                k[0] if k.ndim > 0 else k,
                m_cart[0] if m_cart.ndim > 0 else m_cart,
                m_pole[0] if m_pole.ndim > 0 else m_pole,
                g[0] if g.ndim > 0 else g,
                L[0] if L.ndim > 0 else L,
            ]
            param_names = ["k", "m_cart", "m_pole", "g", "L"]

            invalid_params = []
            for param, name in zip(critical_params, param_names):
                if param == 0 or not np.isfinite(param):
                    invalid_params.append(f"{name}={param}")

            if invalid_params:
                print(
                    f"Warning: Invalid CartPole parameters detected: {invalid_params}"
                )
                print("Restoring parameters from defaults...")
                # Restore parameters from defaults
                (
                    k[...],
                    m_cart[...],
                    m_pole[...],
                    g[...],
                    J_fric[...],
                    M_fric[...],
                    L[...],
                    v_max[...],
                    u_max[...],
                    controlNoiseScale[...],
                    controlNoiseBias[...],
                    controlNoiseCorrelation[...],
                    TrackHalfLength[...],
                    controlNoise_mode,
                ) = CP_PARAMETERS_DEFAULT.export_parameters()

                # Verify restoration was successful
                restored_params = [
                    k[0] if k.ndim > 0 else k,
                    m_cart[0] if m_cart.ndim > 0 else m_cart,
                    m_pole[0] if m_pole.ndim > 0 else m_pole,
                    g[0] if g.ndim > 0 else g,
                    L[0] if L.ndim > 0 else L,
                ]
                print(
                    f"Restored parameters: k={restored_params[0]}, m_cart={restored_params[1]}, m_pole={restored_params[2]}, g={restored_params[3]}, L={restored_params[4]}"
                )

        except Exception as e:
            print(f"Error ensuring global parameters: {e}")
            # If parameter checking fails, at least try to restore defaults
            try:
                from CartPole.cartpole_parameters import CP_PARAMETERS_DEFAULT
                from CartPole.cartpole_parameters import (
                    k,
                    m_cart,
                    m_pole,
                    g,
                    J_fric,
                    M_fric,
                    L,
                    v_max,
                    u_max,
                    controlNoiseScale,
                    controlNoiseBias,
                    controlNoiseCorrelation,
                    TrackHalfLength,
                    controlNoise_mode,
                )

                (
                    k[...],
                    m_cart[...],
                    m_pole[...],
                    g[...],
                    J_fric[...],
                    M_fric[...],
                    L[...],
                    v_max[...],
                    u_max[...],
                    controlNoiseScale[...],
                    controlNoiseBias[...],
                    controlNoiseCorrelation[...],
                    TrackHalfLength[...],
                    controlNoise_mode,
                ) = CP_PARAMETERS_DEFAULT.export_parameters()
            except:
                pass

    def step(self, actions: np.ndarray):
        """
        Takes a single step in all environments.

        :param actions: A numpy array of actions with shape (batch_size,) or (batch_size, 1),
                       representing the dimensionless motor power Q for each environment.
        :return: A tuple containing (next_states, rewards, terminated, info).
                 next_states: shape (batch_size, state_dim)
                 rewards: shape (batch_size,)
                 terminated: shape (batch_size,) - boolean array
                 info: dict with arrays of shape (batch_size,)
        """
        # Ensure actions have the right shape
        actions = np.asarray(actions)
        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        elif actions.ndim == 1:
            pass  # Already correct shape
        else:
            raise ValueError(
                f"Actions must have shape (batch_size,) or (batch_size, 1), got {actions.shape}"
            )

        if len(actions) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} actions, got {len(actions)}")

        # Change to CartPole directory to ensure parameters are properly accessible
        submodule_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "environments", "CartPoleSimulation"
            )
        )
        original_cwd = os.getcwd()

        try:
            os.chdir(submodule_root)

            # Step each environment
            states = []
            for i, (env, action_value) in enumerate(zip(self.envs, actions)):
                # Ensure global parameters are valid before stepping
                self._ensure_global_parameters()

                # Set the action (motor power)
                env.Q = float(action_value)

                # Update the environment state
                env.update_state()

                # Get the new state
                states.append(env.s_with_noise_and_latency)

        finally:
            os.chdir(original_cwd)

        states = np.array(states)  # Shape: (batch_size, state_dim)

        # Render the first environment if visualization is enabled
        if self.visualize:
            self._render()

        # Compute rewards using the original quadratic boundary cost function
        rewards, cost_info = self._compute_reward_from_cost(states, actions)

        # Manual termination after max_steps (no out-of-bounds termination)
        self.step_counts += 1
        terminated = self.step_counts >= self.max_steps

        # Update previous actions for next step's jerk calculation
        self.previous_actions = actions.copy()

        # Info dictionary with cost breakdown and step information
        info = {
            "step_count": self.step_counts.copy(),
            "max_steps": np.full(self.batch_size, self.max_steps),
            **cost_info,
        }

        return self._normalize_state(states), rewards, terminated, info

    def reset(self, batch_size=None, initial_state=None):
        """
        Resets environments to new initial states. Optionally changes the batch size
        and/or sets specific initial states.

        :param batch_size: Number of environments to use. If None, keeps current batch size.
        :param initial_state: Initial states for environments with shape (batch_size, state_dim).
                             If None, uses random initialization.
        :return: The initial states of all environments with shape (batch_size, state_dim).
        """
        # Handle initial_state parameter
        if initial_state is not None:
            initial_state = np.asarray(initial_state)

            # Validate initial_state shape
            if initial_state.ndim == 1:
                # Single state provided, reshape to (1, state_dim)
                initial_state = initial_state.reshape(1, -1)
            elif initial_state.ndim != 2:
                raise ValueError(
                    f"initial_state must have shape (batch_size, state_dim), got shape {initial_state.shape}"
                )

            # Infer batch_size from initial_state if not provided
            if batch_size is None:
                batch_size = initial_state.shape[0]
            elif batch_size != initial_state.shape[0]:
                raise ValueError(
                    f"batch_size ({batch_size}) must match initial_state.shape[0] ({initial_state.shape[0]})"
                )

        # Create/resize environments if batch_size is specified
        if batch_size is not None:
            self._create_environments(batch_size)

        # Change to the CartPole directory for proper parameter loading
        submodule_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "environments", "CartPoleSimulation"
            )
        )
        original_cwd = os.getcwd()

        try:
            os.chdir(submodule_root)

            states = []
            # Reset environments sequentially to avoid global variable conflicts
            for i, env in enumerate(self.envs):
                if initial_state is not None:
                    # Set specific initial state
                    # Convert normalized state back to raw state for the environment
                    raw_state = self._denormalize_state(initial_state[i])
                    env.set_cartpole_state_at_t0(
                        reset_mode=2, s=raw_state, target_position=self.target_position
                    )
                    states.append(env.s)
                else:
                    # Random initialization - ensure each environment gets different random state
                    # Modify the environment's random state slightly to ensure diversity
                    if hasattr(env, "rng_CartPole"):
                        # Add some randomness to each environment's RNG state
                        seed_offset = int((time.time() * 1000000) % 1000000) + i * 1000
                        # Use the proper NumPy random generator API
                        env.rng_CartPole = np.random.default_rng(seed_offset)

                    # Reset to ensure parameters are loaded correctly first
                    env.set_cartpole_state_at_t0(reset_mode=0)  # Safe reset first
                    env.set_cartpole_state_at_t0(
                        reset_mode=1
                    )  # Then random initialization

                    # Add additional randomization to ensure truly different states
                    # Randomly vary position and velocities slightly
                    env.s[self.POSITION_IDX] += np.random.uniform(
                        -0.1, 0.1
                    )  # Small position variation
                    env.s[self.POSITIOND_IDX] += np.random.uniform(
                        -0.1, 0.1
                    )  # Small velocity variation
                    env.s[self.ANGLE_IDX] += np.random.uniform(
                        -0.05, 0.05
                    )  # Additional angle variation
                    env.s[self.ANGLED_IDX] += np.random.uniform(
                        -0.05, 0.05
                    )  # Additional angular velocity variation

                    # Update cos and sin based on modified angle
                    env.s[self.ANGLE_COS_IDX] = np.cos(env.s[self.ANGLE_IDX])
                    env.s[self.ANGLE_SIN_IDX] = np.sin(env.s[self.ANGLE_IDX])

                    # Clamp values to reasonable ranges
                    env.s[self.POSITION_IDX] = np.clip(
                        env.s[self.POSITION_IDX],
                        -self.TrackHalfLength,
                        self.TrackHalfLength,
                    )
                    env.s[self.POSITIOND_IDX] = np.clip(
                        env.s[self.POSITIOND_IDX], -self.v_max, self.v_max
                    )
                    env.s[self.ANGLE_IDX] = np.clip(
                        env.s[self.ANGLE_IDX], -np.pi, np.pi
                    )
                    env.s[self.ANGLED_IDX] = np.clip(
                        env.s[self.ANGLED_IDX],
                        -self.max_angular_velocity,
                        self.max_angular_velocity,
                    )

                    states.append(env.s.copy())

        finally:
            os.chdir(original_cwd)

        # Reset step counters and previous actions
        self.step_counts = np.zeros(self.batch_size, dtype=int)
        self.previous_actions = np.zeros(self.batch_size)

        # Initialize visualization for the first environment if needed
        if self.visualize and not hasattr(self, "fig"):
            self._init_visualization()

        # Render the reset state of the first environment
        if self.visualize:
            self._render()

        states = np.array(states)  # Shape: (batch_size, state_dim)
        return self._normalize_state(states)

    def _denormalize_state(self, normalized_state):
        """
        Converts normalized state back to raw state values for the environment.

        :param normalized_state: Normalized state with values in [0, 1] range
        :return: Raw state values in original ranges
        """
        raw_state = np.zeros_like(normalized_state)

        # Denormalize position: [0, 1] -> [-TrackHalfLength, TrackHalfLength]
        raw_state[self.POSITION_IDX] = (
            normalized_state[self.POSITION_IDX] * 2 * self.TrackHalfLength
        ) - self.TrackHalfLength

        # Denormalize positionD: [0, 1] -> [-v_max, v_max]
        raw_state[self.POSITIOND_IDX] = (
            normalized_state[self.POSITIOND_IDX] * 2 * self.v_max
        ) - self.v_max

        # Denormalize angle: [0, 1] -> [-pi, pi]
        raw_state[self.ANGLE_IDX] = (
            normalized_state[self.ANGLE_IDX] * 2 * np.pi
        ) - np.pi

        # Denormalize angleD: [0, 1] -> [-max_angular_velocity, max_angular_velocity]
        raw_state[self.ANGLED_IDX] = (
            normalized_state[self.ANGLED_IDX] * 2 * self.max_angular_velocity
        ) - self.max_angular_velocity

        # Denormalize angle_cos and angle_sin: [0, 1] -> [-1, 1]
        raw_state[self.ANGLE_COS_IDX] = (normalized_state[self.ANGLE_COS_IDX] * 2) - 1
        raw_state[self.ANGLE_SIN_IDX] = (normalized_state[self.ANGLE_SIN_IDX] * 2) - 1

        return raw_state

    def close(self):
        """Closes the visualization window."""
        if self.visualize:
            import matplotlib.pyplot as plt

            plt.close(self.fig)

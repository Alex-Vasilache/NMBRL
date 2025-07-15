"""
Physical CartPole Wrapper for Neuromorphic MBRL

This module provides a wrapper for the physical cartpole environment that uses the same
interface as the DMC cartpole wrapper, allowing it to be used as a drop-in replacement.

Key Features:
- Same interface as DMCCartpoleWrapper (inherits from VecNormalize)
- Uses physical cartpole hardware via CartPoleEnv with cartpole_type="remote"
- Implements DMC-style reward function from dm_control cartpole swingup task
- Supports vectorized environments and observation normalization

DMC Reward Function Implementation:
The reward function is based on the dm_control cartpole swingup task and computes:

reward = upright * small_control * small_velocity * centered

Where:
- upright = (cos_angle + 1) / 2  # Reward for being upright (0 to 1)
- centered = tolerance(cart_pos, margin=2)  # Reward for being centered
- small_control = tolerance(action, margin=1, sigmoid='quadratic')  # Penalty for large control
- small_velocity = tolerance(angle_vel, margin=5)  # Penalty for high angular velocity

The tolerance function implements reward shaping similar to dm_control's rewards.tolerance
with support for gaussian, linear, and quadratic sigmoid functions.

Usage:
    # Replace this line in dynamic_data_generator.py:
    # from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
    
    # With this line:
    from world_models.physical_cartpole_wrapper import PhysicalCartpoleWrapper as wrapper
    
    # No other changes needed - the interface is identical!
"""

from typing import Optional, Dict, Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add path to access the physical cartpole environment
import sys
import os

MAX_ACTION_CHANGE = 0.4
MAX_ACTION_SCALE = 0.7

_high = np.array(
    [
        np.pi,  # θ
        np.inf,  # θ̇
        1.0,
        1.0,  # sin θ, cos θ
        0.2,  # x
        np.inf,  # ẋ
    ],
    dtype=np.float32,
)
ACTION_SPACE = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
OBSERVATION_SPACE = spaces.Box(-_high, _high, dtype=np.float32)

# Try to import CartPoleEnv, handling case where physical cartpole might not be available
try:
    physical_cartpole_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "environments",
        "physical-cartpole",
        "Driver",
        "CartPoleSimulation",
    )
    if physical_cartpole_path not in sys.path:
        sys.path.append(physical_cartpole_path)

    from GymlikeCartPole.EnvGym.CartpoleEnv import CartPoleEnv
except ImportError as e:
    raise ImportError(
        "Could not import physical cartpole environment. "
        "Please ensure the physical-cartpole environment is available. "
        f"Original error: {e}"
    )


class DMCRewardTask:
    """
    Custom task that implements the DMC cartpole swingup reward function.
    Based on the dm_control cartpole implementation provided by the user.
    """

    def __init__(self, physics):
        self.physics = physics
        self._CART_RANGE = (-0.2, 0.2)
        self._ANGLE_COSINE_RANGE = (0.995, 1)
        self.sparse = False  # Use smooth reward by default

    def _tolerance(
        self, x, bounds=None, margin=0.0, sigmoid="gaussian", value_at_margin=0.1
    ):
        """Tolerance function similar to dm_control rewards.tolerance"""
        if bounds is not None:
            lower, upper = bounds
            # Check if x is within bounds
            if lower <= x <= upper:
                return 1.0
            else:
                # Calculate distance from bounds
                if x < lower:
                    distance = lower - x
                else:
                    distance = x - upper

                if sigmoid == "linear":
                    return max(0.0, 1.0 - distance / margin) if margin > 0 else 0.0
                else:  # gaussian or quadratic
                    if margin > 0:
                        return value_at_margin * np.exp(-0.5 * (distance / margin) ** 2)
                    else:
                        return 0.0
        else:
            # Single value tolerance (for centered values)
            if margin <= 0:
                return 1.0 if abs(x) == 0 else 0.0

            if sigmoid == "linear":
                return max(0.0, 1.0 - abs(x) / margin)
            elif sigmoid == "quadratic":
                normalized_distance = abs(x) / margin
                if normalized_distance <= 1.0:
                    return value_at_margin + (1.0 - value_at_margin) * (
                        1.0 - normalized_distance**2
                    )
                else:
                    return value_at_margin * np.exp(
                        -0.5 * (normalized_distance - 1.0) ** 2
                    )
            else:  # gaussian
                return np.exp(-0.5 * (x / margin) ** 2)

    def get_reward(self, state, action):
        """
        Compute DMC-style cartpole swingup reward.

        State format: [angle, angle_vel, cos_angle, sin_angle, cart_pos, cart_vel]
        """
        # Extract state components
        angle = state[0]
        angle_vel = state[1]
        cos_angle = state[2]
        sin_angle = state[3]
        cart_pos = state[4]
        cart_vel = state[5]

        if self.sparse:
            # Sparse reward version
            cart_in_bounds = self._tolerance(cart_pos, bounds=self._CART_RANGE)
            angle_in_bounds = self._tolerance(
                cos_angle, bounds=self._ANGLE_COSINE_RANGE
            )
            return cart_in_bounds * angle_in_bounds
        else:
            # Smooth reward version (matches DMC implementation)
            upright = (cos_angle + 1) / 2
            centered = self._tolerance(cart_pos, margin=2)
            centered = (1 + centered) / 2

            # Control penalty - action is typically a single value
            action_val = action[0] if hasattr(action, "__len__") else action
            small_control = self._tolerance(
                action_val, margin=1, value_at_margin=0, sigmoid="quadratic"
            )
            small_control = (4 + small_control) / 5

            # Angular velocity penalty
            small_velocity = self._tolerance(angle_vel, margin=5)
            small_velocity = (1 + small_velocity) / 2

            return upright * small_control * small_velocity * centered


class PhysicalCartPoleWrapper(gym.Env):
    """
    Wrapper to adapt the physical CartPoleEnv to work with the same interface as DMCWrapper.
    Uses the DMC reward function instead of the original task reward.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps=1000,
    ):
        # Create the base physical cartpole environment
        self.env = CartPoleEnv(
            render_mode=render_mode,
            task="swingup",  # Use swingup task as base
            cartpole_type="remote",  # Use remote for physical cartpole
        )

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Use the same action and observation spaces as the underlying env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.previous_action = [0.0]

        # Create DMC-style reward task
        # Create a mock physics object for the reward function
        class MockPhysics:
            def __init__(self, env):
                self.x_limit = getattr(
                    env.cartpole_rl, "x_limit", 0.2
                )  # Default track limit

        self.dmc_task = DMCRewardTask(MockPhysics(self.env))

    def step(self, action):
        # Step the underlying environment

        current_position = self.env.state[4]

        if current_position > self.env.cartpole_rl.x_limit * 0.7 and action[0] > 0:
            action = [-0.1]
        elif current_position < -self.env.cartpole_rl.x_limit * 0.7 and action[0] < 0:
            action = [0.1]

        if current_position > self.env.cartpole_rl.x_limit * 0.8 and action[0] > 0:
            action = [0.2]
        elif current_position < -self.env.cartpole_rl.x_limit * 0.8 and action[0] < 0:
            action = [-0.2]

        if current_position > self.env.cartpole_rl.x_limit * 0.9 and action[0] > 0:
            action = [0.3]
        elif current_position < -self.env.cartpole_rl.x_limit * 0.9 and action[0] < 0:
            action = [-0.3]

        change = np.clip(
            action - self.previous_action, -MAX_ACTION_CHANGE, MAX_ACTION_CHANGE
        )

        # Apply change to previous action
        new_action = self.previous_action + change
        # Clamp to action space bounds
        action = np.clip(
            new_action,
            self.env.action_space.low * MAX_ACTION_SCALE,
            self.env.action_space.high * MAX_ACTION_SCALE,
        )
        action = action.reshape(1)

        # Ensure correct dtype to match action space
        action = action.astype(self.action_space.dtype)

        self.previous_action = action.copy()

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Replace the reward with DMC-style reward
        dmc_reward = self.dmc_task.get_reward(obs, action)

        # Add info about stopping due to window closure
        if hasattr(self.env, "cartpole_rl") and hasattr(
            self.env.cartpole_rl, "should_stop"
        ):
            if self.env.cartpole_rl.should_stop:
                info["sim_should_stop"] = True

        return obs, dmc_reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        current_position = self.env.state[4]
        while current_position > 0:
            self.env.step([-0.1])
            current_position = self.env.state[4]

        while current_position < 0:
            self.env.step([0.1])
            current_position = self.env.state[4]

        self.prev_action = [0.0]

        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_physical_cartpole_env(
    render_mode: str = None, max_episode_steps: int = 1000
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = PhysicalCartPoleWrapper(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return Monitor(env)

    return _init


class PhysicalCartpoleWrapper(VecNormalize):
    """
    Main wrapper class that matches the interface of DMCCartpoleWrapper.
    This is what should be imported and used as a drop-in replacement.
    """

    def __init__(
        self,
        seed: int = 42,
        n_envs: int = 1,
        render_mode: str = None,
        max_episode_steps: int = 1000,
    ):
        self.n_envs = n_envs
        self._seed = seed

        env_fns = [
            make_physical_cartpole_env(render_mode, max_episode_steps)
            for _ in range(self.n_envs)
        ]

        if n_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

        super().__init__(
            vec_env,
            norm_obs=False,
            norm_reward=False,
            clip_obs=10.0,
        )
        self.seed(self._seed)

    def step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()

    def render(self, mode="human"):
        if self.venv is None:
            return None
        if mode == "human":
            # For human mode, we need to call render on each individual environment
            return self.venv.env_method("render")
        else:
            # For other modes, use the vectorized environment's render method
            return self.venv.render(mode=mode)

# This file will contain a wrapper for the DeepMind Control Suite CartPole environment.
# This wrapper will implement the `BaseWorldModel` interface, allowing the agent to interact
# with the dm_control environment as if it were a learned world model.

import numpy as np
import collections

from .base_world_model import BaseWorldModel

try:
    from dm_control import suite
    from dm_control.rl.control import Environment
    import cv2
except ImportError:
    print(
        "DeepMind Control Suite or OpenCV is not installed. Please install with: pip install dm_control opencv-python"
    )
    suite = None

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    # Fallback to creating simple mock spaces
    GYMNASIUM_AVAILABLE = False
    print("Gymnasium not available, creating mock spaces")

    class MockSpace:
        def __init__(self, shape, low=-np.inf, high=np.inf):
            self.shape = shape
            self.low = low
            self.high = high

    class MockSpaces:
        @staticmethod
        def Box(low, high, shape, dtype=np.float32):
            return MockSpace(shape, low, high)

    spaces = MockSpaces()


class DMCCartPoleWrapper(BaseWorldModel):
    """
    A wrapper for the DeepMind Control Suite CartPole environment to make it compatible
    with the BaseWorldModel interface. Supports batching for parallel environment execution.
    Uses the 'swingup' task.
    """

    def __init__(
        self,
        batch_size=1,
        max_steps=1000,
        visualize=False,
        render_width=640,
        render_height=480,
        dt_simulation=0.02,
        **kwargs,
    ):
        """
        Initializes the DMCCartPoleWrapper.
        :param batch_size: Number of parallel environments.
        :param max_steps: Maximum number of steps per episode.
        :param visualize: If True, renders the environment.
        :param render_width: Width of the rendered image.
        :param render_height: Height of the rendered image.
        :param dt_simulation: The timestep between agent actions (in seconds).
        """
        if suite is None:
            raise ImportError("DeepMind Control Suite is required to use this wrapper.")

        self.max_steps = max_steps
        self.visualize = visualize
        self.render_width = render_width
        self.render_height = render_height
        self.control_timestep = dt_simulation

        # Create a dummy env to get specs
        dummy_env = suite.load(
            domain_name="cartpole",
            task_name="swingup",
            visualize_reward=True,
            task_kwargs={"time_limit": max_steps},
            environment_kwargs={"control_timestep": self.control_timestep},
        )
        self._action_spec = dummy_env.action_spec()
        self._observation_spec = dummy_env.observation_spec()
        dummy_env.close()

        self.state_dim = sum(
            np.prod(spec.shape) for spec in self._observation_spec.values()
        )
        self.action_dim = np.prod(self._action_spec.shape)

        # Create gymnasium-style action and observation spaces
        self._create_gymnasium_spaces()

        self.envs = []
        self.step_counts = np.array([])

        if self.visualize:
            self._init_visualization()

        self.reset(batch_size=batch_size)

    def _create_gymnasium_spaces(self):
        """Create gymnasium-style action and observation spaces from dm_control specs."""
        # Create action space - ensure float32 to avoid precision warnings
        action_low = np.array(self._action_spec.minimum, dtype=np.float32)
        action_high = np.array(self._action_spec.maximum, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=self._action_spec.shape,
            dtype=np.float32,
        )

        # Create observation space
        # For flattened observations, create a single Box space
        obs_low = -np.inf * np.ones(self.state_dim, dtype=np.float32)
        obs_high = np.inf * np.ones(self.state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(self.state_dim,), dtype=np.float32
        )

    def _init_visualization(self):
        """Initializes visualization elements using OpenCV."""
        self.window_name = "dm_control CartPole"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.render_width, self.render_height)

    def _render(self):
        """Renders the current state of the first environment using OpenCV."""
        if self.visualize and self.batch_size > 0:
            frame = self.envs[0].physics.render(
                width=self.render_width, height=self.render_height
            )
            # OpenCV expects BGR, dm_control renders in RGB.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, frame_bgr)
            cv2.waitKey(1)  # Necessary to process GUI events

    def _flatten_observation(self, obs: collections.OrderedDict) -> np.ndarray:
        """Flattens the observation dictionary from dm_control."""
        return np.concatenate([np.atleast_1d(v) for v in obs.values()])

    def _unflatten_state(self, flat_state: np.ndarray) -> collections.OrderedDict:
        """Unflattens a state vector into an observation dictionary."""
        od = collections.OrderedDict()
        start = 0
        for key, spec in self._observation_spec.items():
            num_elements = np.prod(spec.shape).astype(int)
            end = start + num_elements
            od[key] = flat_state[start:end].reshape(spec.shape)
            start = end
        return od

    def _create_environments(self, batch_size):
        """Create or resize environments to match the target batch size."""
        assert suite is not None, "dm_control suite is not imported"
        current_size = len(self.envs)
        if batch_size > current_size:
            for _ in range(current_size, batch_size):
                env = suite.load(
                    domain_name="cartpole",
                    task_name="swingup",
                    visualize_reward=True,
                    task_kwargs={"time_limit": self.max_steps},
                    environment_kwargs={"control_timestep": self.control_timestep},
                )
                # Add gymnasium-style spaces to each environment
                env.action_space = self.action_space
                env.observation_space = self.observation_space
                self.envs.append(env)
        elif batch_size < current_size:
            for i in range(current_size - 1, batch_size - 1, -1):
                self.envs[i].close()
            self.envs = self.envs[:batch_size]

        self.batch_size = batch_size
        self.step_counts = np.zeros(self.batch_size, dtype=int)

    def step(self, actions: np.ndarray):
        """
        Takes a single step in all environments.

        :param actions: A numpy array of actions with shape (batch_size,) or (batch_size, 1).
        :return: A tuple containing (next_states, rewards, terminated, info).
        """
        if actions.shape[0] != self.batch_size:
            raise ValueError("actions batch size must match environment batch size.")

        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        next_states, rewards, terminated = [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            timestep = env.step(action)

            next_states.append(self._flatten_observation(timestep.observation))
            rewards.append(timestep.reward)

            self.step_counts[i] += 1
            is_last_step = timestep.last()
            terminated.append(is_last_step)

            if is_last_step:
                # dm_control automatically resets on termination, but we get the final observation.
                # We could get the reset observation by calling reset, but that would be
                # inconsistent with other envs in the batch.
                # For now, just mark as terminated.
                pass

        if self.visualize:
            self._render()

        info = {"step_count": self.step_counts.copy()}

        return np.array(next_states), np.array(rewards), np.array(terminated), info

    def reset(self, batch_size=None, initial_state=None):
        """
        Resets environments to new initial states. Optionally changes the batch size
        and/or sets specific initial states.

        :param batch_size: Number of environments to use. If None, keeps current batch size.
        :param initial_state: Initial states for environments with shape (batch_size, state_dim).
                             If None, uses random initialization.
        :return: The initial states of all environments with shape (batch_size, state_dim).
        """
        if batch_size is not None:
            self._create_environments(batch_size)

        if initial_state is not None:
            if initial_state.shape[0] != self.batch_size:
                raise ValueError(
                    "initial_state batch size must match environment batch size."
                )

        states = []
        for i, env in enumerate(self.envs):
            if initial_state is None:
                timestep = env.reset()
                states.append(self._flatten_observation(timestep.observation))
            else:
                # Set the state of the environment based on the initial_state vector.
                with env.physics.reset_context():
                    # initial_state is [cart_pos, cos(angle), sin(angle), cart_vel, pole_vel]
                    cart_pos = initial_state[i, 0]
                    cos_angle = initial_state[i, 1]
                    sin_angle = initial_state[i, 2]
                    cart_vel = initial_state[i, 3]
                    pole_vel = initial_state[i, 4]

                    # Convert cos/sin back to angle
                    pole_angle = np.arctan2(sin_angle, cos_angle)

                    # Set the physics state
                    env.physics.named.data.qpos["slider"] = cart_pos
                    env.physics.named.data.qpos["hinge_1"] = pole_angle
                    env.physics.named.data.qvel["slider"] = cart_vel
                    env.physics.named.data.qvel["hinge_1"] = pole_vel

                # After setting the state, we need to get the corresponding observation.
                # The environment doesn't have a direct way to do this without stepping,
                # so we can call the task's get_observation method.
                states.append(
                    self._flatten_observation(env.task.get_observation(env.physics))
                )

        self.step_counts.fill(0)

        if self.visualize:
            self._render()

        return np.array(states)

    def close(self):
        """Closes all environments."""
        for env in self.envs:
            env.close()
        self.envs = []

        if self.visualize:
            cv2.destroyAllWindows()

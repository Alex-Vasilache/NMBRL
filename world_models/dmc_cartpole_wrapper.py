from typing import Optional, Dict, Any, Callable
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces
from dm_control import suite
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

WINDOW_WIDTH = 360
WINDOW_HEIGHT = 270


class DMCVecEnvWrapper(DummyVecEnv):
    def __init__(
        self,
        batch_size: int = 1,
        max_episode_steps: int = 1000,
    ):
        self.batch_size = batch_size
        super().__init__(
            [
                lambda: DMCWrapper(
                    domain_name="cartpole",
                    task_name="swingup",
                    render_mode=None,
                    max_episode_steps=max_episode_steps,
                )
                for _ in range(batch_size)
            ]
        )


class DMCWrapper(gym.Env):
    """
    Wrapper to convert a dm_control environment to a gymnasium environment.
    It processes observations to be a flat numpy array.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        domain_name,
        task_name,
        render_mode: Optional[str] = None,
        camera_id=0,
        max_episode_steps=1000,
    ):
        self.env = suite.load(
            domain_name,
            task_name,
            visualize_reward=True,
            task_kwargs={"time_limit": float("inf")},
        )
        self.render_mode = render_mode
        self.camera_id = camera_id
        self.max_episode_steps = max_episode_steps
        # Convert dm_control action spec to gymnasium space
        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

        # Convert dm_control observation spec to gymnasium space
        obs_spec = self.env.observation_spec()
        self.observation_space = self._convert_obs_spec_to_space(obs_spec)

    def _convert_obs_spec_to_space(self, obs_spec):
        total_dim = sum(np.prod(spec.shape) for spec in obs_spec.values())
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(total_dim),), dtype=np.float32
        )

    def _flatten_obs(self, obs):
        return np.concatenate([np.ravel(o) for o in obs.values()])

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._flatten_obs(time_step.observation)
        reward = time_step.reward or 0
        terminated = time_step.last()
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        time_step = self.env.reset()
        obs = self._flatten_obs(time_step.observation)
        info = {}
        return obs, info

    def render(self):
        if self.render_mode == "human":
            try:
                import cv2

                # Initialize the window if not already done
                if not hasattr(self, "_window_initialized"):
                    self._window_name = "DMC Cartpole"
                    cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self._window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
                    self._window_initialized = True

                # Get the frame from dm_control
                frame = self.env.physics.render(
                    width=WINDOW_WIDTH,
                    height=WINDOW_HEIGHT,
                    camera_id=self.camera_id,
                )

                # OpenCV expects BGR, dm_control renders in RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Show the image
                cv2.imshow(self._window_name, frame_bgr)
                cv2.waitKey(1)  # Necessary to process GUI events

                return None
            except ImportError:
                print("OpenCV not available, falling back to rgb_array mode")
                return self.env.physics.render(
                    height=WINDOW_HEIGHT, width=WINDOW_WIDTH, camera_id=self.camera_id
                )
            except Exception as e:
                print(f"Error in human rendering: {e}, falling back to rgb_array mode")
                return self.env.physics.render(
                    height=WINDOW_HEIGHT, width=WINDOW_WIDTH, camera_id=self.camera_id
                )
        else:
            return self.env.physics.render(
                height=WINDOW_HEIGHT, width=WINDOW_WIDTH, camera_id=self.camera_id
            )

    def close(self):
        try:
            import cv2

            if hasattr(self, "_window_initialized"):
                cv2.destroyAllWindows()
                delattr(self, "_window_initialized")
        except ImportError:
            pass


def make_dmc_env(
    render_mode: str = "none", max_episode_steps: int = 1000
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = DMCWrapper(
            "cartpole",
            "swingup",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return Monitor(env)

    return _init


class DMCCartpoleWrapper(VecNormalize):
    def __init__(
        self,
        seed: int = 42,
        n_envs: int = 1,
        render_mode: str = "human",
        max_episode_steps: int = 1000,
    ):
        self.n_envs = n_envs
        self._seed = seed

        env_fns = [
            make_dmc_env(render_mode, max_episode_steps) for _ in range(self.n_envs)
        ]

        if n_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

        super().__init__(
            vec_env,
            norm_obs=False,
            norm_reward=False,
            clip_obs=100.0,
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
            # since vectorized environments don't handle human rendering well
            return self.venv.env_method("render")
        else:
            # For other modes, use the vectorized environment's render method
            return self.venv.render(mode=mode)

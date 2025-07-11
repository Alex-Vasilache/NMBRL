# This file will contain a wrapper for the world model.
# The world model will be a neural network that predicts the next state of the environment.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from world_models.world_model_v1 import SimpleModel
import os
import threading
import time
from datetime import datetime


class WorldModelWrapper(DummyVecEnv):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: dict,
        batch_size: int = 1,
        trained_folder: str = "world_models/trained/v1",
    ):
        # --- Config ---
        self.config = config
        wrapper_config = self.config["sac_trainer"]["world_model_wrapper"]
        model_check_interval_s = wrapper_config["model_check_interval_s"]
        obs_clip_range = wrapper_config["obs_clip_range"]
        reward_clip_range = tuple(wrapper_config["reward_clip_range"])

        # --- Basic Env Setup ---
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(self.observation_space, spaces.Box)
        self.state_size = self.observation_space.shape[0]
        assert isinstance(self.action_space, spaces.Box)
        self.action_size = self.action_space.shape[0]

        # --- Clipping Ranges ---
        self.obs_clip_range = obs_clip_range
        self.reward_clip_range = reward_clip_range

        # --- DummyVecEnv Initialization ---
        class DummyGymEnv(gym.Env):
            def __init__(self, obs_space, act_space):
                self.observation_space = obs_space
                self.action_space = act_space

            def reset(self, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0, False, False, {}

        super().__init__(
            [
                lambda: DummyGymEnv(self.observation_space, self.action_space)
                for _ in range(batch_size)
            ]
        )

        # --- Dynamic Model Loading Setup ---
        self.model_folder = trained_folder
        self.model_path = os.path.join(self.model_folder, "model.pth")
        self.nn_model = None
        self.latest_mod_time = None
        self.model_lock = threading.Lock()

        # --- Wait for the first model to be created ---
        print(
            f"[{datetime.now()}] Waiting for the first world model to appear at: {self.model_path}"
        )
        wait_timeout_seconds = 300  # 5 minutes
        wait_start_time = time.time()
        model_found = False
        while not model_found and time.time() - wait_start_time < wait_timeout_seconds:
            if os.path.exists(self.model_path):
                model_found = True
            else:
                time.sleep(2)  # Poll every 2 seconds

        if not model_found:
            raise RuntimeError(
                f"World model did not appear within {wait_timeout_seconds} seconds. Aborting."
            )

        # Attempt to load model at startup
        self._find_and_load_model_if_new()

        if self.nn_model is None:
            # This should ideally not be reached if the wait logic is correct
            raise RuntimeError(
                f"Failed to load the world model from {self.model_path} after it was found."
            )

        # Start the background thread to watch for new models
        self.stop_event = threading.Event()
        self.model_watcher_thread = threading.Thread(
            target=self._model_watcher,
            args=(model_check_interval_s,),
            daemon=True,
        )
        self.model_watcher_thread.start()
        print("Model watcher thread started.")

        # --- State and Buffer Setup ---
        self.state = None
        self.buf_obs = np.zeros(
            (self.num_envs, self.state_size), dtype=self.observation_space.dtype
        )
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def _create_model_instance(self):
        """Creates an instance of the model with the correct dimensions."""
        # Get the hidden dimension from the same config used by the trainer
        world_model_config = self.config["world_model_trainer"]
        hidden_dim = world_model_config["hidden_dim"]

        return SimpleModel(
            input_dim=self.state_size + self.action_size,
            hidden_dim=hidden_dim,
            output_dim=self.state_size + 1,  # next_state + reward
            state_size=self.state_size,
            action_size=self.action_size,
        )

    def _create_placeholder_model(self):
        """Creates a model that always returns zeros."""

        class PlaceholderModel(torch.nn.Module):
            def __init__(self, output_dim):
                super().__init__()
                self.output_dim = output_dim

            def forward(self, x):
                return torch.zeros(x.shape[0], self.output_dim)

        return PlaceholderModel(self.state_size + 1)

    def _find_and_load_model_if_new(self):
        """Checks for a new model file and loads it if it's newer than the current one."""
        if not os.path.exists(self.model_path):
            return False

        try:
            current_mod_time = os.path.getmtime(self.model_path)
            if self.latest_mod_time is None or current_mod_time > self.latest_mod_time:
                print(
                    f"[{datetime.now()}] Detected new or updated model: {self.model_path}"
                )

                # Wait briefly to prevent loading a partially written file
                time.sleep(0.5)

                new_model = self._create_model_instance()
                # Load onto CPU to avoid potential CUDA initialization issues in the thread
                new_model.load_state_dict(
                    torch.load(self.model_path, map_location="cpu")
                )
                new_model.eval()  # Set to evaluation mode

                with torch.no_grad():
                    # --- Test with a dummy input to check for NaNs ---
                    dummy_input = torch.randn(1, self.state_size + self.action_size)
                    test_output = new_model(dummy_input)
                    if torch.isnan(test_output).any():
                        print(
                            f"[{datetime.now()}] ERROR: Model at {self.model_path} produced NaNs on a test input. Skipping load."
                        )
                        return False

                with self.model_lock:
                    self.nn_model = new_model
                self.latest_mod_time = current_mod_time
                print(
                    f"[{datetime.now()}] Successfully loaded model updated at {time.ctime(current_mod_time)}"
                )
                return True
        except Exception as e:
            print(f"Error loading model {self.model_path}: {e}")

        return False

    def _model_watcher(self, check_interval_s: int):
        """Periodically checks for a new model file."""
        while not self.stop_event.is_set():
            self._find_and_load_model_if_new()
            time.sleep(check_interval_s)
        print("Model watcher thread stopped.")

    def step(self, action_np):  # Overrides DummyVecEnv.step
        action_tensor = (
            torch.from_numpy(action_np).float().reshape(self.num_envs, self.action_size)
        )

        with torch.no_grad(), self.model_lock:
            if self.state is None:
                raise RuntimeError("Must call reset() or set_state() before step().")

            nn_input = torch.cat([self.state, action_tensor], dim=1)
            outputs = self.nn_model(nn_input)

            next_state_tensor = outputs[:, : self.state_size]
            reward_tensor = outputs[:, self.state_size]

            # --- NaN/inf check and clamping ---
            if (
                torch.isnan(next_state_tensor).any()
                or torch.isinf(next_state_tensor).any()
            ):
                print(
                    f"[{datetime.now()}] WARNING: NaN/inf detected in predicted next_state. State: {self.state.numpy()}, Action: {action_np}"
                )
                # Replace NaNs with zeros or a reset state
                next_state_tensor = torch.nan_to_num(
                    next_state_tensor,
                    nan=0.0,
                    posinf=self.obs_clip_range,
                    neginf=-self.obs_clip_range,
                )

            clamped_next_state = torch.clamp(
                next_state_tensor, -self.obs_clip_range, self.obs_clip_range
            )
            clamped_reward = torch.clamp(
                reward_tensor, self.reward_clip_range[0], self.reward_clip_range[1]
            )

            self.state = clamped_next_state

        terminated = np.zeros(self.num_envs, dtype=bool)
        # In a real scenario, you might want the world model to predict termination.
        # For now, we assume it never terminates on its own.
        infos = [{} for _ in range(self.num_envs)]
        return (
            clamped_next_state.numpy(),
            clamped_reward.numpy(),
            terminated,
            infos,
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().seed(seed)
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

    def close(self):
        """Cleanly shuts down the model watcher thread."""
        print("Closing WorldModelWrapper and stopping model watcher thread...")
        self.stop_event.set()
        # Wait for the thread to finish
        self.model_watcher_thread.join(timeout=5)
        if self.model_watcher_thread.is_alive():
            print("Warning: Model watcher thread did not shut down cleanly.")
        # No super().close() needed as the dummy envs don't need closing.
        print("WorldModelWrapper closed.")

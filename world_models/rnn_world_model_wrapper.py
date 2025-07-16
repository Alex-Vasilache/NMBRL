# This file contains a wrapper for the RNN world model.
# The RNN world model will be a neural network that predicts sequences of states and rewards.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.serialization
from stable_baselines3.common.vec_env import DummyVecEnv
from networks.world_model_rnn import RNNWorldModel
import os
import threading
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from networks.world_model_rnn import load_model
from utils.tools import resolve_device


class RNNWorldModelWrapper(DummyVecEnv):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        config: dict,
        batch_size: int = 1,
        shared_folder: str = os.path.join(os.path.dirname(__file__), "..", "runs"),
    ):
        # --- Config ---
        self.config = config
        wrapper_config = self.config["agent_trainer"]["world_model_wrapper"]
        model_check_interval_s = wrapper_config["model_check_interval_s"]
        obs_clip_range = wrapper_config["obs_clip_range"]
        reward_clip_range = tuple(wrapper_config["reward_clip_range"])
        self.use_scalers = self.config["world_model_trainer"]["use_scalers"]
        self.use_input_state_scaler = self.use_scalers
        self.use_input_action_scaler = self.use_scalers
        self.use_output_state_scaler = self.config["world_model_trainer"][
            "use_output_state_scaler"
        ]
        self.use_output_reward_scaler = self.config["world_model_trainer"][
            "use_output_reward_scaler"
        ]

        # --- Basic Env Setup ---
        self.observation_space = observation_space
        self.action_space = action_space
        assert isinstance(self.observation_space, spaces.Box)
        self.state_size = self.observation_space.shape[0]
        assert isinstance(self.action_space, spaces.Box)
        self.action_size = self.action_space.shape[0]
        self.terminated = np.zeros(batch_size, dtype=bool)
        self.infos = [{} for _ in range(batch_size)]
        self.max_episode_steps = self.config["agent_trainer"]["max_episode_steps"]

        # --- Clipping Ranges ---
        self.obs_clip_range = obs_clip_range
        self.reward_clip_range = reward_clip_range

        # --- TensorBoard Setup ---
        self.shared_folder = shared_folder
        tb_config = self.config.get("tensorboard", {})
        tb_log_dir = os.path.join(
            shared_folder,
            tb_config.get("log_dir", "tb_logs"),
            "rnn_world_model_wrapper",
        )
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=tb_log_dir, flush_secs=tb_config.get("flush_seconds", 30)
        )
        self.log_frequency = tb_config.get("log_frequency", 10)
        self.step_count_global = 0
        self.model_load_count = 0
        self.last_log_time = time.time()

        print(f"[RNN-WORLD-MODEL-WRAPPER] TensorBoard logging to: {tb_log_dir}")

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
        self.model_path = os.path.join(self.shared_folder, "model.pth")
        self.nn_model = None
        self.latest_mod_time = None
        self.model_lock = threading.Lock()

        # --- Wait for the first model to be created ---
        print(
            f"[{datetime.now()}] Waiting for the first RNN world model to appear at: {self.model_path}"
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
                f"RNN World model did not appear within {wait_timeout_seconds} seconds. Aborting."
            )

        # Attempt to load model at startup
        self._find_and_load_model_if_new()

        if self.nn_model is None:
            raise RuntimeError(
                f"Failed to load the RNN world model from {self.model_path} after it was found."
            )

        # Start the background thread to watch for new models
        self.stop_event = threading.Event()
        self.model_watcher_thread = threading.Thread(
            target=self._model_watcher,
            args=(model_check_interval_s,),
            daemon=True,
        )
        self.model_watcher_thread.start()
        print("RNN Model watcher thread started.")

        # --- State and Buffer Setup ---
        self.state = None
        self.buf_obs = np.zeros(
            (self.num_envs, self.state_size), dtype=self.observation_space.dtype
        )
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def _create_model_instance(self):
        """Creates an instance of the RNN model with the correct dimensions."""
        world_model_config = self.config["world_model_trainer"]
        hidden_dim = world_model_config["hidden_dim"]
        num_layers = world_model_config.get("num_layers", 2)
        dropout = world_model_config.get("dropout", 0.1)

        return RNNWorldModel(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _find_and_load_model_if_new(self):
        """Checks for a new model file and loads it if it's newer than the current one."""
        if not os.path.exists(self.model_path):
            return False

        try:
            current_mod_time = os.path.getmtime(self.model_path)
            if self.latest_mod_time is None or current_mod_time > self.latest_mod_time:
                print(
                    f"[{datetime.now()}] Detected new or updated RNN model: {self.model_path}"
                )

                # Wait briefly to prevent loading a partially written file
                time.sleep(0.5)

                torch.serialization.add_safe_globals([RNNWorldModel])
                device = resolve_device("global", self.config["global"])
                new_model = load_model(
                    self.model_path,
                    with_scalers=self.use_scalers,
                    map_location=device,
                    weights_only=False,
                )
                # Ensure the model is properly moved to the target device
                new_model = new_model.to(device)
                new_model.eval()

                with self.model_lock:
                    self.nn_model = new_model
                    self.latest_mod_time = current_mod_time
                    self.model_load_count += 1

                print(
                    f"[{datetime.now()}] Successfully loaded RNN model (load #{self.model_load_count})"
                )

                # Log to TensorBoard
                if self.writer:
                    self.writer.add_scalar(
                        "Model/Load_Count",
                        self.model_load_count,
                        self.step_count_global,
                    )

                return True

        except Exception as e:
            print(f"[{datetime.now()}] Error loading RNN model: {e}")
            return False

    def _model_watcher(self, check_interval_s):
        """Background thread that watches for new model files."""
        while not self.stop_event.is_set():
            self._find_and_load_model_if_new()
            time.sleep(check_interval_s)

    def step(self, action_np):
        device = resolve_device("global", self.config["global"])
        action_tensor = (
            torch.from_numpy(action_np)
            .float()
            .reshape(self.num_envs, self.action_size)
            .to(device)
        )

        with torch.no_grad(), self.model_lock:
            if self.state is None:
                raise RuntimeError("Must call reset() or set_state() before step().")

            if self.nn_model is None:
                raise RuntimeError("RNN World model is not loaded, cannot step.")

            # Ensure state is on the correct device
            self.state = self.state.to(device)

            # For RNN model, we need to create a sequence input
            # Since we're doing single-step prediction, we create a sequence of length 1
            nn_input = torch.cat([self.state, action_tensor], dim=1)
            nn_input = nn_input.unsqueeze(
                1
            )  # Add sequence dimension: (batch_size, 1, state_size + action_size)

            outputs = self.nn_model(
                nn_input,
                use_input_state_scaler=self.use_input_state_scaler,
                use_input_action_scaler=self.use_input_action_scaler,
                use_output_state_scaler=self.use_output_state_scaler,
                use_output_reward_scaler=self.use_output_reward_scaler,
            )

            # Remove sequence dimension and extract outputs
            outputs = outputs.squeeze(1)  # (batch_size, state_size + 1)
            next_state_tensor = outputs[:, : self.state_size]
            reward_tensor = outputs[:, self.state_size]

            self.step_count += 1
            self.step_count_global += 1
            self.infos = [{} for _ in range(self.num_envs)]
            self.terminated = np.zeros(self.num_envs, dtype=bool)

            if self.step_count >= self.max_episode_steps:
                self.terminated = np.ones(self.num_envs, dtype=bool)
                self.infos = [{"terminal_observation": t} for t in self.state]
                next_state_tensor = torch.from_numpy(self.reset()).to(device)

            # --- NaN/inf check and clamping ---
            nan_detected = False
            if (
                torch.isnan(next_state_tensor).any()
                or torch.isinf(next_state_tensor).any()
            ):
                nan_detected = True
                print(
                    f"[{datetime.now()}] NaN/inf detected in state prediction. Clamping to safe values."
                )
                next_state_tensor = torch.clamp(next_state_tensor, -10.0, 10.0)

            if torch.isnan(reward_tensor).any() or torch.isinf(reward_tensor).any():
                nan_detected = True
                print(
                    f"[{datetime.now()}] NaN/inf detected in reward prediction. Clamping to safe values."
                )
                reward_tensor = torch.clamp(reward_tensor, -2.0, 2.0)

            # --- Clipping based on configuration ---
            next_state_tensor = torch.clamp(
                next_state_tensor, -self.obs_clip_range, self.obs_clip_range
            )
            reward_tensor = torch.clamp(
                reward_tensor, self.reward_clip_range[0], self.reward_clip_range[1]
            )

            # --- Update internal state ---
            self.state = next_state_tensor

            # --- Convert to numpy for return ---
            next_state_np = next_state_tensor.cpu().numpy()
            reward_np = reward_tensor.cpu().numpy()

            # --- Logging ---
            current_time = time.time()
            if (
                self.writer
                and current_time - self.last_log_time > 1.0 / self.log_frequency
            ):
                self.writer.add_scalar(
                    "Prediction/State_Mean",
                    next_state_tensor.mean().item(),
                    self.step_count_global,
                )
                self.writer.add_scalar(
                    "Prediction/State_Std",
                    next_state_tensor.std().item(),
                    self.step_count_global,
                )
                self.writer.add_scalar(
                    "Prediction/Reward_Mean",
                    reward_tensor.mean().item(),
                    self.step_count_global,
                )
                self.writer.add_scalar(
                    "Prediction/Reward_Std",
                    reward_tensor.std().item(),
                    self.step_count_global,
                )
                if nan_detected:
                    self.writer.add_scalar(
                        "Prediction/NaN_Detected", 1.0, self.step_count_global
                    )
                self.last_log_time = current_time

            return next_state_np, reward_np, self.terminated, self.infos

    def reset(
        self,
        *,
        seed=None,
        options=None,
        remove_from_replay_buffer=False,
    ):
        if seed is not None:
            super().seed(seed)
            torch.manual_seed(seed)

        # It's possible to be reset before the first model is loaded and has a valid init state
        if self.nn_model is not None and self.nn_model.valid_init_state is not None:
            buffer_size = self.nn_model.valid_init_state.shape[0]
            half_envs = self.num_envs // 2
            remaining_envs = self.num_envs - half_envs

            # Get latest half from the end of the buffer
            latest_start_idx = max(0, buffer_size - half_envs)
            latest_idxs = np.arange(latest_start_idx, buffer_size)
            if len(latest_idxs) < half_envs:
                # If buffer is smaller than half_envs, pad with random indices
                needed = half_envs - len(latest_idxs)
                random_padding = np.random.randint(0, buffer_size, needed)
                latest_idxs = np.concatenate([latest_idxs, random_padding])

            # Get random half from the rest of the buffer
            if buffer_size > half_envs:
                remaining_buffer_size = buffer_size - half_envs
                random_idxs = np.random.randint(
                    0, remaining_buffer_size, remaining_envs
                )
            else:
                random_idxs = np.random.randint(0, buffer_size, remaining_envs)

            # Combine indices
            combined_idxs = np.concatenate([latest_idxs[:half_envs], random_idxs])

            device = resolve_device("global", self.config["global"])
            self.state = self.nn_model.valid_init_state[combined_idxs]

            # Ensure state is on the correct device
            if isinstance(self.state, torch.Tensor):
                self.state = self.state.to(device)
            else:
                self.state = torch.from_numpy(self.state).float().to(device)

            if remove_from_replay_buffer:
                self.nn_model.valid_init_state = np.delete(
                    self.nn_model.valid_init_state, combined_idxs, axis=0
                )

            if (
                self.use_scalers and not self.use_output_state_scaler
            ):  # scale original range to [-3, 3]
                self.step_count = 0
                return self.nn_model._do_scale(self.state, "state").cpu().numpy()
        else:
            # Fallback to a random state if the model or its buffer isn't ready
            device = resolve_device("global", self.config["global"])
            self.state = (
                torch.from_numpy(self.observation_space.sample())
                .float()
                .unsqueeze(0)
                .to(device)
            )

        self.step_count = 0
        return self.state.cpu().numpy()

    def set_state(self, state_np: np.ndarray):
        """Sets the internal state of the RNN world model."""
        if state_np.ndim == 1:
            state_np = np.expand_dims(state_np, axis=0)

        expected_shape = (self.num_envs, self.state_size)
        if state_np.shape != expected_shape:
            raise ValueError(
                f"Expected state shape {expected_shape}, but got {state_np.shape}"
            )

        device = resolve_device("global", self.config["global"])
        self.state = torch.from_numpy(state_np).float().to(device)

    def close(self):
        """Clean up resources."""
        if hasattr(self, "writer"):
            self.writer.close()
        if hasattr(self, "stop_event"):
            self.stop_event.set()
        super().close()

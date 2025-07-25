# This file will contain a wrapper for the world model.
# The world model will be a neural network that predicts the next state of the environment.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.serialization
from stable_baselines3.common.vec_env import DummyVecEnv
from networks.world_model_v1 import SimpleModel
import os
import threading
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from networks.world_model_v1 import load_model
from networks.world_model_v1 import STATE_SCALER
from utils.tools import resolve_device


class WorldModelWrapper(DummyVecEnv):
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
            shared_folder, tb_config.get("log_dir", "tb_logs"), "world_model_wrapper"
        )
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=tb_log_dir, flush_secs=tb_config.get("flush_seconds", 30)
        )
        self.log_frequency = tb_config.get("log_frequency", 10)
        self.step_count_global = 0
        self.model_load_count = 0
        self.last_log_time = time.time()

        print(f"[WORLD-MODEL-WRAPPER] TensorBoard logging to: {tb_log_dir}")

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

    # def _create_placeholder_model(self):
    #     """Creates a model that always returns zeros."""
    #
    #     class PlaceholderModel(torch.nn.Module):
    #         def __init__(self, output_dim):
    #             super().__init__()
    #             self.output_dim = output_dim
    #             self.valid_init_state = torch.zeros(self.num_envs, output_dim - 1)
    #
    #         def forward(self, x):
    #             return torch.zeros(x.shape[0], self.output_dim)
    #
    #     return PlaceholderModel(self.state_size + 1)

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

                torch.serialization.add_safe_globals([SimpleModel])
                device = resolve_device("global", self.config["global"])
                # Patch: If use_scalers is True but use_existing_scalers is False and scalers are missing, load without scalers
                use_existing_scalers = self.config["global"].get(
                    "use_existing_scalers", True
                )
                scaler_files_exist = all(
                    [
                        os.path.exists(os.path.join(self.shared_folder, f))
                        for f in [
                            "state_scaler.joblib",
                            "action_scaler.joblib",
                            "reward_scaler.joblib",
                        ]
                    ]
                )
                try:
                    if (
                        self.use_scalers
                        and (not scaler_files_exist)
                        and (not use_existing_scalers)
                    ):
                        print(
                            f"[{datetime.now()}] Scaler files missing and use_existing_scalers is False. Loading model without scalers."
                        )
                        new_model = load_model(
                            self.model_path,
                            with_scalers=False,
                            map_location=device,
                            weights_only=False,
                        )
                        if hasattr(new_model, "set_scalers"):
                            new_model.set_scalers(None, None, None)
                    else:
                        new_model = load_model(
                            self.model_path,
                            with_scalers=self.use_scalers,
                            map_location=device,
                            weights_only=False,
                        )
                except Exception as e:
                    print(f"Error loading model {self.model_path}: {e}")
                    return False
                # Ensure the model is properly moved to the target device
                new_model = new_model.to(device)
                new_model.eval()  # Set to evaluation mode

                with torch.no_grad():
                    # --- Test with a dummy input to check for NaNs ---
                    dummy_input = torch.randn(
                        1, self.state_size + self.action_size, device=device
                    )
                    test_output = new_model(
                        dummy_input,
                        use_input_state_scaler=self.use_input_state_scaler,
                        use_input_action_scaler=self.use_input_action_scaler,
                        use_output_state_scaler=self.use_output_state_scaler,
                        use_output_reward_scaler=self.use_output_reward_scaler,
                    )
                    if torch.isnan(test_output).any():
                        print(
                            f"[{datetime.now()}] ERROR: Model at {self.model_path} produced NaNs on a test input. Skipping load."
                        )
                        return False

                with self.model_lock:
                    self.nn_model = new_model
                self.latest_mod_time = current_mod_time
                self.model_load_count += 1

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
                # This should not happen if __init__ completed successfully.
                raise RuntimeError("World model is not loaded, cannot step.")

            # Ensure state is on the correct device
            self.state = self.state.to(device)
            nn_input = torch.cat([self.state, action_tensor], dim=1)
            outputs = self.nn_model(
                nn_input,
                use_input_state_scaler=self.use_input_state_scaler,
                use_input_action_scaler=self.use_input_action_scaler,
                use_output_state_scaler=self.use_output_state_scaler,
                use_output_reward_scaler=self.use_output_reward_scaler,
            )

            next_state_tensor = outputs[:, : self.state_size]
            reward_tensor = outputs[:, self.state_size]

            self.step_count += 1
            self.step_count_global += 1
            self.infos = [{} for _ in range(self.num_envs)]
            self.terminated = np.zeros(self.num_envs, dtype=bool)

            if self.step_count >= self.max_episode_steps:
                self.terminated = np.ones(self.num_envs, dtype=bool)
                # print(f"[{datetime.now()}] Terminated after {self.step_count} steps")
                self.infos = [{"terminal_observation": t} for t in self.state]
                next_state_tensor = torch.from_numpy(self.reset()).to(device)

            # --- NaN/inf check and clamping ---
            nan_detected = False
            if (
                torch.isnan(next_state_tensor).any()
                or torch.isinf(next_state_tensor).any()
            ):
                print(
                    f"[{datetime.now()}] WARNING: NaN/inf detected in predicted next_state. State: {self.state.cpu().numpy()}, Action: {action_np}"
                )
                # Replace NaNs with zeros or a reset state
                next_state_tensor = torch.nan_to_num(
                    next_state_tensor,
                    nan=0.0,
                    posinf=self.obs_clip_range,
                    neginf=-self.obs_clip_range,
                )
                nan_detected = True

            clamped_next_state = torch.clamp(
                next_state_tensor, -self.obs_clip_range, self.obs_clip_range
            )
            clamped_reward = torch.clamp(
                reward_tensor, self.reward_clip_range[0], self.reward_clip_range[1]
            )

            self.state = clamped_next_state

            if (
                self.use_scalers and not self.use_output_state_scaler
            ):  # scale original range to [-3, 3]
                self.state = self.nn_model._do_unscale(self.state, STATE_SCALER)

            # --- TensorBoard Logging ---
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_frequency:
                # Log prediction statistics
                self.writer.add_scalar(
                    "Wrapper/Predictions_Mean_Reward",
                    torch.mean(clamped_reward).item(),
                    self.step_count_global,
                )
                self.writer.add_scalar(
                    "Wrapper/Predictions_Std_Reward",
                    torch.std(clamped_reward).item(),
                    self.step_count_global,
                )

                # Log state statistics
                state_mean = torch.mean(clamped_next_state, dim=0)
                state_std = torch.std(clamped_next_state, dim=0)
                for i in range(self.state_size):
                    self.writer.add_scalar(
                        f"Wrapper/State_Mean_{i}",
                        state_mean[i].item(),
                        self.step_count_global,
                    )
                    self.writer.add_scalar(
                        f"Wrapper/State_Std_{i}",
                        state_std[i].item(),
                        self.step_count_global,
                    )

                # Log environment metrics
                self.writer.add_scalar(
                    "Wrapper/Environment_Episode_Steps",
                    self.step_count,
                    self.step_count_global,
                )
                self.writer.add_scalar(
                    "Wrapper/Environment_Total_Steps",
                    self.step_count_global,
                    self.step_count_global,
                )

                self.last_log_time = current_time

        return (
            clamped_next_state.cpu().numpy(),
            clamped_reward.cpu().numpy(),
            self.terminated,
            self.infos,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
        remove_from_replay_buffer=False,
    ):
        if seed is not None:
            if type(seed) == int:
                super().seed(seed)
                torch.manual_seed(seed)

        if seed is not None and type(seed) == list:
            # list is something like [0,0,0,100,100,100,200,200,200]
            num_unique_seeds = len(set(seed))  # this would be 3

            if self.nn_model is not None and self.nn_model.valid_init_state is not None:
                valid_init_state_size = self.nn_model.valid_init_state.shape[0]

                # Split unique seeds into two halves
                half_unique_seeds = num_unique_seeds // 2
                remaining_unique_seeds = num_unique_seeds - half_unique_seeds

                # Get latest 1000 from buffer for first half
                latest_start_idx = max(0, valid_init_state_size - 1000)
                latest_buffer_size = valid_init_state_size - latest_start_idx

                if latest_buffer_size >= half_unique_seeds:
                    latest_indices = np.random.choice(
                        np.arange(latest_start_idx, valid_init_state_size),
                        half_unique_seeds,
                        replace=False,
                    )
                else:
                    # If latest 1000 is smaller than needed, use all of it and pad with random
                    latest_indices = np.arange(latest_start_idx, valid_init_state_size)
                    needed = half_unique_seeds - len(latest_indices)
                    random_padding = np.random.choice(
                        valid_init_state_size, needed, replace=False
                    )
                    latest_indices = np.concatenate([latest_indices, random_padding])

                # Get remaining seeds from rest of buffer
                if latest_start_idx > 0:
                    rest_indices = np.random.choice(
                        latest_start_idx, remaining_unique_seeds, replace=False
                    )
                else:
                    # If buffer is smaller than 1000, get remaining from entire buffer
                    rest_indices = np.random.choice(
                        valid_init_state_size, remaining_unique_seeds, replace=False
                    )

                # Combine indices
                selected_indices = np.concatenate([latest_indices, rest_indices])
                selected_states = self.nn_model.valid_init_state[selected_indices]

                # Map the selected states to the original number of seeds
                unique_seeds = list(set(seed))
                seed_to_state = dict(zip(unique_seeds, selected_states))
                return_states = torch.stack([seed_to_state[s] for s in seed])
                self.state = return_states

                device = resolve_device("global", self.config["global"])

                if isinstance(self.state, torch.Tensor):
                    self.state = self.state.to(device)
                else:
                    self.state = torch.from_numpy(self.state).float().to(device)

                if (
                    self.use_scalers and not self.use_output_state_scaler
                ):  # scale original range to [-3, 3]
                    self.step_count = 0
                    return (
                        self.nn_model._do_scale(self.state, STATE_SCALER).cpu().numpy()
                    )

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

            # print(
            #     f"[{datetime.now()}] Resetting with valid init state from replay using indices: {combined_idxs}"
            # )

            device = resolve_device("global", self.config["global"])
            self.state = self.nn_model.valid_init_state[
                combined_idxs
            ]  # these are states in the original environment range

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
                return self.nn_model._do_scale(self.state, STATE_SCALER).cpu().numpy()
        else:
            # Fallback to a batch of random states if the model or its buffer isn't ready
            device = resolve_device("global", self.config["global"])
            self.state = (
                torch.from_numpy(
                    np.stack(
                        [self.observation_space.sample() for _ in range(self.num_envs)]
                    )
                )
                .float()
                .to(device)
            )
            # print(
            #     f"[{datetime.now()}] Resetting with random state from observation space"
            # )

        self.step_count = 0
        return self.state.cpu().numpy()

    def set_state(self, state_np: np.ndarray):
        """Sets the internal state of the world model."""
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
        """Cleanly shuts down the model watcher thread."""
        print("Closing WorldModelWrapper and stopping model watcher thread...")
        self.stop_event.set()
        # Wait for the thread to finish
        self.model_watcher_thread.join(timeout=5)
        if self.model_watcher_thread.is_alive():
            print("Warning: Model watcher thread did not shut down cleanly.")
        # Close TensorBoard writer
        self.writer.close()
        print(f"[WORLD-MODEL-WRAPPER] TensorBoard logs saved to: {self.writer.log_dir}")
        # No super().close() needed as the dummy envs don't need closing.
        print("WorldModelWrapper closed.")

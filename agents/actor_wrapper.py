import os
import threading
import numpy as np
from stable_baselines3.common.callbacks import CallbackList
import joblib
from utils.tools import resolve_device

MAX_ACTION_CHANGE = 0.4
MAX_ACTION_SCALE = 0.7


class RandomPolicy:
    """A simple policy that returns random actions."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.previous_action = None

    def predict(self, obs, deterministic=False):
        if self.previous_action is None:
            # First action - can be anywhere in action space
            action = self.action_space.sample().reshape(1, -1)
        else:
            # Ensure change doesn't exceed 0.5
            max_change = MAX_ACTION_CHANGE
            # Sample a change within [-0.5, 0.5]
            change = np.random.uniform(
                -max_change, max_change, size=self.previous_action.shape
            )
            # Apply change to previous action
            new_action = self.previous_action + change
            # Clamp to action space bounds
            action = np.clip(
                new_action,
                self.action_space.low * MAX_ACTION_SCALE,
                self.action_space.high * MAX_ACTION_SCALE,
            )
            action = action.reshape(1, -1)

        # Ensure correct dtype to match action space
        action = action.astype(self.action_space.dtype)

        self.previous_action = action.copy()
        return action, None

    def learn(
        self,
        total_timesteps: int,
        callback: CallbackList = CallbackList([]),
        progress_bar: bool = False,
    ):
        pass


class ActorWrapper:
    """
    A wrapper class that manages loading and providing the latest actor model
    in a non-blocking way using a background thread.
    """

    def __init__(
        self,
        env,
        config: dict,
        training: bool = False,
        shared_folder: str = os.path.join(os.path.dirname(__file__), "..", "runs"),
    ):
        self.shared_folder = shared_folder
        self.agent_folder = os.path.join(self.shared_folder, "actor_logs")
        self.env = env
        self.config = config
        # Use shared TensorBoard logs directory
        tb_config = self.config.get("tensorboard", {})
        self.tb_log_dir = os.path.join(
            self.shared_folder, tb_config.get("log_dir", "tb_logs"), "actor_wrapper"
        )
        self.agent_type = (
            self.config.get("agent_trainer", {}).get("agent_type", "PPO").upper()
        )
        self.training = training
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.state_scaler = None

        if not self.training:
            self.model = RandomPolicy(self.env.action_space)
            self.latest_model_path = None

            # Start the background thread to watch for new models
            self.model_watcher_thread = threading.Thread(
                target=self._model_watcher, daemon=True
            )
            self.model_watcher_thread.start()
        else:
            if self.agent_type == "PPO":
                from stable_baselines3 import PPO

                self.model = PPO(
                    "MlpPolicy",
                    self.env,
                    verbose=self.config["agent_trainer"]["verbose"],
                    seed=self.config["global"]["seed"],
                    tensorboard_log=self.tb_log_dir,
                )
            elif self.agent_type == "SAC":
                from stable_baselines3 import SAC

                self.model = SAC(
                    "MlpPolicy",
                    self.env,
                    verbose=self.config["agent_trainer"]["verbose"],
                    seed=self.config["global"]["seed"],
                    tensorboard_log=self.tb_log_dir,
                )
            elif self.agent_type == "DREAMER":
                from agents.dreamer_ac_agent import DreamerACAgent

                self.model = DreamerACAgent(
                    self.config,
                    self.env,
                    tensorboard_log=self.tb_log_dir,
                )
            elif self.agent_type == "EVO":
                from agents.evo_agent import EvoAgent

                self.model = EvoAgent(
                    self.config,
                    self.env,
                    tensorboard_log=self.tb_log_dir,
                )
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

    def learn(
        self,
        total_timesteps: int,
        callback: CallbackList = CallbackList([]),
        progress_bar: bool = False,
    ):
        if self.training:
            self.model.learn(total_timesteps, callback, progress_bar)
        else:
            raise ValueError("Model is not training, cannot learn.")

    def _find_latest_model(self):
        """Finds the latest model .zip"""
        # The Agent trainer saves models in the 'checkpoints' subfolder
        checkpoints_dir = os.path.join(
            os.path.dirname(self.agent_folder), "checkpoints"
        )
        # print(f"[ACTOR-WRAPPER] Checking for latest model in {checkpoints_dir}")
        try:
            files_in_dir = os.listdir(checkpoints_dir)
        except (FileNotFoundError, NotADirectoryError):
            return None

        model_files = [f for f in files_in_dir if f.endswith(".zip")]
        if not model_files:
            return None

        # Determine target device to decide which model version to prefer
        target_device = resolve_device("global", self.config["global"])

        # If target device is CPU, prefer _cpu versions but fall back to regular files
        if target_device == "cpu":
            cpu_files = [f for f in model_files if f.endswith("_cpu.zip")]
            regular_files = [f for f in model_files if not f.endswith("_cpu.zip")]

            # If we have CPU files, use the latest one
            if cpu_files:
                latest_model_file = max(
                    cpu_files,
                    key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
                )
            # Otherwise, use the latest regular file
            elif regular_files:
                latest_model_file = max(
                    regular_files,
                    key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
                )
            # Fall back to any file if no files found
            else:
                latest_model_file = max(
                    model_files,
                    key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
                )
        else:
            # For non-CPU devices, prefer original files over _cpu versions
            original_files = [f for f in model_files if not f.endswith("_cpu.zip")]
            if original_files:
                # Use the latest original file
                latest_model_file = max(
                    original_files,
                    key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
                )
            else:
                # Fall back to any file if no original files found
                latest_model_file = max(
                    model_files,
                    key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
                )

        # Special handling for "old_" prefixed files - prefer non-old files
        # This ensures newly created checkpoints are preferred over copied old ones
        if "old_" in latest_model_file:
            # Look for non-old files with the same pattern
            non_old_files = [f for f in model_files if not f.startswith("old_")]
            if non_old_files:
                # Use the latest non-old file instead
                latest_model_file = max(
                    non_old_files,
                    key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
                )

        latest_model_path = os.path.join(checkpoints_dir, latest_model_file)
        return latest_model_path

    def _check_and_load_new_model(self):
        """Checks for and loads a new actor model if one is found."""
        try:
            new_model_path = self._find_latest_model()

            model_has_changed = (
                new_model_path and new_model_path != self.latest_model_path
            )

            if model_has_changed and new_model_path:
                print(f"[ACTOR-WRAPPER] Found new model: {new_model_path}")

                # Load the new actor model
                print(f"[ACTOR-WRAPPER] Loading new {self.agent_type} actor model...")

                # Determine the target device for loading
                target_device = resolve_device("global", self.config["global"])
                print(f"[ACTOR-WRAPPER] Loading model to device: {target_device}")

                if self.agent_type == "PPO":
                    from stable_baselines3 import PPO

                    new_model = PPO.load(new_model_path, device=target_device)
                elif self.agent_type == "SAC":
                    from stable_baselines3 import SAC

                    new_model = SAC.load(new_model_path, device=target_device)
                elif self.agent_type == "DREAMER":
                    from agents.dreamer_ac_agent import DreamerACAgent

                    # Patch DreamerACAgent.load to support force-cpu (same as visualize_agent.py)
                    import types
                    from agents import dreamer_ac_agent as dreamer_mod

                    orig_load = dreamer_mod.DreamerACAgent.load

                    def patched_load(path, env=None, force_cpu=False):
                        import torch
                        import zipfile, os, tempfile, pickle
                        from datetime import datetime

                        with tempfile.TemporaryDirectory() as temp_dir:
                            with zipfile.ZipFile(path, "r") as zipf:
                                zipf.extractall(temp_dir)
                            training_info_path = os.path.join(
                                temp_dir, "training_info.pkl"
                            )
                            with open(training_info_path, "rb") as f:
                                training_info = pickle.load(f)
                            config = training_info["config"]
                            global_config = training_info["global_config"]
                            if force_cpu:
                                config["device"] = "cpu"
                                global_config["device"] = "cpu"
                            temp_log_dir = os.path.join(
                                tempfile.gettempdir(),
                                f"dreamer_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            )
                            agent = dreamer_mod.DreamerACAgent(
                                global_config, env, tensorboard_log=temp_log_dir
                            )
                            # Always use CPU if forced, else use config["device"]
                            map_location = "cpu" if force_cpu else config["device"]
                            actor_path = os.path.join(temp_dir, "actor.pth")
                            actor_data = torch.load(
                                actor_path,
                                map_location=map_location,
                                weights_only=False,
                            )
                            agent.agent.actor.load_state_dict(
                                actor_data["model_state_dict"]
                            )
                            critic_path = os.path.join(temp_dir, "critic.pth")
                            critic_data = torch.load(
                                critic_path,
                                map_location=map_location,
                                weights_only=False,
                            )
                            agent.agent.critic.load_state_dict(
                                critic_data["model_state_dict"]
                            )
                            agent.episode_rewards = training_info.get(
                                "episode_rewards", []
                            )
                            agent.episode_lengths = training_info.get(
                                "episode_lengths", []
                            )
                            agent.training_losses = training_info.get(
                                "training_losses", []
                            )
                            agent.agent.actor.eval()
                            agent.agent.critic.eval()
                            return agent

                    # Determine if we should force CPU based on target device
                    force_cpu = target_device == "cpu"

                    dreamer_mod.DreamerACAgent.load = staticmethod(
                        lambda path, env=None: patched_load(
                            path, env, force_cpu=force_cpu
                        )
                    )

                    # Use the patched load method
                    new_model = DreamerACAgent.load(new_model_path, env=self.env)

                    # Restore original method
                    dreamer_mod.DreamerACAgent.load = orig_load

                elif self.agent_type == "EVO":
                    from agents.evo_agent import EvoAgent

                    new_model = EvoAgent.load(new_model_path, env=self.env)
                else:
                    print(
                        f"[ACTOR-WRAPPER] Unknown agent type {self.agent_type}, cannot load model."
                    )
                    return
                print("[ACTOR-WRAPPER] Successfully loaded new actor.")

                # check if action scaler exists
                if not os.path.exists(
                    os.path.join(
                        os.path.dirname(os.path.dirname(new_model_path)),
                        "state_scaler.joblib",
                    )
                ):
                    print(
                        f"[ACTOR-WRAPPER] State scaler not found in {new_model_path}. Using default scaler. This is normal for old models."
                    )
                    state_scaler = None
                else:
                    # search for state scaler
                    state_scaler = joblib.load(
                        os.path.join(
                            os.path.dirname(os.path.dirname(new_model_path)),
                            "state_scaler.joblib",
                        )
                    )

                # Safely update the shared actor and environment
                with self.lock:
                    self.model = new_model
                    self.latest_model_path = new_model_path
                    self.state_scaler = state_scaler

        except Exception as e:
            print(f"[ACTOR-WRAPPER] Error loading actor: {e}. Using previous actor.")
            # Don't raise the exception, just continue with previous actor
            return

    def _model_watcher(self):
        """Periodically calls the model checker."""
        interval = self.config["data_generator"]["actor_check_interval_seconds"]
        while not self.stop_event.wait(interval):
            self._check_and_load_new_model()

    def get_model(self):
        """Returns the current model in a thread-safe way."""
        with self.lock:
            return self.model, self.state_scaler

    def close(self):
        """Stops the background model watcher thread."""
        print("[ACTOR-WRAPPER] Stopping model watcher thread...")
        self.stop_event.set()
        self.model_watcher_thread.join(timeout=5)
        if self.model_watcher_thread.is_alive():
            print(
                "[ACTOR-WRAPPER] Warning: Model watcher thread did not shut down cleanly."
            )

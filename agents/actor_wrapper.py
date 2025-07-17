import os
import threading
import numpy as np
from stable_baselines3.common.callbacks import CallbackList

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

        latest_model_file = max(
            model_files,
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
                if self.agent_type == "PPO":
                    from stable_baselines3 import PPO

                    new_model = PPO.load(new_model_path)
                elif self.agent_type == "SAC":
                    from stable_baselines3 import SAC

                    new_model = SAC.load(new_model_path)
                elif self.agent_type == "DREAMER":
                    from agents.dreamer_ac_agent import DreamerACAgent

                    new_model = DreamerACAgent.load(new_model_path, env=self.env)

                elif self.agent_type == "EVO":
                    from agents.evo_agent import EvoAgent

                    new_model = EvoAgent.load(new_model_path, env=self.env)
                else:
                    print(
                        f"[ACTOR-WRAPPER] Unknown agent type {self.agent_type}, cannot load model."
                    )
                    return
                print("[ACTOR-WRAPPER] Successfully loaded new actor.")

                # Safely update the shared actor and environment
                with self.lock:
                    self.model = new_model
                    self.latest_model_path = new_model_path

        except Exception as e:
            print(f"[ACTOR-WRAPPER] Error loading actor: {e}. Using previous actor.")

    def _model_watcher(self):
        """Periodically calls the model checker."""
        interval = self.config["data_generator"]["actor_check_interval_seconds"]
        while not self.stop_event.wait(interval):
            self._check_and_load_new_model()

    def get_model(self):
        """Returns the current model in a thread-safe way."""
        with self.lock:
            return self.model

    def close(self):
        """Stops the background model watcher thread."""
        print("[ACTOR-WRAPPER] Stopping model watcher thread...")
        self.stop_event.set()
        self.model_watcher_thread.join(timeout=5)
        if self.model_watcher_thread.is_alive():
            print(
                "[ACTOR-WRAPPER] Warning: Model watcher thread did not shut down cleanly."
            )

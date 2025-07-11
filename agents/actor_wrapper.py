import os
import threading
from stable_baselines3 import SAC, PPO


class RandomPolicy:
    """A simple policy that returns random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=False):
        return self.action_space.sample().reshape(1, -1), None


class ActorWrapper:
    """
    A wrapper class that manages loading and providing the latest actor model
    in a non-blocking way using a background thread.
    """

    def __init__(self, actor_path: str, action_space, config: dict):
        self.actor_path = actor_path
        self.action_space = action_space
        self.config = config
        self.agent_type = (
            self.config.get("agent_trainer", {}).get("agent_type", "PPO").upper()
        )

        self.actor = RandomPolicy(self.action_space)

        self.latest_model_path = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Start the background thread to watch for new models
        self.model_watcher_thread = threading.Thread(
            target=self._model_watcher, daemon=True
        )
        self.model_watcher_thread.start()

    def _find_latest_model(self):
        """Finds the latest model .zip"""
        # The Agent trainer saves models in the 'checkpoints' subfolder
        checkpoints_dir = os.path.join(self.actor_path, "actor_logs", "checkpoints")
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
                    new_actor = PPO.load(new_model_path)
                elif self.agent_type == "SAC":
                    new_actor = SAC.load(new_model_path)
                else:
                    print(
                        f"[ACTOR-WRAPPER] Unknown agent type {self.agent_type}, cannot load model."
                    )
                    return
                print("[ACTOR-WRAPPER] Successfully loaded new actor.")

                # Safely update the shared actor and environment
                with self.lock:
                    self.actor = new_actor
                    self.latest_model_path = new_model_path

        except Exception as e:
            print(f"[ACTOR-WRAPPER] Error loading actor: {e}. Using previous actor.")

    def _model_watcher(self):
        """Periodically calls the model checker."""
        interval = self.config["data_generator"]["actor_check_interval_seconds"]
        while not self.stop_event.wait(interval):
            self._check_and_load_new_model()

    def get_actor(self):
        """Returns the current actor in a thread-safe way."""
        with self.lock:
            return self.actor

    def close(self):
        """Stops the background model watcher thread."""
        print("[ACTOR-WRAPPER] Stopping model watcher thread...")
        self.stop_event.set()
        self.model_watcher_thread.join(timeout=5)
        if self.model_watcher_thread.is_alive():
            print(
                "[ACTOR-WRAPPER] Warning: Model watcher thread did not shut down cleanly."
            )

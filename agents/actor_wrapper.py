import os
import threading
from stable_baselines3 import SAC, PPO
from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper


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

    def __init__(self, actor_path: str, base_env, config: dict):
        self.actor_path = actor_path
        self.base_env = base_env
        self.config = config
        self.agent_type = (
            self.config.get("agent_trainer", {}).get("agent_type", "PPO").upper()
        )

        self.active_env = self.base_env
        self.actor = RandomPolicy(self.base_env.action_space)

        self.latest_model_path = None
        self.latest_vecnorm_path = None

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Start the background thread to watch for new models
        self.model_watcher_thread = threading.Thread(
            target=self._model_watcher, daemon=True
        )
        self.model_watcher_thread.start()

    def _find_latest_model(self):
        """Finds the latest model .zip and VecNormalize stats .pkl file."""
        # The Agent trainer saves models in the 'checkpoints' subfolder
        checkpoints_dir = os.path.join(self.actor_path, "actor_logs", "checkpoints")
        try:
            files_in_dir = os.listdir(checkpoints_dir)
        except (FileNotFoundError, NotADirectoryError):
            return None, None

        model_files = [f for f in files_in_dir if f.endswith(".zip")]
        if not model_files:
            return None, None

        latest_model_file = max(
            model_files,
            key=lambda f: os.path.getctime(os.path.join(checkpoints_dir, f)),
        )
        latest_model_path = os.path.join(checkpoints_dir, latest_model_file)

        # Find the corresponding VecNormalize file
        model_name_without_ext = os.path.splitext(latest_model_file)[0]
        vec_normalize_path = os.path.join(
            checkpoints_dir, f"{model_name_without_ext}_vecnorm.pkl"
        )

        if not os.path.exists(vec_normalize_path):
            vec_normalize_path = None

        return latest_model_path, vec_normalize_path

    def _check_and_load_new_model(self):
        """Checks for and loads a new actor model if one is found."""
        try:
            new_model_path, new_vecnorm_path = self._find_latest_model()

            model_has_changed = (
                new_model_path and new_model_path != self.latest_model_path
            )

            if model_has_changed and new_model_path:
                print(f"[ACTOR-WRAPPER] Found new model: {new_model_path}")
                env_for_loading = self.base_env

                if new_vecnorm_path:
                    print(
                        f"[ACTOR-WRAPPER] Loading VecNormalize stats from: {new_vecnorm_path}"
                    )
                    loaded_env = wrapper.load(new_vecnorm_path, self.base_env)
                    # Configure the loaded environment
                    if hasattr(loaded_env, "venv") and loaded_env.venv is not None:
                        if hasattr(loaded_env.venv, "render_mode"):
                            loaded_env.venv.render_mode = "human"
                    else:
                        print(
                            "[ACTOR-WRAPPER] Could not set render_mode on the loaded environment."
                        )
                    loaded_env.training = False
                    loaded_env.norm_reward = False
                    env_for_loading = loaded_env

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
                    if new_vecnorm_path:
                        self.active_env = env_for_loading
                    self.latest_model_path = new_model_path
                    self.latest_vecnorm_path = new_vecnorm_path

        except Exception as e:
            print(f"[ACTOR-WRAPPER] Error loading actor: {e}. Using previous actor.")

    def _model_watcher(self):
        """Periodically calls the model checker."""
        interval = self.config["data_generator"]["actor_check_interval_seconds"]
        while not self.stop_event.wait(interval):
            self._check_and_load_new_model()

    def get_actor_and_env(self):
        """Returns the current actor and active environment in a thread-safe way."""
        with self.lock:
            return self.actor, self.active_env

    def close(self):
        """Stops the background model watcher thread."""
        print("[ACTOR-WRAPPER] Stopping model watcher thread...")
        self.stop_event.set()
        self.model_watcher_thread.join(timeout=5)
        if self.model_watcher_thread.is_alive():
            print(
                "[ACTOR-WRAPPER] Warning: Model watcher thread did not shut down cleanly."
            )

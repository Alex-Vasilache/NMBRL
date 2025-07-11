import subprocess
import time
import os
import threading
import datetime
import shutil
import yaml  # Import the YAML library
from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper


def stream_watcher(identifier, stream):
    """Monitors a stream and prints its output with an identifier."""
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        print(f"[{identifier}] {line}", end="")
    stream.close()


def run_system():
    """
    Creates a unique folder and runs the full dynamic training system:
    1. Data Generator
    2. World Model Trainer
    3. SAC Agent Trainer
    Waits for the SAC agent to finish, then stops the other processes.
    """
    # --- Load configuration from YAML file ---
    config_path = "configs/full_system_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get system run configurations
    run_config = config["run_system"]
    create_new_consoles = run_config.get("create_new_consoles", True)

    # 1. Create unique folder for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(config["global"]["run_folder_prefix"], timestamp)
    world_model_folder = os.path.join(run_folder, "world_model_data")
    os.makedirs(world_model_folder, exist_ok=True)
    print(f"--- Created unique folder for this run: {run_folder} ---")
    print(f"--- World model data will be stored in: {world_model_folder} ---")

    stop_file_path = os.path.join(world_model_folder, "stop_signal.tmp")

    # --- Get environment dimensions ---
    print("--- Getting environment dimensions ---")
    sim_env = wrapper(seed=config["global"]["seed"], n_envs=1)
    sim_env.reset()  # This is crucial to initialize the spaces
    if sim_env.observation_space is None or sim_env.action_space is None:
        raise RuntimeError("Environment spaces were not initialized correctly.")
    state_size = sim_env.observation_space.shape[0]
    action_size = sim_env.action_space.shape[0]
    print(f"State size: {state_size}, Action size: {action_size}")
    sim_env.close()

    # --- Commands to run ---
    # All scripts now receive the path to the config file
    generator_command = [
        "python",
        "-u",
        "learning/dynamic_data_generator.py",
        "--save-folder",
        world_model_folder,
        "--config",
        config_path,
    ]
    world_model_trainer_command = [
        "python",
        "-u",
        "learning/dynamic_train_world_model.py",
        "--save-folder",
        world_model_folder,
        "--state-size",
        str(state_size),
        "--action-size",
        str(action_size),
        "--config",
        config_path,
    ]
    sac_trainer_command = [
        "python",
        "-u",
        "learning/dynamic_train_sac_cartpole.py",
        "--world-model-folder",
        world_model_folder,
        "--config",
        config_path,
    ]

    generator_proc = None
    world_model_trainer_proc = None
    sac_trainer_proc = None
    threads = []

    # --- Setup subprocess arguments based on console config ---
    popen_kwargs = {}
    if create_new_consoles and os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        popen_kwargs.update(
            {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
                "universal_newlines": True,
            }
        )
    if os.name != "nt":
        popen_kwargs["preexec_fn"] = os.setsid

    try:
        # 2. Start all three processes
        print("\n--- Starting Data Generator ---")
        generator_proc = subprocess.Popen(generator_command, **popen_kwargs)

        print("--- Starting World Model Trainer ---")
        world_model_trainer_proc = subprocess.Popen(
            world_model_trainer_command, **popen_kwargs
        )

        # Give the other two a head start to generate some data
        head_start = run_config["head_start_seconds"]
        print(
            f"--- Giving data generator and world model a {head_start}-second head start... ---"
        )
        time.sleep(head_start)

        print("\n--- Starting SAC Agent Trainer ---")
        sac_trainer_proc = subprocess.Popen(sac_trainer_command, **popen_kwargs)

        # --- Start stream watchers if not using new consoles ---
        if not create_new_consoles:
            threads = [
                threading.Thread(
                    target=stream_watcher, args=("GEN-STDOUT", generator_proc.stdout)
                ),
                threading.Thread(
                    target=stream_watcher, args=("GEN-STDERR", generator_proc.stderr)
                ),
                threading.Thread(
                    target=stream_watcher,
                    args=("WM-TRAIN-STDOUT", world_model_trainer_proc.stdout),
                ),
                threading.Thread(
                    target=stream_watcher,
                    args=("WM-TRAIN-STDERR", world_model_trainer_proc.stderr),
                ),
                threading.Thread(
                    target=stream_watcher,
                    args=("SAC-TRAIN-STDOUT", sac_trainer_proc.stdout),
                ),
                threading.Thread(
                    target=stream_watcher,
                    args=("SAC-TRAIN-STDERR", sac_trainer_proc.stderr),
                ),
            ]
            for t in threads:
                t.start()
            print("--- Output streaming to this console. ---")

        # 3. Wait for the main SAC training process to complete
        print(
            "\n--- Processes started. Waiting for SAC Agent Trainer to complete... ---\n"
        )
        # The main script's console will now wait here. The other processes
        # will continue to run in their own windows.
        sac_trainer_proc.wait()
        print("\n--- SAC Agent Trainer finished. ---")

    finally:
        print("\n--- Initiating shutdown sequence ---")

        # 4. Signal generator to stop and terminate world model trainer
        if generator_proc and generator_proc.poll() is None:
            print(
                f"--- Stopping data generator by creating stop file: {stop_file_path} ---"
            )
            with open(stop_file_path, "w") as f:
                pass

        if world_model_trainer_proc and world_model_trainer_proc.poll() is None:
            print("--- Terminating world model trainer ---")
            world_model_trainer_proc.terminate()

        # 5. Wait for termination
        if generator_proc:
            print("--- Waiting for data generator to stop... ---")
            try:
                generator_proc.wait(timeout=20)
                print("--- Data generator stopped gracefully. ---")
            except subprocess.TimeoutExpired:
                print(
                    "--- Data generator did not stop gracefully. Force terminating. ---"
                )
                generator_proc.kill()

        if world_model_trainer_proc:
            print("--- Waiting for world model trainer to terminate... ---")
            try:
                world_model_trainer_proc.wait(timeout=10)
                print("--- World model trainer terminated. ---")
            except subprocess.TimeoutExpired:
                print("--- Trainer did not terminate. Force killing. ---")
                world_model_trainer_proc.kill()

        # Join threads to capture all output if they were started
        if threads:
            print("--- Joining output threads... ---")
            for t in threads:
                t.join()

        print("--- All processes stopped ---")

        # 6. Check for results
        print("\n--- Final Run Summary ---")
        print(f"Run folder: {run_folder}")

        world_model_path = os.path.join(world_model_folder, "model.pth")
        if os.path.exists(world_model_path):
            print(f"INFO: Final world model saved at: {world_model_path}")
        else:
            print("WARNING: World model file was not found.")

        # Check for the SAC model in the new location
        sac_model_dir = os.path.join(world_model_folder, "actor_models", "best_model")
        try:
            model_files = [f for f in os.listdir(sac_model_dir) if f.endswith(".zip")]
            if model_files:
                print(f"INFO: SAC agent best model saved in: {sac_model_dir}")
            else:
                print("WARNING: Could not find a saved SAC agent model (.zip).")
        except FileNotFoundError:
            print("WARNING: SAC agent model directory was not found.")

        print("\n--- Full system run finished ---")


if __name__ == "__main__":
    run_system()

import subprocess
import time
import os
import threading
import datetime
import shutil
from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from world_models.dynamic_world_model_wrapper import WorldModelWrapper


def stream_watcher(identifier, stream):
    """Monitors a stream and prints its output with an identifier."""
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        print(f"[{identifier}] {line}", end="")
    stream.close()


def run_test():
    """
    Creates a unique folder, starts the data generator and world model trainer,
    waits, stops them, and cleans up.
    """
    # 1. Create unique folder for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"temp_test_run_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"--- Created unique folder for this run: {save_folder} ---")

    stop_file_path = os.path.join(save_folder, "stop_signal.tmp")
    model_path = os.path.join(save_folder, "model.pth")
    buffer_path = os.path.join(save_folder, "buffer.pkl")

    # --- Get environment dimensions ---
    print("--- Getting environment dimensions ---")
    sim_env = wrapper(seed=42, n_envs=1)
    sim_env.reset()  # This is crucial to initialize the spaces
    state_size = sim_env.observation_space.shape[0]
    action_size = sim_env.action_space.shape[0]
    print(f"State size: {state_size}, Action size: {action_size}")

    # --- Initialize World Model Wrapper for testing ---
    # This will run in the main process, but we won't step through it.
    # We just want to ensure it can be created and closed.
    world_model_env = WorldModelWrapper(
        observation_space=sim_env.observation_space,
        action_space=sim_env.action_space,
        trained_folder=save_folder,
    )

    # --- Commands to run ---
    generator_command = [
        "python",
        "-u",
        "learning/dynamic_data_generator.py",
        "--save-folder",
        save_folder,
    ]
    trainer_command = [
        "python",
        "-u",
        "learning/dynamic_train_world_model.py",
        "--save-folder",
        save_folder,
        "--state-size",
        str(state_size),
        "--action-size",
        str(action_size),
    ]

    generator_proc = None
    trainer_proc = None

    try:
        # 2. Start both processes
        print("--- Starting Data Generator ---")
        generator_proc = subprocess.Popen(
            generator_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        print("--- Starting World Model Trainer ---")
        trainer_proc = subprocess.Popen(
            trainer_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        # --- Start stream watchers ---
        threads = [
            threading.Thread(
                target=stream_watcher, args=("GEN-STDOUT", generator_proc.stdout)
            ),
            threading.Thread(
                target=stream_watcher, args=("GEN-STDERR", generator_proc.stderr)
            ),
            threading.Thread(
                target=stream_watcher, args=("TRAIN-STDOUT", trainer_proc.stdout)
            ),
            threading.Thread(
                target=stream_watcher, args=("TRAIN-STDERR", trainer_proc.stderr)
            ),
        ]
        for t in threads:
            t.start()

        # 3. Let them run for a while
        print("\n--- Processes started. Running for 90 seconds... ---\n")
        time.sleep(90)

    finally:
        print("\n--- Initiating shutdown sequence ---")

        # 4. Signal generator to stop and terminate trainer
        if generator_proc and generator_proc.poll() is None:
            print(
                f"--- Stopping data generator by creating stop file: {stop_file_path} ---"
            )
            with open(stop_file_path, "w") as f:
                pass

        if trainer_proc and trainer_proc.poll() is None:
            print("--- Terminating world model trainer ---")
            trainer_proc.terminate()

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

        if trainer_proc:
            print("--- Waiting for world model trainer to terminate... ---")
            try:
                trainer_proc.wait(timeout=10)
                print("--- World model trainer terminated. ---")
            except subprocess.TimeoutExpired:
                print("--- Trainer did not terminate. Force killing. ---")
                trainer_proc.kill()

        # Join threads to capture all output
        print("--- Joining output threads... ---")
        for t in threads:
            t.join()
        print("--- All processes and threads stopped ---")

        # 6. Check for results
        print("\n--- Final Checks ---")
        if os.path.exists(model_path):
            print(f"SUCCESS: Model file was created at: {model_path}")
        else:
            print(f"FAILURE: Model file was NOT created at: {model_path}")

        if os.path.exists(buffer_path) and os.path.getsize(buffer_path) > 0:
            print(f"INFO: Buffer file contains leftover data: {buffer_path}")
        else:
            print("INFO: Buffer file is empty or was removed.")

        # 7. Clean up
        print(f"--- Cleaning up test folder: {save_folder} ---")
        # Clean up the environments
        sim_env.close()
        world_model_env.close()
        shutil.rmtree(save_folder, ignore_errors=True)

        print("\n--- Dynamic training test finished ---")


if __name__ == "__main__":
    run_test()

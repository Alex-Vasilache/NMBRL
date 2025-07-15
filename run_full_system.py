import subprocess
import time
import os
import threading
import datetime
import yaml  # Import the YAML library

from utils.tools import seed_everything, save_config_to_shared_folder

# Import keyboard library for sending keystrokes (install with: pip install keyboard)
try:
    import keyboard

    KEYBOARD_AVAILABLE = True
except ImportError:
    print("WARNING: keyboard library not available. Install with: pip install keyboard")
    KEYBOARD_AVAILABLE = False


def stream_watcher(identifier, stream):
    """Monitors a stream and prints its output with an identifier."""
    if stream is None:
        return
    for line in iter(stream.readline, ""):
        print(f"[{identifier}] {line}", end="")
    stream.close()


def control_stream_watcher(identifier, stream):
    """Special stream watcher for control.py that sends shift+k when detecting firmware text."""
    print(f"[{identifier}] Stream watcher started")
    if stream is None:
        print(f"[{identifier}] ERROR: Stream is None!")
        return

    # Import regex for ANSI escape sequence removal
    import re

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    line_count = 0
    for line in iter(stream.readline, ""):
        line_count += 1

        # Print the raw line first (to see carriage returns and other control chars)
        print(f"[{identifier}] Raw line {line_count}: {repr(line)}")

        # Split line on carriage returns to handle overwritten content
        parts = line.split("\r")

        for i, part in enumerate(parts):
            if part.strip():  # Only process non-empty parts
                # Remove ANSI escape sequences first
                part_clean = ansi_escape.sub("", part).strip()

                if part_clean:
                    print(f"[{identifier}] Part {i+1} (cleaned): {part_clean}")

                    # Check for firmware pattern (case insensitive)
                    part_lower = part_clean.lower()

                    # Look specifically for the exact pattern we know is output
                    if ("firmware" in part_lower) and KEYBOARD_AVAILABLE:
                        print(
                            f"[{identifier}] *** DETECTED FIRMWARE CONTROLLER: '{part_clean}' - SENDING SHIFT+K ***"
                        )
                        try:
                            keyboard.send("k")
                            keyboard.send("shift+k")
                            print(f"[{identifier}] Shift+K sent successfully")
                        except Exception as e:
                            print(f"[{identifier}] Error sending keystroke: {e}")
                    # Debug output for any potentially relevant content

    print(f"[{identifier}] Stream ended after {line_count} lines")
    stream.close()


def run_system():
    """
    Creates a unique folder and runs the full dynamic training system:
    1. Data Generator
    2. World Model Trainer
    3. Agent Trainer
    4. Control Script (for physical cartpole only)
    Waits for the agent to finish, then stops the other processes.
    """
    # --- Load configuration from YAML file ---
    config_path = "configs/full_system_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # seed everything
    seed_everything(config["global"]["seed"])

    # Get system run configurations
    run_config = config["run_system"]
    create_new_consoles = run_config.get("create_new_consoles", True)

    # 1. Create unique folder for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_folder = os.path.join(config["global"]["run_folder_prefix"], timestamp)
    os.makedirs(shared_folder, exist_ok=True)
    print(f"--- Created unique folder for this run: {shared_folder} ---")

    # Save the config to the shared folder for reproducibility
    save_config_to_shared_folder(config, config_path, shared_folder, "full_system")

    stop_file_path = os.path.join(shared_folder, "stop_signal.tmp")

    # --- Get environment dimensions ---
    print("--- Getting environment dimensions ---")
    if config["global"]["env_type"] == "dmc":
        state_size = 5
        action_size = 1
    elif config["global"]["env_type"] == "physical":
        state_size = 6
        action_size = 1
    else:
        raise ValueError(f"Invalid environment type: {config['global']['env_type']}")

    print(f"State size: {state_size}, Action size: {action_size}")

    # --- Commands to run ---
    # All scripts now receive the path to the config file
    generator_command = [
        "python",
        "-u",
        "learning/dynamic_data_generator.py",
        "--shared-folder",
        shared_folder,
        "--config",
        config_path,
        "--env-type",
        config["global"]["env_type"],
    ]
    world_model_trainer_command = [
        "python",
        "-u",
        "learning/dynamic_train_world_model.py",
        "--shared-folder",
        shared_folder,
        "--state-size",
        str(state_size),
        "--action-size",
        str(action_size),
        "--config",
        config_path,
        "--env-type",
        config["global"]["env_type"],
    ]
    agent_trainer_command = [
        "python",
        "-u",
        "learning/dynamic_train_agent.py",
        "--shared-folder",
        shared_folder,
        "--config",
        config_path,
        "--env-type",
        config["global"]["env_type"],
    ]

    # Add control command for physical cartpole
    control_command = None
    control_cwd = None
    if config["global"]["env_type"] == "physical":
        control_cwd = os.path.join("environments", "physical-cartpole")
        control_command = [
            "python",
            "-u",  # Unbuffered stdout and stderr
            "-B",  # Don't write .pyc files
            os.path.join("Driver", "control.py"),
        ]

    generator_proc = None
    world_model_trainer_proc = None
    agent_trainer_proc = None
    control_proc = None
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
        # 2. Start control process first if using physical cartpole
        if control_command is not None:
            print("\n--- Starting Physical Cartpole Control Script ---")
            print(f"Command: {' '.join(control_command)}")
            print(f"Working directory: {control_cwd}")

            # Create a copy of popen_kwargs and add the working directory
            control_popen_kwargs = popen_kwargs.copy()
            control_popen_kwargs["cwd"] = control_cwd

            try:
                control_proc = subprocess.Popen(control_command, **control_popen_kwargs)
                print(f"Control process started with PID: {control_proc.pid}")

                # Give control script a moment to initialize
                time.sleep(2)

                # Check if process is still running
                if control_proc.poll() is None:
                    print("Control process is running")
                else:
                    print(
                        f"WARNING: Control process exited with code: {control_proc.returncode}"
                    )

            except Exception as e:
                print(f"ERROR: Failed to start control process: {e}")
                control_proc = None

        # 3. Start the other three processes
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

        print("\n--- Starting Agent Trainer ---")
        agent_trainer_proc = subprocess.Popen(agent_trainer_command, **popen_kwargs)

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
                    args=("AGENT-TRAIN-STDOUT", agent_trainer_proc.stdout),
                ),
                threading.Thread(
                    target=stream_watcher,
                    args=("AGENT-TRAIN-STDERR", agent_trainer_proc.stderr),
                ),
            ]

            # Add control process stream watchers if control process exists
            if control_proc is not None:
                print("Adding control process stream watchers...")
                control_threads = [
                    threading.Thread(
                        target=control_stream_watcher,
                        args=("CONTROL-STDOUT", control_proc.stdout),
                        name="control-stdout-watcher",
                    ),
                    threading.Thread(
                        target=control_stream_watcher,
                        args=("CONTROL-STDERR", control_proc.stderr),
                        name="control-stderr-watcher",
                    ),
                ]
                threads.extend(control_threads)
                print(f"Added {len(control_threads)} control stream watchers")

            for t in threads:
                t.start()
            print("--- Output streaming to this console. ---")

        # 4. Wait for the main training process to complete
        print("\n--- Processes started. Waiting for Agent Trainer to complete... ---\n")
        try:
            # Poll the process to see if it has finished. This non-blocking
            # wait allows the main script to catch the KeyboardInterrupt.
            while agent_trainer_proc.poll() is None:
                time.sleep(1)

            # If the loop exits, the process has finished.
            print("\n--- Agent Trainer finished. ---")
        except KeyboardInterrupt:
            print("\n--- Ctrl-C detected. Initiating shutdown... ---")
            # The 'finally' block will handle the cleanup.

    finally:
        print("\n--- Initiating shutdown sequence ---")

        # 5. Signal all processes to stop
        if generator_proc and generator_proc.poll() is None:
            print(
                f"--- Stopping data generator by creating stop file: {stop_file_path} ---"
            )
            with open(stop_file_path, "w") as f:
                pass  # Create the stop file

        if world_model_trainer_proc and world_model_trainer_proc.poll() is None:
            print("--- Terminating world model trainer ---")
            world_model_trainer_proc.terminate()

        if agent_trainer_proc and agent_trainer_proc.poll() is None:
            print("--- Terminating Agent trainer ---")
            agent_trainer_proc.terminate()

        if control_proc and control_proc.poll() is None:
            print("--- Terminating control script ---")
            control_proc.terminate()

        # 6. Wait for termination
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

        if agent_trainer_proc:
            print("--- Waiting for Agent trainer to terminate... ---")
            try:
                agent_trainer_proc.wait(timeout=10)
                print("--- Agent trainer terminated. ---")
            except subprocess.TimeoutExpired:
                print("--- Agent trainer did not terminate. Force killing. ---")
                agent_trainer_proc.kill()

        if control_proc:
            print("--- Waiting for control script to terminate... ---")
            try:
                control_proc.wait(timeout=10)
                print("--- Control script terminated. ---")
            except subprocess.TimeoutExpired:
                print("--- Control script did not terminate. Force killing. ---")
                control_proc.kill()

        # Join threads to capture all output if they were started
        if threads:
            print("--- Joining output threads... ---")
            for t in threads:
                t.join()

        print("--- All processes stopped ---")

        # 7. Check for results
        print("\n--- Final Run Summary ---")
        print(f"Run folder: {shared_folder}")

        world_model_path = os.path.join(shared_folder, "model.pth")
        if os.path.exists(world_model_path):
            print(f"INFO: Final world model saved at: {world_model_path}")
        else:
            print("WARNING: World model file was not found.")

        # Check for the agent model in the new location
        agent_model_dir = os.path.join(shared_folder, "actor_models", "best_model")
        try:
            model_files = [f for f in os.listdir(agent_model_dir) if f.endswith(".zip")]
            if model_files:
                print(f"INFO: Agent best model saved in: {agent_model_dir}")
            else:
                print("WARNING: Could not find a saved Agent model (.zip).")
        except FileNotFoundError:
            print("WARNING: Agent model directory was not found.")

        print("\n--- Full system run finished ---")


if __name__ == "__main__":
    run_system()

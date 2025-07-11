import numpy as np
import os
import pickle
from typing import Any
import multiprocessing
import portalocker
import queue  # For queue.Empty exception
import argparse
import yaml
from agents.actor_wrapper import ActorWrapper
from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper


def write_to_buffer(buffer_path: str, data: Any) -> None:
    """Write local buffer to file with file locking to prevent race conditions."""
    with portalocker.Lock(buffer_path, "ab+") as f:
        pickle.dump(data, f)


def offload_env_data(
    data_queue: multiprocessing.Queue, inp_data: Any, outp_data: Any
) -> None:
    """Offload data to the shared queue. Must not block. Returns immediately."""
    data_queue.put((inp_data.copy(), outp_data.copy()))


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=False):
        return self.action_space.sample().reshape(1, -1), None


def find_latest_model(actor_dir: str):
    """Finds the latest model .zip file and the VecNormalize stats .pkl file."""

    # The SAC trainer saves the best model in a specific subfolder
    best_model_dir = os.path.join(actor_dir, "actor_logs", "checkpoints")
    try:
        files_in_dir = os.listdir(best_model_dir)
    except (FileNotFoundError, NotADirectoryError):
        return None, None

    model_files = [
        os.path.join(best_model_dir, f) for f in files_in_dir if f.endswith(".zip")
    ]

    if not model_files:
        if not files_in_dir:
            print(
                f"[GENERATOR] Info: Actor directory is empty. Waiting for a model to be saved in '{best_model_dir}'"
            )
        else:
            print(
                f"[GENERATOR] Info: No models (.zip files) found in '{best_model_dir}'."
            )
        return None, None

    latest_model = max(model_files, key=os.path.getctime)

    # Find the corresponding VecNormalize file if it exists
    model_name_without_ext = os.path.splitext(os.path.basename(latest_model))[0]
    vec_normalize_path = os.path.join(
        best_model_dir, f"{model_name_without_ext}_vecnorm.pkl"
    )

    if not os.path.exists(vec_normalize_path):
        vec_normalize_path = None

    return latest_model, vec_normalize_path


def buffer_writer_process(stop_event, data_queue, buffer_path: str, write_interval=1):
    """Periodically writes the contents of the data queue to the file buffer."""

    def write_data():
        if not data_queue.empty():
            data_to_write = []
            while not data_queue.empty():
                try:
                    data_to_write.append(data_queue.get_nowait())
                except queue.Empty:
                    break

            if data_to_write:
                print(f"[GENERATOR] Writing {len(data_to_write)} items to buffer...")
                write_to_buffer(buffer_path, data_to_write)

    while not stop_event.wait(timeout=write_interval):
        write_data()

    # One last write after the loop exits to flush any remaining items
    print("[GENERATOR] Writer process stopping. Performing final flush.")
    write_data()


def main(stop_event, data_queue, actor_path: str, stop_file_path: str, config: dict):
    # Create the base environment that the ActorWrapper will manage
    base_env = wrapper(
        seed=config["global"]["seed"],
        n_envs=1,
        render_mode="human",
        max_episode_steps=config["data_generator"]["max_episode_steps"],
    )

    # Instantiate the actor wrapper
    actor_wrapper = ActorWrapper(
        actor_path=actor_path, base_env=base_env, config=config
    )

    # Initialize state and data buffers from the initial environment
    env = base_env
    state = env.reset()
    state_size = env.observation_space.shape[0]

    inp_data = np.zeros((state_size + 1))
    outp_data = np.zeros((state_size + 2))

    episode_reward = 0
    episode_length = 0
    episode_count = 0

    try:
        while not stop_event.is_set():
            if os.path.exists(stop_file_path):
                print("[GENERATOR] Stop file detected. Shutting down.")
                stop_event.set()
                continue

            # Get the latest actor and environment from the wrapper
            actor, env = actor_wrapper.get_actor_and_env()

            action, _ = actor.predict(state, deterministic=False)

            inp_data[:state_size] = state[0]
            inp_data[state_size] = action[0, 0]
            next_state, reward, terminated, info = env.step(action)

            outp_data[:state_size] = next_state[0]
            outp_data[state_size] = reward[0]
            outp_data[state_size + 1] = terminated[0]
            state = next_state

            env.render()

            sim_should_stop = info[0].get("sim_should_stop", False)
            if sim_should_stop:
                print("[GENERATOR] Render window closed by user, stopping generation.")
                stop_event.set()

            episode_reward += reward[0]
            episode_length += 1

            if terminated[0]:
                print(
                    f"[GENERATOR] Episode {episode_count+1} finished. Reward: {episode_reward}, Length: {episode_length}"
                )
                state = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1

            offload_env_data(data_queue, inp_data, outp_data)
    finally:
        # Ensure the actor wrapper's thread is cleaned up
        actor_wrapper.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dynamic data generator.")
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Path to the folder where data will be saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    save_folder = args.save_folder
    buffer_path = os.path.join(save_folder, "buffer.pkl")
    # The actor path is now the same as the save_folder
    actor_path = save_folder
    stop_file_path = os.path.join(save_folder, "stop_signal.tmp")

    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
    os.makedirs(actor_path, exist_ok=True)

    # Clean up stop file from previous runs at startup
    if os.path.exists(stop_file_path):
        os.remove(stop_file_path)

    stop_event = multiprocessing.Event()
    data_queue = multiprocessing.Queue()

    # Get config for the writer process
    writer_config = config["data_generator"]
    writer_proc = multiprocessing.Process(
        target=buffer_writer_process,
        args=(
            stop_event,
            data_queue,
            buffer_path,
            writer_config["buffer_write_interval_seconds"],
        ),
    )
    writer_proc.start()

    try:
        main(stop_event, data_queue, actor_path, stop_file_path, config)
    except KeyboardInterrupt:
        print("[GENERATOR] Stopping data generation.")
    finally:
        stop_event.set()
        writer_proc.join()
        # Clean up stop file on exit
        if os.path.exists(stop_file_path):
            os.remove(stop_file_path)
        print("[GENERATOR] Data generation stopped.")

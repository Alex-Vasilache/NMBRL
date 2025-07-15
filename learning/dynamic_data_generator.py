import numpy as np
import os
import pickle
from typing import Any
import multiprocessing
import portalocker
import queue  # For queue.Empty exception
import argparse
import yaml
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from agents.actor_wrapper import ActorWrapper
from utils.tools import seed_everything, save_config_to_shared_folder


def write_to_buffer(buffer_path: str, data: Any) -> None:
    """Write local buffer to file with file locking to prevent race conditions."""
    with portalocker.Lock(buffer_path, "ab+") as f:
        pickle.dump(data, f)


def offload_env_data(
    data_queue: multiprocessing.Queue, inp_data: Any, outp_data: Any
) -> None:
    """Offload data to the shared queue. Must not block. Returns immediately."""
    data_queue.put((inp_data.copy(), outp_data.copy()))


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


def main(stop_event, data_queue, shared_folder: str, stop_file_path: str, config: dict):
    # Setup TensorBoard logging
    tb_config = config.get("tensorboard", {})
    tb_log_dir = os.path.join(shared_folder, tb_config.get("log_dir", "tb_logs"))
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=tb_log_dir, flush_secs=tb_config.get("flush_seconds", 30)
    )
    log_frequency = tb_config.get("log_frequency", 10)

    print(f"[GENERATOR] TensorBoard logging to: {tb_log_dir}")

    # Create the base environment that the ActorWrapper will manage
    if args.env_type == "dmc":
        from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
    elif args.env_type == "physical":
        from world_models.physical_cartpole_wrapper import (
            PhysicalCartpoleWrapper as wrapper,
        )
    base_env = wrapper(
        seed=config["global"]["seed"],
        n_envs=1,
        max_episode_steps=config["data_generator"]["max_episode_steps"],
    )

    # Instantiate the actor wrapper
    actor_wrapper = ActorWrapper(
        env=base_env, config=config, training=False, shared_folder=shared_folder
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
    total_steps = 0
    data_generation_start_time = time.time()
    last_log_time = time.time()

    try:
        while not stop_event.is_set():
            if os.path.exists(stop_file_path):
                print("[GENERATOR] Stop file detected. Shutting down.")
                stop_event.set()
                continue

            # Get the latest actor and environment from the wrapper
            actor_model = actor_wrapper.get_model()

            # Extract state from vectorized environment (first and only environment)
            # For single environment, handle vectorized environment observation format
            try:
                # Vectorized environments return observations as arrays/tuples
                if isinstance(state, (tuple, list)) and len(state) > 0:
                    state_obs = (
                        np.array(state[0])
                        if not isinstance(state[0], np.ndarray)
                        else state[0]
                    )
                elif isinstance(state, np.ndarray):
                    state_obs = state
                else:
                    state_obs = np.array(state)

                action, _ = actor_model.predict(state_obs, deterministic=False)
            except Exception as e:
                print(f"Error processing state observation: {e}")
                print(f"State type: {type(state)}, State: {state}")
                continue
            inp_data[:state_size] = state_obs
            inp_data[state_size] = action[0, 0]
            next_state, reward, terminated, info = env.step(action)

            # Extract next state from vectorized environment
            next_state_obs = (
                next_state[0] if isinstance(next_state, tuple) else next_state
            )
            outp_data[:state_size] = next_state_obs
            outp_data[state_size] = reward[0]
            outp_data[state_size + 1] = terminated[0]
            state = next_state

            env.render()

            # sim_should_stop = info[0].get("sim_should_stop", False)
            # if sim_should_stop:
            #     print("[GENERATOR] Render window closed by user, stopping generation.")
            #     stop_event.set()

            episode_reward += reward[0]
            episode_length += 1
            total_steps += 1

            if terminated[0]:
                print(
                    f"[GENERATOR] Episode {episode_count+1} finished. Reward: {episode_reward}, Length: {episode_length}"
                )

                # Log episode statistics to TensorBoard
                writer.add_scalar(
                    "DataGen/Episode_Reward", episode_reward, episode_count
                )
                writer.add_scalar("DataGen/Total_Steps", total_steps, episode_count)

                state = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1

            # Log periodic statistics
            current_time = time.time()
            if current_time - last_log_time >= log_frequency:
                data_generation_rate = total_steps / (
                    current_time - data_generation_start_time
                )
                writer.add_scalar(
                    "DataGen/Steps_Per_Second", data_generation_rate, total_steps
                )
                writer.add_scalar("DataGen/Total_Episodes", episode_count, total_steps)

                last_log_time = current_time

            offload_env_data(data_queue, inp_data, outp_data)
    finally:
        # Ensure the actor wrapper's thread is cleaned up
        actor_wrapper.close()
        # Close TensorBoard writer
        writer.close()
        print(f"[GENERATOR] TensorBoard logs saved to: {tb_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dynamic data generator.")
    parser.add_argument(
        "--shared-folder",
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
    parser.add_argument(
        "--env-type",
        type=str,
        required=False,
        help="Type of environment to use. Options: 'dmc', 'physical'.",
        default="dmc",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # seed everything
    seed_everything(config["global"]["seed"])

    shared_folder = args.shared_folder
    buffer_path = os.path.join(shared_folder, "buffer.pkl")
    stop_file_path = os.path.join(shared_folder, "stop_signal.tmp")

    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

    # Save config to shared folder for reproducibility
    save_config_to_shared_folder(config, args.config, shared_folder, "data_generator")

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
        main(stop_event, data_queue, shared_folder, stop_file_path, config)
    except KeyboardInterrupt:
        print("[GENERATOR] Stopping data generation.")
    finally:
        stop_event.set()
        writer_proc.join()
        # Clean up stop file on exit
        if os.path.exists(stop_file_path):
            os.remove(stop_file_path)
        print("[GENERATOR] Data generation stopped.")

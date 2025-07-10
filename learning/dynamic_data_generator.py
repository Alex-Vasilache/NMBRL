import numpy as np
import torch
import os
import pickle
import time
from typing import Any
import multiprocessing
import portalocker
import queue  # For queue.Empty exception
import argparse


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

    def predict(self, obs):
        return self.action_space.sample().reshape(1, -1)


def get_latest_actor(action_space, actor_path: str) -> Any:
    """Get latest trained actor from local buffer. If no actor is found, return random Policy."""
    if os.path.exists(actor_path):
        try:
            # Assuming the actor is a PyTorch model
            return torch.load(actor_path)
        except Exception as e:
            print(f"Could not load actor, using random policy. Error: {e}")
            return RandomPolicy(action_space)
    return RandomPolicy(action_space)


def buffer_writer_process(stop_event, data_queue, buffer_path: str, write_interval=5):
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
                print(f"Writing {len(data_to_write)} items to buffer...")
                write_to_buffer(buffer_path, data_to_write)

    while not stop_event.wait(timeout=write_interval):
        write_data()

    # One last write after the loop exits to flush any remaining items
    print("Writer process stopping. Performing final flush.")
    write_data()


def main(stop_event, data_queue, actor_path: str, stop_file_path: str):
    from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
    from gymnasium.spaces import Box

    max_episode_steps = 10000

    env = wrapper(
        seed=42, n_envs=1, render_mode="human", max_episode_steps=max_episode_steps
    )

    state = env.reset()
    state_size = env.observation_space.shape[0]

    inp_data = np.zeros((state_size + 1))
    outp_data = np.zeros((state_size + 2))

    episode_reward = 0
    episode_length = 0
    episode_count = 0

    while not stop_event.is_set():
        if os.path.exists(stop_file_path):
            print("Stop file detected. Shutting down.")
            stop_event.set()
            continue

        actor = get_latest_actor(env.action_space, actor_path)

        action = actor.predict(state)

        inp_data[:state_size] = state[0]
        inp_data[state_size] = action[0, 0]
        next_state, reward, terminated, info = env.step(action)

        outp_data[:state_size] = next_state[0]
        outp_data[state_size] = reward[0]
        outp_data[state_size + 1] = terminated[0]
        state = next_state

        env.render()

        if info[0].get("sim_should_stop", False):
            print("Render window closed by user, stopping generation.")
            stop_event.set()

        episode_reward += reward[0]
        episode_length += 1

        if terminated[0]:
            print(
                f"Episode {episode_count+1} finished. Reward: {episode_reward}, Length: {episode_length}"
            )
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1

        offload_env_data(data_queue, inp_data, outp_data)
        time.sleep(0.01)  # Small sleep to avoid busy-waiting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dynamic data generator.")
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Path to the folder where data will be saved.",
    )
    args = parser.parse_args()

    save_folder = args.save_folder
    buffer_path = os.path.join(save_folder, "buffer.pkl")
    actor_path = os.path.join(save_folder, "actor", "latest.pth")
    stop_file_path = os.path.join(save_folder, "stop_signal.tmp")

    os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
    os.makedirs(os.path.dirname(actor_path), exist_ok=True)

    # Clean up stop file from previous runs at startup
    if os.path.exists(stop_file_path):
        os.remove(stop_file_path)

    stop_event = multiprocessing.Event()
    data_queue = multiprocessing.Queue()

    writer_proc = multiprocessing.Process(
        target=buffer_writer_process, args=(stop_event, data_queue, buffer_path)
    )
    writer_proc.start()

    try:
        main(stop_event, data_queue, actor_path, stop_file_path)
    except KeyboardInterrupt:
        print("Stopping data generation.")
    finally:
        stop_event.set()
        writer_proc.join()
        # Clean up stop file on exit
        if os.path.exists(stop_file_path):
            os.remove(stop_file_path)
        print("Data generation stopped.")

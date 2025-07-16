import numpy as np
import torch
import os
from networks.world_model_rnn import RNNWorldModel

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import pickle
import portalocker
import threading
import yaml
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.tools import (
    seed_everything,
    save_config_to_shared_folder,
    resolve_all_device_configs,
)
from sklearn.preprocessing import MinMaxScaler
import joblib
from learning.sequence_dataset import create_sequence_dataset_from_buffer
import matplotlib.pyplot as plt
import io
import torchvision


def train_rnn_model(
    model,
    train_dataset,
    val_dataset,
    config: dict,
    batch_size=32,
    stop_event=None,
    writer=None,
    global_step=0,
    model_save_path=None,
):
    """
    Train an RNN world model on sequence data.

    Parameters:
    - model: The RNN world model to train
    - train_dataset: Training sequence dataset
    - val_dataset: Validation sequence dataset
    - config: Training configuration
    - batch_size: Number of sequences per batch
    - stop_event: Threading event to signal interruption
    - writer: TensorBoard SummaryWriter for logging
    - global_step: Global step counter for TensorBoard logging
    - model_save_path: Path to save the model

    Returns:
    - train_losses: List of training losses
    - val_losses: List of validation losses
    """
    trainer_config = config["world_model_trainer"]
    num_epochs = trainer_config["epochs_per_cycle"]
    learning_rate = trainer_config["learning_rate"]

    # Update model's valid_init_state with initial states from training data
    if hasattr(train_dataset, "get_initial_states"):
        new_init_states = train_dataset.get_initial_states()

        # Accumulate valid initial states over time instead of overwriting
        if (
            hasattr(model, "valid_init_state")
            and model.valid_init_state is not None
            and model.valid_init_state.numel() > 0
        ):
            # Concatenate with existing states
            combined_states = torch.cat(
                [model.valid_init_state, new_init_states], dim=0
            )

            # Optional: limit buffer size to prevent unlimited growth
            max_buffer_size = trainer_config.get("max_valid_init_buffer_size", 10000)
            if combined_states.shape[0] > max_buffer_size:
                # Keep the most recent states
                combined_states = combined_states[-max_buffer_size:]

            model.valid_init_state = combined_states
            print(
                f"[TRAINER] Updated valid_init_state buffer: {model.valid_init_state.shape[0]} states (added {new_init_states.shape[0]} new)"
            )
        else:
            # First time initialization
            model.valid_init_state = new_init_states
            print(
                f"[TRAINER] Initialized valid_init_state buffer: {model.valid_init_state.shape[0]} states"
            )

            # Initialize scalers if this is the first training cycle
            if trainer_config["use_scalers"]:
                state_scaler = MinMaxScaler(feature_range=(-3, 3))
                action_scaler = MinMaxScaler(feature_range=(-3, 3))
                reward_scaler = MinMaxScaler(feature_range=(-3, 3))

                # Get sample data for scaler fitting
                sample_inputs, sample_targets = train_dataset[0]

                # Fit state scaler on initial states
                initial_states = new_init_states.cpu().numpy()
                max_state = np.maximum(
                    np.abs(initial_states.min(axis=0)),
                    np.abs(initial_states.max(axis=0)),
                )
                min_state = -max_state
                state_scaler.fit([min_state, max_state])

                # Fit action scaler on sample actions
                sample_actions = (
                    sample_inputs[:, train_dataset.state_size :].cpu().numpy()
                )
                max_action = np.maximum(
                    np.abs(sample_actions.min(axis=0)),
                    np.abs(sample_actions.max(axis=0)),
                )
                min_action = -max_action
                action_scaler.fit([min_action, max_action])

                # Fit reward scaler
                reward_range = [[0.0], [1.0]]
                reward_scaler.fit(reward_range)

                model.set_scalers(state_scaler, action_scaler, reward_scaler)

                # Save scalers
                if model_save_path is not None:
                    model_dir = os.path.dirname(model_save_path)
                    joblib.dump(
                        state_scaler,
                        os.path.join(model_dir, "state_scaler.joblib"),
                    )
                    joblib.dump(
                        action_scaler,
                        os.path.join(model_dir, "action_scaler.joblib"),
                    )
                    joblib.dump(
                        reward_scaler,
                        os.path.join(model_dir, "reward_scaler.joblib"),
                    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=trainer_config["lr_patience"],
        factor=trainer_config["lr_factor"],
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, (context_states, action_seqs, target_seqs) in enumerate(
            train_loader
        ):
            # Check for stop signal
            if stop_event and stop_event.is_set():
                print("[TRAINER] Training interrupted by stop signal")
                return train_losses, val_losses

            optimizer.zero_grad()

            # Context+autoregressive forward pass
            # context_states: (batch_size, context_length, state_size)
            # action_seqs: (batch_size, context_length + prediction_length, action_size)
            # target_seqs: (batch_size, prediction_length, state_size + 1)
            pred_states, pred_rewards = model.predict_sequence(
                context_states,
                action_seqs,
                use_input_state_scaler=trainer_config["use_scalers"],
                use_input_action_scaler=trainer_config["use_scalers"],
                use_output_state_scaler=trainer_config["use_scalers"],
                use_output_reward_scaler=trainer_config["use_scalers"],
            )
            # Concatenate for loss: (batch_size, prediction_length, state_size + 1)
            outputs = torch.cat([pred_states, pred_rewards], dim=-1)

            # Compute loss
            state_loss = criterion(
                outputs[:, :, : model.state_size], target_seqs[:, :, : model.state_size]
            )
            reward_loss = criterion(
                outputs[:, :, model.state_size :], target_seqs[:, :, model.state_size :]
            )
            loss = state_loss + reward_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

            # Log to TensorBoard
            if writer and batch_idx % 10 == 0:
                writer.add_scalar(
                    "Loss/Train_Batch",
                    loss.item(),
                    global_step + epoch * len(train_loader) + batch_idx,
                )

        # Calculate average training loss for this epoch
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        did_viz = False
        with torch.no_grad():
            for batch_idx, (context_states, action_seqs, target_seqs) in enumerate(
                val_loader
            ):
                pred_states, pred_rewards = model.predict_sequence(
                    context_states,
                    action_seqs,
                    use_input_state_scaler=trainer_config["use_scalers"],
                    use_input_action_scaler=trainer_config["use_scalers"],
                    use_output_state_scaler=trainer_config["use_scalers"],
                    use_output_reward_scaler=trainer_config["use_scalers"],
                )
                outputs = torch.cat([pred_states, pred_rewards], dim=-1)
                state_loss = criterion(
                    outputs[:, :, : model.state_size],
                    target_seqs[:, :, : model.state_size],
                )
                reward_loss = criterion(
                    outputs[:, :, model.state_size :],
                    target_seqs[:, :, model.state_size :],
                )
                loss = state_loss + reward_loss
                val_loss += loss.item()
                num_val_batches += 1
                # Visualization for first batch only
                if not did_viz:
                    tb_log_dir = writer.log_dir if writer is not None else "logs"
                    for i in range(min(2, context_states.shape[0])):
                        plot_and_save_val_predictions(
                            epoch,
                            i,
                            pred_states[i].cpu().numpy(),
                            pred_rewards[i].cpu().numpy().squeeze(),
                            target_seqs[i, :, : model.state_size].cpu().numpy(),
                            target_seqs[i, :, model.state_size :]
                            .cpu()
                            .numpy()
                            .squeeze(),
                            tb_log_dir,
                            writer=writer,
                        )
                    did_viz = True

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Log to TensorBoard
        if writer:
            writer.add_scalar("Loss/Train_Epoch", avg_train_loss, global_step + epoch)
            writer.add_scalar("Loss/Val_Epoch", avg_val_loss, global_step + epoch)
            writer.add_scalar(
                "Learning_Rate", optimizer.param_groups[0]["lr"], global_step + epoch
            )

        # Print progress
        if epoch % 100 == 0:
            print(
                f"[TRAINER] Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return train_losses, val_losses


def plot_and_save_val_predictions(
    epoch,
    sample_idx,
    pred_states,
    pred_rewards,
    target_states,
    target_rewards,
    tb_log_dir,
    writer=None,
):
    """
    Plot and save predicted vs actual state and reward trajectories for a single sample.
    """
    import matplotlib.pyplot as plt
    import io

    try:
        import torchvision

        has_torchvision = True
    except ImportError:
        has_torchvision = False
    save_dir = os.path.join(tb_log_dir, "val_viz")
    os.makedirs(save_dir, exist_ok=True)
    seq_len = pred_states.shape[0]
    state_dim = pred_states.shape[1]
    fig, axes = plt.subplots(state_dim + 1, 1, figsize=(8, 2 * (state_dim + 1)))
    # Ensure axes is always a 1D array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    time = np.arange(seq_len)
    # Plot each state dimension
    for d in range(state_dim):
        axes[d].plot(time, target_states[:, d], label="Actual", color="blue")
        axes[d].plot(
            time, pred_states[:, d], label="Predicted", color="red", linestyle="--"
        )
        axes[d].set_ylabel(f"State {d}")
        axes[d].legend()
        axes[d].grid(True, alpha=0.3)
    # Plot reward
    axes[-1].plot(time, target_rewards, label="Actual Reward", color="blue")
    axes[-1].plot(
        time, pred_rewards, label="Predicted Reward", color="red", linestyle="--"
    )
    axes[-1].set_ylabel("Reward")
    axes[-1].set_xlabel("Time Step")
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, f"epoch_{epoch}_sample_{sample_idx}.png")
    plt.savefig(fname)
    plt.close(fig)
    # Optionally log to TensorBoard
    if writer is not None and has_torchvision:
        # Convert plot to image and log
        image = plt.imread(fname)
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        writer.add_images(
            f"ValViz/Epoch_{epoch}_Sample_{sample_idx}", image_tensor, epoch
        )


def pop_data_from_buffer(buffer_path):
    """
    Reads all data from the buffer file, clears the file, and returns the data.
    Uses file locking to prevent race conditions.
    """
    if not os.path.exists(buffer_path) or os.path.getsize(buffer_path) == 0:
        return []

    try:
        with portalocker.Lock(buffer_path, "rb+", timeout=5) as f:
            all_data = []
            while True:
                try:
                    # Load all pickled objects from the file
                    all_data.extend(pickle.load(f))
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    # This can happen if the writer is in the middle of writing.
                    # We'll just try again later.
                    print(
                        "[TRAINER] Warning: Encountered a partial write. Skipping this read attempt."
                    )
                    return []

            # Truncate the file to clear it
            if all_data:
                f.seek(0)
                f.truncate()

            return all_data
    except portalocker.exceptions.LockException:
        print(
            "[TRAINER] Could not acquire lock on buffer file, another process is using it."
        )
        return []
    except Exception as e:
        print(f"[TRAINER] An unexpected error occurred while reading the buffer: {e}")
        return []


def pop_n_oldest_from_buffer(buffer_path, n):
    """
    Reads the N oldest items from the buffer file, removes them from the buffer,
    and returns them. The remaining items stay in the buffer.
    """
    if not os.path.exists(buffer_path) or os.path.getsize(buffer_path) == 0:
        return []

    try:
        with portalocker.Lock(buffer_path, "rb+", timeout=5) as f:
            all_data = []
            while True:
                try:
                    all_data.extend(pickle.load(f))
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    print(
                        "[TRAINER] Warning: Encountered a partial write. Skipping this read attempt."
                    )
                    return []

            if len(all_data) < n:
                print(
                    f"[TRAINER] Not enough data in buffer ({len(all_data)} < {n}). Skipping."
                )
                return []

            items_to_return = all_data[:n]
            remaining_items = all_data[n:]

            f.seek(0)
            f.truncate()
            if remaining_items:
                pickle.dump(remaining_items, f)

            return items_to_return

    except portalocker.exceptions.LockException:
        print(
            "[TRAINER] Could not acquire lock on buffer file, another process is using it."
        )
        return []
    except Exception as e:
        print(f"[TRAINER] An unexpected error occurred while reading the buffer: {e}")
        return []


def sample_data_from_buffer(buffer_path, sample_size):
    """
    Reads all data from the buffer file and returns a random sample of it.
    Does NOT clear the file.
    """
    if not os.path.exists(buffer_path) or os.path.getsize(buffer_path) == 0:
        return []

    try:
        with portalocker.Lock(buffer_path, "rb", timeout=5) as f:
            all_data = []
            while True:
                try:
                    all_data.extend(pickle.load(f))
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    print(
                        "[TRAINER] Warning: Encountered a partial write. Skipping this read attempt."
                    )
                    return []

            if len(all_data) < sample_size:
                return []

            sampled_data = random.sample(all_data, sample_size)
            return sampled_data

    except portalocker.exceptions.LockException:
        print(
            "[TRAINER] Could not acquire lock on buffer file, another process is using it."
        )
        return []
    except Exception as e:
        print(f"[TRAINER] An unexpected error occurred while sampling the buffer: {e}")
        return []


def peek_buffer_size(buffer_path):
    """Safely checks the number of records in the buffer file without modifying it."""
    if not os.path.exists(buffer_path) or os.path.getsize(buffer_path) == 0:
        return 0
    try:
        with portalocker.Lock(buffer_path, "rb", timeout=1) as f:
            count = 0
            while True:
                try:
                    chunk = pickle.load(f)
                    count += len(chunk)
                except EOFError:
                    break
                except pickle.UnpicklingError:
                    return 0
            return count
    except portalocker.exceptions.LockException:
        return 0
    except Exception:
        return 0


def buffer_watcher(stop_training_event, buffer_path, threshold, interval=5):
    """
    Periodically checks the buffer and sets an event if enough new data is available.
    Runs as a daemon thread.
    """
    while True:
        num_records = peek_buffer_size(buffer_path)
        if num_records >= threshold:
            print(
                f"[TRAINER-WATCHER] Found {num_records} records (threshold: {threshold}). Signaling for training update."
            )
            stop_training_event.set()
        time.sleep(interval)


def main():
    """Main function to train the RNN world model."""

    parser = argparse.ArgumentParser(
        description="Run the dynamic RNN world model trainer."
    )
    parser.add_argument(
        "--shared-folder",
        type=str,
        required=True,
        help="Path to the folder where data is buffered and models will be saved.",
    )
    parser.add_argument(
        "--state-size",
        type=int,
        required=True,
        help="The size of the state space.",
    )
    parser.add_argument(
        "--action-size",
        type=int,
        required=True,
        help="The size of the action space.",
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

    # Resolve all device configurations
    config = resolve_all_device_configs(config)

    # seed everything
    seed_everything(config["global"]["seed"])

    trainer_config = config["world_model_trainer"]
    buffer_policy = trainer_config.get("buffer_policy", "latest")
    batch_size_config = trainer_config.get("batch_size", "all")

    # Get imag_horizon from dreamer config
    imag_horizon = config["dreamer_agent_trainer"]["imag_horizon"]

    buffer_path = os.path.join(args.shared_folder, "buffer.pkl")
    model_save_path = os.path.join(args.shared_folder, "model.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save config to shared folder for reproducibility
    save_config_to_shared_folder(
        config, args.config, args.shared_folder, "rnn_world_model_trainer"
    )

    # Setup TensorBoard logging
    tb_config = config.get("tensorboard", {})
    tb_log_dir = os.path.join(
        args.shared_folder, tb_config.get("log_dir", "tb_logs"), "rnn_world_model"
    )
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=tb_log_dir, flush_secs=tb_config.get("flush_seconds", 30)
    )

    print(f"[TRAINER] TensorBoard logging to: {tb_log_dir}")

    # Get state and action sizes from arguments
    state_size = args.state_size
    action_size = args.action_size

    # Create RNN model
    model = RNNWorldModel(
        state_size=state_size,
        action_size=action_size,
        hidden_dim=trainer_config["hidden_dim"],
        num_layers=2,
        dropout=0.1,
    )

    # Setup for Threaded Training Cycle
    stop_training_event = threading.Event()
    watcher_thread = threading.Thread(
        target=buffer_watcher,
        args=(
            stop_training_event,
            buffer_path,
            trainer_config["new_data_threshold"],
            trainer_config["watcher_interval_seconds"],
        ),
        daemon=True,
    )
    watcher_thread.start()

    print("[TRAINER] RNN World model trainer started. Waiting for data...")

    # Main Training Loop
    global_step = 0
    training_cycle = 0
    try:
        while True:
            print("[TRAINER] Waiting for sufficient data to start training cycle...")
            stop_training_event.wait()
            stop_training_event.clear()

            # Pop or sample data based on the configured policy
            if buffer_policy == "latest":
                new_data = pop_data_from_buffer(buffer_path)
            elif buffer_policy == "oldest_n":
                new_data = pop_n_oldest_from_buffer(
                    buffer_path, trainer_config["new_data_threshold"]
                )
            elif buffer_policy == "random":
                new_data = sample_data_from_buffer(
                    buffer_path, trainer_config["new_data_threshold"]
                )
            else:
                raise ValueError(f"Unknown buffer policy: {buffer_policy}")

            if not new_data:
                print(
                    "[TRAINER] Watcher signaled but no new data was retrieved. Retrying..."
                )
                continue

            # Create sequence datasets
            try:
                train_dataset, val_dataset = create_sequence_dataset_from_buffer(
                    experience_data=new_data,
                    state_size=state_size,
                    action_size=action_size,
                    imag_horizon=imag_horizon,
                    train_val_split=0.8,
                    random_seed=42,
                )
            except ValueError as e:
                print(f"[TRAINER] Error creating sequence dataset: {e}")
                continue

            # Determine batch size for training
            if batch_size_config == "all":
                training_batch_size = min(32, len(train_dataset))
            elif isinstance(batch_size_config, int):
                training_batch_size = batch_size_config
            else:
                raise ValueError(
                    f"Invalid batch size configuration: {batch_size_config}"
                )

            print(
                f"\n[TRAINER] --- Starting RNN training on {len(train_dataset)} sequences "
                f"(batch size: {training_batch_size}, policy: {buffer_policy}, imag_horizon: {imag_horizon}) ---"
            )

            # Clear the event again
            stop_training_event.clear()

            train_losses, val_losses = train_rnn_model(
                model,
                train_dataset,
                val_dataset,
                config=config,
                batch_size=training_batch_size,
                stop_event=stop_training_event,
                writer=writer,
                global_step=global_step,
                model_save_path=model_save_path,
            )

            # Update global step counter
            global_step += len(train_losses)
            training_cycle += 1

            print(
                "[TRAINER] --- Training cycle finished/interrupted. Saving model. ---"
            )
            torch.save(model, model_save_path)
            print(f"[TRAINER] Model saved to {model_save_path}")

    except KeyboardInterrupt:
        print("[TRAINER] Training interrupted by user.")
    finally:
        # Close TensorBoard writer
        writer.close()
        print(f"[TRAINER] TensorBoard logs saved to: {tb_log_dir}")


if __name__ == "__main__":
    main()

import numpy as np
import torch
import os
from world_models.world_model_v1 import SimpleModel

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import argparse
import time
import pickle
import portalocker
import threading


def train_model(
    model,
    dataset,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    validation_split=0.2,
    stop_event=None,
):
    """
    Train a PyTorch model that maps vectors to vectors.

    Parameters:
    - model: The PyTorch model to train.
    - dataset: A tuple (X, y) where X is the input vectors and y is the target vectors.
    - batch_size: Number of samples per batch.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - validation_split: Fraction of the dataset to use for validation.
    - stop_event: A threading.Event to signal interruption from another thread.

    Returns:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    """

    # Unpack dataset
    X, y = dataset

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, shuffle=True
    )

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Create DataLoader for training and validation
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        y_train_tensor,
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        y_val_tensor,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.3
    )

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            epoch_train_loss += loss.item() * inputs.size(0)  # Accumulate loss

        # Average training loss for the epoch
        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        # Validation loss computation
        model.eval()  # Set the model to evaluation mode
        epoch_val_loss = 0.0

        with torch.no_grad():  # No gradient computation during validation
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)

        # Average validation loss for the epoch
        epoch_val_loss /= len(X_val)

        scheduler.step(epoch_val_loss)

        val_losses.append(epoch_val_loss)

        # --- Calculate percentage errors ---
        train_rmse = np.sqrt(epoch_train_loss)
        train_err_perc = (
            train_rmse / (torch.mean(torch.abs(y_train_tensor)) + 1e-8)
        ) * 100

        val_rmse = np.sqrt(epoch_val_loss)
        val_err_perc = (val_rmse / (torch.mean(torch.abs(y_val_tensor)) + 1e-8)) * 100

        current_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(
                f"[TRAINER] Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f} ({train_err_perc:.2f}%), Val Loss: {epoch_val_loss:.6f} ({val_err_perc:.2f}%), LR: {current_lr:.6f}"
            )

        # Check for external stop signal
        if stop_event and stop_event.is_set():
            print(
                f"\n[TRAINER] Training interrupted by signal after {epoch + 1} epochs."
            )
            break

    return train_losses, val_losses, val_dataset


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
                    return 0  # Partial write, ignore for now
            return count
    except portalocker.exceptions.LockException:
        return 0  # Can't get lock, try again later
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
    """Main function to train the world model."""

    parser = argparse.ArgumentParser(description="Run the dynamic world model trainer.")
    parser.add_argument(
        "--save-folder",
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
        "--new-data-threshold",
        type=int,
        default=512,
        help="Number of new records in buffer to trigger a training cycle update.",
    )
    parser.add_argument(
        "--epochs-per-cycle",
        type=int,
        default=10000,
        help="Max epochs to train per cycle if not interrupted.",
    )
    args = parser.parse_args()

    buffer_path = os.path.join(args.save_folder, "buffer.pkl")
    model_save_path = os.path.join(args.save_folder, "model.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Get state and action sizes from arguments
    state_size = args.state_size
    action_size = args.action_size

    model = SimpleModel(
        input_dim=state_size + action_size,
        hidden_dim=1024,
        output_dim=state_size + 1,  # next_state, reward
    )

    # --- Setup for Threaded Training Cycle ---
    stop_training_event = threading.Event()
    watcher_thread = threading.Thread(
        target=buffer_watcher,
        args=(
            stop_training_event,
            buffer_path,
            args.new_data_threshold,
        ),
        daemon=True,
    )
    watcher_thread.start()

    print("[TRAINER] World model trainer started. Waiting for data...")

    # --- Main Training Loop ---
    while True:
        # Wait until the watcher signals that there's enough data
        print("[TRAINER] Waiting for sufficient data to start training cycle...")
        stop_training_event.wait()
        stop_training_event.clear()

        # Pop the new data that triggered the event
        new_data = pop_data_from_buffer(buffer_path)

        if not new_data:
            print("[TRAINER] Watcher signaled but buffer was empty. Retrying...")
            continue

        inp_data, outp_data = zip(*new_data)
        training_inp = np.array(inp_data)
        # We only want to predict next_state and reward, not terminated status
        training_outp = np.array(outp_data)[:, : state_size + 1]

        # --- Data Validation ---
        combined_data = np.hstack([training_inp, training_outp])
        invalid_rows_mask = np.isnan(combined_data).any(axis=1) | np.isinf(
            combined_data
        ).any(axis=1)
        if np.any(invalid_rows_mask):
            num_invalid = np.sum(invalid_rows_mask)
            print(
                f"[TRAINER] Warning: Found {num_invalid} rows with NaN/inf values. Removing them."
            )
            training_inp = training_inp[~invalid_rows_mask]
            training_outp = training_outp[~invalid_rows_mask]

        if len(training_inp) == 0:
            print(
                "[TRAINER] No valid data remaining after sanitation. Skipping training cycle."
            )
            continue

        print(
            f"\n[TRAINER] --- Starting training on new batch of {len(training_inp)} records ---"
        )

        # The watcher thread is still running. We need to clear the event again
        # in case it was set while we were processing the pop.
        # This ensures we train for at least a little while before the *next*
        # batch of data interrupts us.
        stop_training_event.clear()

        train_losses, val_losses, _ = train_model(
            model,
            (training_inp, training_outp),
            batch_size=len(training_inp),
            num_epochs=args.epochs_per_cycle,
            learning_rate=1e-5,
            stop_event=stop_training_event,
        )

        print("[TRAINER] --- Training cycle finished/interrupted. Saving model. ---")
        torch.save(model.state_dict(), model_save_path)
        print(f"[TRAINER] Model saved to {model_save_path}")


if __name__ == "__main__":
    main()

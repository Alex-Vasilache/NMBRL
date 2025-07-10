import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import joblib
import os
from world_models.world_model_v1 import SimpleModel

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy
import argparse
import time
import pickle
import portalocker

save_folder = "world_models/trained/dynamic"


def train_model(
    model,
    dataset,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    validation_split=0.2,
    early_stopping_patience=10,
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
    - early_stopping_patience: Number of epochs to wait for improvement before stopping.

    Returns:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    """

    # Unpack dataset
    X, y = dataset

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )

    # Create DataLoader for training and validation
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    classification_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.3
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_train_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            reg_loss = criterion(outputs[:, -1], targets[:, -1])  # Compute loss
            classification_loss = classification_criterion(
                outputs[:, -1], targets[:, -1]
            )
            reg_weight = outputs.shape[1] - 1
            clf_weight = 1

            loss = reg_weight * reg_loss + clf_weight * classification_loss
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
                reg_loss = criterion(outputs[:, -1], targets[:, -1])  # Compute loss
                classification_loss = classification_criterion(
                    outputs[:, -1], targets[:, -1]
                )
                reg_weight = outputs.shape[1] - 1
                clf_weight = 1
                loss = reg_weight * reg_loss + clf_weight * classification_loss
                epoch_val_loss += loss.item() * inputs.size(0)

        # Average validation loss for the epoch
        epoch_val_loss /= len(X_val)

        scheduler.step(epoch_val_loss)

        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    model.load_state_dict(best_model_wts)
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
                        "Warning: Encountered a partial write. Skipping this read attempt."
                    )
                    return []

            # Truncate the file to clear it
            if all_data:
                f.seek(0)
                f.truncate()

            return all_data
    except portalocker.exceptions.LockException:
        print("Could not acquire lock on buffer file, another process is using it.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading the buffer: {e}")
        return []


def main():
    """Main function to train the world model."""

    parser = argparse.ArgumentParser(description="Run the dynamic world model trainer.")
    parser.add_argument(
        "--save-folder",
        type=str,
        required=True,
        help="Path to the folder where data is buffered and models will be saved.",
    )
    args = parser.parse_args()

    buffer_path = os.path.join(args.save_folder, "buffer.pkl")
    model_save_path = os.path.join(args.save_folder, "model.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # These could be command-line arguments as well
    min_buffer_size = 512  # Minimum number of records to start training
    train_interval_seconds = 10  # How often to check for new data and train
    epochs_per_iteration = 10  # Number of epochs to train each time

    # Dummy environment to get state and action sizes.
    # This part can be improved by saving/loading metadata about the env.
    from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper

    temp_env = wrapper(seed=42, n_envs=1)
    temp_env.reset()
    state_size = temp_env.observation_space.shape[0]
    action_size = temp_env.action_space.shape[0]
    temp_env.close()

    model = SimpleModel(
        input_dim=state_size + action_size,
        hidden_dim=1024,
        output_dim=state_size + 2,  # next_state, reward, terminated
    )

    replay_buffer_inp = np.array([]).reshape(0, state_size + action_size)
    replay_buffer_outp = np.array([]).reshape(0, state_size + 2)
    max_replay_buffer_size = 100000

    print("World model trainer started. Waiting for data...")

    while True:
        new_data = pop_data_from_buffer(buffer_path)

        if new_data:
            print(f"Popped {len(new_data)} new records from the buffer.")

            # Separate inputs and outputs and add to replay buffer
            inp_data, outp_data = zip(*new_data)
            new_inp = np.array(inp_data)
            new_outp = np.array(outp_data)

            replay_buffer_inp = np.vstack([replay_buffer_inp, new_inp])
            replay_buffer_outp = np.vstack([replay_buffer_outp, new_outp])

            # Trim buffer if it exceeds max size
            if len(replay_buffer_inp) > max_replay_buffer_size:
                print(
                    f"Replay buffer full. Trimming oldest {len(replay_buffer_inp) - max_replay_buffer_size} records."
                )
                replay_buffer_inp = replay_buffer_inp[-max_replay_buffer_size:]
                replay_buffer_outp = replay_buffer_outp[-max_replay_buffer_size:]

            print(f"Replay buffer size: {len(replay_buffer_inp)}")

        if len(replay_buffer_inp) >= min_buffer_size:
            print("--- Starting training iteration ---")
            train_losses, val_losses, _ = train_model(
                model,
                (replay_buffer_inp, replay_buffer_outp),
                batch_size=512,
                num_epochs=epochs_per_iteration,
                learning_rate=0.0001,
                early_stopping_patience=5,
            )
            print("--- Finished training iteration ---")

            # Save the model
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        else:
            print(
                f"Not enough data to train. Have {len(replay_buffer_inp)}/{min_buffer_size}. Waiting..."
            )

        time.sleep(train_interval_seconds)


if __name__ == "__main__":
    main()

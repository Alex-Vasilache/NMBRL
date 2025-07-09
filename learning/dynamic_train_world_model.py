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


def main():
    """Main function to train the world model."""

    from world_models.ini_gymlike_cartpole_wrapper import GymlikeCartpoleWrapper
    from gymnasium.spaces import Box

    max_episode_steps = 10000
    env = GymlikeCartpoleWrapper(
        seed=42, n_envs=1, render_mode="human", max_episode_steps=max_episode_steps
    )

    state = env.reset()
    state_size = state.shape[0]

    inp_data = np.zeros((max_episode_steps, state_size + 1))
    outp_data = np.zeros((max_episode_steps, state_size + 2))

    action_space = env.action_space
    assert isinstance(action_space, Box)

    model = SimpleModel(
        input_dim=state_size + 1, hidden_dim=1024, output_dim=state_size + 2
    )

    for i in tqdm(range(max_episode_steps)):

        action = np.array(
            np.random.uniform(
                low=action_space.low, high=action_space.high, size=(1, 1)
            ),
            dtype=action_space.dtype,
        )
        inp_data[i, :state_size] = state[0]
        inp_data[i, state_size] = action[:, 0]
        next_state, reward, terminated, info = env.step(action)

        outp_data[i, :state_size] = next_state[0]
        outp_data[i, state_size] = reward[0]
        outp_data[i, state_size + 1] = terminated[0]
        state = next_state

        if terminated:
            state = env.reset()

    # shuffle the dataset
    idxs = np.arange(len(inp_data))
    np.random.shuffle(idxs)
    inp_data = inp_data[idxs]
    outp_data = outp_data[idxs]

    # Train the model
    train_losses, val_losses, val_dataset = train_model(
        model,
        (inp_data, outp_data),
        batch_size=512,
        num_epochs=200,
        learning_rate=0.0001,
        early_stopping_patience=20,
    )

    # Print training statistics
    print("\n=== Training Statistics ===")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Best training loss: {min(train_losses):.6f}")
    print(f"Best validation loss: {min(val_losses):.6f}")
    print(f"Training loss improvement: {train_losses[0] - train_losses[-1]:.6f}")
    print(f"Validation loss improvement: {val_losses[0] - val_losses[-1]:.6f}")

    X_test, y_test = val_dataset.tensors

    y_hat = model(X_test)

    pred_error = y_test[:, :-1] - y_hat[:, :-1]
    clf_error = y_test[:, -1] - y_hat[:, -1]

    abs_errs = np.abs(pred_error.detach().numpy())
    abs_vals = np.abs(y_test[:, :-1].detach().numpy())
    rel_err = np.mean(np.divide(abs_errs, abs_vals), axis=0)

    clf_error = np.abs(clf_error.detach().numpy())
    clf_vals = np.abs(y_test[:, -1].detach().numpy())
    clf_rel_err = np.mean(np.divide(clf_error, clf_vals))

    print(f"\nMean absolute error per output dimension: {np.mean(abs_errs, axis=0)}")
    print(f"Mean relative error per output dimension: {rel_err}")
    print(f"Overall mean absolute error: {np.mean(abs_errs):.6f}")
    print(f"Overall mean relative error: {np.mean(rel_err):.6f}")
    print(f"Mean classification error: {np.mean(clf_error):.6f}")
    print(f"Mean classification relative error: {np.mean(clf_rel_err):.6f}")

    value_range = np.abs(
        np.max(outp_data[:, :-1], axis=0) - np.min(outp_data[:, :-1], axis=0)
    )
    normalized_error = np.divide(np.mean(abs_errs, axis=0), value_range)
    print(f"Normalized error per dimension (MAE/range): {normalized_error}")

    accuracy = np.mean(clf_error < 0.5)
    print(f"Accuracy: {accuracy:.6f}")

    plt.plot(normalized_error)

    # find next available folder name  (v1, v2, ...)
    i = 1
    while os.path.isdir(f"{save_folder}/v{i}"):
        i += 1
    folder_name = f"v{i}"
    os.makedirs(f"{save_folder}/{folder_name}", exist_ok=True)
    torch.save(model, f"{save_folder}/{folder_name}/model.pth")
    if USE_SCALERS:
        joblib.dump(state_scaler, f"{save_folder}/{folder_name}/state_scaler.joblib")
        joblib.dump(action_scaler, f"{save_folder}/{folder_name}/action_scaler.joblib")

    print(f"\nModel and scalers saved to: {save_folder}/{folder_name}")


if __name__ == "__main__":
    main()

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

save_folder = "world_models/trained"
USE_SCALERS = False


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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
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

    if os.path.exists(f"{save_folder}/cartpole_dataset.npz"):
        data = np.load(f"{save_folder}/cartpole_dataset.npz")
        inp_data = data["inputs"]
        outp_data = data["outputs"]
        print("Loaded dataset from file")
    else:
        print("No dataset found, generating new dataset")

        from world_models.ini_gymlike_cartpole_wrapper import GymlikeCartpoleWrapper
        from gymnasium.spaces import Box

        env = GymlikeCartpoleWrapper(seed=42, n_envs=1)
        num_samples = 50000

        state = env.reset()
        batch_size, state_size = state.shape

        inp_data = np.zeros((num_samples, state_size + 1))
        outp_data = np.zeros((num_samples, state_size + 1))

        print(dir(env.action_space))
        for i in tqdm(range(num_samples)):
            action_space = env.action_space
            assert isinstance(action_space, Box)
            action = np.array(
                np.random.uniform(
                    low=action_space.low, high=action_space.high, size=(1, 1)
                ),
                dtype=action_space.dtype,
            )
            inp_data[i, :state_size] = state[0]
            inp_data[i, state_size] = action[0, 0]
            next_state, reward, terminated, info = env.step(action)

            next_state = next_state
            outp_data[i, :state_size] = next_state[0]
            outp_data[i, state_size] = reward[0]
            state = np.copy(next_state)

        print(np.min(outp_data, axis=0))
        print(np.max(outp_data, axis=0))

        # Save the dataset
        np.savez(
            f"{save_folder}/cartpole_dataset.npz",
            inputs=inp_data,
            outputs=outp_data,
        )

        print(f"Dataset saved with {num_samples} samples")

    if USE_SCALERS:
        # Scale the data to normalize inputs and outputs
        from sklearn.preprocessing import StandardScaler

        if os.path.exists(
            f"{save_folder}/cartpole_input_scaler.joblib"
        ) and os.path.exists(f"{save_folder}/cartpole_output_scaler.joblib"):
            scaler_input = joblib.load(f"{save_folder}/cartpole_input_scaler.joblib")
            scaler_output = joblib.load(f"{save_folder}/cartpole_output_scaler.joblib")
            print("Loaded scalers from file")
        else:
            scaler_input = StandardScaler()
            scaler_output = StandardScaler()

            inp_data_scaled = scaler_input.fit_transform(inp_data)
            outp_data_scaled = scaler_output.fit_transform(outp_data)

            print("Data scaling completed")
            print(
                f"Input data - Min: {np.min(inp_data_scaled, axis=0)}, Max: {np.max(inp_data_scaled, axis=0)}"
            )
            print(
                f"Output data - Min: {np.min(outp_data_scaled, axis=0)}, Max: {np.max(outp_data_scaled, axis=0)}"
            )

            # Save the scalers
            joblib.dump(scaler_input, f"{save_folder}/cartpole_input_scaler.joblib")
            joblib.dump(scaler_output, f"{save_folder}/cartpole_output_scaler.joblib")

        inp_data = inp_data_scaled
        outp_data = outp_data_scaled

    model = SimpleModel(
        input_dim=inp_data.shape[1], hidden_dim=1024, output_dim=outp_data.shape[1]
    )

    # Train the model
    num_examples = 50000
    train_losses, val_losses, val_dataset = train_model(
        model,
        (inp_data[:num_examples, :], outp_data[:num_examples, :]),
        batch_size=512,
        num_epochs=200,
        learning_rate=0.001,
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

    pred_error = y_test - y_hat

    abs_errs = np.abs(pred_error.detach().numpy())
    abs_vals = np.abs(y_test.detach().numpy())
    rel_err = np.mean(np.divide(abs_errs, abs_vals), axis=0)

    print(f"\nMean absolute error per output dimension: {np.mean(abs_errs, axis=0)}")
    print(f"Mean relative error per output dimension: {rel_err}")
    print(f"Overall mean absolute error: {np.mean(abs_errs):.6f}")
    print(f"Overall mean relative error: {np.mean(rel_err):.6f}")

    value_range = np.abs(np.max(outp_data, axis=0) - np.min(outp_data, axis=0))
    normalized_error = np.divide(np.mean(abs_errs, axis=0), value_range)
    print(f"Normalized error per dimension (MAE/range): {normalized_error}")

    plt.plot(normalized_error)

    # find next available folder name  (v1, v2, ...)
    i = 1
    while os.path.isdir(f"{save_folder}/v{i}"):
        i += 1
    folder_name = f"v{i}"
    os.makedirs(f"{save_folder}/{folder_name}", exist_ok=True)
    torch.save(model, f"{save_folder}/{folder_name}/model.pth")
    if USE_SCALERS:
        joblib.dump(scaler_input, f"{save_folder}/{folder_name}/input_scaler.joblib")
        joblib.dump(scaler_output, f"{save_folder}/{folder_name}/output_scaler.joblib")

    print(f"\nModel and scalers saved to: {save_folder}/{folder_name}")


if __name__ == "__main__":
    main()

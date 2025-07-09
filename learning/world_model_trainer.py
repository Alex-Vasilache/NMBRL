import numpy as np
import matplotlib.pyplot as plt
import torch

from world_models.ini_gymlike_cartpole_wrapper import GymlikeCartpoleWrapper
from world_models.world_model_v1 import SimpleModel

env = GymlikeCartpoleWrapper(seed=42, n_envs=1)

num_samples = 50000

state = env.reset()
print(state)

inp_data = np.zeros((num_samples, 6))
outp_data = np.zeros((num_samples, 6))

print(dir(env.action_space))
for i in range(num_samples):
    action = np.random.uniform(
        low=env.action_space.low, high=env.action_space.high, size=(1,)
    )
    inp_data[i, :5] = state
    inp_data[i, 5] = action[0]
    next_state, reward, terminated, info = env.step(action)
    outp_data[i, :5] = next_state
    # print(type(reward))
    outp_data[i, 5] = reward[0]  # if not np.any(np.isnan(reward)) else 0.
    # print(state, action, next_state, reward)
    state = np.copy(next_state)

print(np.min(outp_data, axis=0))
print(np.max(outp_data, axis=0))


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


def train_model(
    model,
    dataset,
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    validation_split=0.2,
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
        epoch_train_loss /= len(train_loader.dataset)
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
        epoch_val_loss /= len(val_loader.dataset)

        scheduler.step(epoch_val_loss)

        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
        )

    return train_losses, val_losses, val_dataset


model = SimpleModel(
    input_dim=inp_data.shape[1], hidden_dim=1000, output_dim=outp_data.shape[1]
)


# Train the model
num_examples = 50000
train_losses, val_losses, val_dataset = train_model(
    model,
    (inp_data[:num_examples, :], outp_data[:num_examples, :]),
    batch_size=512,
    num_epochs=200,
    learning_rate=0.001,
)


X_test, y_test = val_dataset.tensors

y_hat = model(X_test)

pred_error = y_test - y_hat

abs_errs = np.abs(pred_error.detach().numpy())
abs_vals = np.abs(y_test.detach().numpy())
rel_err = np.mean(np.divide(abs_errs, abs_vals), axis=0)

value_range = np.abs(np.max(outp_data, axis=0) - np.min(outp_data, axis=0))
plt.plot(np.divide(np.mean(abs_errs, axis=0), value_range))

torch.save(model, "cartpole_world_model.pth")

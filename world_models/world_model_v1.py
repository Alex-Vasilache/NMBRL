import torch.nn as nn
import joblib
import torch
import os
import torch.serialization


def load_model(model_path, with_scalers=False):
    torch.serialization.add_safe_globals([SimpleModel])
    model = torch.load(model_path, weights_only=False)
    if with_scalers:
        state_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "state_scaler.joblib")
        )
        action_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "action_scaler.joblib")
        )
        model.state_scaler = state_scaler
        model.action_scaler = action_scaler
    return model


class SimpleModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        state_scaler=None,
        action_scaler=None,
    ):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler

    def forward(self, x, use_input_scaler=False, use_output_scaler=False):
        if use_input_scaler:
            x[:, : self.state_size] = self.state_scaler.transform(
                x[:, : self.state_size]
            )
            x[:, self.state_size : self.state_size + 1] = self.action_scaler.transform(
                x[:, self.state_size : self.state_size + 1]
            )
        x = self.hidden(x)  # Pass through hidden layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)  # Pass through output layer
        # apply sigmoid to last value (terminated)
        x[:, -1] = torch.sigmoid(x[:, -1])
        if use_output_scaler:
            x[:, : self.state_size] = self.state_scaler.inverse_transform(
                x[:, : self.state_size]
            )

        return x

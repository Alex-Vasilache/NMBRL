import torch.nn as nn
import joblib
import torch
import os
import torch.serialization


def load_model(model_path, with_scalers=False):
    torch.serialization.add_safe_globals([SimpleModel])
    model = torch.load(model_path, weights_only=False)
    if with_scalers:
        input_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "input_scaler.joblib")
        )
        output_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "output_scaler.joblib")
        )
        model.input_scaler = input_scaler
        model.output_scaler = output_scaler
    return model


class SimpleModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, input_scaler=None, output_scaler=None
    ):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def forward(self, x, use_input_scaler=False, use_output_scaler=False):
        if use_input_scaler:
            x = self.input_scaler.transform(x)
        x = self.hidden(x)  # Pass through hidden layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)  # Pass through output layer
        if use_output_scaler:
            x = self.output_scaler.inverse_transform(x)
        return x

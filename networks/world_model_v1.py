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
        state_size,
        action_size,
        state_scaler=None,
        action_scaler=None,
    ):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_size = state_size
        self.action_size = action_size
        self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler
        # self.register_buffer("valid_init_state", torch.zeros(output_dim - 1))
        self.valid_init_state = None

    def forward(self, x, use_input_scaler=False, use_output_scaler=False):
        if use_input_scaler:
            if self.state_scaler:
                x[:, : self.state_size] = self.state_scaler.transform(
                    x[:, : self.state_size]
                )
            if self.action_scaler:
                x[:, self.state_size : self.state_size + self.action_size] = (
                    self.action_scaler.transform(
                        x[:, self.state_size : self.state_size + self.action_size]
                    )
                )
        x = self.hidden(x)  # Pass through hidden layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)  # Pass through output layer

        # --- Clamp outputs to reasonable ranges ---
        # State is clipped based on the VecNormalize wrapper's clip_obs value
        state_output = torch.clamp(x[:, : self.state_size], -10.0, 10.0)
        # Reward is clipped to a reasonable range for cartpole-swingup (normally 0-1)
        reward_output = torch.clamp(x[:, self.state_size], -2.0, 2.0).unsqueeze(1)

        x = torch.cat([state_output, reward_output], dim=1)

        if use_output_scaler:
            if self.state_scaler:
                x[:, : self.state_size] = self.state_scaler.inverse_transform(
                    x[:, : self.state_size]
                )

        return x

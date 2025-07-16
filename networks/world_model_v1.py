import torch.nn as nn
import joblib
import torch
import os
import torch.serialization


ACTION_SCALER = "action"
STATE_SCALER = "state"
REWARD_SCALER = "reward"


def load_model(model_path, with_scalers=False, map_location=None, weights_only=False):
    torch.serialization.add_safe_globals([SimpleModel])
    model = torch.load(model_path, map_location=map_location, weights_only=weights_only)
    if with_scalers:
        state_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "state_scaler.joblib")
        )
        action_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "action_scaler.joblib")
        )
        reward_scaler = joblib.load(
            os.path.join(os.path.dirname(model_path), "reward_scaler.joblib")
        )
        model.set_scalers(state_scaler, action_scaler, reward_scaler)
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
        reward_scaler=None,
    ):
        super(SimpleModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_size = state_size
        self.action_size = action_size
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        self.set_scalers(state_scaler, action_scaler, reward_scaler)
        self.valid_init_state = None

    def set_scalers(self, state_scaler, action_scaler, reward_scaler):
        self.state_scaler = state_scaler
        self.action_scaler = action_scaler
        self.reward_scaler = reward_scaler

        # Get the device of the model to ensure scaler tensors are on the same device
        device = (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )

        if self.state_scaler:
            self.state_scaler_min = torch.tensor(
                self.state_scaler.min_,
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )
            self.state_scaler_scale = torch.tensor(
                self.state_scaler.scale_,
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )
        if self.action_scaler:
            self.action_scaler_min = torch.tensor(
                self.action_scaler.min_,
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )
            self.action_scaler_scale = torch.tensor(
                self.action_scaler.scale_,
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )
        if self.reward_scaler:
            self.reward_scaler_min = torch.tensor(
                self.reward_scaler.min_,
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )
            self.reward_scaler_scale = torch.tensor(
                self.reward_scaler.scale_,
                dtype=torch.float32,
                requires_grad=False,
                device=device,
            )

    def _do_scale(self, x, scaler=str):
        if scaler == STATE_SCALER:
            scaler_min = self.state_scaler_min
            scaler_scale = self.state_scaler_scale
        elif scaler == ACTION_SCALER:
            scaler_min = self.action_scaler_min
            scaler_scale = self.action_scaler_scale
        elif scaler == REWARD_SCALER:
            scaler_min = self.reward_scaler_min
            scaler_scale = self.reward_scaler_scale
        else:
            raise ValueError(f"Invalid scaler: {scaler}")
        return x * scaler_scale + scaler_min

    def _do_unscale(self, x, scaler=str):
        if scaler == STATE_SCALER:
            scaler_min = self.state_scaler_min
            scaler_scale = self.state_scaler_scale
        elif scaler == ACTION_SCALER:
            scaler_min = self.action_scaler_min
            scaler_scale = self.action_scaler_scale
        elif scaler == REWARD_SCALER:
            scaler_min = self.reward_scaler_min
            scaler_scale = self.reward_scaler_scale
        else:
            raise ValueError(f"Invalid scaler: {scaler}")
        return (x - scaler_min) / scaler_scale

    def scale_input(self, x, use_input_state_scaler, use_input_action_scaler):
        if use_input_state_scaler and self.state_scaler:
            x[:, : self.state_size] = self._do_scale(
                x[:, : self.state_size], STATE_SCALER
            )
        if use_input_action_scaler and self.action_scaler:
            x[:, self.state_size : self.state_size + self.action_size] = self._do_scale(
                x[:, self.state_size : self.state_size + self.action_size],
                ACTION_SCALER,
            )

        return x

    def unscale_output(self, x, use_output_state_scaler, use_output_reward_scaler):
        # preserve gradients
        if use_output_state_scaler and self.state_scaler:
            x[:, : self.state_size] = self._do_unscale(
                x[:, : self.state_size], STATE_SCALER
            )

        if use_output_reward_scaler and self.reward_scaler:
            x[:, self.state_size :] = torch.nn.functional.sigmoid(
                x[:, self.state_size :] * 3.6
            )

        return x

    def forward(
        self,
        x,
        use_input_state_scaler=False,
        use_input_action_scaler=False,
        use_output_state_scaler=False,
        use_output_reward_scaler=False,
    ):
        x = self.scale_input(x, use_input_state_scaler, use_input_action_scaler)

        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)

        x = self.unscale_output(x, use_output_state_scaler, use_output_reward_scaler)

        # # --- Clamp outputs to reasonable ranges ---
        # # State is clipped based on the VecNormalize wrapper's clip_obs value
        # x[:, : self.state_size] = torch.clamp(x[:, : self.state_size], -10.0, 10.0)
        # # Reward is clipped to a reasonable range for cartpole-swingup (normally 0-1)
        # x[:, self.state_size :] = torch.clamp(
        #     x[:, self.state_size :], -2.0, 2.0
        # ).unsqueeze(1)

        return x

    def to(self, device):
        """Override to ensure scaler tensors are also moved to the device."""
        super().to(device)
        # Move scaler tensors to the same device
        if hasattr(self, "state_scaler_min"):
            self.state_scaler_min = self.state_scaler_min.to(device)
            self.state_scaler_scale = self.state_scaler_scale.to(device)
        if hasattr(self, "action_scaler_min"):
            self.action_scaler_min = self.action_scaler_min.to(device)
            self.action_scaler_scale = self.action_scaler_scale.to(device)
        if hasattr(self, "reward_scaler_min"):
            self.reward_scaler_min = self.reward_scaler_min.to(device)
            self.reward_scaler_scale = self.reward_scaler_scale.to(device)
        return self

import torch.nn as nn
import joblib
import torch
import os
import torch.serialization


ACTION_SCALER = "action"
STATE_SCALER = "state"
REWARD_SCALER = "reward"


def load_model(model_path, with_scalers=False, map_location=None, weights_only=False):
    torch.serialization.add_safe_globals([RNNWorldModel])
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


class RNNWorldModel(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_dim,
        num_layers=2,
        dropout=0.1,
        state_scaler=None,
        action_scaler=None,
        reward_scaler=None,
    ):
        super(RNNWorldModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection layer
        self.input_projection = nn.Linear(state_size + action_size, hidden_dim)

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection layers
        self.state_output = nn.Linear(hidden_dim, state_size)
        self.reward_output = nn.Linear(hidden_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize scalers
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

    def _do_scale(self, x, scaler: str):
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

    def _do_unscale(self, x, scaler: str):
        if scaler == STATE_SCALER:
            scaler_min = self.state_scaler_min
            scaler_scale = self.state_scaler_scale
        elif scaler == ACTION_SCALER:
            scaler_min = self.action_scaler_min
            scaler_scale = self.action_scaler_scale
        else:
            raise ValueError(f"Invalid scaler: {scaler}")
        return (x - scaler_min) / scaler_scale

    def scale_input(self, x, use_input_state_scaler, use_input_action_scaler):
        if use_input_state_scaler and self.state_scaler:
            x[:, :, : self.state_size] = self._do_scale(
                x[:, :, : self.state_size], STATE_SCALER
            )
        if use_input_action_scaler and self.action_scaler:
            x[:, :, self.state_size : self.state_size + self.action_size] = (
                self._do_scale(
                    x[:, :, self.state_size : self.state_size + self.action_size],
                    ACTION_SCALER,
                )
            )
        return x

    def unscale_output(
        self, states, rewards, use_output_state_scaler, use_output_reward_scaler
    ):
        # preserve gradients
        if use_output_state_scaler and self.state_scaler:
            states = self._do_unscale(states, STATE_SCALER)

        if use_output_reward_scaler and self.reward_scaler:
            rewards = torch.nn.functional.sigmoid(rewards * 3.6)

        return states, rewards

    def forward(
        self,
        x,
        use_input_state_scaler=False,
        use_input_action_scaler=False,
        use_output_state_scaler=False,
        use_output_reward_scaler=False,
    ):
        # x shape: (batch_size, seq_len, state_size + action_size)
        batch_size, seq_len, _ = x.shape

        # Scale inputs if needed
        x = self.scale_input(x, use_input_state_scaler, use_input_action_scaler)

        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Apply dropout
        x = self.dropout(x)

        # Pass through GRU
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)

        # Apply dropout to GRU output
        gru_out = self.dropout(gru_out)

        # Project to outputs
        states = self.state_output(gru_out)  # (batch_size, seq_len, state_size)
        rewards = self.reward_output(gru_out)  # (batch_size, seq_len, 1)

        # Unscale outputs if needed
        states, rewards = self.unscale_output(
            states, rewards, use_output_state_scaler, use_output_reward_scaler
        )

        # Concatenate states and rewards for compatibility with existing interface
        # Shape: (batch_size, seq_len, state_size + 1)
        outputs = torch.cat([states, rewards], dim=-1)

        return outputs

    def predict_sequence(
        self,
        initial_state,
        actions,
        use_input_state_scaler=False,
        use_input_action_scaler=False,
        use_output_state_scaler=False,
        use_output_reward_scaler=False,
    ):
        """
        Predict a sequence of states and rewards given an initial state and sequence of actions.

        Args:
            initial_state: (batch_size, state_size) initial state
            actions: (batch_size, seq_len, action_size) sequence of actions
            use_*_scaler: boolean flags for scaling

        Returns:
            states: (batch_size, seq_len, state_size) predicted states
            rewards: (batch_size, seq_len, 1) predicted rewards
        """
        batch_size, seq_len, _ = actions.shape

        # Prepare input sequence
        # Start with initial state and first action
        current_state = initial_state.unsqueeze(1)  # (batch_size, 1, state_size)
        first_action = actions[:, 0:1, :]  # (batch_size, 1, action_size)
        first_input = torch.cat(
            [current_state, first_action], dim=-1
        )  # (batch_size, 1, state_size + action_size)

        # Initialize output tensors
        all_states = []
        all_rewards = []

        # Autoregressive prediction
        for t in range(seq_len):
            # Prepare input for this timestep
            if t == 0:
                x = first_input
            else:
                # Use predicted state from previous timestep
                prev_state = all_states[-1].unsqueeze(1)  # (batch_size, 1, state_size)
                current_action = actions[
                    :, t : t + 1, :
                ]  # (batch_size, 1, action_size)
                x = torch.cat(
                    [prev_state, current_action], dim=-1
                )  # (batch_size, 1, state_size + action_size)

            # Get prediction
            with torch.no_grad():
                output = self.forward(
                    x,
                    use_input_state_scaler=use_input_state_scaler,
                    use_input_action_scaler=use_input_action_scaler,
                    use_output_state_scaler=use_output_state_scaler,
                    use_output_reward_scaler=use_output_reward_scaler,
                )

            # Extract state and reward
            state = output[:, 0, : self.state_size]  # (batch_size, state_size)
            reward = output[:, 0, self.state_size :]  # (batch_size, 1)

            all_states.append(state)
            all_rewards.append(reward)

        # Stack all predictions
        states = torch.stack(all_states, dim=1)  # (batch_size, seq_len, state_size)
        rewards = torch.stack(all_rewards, dim=1)  # (batch_size, seq_len, 1)

        return states, rewards

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

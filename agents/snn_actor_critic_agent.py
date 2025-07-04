# This file contains the implementation of the SNN-based actor-critic agent.
# It uses Spiking Neural Networks for both policy (actor) and value function (critic).

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from utils.tools import lambda_return
from .base_agent import BaseAgent


class CriticSNN(nn.Module):
    """
    Spiking Neural Network for the Critic (Value Function).
    Takes state as input and outputs a scalar value estimate.
    """

    def __init__(
        self,
        state_dim=6,
        hidden_dim=128,
        output_dim=1,
        num_steps=1,
        alpha=0.9,
        beta=0.9,
        threshold=1,
        learn_alpha=True,
        learn_beta=True,
        learn_threshold=True,
        weight_init_mean=0.0,
        weight_init_std=0.01,
        max_std=2.0,
        min_std=0.1,
    ):
        """
        Initialize the Critic SNN.

        :param state_dim: Dimension of the state space (default: 6 for CartPole)
        :param hidden_dim: Number of hidden neurons in each layer
        :param output_dim: Output dimension (1 for value function)
        :param num_steps: Number of time steps for SNN simulation
        :param beta: Decay factor for LIF neurons
        """
        super(CriticSNN, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.max_std = max_std
        self.min_std = min_std

        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.lif1 = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            spike_grad=surrogate.fast_sigmoid(),
        )
        self.rec1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rec1.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.lif2 = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            spike_grad=surrogate.fast_sigmoid(),
        )
        self.rec2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rec2.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        # Output layers for mean and std
        self.fc_mean = nn.Linear(hidden_dim, output_dim, bias=False)
        self.fc_mean.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc_std = nn.Linear(hidden_dim, output_dim, bias=False)
        self.fc_std.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        self.reset()

    def forward(self, state):
        """
        Forward pass through the Critic SNN.

        :param state: Input state tensor of shape (batch_size, state_dim)
        :return: Value estimate tensor of shape (batch_size, 1)
        """
        batch_size = state.shape[0]

        # Collect output spikes over time
        spk_rec = []

        # Simulate over time steps
        for step in range(self.num_steps):

            cur1 = self.fc1(state)
            if self.spk1 is None:
                self.spk1 = torch.zeros_like(cur1)
            cur1_rec = self.rec1(self.spk1)
            self.spk1, self.syn1, self.mem1 = self.lif1(
                cur1 + cur1_rec, self.syn1, self.mem1
            )

            cur2 = self.fc2(self.spk1)
            if self.spk2 is None:
                self.spk2 = torch.zeros_like(cur2)
            cur2_rec = self.rec2(self.spk2)
            self.spk2, self.syn2, self.mem2 = self.lif2(
                cur2 + cur2_rec, self.syn2, self.mem2
            )

            spk_rec.append(self.spk2)

        avg_spikes = torch.stack(spk_rec, dim=0).mean(dim=0)
        value_mean = self.fc_mean(avg_spikes)
        value_std = self.fc_std(avg_spikes)

        value_mean = torch.tanh(value_mean)
        value_std = (self.max_std - self.min_std) * torch.sigmoid(
            value_std + 2.0
        ) + self.min_std

        return value_mean, value_std

    def reset(self):
        self.syn1, self.mem1 = self.lif1.init_synaptic()
        self.syn2, self.mem2 = self.lif2.init_synaptic()
        self.spk1 = None
        self.spk2 = None

    def get_states(self):
        return {
            "syn1": self.syn1,
            "syn2": self.syn2,
            "mem1": self.mem1,
            "mem2": self.mem2,
            "spk1": self.spk1,
            "spk2": self.spk2,
        }

    def set_states(self, states):
        self.syn1 = (
            states["syn1"]
            if type(states) == type({})
            else torch.stack([states[i]["syn1"] for i in range(len(states))])
        )
        self.syn2 = (
            states["syn2"]
            if type(states) == type({})
            else torch.stack([states[i]["syn2"] for i in range(len(states))])
        )
        self.mem1 = (
            states["mem1"]
            if type(states) == type({})
            else torch.stack([states[i]["mem1"] for i in range(len(states))])
        )
        self.mem2 = (
            states["mem2"]
            if type(states) == type({})
            else torch.stack([states[i]["mem2"] for i in range(len(states))])
        )
        self.spk1 = (
            states["spk1"]
            if type(states) == type({})
            else (
                None
                if states[0]["spk1"] is None
                else torch.stack([states[i]["spk1"] for i in range(len(states))])
            )
        )
        self.spk2 = (
            states["spk2"]
            if type(states) == type({})
            else (
                None
                if states[0]["spk2"] is None
                else torch.stack([states[i]["spk2"] for i in range(len(states))])
            )
        )


class CriticANN(nn.Module):
    """
    ANN for the Critic (Value Function).
    Takes state as input and outputs a scalar value estimate.
    """

    def __init__(
        self,
        state_dim=6,
        hidden_dim=128,
        output_dim=1,
        weight_init_mean=0.0,
        weight_init_std=0.01,
        max_std=2.0,
        min_std=0.1,
    ):
        """
        Initialize the Critic ANN.

        :param state_dim: Dimension of the state space (default: 6 for CartPole)
        :param hidden_dim: Number of hidden neurons in each layer
        :param output_dim: Output dimension (1 for value function)
        :param weight_init_mean: Mean for weight initialization
        :param weight_init_std: Standard deviation for weight initialization
        :param max_std: Maximum standard deviation for output
        :param min_std: Minimum standard deviation for output
        """
        super(CriticANN, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.max_std = max_std
        self.min_std = min_std

        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc1.bias.data.zero_()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc2.bias.data.zero_()

        # Output layers for mean and std
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_mean.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc_mean.bias.data.zero_()

        self.fc_std = nn.Linear(hidden_dim, output_dim)
        self.fc_std.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc_std.bias.data.zero_()

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, state):
        """
        Forward pass through the Critic ANN.

        :param state: Input state tensor of shape (batch_size, state_dim)
        :return: Tuple of (value_mean, value_std)
        """
        # Forward pass through hidden layers
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))

        # Output value mean and std
        value_mean = self.fc_mean(x)
        value_std = self.fc_std(x)

        # Apply activation functions to match CriticSNN behavior
        value_mean = torch.tanh(value_mean)
        value_std = (self.max_std - self.min_std) * torch.sigmoid(
            value_std + 2.0
        ) + self.min_std

        return value_mean, value_std

    def reset(self):
        """
        Reset method for interface compatibility with CriticSNN.
        ANN doesn't need state reset but keeping for consistency.
        """
        pass

    def get_states(self):
        """
        Get states method for interface compatibility with CriticSNN.
        ANN doesn't have internal states but keeping for consistency.
        """
        return {}

    def set_states(self, states):
        """
        Set states method for interface compatibility with CriticSNN.
        ANN doesn't have internal states but keeping for consistency.
        """
        pass


class ActorSNN(nn.Module):
    """
    Spiking Neural Network for the Actor (Policy).
    Takes state as input and outputs action parameters.
    """

    def __init__(
        self,
        state_dim=6,
        hidden_dim=128,
        action_dim=1,
        num_steps=1,
        alpha=0.9,
        beta=0.9,
        threshold=1,
        learn_alpha=True,
        learn_beta=True,
        learn_threshold=True,
        weight_init_mean=0.0,
        weight_init_std=0.01,
        max_std=2.0,
        min_std=0.1,
    ):
        """
        Initialize the Actor SNN.

        :param state_dim: Dimension of the state space
        :param hidden_dim: Number of hidden neurons in each layer
        :param action_dim: Dimension of the action space
        :param num_steps: Number of time steps for SNN simulation
        :param beta: Decay factor for LIF neurons
        """
        super(ActorSNN, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.learn_threshold = learn_threshold
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.max_std = max_std
        self.min_std = min_std

        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim, bias=False)
        self.fc1.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        self.lif1 = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            spike_grad=surrogate.fast_sigmoid(),
        )

        self.rec1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rec1.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        self.lif2 = snn.Synaptic(
            alpha=alpha,
            beta=beta,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            spike_grad=surrogate.fast_sigmoid(),
        )

        self.rec2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rec2.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        # Output layers for action mean and log_std
        self.fc_mean = nn.Linear(hidden_dim, action_dim, bias=False)
        self.fc_std = nn.Linear(hidden_dim, action_dim, bias=False)

        self.fc_mean.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )
        self.fc_std.weight.data.normal_(
            mean=self.weight_init_mean, std=self.weight_init_std
        )

        self.reset()

    def forward(self, state):
        """
        Forward pass through the Actor SNN.

        :param state: Input state tensor of shape (batch_size, state_dim)
        :return: Tuple of (action_mean, action_std)
        """
        batch_size = state.shape[0]

        # Collect output spikes over time
        spk_rec = []

        # Simulate over time steps
        for step in range(self.num_steps):
            # Encode state as current input
            cur1 = self.fc1(state)
            if self.spk1 is None:
                self.spk1 = torch.zeros_like(cur1)
            cur1_rec = self.rec1(self.spk1)
            self.spk1, self.syn1, self.mem1 = self.lif1(
                cur1 + cur1_rec, self.syn1, self.mem1
            )

            cur2 = self.fc2(self.spk1)
            if self.spk2 is None:
                self.spk2 = torch.zeros_like(cur2)
            cur2_rec = self.rec2(self.spk2)
            self.spk2, self.syn2, self.mem2 = self.lif2(
                cur2 + cur2_rec, self.syn2, self.mem2
            )

            spk_rec.append(self.spk2)

        avg_spikes = torch.stack(spk_rec, dim=0).mean(dim=0)

        # Output action parameters
        action_mean = self.fc_mean(avg_spikes)
        action_std = self.fc_std(avg_spikes)

        action_mean = torch.tanh(action_mean)

        action_std = (self.max_std - self.min_std) * torch.sigmoid(
            action_std + 2.0
        ) + self.min_std

        return action_mean, action_std

    def reset(self):
        self.syn1, self.mem1 = self.lif1.init_synaptic()
        self.syn2, self.mem2 = self.lif2.init_synaptic()
        self.spk1 = None
        self.spk2 = None

    def get_states(self):
        return {
            "syn1": self.syn1,
            "syn2": self.syn2,
            "mem1": self.mem1,
            "mem2": self.mem2,
            "spk1": self.spk1,
            "spk2": self.spk2,
        }

    def set_states(self, states):
        self.syn1 = (
            states["syn1"]
            if type(states) == type({})
            else torch.stack([states[i]["syn1"] for i in range(len(states))])
        )
        self.syn2 = (
            states["syn2"]
            if type(states) == type({})
            else torch.stack([states[i]["syn2"] for i in range(len(states))])
        )
        self.mem1 = (
            states["mem1"]
            if type(states) == type({})
            else torch.stack([states[i]["mem1"] for i in range(len(states))])
        )
        self.mem2 = (
            states["mem2"]
            if type(states) == type({})
            else torch.stack([states[i]["mem2"] for i in range(len(states))])
        )
        self.spk1 = (
            states["spk1"]
            if type(states) == type({})
            else (
                None
                if states[0]["spk1"] is None
                else torch.stack([states[i]["spk1"] for i in range(len(states))])
            )
        )
        self.spk2 = (
            states["spk2"]
            if type(states) == type({})
            else (
                None
                if states[0]["spk2"] is None
                else torch.stack([states[i]["spk2"] for i in range(len(states))])
            )
        )


class SnnActorCriticAgent(BaseAgent):
    """
    An SNN-based Actor-Critic agent using SNNtorch.
    Uses separate SNNs for the actor (policy) and critic (value function).
    """

    def __init__(
        self,
        action_space,
        state_dim=6,
        hidden_dim=128,
        num_steps=1,
        lr=1e-3,
        alpha=0.9,
        beta=0.9,
        threshold=1,
        learn_alpha=True,
        learn_beta=True,
        learn_threshold=True,
        weight_init_mean=0.0,
        weight_init_std=0.01,
        max_std=2.0,
        min_std=0.1,
    ):
        """
        Initialize the SNN Actor-Critic agent.

        :param action_space: Action space from the environment
        :param state_dim: Dimension of the state space
        :param hidden_dim: Number of hidden neurons in each SNN layer
        :param num_steps: Number of time steps for SNN simulation
        :param lr: Learning rate for the optimizers
        """
        self.action_space = action_space
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.action_dim = 1  # CartPole has 1D continuous action space
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std

        # Initialize Actor and Critic SNNs
        self.actor = ActorSNN(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            num_steps=num_steps,
            alpha=alpha,
            beta=beta,
            threshold=threshold,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            weight_init_mean=weight_init_mean,
            weight_init_std=weight_init_std,
        )

        self.critic = CriticANN(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            weight_init_mean=weight_init_mean,
            weight_init_std=weight_init_std,
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )  # TODO test if neuron params are included in the optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Set to training mode
        self.actor.train()
        self.critic.train()

    def get_action(self, state):
        """
        Get action from the actor network using the current policy.

        :param state: Current state of the environment
        :return: Action to take in the environment
        """
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        else:
            state_tensor = torch.FloatTensor([state]).unsqueeze(0)

        # Get action from actor network
        with torch.no_grad():
            action_mean, action_std = self.actor(state_tensor)

            # Sample action from normal distribution
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()

            # Clamp action to action space bounds
            action = torch.clamp(
                action,
                torch.FloatTensor([self.action_space.low[0]]),
                torch.FloatTensor([self.action_space.high[0]]),
            )

        # Return as 1D array to match environment expectation
        return action.squeeze().numpy().reshape(-1)

    def get_value(self, state):
        """
        Get value estimate from the critic network.

        :param state: Current state of the environment
        :return: Value estimate for the state
        """
        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        else:
            state_tensor = torch.FloatTensor([state]).unsqueeze(0)

        with torch.no_grad():
            value_mean, value_std = self.critic(state_tensor)

            dist = torch.distributions.Normal(value_mean, value_std)
            value = dist.sample()

        return value.squeeze().item()

    def compute_action_log_probs(self, states, rollout_actions):
        """
        Compute log probabilities of taking actions in given states.

        :param state: State tensor, shape is (sequence_length, batch_size, state_dim)
        :param action: Action tensor, shape is (sequence_length, batch_size, action_dim)
        :return: Log probabilities of the actions
        """

        batch_size = states.shape[1]
        sequence_length = states.shape[0]

        self.actor.reset()

        action_means = []
        action_stds = []

        for i in range(sequence_length):
            state = states[i]
            action_mean, action_std = self.actor(state)
            action_means.append(action_mean)
            action_stds.append(action_std)

        action_means = torch.stack(action_means, dim=0)
        action_stds = torch.stack(action_stds, dim=0)

        dist = torch.distributions.Normal(action_means, action_stds)
        log_probs = dist.log_prob(rollout_actions)

        return log_probs

    def save_agent_states(self):
        self.actor_states = self.actor.get_states()
        self.critic_states = self.critic.get_states()

    def load_agent_states(self):
        self.actor.set_states(self.actor_states)
        self.critic.set_states(self.critic_states)

    def update(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        gamma=0.997,
        discount_lambda=0.95,
    ):
        """
        Update both actor and critic networks using collected experience.

        :param states: Batch of states, shape is (sequence_length, batch_size, state_dim)
        :param actions: Batch of actions, shape is (sequence_length, batch_size, action_dim)
        :param rewards: Batch of rewards, shape is (sequence_length, batch_size)
        :param next_states: Batch of next states, shape is (sequence_length, batch_size, state_dim)
        :param dones: Batch of done flags, shape is (sequence_length, batch_size)
        :param gamma: Discount factor
        :param discount_lambda: Return mixing factor
        """
        # Convert to tensors
        states = torch.FloatTensor(states).requires_grad_(False)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(2)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(2)

        batch_size = states.shape[1]
        sequence_length = states.shape[0]

        self.critic.reset()

        values_dists = []
        values = []
        for i in range(sequence_length):
            value_mean, value_std = self.critic(states[i])
            value_dist = torch.distributions.Normal(value_mean, value_std)
            values_dists.append(value_dist)
            values.append(value_dist.mode)

        values = torch.stack(values, dim=0)

        # Compute target values (λ-returns)
        target_values, weights = lambda_return(
            rewards[1:],
            values[:-1],
            gamma,
            values[-1],
            discount_lambda,
            axis=0,
        )
        target_values = torch.stack(target_values, dim=1)

        # These targets train the critic
        critic_loss = -torch.stack(
            [
                values_dist.log_prob(target_value.detach())
                for values_dist, target_value in zip(values_dists, target_values)
            ],
            dim=0,
        )
        critic_loss = torch.mean(weights * critic_loss)

        # Compute advantage
        advantages = target_values - values[:-1]

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        log_probs = self.compute_action_log_probs(states, actions)
        actor_target = log_probs[:-1] * (target_values - values[:-1]).detach()
        mix = getattr(self, "imag_gradient_mix", 0.01)
        actor_target = mix * target_values.detach() + (1 - mix) * actor_target
        actor_loss = -torch.mean(weights * actor_target)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_value": values.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

# This file contains the implementation of the SNN-based actor-critic agent.
# It uses Spiking Neural Networks for both policy (actor) and value function (critic).

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from .base_agent import BaseAgent


class CriticSNN(nn.Module):
    """
    Spiking Neural Network for the Critic (Value Function).
    Takes state as input and outputs a scalar value estimate.
    """

    def __init__(
        self, state_dim=6, hidden_dim=128, output_dim=1, num_steps=1, alpha=0.9, beta=0.9, threshold=1, learn_alpha=True, learn_beta=True, learn_threshold=True
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

        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, spike_grad=surrogate.fast_sigmoid())

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, spike_grad=surrogate.fast_sigmoid())

        # Output layer (no spiking neuron, direct value output)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

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
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)

            cur3 = self.fc3(spk2)
            spk3, self.mem3 = self.lif3(cur3, self.mem3)

            spk_rec.append(spk3)

        avg_spikes = torch.stack(spk_rec, dim=0).mean(dim=0)
        value = self.fc_out(avg_spikes)

        return value

    def reset(self):
        self.mem1 = self.lif1.init_synaptic()
        self.mem2 = self.lif2.init_synaptic()
        self.mem3 = self.lif3.init_synaptic()


class ActorSNN(nn.Module):
    """
    Spiking Neural Network for the Actor (Policy).
    Takes state as input and outputs action parameters.
    """

    def __init__(
        self, state_dim=6, hidden_dim=128, action_dim=1, num_steps=1, alpha=0.9, beta=0.9, threshold=1, learn_alpha=True, learn_beta=True, learn_threshold=True
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

        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, spike_grad=surrogate.fast_sigmoid())

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=learn_alpha, learn_beta=learn_beta, learn_threshold=learn_threshold, spike_grad=surrogate.fast_sigmoid())

        # Output layers for action mean and log_std
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        self.reset()

    def forward(self, state):
        """
        Forward pass through the Actor SNN.

        :param state: Input state tensor of shape (batch_size, state_dim)
        :return: Tuple of (action_mean, action_log_std)
        """
        batch_size = state.shape[0]

        # Collect output spikes over time
        spk_rec = []

        # Simulate over time steps
        for step in range(self.num_steps):
            # Encode state as current input
            cur1 = self.fc1(state)
            spk1, self.mem1 = self.lif1(cur1, self.mem1)

            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)

            cur3 = self.fc3(spk2)
            spk3, self.mem3 = self.lif3(cur3, self.mem3)

            spk_rec.append(spk3)

        avg_spikes = torch.stack(spk_rec, dim=0).mean(dim=0)

        # Output action parameters
        action_mean = self.fc_mean(avg_spikes)
        action_log_std = self.fc_log_std(avg_spikes)

        # Clamp log_std to prevent numerical instability
        action_log_std = torch.clamp(action_log_std, min=-10, max=2)

        return action_mean, action_log_std

    def reset(self):
        self.mem1 = self.lif1.init_synaptic()
        self.mem2 = self.lif2.init_synaptic()
        self.mem3 = self.lif3.init_synaptic()


class SnnActorCriticAgent(BaseAgent):
    """
    An SNN-based Actor-Critic agent using SNNtorch.
    Uses separate SNNs for the actor (policy) and critic (value function).
    """

    def __init__(
        self, action_space, state_dim=6, hidden_dim=128, num_steps=20, lr=1e-3
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

        # Initialize Actor and Critic SNNs
        self.actor = ActorSNN(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            num_steps=num_steps,
        )

        self.critic = CriticSNN(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_steps=num_steps,
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
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
            action_mean, action_log_std = self.actor(state_tensor)
            action_std = torch.exp(action_log_std)

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
            value = self.critic(state_tensor)

        return value.squeeze().item()

    def compute_action_log_prob(self, state, action):
        """
        Compute log probability of taking action in given state.

        :param state: State tensor
        :param action: Action tensor
        :return: Log probability of the action
        """
        action_mean, action_log_std = self.actor(state)
        action_std = torch.exp(action_log_std)

        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return log_prob

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """
        Update both actor and critic networks using collected experience.

        :param states: Batch of states
        :param actions: Batch of actions
        :param rewards: Batch of rewards
        :param next_states: Batch of next states
        :param dones: Batch of done flags
        :param gamma: Discount factor
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        # Compute current and next state values
        current_values = self.critic(states)
        next_values = self.critic(next_states)

        # Compute target values (TD target)
        target_values = rewards + gamma * next_values * (~dones)
        target_values = target_values.detach()

        # Compute advantage
        advantages = target_values - current_values

        # Update Critic
        critic_loss = nn.MSELoss()(current_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        log_probs = self.compute_action_log_prob(states, actions.unsqueeze(1))
        actor_loss = -(log_probs * advantages.detach().squeeze()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_value": current_values.mean().item(),
            "mean_advantage": advantages.mean().item(),
        }

import numpy as np
import torch

from utils.tools import lambda_return
from .base_agent import BaseAgent
from networks.mlp import MLP


class ActorCriticAgent(BaseAgent):
    """
    An ANN-based Actor-Critic agent.
    Uses separate ANN for the actor (policy) and critic (value function).
    """

    def __init__(
        self,
        config,
        state_dim=6,
        action_dim=2,
        lr=1e-3,
    ):
        """
        Initialize the Actor-Critic agent.

        :param config: Configuration object
        :param state_dim: Dimension of the state space
        :param action_dim: Dimension of the action space
        :param lr: Learning rate for the optimizers
        """
        # Initialize Actor and Critic
        self.actor = MLP(
            inp_dim=state_dim,
            shape=(action_dim,),
            layers=config.actor["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.actor["dist"],
            std=config.actor["std"],
            min_std=config.actor["min_std"],
            max_std=config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = MLP(
            inp_dim=state_dim,
            shape=(255,) if config.critic["dist"] == "symlog_disc" else (),
            layers=config.critic["layers"],
            units=config.units,
            act=config.act,
            norm=config.norm,
            dist=config.critic["dist"],
            outscale=config.critic["outscale"],
            name="Value",
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )  # TODO test if neuron params are included in the optimizer
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        # Set to training mode
        self.actor.train()
        self.value.train()

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
            value_out = self.critic(state_tensor)

        return value_out.squeeze().item()

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

    def load_agent_states(self):
        self.actor.set_states(self.actor_states)

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

        values = []
        for i in range(sequence_length):
            value = self.critic(states[i])
            values.append(value)

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
        critic_loss = torch.nn.functional.mse_loss(
            values[:-1], target_values.detach(), reduction="none"
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

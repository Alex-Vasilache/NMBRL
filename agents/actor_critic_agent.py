import copy
import numpy as np
import torch
import torch.nn as nn

from utils.tools import Optimizer, RewardEMA
from networks.mlp import MLP


class ActorCriticAgent(nn.Module):
    """
    An ANN-based Actor-Critic agent.
    Uses separate ANN for the actor (policy) and critic (value function).
    """

    def __init__(
        self,
        config,
        state_dim=6,
        action_dim=(2,),
    ):
        """
        Initialize the Actor-Critic agent.

        :param config: Configuration object
        :param state_dim: Dimension of the state space
        :param action_dim: Dimension of the action space
        :param lr: Learning rate for the optimizers
        """
        super().__init__()
        self.config = config
        # Initialize Actor and Critic
        self.actor = MLP(
            inp_dim=state_dim,
            shape=action_dim,
            layers=config["actor"]["layers"],
            units=config["units"],
            act=config["act"],
            norm=config["norm"],
            dist=config["actor"]["dist"],
            std=config["actor"]["std"],
            min_std=config["actor"]["min_std"],
            max_std=config["actor"]["max_std"],
            absmax=1.0,
            temp=config["actor"]["temp"],
            unimix_ratio=config["actor"]["unimix_ratio"],
            outscale=config["actor"]["outscale"],
            name="Actor",
            device=config["device"],
        )
        self.critic = MLP(
            inp_dim=state_dim,
            shape=(255,) if config["critic"]["dist"] == "symlog_disc" else (),
            layers=config["critic"]["layers"],
            units=config["units"],
            act=config["act"],
            norm=config["norm"],
            dist=config["critic"]["dist"],
            outscale=config["critic"]["outscale"],
            name="Value",
            device=config["device"],
        )

        # Optimizers
        self._actor_opt = Optimizer(
            "actor",
            self.actor.parameters(),
            float(config["learning_rate"]),
            float(config["eps"]),
            float(config["grad_clip"]),
        )

        self._value_opt = Optimizer(
            "value",
            self.critic.parameters(),
            float(config["learning_rate"]),
            float(config["eps"]),
            float(config["grad_clip"]),
        )

        if config["critic"]["slow_target"]:
            self._slow_value = copy.deepcopy(self.critic)
            self._updates = 0

        if self.config["reward_EMA"]:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self.config["device"])
            )
            self.reward_ema = RewardEMA(device=self.config["device"])

        # Set to training mode
        self.actor.train()
        self.critic.train()

    def get_action(self, state, deterministic=False):
        """
        Get an action from the actor network.
        """
        # Ensure state is a tensor
        if isinstance(state, np.ndarray) or isinstance(state, list):
            state = torch.tensor(
                state, dtype=torch.float32, device=self.config["device"]
            )
        else:
            # ensure it is on device
            state = state.to(self.config["device"])
        return self.actor(state).mode() if deterministic else self.actor(state).sample()

    def _update_slow_target(self):
        if self.config["critic"]["slow_target"]:
            if self._updates % self.config["critic"]["slow_target_update"] == 0:
                mix = self.config["critic"]["slow_target_fraction"]
                for s, d in zip(
                    self.critic.parameters(), self._slow_value.parameters()
                ):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def update(
        self,
        states,
        actions,
        rewards,
    ):
        """
        Update both actor and critic networks using imagined trajectories.

        :param states: Batch of states, shape is (sequence_length, batch_size, state_dim)
        :param actions: Batch of actions, shape is (sequence_length, batch_size, action_dim)
        :param rewards: Batch of rewards, shape is (sequence_length, batch_size)
        """
        self._update_slow_target()
        batch_size = states.shape[1]
        sequence_length = states.shape[0]

        discount = self.config["gamma"] * torch.ones_like(rewards)

        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()

        value_dists = self.critic(states.detach())
        values = value_dists.mode()

        lambda_returns = [0] * sequence_length
        lambda_returns[-1] = values[-1]

        for i in range(sequence_length - 2, -1, -1):
            lambda_returns[i] = rewards[i] + self.config["gamma"] * (
                (1 - self.config["discount_lambda"]) * values[i + 1]
                + self.config["discount_lambda"] * lambda_returns[i + 1]
            )

        lambda_returns = torch.stack(lambda_returns)  # [sequence_length, batch_size, 1]

        # These targets train the critic
        critic_losses = -value_dists.log_prob(
            lambda_returns.detach()
        )  # [sequence_length, batch_size]

        if self.config["critic"]["slow_target"]:
            slow_target = self._slow_value(states.detach())
            critic_losses -= value_dists.log_prob(slow_target.mode().detach())

        critic_losses = critic_losses.unsqueeze(-1)  # [sequence_length, batch_size, 1]
        if self.config["loss_aggregation"] == "mean":
            critic_loss = torch.mean(critic_losses[:-1] * weights[:-1])
        elif self.config["loss_aggregation"] == "sum":
            critic_loss = torch.sum(critic_losses[:-1] * weights[:-1])
        else:
            raise ValueError(
                f"Invalid loss aggregation method: {self.config['loss_aggregation']}"
            )

        if self.config["reward_EMA"]:
            offset, scale = self.reward_ema(lambda_returns[:-1], self.ema_vals)
            normed_target = (lambda_returns[:-1] - offset) / scale
            normed_base = (values[:-1] - offset) / scale
            advantages = normed_target - normed_base
        else:
            advantages = lambda_returns[:-1] - values[:-1]

        # Compute log probs
        policy_dists = self.actor(states.detach())
        log_probs = policy_dists.log_prob(actions)[
            :-1
        ]  # [sequence_length - 1, batch_size]

        log_probs = log_probs.unsqueeze(-1)  # [sequence_length - 1, batch_size, 1]

        # Compute entropy
        entropy_loss = (
            policy_dists.entropy()[:-1] * float(self.config["actor"]["entropy"])
        ).unsqueeze(
            -1
        )  # [sequence_length - 1, batch_size, 1]

        if self.config["actor"]["imag_grad"] == "dynamics":
            actor_target = advantages
        else:
            actor_target = advantages.detach() * log_probs

        # Separate policy gradient loss from entropy bonus
        policy_gradient_loss = -(actor_target * weights[:-1])
        if self.config["loss_aggregation"] == "mean":
            policy_gradient_loss_mean = torch.mean(policy_gradient_loss)
            entropy_bonus_mean = torch.mean(entropy_loss)
        elif self.config["loss_aggregation"] == "sum":
            policy_gradient_loss_mean = torch.sum(policy_gradient_loss)
            entropy_bonus_mean = torch.sum(entropy_loss)
        else:
            raise ValueError(
                f"Invalid loss aggregation method: {self.config['loss_aggregation']}"
            )

        actor_loss = (
            policy_gradient_loss_mean - entropy_bonus_mean
        )  # entropy_loss already includes entropy coefficient

        self._actor_opt(actor_loss, self.actor.parameters())
        self._value_opt(critic_loss, self.critic.parameters())

        # Calculate additional metrics for detailed logging
        mean_reward = torch.mean(rewards)
        mean_lambda_return = torch.mean(lambda_returns[:-1])
        std_advantage = torch.std(advantages)
        mean_entropy = torch.mean(policy_dists.entropy()[:-1])

        # Calculate slow target loss separately if enabled
        slow_target_loss = torch.tensor(0.0, device=self.config["device"])
        if self.config["critic"]["slow_target"]:
            # Use the same loss calculation as the main critic loss but for slow target
            slow_target_log_prob = -value_dists.log_prob(slow_target.mode().detach())
            slow_target_loss_full = slow_target_log_prob.unsqueeze(
                -1
            )  # [sequence_length, batch_size, 1]
            slow_target_loss = torch.mean(slow_target_loss_full[:-1] * weights[:-1])

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "policy_gradient_loss": policy_gradient_loss_mean.item(),
            "entropy_bonus": entropy_bonus_mean.item(),
            "slow_target_loss": slow_target_loss.item(),
            "mean_value": values.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "std_advantage": std_advantage.item(),
            "mean_reward": mean_reward.item(),
            "mean_lambda_return": mean_lambda_return.item(),
            "mean_entropy": mean_entropy.item(),
        }

# This file implements the training loop for the Actor-Critic agent.
# It manages the agent's interaction with the world model (real or learned),
# and applies learning updates to the SNN-based agent's networks.

import numpy as np
import torch
from collections import deque
import sys
import os

# Add project root to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world_models.ini_cartpole_wrapper import INICartPoleWrapper
from agents.snn_actor_critic_agent import SnnActorCriticAgent


class ActorCriticTrainer:
    """
    Trainer for the SNN Actor-Critic agent.
    Manages the training loop, including agent-environment interaction and learning updates.
    """

    def __init__(self, config):
        """
        Initializes the trainer.

        :param config: A dictionary containing training parameters.
        """
        self.config = config
        self.world_model = INICartPoleWrapper()

        # Initialize the SNN agent
        action_space = self.world_model.env.action_space
        self.agent = SnnActorCriticAgent(
            action_space=action_space,
            state_dim=config.get("state_dim", 6),
            hidden_dim=config.get("hidden_dim", 128),
            num_steps=config.get("snn_time_steps", 1),
            lr=config.get("learning_rate", 1e-3),
            alpha=config.get("alpha", 0.9),
            beta=config.get("beta", 0.9),
            threshold=config.get("threshold", 1),
            learn_alpha=config.get("learn_alpha", True),
            learn_beta=config.get("learn_beta", True),
            learn_threshold=config.get("learn_threshold", True),
        )

        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.buffer_seq_length = config.get("buffer_seq_length", 20)  # Sequence length for SNN training
        self.update_frequency = config.get("update_frequency", 10)
        self.gamma = config.get("gamma", 0.99)

        # Experience buffer for batch training
        self.experience_buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        self.viable_sequence_starts = []

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

    def collect_experience(self, state, action, reward, next_state, done):
        """Store experience in the buffer."""
        self.experience_buffer["states"].append(state.copy())
        self.experience_buffer["actions"].append(action.copy())
        self.experience_buffer["rewards"].append(reward)
        self.experience_buffer["next_states"].append(next_state.copy())
        self.experience_buffer["dones"].append(done)
        self.viable_sequence_starts.append(len(self.experience_buffer["states"]) - 1)

    def can_train(self):
        """Check if we have enough experience to perform training."""
        # Need enough data to sample unique overlapping batch_size sequences of length buffer_seq_length
        min_buffer_size = self.buffer_seq_length + self.batch_size - 1
        return len(self.viable_sequence_starts) >= min_buffer_size

    def sample_batch(self):
        """
        Sample a batch of overlapping but unique sequences for SNN training.
        Returns batch_size sequences of length buffer_seq_length..
        """
        buffer_size = len(self.viable_sequence_starts)

        # Calculate maximum possible starting positions for sequences
        max_start_idx = max(self.viable_sequence_starts) - self.buffer_seq_length + 1

        max_viable_start_idx = 0
        while self.viable_sequence_starts[max_viable_start_idx] <= max_start_idx:
            max_viable_start_idx += 1
        
        # Randomly select unique sequence starting positions 
        sequence_starts = np.random.choice(
            self.viable_sequence_starts[:max_viable_start_idx], 
            size=self.batch_size, 
            replace=False
        )

        # Extract sequences
        batch_sequences = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }

        for start_idx in sequence_starts:
            # Extract sequence of length buffer_seq_length starting from start_idx
            seq_states = []
            seq_actions = []
            seq_rewards = []
            seq_next_states = []
            seq_dones = []
            
            for i in range(self.buffer_seq_length):
                idx = start_idx + i
                seq_states.append(self.experience_buffer["states"][idx])
                seq_actions.append(self.experience_buffer["actions"][idx])
                seq_rewards.append(self.experience_buffer["rewards"][idx])
                seq_next_states.append(self.experience_buffer["next_states"][idx])
                seq_dones.append(self.experience_buffer["dones"][idx])
            
            batch_sequences["states"].append(np.array(seq_states))
            batch_sequences["actions"].append(np.array(seq_actions))
            batch_sequences["rewards"].append(np.array(seq_rewards))
            batch_sequences["next_states"].append(np.array(seq_next_states))
            batch_sequences["dones"].append(np.array(seq_dones))

        # Remove the sequences from the viable starting positions
        for start_idx in sequence_starts:
            self.viable_sequence_starts.remove(start_idx)

        # Convert to numpy arrays
        # Shape: (batch_size, sequence_length, feature_dim) -> (sequence_length, batch_size, feature_dim)
        batch = {
            "states": np.array(batch_sequences["states"]).transpose(1, 0, 2),
            "actions": np.array(batch_sequences["actions"]).transpose(1, 0, 2),
            "rewards": np.array(batch_sequences["rewards"]).transpose(1, 0),
            "next_states": np.array(batch_sequences["next_states"]).transpose(1, 0, 2),
            "dones": np.array(batch_sequences["dones"]).transpose(1, 0),
        }

        return batch

    def train_agent(self):
        """Perform a training step on the agent using collected experience."""
        if not self.can_train():
            return None

        batch = self.sample_batch()

        # Update the agent with sequences
        losses = self.agent.update(
            states=batch["states"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            next_states=batch["next_states"],
            dones=batch["dones"],
            gamma=self.gamma,
        )

        self.training_losses.append(losses)
        return losses

    def train(self):
        """
        Runs the main training loop with proper learning updates.
        """
        num_episodes = self.config.get("num_episodes", 100)
        max_steps_per_episode = self.config.get("max_steps_per_episode", 1000)

        print(f"Starting training for {num_episodes} episodes...")
        print(
            f"Agent configuration: SNN with {self.agent.hidden_dim} hidden units, {self.agent.num_steps} time steps"
        )

        for episode in range(num_episodes):
            state = self.world_model.reset()
            terminated = False
            total_reward = 0
            step_count = 0

            self.agent.actor.reset()
            self.agent.critic.reset()

            # Episode rollout
            while not terminated and step_count < max_steps_per_episode:
                # Get action from agent
                action = self.agent.get_action(state)

                # Environment step
                next_state, reward, terminated, info = self.world_model.step(action)

                # Store experience
                self.collect_experience(state, action, reward, next_state, terminated)

                # Periodic training updates
                if step_count % self.update_frequency == 0 and self.can_train():
                    # We save the agent states before training to avoid forgetting the current membrane potentials and spikes, because during training the agent is reset
                    self.agent.save_agent_states()
                    losses = self.train_agent()
                    self.agent.load_agent_states()
                    if losses and step_count % (self.update_frequency * 5) == 0:
                        print(
                            f"  Step {step_count}: Actor Loss: {losses['actor_loss']:.4f}, "
                            f"Critic Loss: {losses['critic_loss']:.4f}, "
                            f"Mean Value: {losses['mean_value']:.4f}"
                        )

                state = next_state
                total_reward += reward
                step_count += 1

            # Episode summary
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step_count)

            # Print episode results
            if episode % 10 == 0 or episode < 10:
                avg_reward = (
                    np.mean(self.episode_rewards[-10:])
                    if len(self.episode_rewards) >= 10
                    else np.mean(self.episode_rewards)
                )
                print(
                    f"Episode {episode+1:3d}: Reward: {total_reward:8.2f}, Steps: {step_count:4d}, "
                    f"Avg Reward (last 10): {avg_reward:8.2f}"
                )

        # Final training statistics
        print("\n=== Training Complete ===")
        print(f"Average Episode Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"Average Episode Length: {np.mean(self.episode_lengths):.1f}")
        print(f"Best Episode Reward: {np.max(self.episode_rewards):.2f}")

        if self.training_losses:
            final_losses = self.training_losses[-1]
            print(f"Final Actor Loss: {final_losses['actor_loss']:.4f}")
            print(f"Final Critic Loss: {final_losses['critic_loss']:.4f}")

    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained agent without learning updates.
        """
        print(f"\n=== Evaluating Agent over {num_episodes} episodes ===")

        # Set agent to evaluation mode
        self.agent.actor.eval()
        self.agent.critic.eval()

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            state = self.world_model.reset()
            terminated = False
            total_reward = 0
            step_count = 0

            while not terminated and step_count < 1000:
                # Get action (deterministic for evaluation)
                action = self.agent.get_action(state)
                next_state, reward, terminated, info = self.world_model.step(action)

                state = next_state
                total_reward += reward
                step_count += 1

            eval_rewards.append(total_reward)
            eval_lengths.append(step_count)
            print(
                f"Eval Episode {episode+1}: Reward: {total_reward:.2f}, Steps: {step_count}"
            )

        # Set agent back to training mode
        self.agent.actor.train()
        self.agent.critic.train()

        print(f"Evaluation Results:")
        print(
            f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}"
        )
        print(
            f"  Average Length: {np.mean(eval_lengths):.1f} ± {np.std(eval_lengths):.1f}"
        )

        return eval_rewards, eval_lengths


if __name__ == "__main__":
    # Example configuration
    training_config = {
        "num_episodes": 50,
        "batch_size": 32,
        "buffer_seq_length": 20,  # Sequence length for SNN training
        "update_frequency": 10,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_dim": 128,
        "snn_time_steps": 1,
        "buffer_size": 1000,
        "max_steps_per_episode": 1000,
        "alpha": 0.9,
        "beta": 0.9,
        "threshold": 1,
        "learn_alpha": True,
        "learn_beta": True,
        "learn_threshold": True,
    }

    trainer = ActorCriticTrainer(config=training_config)
    trainer.train()
    trainer.evaluate(num_episodes=5)

# This file implements the training loop for the Actor-Critic agent.
# It manages the agent's interaction with the world model (real or learned),
# and applies learning updates to the SNN-based agent's networks.

import numpy as np
import torch
from collections import deque
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

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
        self.world_model = INICartPoleWrapper(
            max_steps=config.get("max_steps_per_episode", 10000),
            visualize=config.get("visualize", False),
            dt_simulation=config.get("dt_simulation", 0.02),
        )

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
            weight_init_mean=config.get("weight_init_mean", 0.0),
            weight_init_std=config.get("weight_init_std", 0.01),
            max_std=config.get("max_std", 2.0),
            min_std=config.get("min_std", 0.1),
        )

        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.buffer_seq_length = config.get("buffer_seq_length", 15)
        self.update_frequency = config.get("update_frequency", 10)
        self.gamma = config.get("gamma", 0.997)
        self.discount_lambda = config.get("discount_lambda", 0.95)

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

        # Model saving directory
        self.save_dir = config.get("save_dir", "saved_models")
        os.makedirs(self.save_dir, exist_ok=True)

        # TensorBoard logging setup
        self.log_dir = config.get("log_dir", "runs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_log_dir = os.path.join(self.log_dir, f"snn_actor_critic_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.global_step = 0
        self.training_step = 0

        print(f"TensorBoard logs will be saved to: {self.tb_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {self.log_dir}")

        # Log hyperparameters to TensorBoard
        hparam_dict = {
            "learning_rate": config.get("learning_rate", 1e-3),
            "batch_size": self.batch_size,
            "buffer_seq_length": self.buffer_seq_length,
            "update_frequency": self.update_frequency,
            "gamma": self.gamma,
            "discount_lambda": self.discount_lambda,
            "hidden_dim": self.agent.hidden_dim,
            "snn_time_steps": self.agent.num_steps,
            "alpha": config.get("alpha", 0.9),
            "beta": config.get("beta", 0.9),
            "threshold": config.get("threshold", 1),
            "weight_init_mean": config.get("weight_init_mean", 0.0),
            "weight_init_std": config.get("weight_init_std", 0.01),
            "max_std": config.get("max_std", 2.0),
            "min_std": config.get("min_std", 0.1),
        }
        metric_dict = {
            "final_avg_reward": 0.0,  # Will be updated at the end of training
            "final_best_reward": 0.0,
            "total_episodes": config.get("num_episodes", 100),
        }
        self.writer.add_hparams(hparam_dict, metric_dict)

    def collect_experience(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """Store experience in the buffer."""
        self.experience_buffer["states"].append(state.copy())  # s at [t]
        self.experience_buffer["actions"].append(action.copy())  # a at [t]
        self.experience_buffer["rewards"].append(reward)  # r at [t]
        self.experience_buffer["next_states"].append(next_state.copy())  # s at [t+1]
        self.experience_buffer["dones"].append(done)  # done at [t]
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
            replace=False,
        )

        # Extract sequences
        batch_sequences = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
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
            discount_lambda=self.discount_lambda,
        )

        self.training_losses.append(losses)

        # Log losses to TensorBoard
        if losses:
            self.writer.add_scalar(
                "Loss/Actor", losses["actor_loss"], self.training_step
            )
            self.writer.add_scalar(
                "Loss/Critic", losses["critic_loss"], self.training_step
            )
            self.writer.add_scalar(
                "Value/Mean_Value", losses["mean_value"], self.training_step
            )
            self.training_step += 1

        return losses

    def save_models(self, episode=None):
        """
        Save the trained actor and critic models along with training configuration.

        :param episode: Optional episode number to include in filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if episode is not None:
            model_name = f"snn_actor_critic_ep{episode}_{timestamp}"
        else:
            model_name = f"snn_actor_critic_final_{timestamp}"

        model_path = os.path.join(self.save_dir, model_name)
        os.makedirs(model_path, exist_ok=True)

        # Save actor model
        torch.save(
            {
                "model_state_dict": self.agent.actor.state_dict(),
                "optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
                "model_config": {
                    "state_dim": self.agent.state_dim,
                    "hidden_dim": self.agent.hidden_dim,
                    "action_dim": self.agent.action_dim,
                    "num_steps": self.agent.num_steps,
                    "weight_init_mean": self.agent.weight_init_mean,
                    "weight_init_std": self.agent.weight_init_std,
                },
            },
            os.path.join(model_path, "actor.pth"),
        )

        # Save critic model
        torch.save(
            {
                "model_state_dict": self.agent.critic.state_dict(),
                "optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
                "model_config": {
                    "state_dim": self.agent.state_dim,
                    "hidden_dim": self.agent.hidden_dim,
                    "output_dim": 1,
                    "weight_init_mean": self.agent.weight_init_mean,
                    "weight_init_std": self.agent.weight_init_std,
                },
            },
            os.path.join(model_path, "critic.pth"),
        )

        # Save training configuration and statistics
        training_info = {
            "config": self.config,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "final_stats": {
                "avg_episode_reward": (
                    np.mean(self.episode_rewards) if self.episode_rewards else 0
                ),
                "avg_episode_length": (
                    np.mean(self.episode_lengths) if self.episode_lengths else 0
                ),
                "best_episode_reward": (
                    np.max(self.episode_rewards) if self.episode_rewards else 0
                ),
                "total_episodes": len(self.episode_rewards),
            },
        }

        torch.save(training_info, os.path.join(model_path, "training_info.pth"))

        print(f"Models saved to: {model_path}")
        return model_path

    def train(self):
        """
        Runs the main training loop with proper learning updates.
        """
        num_episodes = self.config.get("num_episodes", 100)
        max_steps_per_episode = self.config.get("max_steps_per_episode", 10000)
        save_frequency = self.config.get(
            "save_frequency", None
        )  # Save every N episodes

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

            # Episode rollout
            while not terminated and step_count < max_steps_per_episode:
                # Get action from agent

                action = self.agent.get_action(state)

                # Environment step
                next_state, reward, terminated, info = self.world_model.step(action)

                # Store experience
                self.collect_experience(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                )

                # Periodic training updates
                if step_count % self.update_frequency == 0 and self.can_train():
                    # We save the agent states before training to avoid forgetting the current membrane potentials and spikes, because during training the agent is reset
                    # self.agent.save_agent_states()
                    losses = self.train_agent()
                    # self.agent.load_agent_states()
                    self.agent.actor.reset()
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

            # Log episode metrics to TensorBoard
            self.writer.add_scalar("Episode/Reward", total_reward, episode)
            self.writer.add_scalar("Episode/Length", step_count, episode)
            self.writer.add_scalar("Episode/Global_Step", self.global_step, episode)

            # Calculate and log running averages
            if len(self.episode_rewards) >= 10:
                avg_reward_10 = np.mean(self.episode_rewards[-10:])
                avg_length_10 = np.mean(self.episode_lengths[-10:])
                self.writer.add_scalar("Episode/Avg_Reward_10", avg_reward_10, episode)
                self.writer.add_scalar("Episode/Avg_Length_10", avg_length_10, episode)

            if len(self.episode_rewards) >= 100:
                avg_reward_100 = np.mean(self.episode_rewards[-100:])
                avg_length_100 = np.mean(self.episode_lengths[-100:])
                self.writer.add_scalar(
                    "Episode/Avg_Reward_100", avg_reward_100, episode
                )
                self.writer.add_scalar(
                    "Episode/Avg_Length_100", avg_length_100, episode
                )

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

            self.global_step += step_count

            # Save models periodically if requested
            if save_frequency and (episode + 1) % save_frequency == 0:
                self.save_models(episode=episode + 1)

        # Final training statistics
        print("\n=== Training Complete ===")
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        best_reward = np.max(self.episode_rewards)

        print(f"Average Episode Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Best Episode Reward: {best_reward:.2f}")

        # Log final training statistics to TensorBoard
        self.writer.add_scalar("Training/Final_Avg_Reward", avg_reward, num_episodes)
        self.writer.add_scalar("Training/Final_Avg_Length", avg_length, num_episodes)
        self.writer.add_scalar("Training/Best_Reward", best_reward, num_episodes)
        self.writer.add_scalar("Training/Total_Steps", self.global_step, num_episodes)

        if self.training_losses:
            final_losses = self.training_losses[-1]
            print(f"Final Actor Loss: {final_losses['actor_loss']:.4f}")
            print(f"Final Critic Loss: {final_losses['critic_loss']:.4f}")

            self.writer.add_scalar(
                "Training/Final_Actor_Loss", final_losses["actor_loss"], num_episodes
            )
            self.writer.add_scalar(
                "Training/Final_Critic_Loss", final_losses["critic_loss"], num_episodes
            )

        # Close TensorBoard writer
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.tb_log_dir}")

        # Save final models
        final_model_path = self.save_models()
        print(f"Final models saved to: {final_model_path}")

        self.world_model.close()

        return final_model_path

    def close_tensorboard(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
            print(f"TensorBoard writer closed. Logs saved to: {self.tb_log_dir}")

    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained agent without learning updates.
        """
        print(f"\n=== Evaluating Agent over {num_episodes} episodes ===")

        # Set agent to evaluation mode
        self.agent.actor.eval()
        self.agent.critic.eval()
        self.agent.actor.reset()

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

        # Calculate evaluation statistics
        eval_avg_reward = np.mean(eval_rewards)
        eval_std_reward = np.std(eval_rewards)
        eval_avg_length = np.mean(eval_lengths)
        eval_std_length = np.std(eval_lengths)

        print(f"Evaluation Results:")
        print(f"  Average Reward: {eval_avg_reward:.2f} ± {eval_std_reward:.2f}")
        print(f"  Average Length: {eval_avg_length:.1f} ± {eval_std_length:.1f}")

        # Log evaluation results to TensorBoard
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_scalar("Evaluation/Avg_Reward", eval_avg_reward, 0)
            self.writer.add_scalar("Evaluation/Std_Reward", eval_std_reward, 0)
            self.writer.add_scalar("Evaluation/Avg_Length", eval_avg_length, 0)
            self.writer.add_scalar("Evaluation/Std_Length", eval_std_length, 0)
            self.writer.add_scalar("Evaluation/Best_Reward", np.max(eval_rewards), 0)
            self.writer.add_scalar("Evaluation/Worst_Reward", np.min(eval_rewards), 0)

        self.world_model.close()

        return eval_rewards, eval_lengths


if __name__ == "__main__":
    # Example configuration
    training_config = {
        "num_episodes": 100,
        "batch_size": 512,
        "buffer_seq_length": 15,  # Trajectory length for λ-returns
        "update_frequency": 3,
        "learning_rate": 1e-5,
        "gamma": 0.997,  # Discount factor for the reward
        "discount_lambda": 0.95,  # return mixing factor
        "hidden_dim": 32,
        "snn_time_steps": 1,
        "max_steps_per_episode": 1000,
        "alpha": 0.9,
        "beta": 0.9,
        "threshold": 0.01,
        "learn_alpha": True,
        "learn_beta": True,
        "learn_threshold": True,
        "visualize": False,
        "dt_simulation": 0.02,
        "weight_init_mean": 0.03,
        "weight_init_std": 0.03,
        "max_std": 1.0,
        "min_std": 0.01,
        "save_dir": "saved_models",  # Directory to save trained models
        "log_dir": "runs",  # Directory for TensorBoard logs
        "save_frequency": 10,  # Save models every N episodes (None to disable)
    }

    trainer = ActorCriticTrainer(config=training_config)
    trainer.train()
    trainer.evaluate(num_episodes=5)

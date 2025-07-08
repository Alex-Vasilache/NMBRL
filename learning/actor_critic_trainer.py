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
import yaml

from world_models.ini_gymlike_cartpole_wrapper import INIGymlikeCartPoleWrapper
from agents.actor_critic_agent import ActorCriticAgent
from typing import Optional, Union
from gymnasium.spaces import Space


# Add project root to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import tools


class ActorCriticTrainer:
    """
    Trainer for the SNN Actor-Critic agent.
    Manages the training loop, including agent-environment interaction and learning updates.
    """

    world_model: INIGymlikeCartPoleWrapper
    action_space: Space
    state_dim: int

    def __init__(self, config):
        """
        Initializes the trainer.

        :param config: A dictionary containing training parameters.
        """
        self.config = config
        self.world_model = INIGymlikeCartPoleWrapper(
            max_steps=config.get("max_steps_per_episode"),
            visualize=config.get("visualize", False),
            task=config.get("task", "swingup"),
            cartpole_type=config.get("cartpole_type", "custom_sim"),
        )

        # Initialize the Actor-Critic agent
        self.action_space = self.world_model.action_space
        assert self.action_space is not None, "Action space cannot be None"
        self.state_dim = self.world_model.observation_space.shape[0]
        assert self.state_dim is not None, "State dimension cannot be None"
        self.agent = ActorCriticAgent(
            config=config,
            state_dim=self.state_dim,
            action_dim=self.action_space.shape,
        )

        # Training parameters
        self.batch_size = config.get("batch_size")
        self.imag_horizon = config.get("imag_horizon")
        self.update_frequency = config.get("update_frequency")
        self.gamma = config.get("gamma")
        self.discount_lambda = config.get("discount_lambda")

        # Imagined trajectories for batch training
        self.imagined_trajectories = {
            "states": [],
            "actions": [],
            "rewards": [],
        }

        self.initial_state_buffer = []

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

        # Model saving directory
        self.save_dir = config.get("save_dir")
        os.makedirs(self.save_dir, exist_ok=True)

        # TensorBoard logging setup
        self.log_dir = config.get("log_dir")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_log_dir = os.path.join(self.log_dir, f"snn_actor_critic_{timestamp}")
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.global_step = 0
        self.training_step = 0

        print(f"TensorBoard logs will be saved to: {self.tb_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {self.log_dir}")

        # Save configuration files to log directory for reproducibility
        self.save_config_to_logdir()

        # Log hyperparameters to TensorBoard
        hparam_dict = {
            "learning_rate": float(config.get("learning_rate")),
            "batch_size": self.batch_size,
            "imag_horizon": self.imag_horizon,
            "update_frequency": self.update_frequency,
            "gamma": self.gamma,
            "discount_lambda": self.discount_lambda,
            "units": config.get("units"),
            "layers": config.get("actor")["layers"],
            "act": config.get("act"),
            "norm": config.get("norm"),
        }
        metric_dict = {
            "final_avg_reward": 0.0,  # Will be updated at the end of training
            "final_best_reward": 0.0,
            "total_episodes": config.get("num_epochs", 100),
        }
        self.writer.add_hparams(hparam_dict, metric_dict)

        self.fill_initial_state_buffer(
            num_states=self.config.get("batch_size") * self.config.get("batch_length"),
        )

    def fill_initial_state_buffer(self, num_states=None):
        # TODO: fill with initial states from the actual environment (replay buffer of the world model)
        """Prefill the initial state buffer with random sampled states."""

        if num_states is None:
            num_states = self.config.get("batch_size") * self.config.get("batch_length")
        # print(f"Filling initial state buffer with {num_states} states...")
        state = self.world_model.reset()  # [batch_size, state_dim]
        sequential_states = []

        for _ in range(num_states // self.config.get("batch_size") // 2):

            actions = np.random.uniform(
                self.action_space.low,
                self.action_space.high,
                size=(self.config.get("batch_size"), *self.action_space.shape),
            )
            next_states, _, _, _ = self.world_model.step(actions)
            sequential_states.extend(state)
            state = next_states

        state = self.world_model.reset()

        for _ in range(num_states // self.config.get("batch_size") // 2):

            state = torch.tensor(state, dtype=torch.float32)
            actions = self.agent.get_action(state).detach().numpy()

            next_states, _, _, _ = self.world_model.step(actions)
            sequential_states.extend(state)
            state = next_states

        # Shuffle the sequential states
        np.random.shuffle(sequential_states)
        self.initial_state_buffer.extend(sequential_states)

    def store_trajectory(
        self,
        state,
        action,
        reward,
    ):
        """Store experience in the buffer."""
        self.imagined_trajectories["states"].append(state.detach().numpy())  # s at [t]
        self.imagined_trajectories["actions"].append(
            action.detach().numpy()
        )  # a at [t]
        self.imagined_trajectories["rewards"].append(reward)  # r at [t]

    def reset_trajectory(self):
        """Reset the trajectory."""
        self.imagined_trajectories = {
            "states": [],
            "actions": [],
            "rewards": [],
        }

    def get_trajectory(self):
        """Get the imagined trajectory."""
        # Convert to arrays without modifying the original dictionary structure
        states_data = self.imagined_trajectories["states"]
        actions_data = self.imagined_trajectories["actions"]
        rewards_data = self.imagined_trajectories["rewards"]

        # Convert to numpy arrays with proper handling of tensor data
        if isinstance(states_data, list):
            # Convert any tensors in the list to numpy first
            states_list = []
            for state in states_data:
                if torch.is_tensor(state):
                    states_list.append(state.detach().cpu().numpy())
                else:
                    states_list.append(np.array(state))
            states_data = np.array(states_list, dtype=np.float32)

        if isinstance(actions_data, list):
            actions_list = []
            for action in actions_data:
                if torch.is_tensor(action):
                    actions_list.append(action.detach().cpu().numpy())
                else:
                    actions_list.append(np.array(action))
            actions_data = np.array(actions_list, dtype=np.float32)

        if isinstance(rewards_data, list):
            rewards_list = []
            for reward in rewards_data:
                if torch.is_tensor(reward):
                    rewards_list.append(reward.detach().cpu().numpy())
                else:
                    rewards_list.append(np.array(reward))
            rewards_data = np.array(rewards_list, dtype=np.float32)

        states = torch.tensor(
            states_data, dtype=torch.float32
        )  # [imag_horizon, batch_size, state_dim]
        actions = torch.tensor(
            actions_data, dtype=torch.float32
        )  # [imag_horizon, batch_size, action_dim]
        rewards = torch.tensor(rewards_data, dtype=torch.float32).unsqueeze(
            -1
        )  # [imag_horizon, batch_size, 1]
        return states, actions, rewards

    def train_agent(self):
        """Perform a training step on the agent using collected experience."""

        states, actions, rewards = self.get_trajectory()

        # Update the agent with sequences
        losses = self.agent.update(states=states, actions=actions, rewards=rewards)

        self.training_losses.append(losses)

        # Log losses to TensorBoard
        if losses and self.training_step % 10 == 0:
            self.writer.add_scalar(
                "Loss/Actor", losses["actor_loss"], self.training_step
            )
            self.writer.add_scalar(
                "Loss/Critic", losses["critic_loss"], self.training_step
            )

            # Log detailed actor loss components
            self.writer.add_scalar(
                "Loss/Policy_Gradient",
                losses["policy_gradient_loss"],
                self.training_step,
            )
            self.writer.add_scalar(
                "Loss/Entropy_Bonus", losses["entropy_bonus"], self.training_step
            )

            # Log critic loss components
            if losses["slow_target_loss"] > 0:  # Only log if slow target is enabled
                self.writer.add_scalar(
                    "Loss/Slow_Target", losses["slow_target_loss"], self.training_step
                )

            # Log value function and advantage statistics
            self.writer.add_scalar(
                "Values/Mean_Value", losses["mean_value"], self.training_step
            )
            self.writer.add_scalar(
                "Values/Mean_Advantage", losses["mean_advantage"], self.training_step
            )
            self.writer.add_scalar(
                "Values/Std_Advantage", losses["std_advantage"], self.training_step
            )
            self.writer.add_scalar(
                "Values/Mean_Lambda_Return",
                losses["mean_lambda_return"],
                self.training_step,
            )

            # Log reward and policy statistics
            self.writer.add_scalar(
                "Stats/Mean_Reward", losses["mean_reward"], self.training_step
            )
            self.writer.add_scalar(
                "Stats/Mean_Entropy", losses["mean_entropy"], self.training_step
            )
        self.training_step += 1

        return losses

    def save_models(self, epoch=None):
        """
        Save the trained actor and critic models along with training configuration.

        :param epoch: Optional epoch number to include in filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if epoch is not None:
            model_name = f"snn_actor_critic_ep{epoch}_{timestamp}"
        else:
            model_name = f"snn_actor_critic_final_{timestamp}"

        model_path = os.path.join(self.save_dir, model_name)
        os.makedirs(model_path, exist_ok=True)

        # Save actor model
        torch.save(
            {
                "model_state_dict": self.agent.actor.state_dict(),
                "model_config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_space,
                },
            },
            os.path.join(model_path, "actor.pth"),
        )

        # Save critic model
        torch.save(
            {
                "model_state_dict": self.agent.critic.state_dict(),
                "model_config": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_space,
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

    def pop_initial_state(self, batch_size):
        """Pop a batch of initial states from the initial state buffer."""

        if len(self.initial_state_buffer) < batch_size:
            self.fill_initial_state_buffer(
                num_states=batch_size * self.config.get("batch_length"),
            )

        return_value = self.initial_state_buffer[:batch_size]
        self.initial_state_buffer = self.initial_state_buffer[batch_size:]
        # Convert to numpy array first to avoid tensor creation warning
        return_value = np.array(return_value)
        return_value = torch.tensor(return_value, dtype=torch.float32)
        return return_value

    def train(self):
        """
        Runs the main training loop with proper learning updates.
        """
        num_epochs = self.config.get("num_epochs")
        max_steps_per_episode = self.config.get("max_steps_per_episode")
        save_frequency = self.config.get("save_frequency")  # Save every N episodes

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.reset_trajectory()

            states = self.pop_initial_state(self.batch_size)

            states = self.world_model.reset(
                batch_size=self.batch_size, initial_state=states
            )

            for i in range(self.imag_horizon):
                states = torch.tensor(states, dtype=torch.float32)
                actions = self.agent.get_action(states)
                next_states, rewards, terminated, info = self.world_model.step(
                    actions.detach().numpy()
                )
                self.store_trajectory(
                    states,
                    actions,
                    rewards,
                )
                states = next_states

            losses = self.train_agent()
            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch}: Actor Loss: {losses['actor_loss']:.4f}, "
                    f"Critic Loss: {losses['critic_loss']:.4f}, "
                    f"Policy Grad: {losses['policy_gradient_loss']:.4f}, "
                    f"Entropy: {losses['entropy_bonus']:.4f}, "
                    f"Mean Advantage: {losses['mean_advantage']:.4f}"
                )
            # Save models and evaluate periodically if requested
            if save_frequency and (epoch + 1) % save_frequency == 0:
                self.save_models(epoch=epoch + 1)
                # Run intermediate evaluation
                self.run_intermediate_evaluation(epoch + 1)

        # Final training statistics
        print("\n=== Training Complete ===")

        # Log final training statistics to TensorBoard
        if self.training_losses:
            final_losses = self.training_losses[-1]
            print(f"Final Actor Loss: {final_losses['actor_loss']:.4f}")
            print(f"Final Critic Loss: {final_losses['critic_loss']:.4f}")
            print(
                f"Final Policy Gradient Loss: {final_losses['policy_gradient_loss']:.4f}"
            )
            print(f"Final Entropy Bonus: {final_losses['entropy_bonus']:.4f}")
            print(f"Final Mean Advantage: {final_losses['mean_advantage']:.4f}")
            print(f"Final Mean Entropy: {final_losses['mean_entropy']:.4f}")

            self.writer.add_scalar(
                "Training/Final_Actor_Loss", final_losses["actor_loss"], num_epochs
            )
            self.writer.add_scalar(
                "Training/Final_Critic_Loss", final_losses["critic_loss"], num_epochs
            )
            self.writer.add_scalar(
                "Training/Final_Policy_Gradient",
                final_losses["policy_gradient_loss"],
                num_epochs,
            )
            self.writer.add_scalar(
                "Training/Final_Entropy_Bonus",
                final_losses["entropy_bonus"],
                num_epochs,
            )
            self.writer.add_scalar(
                "Training/Final_Mean_Advantage",
                final_losses["mean_advantage"],
                num_epochs,
            )
            self.writer.add_scalar(
                "Training/Final_Mean_Entropy", final_losses["mean_entropy"], num_epochs
            )

        # Close TensorBoard writer
        self.writer.close()
        print(f"TensorBoard logs saved to: {self.tb_log_dir}")

        # Save final models
        final_model_path = self.save_models()
        print(f"Final models saved to: {final_model_path}")

        # Run final evaluation
        print("\n=== Running Final Evaluation ===")
        final_eval_episodes = self.config.get("final_eval_episodes", 10)
        eval_rewards, eval_lengths = self.evaluate(
            num_episodes=final_eval_episodes, visualize=False
        )

        # Log final evaluation to TensorBoard
        final_avg_reward = np.mean(eval_rewards)
        final_std_reward = np.std(eval_rewards)
        self.writer.add_scalar(
            "Evaluation/Final_Avg_Reward", final_avg_reward, num_epochs
        )
        self.writer.add_scalar(
            "Evaluation/Final_Std_Reward", final_std_reward, num_epochs
        )

        self.world_model.close()

        return final_model_path

    def run_intermediate_evaluation(self, epoch):
        """
        Run evaluation during training and log results to TensorBoard.

        :param epoch: Current training epoch
        """
        # Get evaluation parameters from config
        eval_episodes = self.config.get("eval_episodes", 5)
        eval_visualize = self.config.get("eval_visualize", False)

        print(f"\n--- Running Intermediate Evaluation at Epoch {epoch} ---")

        # Set agent to evaluation mode
        original_actor_mode = self.agent.actor.training
        original_critic_mode = self.agent.critic.training
        self.agent.actor.eval()
        self.agent.critic.eval()

        # Store original world model state if it has visualization
        original_visualize = None
        if hasattr(self.world_model, "visualize"):
            original_visualize = self.world_model.visualize
            # Only set visualization if it's safe to do so
            if not eval_visualize or hasattr(self.world_model, "_init_visualization"):
                self.world_model.visualize = eval_visualize

        original_batch_size = self.world_model.batch_size

        try:
            eval_rewards = []
            eval_lengths = []

            for episode in range(eval_episodes):
                state = self.world_model.reset(batch_size=1)
                terminated = False
                total_reward = 0
                step_count = 0

                while not terminated and step_count < 1000:
                    # Get action (deterministic for evaluation)
                    state = torch.tensor(state, dtype=torch.float32)
                    action = self.agent.get_action(state).detach().numpy()
                    next_state, reward, terminated, info = self.world_model.step(action)

                    state = next_state
                    # Extract scalar reward from array
                    reward_scalar = (
                        reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                    )
                    total_reward += reward_scalar
                    step_count += 1

                    # Check if episode terminated (extract from array if needed)
                    terminated = (
                        terminated[0]
                        if isinstance(terminated, (list, np.ndarray))
                        else terminated
                    )

                eval_rewards.append(total_reward)
                eval_lengths.append(step_count)

            # Calculate statistics
            avg_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            avg_length = np.mean(eval_lengths)
            best_reward = np.max(eval_rewards)
            worst_reward = np.min(eval_rewards)

            # Log to TensorBoard
            self.writer.add_scalar(
                "Evaluation/Intermediate_Avg_Reward", avg_reward, epoch
            )
            self.writer.add_scalar(
                "Evaluation/Intermediate_Std_Reward", std_reward, epoch
            )
            self.writer.add_scalar(
                "Evaluation/Intermediate_Avg_Length", avg_length, epoch
            )
            self.writer.add_scalar(
                "Evaluation/Intermediate_Best_Reward", best_reward, epoch
            )
            self.writer.add_scalar(
                "Evaluation/Intermediate_Worst_Reward", worst_reward, epoch
            )

            # Print results
            print(f"  Evaluation Results (Epoch {epoch}):")
            print(f"    Episodes: {eval_episodes}")
            print(f"    Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"    Average Length: {avg_length:.1f}")
            print(f"    Best/Worst Reward: {best_reward:.3f} / {worst_reward:.3f}")

        finally:
            # Restore original modes
            self.agent.actor.train()
            self.agent.critic.train()

            # Restore original visualization state
            if hasattr(self.world_model, "visualize"):
                self.world_model.visualize = original_visualize

            # Restore original batch size
            self.world_model.reset(batch_size=original_batch_size)

        print(f"--- Intermediate Evaluation Complete ---\n")

    def close_tensorboard(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
            print(f"TensorBoard writer closed. Logs saved to: {self.tb_log_dir}")

    def evaluate(self, num_episodes=10, visualize=False):
        """
        Evaluate the trained agent without learning updates.
        """
        print(f"\n=== Evaluating Agent over {num_episodes} episodes ===")

        # Set agent to evaluation mode
        self.agent.actor.eval()
        self.agent.critic.eval()

        # Set visualization if the world model supports it
        if hasattr(self.world_model, "visualize"):
            original_visualize = self.world_model.visualize
            self.world_model.visualize = visualize

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            state = self.world_model.reset(batch_size=1)
            terminated = False
            total_reward = 0
            step_count = 0

            while not terminated and step_count < 1000:
                # Get action (deterministic for evaluation)
                state = torch.tensor(state, dtype=torch.float32)
                action = self.agent.get_action(state).detach().numpy()
                next_state, reward, terminated, info = self.world_model.step(action)

                state = next_state
                # Extract scalar reward from array
                reward_scalar = (
                    reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                )
                total_reward += reward_scalar
                step_count += 1

                # Check if episode terminated (extract from array if needed)
                terminated = (
                    terminated[0]
                    if isinstance(terminated, (list, np.ndarray))
                    else terminated
                )

            eval_rewards.append(total_reward)
            eval_lengths.append(step_count)
            print(
                f"Eval Episode {episode+1}: Reward: {total_reward:.2f}, Steps: {step_count}"
            )

        # Set agent back to training mode
        self.agent.actor.train()
        self.agent.critic.train()

        # Restore original visualization setting
        if original_visualize is not None and hasattr(self.world_model, "visualize"):
            self.world_model.visualize = original_visualize

        # Calculate evaluation statistics
        eval_avg_reward = np.mean(eval_rewards)
        eval_std_reward = np.std(eval_rewards)

        print(f"Evaluation Results:")
        print(f"  Average Reward: {eval_avg_reward:.2f} ± {eval_std_reward:.2f}")

        # Log evaluation results to TensorBoard
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_scalar("Evaluation/Avg_Reward", eval_avg_reward, 0)
            self.writer.add_scalar("Evaluation/Std_Reward", eval_std_reward, 0)
            self.writer.add_scalar("Evaluation/Best_Reward", np.max(eval_rewards), 0)
            self.writer.add_scalar("Evaluation/Worst_Reward", np.min(eval_rewards), 0)

        return eval_rewards, eval_lengths

    def save_config_to_logdir(self):
        """Save the training configuration YAML file to the TensorBoard log directory."""
        # Save the actual config used for training
        config_path = os.path.join(self.tb_log_dir, "training_config.yaml")

        # Add training metadata
        config_with_metadata = {
            "training_metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "log_directory": self.tb_log_dir,
                "save_directory": self.save_dir,
            },
            "training_config": self.config,
        }

        with open(config_path, "w") as f:
            yaml.dump(config_with_metadata, f, default_flow_style=False, indent=2)
        print(f"Training configuration saved to: {config_path}")

        # Also save a copy of the original config file if we can find it
        original_config_files = [
            "configs/actor_critic_config.yaml",
            "actor_critic_config.yaml",
        ]
        for original_path in original_config_files:
            if os.path.exists(original_path):
                import shutil

                original_copy_path = os.path.join(
                    self.tb_log_dir, "original_config.yaml"
                )
                shutil.copy2(original_path, original_copy_path)
                print(f"Original config file copied to: {original_copy_path}")
                break


if __name__ == "__main__":
    # Example configuration
    training_config = yaml.load(
        open("configs/actor_critic_config.yaml"), Loader=yaml.FullLoader
    )

    tools.set_seed_everywhere(training_config["seed"])
    if training_config["deterministic_run"]:
        tools.enable_deterministic_run()

    trainer = ActorCriticTrainer(config=training_config)
    trainer.train()
    trainer.evaluate(num_episodes=5)
    trainer.evaluate(num_episodes=1, visualize=True)

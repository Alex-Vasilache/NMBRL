import numpy as np
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
import torch
import sys
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml
import zipfile
import tempfile
import pickle

# Add project root to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import tools
from agents.actor_critic_agent import ActorCriticAgent


class DreamerACAgent:
    """
    Trainer for the DREAMER Actor-Critic agent.
    Manages the training loop, including agent-environment interaction and learning updates.
    """

    def __init__(
        self,
        config,
        env,
        tensorboard_log=str(os.path.join(os.path.dirname(__file__), "..", "runs")),
    ):
        """
        Initializes the trainer.

        :param config: A dictionary containing training parameters.
        """
        self.config = config["dreamer_agent_trainer"]
        self.global_config = config
        self.env = env

        # Initialize the Actor-Critic agent
        self.action_space = self.env.action_space
        self.state_dim = self.env.observation_space.shape[0]
        self.agent = ActorCriticAgent(
            config=self.config,
            state_dim=self.state_dim,
            action_dim=self.action_space.shape,
        )

        # Training parameters
        self.batch_size = self.config.get("batch_size")
        self.imag_horizon = self.config.get("imag_horizon")
        self.update_frequency = self.config.get("update_frequency")
        self.gamma = self.config.get("gamma")
        self.discount_lambda = self.config.get("discount_lambda")

        # Imagined trajectories for batch training
        self.imagined_trajectories = {
            "states": [],
            "actions": [],
            "rewards": [],
        }

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []

        # TensorBoard logging setup
        self.tb_log_dir = os.path.join(tensorboard_log, "dreamer_agent")
        os.makedirs(self.tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.global_step = 0
        self.training_step = 0

        # print(f"TensorBoard logs will be saved to: {self.tb_log_dir}")

        tools.seed_everything(self.global_config["global"]["seed"])
        if self.config["deterministic_run"]:
            tools.enable_deterministic_run()

        # Log hyperparameters to TensorBoard
        hparam_dict = {
            "learning_rate": float(self.config.get("learning_rate")),
            "batch_size": self.batch_size,
            "imag_horizon": self.imag_horizon,
            "update_frequency": self.update_frequency,
            "gamma": self.gamma,
            "discount_lambda": self.discount_lambda,
            "units": self.config.get("units"),
            "layers": self.config.get("actor")["layers"],
            "act": self.config.get("act"),
            "norm": self.config.get("norm"),
        }

    def store_trajectory(
        self,
        state,
        action,
        reward,
    ):
        """Store experience in the buffer."""
        self.imagined_trajectories["states"].append(
            state.cpu().detach().numpy()
        )  # s at [t]
        self.imagined_trajectories["actions"].append(
            action.cpu().detach().numpy()
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
                    states_list.append(state.cpu().detach().numpy())
                else:
                    states_list.append(np.array(state))
            states_data = np.array(states_list, dtype=np.float32)

        if isinstance(actions_data, list):
            actions_list = []
            for action in actions_data:
                if torch.is_tensor(action):
                    actions_list.append(action.cpu().detach().numpy())
                else:
                    actions_list.append(np.array(action))
            actions_data = np.array(actions_list, dtype=np.float32)

        if isinstance(rewards_data, list):
            rewards_list = []
            for reward in rewards_data:
                if torch.is_tensor(reward):
                    rewards_list.append(reward.cpu().detach().numpy())
                else:
                    rewards_list.append(np.array(reward))
            rewards_data = np.array(rewards_list, dtype=np.float32)

        states = torch.tensor(
            states_data, dtype=torch.float32, device=self.config["device"]
        )  # [imag_horizon, batch_size, state_dim]
        actions = torch.tensor(
            actions_data, dtype=torch.float32, device=self.config["device"]
        )  # [imag_horizon, batch_size, action_dim]
        rewards = torch.tensor(
            rewards_data, dtype=torch.float32, device=self.config["device"]
        ).unsqueeze(
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
        Compatible with SB3 format (.zip file).

        :param epoch: Optional epoch number to include in filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if epoch is not None:
            model_name = f"dreamer_ac_agent_ep{epoch}_{timestamp}.zip"
        else:
            model_name = f"dreamer_ac_agent_final_{timestamp}.zip"

        model_path = os.path.join(self.save_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Create a temporary directory to store files before zipping
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save actor model
            actor_path = os.path.join(temp_dir, "actor.pth")
            torch.save(
                {
                    "model_state_dict": self.agent.actor.state_dict(),
                    "model_config": {
                        "state_dim": self.state_dim,
                        "action_dim": self.action_space,
                    },
                },
                actor_path,
            )

            # Save critic model
            critic_path = os.path.join(temp_dir, "critic.pth")
            torch.save(
                {
                    "model_state_dict": self.agent.critic.state_dict(),
                    "model_config": {
                        "state_dim": self.state_dim,
                        "action_dim": self.action_space,
                    },
                },
                critic_path,
            )

            # Save training configuration and statistics
            training_info = {
                "config": self.config,
                "global_config": self.global_config,
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

            training_info_path = os.path.join(temp_dir, "training_info.pkl")
            with open(training_info_path, "wb") as f:
                pickle.dump(training_info, f)

            # Create the zip file
            with zipfile.ZipFile(model_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(actor_path, "actor.pth")
                zipf.write(critic_path, "critic.pth")
                zipf.write(training_info_path, "training_info.pkl")

        print(f"Models saved to: {model_path}")
        return model_path

    def predict(self, obs, deterministic=False):
        """
        Predict action for given observation.
        Compatible with SB3 interface.

        :param obs: Observation
        :param deterministic: Whether to use deterministic action
        :return: Action and state (None for this implementation)
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.config["device"])

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            action = self.agent.get_action(obs, deterministic=deterministic)
            return action.cpu().numpy(), None

    @classmethod
    def load(cls, path, env=None):
        """
        Load a DREAMER agent from a .zip file.
        Compatible with SB3 loading pattern.

        :param path: Path to the .zip file
        :param env: Environment (optional, for compatibility)
        :return: Loaded DreamerACAgent instance
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the zip file
            with zipfile.ZipFile(path, "r") as zipf:
                zipf.extractall(temp_dir)

            # Load training info
            training_info_path = os.path.join(temp_dir, "training_info.pkl")
            with open(training_info_path, "rb") as f:
                training_info = pickle.load(f)

            config = training_info["config"]
            global_config = training_info["global_config"]

            # Create a new agent instance (with minimal tensorboard setup for loading)
            temp_log_dir = os.path.join(
                tempfile.gettempdir(),
                f"dreamer_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            agent = cls(global_config, env, tensorboard_log=temp_log_dir)

            # Load actor model
            actor_path = os.path.join(temp_dir, "actor.pth")
            actor_data = torch.load(
                actor_path, map_location=config["device"], weights_only=False
            )
            agent.agent.actor.load_state_dict(actor_data["model_state_dict"])

            # Load critic model
            critic_path = os.path.join(temp_dir, "critic.pth")
            critic_data = torch.load(
                critic_path, map_location=config["device"], weights_only=False
            )
            agent.agent.critic.load_state_dict(critic_data["model_state_dict"])

            # Restore training statistics
            agent.episode_rewards = training_info.get("episode_rewards", [])
            agent.episode_lengths = training_info.get("episode_lengths", [])
            agent.training_losses = training_info.get("training_losses", [])

            # Set to evaluation mode
            agent.agent.actor.eval()
            agent.agent.critic.eval()

            # print(f"DREAMER agent loaded from: {path}")
            return agent

    def close_tensorboard(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()
            print(f"TensorBoard writer closed. Logs saved to: {self.tb_log_dir}")

    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained agent without learning updates.
        """
        if self.eval_env is None:
            print("No evaluation environment available. Skipping evaluation.")
            return [], []

        print(f"\n=== Evaluating Agent over {num_episodes} episodes ===")

        # Set agent to evaluation mode
        self.agent.actor.eval()
        self.agent.critic.eval()

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            state, info = self.eval_env.reset()
            terminated = False
            total_reward = 0
            step_count = 0

            while not terminated and step_count < self.config.get(
                "max_steps_per_episode"
            ):
                # Get action (deterministic for evaluation)
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.config["device"]
                )
                action = self.agent.get_action(state).cpu().detach().numpy()
                try:
                    step_result = self.eval_env.step(action)
                    # Handle different return formats from step
                    step_len = len(step_result)
                    if step_len == 4:
                        next_state, reward, terminated, info = step_result[:4]
                        truncated = False
                    elif step_len == 5:
                        next_state, reward, terminated, truncated, info = step_result[
                            :5
                        ]
                    else:
                        raise ValueError(
                            f"Unexpected step return format: {step_len} elements"
                        )
                except Exception as e:
                    print(f"Error during evaluation step: {e}")
                    break

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

        # Calculate evaluation statistics
        eval_avg_reward = np.mean(eval_rewards)
        eval_std_reward = np.std(eval_rewards)

        print(f"Evaluation Results:")
        print(f"  Average Reward: {eval_avg_reward:.2f} ± {eval_std_reward:.2f}")

        # Log evaluation results to TensorBoard
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_scalar(
                "Evaluation/Avg_Reward", eval_avg_reward, self.global_step
            )
            self.writer.add_scalar(
                "Evaluation/Std_Reward", eval_std_reward, self.global_step
            )
            self.writer.add_scalar(
                "Evaluation/Best_Reward", np.max(eval_rewards), self.global_step
            )
            self.writer.add_scalar(
                "Evaluation/Worst_Reward", np.min(eval_rewards), self.global_step
            )

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

    def learn(
        self,
        total_timesteps: int,
        callback: CallbackList = CallbackList([]),
        progress_bar: bool = False,
    ):
        """
        Runs the main training loop with proper learning updates.
        """
        num_epochs = self.config.get("num_epochs")
        save_frequency = self.config.get("save_frequency")  # Save every N episodes
        eval_frequency = self.config.get("eval_frequency")  # Evaluate every N episodes
        print_frequency = self.config.get("print_frequency")  # Print every N episodes
        self.save_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(self.tb_log_dir)))
            ),
            "checkpoints",
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.eval_env = None

        self.save_config_to_logdir()

        for individual_callback in callback.callbacks:
            if isinstance(individual_callback, EvalCallback):
                self.eval_env = individual_callback.eval_env

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.reset_trajectory()
            states = self.env.reset(
                remove_from_replay_buffer=False
            )  # directly contains samples from the replay buffer

            for i in range(self.imag_horizon):
                states = torch.tensor(
                    states, dtype=torch.float32, device=self.config["device"]
                )
                actions = self.agent.get_action(states)
                next_states, rewards, terminated, info = self.env.step(
                    actions.cpu().detach().numpy()
                )
                self.store_trajectory(
                    states,
                    actions,
                    rewards,
                )
                states = next_states

            losses = self.train_agent()
            if print_frequency and (epoch + 1) % print_frequency == 0:
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
            if eval_frequency and (epoch + 1) % eval_frequency == 0 and self.eval_env:
                self.evaluate(num_episodes=10)

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
        # print(f"TensorBoard logs saved to: {self.tb_log_dir}")

        # Save final models
        final_model_path = self.save_models()
        print(f"Final models saved to: {final_model_path}")

        # Run final evaluation
        print("\n=== Running Final Evaluation ===")
        final_eval_episodes = self.config.get("final_eval_episodes", 10)
        eval_rewards, eval_lengths = self.evaluate(num_episodes=final_eval_episodes)

        # Log final evaluation to TensorBoard
        final_avg_reward = np.mean(eval_rewards)
        final_std_reward = np.std(eval_rewards)
        self.writer.add_scalar(
            "Evaluation/Final_Avg_Reward", final_avg_reward, num_epochs
        )
        self.writer.add_scalar(
            "Evaluation/Final_Std_Reward", final_std_reward, num_epochs
        )

        self.env.close()

        return final_model_path

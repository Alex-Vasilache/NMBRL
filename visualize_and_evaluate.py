# Script for loading trained SNN Actor-Critic models and running visualization and evaluation.
# This script allows you to load saved models and evaluate their performance with visualization.

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse
from collections import deque
import time

# Add project root to path for proper imports
sys.path.insert(0, os.path.dirname(__file__))

from world_models.ini_cartpole_wrapper import INICartPoleWrapper
from agents.snn_actor_critic_agent import SnnActorCriticAgent


class ModelEvaluator:
    """
    Evaluator for trained SNN Actor-Critic models.
    Loads saved models and runs evaluation episodes with visualization and statistics.
    """

    def __init__(self, model_path, visualize=True):
        """
        Initialize the evaluator.

        :param model_path: Path to the directory containing saved models
        :param visualize: Whether to show visual environment during evaluation
        """
        self.model_path = model_path
        self.visualize = visualize

        # Load training info and configuration
        training_info_path = os.path.join(model_path, "training_info.pth")
        if not os.path.exists(training_info_path):
            raise FileNotFoundError(f"Training info not found at {training_info_path}")

        self.training_info = torch.load(training_info_path, weights_only=False)
        self.config = self.training_info["config"]

        print(f"Loading model from: {model_path}")
        print(
            f"Model was trained for {self.training_info['final_stats']['total_episodes']} episodes"
        )
        print(
            f"Best training reward: {self.training_info['final_stats']['best_episode_reward']:.2f}"
        )

        # Initialize environment
        self.world_model = INICartPoleWrapper(
            visualize=visualize,
            dt_simulation=self.config.get("dt_simulation", 0.02),
        )

        # Initialize agent
        action_space = self.world_model.env.action_space
        self.agent = SnnActorCriticAgent(
            action_space=action_space,
            state_dim=self.config.get("state_dim", 6),
            hidden_dim=self.config.get("hidden_dim", 128),
            num_steps=self.config.get("snn_time_steps", 1),
            lr=self.config.get("learning_rate", 1e-3),
            alpha=self.config.get("alpha", 0.9),
            beta=self.config.get("beta", 0.9),
            threshold=self.config.get("threshold", 1),
            learn_alpha=self.config.get("learn_alpha", True),
            learn_beta=self.config.get("learn_beta", True),
            learn_threshold=self.config.get("learn_threshold", True),
            weight_init_mean=self.config.get("weight_init_mean", 0.0),
            weight_init_std=self.config.get("weight_init_std", 0.01),
            max_std=self.config.get("max_std", 2.0),
            min_std=self.config.get("min_std", 0.1),
        )

        # Load trained models
        self.load_models()

    def load_models(self):
        """Load the actor and critic models from saved checkpoints."""
        actor_path = os.path.join(self.model_path, "actor.pth")
        critic_path = os.path.join(self.model_path, "critic.pth")

        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Actor model not found at {actor_path}")
        if not os.path.exists(critic_path):
            raise FileNotFoundError(f"Critic model not found at {critic_path}")

        # Load actor
        actor_checkpoint = torch.load(actor_path, weights_only=False)
        self.agent.actor.load_state_dict(actor_checkpoint["model_state_dict"])

        # Load critic
        critic_checkpoint = torch.load(critic_path, weights_only=False)
        self.agent.critic.load_state_dict(critic_checkpoint["model_state_dict"])

        # Set to evaluation mode
        self.agent.actor.eval()
        self.agent.critic.eval()

        print("Models loaded successfully!")

    def evaluate(self, num_episodes=10, max_steps_per_episode=1000, render_delay=0.02):
        """
        Evaluate the loaded model over multiple episodes.

        :param num_episodes: Number of evaluation episodes
        :param max_steps_per_episode: Maximum steps per episode
        :param render_delay: Delay between visualization frames (seconds)
        :return: Dictionary with evaluation results
        """
        print(f"\n=== Evaluating Model over {num_episodes} episodes ===")

        eval_rewards = []
        eval_lengths = []
        episode_trajectories = []

        for episode in range(num_episodes):
            print(f"Running evaluation episode {episode + 1}/{num_episodes}")

            state = self.world_model.reset()
            terminated = False
            total_reward = 0
            step_count = 0

            # Reset agent states
            self.agent.actor.reset()
            self.agent.critic.reset()

            # Store trajectory for analysis
            trajectory = {"states": [], "actions": [], "rewards": [], "values": []}

            while not terminated and step_count < max_steps_per_episode:
                # Get action from agent
                action = self.agent.get_action(state)
                value = self.agent.get_value(state)

                # Store trajectory data
                trajectory["states"].append(state.copy())
                trajectory["actions"].append(action.copy())
                trajectory["values"].append(value)

                # Environment step
                next_state, reward, terminated, info = self.world_model.step(action)

                trajectory["rewards"].append(reward)

                # Add visualization delay if enabled
                if self.visualize and render_delay > 0:
                    time.sleep(render_delay)

                state = next_state
                total_reward += reward
                step_count += 1

            eval_rewards.append(total_reward)
            eval_lengths.append(step_count)
            episode_trajectories.append(trajectory)

            print(
                f"  Episode {episode + 1}: Reward: {total_reward:.2f}, Steps: {step_count}"
            )

        # Calculate statistics
        results = {
            "rewards": eval_rewards,
            "lengths": eval_lengths,
            "trajectories": episode_trajectories,
            "stats": {
                "mean_reward": np.mean(eval_rewards),
                "std_reward": np.std(eval_rewards),
                "mean_length": np.mean(eval_lengths),
                "std_length": np.std(eval_lengths),
                "min_reward": np.min(eval_rewards),
                "max_reward": np.max(eval_rewards),
                "success_rate": np.mean(
                    [r > 0 for r in eval_rewards]
                ),  # Assuming positive reward indicates success
            },
        }

        # Print results
        print(f"\n=== Evaluation Results ===")
        print(
            f"Mean Reward: {results['stats']['mean_reward']:.2f} ± {results['stats']['std_reward']:.2f}"
        )
        print(
            f"Mean Length: {results['stats']['mean_length']:.1f} ± {results['stats']['std_length']:.1f}"
        )
        print(
            f"Min/Max Reward: {results['stats']['min_reward']:.2f} / {results['stats']['max_reward']:.2f}"
        )
        print(f"Success Rate: {results['stats']['success_rate']:.2%}")

        self.world_model.close()

        return results

    def plot_training_progress(self, save_path=None):
        """
        Plot training progress from the loaded model.

        :param save_path: Optional path to save the plot
        """
        if "episode_rewards" not in self.training_info:
            print("No training progress data available.")
            return

        episode_rewards = self.training_info["episode_rewards"]
        episode_lengths = self.training_info["episode_lengths"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot episode rewards
        episodes = range(1, len(episode_rewards) + 1)
        ax1.plot(episodes, episode_rewards, alpha=0.6, label="Episode Reward")

        # Add moving average
        if len(episode_rewards) > 10:
            moving_avg = np.convolve(episode_rewards, np.ones(10) / 10, mode="valid")
            ax1.plot(
                episodes[9:],
                moving_avg,
                "r-",
                linewidth=2,
                label="Moving Average (10 episodes)",
            )

        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Training Progress: Episode Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot episode lengths
        ax2.plot(
            episodes, episode_lengths, alpha=0.6, color="green", label="Episode Length"
        )

        # Add moving average for lengths
        if len(episode_lengths) > 10:
            moving_avg_length = np.convolve(
                episode_lengths, np.ones(10) / 10, mode="valid"
            )
            ax2.plot(
                episodes[9:],
                moving_avg_length,
                "orange",
                linewidth=2,
                label="Moving Average (10 episodes)",
            )

        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Training Progress: Episode Lengths")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training progress plot saved to: {save_path}")

        plt.show()

    def plot_evaluation_results(self, results, save_path=None):
        """
        Plot evaluation results.

        :param results: Results dictionary from evaluate() method
        :param save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        episodes = range(1, len(results["rewards"]) + 1)
        axes[0, 0].bar(episodes, results["rewards"])
        axes[0, 0].axhline(
            y=results["stats"]["mean_reward"],
            color="r",
            linestyle="--",
            label=f"Mean: {results['stats']['mean_reward']:.2f}",
        )
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Evaluation Episode Rewards")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].bar(episodes, results["lengths"], color="green")
        axes[0, 1].axhline(
            y=results["stats"]["mean_length"],
            color="r",
            linestyle="--",
            label=f"Mean: {results['stats']['mean_length']:.1f}",
        )
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].set_title("Evaluation Episode Lengths")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Reward distribution
        axes[1, 0].hist(
            results["rewards"],
            bins=min(10, len(results["rewards"])),
            alpha=0.7,
            edgecolor="black",
        )
        axes[1, 0].axvline(
            x=results["stats"]["mean_reward"],
            color="r",
            linestyle="--",
            label=f"Mean: {results['stats']['mean_reward']:.2f}",
        )
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Value function estimates (from first episode)
        if results["trajectories"]:
            first_episode_values = results["trajectories"][0]["values"]
            steps = range(len(first_episode_values))
            axes[1, 1].plot(steps, first_episode_values, "b-", linewidth=2)
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Value Estimate")
            axes[1, 1].set_title("Value Function Estimates (First Episode)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Evaluation results plot saved to: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SNN Actor-Critic model"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the saved model directory"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable environment visualization"
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.02,
        help="Delay between frames during visualization",
    )
    parser.add_argument(
        "--plot-training", action="store_true", help="Plot training progress"
    )
    parser.add_argument(
        "--save-plots", type=str, default=None, help="Directory to save plots"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path, visualize=not args.no_visualize
    )

    # Plot training progress if requested
    if args.plot_training:
        save_path = None
        if args.save_plots:
            os.makedirs(args.save_plots, exist_ok=True)
            save_path = os.path.join(args.save_plots, "training_progress.png")
        evaluator.plot_training_progress(save_path=save_path)

    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.episodes, render_delay=args.render_delay
    )

    # Plot evaluation results
    save_path = None
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)
        save_path = os.path.join(args.save_plots, "evaluation_results.png")
    evaluator.plot_evaluation_results(results, save_path=save_path)


if __name__ == "__main__":
    main()

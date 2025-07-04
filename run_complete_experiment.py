#!/usr/bin/env python3
"""
Complete experiment script for SNN Actor-Critic training and evaluation.
This script demonstrates the full workflow: training the agent and then evaluating it.
"""

import os
import sys
import argparse

# Add project root to path for proper imports
sys.path.insert(0, os.path.dirname(__file__))

from learning.actor_critic_trainer import ActorCriticTrainer
from visualize_and_evaluate import ModelEvaluator


def run_complete_experiment(config, eval_episodes=5, visualize_eval=True):
    """
    Run a complete experiment: train the agent and then evaluate it.

    :param config: Training configuration dictionary
    :param eval_episodes: Number of episodes for evaluation
    :param visualize_eval: Whether to visualize during evaluation
    :return: Tuple of (model_path, evaluation_results)
    """
    print("=" * 60)
    print("SNN ACTOR-CRITIC COMPLETE EXPERIMENT")
    print("=" * 60)

    # Phase 1: Training
    print("\n🚀 Phase 1: Training the SNN Actor-Critic Agent")
    print("-" * 50)

    trainer = ActorCriticTrainer(config=config)
    model_path = trainer.train()

    print(f"\n✅ Training completed! Models saved to: {model_path}")

    # Phase 2: Evaluation
    print(f"\n🔍 Phase 2: Evaluating the Trained Agent")
    print("-" * 50)

    # Temporarily turn off visualization for training config but enable for evaluation
    evaluator = ModelEvaluator(model_path=model_path, visualize=visualize_eval)

    # Plot training progress
    print("\n📊 Plotting training progress...")
    evaluator.plot_training_progress()

    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=eval_episodes,
        render_delay=0.05 if visualize_eval else 0,  # Slower for better visualization
    )

    # Plot evaluation results
    print("\n📈 Plotting evaluation results...")
    evaluator.plot_evaluation_results(results)

    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return model_path, results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete SNN Actor-Critic experiment"
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of training episodes"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--no-train-viz",
        action="store_true",
        help="Disable visualization during training",
    )
    parser.add_argument(
        "--no-eval-viz",
        action="store_true",
        help="Disable visualization during evaluation",
    )
    parser.add_argument(
        "--save-dir", type=str, default="saved_models", help="Directory to save models"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension for SNN layers"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for training"
    )

    args = parser.parse_args()

    # Configure the experiment
    training_config = {
        # Training episodes
        "num_episodes": args.episodes,
        "max_steps_per_episode": 1000,
        # Learning parameters
        "batch_size": args.batch_size,
        "buffer_seq_length": 15,  # Trajectory length for λ-returns
        "update_frequency": 1,
        "learning_rate": args.lr,
        "gamma": 0.997,  # Discount factor for the reward
        "discount_lambda": 0.95,  # Return mixing factor
        # SNN Architecture
        "hidden_dim": args.hidden_dim,
        "snn_time_steps": 1,
        "state_dim": 6,  # CartPole state dimension
        # SNN Parameters
        "alpha": 0.9,
        "beta": 0.9,
        "threshold": 1.0,
        "learn_alpha": False,  # Keep SNN parameters fixed for stability
        "learn_beta": False,
        "learn_threshold": False,
        # Network initialization
        "weight_init_mean": 0.0,
        "weight_init_std": 0.01,
        "max_std": 1.0,
        "min_std": 0.1,
        # Environment and visualization
        "visualize": False,
        "dt_simulation": 0.02,
        # Model saving
        "save_dir": args.save_dir,
        "save_frequency": max(1, args.episodes // 5),  # Save 5 times during training
    }

    print("Experiment Configuration:")
    print(f"  Training Episodes: {training_config['num_episodes']}")
    print(f"  Evaluation Episodes: {args.eval_episodes}")
    print(f"  Hidden Dimension: {training_config['hidden_dim']}")
    print(f"  Learning Rate: {training_config['learning_rate']}")
    print(f"  Batch Size: {training_config['batch_size']}")
    print(f"  Training Visualization: {training_config['visualize']}")
    print(f"  Evaluation Visualization: {not args.no_eval_viz}")
    print(f"  Save Directory: {training_config['save_dir']}")

    # Run the complete experiment
    model_path, results = run_complete_experiment(
        config=training_config,
        eval_episodes=args.eval_episodes,
        visualize_eval=not args.no_eval_viz,
    )

    print(f"\n📁 Final model location: {model_path}")
    print(
        f"📊 Evaluation mean reward: {results['stats']['mean_reward']:.2f} ± {results['stats']['std_reward']:.2f}"
    )


if __name__ == "__main__":
    # Example usage with different configurations
    if len(sys.argv) == 1:
        print("🔧 Running with default configuration...")
        print("   Use --help to see available options")
        print(
            "   Example: python run_complete_experiment.py --episodes 50 --eval-episodes 10 --hidden-dim 128"
        )

    main()

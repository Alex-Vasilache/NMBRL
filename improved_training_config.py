#!/usr/bin/env python3
"""
Improved training configurations for SNN Actor-Critic agent.
This script provides better hyperparameters and debugging options for more effective training.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from learning.actor_critic_trainer import ActorCriticTrainer


def get_stable_config():
    """
    Configuration optimized for stable learning with CartPole.
    Uses conservative hyperparameters for reliable convergence.
    """
    return {
        # Training episodes - start smaller for debugging
        "num_episodes": 100,
        "max_steps_per_episode": 1000,
        # Learning parameters - more conservative
        "batch_size": 128,  # Smaller batch for more frequent updates
        "buffer_seq_length": 10,  # Shorter sequences for faster learning
        "update_frequency": 5,  # More frequent updates
        "learning_rate": 5e-4,  # Slightly higher learning rate
        "gamma": 0.99,  # Standard discount factor
        "discount_lambda": 0.95,  # λ-return mixing
        # SNN Architecture - simpler for debugging
        "hidden_dim": 32,  # Smaller network, easier to train
        "snn_time_steps": 1,  # Single time step for simplicity
        "state_dim": 6,
        # SNN Parameters - keep fixed for stability
        "alpha": 0.9,
        "beta": 0.9,
        "threshold": 1.0,
        "learn_alpha": False,  # Keep SNN params fixed initially
        "learn_beta": False,
        "learn_threshold": False,
        # Network initialization - smaller std for stability
        "weight_init_mean": 0.0,
        "weight_init_std": 0.005,  # Smaller initialization
        "max_std": 0.5,  # Smaller action noise
        "min_std": 0.01,
        # Environment and visualization
        "visualize": False,  # Disable for faster training
        "dt_simulation": 0.02,
        # Model saving
        "save_dir": "improved_models",
        "save_frequency": 25,  # Save every 25 episodes
    }


def get_debug_config():
    """
    Configuration for debugging training issues.
    Very small network and frequent updates for quick feedback.
    """
    return {
        # Short training for quick debugging
        "num_episodes": 20,
        "max_steps_per_episode": 500,  # Shorter episodes
        # Aggressive learning parameters
        "batch_size": 32,  # Small batch size
        "buffer_seq_length": 5,  # Very short sequences
        "update_frequency": 1,  # Update every step
        "learning_rate": 1e-3,  # Higher learning rate
        "gamma": 0.95,  # Lower discount for faster learning
        "discount_lambda": 0.9,
        # Minimal SNN Architecture
        "hidden_dim": 16,  # Very small network
        "snn_time_steps": 1,
        "state_dim": 6,
        # SNN Parameters
        "alpha": 0.8,
        "beta": 0.8,
        "threshold": 0.8,  # Lower threshold
        "learn_alpha": False,
        "learn_beta": False,
        "learn_threshold": False,
        # Conservative initialization
        "weight_init_mean": 0.0,
        "weight_init_std": 0.001,  # Very small weights
        "max_std": 0.3,
        "min_std": 0.05,
        # Environment
        "visualize": True,  # Enable to see what's happening
        "dt_simulation": 0.02,
        # Model saving
        "save_dir": "debug_models",
        "save_frequency": 5,  # Save frequently for debugging
    }


def get_performance_config():
    """
    Configuration optimized for best performance.
    Larger network and longer training for maximum results.
    """
    return {
        # Extended training
        "num_episodes": 200,
        "max_steps_per_episode": 1000,
        # Optimized learning parameters
        "batch_size": 256,
        "buffer_seq_length": 15,
        "update_frequency": 10,
        "learning_rate": 2e-4,  # Conservative learning rate
        "gamma": 0.997,
        "discount_lambda": 0.95,
        # Larger SNN Architecture
        "hidden_dim": 128,  # Larger network capacity
        "snn_time_steps": 1,
        "state_dim": 6,
        # SNN Parameters
        "alpha": 0.95,  # Slower decay
        "beta": 0.95,
        "threshold": 1.0,
        "learn_alpha": False,  # Keep fixed for stability
        "learn_beta": False,
        "learn_threshold": False,
        # Careful initialization
        "weight_init_mean": 0.0,
        "weight_init_std": 0.01,
        "max_std": 1.0,
        "min_std": 0.01,
        # Environment
        "visualize": False,  # Disable for speed
        "dt_simulation": 0.02,
        # Model saving
        "save_dir": "performance_models",
        "save_frequency": 50,
    }


def run_improved_training(config_name="stable"):
    """
    Run training with improved configuration.

    :param config_name: One of "stable", "debug", or "performance"
    """
    configs = {
        "stable": get_stable_config(),
        "debug": get_debug_config(),
        "performance": get_performance_config(),
    }

    if config_name not in configs:
        print(f"❌ Unknown config: {config_name}")
        print(f"Available configs: {list(configs.keys())}")
        return

    config = configs[config_name]

    print(f"🚀 Starting IMPROVED training with '{config_name}' configuration")
    print("=" * 60)
    print(f"Episodes: {config['num_episodes']}")
    print(f"Hidden Dim: {config['hidden_dim']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Update Frequency: {config['update_frequency']}")
    print("=" * 60)

    # Create trainer and run
    trainer = ActorCriticTrainer(config=config)
    model_path = trainer.train()

    print(f"\n✅ Training completed!")
    print(f"📁 Model saved to: {model_path}")

    # Quick evaluation
    print(f"\n🔍 Running quick evaluation...")
    trainer.evaluate(num_episodes=3)

    return model_path


def diagnose_training_issues():
    """
    Print diagnostic information for troubleshooting training problems.
    """
    print("🔧 SNN ACTOR-CRITIC TRAINING DIAGNOSTICS")
    print("=" * 50)

    print("\n📊 Common Training Issues & Solutions:")
    print("1. NEGATIVE REWARDS:")
    print("   - Check environment reward structure")
    print("   - Verify action space mapping")
    print("   - Try smaller learning rate (1e-4 to 5e-5)")
    print("   - Use smaller network (hidden_dim=16-32)")

    print("\n2. HITTING EPISODE LIMITS:")
    print("   - Agent not learning termination conditions")
    print("   - Try more frequent updates (update_frequency=1-5)")
    print("   - Shorter episode lengths for debugging")
    print("   - Check if environment termination is working")

    print("\n3. UNSTABLE TRAINING:")
    print("   - Reduce learning rate")
    print("   - Increase batch size")
    print("   - Use smaller weight initialization")
    print("   - Keep SNN parameters fixed (learn_*=False)")

    print("\n4. SLOW CONVERGENCE:")
    print("   - Increase learning rate carefully")
    print("   - More frequent updates")
    print("   - Shorter sequence lengths")
    print("   - Check gradient flow")

    print("\n🎯 Recommended Quick Tests:")
    print("1. Run debug config: python improved_training_config.py debug")
    print("2. Check with minimal episodes and visualization enabled")
    print("3. Monitor actor/critic losses during training")
    print("4. Verify environment rewards are reasonable")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Improved SNN Actor-Critic Training")
    parser.add_argument(
        "config",
        nargs="?",
        default="stable",
        choices=["stable", "debug", "performance"],
        help="Training configuration to use",
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Show diagnostic information"
    )

    args = parser.parse_args()

    if args.diagnose:
        diagnose_training_issues()
    else:
        run_improved_training(args.config)

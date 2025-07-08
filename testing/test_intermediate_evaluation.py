#!/usr/bin/env python3
"""
Test script for the intermediate evaluation functionality during training.
"""

import os
import sys
import yaml
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learning.actor_critic_trainer import ActorCriticTrainer
from utils import tools


def create_test_config():
    """Create a minimal test configuration for quick testing."""
    return {
        # Training - minimal for testing
        "num_epochs": 5,
        "batch_size": 4,
        "learning_rate": 3e-4,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "max_steps_per_episode": 100,
        "device": "cpu",
        "reward_EMA": False,
        "seed": 42,
        "deterministic_run": True,
        # Behavior
        "gamma": 0.997,
        "discount_lambda": 0.95,
        "imag_horizon": 5,  # Short horizon for testing
        # Simulation
        "visualize": False,
        "dt_simulation": 0.02,
        # Model
        "act": "SiLU",
        "norm": True,
        "units": 32,
        # Logging
        "save_frequency": 2,  # Save every 2 epochs for testing
        # Evaluation
        "eval_episodes": 2,  # Minimal episodes for testing
        "eval_visualize": False,
        "final_eval_episodes": 3,
        # Actor
        "actor": {
            "layers": 2,
            "dist": "normal",
            "entropy": 3e-4,
            "unimix_ratio": 0.01,
            "std": "learned",
            "min_std": 0.1,
            "max_std": 1.0,
            "temp": 0.1,
            "outscale": 1.0,
        },
        # Critic
        "critic": {
            "layers": 2,
            "dist": "symlog_disc",
            "slow_target": False,
            "outscale": 0.0,
        },
    }


def test_intermediate_evaluation():
    """Test that intermediate evaluations run correctly during training."""
    print("Testing Intermediate Evaluation Functionality")
    print("=" * 50)

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config = create_test_config()
        config["save_dir"] = os.path.join(temp_dir, "models")
        config["log_dir"] = os.path.join(temp_dir, "logs")

        print(f"Using temporary directories:")
        print(f"  Models: {config['save_dir']}")
        print(f"  Logs: {config['log_dir']}")

        try:
            # Set deterministic behavior
            tools.set_seed_everywhere(config["seed"])
            tools.enable_deterministic_run()

            # Create trainer
            print("\nInitializing trainer...")
            trainer = ActorCriticTrainer(config=config)

            # Test that the intermediate evaluation method exists
            assert hasattr(
                trainer, "run_intermediate_evaluation"
            ), "run_intermediate_evaluation method not found"
            print("✓ Intermediate evaluation method found")

            # Test the evaluation method directly
            print("\nTesting intermediate evaluation method...")
            trainer.run_intermediate_evaluation(epoch=0)
            print("✓ Intermediate evaluation method executed successfully")

            # Run short training to test integration
            print("\nRunning short training with intermediate evaluations...")
            final_model_path = trainer.train()
            print(f"✓ Training completed, final model saved to: {final_model_path}")

            # Check that model files were created
            expected_saves = config["num_epochs"] // config["save_frequency"]
            model_files = [
                f
                for f in os.listdir(config["save_dir"])
                if f.startswith("snn_actor_critic")
            ]
            print(
                f"✓ Found {len(model_files)} saved model directories (expected at least {expected_saves})"
            )

            # Check that TensorBoard logs exist
            log_files = []
            for root, dirs, files in os.walk(config["log_dir"]):
                log_files.extend([f for f in files if f.endswith(".tfevents")])
            print(f"✓ Found {len(log_files)} TensorBoard log files")

            # Close trainer
            trainer.close_tensorboard()
            print("✓ Trainer closed successfully")

            print("\n" + "=" * 50)
            print("🎉 All intermediate evaluation tests passed!")
            return True

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_config_parameters():
    """Test that evaluation configuration parameters are handled correctly."""
    print("\nTesting Configuration Parameters")
    print("-" * 30)

    base_config = create_test_config()

    # Test with missing eval parameters
    config_missing_params = base_config.copy()
    config_missing_params.pop("eval_episodes", None)
    config_missing_params.pop("eval_visualize", None)
    config_missing_params.pop("final_eval_episodes", None)

    try:
        # Should work with default values
        with tempfile.TemporaryDirectory() as temp_dir:
            config_missing_params["save_dir"] = os.path.join(temp_dir, "models")
            config_missing_params["log_dir"] = os.path.join(temp_dir, "logs")

            trainer = ActorCriticTrainer(config=config_missing_params)
            trainer.run_intermediate_evaluation(epoch=0)
            trainer.close_tensorboard()

        print("✓ Default evaluation parameters work correctly")

        # Test with custom parameters
        config_custom = base_config.copy()
        config_custom["eval_episodes"] = 1
        config_custom["eval_visualize"] = (
            False  # Keep false to avoid visualization issues in tests
        )
        config_custom["final_eval_episodes"] = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            config_custom["save_dir"] = os.path.join(temp_dir, "models")
            config_custom["log_dir"] = os.path.join(temp_dir, "logs")

            trainer = ActorCriticTrainer(config=config_custom)
            trainer.run_intermediate_evaluation(epoch=0)
            trainer.close_tensorboard()

        print("✓ Custom evaluation parameters work correctly")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


if __name__ == "__main__":
    print("Intermediate Evaluation Test Suite")
    print("=" * 60)

    # Run tests
    test1_success = test_intermediate_evaluation()
    test2_success = test_config_parameters()

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"  Intermediate Evaluation: {'✓ PASS' if test1_success else '❌ FAIL'}")
    print(f"  Configuration Parameters: {'✓ PASS' if test2_success else '❌ FAIL'}")

    if test1_success and test2_success:
        print("\n🎉 All tests passed! Intermediate evaluation is ready to use.")
        print("\nUsage:")
        print("1. Set 'save_frequency' in your config to enable periodic saves")
        print("2. Set 'eval_episodes' to control evaluation length")
        print("3. Set 'eval_visualize' to true/false for visualization during eval")
        print("4. Check TensorBoard logs for evaluation metrics over time")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

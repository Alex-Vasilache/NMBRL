#!/usr/bin/env python3
"""
Test script to verify EvoAgent save/load functionality.
This script tests the save and load methods without running full training.
"""

import os
import sys
import tempfile
import numpy as np
import torch

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_evo_agent_save_load():
    """Test that EvoAgent save/load functionality works correctly."""
    print("Testing EvoAgent save/load functionality...")

    # Create a mock configuration
    config = {
        "evo_agent_trainer": {
            "general": {"random_seed": 42, "device": "cpu"},
            "environment": {"game_name": "CartPole-v1", "max_env_steps": 500},
            "network": {
                "net_size": [3, 3, 1],
                "spike_steps": 4,
                "max_vthr": 1000,
                "spatial": True,
                "prune_unconnected": True,
            },
            "evolution": {
                "num_iterations": 100,
                "num_gene_samples": 10,
                "evolution_method": "classic",
            },
            "map_elites": {"sigma_bins": 10, "sparsity_bins": 10},
            "training": {
                "batch_size_gene": 5,
                "num_data_samples": 2,
                "batch_size_data": 2,
                "curiculum_learning": False,
            },
        }
    }

    # Create a mock environment
    class MockEnv:
        def __init__(self):
            self.action_space = type("ActionSpace", (), {"shape": (1,), "n": 2})()
            self.observation_space = type("ObsSpace", (), {"shape": (4,)})()

    env = MockEnv()

    try:
        # Test agent creation
        print("Creating EvoAgent...")
        from evo_agent import EvoAgent

        agent = EvoAgent(config, env)
        print("✅ EvoAgent created successfully")

        # Test save functionality
        print("Testing save functionality...")
        save_path = agent.save_models(epoch=1)
        print(f"✅ Agent saved to: {save_path}")

        # Test load functionality
        print("Testing load functionality...")
        loaded_agent = EvoAgent.load(save_path, env)
        print("✅ Agent loaded successfully")

        # Test prediction with loaded agent
        print("Testing prediction with loaded agent...")
        obs = np.random.randn(4).astype(np.float32)
        action, _ = loaded_agent.predict(obs)
        print(f"✅ Prediction successful, action shape: {action.shape}")

        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)
            print("✅ Cleaned up test files")

        print(
            "🎉 All tests passed! EvoAgent save/load functionality is working correctly."
        )
        return True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("=" * 50)
    print("Testing EvoAgent Save/Load Functionality")
    print("=" * 50)

    success = test_evo_agent_save_load()

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())

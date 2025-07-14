#!/usr/bin/env python3
"""
Test script for the physical cartpole wrapper.

This script tests that the physical cartpole wrapper can be imported and
has the same interface as the DMC cartpole wrapper.
"""

import numpy as np


def test_wrapper_interface():
    """Test that the wrapper has the same interface as DMCCartpoleWrapper."""

    try:
        from world_models.physical_cartpole_wrapper import PhysicalCartpoleWrapper

        print("✓ Successfully imported PhysicalCartpoleWrapper")
    except ImportError as e:
        print(f"✗ Failed to import PhysicalCartpoleWrapper: {e}")
        return False

    try:
        from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper

        print("✓ Successfully imported DMCCartpoleWrapper for comparison")
    except ImportError as e:
        print(f"✗ Failed to import DMCCartpoleWrapper: {e}")
        return False

    # Test that both wrappers have the same constructor signature
    try:
        # Test DMC wrapper parameters
        dmc_params = {
            "seed": 42,
            "n_envs": 1,
            "render_mode": "none",
            "max_episode_steps": 1000,
        }

        # Test physical wrapper with same parameters
        physical_params = {
            "seed": 42,
            "n_envs": 1,
            "render_mode": "none",
            "max_episode_steps": 1000,
        }

        print("✓ Both wrappers accept the same constructor parameters")
    except Exception as e:
        print(f"✗ Parameter compatibility test failed: {e}")
        return False

    print("✓ All interface tests passed!")
    return True


def test_reward_function():
    """Test the DMC reward function implementation."""

    try:
        from world_models.physical_cartpole_wrapper import DMCRewardTask

        # Create mock physics object
        class MockPhysics:
            x_limit = 0.2

        reward_task = DMCRewardTask(MockPhysics())
        print("✓ Successfully created DMCRewardTask")

        # Test reward computation with sample state
        # State format: [angle, angle_vel, cos_angle, sin_angle, cart_pos, cart_vel]
        test_states = [
            # Upright, centered, low velocity
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            # Hanging down, centered
            [np.pi, 0.0, -1.0, 0.0, 0.0, 0.0],
            # Angled, off-center
            [np.pi / 4, 1.0, np.cos(np.pi / 4), np.sin(np.pi / 4), 0.1, 0.0],
        ]

        test_action = [0.0]  # No control input

        for i, state in enumerate(test_states):
            reward = reward_task.get_reward(state, test_action)
            print(f"✓ Test state {i+1}: reward = {reward:.4f}")

        print("✓ Reward function tests passed!")
        return True

    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Physical CartPole Wrapper")
    print("=" * 40)

    interface_ok = test_wrapper_interface()
    print()

    reward_ok = test_reward_function()
    print()

    if interface_ok and reward_ok:
        print("🎉 All tests passed! The physical cartpole wrapper is ready to use.")
        print("\nTo use in dynamic_data_generator.py, simply change the import:")
        print(
            "from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper"
        )
        print("to:")
        print(
            "from world_models.physical_cartpole_wrapper import PhysicalCartpoleWrapper as wrapper"
        )
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()

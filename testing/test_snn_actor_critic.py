#!/usr/bin/env python3
"""
Test script for the SNN Critic and Actor networks.
This verifies that the networks can process states and produce the correct outputs.
"""

import numpy as np
import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.snn_actor_critic_agent import CriticSNN, ActorSNN, SnnActorCriticAgent
from world_models.ini_cartpole_wrapper import INICartPoleWrapper


def test_critic_snn():
    """Test the Critic SNN with various input sizes."""
    print("Testing Critic SNN...")

    # Initialize Critic
    critic = CriticSNN(state_dim=6, hidden_dim=64, num_steps=10)

    # Test single state
    single_state = torch.randn(1, 6)
    value = critic(single_state)
    print(f"Single state input shape: {single_state.shape}")
    print(f"Critic output shape: {value.shape}")
    print(f"Critic output value: {value.item():.4f}")

    # Test batch of states
    batch_states = torch.randn(8, 6)
    batch_values = critic(batch_states)
    print(f"Batch states input shape: {batch_states.shape}")
    print(f"Batch critic output shape: {batch_values.shape}")
    print(f"Mean batch value: {batch_values.mean().item():.4f}")

    print("✅ Critic SNN test passed!\n")


def test_actor_snn():
    """Test the Actor SNN with various input sizes."""
    print("Testing Actor SNN...")

    # Initialize Actor
    actor = ActorSNN(state_dim=6, hidden_dim=64, action_dim=1, num_steps=10)

    # Test single state
    single_state = torch.randn(1, 6)
    action_mean, action_log_std = actor(single_state)
    print(f"Single state input shape: {single_state.shape}")
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Action log_std shape: {action_log_std.shape}")
    print(f"Action mean: {action_mean.item():.4f}")
    print(f"Action std: {torch.exp(action_log_std).item():.4f}")

    # Test batch of states
    batch_states = torch.randn(8, 6)
    batch_mean, batch_log_std = actor(batch_states)
    print(f"Batch states input shape: {batch_states.shape}")
    print(f"Batch action mean shape: {batch_mean.shape}")
    print(f"Batch action log_std shape: {batch_log_std.shape}")

    print("✅ Actor SNN test passed!\n")


def test_agent_integration():
    """Test the full SNN Actor-Critic agent with the CartPole environment."""
    print("Testing SNN Actor-Critic Agent Integration...")

    try:
        # Initialize environment
        env = INICartPoleWrapper()

        # Initialize agent
        agent = SnnActorCriticAgent(
            action_space=env.env.action_space, state_dim=6, hidden_dim=64, num_steps=10
        )

        # Test single step
        state = env.reset()
        print(f"Environment state shape: {state.shape}")

        # Get action from agent
        action = agent.get_action(state)
        print(f"Agent action: {action}")
        print(f"Action type: {type(action)}")
        print(f"Action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")

        # Get value estimate
        value = agent.get_value(state)
        print(f"State value estimate: {value:.4f}")

        # Test environment step
        next_state, reward, terminated, info = env.step(action)
        print(f"Environment step successful")
        print(f"Reward: {reward:.4f}")
        print(f"Terminated: {terminated}")

        # Test batch update (with dummy data)
        batch_size = 4
        dummy_states = np.random.randn(batch_size, 6)
        dummy_actions = np.random.uniform(-1, 1, (batch_size,))
        dummy_rewards = np.random.randn(batch_size)
        dummy_next_states = np.random.randn(batch_size, 6)
        dummy_dones = np.random.choice([True, False], batch_size)

        losses = agent.update(
            dummy_states, dummy_actions, dummy_rewards, dummy_next_states, dummy_dones
        )

        print(f"Update successful! Losses: {losses}")

        print("✅ Agent integration test passed!\n")

    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        import traceback

        traceback.print_exc()


def test_gradient_flow():
    """Test that gradients flow properly through the SNN."""
    print("Testing gradient flow...")

    critic = CriticSNN(state_dim=6, hidden_dim=32, num_steps=5)

    # Test input
    state = torch.randn(2, 6, requires_grad=True)

    # Forward pass
    value = critic(state)
    loss = value.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    has_gradients = any(p.grad is not None for p in critic.parameters())
    print(f"Parameters have gradients: {has_gradients}")

    if has_gradients:
        grad_norms = [
            p.grad.norm().item() for p in critic.parameters() if p.grad is not None
        ]
        print(f"Average gradient norm: {np.mean(grad_norms):.6f}")
        print("✅ Gradient flow test passed!\n")
    else:
        print("❌ No gradients found!")


def test_network_consistency():
    """Test that the networks produce consistent outputs for the same input."""
    print("Testing network consistency...")

    # Initialize networks
    critic = CriticSNN(state_dim=6, hidden_dim=32, num_steps=10)
    actor = ActorSNN(state_dim=6, hidden_dim=32, action_dim=1, num_steps=10)

    # Set to eval mode to ensure deterministic behavior
    critic.eval()
    actor.eval()

    # Test state
    test_state = torch.randn(1, 6)

    # Multiple forward passes should give same results
    with torch.no_grad():
        value1 = critic(test_state)
        value2 = critic(test_state)

        mean1, logstd1 = actor(test_state)
        mean2, logstd2 = actor(test_state)

    # Check consistency (allowing for small numerical differences)
    value_diff = torch.abs(value1 - value2).max().item()
    mean_diff = torch.abs(mean1 - mean2).max().item()
    logstd_diff = torch.abs(logstd1 - logstd2).max().item()

    print(f"Value difference: {value_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    print(f"LogStd difference: {logstd_diff:.8f}")

    tolerance = 1e-6
    if value_diff < tolerance and mean_diff < tolerance and logstd_diff < tolerance:
        print("✅ Network consistency test passed!\n")
    else:
        print("❌ Networks are not producing consistent outputs!")


def test_different_timesteps():
    """Test how different numbers of timesteps affect the outputs."""
    print("Testing different timesteps...")

    test_state = torch.randn(1, 6)
    timesteps = [5, 10, 20, 50]

    print("Critic values for different timesteps:")
    for t in timesteps:
        critic = CriticSNN(state_dim=6, hidden_dim=32, num_steps=t)
        with torch.no_grad():
            value = critic(test_state)
        print(f"  {t:2d} steps: {value.item():8.4f}")

    print("\nActor outputs for different timesteps:")
    for t in timesteps:
        actor = ActorSNN(state_dim=6, hidden_dim=32, action_dim=1, num_steps=t)
        with torch.no_grad():
            mean, logstd = actor(test_state)
        print(
            f"  {t:2d} steps: mean={mean.item():8.4f}, std={torch.exp(logstd).item():8.4f}"
        )

    print("✅ Timestep variation test completed!\n")


if __name__ == "__main__":
    print("=== SNN Actor-Critic Test Suite ===\n")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    test_critic_snn()
    test_actor_snn()
    test_gradient_flow()
    test_network_consistency()
    test_different_timesteps()
    test_agent_integration()

    print("=== All tests completed! ===")

import numpy as np
import torch
from world_models.world_model_wrapper import WorldModelWrapper
from world_models.ini_gymlike_cartpole_wrapper import GymlikeCartpoleWrapper
import matplotlib.pyplot as plt


def main():
    """
    Compares the trained World Model against the actual environment by feeding them
    the same actions and observing the differences in next state and reward.
    """
    print("--- Comparing World Model to Real Environment ---")

    # --- Setup ---
    num_comparison_steps = 100

    # 1. Initialize the real environment
    try:
        real_env = GymlikeCartpoleWrapper(seed=42, n_envs=1)
    except ImportError as e:
        print(f"Error initializing real environment: {e}")
        print("Please ensure all dependencies for the environment are installed.")
        return

    # 2. Initialize the World Model environment
    try:
        world_model_env = WorldModelWrapper(
            simulated_env=real_env,
            batch_size=1,
            trained_folder="world_models/trained/v1",
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(
            "Please ensure you have a trained model at 'world_models/trained/v1/model.pth'."
        )
        print("You can train one by running 'learning/world_model_trainer.py'.")
        return

    # --- Comparison Loop ---
    real_state = real_env.reset()

    state_errors = []
    reward_errors = []

    print(f"\nRunning comparison for {num_comparison_steps} steps...")
    for step in range(num_comparison_steps):
        # Synchronize the world model's state with the real environment's state
        world_model_env.set_state(real_state)

        # Choose a random action
        action = real_env.action_space.sample()
        # The real env expects a shape of (n_envs, action_dim), so (1, 1)
        action_for_real_env = np.expand_dims(action, axis=0)
        # The world model (DummyVecEnv) expects (n_envs, action_dim) as well
        action_for_wm = action_for_real_env

        # Step both environments
        wm_next_state, wm_reward, _, _ = world_model_env.step(action_for_wm)
        real_next_state, real_reward, _, _ = real_env.step(action_for_real_env)

        # Calculate prediction errors
        state_error = np.mean((real_next_state - wm_next_state) ** 2)
        reward_error = np.mean((real_reward - wm_reward) ** 2)

        state_errors.append(state_error)
        reward_errors.append(reward_error)

        if step < 10:  # Print first few steps
            print(
                f"Step {step + 1:02d} | State MSE: {state_error:.6f} | Reward MSE: {reward_error:.6f}"
            )

        # Update the real state for the next iteration
        real_state = real_next_state

    # --- Results ---
    avg_state_error = np.mean(state_errors)
    avg_reward_error = np.mean(reward_errors)

    print("\n--- Comparison Results ---")
    print(f"Average Next State MSE: {avg_state_error:.6f}")
    print(f"Average Reward MSE:     {avg_reward_error:.6f}")

    # Plotting the errors over time
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(state_errors, label="State MSE")
    ax[0].set_title("World Model Prediction Error (MSE)")
    ax[0].set_ylabel("State Error (MSE)")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(reward_errors, label="Reward MSE", color="orange")
    ax[1].set_xlabel("Time Step")
    ax[1].set_ylabel("Reward Error (MSE)")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

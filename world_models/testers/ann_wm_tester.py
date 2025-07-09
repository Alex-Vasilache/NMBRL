import numpy as np
from world_models.world_model_wrapper import WorldModelWrapper
from world_models.ini_gymlike_cartpole_wrapper import GymlikeCartpoleWrapper


def main():
    """
    Tests the WorldModelWrapper to ensure it loads a model and can be stepped through.
    """
    print("--- Testing WorldModelWrapper ---")

    try:
        # Batch size is 1 for this test
        simulated_env = GymlikeCartpoleWrapper(seed=42, n_envs=1)
        world_model_env = WorldModelWrapper(
            simulated_env=simulated_env,
            batch_size=1,
            trained_folder="world_models/trained/v1",
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(
            "Please ensure you have a trained model at 'world_models/trained/v1/model.pth'."
        )
        print("You can train a model by running 'learning/world_model_trainer.py'.")
        return

    # --- Test 1: Reset and check observation ---
    print("\n--- Test 1: Resetting environment ---")
    obs = world_model_env.reset()
    print(f"Initial observation on reset: {obs}")
    assert obs is not None, "Reset should return a valid observation."
    assert (
        obs.shape[1] == world_model_env.observation_space.shape[0]
    ), "Observation shape mismatch."

    # --- Test 2: Set a specific state ---
    print("\n--- Test 2: Setting a custom initial state ---")
    initial_state_shape = (1, world_model_env.observation_space.shape[0])
    initial_state = np.random.randn(*initial_state_shape).astype(np.float32)
    world_model_env.set_state(initial_state)
    print(f"Set initial state to: {initial_state}")

    # --- Test 3: Step through the environment ---
    print("\n--- Test 3: Stepping with random actions for 5 steps ---")
    for i in range(5):
        action = world_model_env.action_space.sample()
        action = np.expand_dims(action, axis=0)  # Add batch dimension
        obs, reward, terminated, info = world_model_env.step(action)
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Next Observation: {obs}")
        print(f"  Reward: {reward}")

        assert obs is not None
        assert isinstance(reward, np.ndarray)

    print("\n--- WorldModelWrapper test completed ---")


if __name__ == "__main__":
    main()

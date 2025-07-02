import sys
import os
import numpy as np

# Add the root directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_models.ini_cartpole_wrapper import INICartPoleWrapper


def test_ini_cartpole_wrapper():
    """
    Tests the INICartPoleWrapper by running a short simulation.
    """
    print("Testing INICartPoleWrapper...")
    env = INICartPoleWrapper()
    state = env.reset()
    print(f"Initial state: {state}")

    terminated = False
    for i in range(100):
        if terminated:
            print(f"Episode terminated at step {i}. Resetting environment.")
            state = env.reset()
            print(f"New initial state: {state}")

        # Take a random action
        action = np.random.uniform(-1.0, 1.0, size=(1,))

        state, reward, terminated, info = env.step(action)

        print(f"Step {i+1}:")
        print(f"  Action: {action[0]:.2f}")
        print(f"  State: {state}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}")
        print(f"  Info: {info}")

        if terminated:
            print("Terminated. Resetting.")
            env.reset()

    print("\nTest finished.")


if __name__ == "__main__":
    test_ini_cartpole_wrapper()

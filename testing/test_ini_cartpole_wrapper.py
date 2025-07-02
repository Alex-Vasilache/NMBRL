import sys
import os
import numpy as np

# Add the root directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_models.ini_cartpole_wrapper import INICartPoleWrapper


def test_ini_cartpole_wrapper():
    """
    Tests the INICartPoleWrapper by running a short simulation with the quadratic boundary cost function.
    """
    print("Testing INICartPoleWrapper with quadratic boundary cost...")

    # Initialize with custom parameters
    env = INICartPoleWrapper(max_steps=50, target_position=0.0)
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"Target position: {env.target_position}")
    print(f"Max steps: {env.max_steps}")
    print()

    terminated = False
    total_reward = 0.0

    for i in range(60):  # Try more steps than max_steps to test termination
        if terminated:
            print(f"Episode terminated at step {i}. Resetting environment.")
            state = env.reset()
            print(f"New initial state: {state}")
            total_reward = 0.0
            terminated = False
            print()

        # Take a random action
        action = np.random.uniform(-1.0, 1.0, size=(1,))

        state, reward, terminated, info = env.step(action)
        total_reward += reward

        print(f"Step {i+1}:")
        print(f"  Action: {action[0]:.3f}")
        print(
            f"  Position: {state[env.POSITION_IDX]:.3f} (target: {env.target_position})"
        )
        print(
            f"  Angle: {state[env.ANGLE_IDX]:.3f} rad ({np.degrees(state[env.ANGLE_IDX]):.1f}°)"
        )
        print(f"  Angle cos: {state[env.ANGLE_COS_IDX]:.3f}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Terminated: {terminated}")
        print(f"  Step count: {info['step_count']}/{info['max_steps']}")
        print(f"  Cost breakdown:")
        print(f"    Distance cost: {info['distance_cost']:.3f}")
        print(f"    Angle cost: {info['angle_cost']:.3f}")
        print(f"    Control cost: {info['control_cost']:.3f}")
        print(f"    Jerk cost: {info['jerk_cost']:.3f}")
        print(f"    Total cost: {info['total_cost']:.3f}")
        print()

        if terminated:
            print(f"Episode completed after {info['step_count']} steps.")
            print(f"Final total reward: {total_reward:.3f}")
            break

    print("\nTest finished.")


if __name__ == "__main__":
    test_ini_cartpole_wrapper()

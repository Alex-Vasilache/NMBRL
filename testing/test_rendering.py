import numpy as np
import sys
import os
import time

# Add project root to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world_models.ini_gymlike_cartpole_wrapper import INIGymlikeCartPoleWrapper


def test_long_rendering_gymlike():
    """
    Tests the rendering of the INIGymlikeCartPoleWrapper over a very long period with random actions.
    """
    print(
        "Testing long-term rendering with random actions (Gymlike) for 10,000 steps..."
    )
    try:
        num_steps = 10000
        # The environment will now run for the full num_steps before terminating
        env = INIGymlikeCartPoleWrapper(
            max_steps=num_steps,
            visualize=True,
            task="swingup",
            cartpole_type="custom_sim",
        )
        env.reset(batch_size=1)

        print(f"\n--- Running for {num_steps} steps with random actions ---")

        for i in range(num_steps):
            action = np.random.uniform(
                env.action_space.low,
                env.action_space.high,
                size=(1, *env.action_space.shape),
            )
            next_state, reward, terminated, info = env.step(action)

            # Print status less frequently to avoid flooding the console
            if (i + 1) % 100 == 0:
                print(
                    f"Step {i+1}/{num_steps}: Reward = {reward[0]:.3f}, Terminated = {terminated[0]}"
                )

            time.sleep(0.001)

            if terminated[0]:
                print(
                    f"Episode terminated at step {i+1}, which is expected with random actions. Resetting."
                )
                env.reset()

        print("\nClosing environment...")
        env.close()
        print("Long-term rendering test complete!")

    except ImportError as e:
        print(f"Skipping test: {e}")
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_long_rendering_gymlike()

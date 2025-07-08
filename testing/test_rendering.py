import numpy as np
import sys
import os
import time

# Add project root to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from world_models.dmc_cartpole_wrapper import DMCCartPoleWrapper


def test_long_rendering_opencv():
    """
    Tests the rendering of the DMCCartPoleWrapper over a very long period with random actions,
    using the faster OpenCV implementation.
    """
    print(
        "Testing long-term rendering with random actions (OpenCV) for 10,000 steps..."
    )
    try:
        num_steps = 10000
        # The environment will now run for the full num_steps before terminating
        env = DMCCartPoleWrapper(
            batch_size=1,
            max_steps=num_steps,
            visualize=True,
            dt_simulation=0.02,
        )
        env.reset()

        print(f"\n--- Running for {num_steps} steps with random actions ---")

        for i in range(num_steps):
            action = np.random.uniform(-1.0, 1.0, size=(1, env.action_dim))
            next_state, reward, terminated, info = env.step(action)

            # Print status less frequently to avoid flooding the console
            if (i + 1) % 100 == 0:
                print(
                    f"Step {i+1}/{num_steps}: Reward = {reward[0]:.3f}, Terminated = {terminated[0]}"
                )

            time.sleep(0.001)

            if terminated[0] and i < num_steps - 1:
                print(
                    f"Episode terminated unexpectedly at step {i+1}. This shouldn't happen."
                )
                break

        print("\nClosing environment...")
        env.close()
        print("Long-term rendering test complete!")

    except ImportError as e:
        print(f"Skipping test: {e}")
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_long_rendering_opencv()

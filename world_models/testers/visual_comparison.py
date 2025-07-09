import numpy as np
import cv2
from world_models.world_model_wrapper import WorldModelWrapper
from world_models.ini_gymlike_cartpole_wrapper import make_env, GymlikeCartpoleWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import time


# A simpler wrapper for visualization that doesn't use SubprocVecEnv
class VisCartpoleWrapper(DummyVecEnv):
    def __init__(self, n_envs=1):
        super().__init__([make_env for _ in range(n_envs)])
        self.n_envs = n_envs

    def set_state(self, state, env_idx=0):
        # This is a bit of a hack to set the state of the underlying environment
        # It assumes the base env is a CartPoleEnv with a 'state' attribute.
        self.envs[env_idx].unwrapped.state = state

    def render(self, *args, **kwargs):
        return self.envs[0].render()


def main():
    """
    Compares the trained World Model against the actual environment by rendering
    their outputs side-by-side for visual comparison.
    """
    print("--- Visually Comparing World Model to Real Environment ---")

    # --- Setup ---
    num_comparison_steps = 200

    # 1. Initialize the real environment for stepping
    real_env = GymlikeCartpoleWrapper(seed=42, n_envs=1)

    # 2. Initialize a separate, renderable environment for visualization
    vis_env = VisCartpoleWrapper(n_envs=1)

    # 3. Initialize the World Model environment
    try:
        world_model_env = WorldModelWrapper(
            simulated_env=vis_env,
            batch_size=1,
            trained_folder="world_models/trained/v1",
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(
            "Please ensure you have a trained model at 'world_models/trained/v1/model.pth'."
        )
        return

    # --- Comparison Loop ---
    real_state = real_env.reset()
    vis_env.reset()

    print(f"\nRunning visual comparison for {num_comparison_steps} steps...")
    for step in range(num_comparison_steps):
        # Synchronize the world model's state with the real environment's state
        world_model_env.set_state(real_state)

        # Choose a random action
        action = real_env.action_space.sample()

        # Step both environments
        wm_next_state, _, _, _ = world_model_env.step(np.array([action]))
        real_next_state, _, _, _ = real_env.step(np.array([action]))

        # --- Render both states ---
        # Render the real environment state
        vis_env.set_state(real_next_state[0])
        img_real = vis_env.render()

        # Render the world model's predicted state
        vis_env.set_state(wm_next_state[0])
        img_wm = vis_env.render()

        # Combine images side-by-side
        combined_img = np.concatenate((img_real, img_wm), axis=1)

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            combined_img,
            "Real Environment",
            (50, 50),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined_img,
            "World Model",
            (img_real.shape[1] + 50, 50),
            font,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Real vs. World Model", combined_img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        # Update the real state for the next iteration
        real_state = real_next_state
        time.sleep(0.02)  # Slow down for better visualization

    print("\n--- Visual comparison finished ---")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

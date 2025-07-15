#!/usr/bin/env python3
"""
Keyboard Control for Physical CartPole

This script allows manual control of the physical cartpole using keyboard input.
Use left and right arrow keys to control the cart movement.

Controls:
- Left Arrow:  Move cart left (negative action)
- Right Arrow: Move cart right (positive action)  
- 'q' or ESC:  Quit the program
- 'r':         Reset the environment
- 'h':         Show help

The script uses the PhysicalCartpoleWrapper with DMC reward function.
"""

import sys
import os
import time
import numpy as np

# Add path to access the physical cartpole utilities
sys.path.append(
    os.path.join("environments", "physical-cartpole", "Driver", "DriverFunctions")
)

try:
    from kbhit import KBHit
except ImportError:
    print(
        "Error: Could not import KBHit. Please ensure you're running from the correct directory."
    )
    print(
        "Expected path: environments/physical-cartpole/Driver/DriverFunctions/kbhit.py"
    )
    sys.exit(1)

from world_models.physical_cartpole_wrapper import PhysicalCartpoleWrapper


class KeyboardCartPoleController:
    """Keyboard controller for the physical cartpole."""

    def __init__(self):
        self.kb = None
        self.env = None
        self.running = False
        self.action_magnitude = 0.5  # Default action strength
        self.current_action = 0.0

        # Initialize keyboard input
        try:
            self.kb = KBHit()
            print("✓ Keyboard input initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize keyboard input: {e}")
            raise

    def setup_environment(self):
        """Initialize the physical cartpole environment."""
        try:
            print("Initializing physical cartpole environment...")
            self.env = PhysicalCartpoleWrapper(
                seed=42,
                n_envs=1,
                render_mode=None,  # Enable visual feedback
                max_episode_steps=1000,
            )
            print("✓ Physical cartpole environment initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize environment: {e}")
            print(
                "Please ensure the physical cartpole hardware is connected and accessible."
            )
            return False

    def print_help(self):
        """Print control instructions."""
        print("\n" + "=" * 50)
        print("PHYSICAL CARTPOLE KEYBOARD CONTROL")
        print("=" * 50)
        print("Controls:")
        print("  ← Left Arrow:  Move cart left")
        print("  → Right Arrow: Move cart right")
        print("  ↑ Up Arrow:    Increase action magnitude")
        print("  ↓ Down Arrow:  Decrease action magnitude")
        print("  'r':           Reset environment")
        print("  '+':           Increase action magnitude")
        print("  '-':           Decrease action magnitude")
        print("  'h':           Show this help")
        print("  'q' or ESC:    Quit")
        print(f"\nCurrent action magnitude: {self.action_magnitude:.2f}")
        print("=" * 50)

    def process_keyboard_input(self):
        """Process keyboard input and return corresponding action."""
        if not self.kb.kbhit():
            return 0.0  # No input, return neutral action

        try:
            # Try to get arrow key first
            if hasattr(self.kb, "getarrow"):
                try:
                    arrow = self.kb.getarrow()
                    if arrow == 1:  # Right arrow
                        return self.action_magnitude
                    elif arrow == 3:  # Left arrow
                        return -self.action_magnitude
                    elif arrow == 0:  # Up arrow - increase magnitude
                        self.action_magnitude = min(1.0, self.action_magnitude + 0.1)
                        print(
                            f"Action magnitude increased to: {self.action_magnitude:.2f}"
                        )
                        return 0.0
                    elif arrow == 2:  # Down arrow - decrease magnitude
                        self.action_magnitude = max(0.1, self.action_magnitude - 0.1)
                        print(
                            f"Action magnitude decreased to: {self.action_magnitude:.2f}"
                        )
                        return 0.0
                except:
                    # If getarrow fails, fall back to regular character input
                    pass

            # Regular character input
            c = self.kb.getch()

            if c == "q" or ord(c) == 27:  # 'q' or ESC
                print("\nQuitting...")
                self.running = False
                return 0.0
            elif c == "r":
                print("Resetting environment...")
                self.reset_environment()
                return 0.0
            elif c == "h":
                self.print_help()
                return 0.0
            elif c == "+" or c == "=":
                self.action_magnitude = min(1.0, self.action_magnitude + 0.1)
                print(f"Action magnitude increased to: {self.action_magnitude:.2f}")
                return 0.0
            elif c == "-":
                self.action_magnitude = max(0.1, self.action_magnitude - 0.1)
                print(f"Action magnitude decreased to: {self.action_magnitude:.2f}")
                return 0.0

        except Exception as e:
            print(f"Error processing keyboard input: {e}")

        return 0.0

    def reset_environment(self):
        """Reset the environment."""
        if self.env is not None:
            try:
                obs = self.env.reset()
                print("Environment reset successfully")
                return obs
            except Exception as e:
                print(f"Error resetting environment: {e}")
                return None
        return None

    def run(self):
        """Main control loop."""
        print("Starting Physical CartPole Keyboard Control...")

        # Setup environment
        if not self.setup_environment():
            return

        # Reset environment
        obs = self.reset_environment()
        if obs is None:
            print("Failed to reset environment. Exiting.")
            return

        # Show help
        self.print_help()
        print("\nStarting control loop... Press 'h' for help anytime.")
        print("Make sure this terminal window is focused to receive keyboard input.\n")

        self.running = True
        step_count = 0
        total_reward = 0.0

        try:
            # Variables for responsive control
            current_action = 0.0  # Current action to apply
            last_env_step_time = time.time()
            last_render_time = time.time()
            env_step_interval = 0.02  # Step environment every 20ms (50 Hz)
            render_interval = 0.1  # Render every 100ms (10 Hz)
            keyboard_check_interval = (
                0.001  # Check keyboard every 1ms for responsiveness
            )

            while self.running:
                current_time = time.time()

                # High-frequency keyboard input checking for maximum responsiveness
                new_action = self.process_keyboard_input()
                if new_action != 0.0:
                    current_action = new_action
                elif current_time - getattr(self, "_last_action_time", 0) > 0.05:
                    # If no new input for 50ms, decay action to zero for more natural control
                    current_action = 0.0

                if new_action != 0.0:
                    self._last_action_time = current_time

                if not self.running:
                    break

                # Step environment at controlled rate to avoid overwhelming physical hardware
                if current_time - last_env_step_time >= env_step_interval:
                    try:
                        # Convert scalar action to proper format for vectorized environment
                        # VecEnv expects actions for each environment in the batch
                        action_array = np.array(
                            [[current_action]], dtype=np.float32
                        )  # Shape: (n_envs, action_dim)
                        obs, reward, terminated, info = self.env.step(action_array)

                        step_count += 1
                        total_reward += (
                            reward[0] if hasattr(reward, "__len__") else reward
                        )
                        last_env_step_time = current_time

                        # Print status less frequently to reduce terminal spam
                        if step_count % 1 == 0:  # Every 50 steps instead of every step
                            print(
                                f"Step {step_count}, Total Reward: {total_reward:.2f}, Current Action: {current_action:+.2f}"
                            )
                            print(f"Observation: {obs}")

                        # Check if episode terminated
                        if (
                            terminated[0]
                            if hasattr(terminated, "__len__")
                            else terminated
                        ):
                            print(f"Episode terminated at step {step_count}")
                            print(f"Total reward: {total_reward:.2f}")
                            print("Resetting environment...")
                            obs = self.reset_environment()
                            step_count = 0
                            total_reward = 0.0

                        # Check for simulation stop signal
                        if hasattr(info, "__len__") and len(info) > 0:
                            if info[0].get("sim_should_stop", False):
                                print("Simulation stop signal received.")
                                self.running = False

                    except Exception as e:
                        print(f"Error during environment step: {e}")
                        print("Attempting to reset environment...")
                        obs = self.reset_environment()
                        if obs is None:
                            print("Failed to recover. Exiting.")
                            break
                        last_env_step_time = current_time  # Reset timer even on error

                # Render at a separate, lower frequency to improve performance
                if current_time - last_render_time >= render_interval:
                    try:
                        self.env.render()
                        last_render_time = current_time
                    except Exception as e:
                        print(f"Error rendering environment: {e}")

                # Small sleep to prevent excessive CPU usage while maintaining responsiveness
                time.sleep(keyboard_check_interval)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")

        if self.env is not None:
            try:
                self.env.close()
            except:
                pass

        if self.kb is not None:
            try:
                self.kb.set_normal_term()
            except:
                pass

        print("Cleanup complete.")


def main():
    """Main function."""
    print("Physical CartPole Keyboard Control")
    print("Make sure the physical cartpole hardware is connected and ready.")

    controller = KeyboardCartPoleController()

    try:
        controller.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        controller.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to visualize world model predictions by comparing predicted vs actual states and rewards.
This script loads a trained world model and runs it alongside the real environment to show
the quality of predictions and render frames for both predicted and actual states.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from networks.world_model_v1 import load_model, SimpleModel
from networks.world_model_rnn import load_model as load_rnn_model, RNNWorldModel
from utils.tools import resolve_device


class WorldModelVisualizer:
    """Visualizes world model predictions against actual environment states."""

    def __init__(self, config_path: str, model_path: str, env_type: str = "dmc"):
        """
        Initialize the visualizer.

        Args:
            config_path: Path to the configuration file
            model_path: Path to the trained world model
            env_type: Environment type ("dmc" or "gym")
        """
        self.config_path = config_path
        self.model_path = model_path
        self.env_type = env_type

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Resolve device
        self.device = resolve_device(self.config["global"]["device"])
        print(f"Using device: {self.device}")

        # Load world model
        self.world_model = self._load_world_model()

        # Display model type information
        self._display_model_info()

        # Display information about the valid init buffer
        self._display_init_buffer_info()

        # Create environments
        self.real_env, self.pred_env, self.pred_env_visual = self._create_environments()

        # Initialize data storage
        self.actual_states = []
        self.predicted_states = []
        self.actual_rewards = []
        self.predicted_rewards = []
        self.actions = []
        self.frames_actual = []
        self.frames_predicted = []

    def _load_world_model(self):
        """Load the trained world model (MLP or RNN)."""
        print(f"Loading world model from: {self.model_path}")

        # Try to load as RNN model first
        try:
            model = load_rnn_model(
                self.model_path, with_scalers=False, map_location=self.device
            )
            model.eval()
            print("Successfully loaded RNN world model")
            return model
        except Exception as e:
            print(f"Failed to load as RNN model: {e}")
            print("Trying to load as MLP model...")

            # Try to load as MLP model
            try:
                model = load_model(
                    self.model_path, with_scalers=False, map_location=self.device
                )
                model.eval()
                print("Successfully loaded MLP world model")
                return model
            except Exception as e2:
                print(f"Failed to load as MLP model: {e2}")
                raise RuntimeError("Could not load world model as either RNN or MLP")

        # Set up scalers if they exist
        model_dir = os.path.dirname(self.model_path)
        scaler_paths = {
            "state": os.path.join(model_dir, "state_scaler.joblib"),
            "action": os.path.join(model_dir, "action_scaler.joblib"),
            "reward": os.path.join(model_dir, "reward_scaler.joblib"),
        }

    def _display_model_info(self):
        """Display information about the loaded model type."""
        if isinstance(self.world_model, RNNWorldModel):
            print(f"Model type: RNN World Model")
            print(f"  Hidden dimension: {self.world_model.hidden_dim}")
            print(f"  Number of layers: {self.world_model.num_layers}")
            print(f"  State size: {self.world_model.state_size}")
            print(f"  Action size: {self.world_model.action_size}")
        else:
            print(f"Model type: MLP World Model")
            hidden_dim = getattr(self.world_model, "hidden_dim", None)
            if hidden_dim is not None:
                print(f"  Hidden dimension: {hidden_dim}")
            print(f"  State size: {self.world_model.state_size}")
            print(f"  Action size: {self.world_model.action_size}")

    def _display_init_buffer_info(self):
        """Display information about the valid init buffer."""
        if self.world_model.valid_init_state is not None:
            buffer_size = len(self.world_model.valid_init_state)
            print(f"Valid init buffer loaded: {buffer_size} states")

            # Show statistics about the buffer
            if buffer_size > 0:
                buffer_array = np.array(self.world_model.valid_init_state)
                print(f"  Buffer shape: {buffer_array.shape}")
                print(
                    f"  State dimension: {buffer_array.shape[1] if len(buffer_array.shape) > 1 else 'scalar'}"
                )

                # Show some statistics
                if len(buffer_array.shape) > 1:
                    print(
                        f"  State range: [{buffer_array.min():.3f}, {buffer_array.max():.3f}]"
                    )
                    print(f"  State mean: {buffer_array.mean():.3f}")
                    print(f"  State std: {buffer_array.std():.3f}")
        else:
            print("Warning: No valid init buffer found in the world model")
            print(
                "  This may indicate the model hasn't been trained yet or the buffer wasn't saved"
            )

    def _create_environments(self) -> Tuple:
        """Create real and predicted environments."""
        import platform

        if self.env_type == "dmc":
            try:
                from world_models.dmc_cartpole_wrapper import (
                    DMCWrapper,
                    make_dmc_env,
                    DMCCartpoleWrapper,
                )
            except ImportError as e:
                print(
                    "[ERROR] Failed to import dm_control or dmc_cartpole_wrapper. Mujoco rendering will not work."
                )
                print("        Error details:", e)
                print(
                    "        If you do not need Mujoco rendering, you can ignore this. Otherwise, check your installation."
                )
                raise RuntimeError("dm_control import failed: " + str(e))
            # On Windows, do not set MUJOCO_GL=egl. Use 'rgb_array' render mode for frame capture.
            render_mode = "rgb_array"
            real_env = DMCWrapper(
                "cartpole",
                "swingup",
                render_mode=render_mode,
                max_episode_steps=1000,
                dt_simulation=0.02,
            )
            # Create a wrapper for the world model predictions
            pred_env = self._create_world_model_env()
            pred_env_visual = DMCWrapper(
                "cartpole",
                "swingup",
                render_mode="human",
                max_episode_steps=1000,
                dt_simulation=0.02,
            )
        else:
            raise ValueError(
                f"Unsupported environment type: {self.env_type}. Only 'dmc' is supported."
            )
        return real_env, pred_env, pred_env_visual

    def _create_world_model_env(self):
        """Create a wrapper environment that uses the world model for predictions."""

        class WorldModelEnv:
            def __init__(self, world_model, config, device):
                self.world_model = world_model
                self.config = config
                self.device = device
                self.state = None
                self.step_count = 0
                self.max_episode_steps = 1000
                self.last_action = None  # Store last action for visualization

                # Get environment dimensions from world model
                self.state_size = world_model.state_size
                self.action_size = world_model.action_size

                # Get configuration settings
                self.use_scalers = config["world_model_trainer"]["use_scalers"]
                self.use_output_state_scaler = config["world_model_trainer"][
                    "use_output_state_scaler"
                ]
                self.use_output_reward_scaler = config["world_model_trainer"][
                    "use_output_reward_scaler"
                ]

                # Check if this is an RNN model
                self.is_rnn = isinstance(world_model, RNNWorldModel)

            def reset(self):
                """Reset the environment with an initial state from the valid init buffer."""
                # Get initial state from the valid init buffer if available
                if (
                    self.world_model.valid_init_state is not None
                    and len(self.world_model.valid_init_state) > 0
                ):
                    # Sample a random state from the buffer
                    buffer_size = len(self.world_model.valid_init_state)
                    random_idx = np.random.randint(0, buffer_size)
                    init_state = self.world_model.valid_init_state[random_idx]

                    # Convert to tensor and ensure correct shape
                    if isinstance(init_state, np.ndarray):
                        self.state = torch.tensor(
                            init_state, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                    else:
                        self.state = (
                            init_state.unsqueeze(0)
                            if init_state.dim() == 1
                            else init_state
                        )

                    print(
                        f"  Using initial state from buffer (index {random_idx}/{buffer_size})"
                    )
                else:
                    # Fallback to random state if buffer is not available
                    print(
                        "  Warning: No valid init buffer available, using random state"
                    )
                    self.state = torch.randn(1, self.state_size, device=self.device)

                self.step_count = 0
                return self.state.cpu().numpy().flatten()

            def step(self, action):
                """Take a step using the world model."""
                if self.state is None:
                    raise RuntimeError("Environment not reset")
                # Store the last action for visualization
                self.last_action = action

                # Convert action to tensor
                action_tensor = torch.tensor(
                    action, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                if self.is_rnn:
                    # For RNN model, we need to create a sequence input
                    # Since we're doing single-step prediction, we create a sequence of length 1
                    model_input = torch.cat([self.state, action_tensor], dim=1)
                    model_input = model_input.unsqueeze(
                        1
                    )  # Add sequence dimension: (batch_size, 1, state_size + action_size)

                    # Get prediction from RNN world model
                    with torch.no_grad():
                        prediction = self.world_model(
                            model_input,
                            use_input_state_scaler=self.use_scalers,
                            use_input_action_scaler=self.use_scalers,
                            use_output_state_scaler=True,
                            use_output_reward_scaler=self.use_output_reward_scaler,
                        )

                    # Remove sequence dimension and extract outputs
                    prediction = prediction.squeeze(1)  # (batch_size, state_size + 1)
                    next_state = prediction[:, : self.state_size]
                    reward = prediction[:, self.state_size]
                else:
                    # For MLP model, use the original approach
                    model_input = torch.cat([self.state, action_tensor], dim=1)

                    # Get prediction from world model
                    with torch.no_grad():
                        prediction = self.world_model(
                            model_input,
                            use_input_state_scaler=self.use_scalers,
                            use_input_action_scaler=self.use_scalers,
                            use_output_state_scaler=True,
                            use_output_reward_scaler=self.use_output_reward_scaler,
                        )

                    # Extract next state and reward
                    next_state = prediction[:, : self.state_size]
                    reward = prediction[:, self.state_size]

                # Update state
                self.state = next_state
                self.step_count += 1

                # Check if episode is done
                done = self.step_count >= self.max_episode_steps

                return (
                    next_state.cpu().numpy().flatten(),
                    reward.cpu().numpy().item(),
                    done,
                    {},
                )

            def render(self, mode="rgb_array"):
                """Render the current state using a custom renderer for DMC states."""
                try:
                    # Create a simple visualization of the predicted state
                    frame = np.zeros((400, 600, 3), dtype=np.uint8)
                    frame.fill(255)  # White background

                    if self.state is not None:
                        state = self.state.cpu().numpy().flatten()

                        # For DMC cartpole, the state structure is typically:
                        # [position, cos(angle), sin(angle), velocity, angular_velocity]
                        # But we need to handle the flattened structure from DMC

                        # Try to extract cart position and angle from the flattened state
                        # This is a simplified approach - you might need to adjust based on your specific DMC setup
                        if len(state) >= 5:
                            # Assume first element is position, 2nd and 3rd are sin/cos of angle
                            cart_pos = state[0] if len(state) > 0 else 0
                            angle_cos = state[1] if len(state) > 1 else 1
                            angle_sin = state[2] if len(state) > 2 else 0
                            angle = np.arctan2(angle_sin, angle_cos)
                            cart_vel = state[3] if len(state) > 3 else 0
                            angle_vel = state[4] if len(state) > 4 else 0
                        else:
                            # Fallback for different state structures
                            cart_pos = 0
                            angle = 0

                        # self.real_env.physics.set_state(  # position, angle, vel, angle_vel
                        #     [cart_pos, angle, cart_vel, angle_vel]
                        # )
                        # self.real_env.physics.step()
                        # real_frame = self.real_env.render()

                        # cv2.imshow("Real Environment", real_frame)
                        # cv2.waitKey(1)

                        # Draw cart and pole (simplified visualization)
                        # Scale to image coordinates
                        scale = 150
                        center_x = 300 + int(cart_pos * scale)
                        center_y = 200

                        # Draw cart (rectangle)
                        cart_width, cart_height = 40, 20
                        cv2.rectangle(
                            frame,
                            (center_x - cart_width // 2, center_y - cart_height // 2),
                            (center_x + cart_width // 2, center_y + cart_height // 2),
                            (0, 0, 0),
                            -1,
                        )

                        # Draw pole (line) - flip the angle to match actual environment
                        pole_length = 150
                        pole_end_x = center_x + int(pole_length * np.sin(angle))
                        pole_end_y = center_y - int(pole_length * np.cos(angle))
                        cv2.line(
                            frame,
                            (center_x, center_y),
                            (pole_end_x, pole_end_y),
                            (139, 69, 19),
                            5,
                        )  # Brown pole

                        # Draw ground line
                        cv2.line(
                            frame,
                            (0, center_y + cart_height // 2),
                            (600, center_y + cart_height // 2),
                            (0, 0, 0),
                            2,
                        )

                        # Draw action arrow below the cart
                        if self.last_action is not None:
                            # Assume action is a 1D array or scalar
                            action_val = (
                                float(self.last_action[0])
                                if hasattr(self.last_action, "__len__")
                                and len(self.last_action) > 0
                                else float(self.last_action)
                            )
                            arrow_length = int(
                                40 + 60 * abs(action_val)
                            )  # min 40, max 100 px
                            arrow_color = (0, 0, 255)  # Red arrow
                            arrow_thickness = 4
                            base_y = center_y + cart_height // 2 + 20
                            base_x = center_x
                            if action_val >= 0:
                                tip_x = base_x + arrow_length
                            else:
                                tip_x = base_x - arrow_length
                            tip_y = base_y
                            cv2.arrowedLine(
                                frame,
                                (base_x, base_y),
                                (tip_x, tip_y),
                                arrow_color,
                                arrow_thickness,
                                tipLength=0.3,
                            )
                            # Optionally, add text for action value
                            cv2.putText(
                                frame,
                                f"Action: {action_val:.2f}",
                                (base_x - 50, base_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )

                        # Add text to indicate this is a prediction
                        cv2.putText(
                            frame,
                            "PREDICTED",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )

                        # Add state info for debugging
                        cv2.putText(
                            frame,
                            f"State dim: {len(state)}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

                    return frame
                except Exception as e:
                    # Fallback to placeholder with error info
                    frame = np.zeros((400, 600, 3), dtype=np.uint8)
                    frame.fill(255)
                    cv2.putText(
                        frame,
                        f"Render Error: {str(e)[:20]}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )
                    return frame

        return WorldModelEnv(self.world_model, self.config, self.device)

    def _get_state_from_dmc_state(self, state):
        position = state[0]
        angle_cos = state[1]
        angle_sin = state[2]
        angle = np.arctan2(angle_sin, angle_cos)
        velocity = state[3]
        angle_vel = state[4]
        return np.array([position, angle, velocity, angle_vel])

    def run_comparison(
        self,
        num_episodes: int = 3,
        rollout_length: int = 16,
        random_actions: bool = False,
        save_frames: bool = True,
    ):
        """
        Run comparison between real and predicted environments using autoregressive rollouts.

        Args:
            num_episodes: Number of episodes to run
            rollout_length: Length of autoregressive rollouts (default: 16)
            random_actions: Whether to use random actions or a simple policy
            save_frames: Whether to save rendered frames
        """
        print(
            f"Running autoregressive rollout comparison for {num_episodes} episodes..."
        )
        print(f"Rollout length: {rollout_length} steps")

        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")

            # Reset environments
            actual_state = self.real_env.reset()[0]
            predicted_state = self.pred_env.reset()

            # Run real environment for a few steps with random actions to get a more realistic starting state
            print("  Running real environment for warm-up steps...")
            warmup_steps = 100  # Number of warm-up steps
            for step in range(warmup_steps):
                random_action = self.real_env.action_space.sample()
                actual_state, _, _, _, _ = self.real_env.step(random_action)
                print(
                    f"    Warm-up step {step + 1}: action={random_action[0]:.3f}, state={actual_state[:3]}"
                )

            # Use the final state from warm-up as the starting point for both environments
            if hasattr(self.pred_env, "state") and self.pred_env.state is not None:
                # Set the predicted environment to use the warm-up state
                warmup_state = torch.tensor(
                    actual_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                self.pred_env.state = warmup_state
                predicted_state = actual_state.copy()
                print(
                    f"  Using warm-up state as starting point: {actual_state[:3]}..."
                )  # Show first 3 elements

            state = self._get_state_from_dmc_state(actual_state)

            self.pred_env_visual.env.physics.set_state(state)

            episode_actual_states = [actual_state.copy()]
            episode_predicted_states = [predicted_state.copy()]
            episode_actual_rewards = []
            episode_predicted_rewards = []
            episode_actions = []
            episode_frames_actual = []
            episode_frames_predicted = []

            # Generate action sequence for the entire rollout
            if random_actions:
                actions = [
                    self.real_env.action_space.sample() for _ in range(rollout_length)
                ]
            else:
                # Generate actions using a simple policy
                actions = []
                for step in range(rollout_length):
                    state = self._get_state_from_dmc_state(
                        actual_state
                    )  # pos, angle, vel, angle_vel
                    # Simple swing-up policy
                    action = np.array(
                        [0.5 * np.sin(step * 0.1) - 0.1 * state[0] - 0.05 * state[2]]
                    )
                    action = np.clip(action, -1.0, 1.0)
                    actions.append(action)

            # Execute real environment rollout
            print("  Executing real environment rollout...")
            actual_rollout_states = [actual_state.copy()]
            actual_rollout_rewards = []

            for step, action in enumerate(actions):

                actual_state, actual_reward, done, _, info = self.real_env.step(action)
                actual_rollout_states.append(actual_state.copy())
                actual_rollout_rewards.append(actual_reward)

                if save_frames:
                    try:
                        actual_frame = self.real_env.render().copy()
                        if actual_frame is None:
                            print(f"[Warning] Actual frame at step {step} is None.")
                        episode_frames_actual.append(actual_frame)
                    except Exception as e:
                        print(
                            f"[Error] Exception during actual frame render at step {step}: {e}"
                        )
                        episode_frames_actual.append(None)

                if done:
                    break

            # Execute predicted environment rollout (autoregressive)
            print("  Executing predicted environment rollout...")
            predicted_rollout_states = [predicted_state.copy()]
            predicted_rollout_rewards = []

            # Reset the predicted environment to the same initial state (warm-up state)
            self.pred_env.state = torch.tensor(
                actual_rollout_states[0], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            for step, action in enumerate(actions):

                predicted_state, predicted_reward, pred_done, pred_info = (
                    self.pred_env.step(action)
                )
                predicted_rollout_states.append(predicted_state.copy())
                predicted_rollout_rewards.append(predicted_reward)

                # Render frames if requested
                if save_frames:
                    try:
                        state = self._get_state_from_dmc_state(predicted_state)
                        self.pred_env_visual.env.physics.set_state(state)
                        self.pred_env_visual.env.physics.step()
                        predicted_frame = self.pred_env_visual.env.physics.render(
                            camera_id=0
                        )
                        # Add action visualization to predicted frame
                        if (
                            predicted_frame is not None
                            and len(predicted_frame.shape) == 3
                        ):
                            # Convert to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR)

                            # Add action text in yellow
                            action_text = (
                                f"Action: {action[0]:.3f}"
                                if len(action) > 0
                                else "Action: N/A"
                            )
                            cv2.putText(
                                frame_bgr,
                                action_text,
                                (10, 30),  # Position (x, y)
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,  # Font scale
                                (0, 255, 255),  # Yellow color in BGR
                                2,  # Thickness
                                cv2.LINE_AA,
                            )

                            # Add arrow based on action magnitude and sign
                            if len(action) > 0:
                                action_val = action[0]
                                frame_height, frame_width = frame_bgr.shape[:2]

                                # Arrow parameters
                                center_x = frame_width // 2
                                center_y = frame_height - 50
                                max_arrow_length = 80

                                # Calculate arrow length based on magnitude (clamp to reasonable range)
                                arrow_length = int(
                                    min(
                                        abs(action_val) * max_arrow_length,
                                        max_arrow_length,
                                    )
                                )

                                # Calculate arrow direction based on sign
                                if action_val > 0:
                                    # Positive action - arrow points right
                                    end_x = center_x + arrow_length
                                    end_y = center_y
                                elif action_val < 0:
                                    # Negative action - arrow points left
                                    end_x = center_x - arrow_length
                                    end_y = center_y
                                else:
                                    # Zero action - small circle
                                    cv2.circle(
                                        frame_bgr,
                                        (center_x, center_y),
                                        5,
                                        (0, 255, 255),
                                        -1,
                                    )
                                    end_x = center_x
                                    end_y = center_y

                                # Draw arrow if action is non-zero
                                if action_val != 0:
                                    cv2.arrowedLine(
                                        frame_bgr,
                                        (center_x, center_y),
                                        (end_x, end_y),
                                        (0, 255, 255),  # Yellow color in BGR
                                        3,  # Thickness
                                        tipLength=0.3,
                                    )

                            # Convert back to RGB
                            predicted_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        if predicted_frame is None:
                            print(f"[Warning] Predicted frame at step {step} is None.")
                        episode_frames_predicted.append(predicted_frame)
                    except Exception as e:
                        print(
                            f"[Error] Exception during predicted frame render at step {step}: {e}"
                        )
                        episode_frames_predicted.append(None)

                if pred_done:
                    break

            # Store rollout data
            episode_actual_states.extend(
                actual_rollout_states[1:]
            )  # Skip initial state (already added)
            episode_predicted_states.extend(predicted_rollout_states[1:])
            episode_actual_rewards.extend(actual_rollout_rewards)
            episode_predicted_rewards.extend(predicted_rollout_rewards)
            episode_actions.extend(actions)

            # Store episode data
            self.actual_states.append(np.array(episode_actual_states))
            self.predicted_states.append(np.array(episode_predicted_states))
            self.actual_rewards.append(np.array(episode_actual_rewards))
            self.predicted_rewards.append(np.array(episode_predicted_rewards))
            self.actions.append(np.array(episode_actions))
            self.frames_actual.append(episode_frames_actual)
            self.frames_predicted.append(episode_frames_predicted)

            print(f"  Rollout completed in {len(actions)} steps")
            print(f"  Total actual reward: {sum(actual_rollout_rewards):.3f}")
            print(f"  Total predicted reward: {sum(predicted_rollout_rewards):.3f}")

            # Print step-by-step comparison for first few steps
            print("  Step-by-step comparison (first 5 steps):")
            for step in range(min(5, len(actual_rollout_rewards))):
                actual_reward = actual_rollout_rewards[step]
                predicted_reward = predicted_rollout_rewards[step]
                print(
                    f"    Step {step}: Actual: {actual_reward:.3f}, Predicted: {predicted_reward:.3f}"
                )

    def plot_comparisons(self, save_dir: str = "world_model_visualization"):
        """Create comprehensive plots comparing actual vs predicted data."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"Saving plots to: {save_dir}")

        # Plot 1: State comparison over time
        self._plot_states_comparison(save_dir)

        # Plot 2: Reward comparison over time
        self._plot_rewards_comparison(save_dir)

        # Plot 3: State prediction errors
        self._plot_prediction_errors(save_dir)

        # Plot 4: Reward prediction errors
        self._plot_reward_errors(save_dir)

        # Plot 5: Action sequences
        self._plot_actions(save_dir)

        # Save frames if available
        self._save_frames(save_dir)

        print(f"All plots saved to: {save_dir}")

    def _plot_states_comparison(self, save_dir: str):
        """Plot actual vs predicted states over time."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("State Comparison: Actual vs Predicted", fontsize=16)

        state_names = [
            "Cart Position",
            "Angle Cos",
            "Angle Sin",
            "Cart Velocity",
            "Angle Velocity",
            "Angle",
        ]

        for episode_idx in range(
            min(len(self.actual_states), 2)
        ):  # Plot first 2 episodes
            actual_states = self.actual_states[episode_idx]
            angle = np.arctan2(actual_states[:, 2], actual_states[:, 1]).reshape(-1, 1)
            actual_states = np.append(actual_states, angle, axis=1)

            predicted_states = self.predicted_states[episode_idx]
            angle = np.arctan2(predicted_states[:, 2], predicted_states[:, 1]).reshape(
                -1, 1
            )
            predicted_states = np.append(predicted_states, angle, axis=1)

            for state_idx in range(min(6, actual_states.shape[1])):
                row = episode_idx
                col = state_idx

                if col < 3:  # First row
                    ax = axes[0, col]
                else:  # Second row
                    ax = axes[1, col - 3]

                time_steps = np.arange(len(actual_states))
                ax.plot(
                    time_steps,
                    actual_states[:, state_idx],
                    "b-",
                    label="Actual",
                    linewidth=2,
                )
                ax.plot(
                    time_steps,
                    predicted_states[:, state_idx],
                    "r--",
                    label="Predicted",
                    linewidth=2,
                )
                ax.set_title(f"{state_names[state_idx]} (Episode {episode_idx + 1})")
                ax.set_xlabel("Time Step")
                ax.set_ylabel("State Value")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "state_comparison.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_rewards_comparison(self, save_dir: str):
        """Plot actual vs predicted rewards over time."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Reward Comparison: Actual vs Predicted", fontsize=16)

        for episode_idx in range(min(len(self.actual_rewards), 4)):
            row = episode_idx // 2
            col = episode_idx % 2

            actual_rewards = self.actual_rewards[episode_idx]
            predicted_rewards = self.predicted_rewards[episode_idx]

            time_steps = np.arange(len(actual_rewards))

            # Individual rewards
            axes[row, col].plot(
                time_steps, actual_rewards, "b-", label="Actual", linewidth=2
            )
            axes[row, col].plot(
                time_steps, predicted_rewards, "r--", label="Predicted", linewidth=2
            )
            axes[row, col].set_title(f"Episode {episode_idx + 1}")
            axes[row, col].set_xlabel("Time Step")
            axes[row, col].set_ylabel("Reward")
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "reward_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_prediction_errors(self, save_dir: str):
        """Plot state prediction errors."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("State Prediction Errors", fontsize=16)

        state_names = [
            "Cart Position",
            "Angle Cos",
            "Angle Sin",
            "Cart Velocity",
            "Angle Velocity",
        ]

        # Combine all episodes
        all_actual = np.concatenate(self.actual_states, axis=0)
        all_predicted = np.concatenate(self.predicted_states, axis=0)

        errors = all_actual - all_predicted

        for state_idx in range(min(6, errors.shape[1])):
            row = state_idx // 3
            col = state_idx % 3

            ax = axes[row, col]

            time_steps = np.arange(len(errors))
            ax.plot(time_steps, errors[:, state_idx], "g-", linewidth=1)
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
            ax.set_title(f"{state_names[state_idx]} Error")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Error (Actual - Predicted)")
            ax.grid(True, alpha=0.3)

            # Add error statistics
            mean_error = np.mean(errors[:, state_idx])
            std_error = np.std(errors[:, state_idx])
            ax.text(
                0.02,
                0.98,
                f"Mean: {mean_error:.3f}\nStd: {std_error:.3f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "state_prediction_errors.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_reward_errors(self, save_dir: str):
        """Plot reward prediction errors."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Reward Prediction Errors", fontsize=16)

        for episode_idx in range(min(len(self.actual_rewards), 4)):
            row = episode_idx // 2
            col = episode_idx % 2

            actual_rewards = self.actual_rewards[episode_idx]
            predicted_rewards = self.predicted_rewards[episode_idx]

            reward_errors = actual_rewards - predicted_rewards
            time_steps = np.arange(len(reward_errors))

            axes[row, col].plot(time_steps, reward_errors, "g-", linewidth=2)
            axes[row, col].axhline(y=0, color="k", linestyle="--", alpha=0.5)
            axes[row, col].set_title(f"Episode {episode_idx + 1} Reward Error")
            axes[row, col].set_xlabel("Time Step")
            axes[row, col].set_ylabel("Reward Error (Actual - Predicted)")
            axes[row, col].grid(True, alpha=0.3)

            # Add error statistics
            mean_error = np.mean(reward_errors)
            std_error = np.std(reward_errors)
            axes[row, col].text(
                0.02,
                0.98,
                f"Mean: {mean_error:.3f}\nStd: {std_error:.3f}",
                transform=axes[row, col].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "reward_prediction_errors.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_actions(self, save_dir: str):
        """Plot action sequences."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Action Sequences", fontsize=16)

        for episode_idx in range(min(len(self.actions), 4)):
            row = episode_idx // 2
            col = episode_idx % 2

            actions = self.actions[episode_idx]
            time_steps = np.arange(len(actions))

            axes[row, col].plot(time_steps, actions, "purple", linewidth=2)
            axes[row, col].set_title(f"Episode {episode_idx + 1} Actions")
            axes[row, col].set_xlabel("Time Step")
            axes[row, col].set_ylabel("Action Value")
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_ylim(-1.1, 1.1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, "action_sequences.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _save_frames(self, save_dir: str):
        """Save rendered frames if available."""
        import cv2

        frames_dir = os.path.join(save_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        TARGET_HEIGHT, TARGET_WIDTH = 270, 360  # Match DMCWrapper real env

        for episode_idx, (actual_frames, predicted_frames) in enumerate(
            zip(self.frames_actual, self.frames_predicted)
        ):
            episode_dir = os.path.join(frames_dir, f"episode_{episode_idx + 1}")
            os.makedirs(episode_dir, exist_ok=True)

            actual_dir = os.path.join(episode_dir, "actual")
            predicted_dir = os.path.join(episode_dir, "predicted")
            os.makedirs(actual_dir, exist_ok=True)
            os.makedirs(predicted_dir, exist_ok=True)

            actual_row = []
            predicted_row = []

            for step_idx, (actual_frame, predicted_frame) in enumerate(
                zip(actual_frames, predicted_frames)
            ):
                # Save actual frame
                if actual_frame is not None:
                    print(
                        f"[DEBUG] Actual frame at step {step_idx}: type={type(actual_frame)}, shape={getattr(actual_frame, 'shape', None)}, dtype={getattr(actual_frame, 'dtype', None)}"
                    )
                    try:
                        if actual_frame.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                            actual_frame = cv2.resize(
                                actual_frame, (TARGET_WIDTH, TARGET_HEIGHT)
                            )
                        out_path = os.path.join(actual_dir, f"step_{step_idx:03d}.png")
                        result = cv2.imwrite(
                            out_path, cv2.cvtColor(actual_frame, cv2.COLOR_RGB2BGR)
                        )
                        print(
                            f"[DEBUG] cv2.imwrite(actual) returned {result} for {out_path}"
                        )
                        if not result:
                            print(f"[ERROR] Failed to save actual frame at {out_path}")
                        actual_row.append(actual_frame)
                    except Exception as e:
                        print(
                            f"[ERROR] Exception saving actual frame at step {step_idx}: {e}"
                        )
                        actual_row.append(
                            np.full(
                                (TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8
                            )
                        )
                else:
                    actual_row.append(
                        np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8)
                    )

                # Save predicted frame
                if predicted_frame is not None:
                    print(
                        f"[DEBUG] Predicted frame at step {step_idx}: type={type(predicted_frame)}, shape={getattr(predicted_frame, 'shape', None)}, dtype={getattr(predicted_frame, 'dtype', None)}"
                    )
                    try:
                        if predicted_frame.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                            predicted_frame = cv2.resize(
                                predicted_frame, (TARGET_WIDTH, TARGET_HEIGHT)
                            )
                        out_path = os.path.join(
                            predicted_dir, f"step_{step_idx:03d}.png"
                        )
                        result = cv2.imwrite(
                            out_path, cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR)
                        )
                        print(
                            f"[DEBUG] cv2.imwrite(predicted) returned {result} for {out_path}"
                        )
                        if not result:
                            print(
                                f"[ERROR] Failed to save predicted frame at {out_path}"
                            )
                        predicted_row.append(predicted_frame)
                    except Exception as e:
                        print(
                            f"[ERROR] Exception saving predicted frame at step {step_idx}: {e}"
                        )
                        predicted_row.append(
                            np.full(
                                (TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8
                            )
                        )
                else:
                    predicted_row.append(
                        np.full((TARGET_HEIGHT, TARGET_WIDTH, 3), 255, dtype=np.uint8)
                    )

            # Combine all actual and predicted frames into a single image
            if actual_row and predicted_row:
                combined_actual = np.concatenate(actual_row, axis=1)
                combined_predicted = np.concatenate(predicted_row, axis=1)
                combined_image = np.concatenate(
                    [combined_actual, combined_predicted], axis=0
                )
                combined_path = os.path.join(
                    episode_dir, f"combined_episode_{episode_idx + 1}.png"
                )
                result = cv2.imwrite(
                    combined_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
                )
                print(f"[DEBUG] Combined image saved: {combined_path}, result={result}")

        print(f"Frames saved to: {frames_dir}")

    def test_sequence_prediction(self):
        """Test sequence prediction if using RNN model."""
        if isinstance(self.world_model, RNNWorldModel):
            print("\n" + "=" * 50)
            print("TESTING RNN SEQUENCE PREDICTION")
            print("=" * 50)

            # Get imag_horizon from config
            imag_horizon = self.config["dreamer_agent_trainer"]["imag_horizon"]
            print(f"Testing sequence prediction with imag_horizon: {imag_horizon}")

            # Create test data
            batch_size = 2
            initial_state = torch.randn(
                batch_size, self.world_model.state_size, device=self.device
            )
            actions = torch.randn(
                batch_size,
                imag_horizon,
                self.world_model.action_size,
                device=self.device,
            )

            print(f"Initial state shape: {initial_state.shape}")
            print(f"Actions shape: {actions.shape}")

            # Test sequence prediction
            with torch.no_grad():
                states, rewards = self.world_model.predict_sequence(
                    initial_state,
                    actions,
                    use_input_state_scaler=self.config["world_model_trainer"][
                        "use_scalers"
                    ],
                    use_input_action_scaler=self.config["world_model_trainer"][
                        "use_scalers"
                    ],
                    use_output_state_scaler=True,
                    use_output_reward_scaler=self.config["world_model_trainer"][
                        "use_output_reward_scaler"
                    ],
                )

            print(f"Predicted states shape: {states.shape}")
            print(f"Predicted rewards shape: {rewards.shape}")

            # Show some statistics
            print(
                f"States range: [{states.min().item():.3f}, {states.max().item():.3f}]"
            )
            print(
                f"Rewards range: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]"
            )
            print("RNN sequence prediction test completed successfully!")
            print("=" * 50)

    def print_statistics(self):
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("WORLD MODEL PREDICTION STATISTICS")
        print("=" * 50)

        # Combine all episodes
        all_actual_states = np.concatenate(self.actual_states, axis=0)
        all_predicted_states = np.concatenate(self.predicted_states, axis=0)
        all_actual_rewards = np.concatenate(self.actual_rewards, axis=0)
        all_predicted_rewards = np.concatenate(self.predicted_rewards, axis=0)

        # State prediction statistics
        state_errors = all_actual_states - all_predicted_states
        state_rmse = np.sqrt(np.mean(state_errors**2, axis=0))
        state_mae = np.mean(np.abs(state_errors), axis=0)

        print("\nState Prediction Statistics:")
        state_names = [
            "Cart Position",
            "Angle Cos",
            "Angle Sin",
            "Cart Velocity",
            "Angle Velocity",
        ]
        for i, name in enumerate(state_names):
            if i < len(state_rmse):
                print(f"  {name}:")
                print(f"    RMSE: {state_rmse[i]:.4f}")
                print(f"    MAE:  {state_mae[i]:.4f}")

        # Reward prediction statistics
        reward_errors = all_actual_rewards - all_predicted_rewards
        reward_rmse = np.sqrt(np.mean(reward_errors**2))
        reward_mae = np.mean(np.abs(reward_errors))
        reward_corr = np.corrcoef(all_actual_rewards, all_predicted_rewards)[0, 1]

        print(f"\nReward Prediction Statistics:")
        print(f"  RMSE: {reward_rmse:.4f}")
        print(f"  MAE:  {reward_mae:.4f}")
        print(f"  Correlation: {reward_corr:.4f}")

        # Episode statistics
        print(f"\nEpisode Statistics:")
        print(f"  Number of episodes: {len(self.actual_states)}")
        print(
            f"  Average episode length: {np.mean([len(r) for r in self.actual_rewards]):.1f} steps"
        )
        print(f"  Total actual reward: {np.sum(all_actual_rewards):.3f}")
        print(f"  Total predicted reward: {np.sum(all_predicted_rewards):.3f}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Visualize world model predictions")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/full_system_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained world model",
        default="C:/Users/tz124/work-local/telluride/Neuromorphic_MBRL/runs/20250715_222251/model.pth",
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="dmc",
        choices=["dmc"],
        help="Environment type (only 'dmc' supported)",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument(
        "--rollout_length",
        type=int,
        default=16,
        help="Length of autoregressive rollouts",
    )
    parser.add_argument(
        "--random_actions",
        action="store_true",
        help="Use random actions instead of simple policy",
    )
    parser.add_argument(
        "--save_frames", action="store_true", help="Save rendered frames"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="world_model_visualization",
        help="Output directory for plots and frames",
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = WorldModelVisualizer(args.config, args.model, args.env_type)

    # Test sequence prediction if using RNN model
    visualizer.test_sequence_prediction()

    # Run comparison
    visualizer.run_comparison(
        num_episodes=args.episodes,
        rollout_length=args.rollout_length,
        random_actions=args.random_actions,
        save_frames=args.save_frames,
    )

    # Create plots
    visualizer.plot_comparisons(args.output_dir)

    # Print statistics
    visualizer.print_statistics()

    print(f"\nVisualization complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example script demonstrating how to use the world model visualization tool.
This script shows how to run autoregressive rollouts and visualize the results.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.visualize_world_model_predictions import WorldModelVisualizer


def main():
    """Example usage of the WorldModelVisualizer."""

    # Configuration
    config_path = "configs/full_system_config.yaml"
    model_path = "runs/model.pth"  # Update this to your actual model path
    env_type = "dmc"  # Only DMC is supported

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a world model first or update the model_path variable.")
        return

    # Create visualizer
    print("Creating world model visualizer...")
    visualizer = WorldModelVisualizer(config_path, model_path, env_type)

    # Run autoregressive rollouts
    print("\nRunning autoregressive rollouts...")
    visualizer.run_comparison(
        num_episodes=3,
        rollout_length=16,  # 16-step autoregressive rollouts
        random_actions=False,  # Use simple policy instead of random actions
        save_frames=True,
    )

    # Create plots and save results
    print("\nCreating visualization plots...")
    output_dir = "world_model_visualization_results"
    visualizer.plot_comparisons(output_dir)

    # Print statistics
    visualizer.print_statistics()

    print(f"\nVisualization complete! Check {output_dir} for results.")
    print("\nGenerated files:")
    print(f"  - {output_dir}/state_comparison.png")
    print(f"  - {output_dir}/reward_comparison.png")
    print(f"  - {output_dir}/state_prediction_errors.png")
    print(f"  - {output_dir}/reward_prediction_errors.png")
    print(f"  - {output_dir}/action_sequences.png")
    print(f"  - {output_dir}/frames/ (rendered frames)")


if __name__ == "__main__":
    main()

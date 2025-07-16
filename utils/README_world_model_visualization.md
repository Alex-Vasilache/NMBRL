# World Model Visualization Tool

This tool visualizes world model predictions by comparing predicted states and rewards against actual environment states using autoregressive rollouts.

## Features

- **Autoregressive Rollouts**: Performs 16-step (or custom length) autoregressive rollouts where the world model predicts the next state based on previous predicted states
- **State Comparison**: Plots actual vs predicted states over time
- **Reward Comparison**: Plots actual vs predicted rewards over time
- **Error Analysis**: Shows prediction errors for both states and rewards
- **Frame Rendering**: Renders frames for both actual and predicted states using DMC environment
- **Statistics**: Provides detailed statistics on prediction accuracy

## Usage

### Command Line Interface

```bash
python utils/visualize_world_model_predictions.py --model path/to/model.pth [options]
```

#### Arguments

- `--model`: Path to the trained world model (required)
- `--config`: Path to configuration file (default: `configs/full_system_config.yaml`)
- `--env_type`: Environment type - "dmc" only (default: "dmc")
- `--episodes`: Number of episodes to run (default: 3)
- `--rollout_length`: Length of autoregressive rollouts (default: 16)
- `--random_actions`: Use random actions instead of simple policy
- `--save_frames`: Save rendered frames
- `--output_dir`: Output directory for plots and frames (default: "world_model_visualization")

#### Examples

```bash
# Basic usage with 16-step rollouts
python utils/visualize_world_model_predictions.py --model runs/model.pth

# Custom rollout length
python utils/visualize_world_model_predictions.py --model runs/model.pth --rollout_length 32

# Use random actions
python utils/visualize_world_model_predictions.py --model runs/model.pth --random_actions

# Save frames and specify output directory
python utils/visualize_world_model_predictions.py --model runs/model.pth --save_frames --output_dir my_results
```

### Programmatic Usage

```python
from utils.visualize_world_model_predictions import WorldModelVisualizer

# Create visualizer
visualizer = WorldModelVisualizer(
    config_path="configs/full_system_config.yaml",
    model_path="runs/model.pth",
    env_type="dmc"  # Only DMC is supported
)

# Run autoregressive rollouts
visualizer.run_comparison(
    num_episodes=3,
    rollout_length=16,
    random_actions=False,
    save_frames=True
)

# Create plots
visualizer.plot_comparisons("output_directory")

# Print statistics
visualizer.print_statistics()
```

## Output

The script generates several visualization files:

### Plots

1. **state_comparison.png**: Actual vs predicted states over time
2. **reward_comparison.png**: Actual vs predicted rewards over time
3. **state_prediction_errors.png**: State prediction errors with statistics
4. **reward_prediction_errors.png**: Reward prediction errors with statistics
5. **action_sequences.png**: Action sequences used in rollouts

### Frames

- `frames/episode_X/actual/`: Rendered frames from the real environment
- `frames/episode_X/predicted/`: Rendered frames from the world model predictions

### Statistics

The script prints detailed statistics including:
- RMSE and MAE for each state dimension
- Reward prediction accuracy
- Correlation between actual and predicted rewards
- Episode statistics

## How Autoregressive Rollouts Work

1. **Initial State**: Both real and predicted environments start from the same initial state, sampled from the world model's valid init buffer
2. **Action Sequence**: A sequence of actions is generated (either random or using a simple policy)
3. **Real Rollout**: The real environment executes the action sequence step-by-step
4. **Predicted Rollout**: The world model performs autoregressive prediction:
   - Takes the initial state and first action
   - Predicts the next state and reward
   - Uses the predicted state as input for the next step
   - Continues for the full rollout length

This allows comparison of how well the world model can predict long sequences of states and rewards using realistic starting conditions.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- OpenCV (for frame rendering)
- PyYAML
- dm_control (for DMC environment)
- The trained world model and its associated scalers

## Notes

- The predicted environment uses a custom renderer to visualize predicted states
- Frames are saved as PNG files in the specified output directory
- The script only supports DMC CartPole environment
- Error statistics are computed across all episodes for comprehensive analysis 
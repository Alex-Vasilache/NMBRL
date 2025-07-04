# SNN Actor-Critic Training and Evaluation

This guide explains how to train and evaluate the Spiking Neural Network (SNN) Actor-Critic agent for the CartPole environment.

## 📁 Files Overview

- **`learning/actor_critic_trainer.py`** - Main training loop for the SNN Actor-Critic agent
- **`visualize_and_evaluate.py`** - Script for loading and evaluating trained models
- **`run_complete_experiment.py`** - Complete workflow script (training + evaluation)

## 🚀 Quick Start

### Option 1: Complete Experiment (Recommended)

Run the complete workflow with default settings:

```bash
python run_complete_experiment.py
```

Run with custom parameters:

```bash
python run_complete_experiment.py --episodes 50 --eval-episodes 10 --hidden-dim 128 --lr 1e-3
```

### Option 2: Training Only

Train a model with the example configuration:

```bash
python learning/actor_critic_trainer.py
```

### Option 3: Evaluation Only

Evaluate a pre-trained model:

```bash
python visualize_and_evaluate.py saved_models/snn_actor_critic_final_20231201_123456
```

## 📊 Training Configuration

### Key Parameters

| Parameter           | Description                   | Default | Recommended Range |
| ------------------- | ----------------------------- | ------- | ----------------- |
| `num_episodes`      | Number of training episodes   | 50      | 20-200            |
| `hidden_dim`        | SNN hidden layer size         | 64      | 32-256            |
| `learning_rate`     | Learning rate for optimizers  | 1e-4    | 1e-5 to 1e-3      |
| `batch_size`        | Training batch size           | 256     | 32-1024           |
| `snn_time_steps`    | SNN simulation time steps     | 1       | 1-10              |
| `buffer_seq_length` | Sequence length for λ-returns | 15      | 5-50              |

### SNN-Specific Parameters

| Parameter         | Description              | Default | Notes                 |
| ----------------- | ------------------------ | ------- | --------------------- |
| `alpha`           | Synaptic decay parameter | 0.9     | 0.5-0.95              |
| `beta`            | Membrane decay parameter | 0.9     | 0.5-0.95              |
| `threshold`       | Spike threshold          | 1.0     | 0.5-2.0               |
| `learn_alpha`     | Learn α parameter        | False   | Set True for adaptive |
| `learn_beta`      | Learn β parameter        | False   | Set True for adaptive |
| `learn_threshold` | Learn threshold          | False   | Set True for adaptive |

## 🎯 Model Saving

Models are automatically saved with timestamps in the following structure:

```
saved_models/
├── snn_actor_critic_final_20231201_123456/
│   ├── actor.pth          # Actor network weights + config
│   ├── critic.pth         # Critic network weights + config  
│   └── training_info.pth  # Training statistics + config
```

### Periodic Saving

Configure periodic saving during training:

```python
training_config = {
    "save_frequency": 10,  # Save every 10 episodes
    "save_dir": "my_models",  # Custom save directory
    # ... other parameters
}
```

## 📈 Evaluation and Visualization

### Basic Evaluation

```bash
# Evaluate with visualization (default)
python visualize_and_evaluate.py saved_models/model_directory

# Evaluate without visualization
python visualize_and_evaluate.py saved_models/model_directory --no-visualize

# Custom number of episodes
python visualize_and_evaluate.py saved_models/model_directory --episodes 20
```

### Advanced Options

```bash
# Full evaluation with all plots and saved outputs
python visualize_and_evaluate.py saved_models/model_directory \
    --episodes 20 \
    --plot-training \
    --save-plots evaluation_plots \
    --render-delay 0.05
```

### Available Visualizations

1. **Training Progress**
   - Episode rewards over time
   - Episode lengths over time
   - Moving averages

2. **Evaluation Results**
   - Episode rewards distribution
   - Episode lengths distribution  
   - Reward histogram
   - Value function estimates

## 🛠 Customization Examples

### Fast Training (for testing)

```bash
python run_complete_experiment.py \
    --episodes 10 \
    --eval-episodes 3 \
    --batch-size 64 \
    --no-train-viz
```

### Deep Network Training

```bash
python run_complete_experiment.py \
    --episodes 100 \
    --hidden-dim 256 \
    --lr 5e-5 \
    --batch-size 512
```

### SNN Parameter Exploration

Modify the training configuration to explore SNN parameters:

```python
training_config = {
    "snn_time_steps": 5,      # Multi-step SNN simulation
    "learn_alpha": True,      # Adaptive synaptic decay
    "learn_beta": True,       # Adaptive membrane decay
    "learn_threshold": True,  # Adaptive spike threshold
    "alpha": 0.8,            # Initial α value
    "beta": 0.7,             # Initial β value
    "threshold": 0.8,        # Initial threshold
    # ... other parameters
}
```

## 📊 Performance Monitoring

### Training Metrics

During training, monitor these key metrics:

- **Episode Reward**: Target > 800 for good performance
- **Episode Length**: Target > 900 steps for stability
- **Actor Loss**: Should decrease over time
- **Critic Loss**: Should stabilize at low values
- **Mean Value**: Should reflect expected returns

### Evaluation Metrics

- **Mean Reward**: Average performance across episodes
- **Success Rate**: Percentage of successful episodes
- **Consistency**: Low standard deviation in rewards

## 🔧 Troubleshooting

### Common Issues

1. **Training Instability**
   - Reduce learning rate (try 1e-5)
   - Increase batch size
   - Reduce buffer sequence length

2. **Poor Performance**
   - Increase hidden dimension
   - Adjust SNN parameters (α, β)
   - Try different weight initialization

3. **Memory Issues**
   - Reduce batch size
   - Reduce buffer sequence length
   - Use shorter episodes

### Debug Mode

Add debug prints to track training:

```python
training_config["update_frequency"] = 1  # More frequent updates
# Monitor losses more closely during training
```

## 📋 Example Workflows

### Baseline Experiment

```bash
# Quick baseline with visualization
python run_complete_experiment.py --episodes 30 --eval-episodes 5

# Production run without visualization
python run_complete_experiment.py \
    --episodes 100 \
    --eval-episodes 20 \
    --no-train-viz \
    --save-dir experiments/baseline
```

### Hyperparameter Sweep

```bash
# Test different network sizes
for hidden_dim in 32 64 128 256; do
    python run_complete_experiment.py \
        --episodes 50 \
        --hidden-dim $hidden_dim \
        --save-dir experiments/hidden_$hidden_dim \
        --no-train-viz
done
```

### Model Comparison

```bash
# Evaluate multiple models
python visualize_and_evaluate.py experiments/hidden_64 --episodes 10 --save-plots plots/hidden_64
python visualize_and_evaluate.py experiments/hidden_128 --episodes 10 --save-plots plots/hidden_128
```

## 🎯 Performance Targets

| Metric            | Beginner | Good  | Excellent |
| ----------------- | -------- | ----- | --------- |
| Episode Reward    | > 200    | > 600 | > 900     |
| Episode Length    | > 200    | > 600 | > 950     |
| Training Episodes | < 100    | < 50  | < 30      |
| Stability         | ±100     | ±50   | ±20       |

## 📝 Notes

- The SNN implementation uses SNNTorch library
- CartPole environment provides 6-dimensional continuous state space
- Actor outputs continuous actions (1D force)
- Critic estimates state values for training
- λ-returns are used for more stable value estimates 
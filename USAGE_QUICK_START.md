# SNN Actor-Critic Quick Start Guide

## 🚀 Quick Commands

### 1. **IMPROVED Training** (Recommended)
```bash
# Stable configuration (good starting point)
python improved_training_config.py stable

# Debug configuration (quick test with visualization)
python improved_training_config.py debug

# Performance configuration (longer training, best results)
python improved_training_config.py performance
```

### 2. Run Complete Experiment (Training + Evaluation)
```bash
# Default settings (20 episodes training, 5 evaluation)
python run_complete_experiment.py

# Custom settings  
python run_complete_experiment.py --episodes 50 --eval-episodes 10 --hidden-dim 128
```

### 3. Train Only (Original)
```bash
python learning/actor_critic_trainer.py
```

### 4. Evaluate Existing Model
```bash
python visualize_and_evaluate.py saved_models/your_model_directory
```

## 📊 What You'll Get

### During Training:
- Real-time episode progress
- Loss metrics every few steps
- Periodic model saving
- Final model saved automatically

### During Evaluation:
- Visual environment simulation
- Performance statistics
- Training progress plots
- Evaluation results plots

## 🎯 Key Files Created

```
saved_models/
└── snn_actor_critic_final_TIMESTAMP/
    ├── actor.pth          # Trained actor network
    ├── critic.pth         # Trained critic network  
    └── training_info.pth  # Training stats & config
```

## ⚡ Common Use Cases

```bash
# Quick test (RECOMMENDED - improved training)
python improved_training_config.py debug

# Production training (stable, reliable)
python improved_training_config.py stable

# Best performance (longer training)
python improved_training_config.py performance

# Original workflow (if needed)
python run_complete_experiment.py --episodes 10 --no-train-viz

# Evaluate with detailed plots
python visualize_and_evaluate.py saved_models/model_dir --plot-training --save-plots results
```

## 🔧 Troubleshooting

### ✅ **FIXED: PyTorch Loading Error**
The "weights_only" error has been resolved! All model loading now works correctly.

### Common Issues:
- **Training too slow?** Add `--no-train-viz` or use improved configs
- **Poor performance/negative rewards?** Use `python improved_training_config.py debug`
- **Memory issues?** Reduce `--batch-size`
- **PyTorch loading errors?** These are now fixed automatically

### Quick Diagnostics:
```bash
# Test if saved models load correctly
python test_model_loading.py

# Get detailed troubleshooting help  
python improved_training_config.py --diagnose

# Quick debug training (20 episodes with visualization)
python improved_training_config.py debug
```

See `README_SNN_Training.md` for detailed documentation. 
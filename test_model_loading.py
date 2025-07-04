#!/usr/bin/env python3
"""
Quick test script to verify that saved models can be loaded correctly.
This script tests the model loading functionality without running full evaluation.
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def test_model_loading(model_path):
    """
    Test if a saved model can be loaded successfully.

    :param model_path: Path to the saved model directory
    :return: True if loading successful, False otherwise
    """
    print(f"Testing model loading from: {model_path}")

    try:
        # Test loading training info
        training_info_path = os.path.join(model_path, "training_info.pth")
        if not os.path.exists(training_info_path):
            print(f"❌ Training info file not found: {training_info_path}")
            return False

        print("📋 Loading training info...")
        training_info = torch.load(training_info_path, weights_only=False)
        config = training_info["config"]
        print(f"✅ Training info loaded successfully")
        print(
            f"   - Episodes trained: {training_info['final_stats']['total_episodes']}"
        )
        print(
            f"   - Best reward: {training_info['final_stats']['best_episode_reward']:.2f}"
        )
        print(f"   - Hidden dim: {config.get('hidden_dim', 'N/A')}")

        # Test loading actor model
        actor_path = os.path.join(model_path, "actor.pth")
        if not os.path.exists(actor_path):
            print(f"❌ Actor model file not found: {actor_path}")
            return False

        print("🎭 Loading actor model...")
        actor_checkpoint = torch.load(actor_path, weights_only=False)
        print(f"✅ Actor model loaded successfully")
        print(f"   - State dict keys: {len(actor_checkpoint['model_state_dict'])}")

        # Test loading critic model
        critic_path = os.path.join(model_path, "critic.pth")
        if not os.path.exists(critic_path):
            print(f"❌ Critic model file not found: {critic_path}")
            return False

        print("🧠 Loading critic model...")
        critic_checkpoint = torch.load(critic_path, weights_only=False)
        print(f"✅ Critic model loaded successfully")
        print(f"   - State dict keys: {len(critic_checkpoint['model_state_dict'])}")

        print("\n🎉 All model files loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def find_latest_model():
    """Find the most recently saved model in the saved_models directory."""
    saved_models_dir = "saved_models"
    if not os.path.exists(saved_models_dir):
        return None

    model_dirs = [
        d
        for d in os.listdir(saved_models_dir)
        if os.path.isdir(os.path.join(saved_models_dir, d))
    ]

    if not model_dirs:
        return None

    # Sort by modification time, most recent first
    model_dirs.sort(
        key=lambda x: os.path.getmtime(os.path.join(saved_models_dir, x)), reverse=True
    )

    return os.path.join(saved_models_dir, model_dirs[0])


def main():
    print("🧪 Model Loading Test Script")
    print("=" * 40)

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Try to find the latest model
        model_path = find_latest_model()
        if model_path is None:
            print("❌ No saved models found and no model path provided.")
            print("Usage: python test_model_loading.py [model_path]")
            print("   or train a model first using run_complete_experiment.py")
            return
        else:
            print(f"📁 Using latest model: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return

    success = test_model_loading(model_path)

    if success:
        print("\n✅ Model loading test PASSED!")
        print("🚀 You can now run evaluation with:")
        print(f'   python visualize_and_evaluate.py "{model_path}"')
    else:
        print("\n❌ Model loading test FAILED!")
        print("🔧 Check the model files and try again.")


if __name__ == "__main__":
    main()

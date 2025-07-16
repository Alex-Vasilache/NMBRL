#!/usr/bin/env python3
import torch
from torch import distributions as torchd
from utils.tools import TanhBijector


def test_tanh_bijector():
    print("Testing TanhBijector...")

    # Create a normal distribution
    mean = torch.tensor([0.0])
    std = torch.tensor([1.0])
    base_dist = torchd.normal.Normal(mean, std)

    # Create the transformed distribution
    try:
        transformed_dist = torchd.transformed_distribution.TransformedDistribution(
            base_dist, TanhBijector()
        )
        print("✓ TanhBijector works correctly!")

        # Test sampling
        sample = transformed_dist.sample()
        print(f"✓ Sample shape: {sample.shape}")
        print(f"✓ Sample value: {sample.item():.4f}")

        # Test log_prob
        log_prob = transformed_dist.log_prob(sample)
        print(f"✓ Log probability: {log_prob.item():.4f}")

        return True
    except Exception as e:
        import traceback

        print(f"✗ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_tanh_bijector()
    if success:
        print("\nTanhBijector test passed!")
    else:
        print("\nTanhBijector test failed!")

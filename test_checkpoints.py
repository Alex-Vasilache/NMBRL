#!/usr/bin/env python3
import os
import argparse
import glob
import yaml
import time
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import joblib

from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from utils.tools import seed_everything


def find_checkpoints(run_dir):
    """Find all agent checkpoints in the run directory."""
    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    if not os.path.exists(checkpoints_dir):
        print(
            f"[TESTER] Error: Checkpoints directory '{checkpoints_dir}' does not exist."
        )
        return []

    # Find all .zip files
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.zip"))

    if not checkpoint_files:
        print(f"[TESTER] No checkpoint files found in '{checkpoints_dir}'.")
        return []

    # Sort by filename to get chronological order
    checkpoint_files.sort()

    print(f"[TESTER] Found {len(checkpoint_files)} checkpoints:")
    for cp in checkpoint_files:
        print(f"  - {os.path.basename(cp)}")

    return checkpoint_files


def load_agent(checkpoint_path, agent_type="DREAMER", env=None, force_cpu=False):
    """Load an agent from a checkpoint file."""
    print(
        f"[TESTER] Loading {agent_type} agent from: {os.path.basename(checkpoint_path)}"
    )

    if agent_type == "DREAMER":
        from agents.dreamer_ac_agent import DreamerACAgent

        # Patch DreamerACAgent.load to support force-cpu
        import types
        from agents import dreamer_ac_agent as dreamer_mod

        orig_load = dreamer_mod.DreamerACAgent.load

        def patched_load(path, env=None, force_cpu=False):
            import torch
            import zipfile, os, tempfile, pickle
            from datetime import datetime

            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(path, "r") as zipf:
                    zipf.extractall(temp_dir)
                training_info_path = os.path.join(temp_dir, "training_info.pkl")
                with open(training_info_path, "rb") as f:
                    training_info = pickle.load(f)
                config = training_info["config"]
                global_config = training_info["global_config"]
                if force_cpu:
                    config["device"] = "cpu"
                    global_config["device"] = "cpu"
                temp_log_dir = os.path.join(
                    tempfile.gettempdir(),
                    f"dreamer_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                agent = dreamer_mod.DreamerACAgent(
                    global_config, env, tensorboard_log=temp_log_dir
                )
                # Always use CPU if forced, else use config["device"]
                map_location = "cpu" if force_cpu else config["device"]
                actor_path = os.path.join(temp_dir, "actor.pth")
                actor_data = torch.load(
                    actor_path, map_location=map_location, weights_only=False
                )
                agent.agent.actor.load_state_dict(actor_data["model_state_dict"])
                critic_path = os.path.join(temp_dir, "critic.pth")
                critic_data = torch.load(
                    critic_path, map_location=map_location, weights_only=False
                )
                agent.agent.critic.load_state_dict(critic_data["model_state_dict"])
                agent.episode_rewards = training_info.get("episode_rewards", [])
                agent.episode_lengths = training_info.get("episode_lengths", [])
                agent.training_losses = training_info.get("training_losses", [])
                agent.agent.actor.eval()
                agent.agent.critic.eval()
                return agent

        dreamer_mod.DreamerACAgent.load = staticmethod(
            lambda path, env=None: patched_load(path, env, force_cpu=force_cpu)
        )

        return DreamerACAgent.load(checkpoint_path, env=env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def evaluate_agent_single_seed(
    agent,
    env,
    seed,
    n_episodes=10,
    state_scaler=None,
    use_output_state_scaler=False,
):
    """Evaluate an agent for multiple episodes on a single seed."""
    seed_everything(seed)
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state = env.reset()

        # Randomize initial state for DMC CartPole
        try:
            base_env = env
            # Unwrap VecNormalize, DummyVecEnv, etc.
            if hasattr(base_env, "venv"):
                base_env = base_env.venv
            if (
                hasattr(base_env, "envs")
                and isinstance(base_env.envs, list)
                and len(base_env.envs) > 0
            ):
                base_env = base_env.envs[0]
            if hasattr(base_env, "env"):
                base_env = base_env.env
            if hasattr(base_env, "physics"):
                qpos = np.array(base_env.physics.data.qpos)
                qvel = np.array(base_env.physics.data.qvel)
                # Randomize cart position and pole angle
                qpos[0] = np.random.uniform(-1.0, 1.0)  # cart position
                qpos[1] = np.random.uniform(-0.5, 0.5)  # pole angle (radians)
                # Randomize velocities
                qvel[:] = np.random.uniform(-0.2, 0.2, size=qvel.shape)
                base_env.physics.set_state(np.concatenate([qpos, qvel]))
                # Get the new observation after setting the state
                if hasattr(base_env, "_get_obs"):
                    state = base_env._get_obs()
                elif hasattr(base_env, "get_observation"):
                    state = base_env.get_observation()
                else:
                    # As fallback, step with zero action to get new obs
                    zero_action = np.zeros(base_env.action_space.shape)
                    state = base_env.step(zero_action)[0]
        except Exception as e:
            print(f"[TESTER] Could not randomize initial state: {e}")

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Handle different observation formats
            if isinstance(state, tuple):
                obs = state[0] if hasattr(state[0], "shape") else state
            else:
                obs = state

            # Apply state scaler if available and not using output state scaler
            scaled_obs = obs
            if state_scaler is not None and not use_output_state_scaler:
                try:
                    if isinstance(scaled_obs, np.ndarray) and scaled_obs.ndim == 1:
                        scaled_obs = scaled_obs.reshape(1, -1)
                    scaled_obs = state_scaler.transform(scaled_obs)
                except Exception as e:
                    print(f"[TESTER] Warning: State scaling failed: {e}")
                    scaled_obs = obs

            # Get action from agent
            if hasattr(agent, "predict"):
                # Stable-Baselines3 format
                action, _ = agent.predict(scaled_obs, deterministic=True)
            else:
                # Custom agent format (like DREAMER)
                action = agent.get_action(scaled_obs, deterministic=True)

            try:
                step_result = base_env.step(action)
            except Exception as e:
                print(f"[TESTER] Exception during step: {e}")
                break

            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
            else:
                next_state, reward, terminated, info = step_result
                truncated = False
            done = terminated or truncated

            # Handle different reward formats
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0

            episode_reward += reward
            episode_length += 1
            state = next_state

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return episode_rewards, episode_lengths


def test_checkpoint(
    checkpoint_path,
    run_dir,
    seeds=[42, 123, 456],
    n_episodes=10,
    state_scaler=None,
    use_output_state_scaler=False,
    force_cpu=False,
):
    """Test a single checkpoint over multiple seeds."""
    checkpoint_name = os.path.basename(checkpoint_path)
    print(f"\n[TESTER] Testing checkpoint: {checkpoint_name}")

    all_rewards = []
    all_lengths = []

    for seed in seeds:
        print(f"[TESTER] Testing with seed {seed}...")

        # Create environment for this seed
        env = wrapper(
            seed=seed,
            n_envs=1,
            render_mode=None,  # No rendering for testing
            max_episode_steps=1000,
        )

        try:
            # Load agent
            agent = load_agent(checkpoint_path, "DREAMER", env=env, force_cpu=force_cpu)

            # Evaluate agent
            episode_rewards, episode_lengths = evaluate_agent_single_seed(
                agent,
                env,
                seed,
                n_episodes=n_episodes,
                state_scaler=state_scaler,
                use_output_state_scaler=use_output_state_scaler,
            )

            all_rewards.extend(episode_rewards)
            all_lengths.extend(episode_lengths)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)

            print(
                f"[TESTER] Seed {seed}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}, Mean length = {mean_length:.1f}"
            )

        except Exception as e:
            print(
                f"[TESTER] Error testing checkpoint {checkpoint_name} with seed {seed}: {e}"
            )
            import traceback

            traceback.print_exc()
        finally:
            env.close()

    if all_rewards:
        overall_mean = np.mean(all_rewards)
        overall_std = np.std(all_rewards)
        overall_mean_length = np.mean(all_lengths)

        print(f"[TESTER] Overall for {checkpoint_name}:")
        print(f"  Mean reward: {overall_mean:.2f} ± {overall_std:.2f}")
        print(f"  Mean episode length: {overall_mean_length:.1f}")

        return {
            "checkpoint": checkpoint_name,
            "mean_reward": overall_mean,
            "std_reward": overall_std,
            "mean_length": overall_mean_length,
            "all_rewards": all_rewards,
            "all_lengths": all_lengths,
        }
    else:
        print(f"[TESTER] No valid results for {checkpoint_name}")
        return None


def load_run_config(run_dir):
    """Load the configuration used for this run."""
    # Look for saved config files
    config_patterns = [
        os.path.join(run_dir, "*_config.yaml"),
        os.path.join(run_dir, "training_config.yaml"),
        os.path.join(run_dir, "original_config.yaml"),
    ]

    for pattern in config_patterns:
        config_files = glob.glob(pattern)
        if config_files:
            config_path = config_files[0]
            print(f"[TESTER] Loading config from: {config_path}")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

    # If no config found, return default values
    print("[TESTER] No config file found, using defaults")
    return {
        "global": {"seed": 42},
        "data_generator": {"max_episode_steps": 1000},
        "tensorboard": {"log_dir": "tb_logs", "flush_seconds": 30},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test all agent checkpoints in a run directory over multiple seeds."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="20250717_124546",
        help="Path to run directory (default: 20250717_124546)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds to test (default: 42 123 456)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes per seed (default: 10)",
    )
    parser.add_argument(
        "--use-state-scaler",
        action="store_true",
        default=True,
        help="Use state scaler from run directory if available (default: True)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=False,
        help="Force model loading and evaluation on CPU (default: False)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoint_results.csv",
        help="Output CSV file for results (default: checkpoint_results.csv)",
    )
    args = parser.parse_args()

    # Check if run directory exists
    if not os.path.exists(args.run_dir):
        print(f"[TESTER] Error: Run directory '{args.run_dir}' does not exist.")
        return

    print(f"[TESTER] Testing checkpoints in: {args.run_dir}")
    print(f"[TESTER] Seeds: {args.seeds}")
    print(f"[TESTER] Episodes per seed: {args.n_episodes}")

    # Load configuration
    config = load_run_config(args.run_dir)

    # Load state scaler if requested and available
    state_scaler = None
    use_scaler = True  # Always use state scaler if available
    scaler_path = os.path.join(args.run_dir, "state_scaler.joblib")
    if use_scaler and os.path.exists(scaler_path):
        try:
            state_scaler = joblib.load(scaler_path)
            print(f"[TESTER] Loaded state scaler from: {scaler_path}")
        except Exception as e:
            print(f"[TESTER] Warning: Failed to load state scaler: {e}")
            state_scaler = None
    elif use_scaler:
        print(
            f"[TESTER] Warning: State scaler requested but not found at {scaler_path}"
        )

    # Find all checkpoints
    checkpoint_files = find_checkpoints(args.run_dir)
    if not checkpoint_files:
        return

    # Test each checkpoint
    results = []
    use_output_state_scaler = config.get("world_model_trainer", {}).get(
        "use_output_state_scaler", False
    )

    for i, checkpoint_path in enumerate(checkpoint_files):
        print(f"\n[TESTER] Progress: {i+1}/{len(checkpoint_files)}")

        result = test_checkpoint(
            checkpoint_path,
            args.run_dir,
            seeds=args.seeds,
            n_episodes=args.n_episodes,
            state_scaler=state_scaler,
            use_output_state_scaler=use_output_state_scaler,
            force_cpu=args.force_cpu,
        )

        if result:
            results.append(result)

    # Sort results by mean reward (descending)
    results.sort(key=lambda x: x["mean_reward"], reverse=True)

    # Print summary
    print("\n" + "=" * 80)
    print("CHECKPOINT TESTING RESULTS")
    print("=" * 80)

    if results:
        print(f"\nBest performing checkpoint: {results[0]['checkpoint']}")
        print(
            f"Mean reward: {results[0]['mean_reward']:.2f} ± {results[0]['std_reward']:.2f}"
        )
        print(f"Mean episode length: {results[0]['mean_length']:.1f}")

        print(f"\nAll results (sorted by mean reward):")
        print(
            f"{'Checkpoint':<40} {'Mean Reward':<15} {'Std Reward':<15} {'Mean Length':<15}"
        )
        print("-" * 85)

        for result in results:
            print(
                f"{result['checkpoint']:<40} {result['mean_reward']:<15.2f} {result['std_reward']:<15.2f} {result['mean_length']:<15.1f}"
            )

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n[TESTER] Results saved to: {args.output}")

    else:
        print("[TESTER] No valid results obtained.")

    print("=" * 80)


if __name__ == "__main__":
    main()

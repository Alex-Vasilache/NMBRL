#!/usr/bin/env python3
import os
import argparse
import glob
import yaml
import time
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from utils.tools import seed_everything


def find_latest_run_directory(base_dir):
    """Find the most recently created run directory."""
    if not os.path.exists(base_dir):
        return None

    run_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not run_dirs:
        return None

    return max(run_dirs, key=os.path.getctime)


def find_latest_agent_checkpoint(run_dir):
    """Find the latest agent checkpoint (.zip file) in the run directory."""
    # Look in actor_logs/checkpoints/ subdirectory
    checkpoints_dir = os.path.join(run_dir, "actor_logs", "checkpoints")

    if not os.path.exists(checkpoints_dir):
        return None, None

    # Find all .zip files
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.zip"))

    if not checkpoint_files:
        return None, None

    # Get the latest checkpoint by creation time
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)

    # Determine agent type from filename
    filename = os.path.basename(latest_checkpoint)
    if filename.startswith("dreamer_ac_agent"):
        agent_type = "DREAMER"
    elif filename.startswith("ppo_cp"):
        agent_type = "PPO"
    elif filename.startswith("sac_cp"):
        agent_type = "SAC"
    else:
        # Try to infer from file structure or default to DREAMER
        agent_type = "DREAMER"
        print(
            f"[VISUALIZER] Warning: Could not determine agent type from filename '{filename}', assuming DREAMER"
        )

    return latest_checkpoint, agent_type


def load_agent(checkpoint_path, agent_type, env=None):
    """Load an agent from a checkpoint file."""
    print(f"[VISUALIZER] Loading {agent_type} agent from: {checkpoint_path}")

    if agent_type == "DREAMER":
        from agents.dreamer_ac_agent import DreamerACAgent

        return DreamerACAgent.load(checkpoint_path, env=env)
    elif agent_type == "PPO":
        from stable_baselines3 import PPO

        return PPO.load(checkpoint_path)
    elif agent_type == "SAC":
        from stable_baselines3 import SAC

        return SAC.load(checkpoint_path)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


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
            print(f"[VISUALIZER] Loading config from: {config_path}")
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

    # If no config found, return default values
    print("[VISUALIZER] No config file found, using defaults")
    return {
        "global": {"seed": 42},
        "data_generator": {"max_episode_steps": 1000},
        "tensorboard": {"log_dir": "tb_logs", "flush_seconds": 30},
    }


def evaluate_agent(agent, env, n_episodes=10, render=True, verbose=True):
    """Evaluate an agent for multiple episodes and return statistics."""
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False

        start_time = time.time()

        while not terminated:
            # Handle different observation formats
            if isinstance(state, tuple):
                obs = state[0] if hasattr(state[0], "shape") else state
            else:
                obs = state

            # Get action from agent
            if hasattr(agent, "predict"):
                # Stable-Baselines3 format
                action, _ = agent.predict(obs, deterministic=True)
            else:
                # Custom agent format (like DREAMER)
                action = agent.get_action(obs, deterministic=True)

            # Step environment
            next_state, reward, terminated, info = env.step(action)

            if render:
                env.render()
                # Small delay to make visualization visible
                time.sleep(0.02)

            # Handle different reward formats
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0

            episode_reward += reward
            episode_length += 1
            state = next_state

            # Check for user closing render window
            if render and isinstance(info, (list, tuple)) and len(info) > 0:
                info_dict = info[0] if isinstance(info[0], dict) else {}
                if info_dict.get("sim_should_stop", False):
                    print(
                        "[VISUALIZER] Render window closed by user, stopping evaluation."
                    )
                    return episode_rewards, episode_lengths

        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if verbose:
            print(
                f"[VISUALIZER] Episode {episode + 1}/{n_episodes}: "
                f"Reward = {episode_reward:.2f}, "
                f"Length = {episode_length}, "
                f"Time = {episode_time:.1f}s"
            )

    return episode_rewards, episode_lengths


def print_statistics(episode_rewards, episode_lengths):
    """Print evaluation statistics."""
    if not episode_rewards:
        print("[VISUALIZER] No episodes completed.")
        return

    print("\n" + "=" * 50)
    print("EVALUATION STATISTICS")
    print("=" * 50)
    print(f"Episodes completed: {len(episode_rewards)}")
    print(
        f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    print(
        f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(f"Max episode length: {np.max(episode_lengths)}")
    print(f"Min episode length: {np.min(episode_lengths)}")
    print("=" * 50)


def log_to_tensorboard(
    writer, episode_rewards, episode_lengths, agent_type, checkpoint_path
):
    """Log evaluation results to TensorBoard."""
    if not episode_rewards:
        return

    # Log summary statistics
    writer.add_scalar("Evaluation/Mean_Reward", np.mean(episode_rewards), 0)
    writer.add_scalar("Evaluation/Std_Reward", np.std(episode_rewards), 0)
    writer.add_scalar("Evaluation/Max_Reward", np.max(episode_rewards), 0)
    writer.add_scalar("Evaluation/Min_Reward", np.min(episode_rewards), 0)
    writer.add_scalar("Evaluation/Mean_Episode_Length", np.mean(episode_lengths), 0)

    # Log individual episode results
    for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
        writer.add_scalar("Evaluation/Episode_Reward", reward, i)
        writer.add_scalar("Evaluation/Episode_Length", length, i)

    # Log metadata
    writer.add_text("Evaluation/Agent_Type", agent_type, 0)
    writer.add_text("Evaluation/Checkpoint_Path", checkpoint_path, 0)
    writer.add_text(
        "Evaluation/Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the latest trained agent from a run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to specific run directory. If not provided, finds the latest in completed_runs/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint file to load instead of finding latest",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["DREAMER", "PPO", "SAC"],
        help="Agent type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (run headless)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Determine run directory
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.exists(run_dir):
            print(f"[VISUALIZER] Error: Run directory '{run_dir}' does not exist.")
            return
    else:
        # Find latest run in completed_runs
        run_dir = find_latest_run_directory("completed_runs")
        if not run_dir:
            print("[VISUALIZER] Error: No run directories found in 'completed_runs/'.")
            print("[VISUALIZER] Use --run-dir to specify a specific directory.")
            return

    print(f"[VISUALIZER] Using run directory: {run_dir}")

    # Load configuration
    config = load_run_config(run_dir)
    seed_everything(args.seed)

    # Determine checkpoint and agent type
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        agent_type = args.agent_type
        if not agent_type:
            # Try to infer from filename
            filename = os.path.basename(checkpoint_path)
            if "dreamer" in filename.lower():
                agent_type = "DREAMER"
            elif "ppo" in filename.lower():
                agent_type = "PPO"
            elif "sac" in filename.lower():
                agent_type = "SAC"
            else:
                print(
                    "[VISUALIZER] Error: Could not determine agent type. Use --agent-type."
                )
                return
    else:
        checkpoint_path, agent_type = find_latest_agent_checkpoint(run_dir)
        if not checkpoint_path:
            print(f"[VISUALIZER] Error: No agent checkpoints found in '{run_dir}'.")
            return

    if not os.path.exists(checkpoint_path):
        print(
            f"[VISUALIZER] Error: Checkpoint file '{checkpoint_path}' does not exist."
        )
        return

    print(f"[VISUALIZER] Loading checkpoint: {checkpoint_path}")
    print(f"[VISUALIZER] Agent type: {agent_type}")

    # Setup TensorBoard logging
    tb_config = config.get("tensorboard", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join(
        run_dir, tb_config.get("log_dir", "tb_logs"), f"evaluation_{timestamp}"
    )
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(
        log_dir=tb_log_dir, flush_secs=tb_config.get("flush_seconds", 30)
    )

    # Create environment
    render_mode = None if args.no_render else "human"
    env = wrapper(
        seed=args.seed,
        n_envs=1,
        render_mode=render_mode,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        # Load agent
        agent = load_agent(checkpoint_path, agent_type, env=env)
        print(f"[VISUALIZER] Successfully loaded {agent_type} agent.")

        # Evaluate agent
        print(f"[VISUALIZER] Starting evaluation for {args.n_episodes} episodes...")
        episode_rewards, episode_lengths = evaluate_agent(
            agent,
            env,
            n_episodes=args.n_episodes,
            render=not args.no_render,
            verbose=True,
        )

        # Print and log results
        print_statistics(episode_rewards, episode_lengths)
        log_to_tensorboard(
            writer, episode_rewards, episode_lengths, agent_type, checkpoint_path
        )

        print(f"[VISUALIZER] TensorBoard logs saved to: {tb_log_dir}")

    except KeyboardInterrupt:
        print("\n[VISUALIZER] Evaluation interrupted by user.")
    except Exception as e:
        print(f"[VISUALIZER] Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        writer.close()
        print("[VISUALIZER] Evaluation completed.")


if __name__ == "__main__":
    main()

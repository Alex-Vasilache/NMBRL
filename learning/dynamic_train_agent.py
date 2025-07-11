#!/usr/bin/env python3
import os
import argparse
from stable_baselines3.common.monitor import Monitor
import yaml

from agents.actor_wrapper import ActorWrapper
from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from world_models.dynamic_world_model_wrapper import WorldModelWrapper
from utils.tools import seed_everything

from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train an agent using a dynamic world model."
    )
    parser.add_argument(
        "--shared-folder",
        type=str,
        required=True,
        help="Path to the folder where the trained world model is stored and updated.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # --- Load configuration from YAML file ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    global_config = config["global"]
    agent_config = config["agent_trainer"]
    agent_type = agent_config.get("agent_type", "PPO").upper()

    # seed everything
    seed_everything(global_config["seed"])

    # The --shared-folder is the shared space for all components
    shared_folder = args.shared_folder
    LOG_DIR = os.path.join(shared_folder, "actor_logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create a temporary real environment just to get the observation and action spaces
    # This env is then closed and not used for training.
    print("--- Creating environment to extract space info ---")
    temp_real_env = wrapper(seed=global_config["seed"], n_envs=1)
    obs_space = temp_real_env.observation_space
    act_space = temp_real_env.action_space
    temp_real_env.close()
    print(f"Obs space: {obs_space}, Action space: {act_space}")

    print(
        f"--- Initializing World Model environment wrapper (monitoring: {shared_folder}) ---"
    )
    train_env = WorldModelWrapper(
        observation_space=obs_space,
        action_space=act_space,
        batch_size=agent_config["n_envs"],
        shared_folder=shared_folder,
        config=config,
    )

    agent = ActorWrapper(
        env=train_env,
        config=config,
        training=True,
        shared_folder=shared_folder,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=agent_config["checkpoint_freq"],
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix=f"{agent_type.lower()}_cp",
    )

    callbacks = CallbackList([checkpoint_callback])

    agent.learn(
        total_timesteps=agent_config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    # The callbacks handle saving the best model and checkpoints, so a final save is not needed.

    train_env.close()
    eval_env.close()
    print("\n[AGENT-TRAINER] Training finished and environments closed.")


if __name__ == "__main__":
    main()

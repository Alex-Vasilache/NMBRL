#!/usr/bin/env python3
import os
import argparse
import yaml  # Import the YAML library

from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from world_models.dynamic_world_model_wrapper import WorldModelWrapper


from stable_baselines3 import SAC, PPO
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
        "--world-model-folder",
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

    # The --world-model-folder is the shared space for all components
    shared_folder = args.world_model_folder
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
        f"--- Initializing World Model environment wrapper (monitoring: {args.world_model_folder}) ---"
    )
    train_env = WorldModelWrapper(
        observation_space=obs_space,
        action_space=act_space,
        batch_size=agent_config["n_envs"],
        trained_folder=args.world_model_folder,
        config=config,
    )

    if agent_type == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=2,
            seed=global_config["seed"],
            tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        )
    elif agent_type == "SAC":
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=2,
            seed=global_config["seed"],
            tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    eval_env = wrapper(
        seed=global_config["seed"],
        n_envs=1,
        max_episode_steps=agent_config["max_episode_steps"],
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=agent_config["eval_freq"],
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=agent_config["checkpoint_freq"],
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix=f"{agent_type.lower()}_cp",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    model.learn(
        total_timesteps=agent_config["total_timesteps"],
        callback=callbacks,
        progress_bar=False,
    )

    # The callbacks handle saving the best model and checkpoints, so a final save is not needed.

    train_env.close()
    eval_env.close()
    print("\n[AGENT-TRAINER] Training finished and environments closed.")


if __name__ == "__main__":
    main()

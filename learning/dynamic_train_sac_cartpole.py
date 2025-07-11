#!/usr/bin/env python3
import os
import random
from datetime import datetime
import argparse
import yaml  # Import the YAML library

from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from world_models.dynamic_world_model_wrapper import WorldModelWrapper

import numpy as np
import torch

from typing import Callable
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train an SAC agent using a dynamic world model."
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
    sac_config = config["sac_trainer"]

    # ─── Directory & Path Setup ──────────────────────────────────────────────────
    # The --world-model-folder is the shared space for all components
    shared_folder = args.world_model_folder
    LOG_DIR = os.path.join(shared_folder, "actor_logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Agent Setup --------------------------------------------------------------
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
        batch_size=sac_config["n_envs"],
        trained_folder=args.world_model_folder,
        config=config,
    )

    # ─── 4) MODEL SETUP ───────────────────────────────────────────────────────────
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """Linear decay from initial_value to zero over training."""

        def lr_fn(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return lr_fn

    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(
            net_arch=sac_config["net_arch"],
            log_std_init=-3,
        ),
        buffer_size=sac_config["buffer_size"],
        batch_size=sac_config["batch_size"],
        learning_starts=sac_config["learning_starts"],
        train_freq=(sac_config["train_freq_steps"], "step"),
        gradient_steps=sac_config["gradient_steps"],
        gamma=sac_config["gamma"],
        tau=sac_config["tau"],
        ent_coef=sac_config["ent_coef"],
        target_update_interval=sac_config["target_update_interval"],
        use_sde=sac_config["use_sde"],
        sde_sample_freq=-1,  # Not used when SDE is false
        learning_rate=linear_schedule(sac_config["initial_lr"]),
        verbose=2,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        seed=global_config["seed"],
        device="auto",
    )

    # ─── 5) CALLBACKS & EVAL ENV ───────────────────────────────────────────────────
    # Create evaluation env with identical normalization (without loading any prior stats)
    eval_env = wrapper(
        seed=global_config["seed"],
        n_envs=1,
        max_episode_steps=sac_config["max_episode_steps"],
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=sac_config["eval_freq"],
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=sac_config["checkpoint_freq"],
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="sac_cp",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ─── 6) TRAINING & SAVING ────────────────────────────────────────────────────
    model.learn(
        total_timesteps=sac_config["total_timesteps"],
        callback=callbacks,
        progress_bar=False,
    )

    # The callbacks handle saving the best model and checkpoints, so a final save is not needed.

    # ─── 7) CLEANUP ─────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()
    print("\n[SAC-TRAINER] Training finished and environments closed.")


if __name__ == "__main__":
    main()

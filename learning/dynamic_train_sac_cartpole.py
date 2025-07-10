#!/usr/bin/env python3
import os
import random
from datetime import datetime
import argparse

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

# ─── 0) CONFIGURATION ─────────────────────────────────────────────────────────

SEED = 42
N_ENVS = 32
TOTAL_TIMESTEPS = 1_000_000

NET_ARCH = [64, 64]
BATCH_SIZE = 64
INITIAL_LR = 1e-4

MAX_EPISODE_STEPS = 1000


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
    args = parser.parse_args()

    # ─── Directory & Path Setup ──────────────────────────────────────────────────
    # The --world-model-folder is the shared space for all components
    shared_folder = args.world_model_folder
    LOG_DIR = os.path.join(shared_folder, "actor_logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Agent Setup --------------------------------------------------------------
    # Create a temporary real environment just to get the observation and action spaces
    # This env is then closed and not used for training.
    print("--- Creating environment to extract space info ---")
    temp_real_env = wrapper(seed=SEED, n_envs=1)
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
        batch_size=N_ENVS,
        trained_folder=args.world_model_folder,
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
        policy_kwargs=dict(net_arch=NET_ARCH, log_std_init=-3),
        buffer_size=100_000,
        batch_size=BATCH_SIZE,
        learning_starts=1_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        gamma=0.98,
        tau=0.02,
        ent_coef="auto",
        target_update_interval=1,
        use_sde=False,  # Disable SDE for more stable training
        sde_sample_freq=-1,  # Not used when SDE is false
        learning_rate=linear_schedule(INITIAL_LR),
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        seed=SEED,
        device="auto",
    )

    # ─── 5) CALLBACKS & EVAL ENV ───────────────────────────────────────────────────
    # Create evaluation env with identical normalization (without loading any prior stats)
    eval_env = wrapper(seed=SEED, n_envs=1, max_episode_steps=MAX_EPISODE_STEPS)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best_model"),
        log_path=os.path.join(LOG_DIR, "eval_logs"),
        eval_freq=1_000,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix="sac_cp",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # ─── 6) TRAINING & SAVING ────────────────────────────────────────────────────
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=False)

    # The callbacks handle saving the best model and checkpoints, so a final save is not needed.

    # ─── 7) CLEANUP ─────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()
    print("\n[SAC-TRAINER] Training finished and environments closed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import random
from datetime import datetime

from world_models.ini_gymlike_cartpole_wrapper import GymlikeCartpoleWrapper

import numpy as np
import torch

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
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
N_ENVS = 16
TOTAL_TIMESTEPS = 500_000

NET_ARCH = [128, 128]
BATCH_SIZE = 64
INITIAL_LR = 3e-4

MAX_EPISODE_STEPS = 500


def main():
    # ─── Run-specific directory setup ─────────────────────────────────────────────
    # Generate a unique folder for everything produced this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"sac_cartpole_{timestamp}")

    # Inside that, separate subfolders for models and various logs
    MODEL_DIR = os.path.join(run_dir, "models")
    LOG_DIR = os.path.join(run_dir, "logs")

    # Create the entire hierarchy in one go
    for d in (
        MODEL_DIR,
        os.path.join(LOG_DIR, "tensorboard"),
        os.path.join(LOG_DIR, "best_model"),
        os.path.join(LOG_DIR, "eval_logs"),
        os.path.join(LOG_DIR, "checkpoints"),
    ):
        os.makedirs(d, exist_ok=True)

    # ─── 1) REPRODUCIBILITY ────────────────────────────────────────────────────────
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    # ─── 2) ENV FACTORY ───────────────────────────────────────────────────────────
    train_env = GymlikeCartpoleWrapper(
        seed=SEED, n_envs=N_ENVS, max_episode_steps=MAX_EPISODE_STEPS
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
        policy_kwargs=dict(net_arch=NET_ARCH),
        buffer_size=100_000,
        batch_size=BATCH_SIZE,
        learning_starts=1_000,
        train_freq=(1, "step"),
        gradient_steps=4,
        tau=0.02,
        ent_coef="auto",
        target_update_interval=1,
        use_sde=True,
        sde_sample_freq=4,
        learning_rate=linear_schedule(INITIAL_LR),
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tensorboard"),
        seed=SEED,
        device="auto",
    )

    # ─── 5) CALLBACKS & EVAL ENV ───────────────────────────────────────────────────
    # Create evaluation env with identical normalization (without loading any prior stats)
    eval_env = GymlikeCartpoleWrapper(
        seed=SEED, n_envs=1, max_episode_steps=MAX_EPISODE_STEPS
    )

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

    rolling_cb = RollingSuccessCountCallback(n_episodes=20, verbose=1)

    callbacks = CallbackList([eval_callback, checkpoint_callback, rolling_cb])

    # ─── 6) TRAINING & SAVING ────────────────────────────────────────────────────
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)

    model_name = f"sac_cartpole_arch{'x'.join(map(str, NET_ARCH))}_bs{BATCH_SIZE}_lr{INITIAL_LR:.0e}"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
    vec_path = os.path.join(MODEL_DIR, f"{model_name}_vecnorm.pkl")

    model.save(model_path)
    train_env.save(vec_path)

    # ─── 7) CLEANUP ─────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()


class RollingSuccessCountCallback(BaseCallback):
    """
    After each episode, logs:
      - roll/episode                         : total number of episodes so far
      - roll/last_N_successes_by_episode     : successes in last N episodes (indexed by episode count)
      - roll/last_N_successes_by_timestep    : same successes (indexed by total timesteps)
      - roll/total_timesteps                 : cumulative number of environment steps so far
    """

    def __init__(self, n_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.n_episodes = n_episodes
        # deque automatically drops oldest entries once full
        self._window = deque(maxlen=n_episodes)
        # count of completed episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        Called at every environment step. We only act when an env reports done=True.
        """
        # ─── detect which sub-envs just ended ────────────────────────────────────
        # `dones` is a list/array of bools for each parallel env
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for idx, done in enumerate(dones):
            if not done:
                continue  # nothing to do until this env finishes an episode

            self.episode_count += 1

            info = infos[idx]
            # ⮕ Gymnasium will set this when the episode ends due to hitting the time limit
            truncated = info.get("TimeLimit.truncated", False)
            # ⮕ we treat a truncation by timeout as a “success”
            success = int(truncated)

            self._window.append(success)

            # ─── rolling sum over last N episodes ───────────────────────────────
            count = sum(self._window)
            print(f"[Callback] Last {self.n_episodes} successes: {count}")

            # ─── log under the episode-indexed tag ───────────────────────────────
            self.logger.record("roll/last_N_successes_by_episode", count)
            # use episode_count as the step for this series
            self.logger.dump(self.episode_count)

            # ─── log under the timestep-indexed tag ──────────────────────────────
            self.logger.record("roll/last_N_successes_by_timestep", count)
            self.logger.record("roll/total_timesteps", self.num_timesteps)
            # use num_timesteps as the step for this series
            self.logger.dump(self.num_timesteps)

            # (Optional) also log the raw episode index
            self.logger.record("roll/episode", self.episode_count)
            # it will show up on the next dump

        return True


if __name__ == "__main__":
    main()

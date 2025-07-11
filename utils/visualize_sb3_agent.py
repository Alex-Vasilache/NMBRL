#!/usr/bin/env python3
import os
from stable_baselines3 import SAC
from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper


def find_latest_model_and_vecnormalize(run_dir):
    """Finds the latest model .zip file and the VecNormalize stats .pkl file in a run directory."""
    # We load from the 'models' directory because the `_vecnorm.pkl` file, which contains
    # essential normalization statistics, is saved there alongside the final model.
    # The 'best_model' directory only contains the model parameters.
    models_dir = os.path.join(run_dir, "logs", "best_model")

    if not os.path.isdir(models_dir):
        return None, None

    # Find the latest model file
    model_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".zip")
    ]
    if not model_files:
        return None, None
    latest_model = max(model_files, key=os.path.getctime)

    # Find the corresponding VecNormalize file
    model_name_without_ext = os.path.splitext(os.path.basename(latest_model))[0]
    vec_normalize_path = os.path.join(
        models_dir, f"{model_name_without_ext}_vecnorm.pkl"
    )

    if not os.path.exists(vec_normalize_path):
        print(
            f"Warning: Corresponding VecNormalize file not found at {vec_normalize_path}"
        )
        return latest_model, None

    return latest_model, vec_normalize_path


def main():
    # --- Configuration ---
    RUNS_DIR = "runs"
    N_EVAL_EPISODES = 100

    # --- Find the directory of the latest run ---
    run_dirs = [
        os.path.join(RUNS_DIR, d)
        for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    ]
    if not run_dirs:
        print("No run directories found.")
        return
    latest_run_dir = max(run_dirs, key=os.path.getctime)

    # --- Find latest model and VecNormalize stats ---
    latest_model_path, vec_normalize_path = find_latest_model_and_vecnormalize(
        latest_run_dir
    )

    if not latest_model_path:
        print(f"No model found in {latest_run_dir}")
        return

    # --- Create and load evaluation environment ---
    # The stats file (vec_normalize_path) contains the running average of observations
    # It's important to load this to ensure the model sees data in the same distribution it was trained on
    eval_env = wrapper(seed=42, n_envs=1, render_mode="human", max_episode_steps=5000)
    if vec_normalize_path:
        print(f"Loading VecNormalize stats from: {vec_normalize_path}")
        eval_env = wrapper.load(vec_normalize_path, eval_env)
        # The render_mode is not saved in the vecnormalize file, so we need to set it again
        eval_env.venv.render_mode = "human"

    # We need to set training mode to False so that running average of observations and rewards is not updated
    eval_env.training = False
    # We need to set norm_reward to False because we want to see the real reward
    eval_env.norm_reward = False
    seed = 42
    # --- Evaluate ---
    for episode in range(N_EVAL_EPISODES):
        # --- Load Model ---
        print(f"Loading model from: {latest_model_path}")
        model = SAC.load(latest_model_path, env=eval_env)
        eval_env.seed(seed)
        # --- Evaluate ---
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # By pumping pygame events, we keep the window responsive
            # pygame.event.pump()

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            eval_env.render()
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

        seed += 1

    eval_env.close()


if __name__ == "__main__":
    main()

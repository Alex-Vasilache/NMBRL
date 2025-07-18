#!/usr/bin/env python3
import os
import argparse
from stable_baselines3.common.monitor import Monitor
import yaml
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from agents.actor_wrapper import ActorWrapper

# from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper
from world_models.dynamic_world_model_wrapper import WorldModelWrapper
from utils.tools import (
    seed_everything,
    save_config_to_shared_folder,
    resolve_all_device_configs,
)

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
    parser.add_argument(
        "--env-type",
        type=str,
        required=False,
        help="Type of environment to use. Options: 'dmc', 'physical'.",
        default="dmc",
    )

    args = parser.parse_args()

    # --- Load configuration from YAML file ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve all device configurations
    config = resolve_all_device_configs(config)

    global_config = config["global"]
    agent_config = config["agent_trainer"]
    agent_type = agent_config.get("agent_type", "PPO").upper()

    # Save config to shared folder for reproducibility
    save_config_to_shared_folder(
        config, args.config, args.shared_folder, "agent_trainer"
    )

    # seed everything
    seed_everything(global_config["seed"])

    # The --shared-folder is the shared space for all components
    shared_folder = args.shared_folder
    LOG_DIR = os.path.join(shared_folder, "actor_logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Setup TensorBoard logging
    tb_config = config.get("tensorboard", {})
    tb_log_dir = os.path.join(
        shared_folder, tb_config.get("log_dir", "tb_logs"), "agent"
    )
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"[AGENT-TRAINER] TensorBoard logging to: {tb_log_dir}")

    # Create a temporary real environment just to get the observation and action spaces
    # This env is then closed and not used for training.
    print("--- Creating environment to extract space info ---")
    if args.env_type == "dmc":
        from world_models.dmc_cartpole_wrapper import DMCCartpoleWrapper as wrapper

        temp_real_env = wrapper(seed=global_config["seed"], n_envs=1, render_mode=None)

        obs_space = temp_real_env.observation_space
        act_space = temp_real_env.action_space
        temp_real_env.close()

    elif args.env_type == "physical":
        from world_models.physical_cartpole_wrapper import (
            ACTION_SPACE,
            OBSERVATION_SPACE,
        )

        obs_space = OBSERVATION_SPACE
        act_space = ACTION_SPACE

    else:
        raise ValueError(f"Invalid environment type: {args.env_type}")

    print(f"Obs space: {obs_space}, Action space: {act_space}")

    # Ensure action_space is Box type for WorldModelWrapper
    from gymnasium.spaces import Box

    if not isinstance(act_space, Box):
        raise TypeError(f"Expected action_space to be Box, got {type(act_space)}")

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

    # Check if we should load from a checkpoint
    # The DreamerACAgent saves checkpoints in the 'checkpoints' directory
    checkpoint_agent_dir = os.path.join(shared_folder, "checkpoints")
    if os.path.exists(checkpoint_agent_dir):
        # Look for the latest checkpoint
        checkpoint_files = [
            f for f in os.listdir(checkpoint_agent_dir) if f.endswith(".zip")
        ]
        if checkpoint_files:
            # Sort by creation time to get the latest
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda f: os.path.getctime(os.path.join(checkpoint_agent_dir, f)),
            )
            latest_checkpoint_path = os.path.join(
                checkpoint_agent_dir, latest_checkpoint
            )
            try:
                print(
                    f"[AGENT-TRAINER] Loading existing agent from checkpoint: {latest_checkpoint}"
                )
                # Load the checkpoint into the ActorWrapper's model
                if agent_type == "DREAMER":
                    from agents.dreamer_ac_agent import DreamerACAgent

                    # Patch DreamerACAgent.load to support force-cpu (same as ActorWrapper)
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
                            training_info_path = os.path.join(
                                temp_dir, "training_info.pkl"
                            )
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
                            print(
                                f"[AGENT-TRAINER] Loading model to device: {map_location}"
                            )
                            actor_path = os.path.join(temp_dir, "actor.pth")
                            actor_data = torch.load(
                                actor_path,
                                map_location=map_location,
                                weights_only=False,
                            )
                            agent.agent.actor.load_state_dict(
                                actor_data["model_state_dict"]
                            )
                            critic_path = os.path.join(temp_dir, "critic.pth")
                            critic_data = torch.load(
                                critic_path,
                                map_location=map_location,
                                weights_only=False,
                            )
                            agent.agent.critic.load_state_dict(
                                critic_data["model_state_dict"]
                            )
                            agent.episode_rewards = training_info.get(
                                "episode_rewards", []
                            )
                            agent.episode_lengths = training_info.get(
                                "episode_lengths", []
                            )
                            agent.training_losses = training_info.get(
                                "training_losses", []
                            )
                            agent.agent.actor.eval()
                            agent.agent.critic.eval()
                            return agent

                    dreamer_mod.DreamerACAgent.load = staticmethod(
                        lambda path, env=None: patched_load(path, env, force_cpu=True)
                    )

                    # Load the checkpoint with force_cpu=True
                    loaded_agent = DreamerACAgent.load(
                        latest_checkpoint_path, env=train_env
                    )

                    # Fix the save directory to point to the new run directory
                    # The loaded agent has the old tb_log_dir, so we need to update it
                    loaded_agent.tb_log_dir = os.path.join(
                        shared_folder,
                        tb_config.get("log_dir", "tb_logs"),
                        "dreamer_agent",
                    )
                    os.makedirs(loaded_agent.tb_log_dir, exist_ok=True)

                    # Update the save_dir to point to the new run's checkpoints directory
                    loaded_agent.save_dir = os.path.join(shared_folder, "checkpoints")
                    os.makedirs(loaded_agent.save_dir, exist_ok=True)

                    # Update the writer to use the new log directory
                    if (
                        hasattr(loaded_agent, "writer")
                        and loaded_agent.writer is not None
                    ):
                        loaded_agent.writer.close()
                    loaded_agent.writer = SummaryWriter(log_dir=loaded_agent.tb_log_dir)

                    # Replace the ActorWrapper's model with the loaded one
                    agent.model = loaded_agent

                    print("[AGENT-TRAINER] Successfully loaded agent from checkpoint")
                    print(
                        f"[AGENT-TRAINER] Updated save directory to: {loaded_agent.save_dir}"
                    )
                else:
                    print(
                        "[AGENT-TRAINER] Agent checkpoint found, will be loaded by ActorWrapper"
                    )
            except Exception as e:
                print(
                    f"[AGENT-TRAINER] Warning: Failed to load agent from checkpoint: {e}"
                )
                print("[AGENT-TRAINER] Starting with fresh agent")
        else:
            print(
                "[AGENT-TRAINER] No existing agent checkpoint files found, starting with fresh agent"
            )
    else:
        print(
            "[AGENT-TRAINER] No agent checkpoint directory found, starting with fresh agent"
        )

    # Setup TensorBoard logging for callbacks
    class TensorBoardLoggingCallback:
        def __init__(self, log_dir):
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
            self.step_count = 0

        def on_step(self, agent) -> bool:
            # This would be called during training to log metrics
            # The actual implementation depends on the agent type
            self.step_count += 1
            return True

        def close(self):
            self.writer.close()

    tb_callback = TensorBoardLoggingCallback(tb_log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=agent_config["checkpoint_freq"],
        save_path=os.path.join(LOG_DIR, "checkpoints"),
        name_prefix=f"{agent_type.lower()}_cp",
    )

    callbacks = CallbackList([checkpoint_callback])

    try:
        agent.learn(
            total_timesteps=agent_config["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )
    finally:
        tb_callback.close()
        print(f"[AGENT-TRAINER] TensorBoard logs saved to: {tb_log_dir}")

    train_env.close()
    print("\n[AGENT-TRAINER] Training finished and environments closed.")


if __name__ == "__main__":
    main()

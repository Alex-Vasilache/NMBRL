from stable_baselines3.common.callbacks import CallbackList
import os
import sys
import tempfile
import zipfile
import pickle
from datetime import datetime

import torch

# This is a hacky way to ensure the imports from the submodule work
submodule_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "Spatial-SNNs",
    )
)

if submodule_root not in sys.path:
    sys.path.insert(0, submodule_root)

from src.task import Task
from util.config_loader import map_config_to_args
from util.args import dotdict
from src.search.evolution import Evolution
import numpy as np
from util.training import ProgressBar


class EvoAgent:
    def __init__(
        self,
        config,
        env,
        tensorboard_log=str(os.path.join(os.path.dirname(__file__), "..", "runs")),
    ):
        self.config = config
        self.env = env
        self.tensorboard_log = tensorboard_log
        yaml_args = config.get("evo_agent_trainer", {})
        yaml_args = map_config_to_args(yaml_args)
        self.args = dotdict(yaml_args)

        if "max_env_steps" not in self.args:
            if self.args.curiculum_learning:
                self.args.max_env_steps = 10
            else:
                self.args.max_env_steps = 1000
        else:
            if self.args.curiculum_learning:
                self.args.max_env_steps = self.args.max_env_steps
            else:
                self.args.max_env_steps = self.args.max_env_steps

        self.args.time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task = Task(self.args, env=self.env)
        self.data_path = self.task.data_path
        self.args.data_path = self.data_path
        self.increase_every_nth = 1

        output_features = self.task.out_feat

        if not self.task.action_bounds is None:
            if self.task.discretize_intervals:
                output_features = int(output_features * self.task.discretize_intervals)

        self.search_dist = Evolution(
            net_size=self.args.net_size,
            pool_size=self.args.num_gene_samples,
            input_neurons=self.task.inp_feat,
            output_neurons=output_features,
            f1=self.task.of,
            f2=self.args.sigma_bins,
            f3=self.args.sparsity_bins,
            max_vthr=self.args.max_vthr,
            prune_unconnected=self.args.prune_unconnected,
            evolution_method=self.args.evolution_method,
            spatial=self.args.spatial,
        )

        self.info = self.args.__dict__
        self.info["game_name"] = self.task.game_name
        self.info["weights_iteration"] = 0
        self.info["avg_score"] = "None"
        self.info["net_size"] = self.args.net_size
        self.info["evolution_method"] = self.args.evolution_method
        self.info["num_gene_samples"] = self.args.num_gene_samples

        self.info["max_score"] = "None"
        self.info["map_scores"] = np.array2string(
            np.array(self.search_dist.map_scores),
            formatter={"float_kind": lambda x: "%4.0f" % x},
        )

        map_scores = np.array(self.search_dist.map_scores)
        # Ensure map_scores is a 2D array for saving
        if len(map_scores.shape) == 1:
            map_scores = map_scores.reshape(-1, 1)
        elif len(map_scores.shape) == 3:
            map_scores = map_scores.reshape(-1, map_scores.shape[-1])

        self.just_created = False
        self.test_task = None  # Will be initialized when needed for prediction

    def save_models(self, epoch=None):
        """
        Save the trained evolutionary agent along with training configuration.
        Compatible with SB3 format (.zip file).

        :param epoch: Optional epoch number to include in filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if epoch is not None:
            model_name = f"evo_agent_ep{epoch}_{timestamp}.zip"
        else:
            model_name = f"evo_agent_final_{timestamp}.zip"

        # Create save directory if it doesn't exist
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self.tensorboard_log))),
            "checkpoints",
        )
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, model_name)

        # Create a temporary directory to store files before zipping
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the search distribution (evolution state)
            search_dist_path = os.path.join(temp_dir, "search_dist.save")
            import joblib

            joblib.dump(self.search_dist, search_dist_path)

            # Save the task configuration
            task_config_path = os.path.join(temp_dir, "task_config.pkl")
            with open(task_config_path, "wb") as f:
                pickle.dump(
                    {
                        "args": self.args.__dict__,
                        "task_config": {
                            "inp_feat": self.task.inp_feat,
                            "out_feat": self.task.out_feat,
                            "of": self.task.of,
                            "action_bounds": self.task.action_bounds,
                            "discretize_intervals": self.task.discretize_intervals,
                            "size": self.task.size,
                            "spike_steps": self.task.spike_steps,
                            "game_name": self.task.game_name,
                        },
                    },
                    f,
                )

            # Save training configuration and statistics
            training_info = {
                "config": self.config,
                "args": self.args.__dict__,
                "info": self.info,
                "final_stats": {
                    "best_score": (
                        self.search_dist.scores.max()
                        if self.search_dist.scores.size > 0
                        else 0
                    ),
                    "avg_score": (
                        self.search_dist.scores.mean()
                        if self.search_dist.scores.size > 0
                        else 0
                    ),
                    "total_elites": (
                        len(self.search_dist.elites)
                        if hasattr(self.search_dist, "elites")
                        else 0
                    ),
                    "evolution_method": self.args.evolution_method,
                    "net_size": self.args.net_size,
                },
            }

            training_info_path = os.path.join(temp_dir, "training_info.pkl")
            with open(training_info_path, "wb") as f:
                pickle.dump(training_info, f)

            # Create the zip file
            with zipfile.ZipFile(model_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(search_dist_path, "search_dist.save")
                zipf.write(task_config_path, "task_config.pkl")
                zipf.write(training_info_path, "training_info.pkl")

        print(f"EvoAgent saved to: {model_path}")
        return model_path

    @classmethod
    def load(cls, path, env=None):
        """
        Load an EvoAgent from a .zip file.
        Compatible with SB3 loading pattern.

        :param path: Path to the .zip file
        :param env: Environment (optional, for compatibility)
        :return: Loaded EvoAgent instance
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the zip file
            with zipfile.ZipFile(path, "r") as zipf:
                zipf.extractall(temp_dir)

            # Load training info
            training_info_path = os.path.join(temp_dir, "training_info.pkl")
            with open(training_info_path, "rb") as f:
                training_info = pickle.load(f)

            config = training_info["config"]
            args_dict = dotdict(map_config_to_args(config["evo_agent_trainer"]))

            # Create a new agent instance with the loaded configuration
            temp_log_dir = os.path.join(
                tempfile.gettempdir(),
                f"evo_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            agent = cls(config, env, tensorboard_log=temp_log_dir)

            # Load the search distribution
            search_dist_path = os.path.join(temp_dir, "search_dist.save")
            import joblib

            agent.search_dist = joblib.load(search_dist_path)

            # Load task configuration
            task_config_path = os.path.join(temp_dir, "task_config.pkl")
            with open(task_config_path, "rb") as f:
                task_config = pickle.load(f)

            # Update agent's args with loaded configuration
            agent.args = dotdict(args_dict)
            agent.info = training_info["info"]
            agent.args.time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create test task for prediction
            test_args = dotdict(args_dict.copy())
            test_args.device = agent.config["global"]["device"]
            test_args.test = True
            test_args.time = agent.args.time
            test_args.random_seed = 99
            test_args.num_data_samples = 1
            test_args.batch_size_data = 1
            test_args.batch_size_gene = 1
            test_args.num_data_samples = 1
            test_args.num_gene_samples = 1
            test_args.max_env_steps = 1000
            agent.test_task = Task(test_args, env=env)
            agent.just_created = True

            agent.test_task.set_params([agent.search_dist.elites[0]])

            print(f"EvoAgent loaded from: {path}")
            return agent

    def learn(
        self,
        total_timesteps: int,
        callback: CallbackList = CallbackList([]),
        progress_bar: bool = False,
    ):

        test_elites = []
        test_scores = []
        test_ftdesc = []
        test_stds = []

        train_log = []
        test_log = []

        test_args = dotdict(self.args.copy())
        test_args.test = True
        test_args.random_seed = 99
        test_args.num_data_samples = 100
        test_args.batch_size_data = 100
        test_args.max_env_steps = 1000
        total_steps = 0

        pb = ProgressBar(num_iterations=self.args.num_iterations)

        for iteration in range(self.args.weights_iteration, self.args.num_iterations):
            samples = self.search_dist.pool
            rewards, std_devs, ft_desc = self.task.reward(samples)

            self.search_dist.update_params(rewards, ft_desc[:-1])
            total_steps += int(ft_desc[-1].sum())

            max_rew_idx = np.argmax(rewards)

            train_log.append(
                (
                    iteration,
                    total_steps,
                    rewards.mean(),
                    rewards.max(),
                    std_devs[max_rew_idx],
                )
            )

            self.info["weights_iteration"] = str(iteration)
            self.info["total_steps"] = str(total_steps)
            self.info["avg_reward"] = (
                str(self.search_dist.scores.mean())
                if self.search_dist.scores.size > 0
                else "None"
            )
            self.info["max_reward"] = (
                str(self.search_dist.scores.max())
                if self.search_dist.scores.size > 0
                else "None"
            )
            if self.search_dist.evolution_method == "map_elites":
                self.info["map_scores"] = np.array2string(
                    np.array(self.search_dist.map_scores),
                    max_line_width=1000,
                    formatter={"float_kind": lambda x: "%4.0f" % x},
                )

            if (iteration + 1) % 10 == 0 and iteration != 0:
                self.save_models(iteration)

            # --- Curriculum Learning ---
            if (
                iteration % self.increase_every_nth == 0
                and self.args.curiculum_learning
            ):
                self.args.max_env_steps = min(self.args.max_env_steps * 1.1, 1000)
                del self.task
                self.task = Task(self.args)
                self.increase_every_nth = self.increase_every_nth + 1
                self.info["max_env_steps"] = self.args.max_env_steps

            pb(
                avg_r1=(
                    self.search_dist.scores.mean()
                    if self.search_dist.scores.size > 0
                    else 0
                ),
                min_r1=(
                    self.search_dist.scores.min()
                    if self.search_dist.scores.size > 0
                    else 0
                ),
                max_r1=(
                    self.search_dist.scores.max()
                    if self.search_dist.scores.size > 0
                    else 0
                ),
            )

        self.task.cleanup()
        del self.task

        # Save final model
        final_model_path = self.save_models()
        print(f"Training complete. Final model saved to: {final_model_path}")

    def predict(self, obs, deterministic=False):
        """
        Predict action for given observation.
        Compatible with SB3 interface.

        :param obs: Observation
        :param deterministic: Whether to use deterministic action
        :return: Action and state (None for this implementation)
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(
                obs, dtype=torch.float32, device=self.config["global"]["device"]
            )

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        obs = np.array(obs)

        if self.just_created and self.test_task is not None:
            self.test_task.set_params([self.search_dist.elites[0]])
            self.just_created = False

        if self.test_task is None:
            # Initialize test task if not already done
            test_args = dotdict(self.args.__dict__.copy())
            test_args.test = True
            test_args.random_seed = 99
            test_args.num_data_samples = 1
            test_args.batch_size_data = 1
            test_args.max_env_steps = 1000
            self.test_task = Task(test_args, env=self.env)
            self.test_task.set_params([self.search_dist.elites[0]])

        with torch.no_grad():
            states = obs
            t, b, f = (
                self.test_task.spike_steps,
                self.test_task.batch_size,
                self.test_task.inp_feat,
            )
            input_state = (
                states.reshape(b, f)
                .repeat(t)
                .reshape(b, f, t)
                .swapaxes(0, 2)
                .swapaxes(1, 2)
            )

            input_state = input_state.reshape(
                self.test_task.spike_steps,
                self.test_task.batch_size_data,
                self.test_task.batch_size_gene * self.test_task.inp_feat,
            )

            actions, sigmoid_actions = self.test_task.net.forward(input_state)

        return actions, None

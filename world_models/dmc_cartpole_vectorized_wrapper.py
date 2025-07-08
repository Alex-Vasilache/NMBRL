# This file will contain a vectorized wrapper for the DeepMind Control Suite,
# using multiprocessing to run multiple environments in parallel.

import numpy as np
import multiprocessing as mp
import cv2
from .base_world_model import BaseWorldModel
import collections
import time

try:
    from dm_control import suite
    from dm_control.rl.control import Environment
except ImportError:
    print(
        "DeepMind Control Suite is not installed. Please install it with: pip install dm_control"
    )
    suite = None


def worker(conn, task_name, time_limit, control_timestep, render_kwargs):
    """
    Worker process for running a single dm_control environment.
    """
    if suite is None:
        return
    env = suite.load(
        domain_name="cartpole",
        task_name=task_name,
        visualize_reward=True,
        task_kwargs={"time_limit": time_limit},
        environment_kwargs={"control_timestep": control_timestep},
    )

    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            action = data
            time_step = env.step(action)
            conn.send((time_step.observation, time_step.reward, time_step.last()))
        elif cmd == "reset":
            time_step = env.reset()
            if data is not None:
                with env.physics.reset_context():
                    env.physics.set_state(data)
            conn.send(env.physics.get_state())
        elif cmd == "close":
            env.close()
            conn.close()
            break
        elif cmd == "render":
            if render_kwargs:
                conn.send(env.physics.render(**render_kwargs))
            else:
                conn.send(None)
        elif cmd == "_get_action_spec":
            conn.send(env.action_spec())
        elif cmd == "_get_observation_spec":
            conn.send(env.observation_spec())
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class DMCVectorizedCartPoleWrapper(BaseWorldModel):
    """
    A vectorized wrapper for dm_control environments that runs them in parallel.
    This version does not support visualization or setting initial states.
    """

    def __init__(
        self,
        batch_size=1,
        max_steps=1000,
        visualize=False,
        render_width=640,
        render_height=480,
        dt_simulation=0.02,
    ):
        if suite is None:
            raise ImportError(
                "dm_control not found. Please install with 'pip install dm_control'"
            )

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.visualize = visualize
        self.render_width = render_width
        self.render_height = render_height
        self.dt_simulation = dt_simulation

        render_kwargs = {
            "width": self.render_width,
            "height": self.render_height,
            "camera_id": 0,
        }

        # Create pipes for communication
        parent_conns, worker_conns = zip(*[mp.Pipe() for _ in range(batch_size)])
        self.parent_conns = parent_conns

        # Create worker processes
        self.workers = [
            mp.Process(
                target=worker,
                args=(
                    worker_conns[i],
                    "swingup",
                    max_steps,
                    dt_simulation,
                    render_kwargs if i == 0 and visualize else None,
                ),
            )
            for i in range(batch_size)
        ]

        for p in self.workers:
            p.start()

        for pipe in worker_conns:
            pipe.close()

        self._action_spec = self.get_action_spec()
        self._observation_spec = self.get_observation_spec()

        self.state_dim = sum(
            np.prod(spec.shape) for spec in self._observation_spec.values()
        )
        self.action_dim = np.prod(self._action_spec.shape)

    def get_action_spec(self):
        self.parent_conns[0].send(("_get_action_spec", None))
        return self.parent_conns[0].recv()

    def get_observation_spec(self):
        self.parent_conns[0].send(("_get_observation_spec", None))
        return self.parent_conns[0].recv()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reset(self, initial_state=None):
        if initial_state is not None:
            # If initial_state is a single state, broadcast to all envs
            if not isinstance(initial_state, list):
                initial_state = [initial_state] * self.batch_size
        else:
            initial_state = [None] * self.batch_size

        for i, pipe in enumerate(self.parent_conns):
            pipe.send(("reset", initial_state[i]))

        obs = [pipe.recv() for pipe in self.parent_conns]
        return self._flatten_obs(obs)

    def step(self, actions):
        for i, pipe in enumerate(self.parent_conns):
            pipe.send(("step", actions[i]))

        results = [pipe.recv() for pipe in self.parent_conns]

        obs, rewards, dones = zip(*results)

        return self._flatten_obs(obs), np.array(rewards), np.array(dones)

    def close(self):
        for pipe in self.parent_conns:
            pipe.send(("close", None))
        for p in self.workers:
            p.join()
        for pipe in self.parent_conns:
            pipe.close()

    def render(self):
        if self.visualize:
            self.parent_conns[0].send(("render", None))
            img = self.parent_conns[0].recv()
            if img is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("DMC CartPole", img_bgr)
                cv2.waitKey(1)
        else:
            print(
                "Visualization is not enabled. Please set visualize=True during initialization."
            )
            return None

    def _flatten_obs(self, obs_batch):
        # Flattens a batch of dm_control observations
        flattened_batch = []
        for obs_dict in obs_batch:
            if isinstance(obs_dict, collections.OrderedDict):
                # For initial reset which might not have the full structure
                flat_obs = np.concatenate([v.flatten() for v in obs_dict.values()])
                flattened_batch.append(flat_obs)
            else:  # This handles the case where obs is a numpy array from physics.get_state()
                flattened_batch.append(obs_dict)
        return np.array(flattened_batch)

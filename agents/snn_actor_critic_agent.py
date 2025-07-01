# This file will contain the implementation of the SNN-based actor-critic agent.
# It will inherit from the BaseAgent and use Spiking Neural Networks for its policy and value functions.

import numpy as np
from Neuromorphic_MBRL.agents.base_agent import BaseAgent


class SnnActorCriticAgent(BaseAgent):
    """
    An SNN-based Actor-Critic agent.
    For now, it will just select random actions. The actual SNN networks and learning
    logic will be added in subsequent tasks.
    """

    def __init__(self, action_space):
        # The action space is typically a Box from gymnasium, which has a `sample()` method.
        self.action_space = action_space

    def get_action(self, state):
        """
        Returns a random action from the action space.

        :param state: The current state of the environment (not used for this initial version).
        :return: A random action.
        """
        return self.action_space.sample()

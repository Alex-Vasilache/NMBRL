# This file will contain the implementation of the Actor-Critic agent.
# It will house the actor and critic networks (both standard and SNN-based)
# and the logic for how the agent selects actions based on observations.
import numpy as np
from Neuromorphic_MBRL.agents.base_agent import BaseAgent


class ActorCriticAgent(BaseAgent):
    """
    An Actor-Critic agent.
    For now, it will just select random actions. The actual networks and learning
    logic will be added in subsequent tasks.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        """
        Returns a random action from the action space.

        :param state: The current state of the environment (not used for this initial version).
        :return: A random action.
        """
        return self.action_space.sample()

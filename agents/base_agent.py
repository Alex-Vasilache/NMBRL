# This file will define the abstract base class for all agents.
# It will enforce a common interface for agents, including methods like `get_action()`.
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseAgent(ABC, nn.Module):
    @abstractmethod
    def get_action(self, state):
        pass

# This file will define the abstract base class for all world models.
# It will enforce a common interface for models, including `step()` and `reset()` methods,
# allowing the agent to interact with both the real environment and learned models seamlessly.
from abc import ABC, abstractmethod


class BaseWorldModel(ABC):
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

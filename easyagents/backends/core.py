"""This module contains backend core classes like Backend and BackendAgent.

    The concrete backends like tfagent or baselines are implemented in seprate modules.
"""

from abc import ABC, abstractmethod
from typing import List
from easyagents.core import TrainCallback, AgentConfig


class BackendAgent(ABC):
    """Base class for all backend agent implementations.

        Implements the train loop and calls the TrainCallbacks.
    """

    def train(self, train: List[TrainCallback]):
        pass


class Backend(ABC):
    """Abstract base class for all backends.

    It contains in particular the code shared between all backends.
    """

    @abstractmethod
    def create_ppo_agent(self, agent_config: AgentConfig) -> BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.

        Args:
            agent_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.
        """
        pass


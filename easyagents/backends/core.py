"""This module contains backend core classes like Backend and BackendAgent.

    The concrete backends like tfagent or baselines are implemented in seprate modules.
"""

from abc import ABC, abstractmethod
from typing import List
from easyagents import core


class BackendAgent(ABC):
    """Base class for all backend agent implementations.

        Implements the train loop and calls the TrainCallbacks.
    """

    def __init__(self, model_config: core.ModelConfig):
        assert model_config is not None, "model_config not set."
        self._model_config = model_config

    def train(self,
              train_context: core.TrainContext,
              train: List[core.TrainCallback],
              play: List[core.PlayCallback],
              ):
        """
            for i in num_iterations
                for e in num_episodes_per_iterations
                    play episode and record steps (while steps_in_episode < max_steps_per_episode and)
                train policy for num_epochs_per_iteration epochs
                if current_episode % num_iterations_between_eval == 0:
                    evaluate policy
                if training_done
                    break

            Args:
                train: list of callbacks called during the train loop (but not during the intermittent evaluations)
                play: list of callbacks called during the evaluation (but not during data collection for the training)
                train_context: training configuration to be used
        """
        assert train_context is not None , "train_context not set"
        assert train is not None, "train not set"
        assert play is not None, "play not set"
        pass


class Backend(ABC):
    """Abstract base class for all backends.

    It contains in particular the code shared between all backends.
    """

    @abstractmethod
    def create_ppo_agent(self, model_config: core.ModelConfig) -> BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.

        Args:
            model_config: the agents configuration containing in patricular the name of the gym environment
                to be used and the nn architecture.
        """
        pass

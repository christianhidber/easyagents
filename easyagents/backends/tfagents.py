"""This module contains the backend implementation for tf Agents (see https://github.com/tensorflow/agents)"""

from easyagents import core
from easyagents.backends import core as bcore


class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend for TfAgents.

        Serves as a factory to create algorithm specific wrappers for the TfAgents implementations.
    """

    def create_ppo_agent(self, model_config: core.ModelConfig) -> bcore.BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.
        """
        return TfPpoAgent(model_config=model_config)


class TfAgent(bcore.BackendAgent):
    """Base class of all TfAgent based agent implementations.

        Contains all TfAgents specific code shared among all implementations.
    """

    def train_implementation(self, train_context: core.TrainContext):
        pass


class TfPpoAgent(TfAgent):
    """TfAgents Implementation for Ppo, deriving from the TfAgent"""

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config)

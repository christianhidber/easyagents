"""This module contains the backend implementation for tf Agents (see https://github.com/tensorflow/agents)"""

from easyagents.backends.core import Backend, BackendAgent
from easyagents.core import ModelConfig


class TfAgentsBackend(Backend):
    """Backend for TfAgents.

        Serves as a factory to create algorithm specific wrappers for the TfAgents implementations.
    """

    def create_ppo_agent(self, model_config: ModelConfig) -> BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.
        """
        return TfPpoAgent(model_config=model_config)


class TfAgent(BackendAgent):
    """Base class of all TfAgent based agent implementations.

        Contains all TfAgents specific code shared among all implementations.
    """


class TfPpoAgent(TfAgent):
    """TfAgents Implementation for Ppo, deriving from the TfAgent, implementig the PpoBackendAgent interface"""

    def __init__(self, model_config: ModelConfig):
        pass

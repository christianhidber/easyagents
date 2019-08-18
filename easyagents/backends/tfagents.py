from easyagents.backends.core import Backend, BackendAgent
from easyagents.core import AgentConfig


class TfAgent(BackendAgent):
    """Base class of all TfAgent based agent implementations.

        Contains all TfAgents specific code shared among all implementations.
    """


class TfPpoAgent(TfAgent):
    """TfAgents Implementation for Ppo, deriving from the TfAgent, implementig the PpoBackendAgent interface"""

    def __init__(self, agent_config: AgentConfig):
        pass


class TfAgentsBackend(Backend):
    """Backend for TfAgents.

        Serves as a factory to create algorithm specific wrappers for the TfAgents implementations.
    """

    def create_ppo_agent(self, agent_config: AgentConfig) -> BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation.

            If this backend does not implement PpoAgent then throw a NotImplementedError exception.
        """
        return TfPpoAgent(agent_config=agent_config)

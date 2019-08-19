from easyagents.backends.core import Backend, BackendAgent
from easyagents.core import ModelConfig
from easyagents.backends.tfagents import TfAgentsBackend


class DefaultBackend(Backend):
    """Backend which redirects all calls to the some default implementation."""

    def __init__(self):
        self._tfagents = TfAgentsBackend()

    def create_ppo_agent(self, agent_config: ModelConfig) -> BackendAgent:
        return self._tfagents.create_ppo_agent(agent_config)

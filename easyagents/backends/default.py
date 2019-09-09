from easyagents import core
from easyagents.backends import core as bcore
from easyagents.backends import tfagents

class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend which redirects all calls to the some default implementation."""

    name = 'default'

    def __init__(self):
        self._tfagents = tfagents.BackendAgentFactory()

    def create_dqn_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return self._tfagents.create_dqn_agent(model_config)

    def create_ppo_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return self._tfagents.create_ppo_agent(model_config)

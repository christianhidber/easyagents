from easyagents import core
from easyagents.backends import core as bcore
import easyagents.backends.tfagents

class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend which redirects all calls to the some default implementation."""

    name = 'default'

    def __init__(self):
        self._tfagents =  easyagents.backends.tfagents.BackendAgentFactory()

    def create_dqn_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return self._tfagents.create_dqn_agent(model_config)

    def create_ppo_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return self._tfagents.create_ppo_agent(model_config)

    def create_random_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return self._tfagents.create_random_agent(model_config)

    def create_reinforce_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        return self._tfagents.create_reinforce_agent(model_config)
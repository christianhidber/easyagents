from easyagents import core
from easyagents.backends import core as bcore
from easyagents.backends import tfagents


class Backend(bcore.Backend):
    """Backend which redirects all calls to the some default implementation."""

    def __init__(self):
        self._tfagents = tfagents.Backend()

    def create_ppo_agent(self, model_config: core.ModelConfig) -> bcore.BackendAgent:
        return self._tfagents.create_ppo_agent(model_config)

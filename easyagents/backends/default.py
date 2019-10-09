from typing import Type, Dict
from easyagents.backends import core as bcore
import easyagents.backends.tfagents
import easyagents.backends.kerasrl


class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend which redirects all calls to the some default implementation."""

    backend_name = 'default'

    def get_algorithms(self) -> Dict[Type, Type[easyagents.backends.core.BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        return {easyagents.agents.CemAgent: easyagents.backends.kerasrl.KerasRlCemAgent,
                easyagents.agents.DqnAgent: easyagents.backends.tfagents.TfDqnAgent,
                easyagents.agents.DoubleDqnAgent: easyagents.backends.kerasrl.KerasRlDoubleDqnAgent,
                easyagents.agents.DuelingDqnAgent: easyagents.backends.kerasrl.KerasRlDuelingDqnAgent,
                easyagents.agents.PpoAgent: easyagents.backends.tfagents.TfPpoAgent,
                easyagents.agents.RandomAgent: easyagents.backends.tfagents.TfRandomAgent,
                easyagents.agents.ReinforceAgent: easyagents.backends.tfagents.TfReinforceAgent,
                easyagents.agents.SacAgent: easyagents.backends.tfagents.TfSacAgent}

from typing import Type, Dict

from easyagents.backends import core as bcore
import easyagents.core


class TensorforceNotRegisteredAgent(bcore.BackendAgent):

    def __init__(self, model_config: easyagents.core.ModelConfig, backend_name: str):
        raise NotImplementedError(
            "Call agents.register_tensorforce() to activate tensorforce before instantiating the first agent."
            "This agent is implemented by tensorforce. Due to an incompatibility"
            "between tensorforce and tfagents their agents can not be instantiated in the"
            "same python runtime instance (conflicting excpectations on tensorflows eager execution mode)."
            )


class TfagentsNotRegisteredAgent(bcore.BackendAgent):

    def __init__(self, model_config: easyagents.core.ModelConfig, backend_name: str):
        raise NotImplementedError(
            "Do not call agents.register_tensorforce() before instantiating a tfagents based agent."
            "This agent is implemented by tfagents. Due to an incompatibility"
            "between tensorforce and tfagents their agents can not be instantiated in the"
            "same python runtime instance (conflicting excpectations on tensorflows eager execution mode)."
            )

class TfagentsDefaultNotRegisteredAgent(bcore.BackendAgent):

    def __init__(self, model_config: easyagents.core.ModelConfig, backend_name: str):
        raise NotImplementedError(
            "Set the backend='tensorforce' argument in the easyagents constructor call. "
            "This agents default implementation is implemented by tfagents. Due to an incompatibility"
            "between tensorforce and tfagents their agents can not be instantiated in the"
            "same python runtime instance (conflicting excpectations on tensorflows eager execution mode)."
            )


class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend which redirects all calls to the some default implementation."""

    def __init__(self, register_tensorforce: bool = False):
        self.register_tensorforce = register_tensorforce

    backend_name = 'default'

    def get_algorithms(self) -> Dict[Type, Type[easyagents.backends.core.BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        #                easyagents.agents.CemAgent: easyagents.backends.kerasrl.KerasRlCemAgent,
        #                easyagents.agents.DoubleDqnAgent: easyagents.backends.kerasrl.KerasRlDoubleDqnAgent,
        if self.register_tensorforce:
            import easyagents.backends.tforce

            result = {
                easyagents.agents.DqnAgent: TfagentsDefaultNotRegisteredAgent,
                easyagents.agents.DuelingDqnAgent: easyagents.backends.tforce.TforceDuelingDqnAgent,
                easyagents.agents.PpoAgent: TfagentsDefaultNotRegisteredAgent,
                easyagents.agents.RandomAgent: TfagentsDefaultNotRegisteredAgent,
                easyagents.agents.ReinforceAgent: TfagentsDefaultNotRegisteredAgent,
                easyagents.agents.SacAgent: TfagentsNotRegisteredAgent}
        else:
            import easyagents.backends.tfagents

            result = {
                easyagents.agents.DqnAgent: easyagents.backends.tfagents.TfDqnAgent,
                easyagents.agents.DuelingDqnAgent: TensorforceNotRegisteredAgent,
                easyagents.agents.PpoAgent: easyagents.backends.tfagents.TfPpoAgent,
                easyagents.agents.RandomAgent: easyagents.backends.tfagents.TfRandomAgent,
                easyagents.agents.ReinforceAgent: easyagents.backends.tfagents.TfReinforceAgent,
                easyagents.agents.SacAgent: easyagents.backends.tfagents.TfSacAgent}
        return result

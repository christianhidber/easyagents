"""This module contains the public api of the EasyAgents reinforcement learning library.

    It consist mainly of the class hierarchy of the available agents (algorithms), registrations and
    the management of the available backends. In their implementation the agents forward their calls
    to the chosen backend.
"""

from abc import ABC
from typing import Dict, List
from easyagents.core import ModelConfig, TrainCallback
from easyagents.backends.core import Backend
from easyagents.backends.default import DefaultBackend

_DFEAULT_BACKEND_NAME = 'default'

_backends: Dict[str, Backend] = {_DFEAULT_BACKEND_NAME: DefaultBackend()}


def get_backends():
    """returns a list of all registered backend identifiers."""
    return _backends.keys()


def register_backend(backend_name: str, backend: Backend):
    assert backend_name is not None, "backend_name not set"
    assert backend_name, "backend_name is empty"
    assert backend is not None, "backend not set"
    _backends[backend_name] = backend


class EasyAgent(ABC):
    """Abstract base class for all easy reinforcment learning agents.

        Implementations must set _backend_agent and _agent_config.

        Args:
            backend_name: the backend (implementation) to be used, if None the a default implementation is used
    """

    def __init__(self, backend_name: str = None):
        if backend_name is None:
            backend_name = _DFEAULT_BACKEND_NAME

        assert backend_name in get_backends(), \
            f'"{backend_name}" is not admissible. The registered backends are {get_backends()}.'

        self._agent_config = None
        self._backend = _backends[backend_name]
        self._backend_agent = None
        return

    def train(self, train: List[TrainCallback] = None):
        """Trains a new model using the gym environment passed during instantiation.

        Args:
            train: list of callbacks called during the train loop (but not during evaluation after each iteration)
            """
        self._backend_agent.train(train=train)
        return


class PpoAgent(EasyAgent):
    """creates a new agent based on the PPO algorithm.

        PPO is an actor-critic algorithm using 2 neural networks. The actor network
        to predict the next action to be taken and the critic network to estimate
        the value of the game state we are currently in (the expected, discounted
        sum of future rewards when following the current actor network).

        see also: https://spinningup.openai.com/en/latest/algorithms/ppo.html

        Args:
            gym_env_name: name of an OpenAI gym environment to be used for training and evaluation
            fc_layers: defines the neural network to be used, a sequence of fully connected
                layers of the given size. Eg (75,40) yields a neural network consisting
                out of 2 hidden layers, the first one containing 75 and the second layer
                containing 40 neurons.
            backend=the backend to be used (eg 'tfagents'), if None a default implementation is used.
                call get_backends() to get a list of the available backends.
    """

    def __init__(self, gym_env_name: str, fc_layers=None, backend_name: str = None):
        super().__init__(backend_name=backend_name)
        self._agent_config = ModelConfig(gym_env_name=gym_env_name, fc_layers=fc_layers)
        self._backendAgent = self._backend.create_ppo_agent(self._agent_config)
        return

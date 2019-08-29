"""This module contains the public api of the EasyAgents reinforcement learning library.

    It consist mainly of the class hierarchy of the available agents (algorithms), registrations and
    the management of the available backends. In their implementation the agents forward their calls
    to the chosen backend.
"""

from abc import ABC
from typing import Dict, List, Tuple, Optional
from easyagents import core
from easyagents.backends import core as bcore
import easyagents.backends.default
import easyagents.backends.tfagents

_backends: Dict[str, bcore.BackendAgentFactory] = {
    easyagents.backends.default.BackendAgentFactory.name: easyagents.backends.default.BackendAgentFactory(),
    easyagents.backends.tfagents.BackendAgentFactory.name: easyagents.backends.tfagents.BackendAgentFactory()
}


def get_backends():
    """returns a list of all registered backend identifiers."""
    return _backends.keys()


def register_backend(backend_name: str, backend: bcore.BackendAgentFactory):
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

    def __init__(self,
                 gym_env_name: str = None,
                 fc_layers: Tuple[int, ...] = None,
                 model_config: core.ModelConfig = None,
                 backend_name: str = None):

        if model_config is None:
            model_config = core.ModelConfig(gym_env_name=gym_env_name, fc_layers=fc_layers)
        if backend_name is None:
            backend_name = easyagents.backends.default.BackendAgentFactory.name

        assert model_config is not None, "model_config not set."
        assert backend_name in get_backends(), \
            f'{backend_name} is not admissible. The registered backends are {get_backends()}.'

        self._model_config: core.ModelConfig = model_config
        self._backend: bcore.BackendAgentFactory = _backends[backend_name]
        self._backend_agent: Optional[bcore._BackendAgent] = None
        return

    def train(self, train_context: core.TrainContext, callbacks: List[core.AgentCallback] = None):
        """Trains a new model using the gym environment passed during instantiation.

        Args:
            callbacks: list of callbacks called during the training and evaluation
            train_context: training configuration to be used (num_iterations,num_episodes_per_iteration,...)
        """
        assert train_context is not None
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks,list):
            assert isinstance(callbacks,core.AgentCallback), "callback not a AgentCallback or a list thereof."
            callbacks=[callbacks]
        self._backend_agent.train(train_context=train_context, callbacks=callbacks)


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

    def __init__(self,
                 gym_env_name: str,
                 fc_layers: Optional[Tuple[int, ...]] = None,

                 backend_name: str = None):
        super().__init__(gym_env_name=gym_env_name, fc_layers=fc_layers, backend_name=backend_name)
        self._backend_agent = self._backend.create_ppo_agent(self._model_config)
        return

    def train(self,
              callbacks: List[core.AgentCallback] = None,
              num_iterations: int = 1000,
              num_episodes_per_iteration: int = 10,
              max_steps_per_episode: int = 1000,
              num_epochs_per_iteration: int = 10,
              learning_rate: float = 1,
              train_context: core.TrainContext = None):
        """Trains a new model using the gym environment passed during instantiation.

        Args:
            callbacks: list of callbacks called during training and evaluation
            num_iterations: number of times the training is repeated (with additional data)
            num_episodes_per_iteration: number of episodes played per training iteration
            max_steps_per_episode: maximum number of steps per episode
            num_epochs_per_iteration: number of times the data collected for the current iteration
                is used to retrain the current policy.
            learning_rate: the learning rate used in the next iteration's policy training (0,1]

            train_context: training configuration to be used. if set overrides all other training context arguments.
        """
        if train_context is None:
            train_context = core.TrainContext()
            train_context.num_iterations = num_iterations
            train_context.num_episodes_per_iteration = num_episodes_per_iteration
            train_context.max_steps_per_episode = max_steps_per_episode
            train_context.num_epochs_per_iteration = num_epochs_per_iteration
            train_context.learning_rate = learning_rate

        super().train(train_context=train_context, callbacks=callbacks)

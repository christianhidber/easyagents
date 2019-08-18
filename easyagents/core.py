"""This module contains the core datastructures shared between fronten and backend like the definition
    of all callbacks and agent configurations.
"""

from abc import ABC, abstractmethod
from easyagents.env import _is_registered_with_gym


class AgentConfig(ABC):
    """Base class for all agent configurations, containing in particular the name of the gym environment.

    Derived classes may contain algorithm specific configuration data.
    Note that parameters which may change during training or evaluation should be placed in the
    TrainingContext and PlayContext respectively.

    Args:
        gym_env_name: the name of the registered gym environment to use, eg 'CartPole-v0'
        fc_layers: int tuple defining the number and size of each fully connected layer.
    """

    def __init__(self, gym_env_name: str, fc_layers=None):
        if fc_layers is None:
            fc_layers = (100, 100)
        if isinstance(fc_layers, int):
            fc_layers = (fc_layers,)

        assert isinstance(gym_env_name, str), "passed gym_env_name not a string."
        assert gym_env_name != "", "gym environment name is empty."
        assert _is_registered_with_gym(gym_env_name), \
            f'"{gym_env_name}" is not the name of an environment registered with OpenAI gym.' + \
            'Consider using easyagents.env.register_with_gym to register your environment.'
        assert fc_layers is not None, "fc_layers not set"
        assert isinstance(fc_layers, tuple), "fc_layers not a tuple"
        assert fc_layers, "fc_layers must contain at least 1 int"
        for i in fc_layers:
            assert isinstance(i, int) and i >= 1, f'{i} is not a valid size for a hidden layer'

        self.gym_env_name = gym_env_name
        self.fc_layers = fc_layers


class TrainCallback(ABC):
    """Base class for all callbacks controlling / monitoring the training loop of an agent"""

    @abstractmethod
    def on_train_begin(self):
        """Called once at the entry of an agent.train() call. """

    @abstractmethod
    def on_train_end(self):
        """Called once before exiting an agent.train() call"""

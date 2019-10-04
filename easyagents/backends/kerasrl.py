"""This module contains the backend implementation for keras-rl (see https://github.com/keras-rl/keras-rl)"""
from abc import ABCMeta
from typing import Dict, Type
import math

# noinspection PyUnresolvedReferences
import easyagents.agents
from easyagents import core
from easyagents.backends import core as bcore
from easyagents.backends import monitor

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

import gym
import gym.spaces


# noinspection PyUnresolvedReferences,PyAbstractClass
class KerasRlAgent(bcore.BackendAgent, metaclass=ABCMeta):
    """Reinforcement learning agents based on keras-rl originally developed by matthias plappert

        https://github.com/keras-rl/keras-rl
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    def _create_env(self):
        """Creates a new gym instance."""
        result = gym.make(self.model_config.gym_env_name)
        return result


class CemAgent(KerasRlAgent):
    """ creates a new agent based on the CEM algorithm using the keras-rl implementation.

        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf

         Args:
             model_config: the model configuration including the name of the target gym environment
                 as well as the neural network architecture.
     """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    def train_implementation(self, train_context: core.DqnTrainContext):
        train_env = self._create_env()

        assert isinstance(train_env,gym.spaces.Discrete), "Only discrete actions environment are supported."

        action_space : gym.spaces.Discrete = train_env.action_space

        memory = EpisodeParameterMemory(limit=train_context.max_steps_in_buffer, window_length=1)
        cem = CEMAgent(model=model, nb_actions=action_space.n, memory=memory,
                       batch_size=train_context.num_steps_sampled_from_buffer,
                       nb_steps_warmup=train_context.num_steps_buffer_preload,
                       train_interval=50,
                       elite_frac=0.05)
        cem.compile()
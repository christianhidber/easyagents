"""This module contains the backend implementation for tensorforce

    see https://github.com/tensorforce/tensorforce
"""
from abc import ABCMeta
from typing import List, Dict
import math

import gym
import gym.core
import gym.spaces
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# noinspection PyUnresolvedReferences
import easyagents
import easyagents.backends.core

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


class TensorforcePpoAgent(easyagents.backends.core.BackendAgent, metaclass=ABCMeta):
    """ Agent based on the PPO algorithm using the tensorforce implementation."""

    def __init__(self, model_config: easyagents.core.ModelConfig):
        """
        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
        """
        super().__init__(model_config=model_config)

    def _create_env(self) -> Environment:
        """Creates a tensorforce Environment encapsulating the underlying gym environment given in self.model_config"""
        result = Environment.create(environment='gym', level=self.model_config.gym_env_name)
        return result

    def _create_network_specification(self):
        """Creates a tensorforce network specification based on the layer specification given in self.model_config"""
        result: List[Dict] = []
        layer_sizes = self.model_config.fc_layers
        for layer_size in layer_sizes:
            result.append(dict(type='dense', size=layer_size, activation='relu'))
        return result

    def play_implementation(self, play_context: easyagents.core.PlayContext):
        """Agent specific implementation of playing a single episode with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."

        return
        if self._play_env is None:
            self._play_env = self._create_env()
        while True:
            self.on_play_episode_begin(env=self._play_env)
            state = self._play_env.reset()
            done = False
            while not done:
                action = None
                state, reward, done, _ = self._play_env.step(action)
            self.on_play_episode_end()
            if play_context.play_done:
                break
        return

    def train_implementation(self, train_context: easyagents.core.ActorCriticTrainContext):
        """Tensorforce Ppo Implementation of the train loop.

            The implementation follows https://github.com/tensorforce/tensorforce/blob/master/examples/quickstart.py
        """
        tc = train_context
        self.log('Creating Environment...')
        train_env = self._create_env()

        self.log('Creating network specification...')
        network = self._create_network_specification()

        self.log_api('Agent.create', "(agent='ppo',environment=...,network=...)")
        ppoAgent = Agent.create(
            agent='ppo',
            environment=train_env,
            network=network,
            learning_rate=tc.learning_rate,
            optimization_steps=tc.num_epochs_per_iteration,
            discount=tc.reward_discount_gamma,
            seed=self.model_config.seed
        )

        def callback(runner: Runner) -> bool:
            return True

        # Initialize the runner
        runner = Runner(agent=ppoAgent, environment=train_env)

        # Start the runner
        runner.run(num_episodes=tc.num_iterations * tc.num_episodes_per_iteration,
                   max_episode_timesteps=tc.max_steps_per_episode,
                   use_tqdm=False,
                   callback=callback
                   )
        # self.on_train_iteration_begin()
        # self.on_train_iteration_end(0)
        runner.close()


class BackendAgentFactory(easyagents.backends.core.BackendAgentFactory):
    """Backend for Tensorforce.

        Serves as a factory to create algorithm specific wrappers for the tensorforce implementations.
    """

    name: str = 'tensorforce'

    def create_dqn_agent(self, model_config: easyagents.core.ModelConfig) -> easyagents.backends.core._BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation."""
        return TensorforcePpoAgent(model_config=model_config)

"""This module contains the backend implementation for tensorforce

    see https://github.com/tensorforce/tensorforce
"""
from abc import ABCMeta
from typing import List, Dict, Optional, Type
import math
import os
import tempfile
import datetime

import gym
import easyagents.backends.core

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


class TforceAgent(easyagents.backends.core.BackendAgent, metaclass=ABCMeta):
    """ Base class for agents based on the tensorforce implementation."""

    def __init__(self, model_config: easyagents.core.ModelConfig):
        """
        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
        """
        super().__init__(model_config=model_config)
        self._agent: Optional[Agent] = None
        self._play_env: Optional[Environment] = None

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

    def _get_temp_path(self):
        result = os.path.join(tempfile.gettempdir(), tempfile.gettempprefix())
        n = datetime.datetime.now()
        result = result + f'-{n.year % 100:2}{n.month:02}{n.day:02}-{n.hour:02}{n.minute:02}{n.second:02}-' + \
                 f'{n.microsecond:06}'
        return result

    def _train_with_tensorforce_runner(self, train_env: Environment, train_context: easyagents.core.TrainContext):
        """Trains the self._agent using a tensorforce runner.

        Args:
            train_env: the tensorforce environment to use for the training
            train_context: context containing the training parameters
        """
        assert train_context
        assert train_env
        assert self._agent

        def train_callback(_: Runner) -> bool:
            result = not train_context.training_done
            return result

        def eval_callback(_: Runner) -> bool:
            result = not train_context.training_done
            if result:
                self.on_train_iteration_end(loss=math.nan, actor_loss=math.nan, critic_loss=math.nan)
                result = not train_context.training_done
                if result:
                    self.on_train_iteration_begin()
            return result

        # Initialize the runner
        self.log_api('Runner.create', "(agent=..., environment=...)")
        runner = Runner(agent=self._agent, environment=train_env)

        # Start the runner
        self.log_api('runner.run', f'(num_episodes=None, max_episode_timesteps={train_context.max_steps_per_episode})')
        self.on_train_iteration_begin()
        runner.run(num_episodes=None,
                   max_episode_timesteps=train_context.max_steps_per_episode,
                   use_tqdm=False,
                   callback=train_callback,
                   evaluation_callback=eval_callback,
                   evaluation_frequency=None,
                   evaluation=False,
                   num_evaluation_iterations=0
                   )
        if not train_context.training_done:
            self.on_train_iteration_end(loss=math.nan, actor_loss=math.nan, critic_loss=math.nan)
        runner.close()

    def play_implementation(self, play_context: easyagents.core.PlayContext):
        """Agent specific implementation of playing a single episode with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."
        assert self._agent, "agent not set. call train first."

        if self._play_env is None:
            self._play_env = self._create_env()
        while True:
            # noinspection PyUnresolvedReferences
            gym_env: gym.Env = self._play_env.environment
            self.on_play_episode_begin(env=gym_env)
            state = self._play_env.reset()
            done = False
            while not done:
                action = self._agent.act(state, evaluation=True)
                state, terminal, reward = self._play_env.execute(actions=action)
                if isinstance(terminal, bool):
                    done = terminal
                else:
                    done = terminal > 0
            self.on_play_episode_end()
            if play_context.play_done:
                break
        return


class TforcePpoAgent(TforceAgent):
    """ Agent based on the PPO algorithm using the tensorforce implementation."""

    def __init__(self, model_config: easyagents.core.ModelConfig):
        """
        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
        """
        super().__init__(model_config=model_config)

    def train_implementation(self, train_context: easyagents.core.ActorCriticTrainContext):
        """Tensorforce Ppo Implementation of the train loop.

            The implementation follows https://github.com/tensorforce/tensorforce/blob/master/examples/quickstart.py
        """
        tc = train_context
        self.log('Creating Environment...')
        train_env = self._create_env()

        self.log('Creating network specification...')
        network = self._create_network_specification()

        self.log_api('Agent.create', f'(agent="ppo", learning_rate={tc.learning_rate}, ' +
                     f'batch_size={tc.num_episodes_per_iteration}, ' +
                     f'optimization_steps={tc.num_epochs_per_iteration}, ' +
                     f'discount={tc.reward_discount_gamma},seed={self.model_config.seed})')
        tempdir = self._get_temp_path()
        self._agent = Agent.create(
            agent='ppo',
            environment=train_env,
            network=network,
            learning_rate=tc.learning_rate,
            batch_size=tc.num_episodes_per_iteration,
            optimization_steps=tc.num_epochs_per_iteration,
            discount=tc.reward_discount_gamma,
            seed=self.model_config.seed,
            summarizer=dict(directory=tempdir, labels=['losses']),
        )
        self._train_with_tensorforce_runner(train_env, tc)


class BackendAgentFactory(easyagents.backends.core.BackendAgentFactory):
    """Backend for Tensorforce.

        Serves as a factory to create algorithm specific wrappers for the tensorforce implementations.
    """

    name: str = 'tensorforce'

    def get_algorithms(self) -> Dict[Type, Type[easyagents.backends.core.BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        return {easyagents.agents.PpoAgent: TforcePpoAgent}

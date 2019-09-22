"""This module contains the backend implementation for huskarl

    see https://github.com/danaugrs/huskarl
"""
from abc import ABCMeta
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

import huskarl
import huskarl.core


class HuskarlAgentWrapper(huskarl.core.Agent):
    """Wrapper for huskarl agents to propagate callbacks to the easyagent infrastructure"""

    def __init__(self, hk_agent: huskarl.core.Agent, backend_agent: easyagents.backends.core.BackendAgent):
        assert hk_agent is not None
        assert backend_agent is not None
        self._hk_agent: huskarl.core.Agent = hk_agent
        self._backend_agent: easyagents.backends.core.BackendAgent = backend_agent

    def act(self, state, instance=0):
        return self._hk_agent.act(state, instance=instance)

    def push(self, transition, instance=0):
        return self._hk_agent.push(transition, instance=instance)

    def save(self, filename, overwrite=False):
        return self._hk_agent.save(filename, overwrite=overwrite)

    def train(self, step):
        result = self._hk_agent.train(step)
        " currently (19Q3) we don't know how to access to the loss => return nan. results in a text message in"
        " the plot.Loss plot."
        self._backend_agent.on_train_iteration_end(math.nan)
        self._backend_agent.on_train_iteration_begin()
        return result


class HuskarlDqnAgent(easyagents.backends.core.BackendAgent, metaclass=ABCMeta):
    """Reinforcement learning agents implemented in the huskarl.

        see https://github.com/danaugrs/huskarl
    """

    """ creates a new agent based on the DQN algorithm using the huskarl implementation.

        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
    """

    def __init__(self, model_config: easyagents.core.ModelConfig):
        super().__init__(model_config=model_config)
        self._hk_agent = None
        self._play_env = None

    def _create_env(self) -> gym.Env:
        return gym.make(self.model_config.gym_env_name)

    def _create_model(self, observation_space: gym.core.Space):
        activation = 'relu'
        layer_sizes = self.model_config.fc_layers
        result: Sequential = Sequential()
        result.add(Dense(layer_sizes[0], activation=activation, input_shape=observation_space.shape))
        for layer_size in layer_sizes[1:]:
            result.add(Dense(layer_size, activation=activation))
        return result

    def play_implementation(self, play_context: easyagents.core.PlayContext):
        """Agent specific implementation of playing a single episode with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."
        assert self._hk_agent, "huskarl agent not set. call train() first."

        if self._play_env is None:
            self._play_env = self._create_env()
        while True:
            self.on_play_episode_begin(env=self._play_env)
            state = self._play_env.reset()
            done = False
            while not done:
                action = self._hk_agent.act(state)
                state, reward, done, _ = self._play_env.step(action)
            self.on_play_episode_end()
            if play_context.play_done:
                break
        return

    def train_implementation(self, train_context: easyagents.core.DqnTrainContext):
        """Huskarl Dqn Implementation of the train loop.

        The implementation follows
        https://github.com/danaugrs/huskarl/blob/master/examples/dqn-cartpole.py
        """
        self.log('Creating environment...')
        train_env = self._create_env()
        observation_space = train_env.observation_space
        action_space = train_env.action_space

        assert isinstance(action_space, gym.spaces.Discrete)

        self.log('Creating model...')
        model = self._create_model(observation_space)

        self.log_api('agent.DQN', 'Create')
        optimizer = Adam(lr=train_context.learning_rate)
        self._hk_agent = huskarl.agent.DQN(model,
                                           actions=action_space.n,
                                           optimizer=optimizer,
                                           nsteps=2,
                                           memsize=train_context.max_steps_in_buffer,
                                           gamma=train_context.reward_discount_gamma,
                                           batch_size=train_context.num_steps_sampled_from_buffer)
        self._hk_agent = HuskarlAgentWrapper(self._hk_agent, self)

        self.log_api('Simulation', 'Create')
        sim = huskarl.Simulation(self._create_env, self._hk_agent)
        # huskarl collects env experiences inside sim.train but not insirde agent.train.
        # to monitor the episodes played we immediately start with an iteration.
        self.on_train_iteration_begin()
        sim.train(max_steps=train_context.num_iterations)
        self.on_train_iteration_end(0)


class BackendAgentFactory(easyagents.backends.core.BackendAgentFactory):
    """Backend for Huskarl.

        Serves as a factory to create algorithm specific wrappers for the huskarl implementations.
    """

    name: str = 'huskarl'

    def create_dqn_agent(self, model_config: easyagents.core.ModelConfig) -> easyagents.backends.core._BackendAgent:
        """Create an instance of DqnAgent wrapping this backends implementation."""
        return HuskarlDqnAgent(model_config=model_config)

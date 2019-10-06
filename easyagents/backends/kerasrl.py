"""This module contains the backend implementation for keras-rl (see https://github.com/keras-rl/keras-rl)"""
from abc import ABCMeta
from typing import Optional, Dict, Type
import math

# noinspection PyUnresolvedReferences
import easyagents.agents
from easyagents import core
from easyagents.backends import core as bcore

from keras.models import Sequential
from keras.layers import Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Dense

import rl.core
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.callbacks import Callback

import gym
import gym.spaces


# noinspection PyUnresolvedReferences,PyAbstractClass
class KerasRlAgent(bcore.BackendAgent, metaclass=ABCMeta):
    """Reinforcement learning agents based on keras-rl originally developed by matthias plappert

        https://github.com/keras-rl/keras-rl
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config, tensorflow_v2_eager=False)
        self._agent: Optional[rl.core.Agent] = None
        self._play_env: Optional[gym.Env] = None

    def _create_env(self) -> gym.Env:
        """Creates a new gym instance."""
        self.log_api(f'gym.make', f'("{self.model_config.original_env_name}")')
        result = gym.make(self.model_config.gym_env_name)
        return result

    def _create_model(self, gym_env: gym.Env) -> Sequential:
        """Creates a model consisting of  fully connected layers as given by self.model_config.fc_layers
        with relu as activation function.

        Args:
            gym_env: gym_env whose observation shape ofdefines the size of the input layer and
                whose action_space defines the size of the output layer.

        Returns:
            Keras Sequential model according to model_config.fc_layers
        """
        assert gym_env

        num_actions = gym_env.action_space.n
        self.log_api(f'Sequential', f'()')
        result = Sequential()
        input_shape = (1,) + gym_env.observation_space.shape
        self.log_api(f'model.add', f'(Flatten(input_shape={input_shape}))')
        result.add(Flatten(input_shape=input_shape))
        for layer_size in self.model_config.fc_layers:
            self.log_api(f'model.add', f'(Dense({layer_size}))')
            result.add(Dense(layer_size))
            self.log_api(f'model.add', f'(Activation("relu"))')
            result.add(Activation('relu'))
        self.log_api(f'model.add', f'(Dense({num_actions}))')
        result.add(Dense(num_actions))
        self.log_api(f'model.add', f'(Activation("linear"))')
        result.add(Activation('linear'))
        return result

    def play_implementation(self, play_context: core.PlayContext):
        """Agent specific implementation of playing a single episodes with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."
        assert self._agent, "_agent not set. call train() first."

        if self._play_env is None:
            self._play_env = self._create_env()
        while True:
            self.on_play_episode_begin(env=self._play_env)
            self.log_api('agent.test', f'(env=..., nb_episodes=1, nb_max_start_steps=0, start_step_policy=None)')
            self._agent.test(env=self._play_env, nb_episodes=1, visualize=False, nb_max_episode_steps=None,
                             nb_max_start_steps=0, start_step_policy=None, verbose=0)
            self.on_play_episode_end()
            if play_context.play_done:
                break


class KerasRlDqnAgent(KerasRlAgent):
    """Keras-rl implementation of the algorithm described in in Mnih (2013) and Mnih (2015).
        http://arxiv.org/pdf/1312.5602.pdf and http://arxiv.org/abs/1509.06461
        """

    class DQNAgentWrapper(DQNAgent):
        """Override of the KerasRl DqnAgennt instantiation due to a conflict with tensorflow 1.15.

        Essentially a copy of  https://raw.githubusercontent.com/keras-rl/keras-rl/master/rl/agents/dqn.py
        """

        def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                     dueling_type='avg', *args, **kwargs):
            super(DQNAgent, self).__init__(*args, **kwargs)

            if model.output._keras_shape != (None, self.nb_actions):
                raise ValueError(f'Model output "{model.output}" has invalid shape. Dqn expects ' +
                                 f'a model that has one dimension for each action, in this case {self.nb_actions}.')

            self.enable_double_dqn = enable_double_dqn
            self.enable_dueling_network = enable_dueling_network
            self.dueling_type = dueling_type
            if self.enable_dueling_network:
                layer = model.layers[-2]
                nb_action = model.output._keras_shape[-1]
                y = Dense(nb_action + 1, activation='linear')(layer.output)
                if self.dueling_type == 'avg':
                    outputlayer = Lambda(
                        lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True),
                        output_shape=(nb_action,))(y)
                elif self.dueling_type == 'max':
                    outputlayer = Lambda(
                        lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True),
                        output_shape=(nb_action,))(y)
                elif self.dueling_type == 'naive':
                    outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
                else:
                    assert False, "dueling_type must be one of {'avg','max','naive'}"
                model = Model(inputs=model.input, outputs=outputlayer)

            self.model = model
            if policy is None:
                policy = EpsGreedyQPolicy()
            if test_policy is None:
                test_policy = GreedyQPolicy()
            self.policy = policy
            self.test_policy = test_policy
            self.reset_states()

    class DqnCallback(rl.callbacks.Callback):
        """Callback registered with keras rl agents to propagate iteration and episode updates."""

        def __init__(self, agent: bcore.BackendAgent, dqn_context: core.DqnTrainContext,
                     loss_metric_idx: Optional[int]):
            """
            Args:
                agent: the agent to propagate iteration begn/end events to.
                dqn_context: the train_context containing the iteration definitions
                loss_metric_idx: the index of the loss in the metrics list, or None
            """
            assert agent
            assert dqn_context
            self._agent = agent
            self._dqn_context = dqn_context
            self._loss_metric_idx = loss_metric_idx
            super().__init__()

        def on_step_end(self, step, logs=None):
            """Signals the base class the end / begin of a training iteration."""
            if self._dqn_context.steps_done_in_training % self._dqn_context.num_steps_per_iteration == 0:
                loss = math.nan
                if logs and 'metrics' in logs and (self._loss_metric_idx is not None):
                    metrics = logs['metrics']
                    if len(metrics) > self._loss_metric_idx:
                        loss = metrics[self._loss_metric_idx]
                self._agent.on_train_iteration_end(loss)
                if not self._dqn_context.training_done:
                    self._agent.on_train_iteration_begin()

    def __init__(self, model_config: core.ModelConfig):
        """ creates a new agent based on the DQN algorithm using the keras-rl implementation.

            Args:
                model_config: the model configuration including the name of the target gym environment
                    as well as the neural network architecture.
        """
        super().__init__(model_config=model_config)

    def train_implementation(self, train_context: core.DqnTrainContext):
        assert train_context
        dc: core.DqnTrainContext = train_context
        train_env = self._create_env()
        keras_model = self._create_model(train_env)
        self.log_api(f'SequentialMemory', f'(limit={dc.max_steps_in_buffer}, window_length=1)')
        memory = SequentialMemory(limit=dc.max_steps_in_buffer, window_length=1)
        self.log_api(f'BoltzmannQPolicy', f'()')
        policy = BoltzmannQPolicy()
        num_actions = train_env.action_space.n
        self.log_api(f'DQNAgent', f'(nb_actions={num_actions}, ' +
                     f'nb_steps_warmup={dc.num_steps_buffer_preload}, target_model_update=1e-2,' +
                     f'gamma={dc.reward_discount_gamma}, batch_size={dc.num_steps_sampled_from_buffer}, ' +
                     f'train_interval={dc.num_steps_per_iteration}, model=..., memory=..., policy=...)')
        self._agent = KerasRlDqnAgent.DQNAgentWrapper(
            model=keras_model,
            nb_actions=num_actions,
            memory=memory,
            nb_steps_warmup=dc.num_steps_buffer_preload,
            target_model_update=1e-2,
            gamma=dc.reward_discount_gamma,
            batch_size=dc.num_steps_sampled_from_buffer,
            train_interval=dc.num_steps_per_iteration,
            policy=policy)
        self.log_api(f'agent.compile', f'(Adam(lr=1e-3), metrics=["mae"]')
        self._agent.compile(Adam(lr=1e-3), metrics=['mae'])
        num_steps = dc.num_iterations * dc.num_steps_per_iteration

        loss_metric_idx = None
        if 'loss' in self._agent.metrics_names:
            loss_metric_idx = self._agent.metrics_names.index("loss")
        dqn_callback = KerasRlDqnAgent.DqnCallback(self, dc, loss_metric_idx)
        self.on_train_iteration_begin()
        self.log_api(f'agent.fit', f'(train_env, nb_steps={num_steps})')
        self._agent.fit(train_env, nb_steps=num_steps, visualize=False, verbose=0, callbacks=[dqn_callback])
        if not dc.training_done:
            self.on_train_iteration_end(math.nan)


class CemKerasRlAgent(KerasRlAgent):
    """ creates a new agent based on the CEM algorithm using the keras-rl implementation.

        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf

         Args:
             model_config: the model configuration including the name of the target gym environment
                 as well as the neural network architecture.
     """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    def train_implementation(self, train_context: core.DqnTrainContext):
        """
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
        """


class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend for TfAgents.

        Serves as a factory to create algorithm specific wrappers for the keras-rl implementations.
    """

    name: str = 'kerasrl'

    tensorflow_v2_eager_compatible: bool = False

    def get_algorithms(self) -> Dict[Type, Type[easyagents.backends.core.BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        return {easyagents.agents.DqnAgent: KerasRlDqnAgent}

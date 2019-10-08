"""This module contains the backend implementation for keras-rl (see https://github.com/keras-rl/keras-rl)"""
from abc import ABCMeta
from typing import Optional, Dict, Type
import math

# noinspection PyUnresolvedReferences
import easyagents.agents
from easyagents import core
from easyagents.backends import core as bcore

import keras.backend as K
from keras.layers import Activation, Flatten, Lambda, Dense
from keras.models import Sequential, Model
from keras.optimizers import Adam

import rl.core
from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.callbacks import Callback
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import EpisodeParameterMemory, SequentialMemory

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

    def _create_model(self, gym_env: gym.Env, activation: str) -> Sequential:
        """Creates a model consisting of  fully connected layers as given by self.model_config.fc_layers
        with relu as activation function.

        Args:
            gym_env: gym_env whose observation shape ofdefines the size of the input layer and
                whose action_space defines the size of the output layer.
            activation: output activation function eg 'linear' or 'softmax'

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
        self.log_api(f'model.add', f'(Activation("{activation}"))')
        result.add(Activation(activation))
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
            observation = self._play_env.reset()
            done = False
            while not done:
                action = self._agent.forward(observation)
                observation, reward, done, info = self._play_env.step(action)
            self.on_play_episode_end()
            if play_context.play_done:
                break


class KerasRlCemAgent(KerasRlAgent):
    """Keras-rl implementation of the cross-entropy method algorithm.

        see "https://learning.mpi-sws.org/mlss2016/slides/2016-MLSS-RL.pdf" and
            "https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6579&rep=rep1&type=pdf"
    """

    class CemCallback(rl.callbacks.Callback):
        """Callback registered with keras rl agents to propagate iteration and episode updates."""

        def __init__(self, cem_agent: KerasRlAgent, cem_context: core.CemTrainContext, nb_steps: int):
            """
            Args:
                cem_agent: the agent to propagate iteration begn/end events to.
                cem_context: the train_context containing the iteration definitions
                nb_steps: value set in the keras cem agent.
            """
            assert cem_agent
            assert cem_context
            assert nb_steps
            self._cem_agent: KerasRlAgent = cem_agent
            self._cem_context: core.CemTrainContext = cem_context
            self._nb_steps = nb_steps
            super().__init__()

        def on_episode_end(self, episode, logs=None):
            """Signals the base class the end / begin of a training iteration."""
            cc: core.CemTrainContext = self._cem_context
            episode = episode + 1
            if episode % cc.num_episodes_per_iteration == 0:
                self._cem_agent.on_train_iteration_end(math.nan)
                if self._cem_context.training_done:
                    self._cem_agent._agent.step = self._nb_steps
                else:
                    self._cem_agent.on_train_iteration_begin()

    def train_implementation(self, train_context: core.CemTrainContext):
        assert train_context
        cc: core.CemTrainContext = train_context
        train_env = self._create_env()
        keras_model = self._create_model(gym_env=train_env, activation='softmax')

        policy_buffer_size = 5 * cc.num_episodes_per_iteration
        self.log_api(f'EpisodeParameterMemory', f'(limit={policy_buffer_size}, window_length=1)')
        memory = EpisodeParameterMemory(limit=policy_buffer_size, window_length=1)
        num_actions = train_env.action_space.n
        self.log_api(f'CEMAgent', f'(model=..., nb_actions={num_actions}, memory=..., ' + \
                     f'nb_steps_warmup={cc.num_steps_warmup}, ' + \
                     f'train_interval={cc.num_episodes_per_iteration}, ' + \
                     f'batch_size={cc.num_episodes_per_iteration}, ' + \
                     f'elite_frac={cc.elite_set_fraction})')
        self._agent = CEMAgent(model=keras_model, nb_actions=num_actions, memory=memory,
                               nb_steps_warmup=cc.num_steps_warmup,
                               batch_size=cc.num_episodes_per_iteration,
                               train_interval=cc.num_episodes_per_iteration,
                               elite_frac=cc.elite_set_fraction)
        self.log_api(f'agent.compile', '()')
        self._agent.compile()
        nb_steps = cc.num_iterations * cc.num_episodes_per_iteration * cc.max_steps_per_episode
        callback = KerasRlCemAgent.CemCallback(self, cc, nb_steps)
        self.on_train_iteration_begin()
        self.log_api(f'agent.fit', f'(train_env, nb_steps={nb_steps})')
        self._agent.fit(train_env, nb_steps=nb_steps, visualize=False, verbose=0, callbacks=[callback])
        if not cc.training_done:
            self.on_train_iteration_end(math.nan)


class KerasRlDqnAgent(KerasRlAgent):
    """Keras-rl implementation of the algorithm described in in Mnih (2013) and Mnih (2015).
        http://arxiv.org/pdf/1312.5602.pdf and http://arxiv.org/abs/1509.06461

        includes implementations for the double dqn and dueling dqn variations.
        """

    class DQNAgentWrapper(DQNAgent):
        """Override of the KerasRl DqnAgennt instantiation due to a conflict with tensorflow 1.15.

        Essentially a copy of  https://raw.githubusercontent.com/keras-rl/keras-rl/master/rl/agents/dqn.py
        """

        def __init__(self, model, policy=None, test_policy=None, enable_double_dqn=False, enable_dueling_network=False,
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
            steps_done = self._dqn_context.steps_done_in_training - self._dqn_context.num_steps_buffer_preload
            if steps_done > 0 and steps_done % self._dqn_context.num_steps_per_iteration == 0:
                loss = math.nan
                if logs and 'metrics' in logs and (self._loss_metric_idx is not None):
                    metrics = logs['metrics']
                    if len(metrics) > self._loss_metric_idx:
                        loss = metrics[self._loss_metric_idx]
                self._agent.on_train_iteration_end(loss)
                if not self._dqn_context.training_done:
                    self._agent.on_train_iteration_begin()

    def __init__(self, model_config: core.ModelConfig,
                 enable_dueling_dqn: bool = False, enable_double_dqn=False):
        """ creates a new agent based on the DQN algorithm using the keras-rl implementation.

            Args:
                model_config: the model configuration including the name of the target gym environment
                    as well as the neural network architecture.
                enable_double_dqn: use the double dqn algorithm instead
                enable_dueling_dqn: use the dueling dqn algorithm instead
        """
        super().__init__(model_config=model_config)
        self._enable_double_dqn: bool = enable_double_dqn
        self._enable_dueling_network: bool = enable_dueling_dqn

    def train_implementation(self, train_context: core.DqnTrainContext):
        assert train_context
        dc: core.DqnTrainContext = train_context
        train_env = self._create_env()
        keras_model = self._create_model(gym_env=train_env, activation='linear')
        self.log_api(f'SequentialMemory', f'(limit={dc.max_steps_in_buffer}, window_length=1)')
        memory = SequentialMemory(limit=dc.max_steps_in_buffer, window_length=1)
        self.log_api(f'BoltzmannQPolicy', f'()')
        policy = BoltzmannQPolicy()
        num_actions = train_env.action_space.n
        self.log_api(f'DQNAgent', f'(nb_actions={num_actions}, ' +
                     f'enable_double_dqn={self._enable_double_dqn}, ' +
                     f'enable_dueling_network={self._enable_dueling_network}, ' +
                     f'nb_steps_warmup={dc.num_steps_buffer_preload}, target_model_update=1e-2,' +
                     f'gamma={dc.reward_discount_gamma}, batch_size={dc.num_steps_sampled_from_buffer}, ' +
                     f'train_interval={dc.num_steps_per_iteration}, model=..., memory=..., policy=...)')
        self._agent = KerasRlDqnAgent.DQNAgentWrapper(
            enable_double_dqn=self._enable_double_dqn,
            enable_dueling_network=self._enable_dueling_network,
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


class KerasRlDoubleDqnAgent(KerasRlDqnAgent):
    """Keras-rl implementation of the algorithm described in https://arxiv.org/abs/1509.06461 """

    def __init__(self, model_config: core.ModelConfig):
        """Args:
                model_config: the model configuration including the name of the target gym environment
                    as well as the neural network architecture.
        """
        super().__init__(model_config=model_config, enable_double_dqn=True)


class KerasRlDuelingDqnAgent(KerasRlDqnAgent):
    """Keras-rl implementation of the algorithm described in https://arxiv.org/abs/1511.06581 """

    def __init__(self, model_config: core.ModelConfig):
        """ creates a new agent based on the DQN algorithm using the keras-rl implementation.

            Args:
                model_config: the model configuration including the name of the target gym environment
                    as well as the neural network architecture.
                enable_double_dqn:
        """
        super().__init__(model_config=model_config, enable_dueling_dqn=True)


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
        return {
            easyagents.agents.CemAgent: KerasRlCemAgent,
            easyagents.agents.DqnAgent: KerasRlDqnAgent,
            easyagents.agents.DoubleDqnAgent: KerasRlDoubleDqnAgent,
            easyagents.agents.DuelingDqnAgent: KerasRlDuelingDqnAgent,
        }

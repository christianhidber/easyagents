"""This module contains the backend implementation for tf Agents (see https://github.com/tensorflow/agents)"""
from abc import ABCMeta
from typing import Dict, Type
import math
import os

# noinspection PyUnresolvedReferences
import easyagents.agents
from easyagents import core
from easyagents.backends import core as bcore
from easyagents.backends import monitor

# noinspection PyPackageRequirements
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import gym_wrapper, py_environment, tf_py_environment
from tf_agents.networks import actor_distribution_network, normal_projection_network, q_network, value_network
from tf_agents.policies import greedy_policy, tf_policy, random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import gym


# noinspection PyUnresolvedReferences,PyAbstractClass
class TfAgent(bcore.BackendAgent, metaclass=ABCMeta):
    """Reinforcement learning agents based on googles tf_agent implementations

        https://github.com/tensorflow/agents
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config, backend_name=TfAgentAgentFactory.backend_name)
        self._trained_policy = None
        self._play_env: Optional[gym.Env] = None

    def _create_gym_with_wrapper(self, discount):
        gym_spec = gym.spec(self.model_config.gym_env_name)
        gym_env = gym_spec.make()

        # simplify_box_bounds: Whether to replace bounds of Box space that are arrays
        #  with identical values with one number and rely on broadcasting.
        # important, simplify_box_bounds True crashes environments with boundaries with identical values
        env = gym_wrapper.GymWrapper(
            gym_env,
            discount=discount,
            simplify_box_bounds=False)
        return env

    def _create_env(self, discount: float = 1) -> tf_py_environment.TFPyEnvironment:
        """ creates a new instance of the gym environment and wraps it in a tfagent TFPyEnvironment

            Args:
                discount: the reward discount factor
        """
        assert 0 < discount <= 1, "discount not admissible"

        self.log_api(f'TFPyEnvironment', f'( suite_gym.load( ... ) )')
        # suit_gym.load crashes our environment 
        # py_env = suite_gym.load(self.model_config.gym_env_name, discount=discount)
        py_env = self._create_gym_with_wrapper(discount)

        result = tf_py_environment.TFPyEnvironment(py_env)
        return result

    def _get_gym_env(self, tf_py_env: tf_py_environment.TFPyEnvironment) -> monitor._MonitorEnv:
        """ extracts the underlying _MonitorEnv from tf_py_env created by _create_tfagent_env"""
        assert isinstance(tf_py_env, tf_py_environment.TFPyEnvironment), \
            "passed tf_py_env is not an instance of TFPyEnvironment"
        assert isinstance(tf_py_env.pyenv, py_environment.PyEnvironment), \
            "passed TFPyEnvironment.pyenv does not contain a PyEnvironment"
        assert len(tf_py_env.pyenv.envs) == 1, "passed TFPyEnvironment.pyenv does not contain a unique environment"

        result = tf_py_env.pyenv.envs[0].gym
        assert isinstance(result, monitor._MonitorEnv), "passed TFPyEnvironment does not contain a _MonitorEnv"
        return result

    def play_implementation(self, play_context: core.PlayContext):
        """Agent specific implementation of playing a single episodes with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."
        assert self._trained_policy, "trained_policy not set. call train() first."

        if self._play_env is None:
            self._play_env = self._create_env()
        gym_env = self._get_gym_env(self._play_env)
        while True:
            self.on_play_episode_begin(env=gym_env)
            time_step = self._play_env.reset()
            while not time_step.is_last():
                action_step = self._trained_policy.action(time_step)
                time_step = self._play_env.step(action_step.action)
            self.on_play_episode_end()
            if play_context.play_done:
                break


    def load_implementation(self, directory: str):
        """Loads a previously saved actor policy from the directory

        Args:
             directory: the directory to load the policy from.
        """
        assert directory

        self.log_api('saved_model.load', f'({directory})')
        self._trained_policy = tf.compat.v2.saved_model.load(directory)


    def save_implementation(self, directory: str):
        """Saves the trained actor policy in directory.
           If no policy was trained yet, no file is written
.
        Args:
             directory: the directory to save the policy weights to.
        """
        assert self._trained_policy, "no policy trained yet."

        self.log_api('PolicySaver', f'(trained_policy,seed={self.model_config.seed})')
        saver = policy_saver.PolicySaver(self._trained_policy, seed=self.model_config.seed)
        self.log_api('policy_saver.save', f'({directory})')
        saver.save(directory)


# noinspection PyUnresolvedReferences
class TfDqnAgent(TfAgent):
    """ creates a new agent based on the DQN algorithm using the tfagents implementation.

        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    def collect_step(self, env: tf_py_environment.TFPyEnvironment, policy: tf_policy.Base,
                     replay_buffer: TFUniformReplayBuffer):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        replay_buffer.add_batch(traj)

    # noinspection DuplicatedCode
    def train_implementation(self, train_context: core.TrainContext):
        """Tf-Agents Ppo Implementation of the train loop.

        The implementation follows
        https://colab.research.google.com/github/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb
        """
        assert isinstance(train_context, core.StepsTrainContext)
        dc: core.StepsTrainContext = train_context

        train_env = self._create_env(discount=dc.reward_discount_gamma)
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and DqnAgent
        self.log_api('AdamOptimizer', '()')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=dc.learning_rate)
        self.log_api('QNetwork', '()')
        q_net = q_network.QNetwork(observation_spec, action_spec, fc_layer_params=self.model_config.fc_layers)
        self.log_api('DqnAgent', '()')
        tf_agent = dqn_agent.DqnAgent(timestep_spec, action_spec,
                                      q_network=q_net, optimizer=optimizer,
                                      td_errors_loss_fn=common.element_wise_squared_loss)

        self.log_api('tf_agent.initialize', f'()')
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        self.log_api('TFUniformReplayBuffer', '()')
        replay_buffer = TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
                                              batch_size=train_env.batch_size,
                                              max_length=dc.max_steps_in_buffer)
        self.log_api('RandomTFPolicy', '()')
        random_policy = random_tf_policy.RandomTFPolicy(timestep_spec, action_spec)
        self.log_api('replay_buffer.add_batch', '(trajectory)')
        for _ in range(dc.num_steps_buffer_preload):
            self.collect_step(env=train_env, policy=random_policy, replay_buffer=replay_buffer)

        # Train
        tf_agent.train = common.function(tf_agent.train, autograph=False)
        self.log_api('replay_buffer.as_dataset', f'(num_parallel_calls=3, ' +
                     f'sample_batch_size={dc.num_steps_sampled_from_buffer}, num_steps=2).prefetch(3)')
        dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=dc.num_steps_sampled_from_buffer,
                                           num_steps=2).prefetch(3)
        iter_dataset = iter(dataset)
        self.log_api('for each iteration')
        self.log_api('  replay_buffer.add_batch', '(trajectory)')
        self.log_api('  tf_agent.train', '(experience=trajectory)')
        while True:
            self.on_train_iteration_begin()
            for _ in range(dc.num_steps_per_iteration):
                self.collect_step(env=train_env, policy=tf_agent.collect_policy, replay_buffer=replay_buffer)
            trajectories, _ = next(iter_dataset)
            tf_loss_info = tf_agent.train(experience=trajectories)
            self.on_train_iteration_end(tf_loss_info.loss)
            if train_context.training_done:
                break
        return


# noinspection PyUnresolvedReferences
class TfPpoAgent(TfAgent):
    """ creates a new agent based on the PPO algorithm using the tfagents implementation.
        PPO is an actor-critic algorithm using 2 neural networks. The actor network
        to predict the next action to be taken and the critic network to estimate
        the value of the game state we are currently in (the expected, discounted
        sum of future rewards when following the current actor network).

        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    # noinspection DuplicatedCode
    def train_implementation(self, train_context: core.TrainContext):
        """Tf-Agents Ppo Implementation of the train loop."""

        assert isinstance(train_context, core.PpoTrainContext)
        tc: core.PpoTrainContext = train_context
        train_env = self._create_env(discount=tc.reward_discount_gamma)
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and PpoAgent
        self.log_api('AdamOptimizer', '()')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=tc.learning_rate)

        self.log_api('ActorDistributionNetwork', '()')
        actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec, action_spec,
                                                                        fc_layer_params=self.model_config.fc_layers)
        self.log_api('ValueNetwork', '()')
        value_net = value_network.ValueNetwork(observation_spec, fc_layer_params=self.model_config.fc_layers)

        self.log_api('PpoAgent', '()')
        tf_agent = ppo_agent.PPOAgent(timestep_spec, action_spec, optimizer,
                                      actor_net=actor_net, value_net=value_net,
                                      num_epochs=tc.num_epochs_per_iteration)
        self.log_api('tf_agent.initialize', '()')
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        collect_data_spec = tf_agent.collect_data_spec
        self.log_api('TFUniformReplayBuffer', '()')
        replay_buffer = TFUniformReplayBuffer(collect_data_spec,
                                              batch_size=1, max_length=tc.max_steps_in_buffer)

        collect_policy = tf_agent.collect_policy
        self.log_api('DynamicEpisodeDriver', '()')
        collect_driver = DynamicEpisodeDriver(train_env, collect_policy, observers=[replay_buffer.add_batch],
                                              num_episodes=tc.num_episodes_per_iteration)

        # Train
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        tf_agent.train = common.function(tf_agent.train, autograph=False)

        while True:
            self.on_train_iteration_begin()
            self.log_api('-----', f'iteration {tc.iterations_done_in_training:4} of {tc.num_iterations:<4}      -----')
            self.log_api('collect_driver.run', '()')
            collect_driver.run()

            self.log_api('replay_buffer.gather_all', '()')
            trajectories = replay_buffer.gather_all()

            self.log_api('tf_agent.train', '(experience=...)')
            loss_info = tf_agent.train(experience=trajectories)
            total_loss = loss_info.loss.numpy()
            actor_loss = loss_info.extra.policy_gradient_loss.numpy()
            critic_loss = loss_info.extra.value_estimation_loss.numpy()
            self.log_api('', f'loss={total_loss:<7.1f} [actor={actor_loss:<7.1f} critic={critic_loss:<7.1f}]')

            self.log_api('replay_buffer.clear', '()')
            replay_buffer.clear()

            self.on_train_iteration_end(loss=total_loss, actor_loss=actor_loss, critic_loss=critic_loss)
            if tc.training_done:
                break
        return


# noinspection PyUnresolvedReferences
class TfRandomAgent(TfAgent):
    """ creates a new random agent based on uniform random actions.

        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)
        self._set_trained_policy()

    def _set_trained_policy(self):
        """Tf-Agents Random Implementation of the train loop."""
        self.log('Creating environment...')
        train_env = self._create_env()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        self.log_api('RandomTFPolicy', 'create')
        self._trained_policy = random_tf_policy.RandomTFPolicy(timestep_spec, action_spec)
        self._agent_context._is_policy_trained = True

    def load_implementation(self, directory: str):
        """NoOps implementation, since we don't save/load random policies."""
        pass

    def save_implementation(self, directory: str):
        """NoOps implementation, since we don't save/load random policies."""
        pass

    # noinspection DuplicatedCode
    def train_implementation(self, train_context: core.TrainContext):
        self.log("Training...")
        train_env = self._create_env()
        while True:
            self.on_train_iteration_begin()
            # ensure that 1 episode is played during the iteration
            time_step = train_env.reset()
            while not time_step.is_last():
                action_step = self._trained_policy.action(time_step)
                time_step = train_env.step(action_step.action)
            self.on_train_iteration_end(math.nan)
            if train_context.training_done:
                break
        return


# noinspection PyUnresolvedReferences
class TfReinforceAgent(TfAgent):
    """ creates a new agent based on the Reinforce algorithm using the tfagents implementation.
        Reinforce is a vanilla policy gradient algorithm using a single neural networks to predict
        the actions.

        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    # noinspection DuplicatedCode
    def train_implementation(self, train_context: core.TrainContext):
        """Tf-Agents Reinforce Implementation of the train loop."""

        assert isinstance(train_context, core.EpisodesTrainContext)
        tc: core.EpisodesTrainContext = train_context
        self.log('Creating environment...')
        train_env = self._create_env(discount=tc.reward_discount_gamma)
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and PpoAgent
        self.log_api('AdamOptimizer', 'create')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=tc.learning_rate)

        self.log_api('ActorDistributionNetwork', 'create')
        actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec, action_spec,
                                                                        fc_layer_params=self.model_config.fc_layers)

        self.log_api('ReinforceAgent', 'create')
        tf_agent = reinforce_agent.ReinforceAgent(timestep_spec, action_spec, actor_network=actor_net,
                                                  optimizer=optimizer)

        self.log_api('tf_agent.initialize()')
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        collect_data_spec = tf_agent.collect_data_spec
        self.log_api('TFUniformReplayBuffer', 'create')
        replay_buffer = TFUniformReplayBuffer(collect_data_spec, batch_size=1, max_length=tc.max_steps_in_buffer)
        self.log_api('DynamicEpisodeDriver', 'create')
        collect_driver = DynamicEpisodeDriver(train_env, tf_agent.collect_policy,
                                              observers=[replay_buffer.add_batch],
                                              num_episodes=tc.num_episodes_per_iteration)

        # Train
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        tf_agent.train = common.function(tf_agent.train, autograph=False)

        self.log('Starting training...')
        while True:
            self.on_train_iteration_begin()
            msg = f'iteration {tc.iterations_done_in_training:4} of {tc.num_iterations:<4}'
            self.log_api('collect_driver.run', msg)
            collect_driver.run()

            self.log_api('replay_buffer.gather_all', msg)
            trajectories = replay_buffer.gather_all()

            self.log_api('tf_agent.train', msg)
            loss_info = tf_agent.train(experience=trajectories)
            total_loss = loss_info.loss.numpy()
            self.log_api('', f'loss={total_loss:<7.1f}')

            self.log_api('replay_buffer.clear', msg)
            replay_buffer.clear()

            self.on_train_iteration_end(loss=total_loss)
            if tc.training_done:
                break
        return


# noinspection PyUnresolvedReferences
class TfSacAgent(TfAgent):
    """ creates a new agent based on the SAC algorithm using the tfagents implementation.

        adapted from
            https://github.com/tensorflow/agents/blob/master/tf_agents/colabs/7_SAC_minitaur_tutorial.ipynb

        Args:
            model_config: the model configuration including the name of the target gym environment
                as well as the neural network architecture.
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)

    # noinspection DuplicatedCode
    def train_implementation(self, train_context: core.TrainContext):
        """Tf-Agents Ppo Implementation of the train loop."""

        assert isinstance(train_context, core.StepsTrainContext)
        tc: core.StepsTrainContext = train_context
        train_env = self._create_env(discount=tc.reward_discount_gamma)
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        self.log_api('CriticNetwork',
                     f'(observation_spec, action_spec), observation_fc_layer_params=None, ' +
                     f'action_fc_layer_params=None, joint_fc_layer_params={self.model_config.fc_layers})')
        critic_net = critic_network.CriticNetwork((observation_spec, action_spec),
                                                  observation_fc_layer_params=None, action_fc_layer_params=None,
                                                  joint_fc_layer_params=self.model_config.fc_layers)

        def normal_projection_net(action_spec_arg, init_means_output_factor=0.1):
            return normal_projection_network.NormalProjectionNetwork(action_spec_arg,
                                                                     mean_transform=None,
                                                                     state_dependent_std=True,
                                                                     init_means_output_factor=init_means_output_factor,
                                                                     std_transform=sac_agent.std_clip_transform,
                                                                     scale_distribution=True)

        self.log_api('ActorDistributionNetwork',
                     f'observation_spec, action_spec, fc_layer_params={self.model_config.fc_layers}), ' +
                     f'continuous_projection_net=...)')
        actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec, action_spec,
                                                                        fc_layer_params=self.model_config.fc_layers,
                                                                        continuous_projection_net=normal_projection_net)
        # self.log_api('tf.compat.v1.train.get_or_create_global_step','()')
        # global_step = tf.compat.v1.train.get_or_create_global_step()
        self.log_api('SacAgent', f'(timestep_spec, action_spec, actor_network=..., critic_network=..., ' +
                     f'actor_optimizer=AdamOptimizer(learning_rate={tc.learning_rate}), ' +
                     f'critic_optimizer=AdamOptimizer(learning_rate={tc.learning_rate}), ' +
                     f'alpha_optimizer=AdamOptimizer(learning_rate={tc.learning_rate}), ' +
                     f'gamma={tc.reward_discount_gamma})')
        tf_agent = sac_agent.SacAgent(
            timestep_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=tc.learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=tc.learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=tc.learning_rate),
            # target_update_tau=0.005,
            # target_update_period=1,
            # td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
            gamma=tc.reward_discount_gamma)
        # reward_scale_factor=1.0,
        # gradient_clipping=None,
        # train_step_counter=global_step)
        self.log_api('tf_agent.initialize', '()')
        tf_agent.initialize()

        self._trained_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
        collect_policy = tf_agent.collect_policy

        # setup and preload replay buffer
        self.log_api('TFUniformReplayBuffer', f'(data_spec=tf_agent.collect_data_spec, ' +
                     f'batch_size={train_env.batch_size}, max_length={tc.max_steps_in_buffer})')
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
                                                                       batch_size=train_env.batch_size,
                                                                       max_length=tc.max_steps_in_buffer)

        self.log_api('DynamicStepDriver', f'(env, collect_policy, observers=[replay_buffer.add_batch], ' +
                     f'num_steps={tc.num_steps_buffer_preload})')
        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(train_env,
                                                                       collect_policy,
                                                                       observers=[replay_buffer.add_batch],
                                                                       num_steps=tc.num_steps_buffer_preload)
        self.log_api('initial_collect_driver.run()')
        initial_collect_driver.run()

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=tc.num_steps_sampled_from_buffer,
                                           num_steps=2).prefetch(3)
        iterator = iter(dataset)

        self.log_api('DynamicStepDriver', f'(env, collect_policy, observers=[replay_buffer.add_batch], ' +
                     f'num_steps={tc.num_steps_per_iteration})')
        collect_driver = dynamic_step_driver.DynamicStepDriver(train_env,
                                                               collect_policy,
                                                               observers=[replay_buffer.add_batch],
                                                               num_steps=tc.num_steps_per_iteration)

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        tf_agent.train = common.function(tf_agent.train)
        collect_driver.run = common.function(collect_driver.run)

        self.log_api('for each iteration')
        self.log_api('  collect_driver.run', '()')
        self.log_api('  tf_agent.train', '(experience=...)')
        while True:
            self.on_train_iteration_begin()
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(tc.num_steps_per_iteration):
                collect_driver.run()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(iterator)
            loss_info = tf_agent.train(experience)
            total_loss = loss_info.loss.numpy()
            actor_loss = loss_info.extra.actor_loss
            alpha_loss = loss_info.extra.alpha_loss
            critic_loss = loss_info.extra.critic_loss
            self.on_train_iteration_end(loss=total_loss, actor_loss=actor_loss, critic_loss=critic_loss,
                                        alpha_loss=alpha_loss)
            if tc.training_done:
                break
        return


class TfAgentAgentFactory(bcore.BackendAgentFactory):
    """Backend for TfAgents.

        Serves as a factory to create algorithm specific wrappers for the TfAgents implementations.
    """

    backend_name: str = 'tfagents'

    def get_algorithms(self) -> Dict[Type, Type[easyagents.backends.core.BackendAgent]]:
        """Yields a mapping of EasyAgent types to the implementations provided by this backend."""
        return {easyagents.agents.DqnAgent: TfDqnAgent,
                easyagents.agents.PpoAgent: TfPpoAgent,
                easyagents.agents.RandomAgent: TfRandomAgent,
                easyagents.agents.ReinforceAgent: TfReinforceAgent,
                easyagents.agents.SacAgent: TfSacAgent}

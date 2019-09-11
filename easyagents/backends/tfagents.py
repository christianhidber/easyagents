"""This module contains the backend implementation for tf Agents (see https://github.com/tensorflow/agents)"""
from abc import ABCMeta

# noinspection PyUnresolvedReferences
import easyagents.agents
from easyagents import core
from easyagents.backends import core as bcore
from easyagents.backends import monitor

# noinspection PyPackageRequirements
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import tf_policy, random_tf_policy
from tf_agents.trajectories import trajectory


# noinspection PyUnresolvedReferences,PyAbstractClass
class TfAgent(bcore.BackendAgent, metaclass=ABCMeta):
    """Reinforcement learning agents based on googles tf_agent implementations

        https://github.com/tensorflow/agents
    """

    def __init__(self, model_config: core.ModelConfig):
        super().__init__(model_config=model_config)
        self._initialize()
        self._trained_policy = None
        self._eval_env = None

    def _initialize(self):
        """ initializes TensorFlow behaviour and random seeds."""
        self.log_api('v1.enable_v2_behavior')
        tf.compat.v1.enable_v2_behavior()
        self.log_api('v1.enable_eager_execution')
        tf.compat.v1.enable_eager_execution()
        if self.model_config.seed:
            self.log_api(f'v1.set_random_seed({self.model_config.seed})')
            tf.compat.v1.set_random_seed(self.model_config.seed)
        return

    def _create_tfagent_env(self, discount: float = 1) -> tf_py_environment.TFPyEnvironment:
        """ creates a new instance of the gym environment and wraps it in a tfagent TFPyEnvironment

            Args:
                discount: the reward discount factor
        """
        assert 0 < discount <= 1, "discount not admissible"

        self.log_api(f'creating TFPyEnvironment( suite_gym.load( ... ) )')
        py_env = suite_gym.load(self.model_config.gym_env_name, discount=discount)
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
        """Agent specific implementation of playing a single episode with the current policy.

            Args:
                play_context: play configuration to be used
        """
        assert play_context, "play_context not set."
        assert self._trained_policy, "trained_policy not set. call train() first."

        if self._eval_env is None:
            self._eval_env = self._create_tfagent_env()
        gym_env = self._get_gym_env(self._eval_env)
        while True:
            self.on_play_episode_begin(env=gym_env)
            time_step = self._eval_env.reset()
            while not time_step.is_last():
                action_step = self._trained_policy.action(time_step)
                time_step = self._eval_env.step(action_step.action)
            self.on_play_episode_end()
            if play_context.play_done:
                break


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
        assert isinstance(train_context, core.DqnTrainContext)
        dc: core.DqnTrainContext = train_context

        self.log('Creating environment...')
        train_env = self._create_tfagent_env(discount=dc.reward_discount_gamma)
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and DqnAgent
        self.log_api('AdamOptimizer', 'create')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=dc.learning_rate)
        self.log_api('QNetwork', 'create')
        q_net = q_network.QNetwork(observation_spec, action_spec, fc_layer_params=self.model_config.fc_layers)
        self.log_api('DqnAgent', 'create')
        tf_agent = dqn_agent.DqnAgent(timestep_spec, action_spec,
                                      q_network=q_net, optimizer=optimizer,
                                      td_errors_loss_fn=common.element_wise_squared_loss)

        self.log_api('tf_agent.initialize()')
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        self.log_api('TFUniformReplayBuffer', 'create')
        replay_buffer = TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
                                              batch_size=train_env.batch_size,
                                              max_length=dc.max_steps_in_buffer)
        self.log_api('RandomTFPolicy', 'create')
        self.log("Preloading replay buffer...")
        random_policy = random_tf_policy.RandomTFPolicy(timestep_spec, action_spec)
        for _ in range(dc.num_steps_buffer_preload):
            self.collect_step(env=train_env, policy=random_policy, replay_buffer=replay_buffer)

        # Train
        tf_agent.train = common.function(tf_agent.train, autograph=False)
        self.log_api('replay_buffer.as_dataset', 'create')
        dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=dc.num_steps_sampled_from_buffer,
                                           num_steps=2).prefetch(3)
        iter_dataset = iter(dataset)
        self.log("Training...")
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

        assert isinstance(train_context, core.ActorCriticTrainContext)
        tc: core.ActorCriticTrainContext = train_context
        self.log('Creating environment...')
        train_env = self._create_tfagent_env(discount=tc.reward_discount_gamma)
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and PpoAgent
        self.log_api('AdamOptimizer', 'create')
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=tc.learning_rate)

        self.log_api('ActorDistributionNetwork', 'create')
        actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec, action_spec,
                                                                        fc_layer_params=self.model_config.fc_layers)
        self.log_api('ValueNetwork', 'create')
        value_net = value_network.ValueNetwork(observation_spec, fc_layer_params=self.model_config.fc_layers)

        self.log_api('PpoAgent', 'create')
        tf_agent = ppo_agent.PPOAgent(timestep_spec, action_spec, optimizer,
                                      actor_net=actor_net, value_net=value_net,
                                      num_epochs=tc.num_epochs_per_iteration)
        self.log_api('tf_agent.initialize()')
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        collect_data_spec = tf_agent.collect_data_spec
        self.log_api('TFUniformReplayBuffer', 'create')
        replay_buffer = TFUniformReplayBuffer(collect_data_spec,
                                              batch_size=1, max_length=tc.max_steps_in_buffer)

        collect_policy = tf_agent.collect_policy
        self.log_api('DynamicEpisodeDriver', 'create')
        collect_driver = DynamicEpisodeDriver(train_env, collect_policy, observers=[replay_buffer.add_batch],
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
            actor_loss = loss_info.extra.policy_gradient_loss.numpy()
            critic_loss = loss_info.extra.value_estimation_loss.numpy()
            self.log_api('', f'loss={total_loss:<7.1f} [actor={actor_loss:<7.1f} critic={critic_loss:<7.1f}]')

            self.log_api('replay_buffer.clear', msg)
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

    def _set_trained_policy(self):
        """Tf-Agents Random Implementation of the train loop."""
        self.log('Creating environment...')
        train_env = self._create_tfagent_env()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Data collection & Buffering
        self.log_api('RandomTFPolicy', 'create')
        self._trained_policy = random_tf_policy.RandomTFPolicy(timestep_spec, action_spec)

    # noinspection DuplicatedCode
    def train_implementation(self, train_context: core.TrainContext):
        self.log("Training...")
        while True:
            self.on_train_iteration_begin()
            # ensure that 1 episode is played during the iteration
            time_step = train_env.reset()
            while not time_step.is_last():
                action_step = self._trained_policy.action(time_step)
                time_step = train_env.step(action_step.action)
            self.on_train_iteration_end(0)
            if train_context.training_done:
                break
        return


class BackendAgentFactory(bcore.BackendAgentFactory):
    """Backend for TfAgents.

        Serves as a factory to create algorithm specific wrappers for the TfAgents implementations.
    """

    name: str = 'tfagents'

    def create_dqn_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        """Create an instance of DqnAgent wrapping this backends implementation."""
        return TfDqnAgent(model_config=model_config)

    def create_ppo_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        """Create an instance of PpoAgent wrapping this backends implementation."""
        return TfPpoAgent(model_config=model_config)

    def create_random_agent(self, model_config: core.ModelConfig) -> bcore._BackendAgent:
        """Create an instance of RandomAgent wrapping this backends implementation."""
        return TfRandomAgent(model_config=model_config)

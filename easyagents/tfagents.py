import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import q_network
from tf_agents.networks import value_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from easyagents.agents import AbstractAgent
from easyagents.config import Logging
from easyagents.config import Training
from easyagents.easyenv import EasyEnv

class TfAgent(AbstractAgent):
    """ Reinforcement learning agents based on googles tf_agent implementations
        https://github.com/tensorflow/agents
    """

    def __init__(self,
                 gym_env_name: str,
                 training: Training = None,
                 fc_layers=None,
                 reward_discount_gamma: float = 1,
                 learning_rate: float = 0.001,
                 logging: Logging = None
                 ):
        super().__init__(gym_env_name=gym_env_name,
                         training=training,
                         fc_layers=fc_layers,
                         reward_discount_gamma=reward_discount_gamma,
                         learning_rate=learning_rate,
                         logging=logging)
        self.__initialize()
        self._trained_policy = None
        self._gym_eval_env = None
        return

    def __initialize(self):
        """ initializes TensorFlow behaviour and random seeds.
        """
        self._gym_eval_env = None
        self._log_agent("executing: tf.compat.v1.enable_v2_behavior()")
        tf.compat.v1.enable_v2_behavior()
        self._log_agent("executing: tf.enable_eager_execution()")
        tf.compat.v1.enable_eager_execution()
        self._log_agent("executing: tf.compat.v1.set_random_seed(0)")
        tf.compat.v1.set_random_seed(0)
        return

    def _create_tfagent_env(self) -> tf_py_environment.TFPyEnvironment:
        """ creates a new instance of the gym environment and wraps it in a tfagent TFPyEnvironment
        """
        self._log_agent("   executing tf_py_environment.TFPyEnvironment( suite_gym.load )")
        py_env = suite_gym.load(self._gym_env_name,
                                discount=self._reward_discount_gamma,
                                max_episode_steps=self._training.max_steps_per_episode)
        result = tf_py_environment.TFPyEnvironment(py_env)
        return result

    def _get_easyenv(self, tf_py_env: tf_py_environment.TFPyEnvironment) -> EasyEnv:
        """ extracts the underlying EasyEnv from tf_py_env created by _create_tfagent_env
        """
        assert isinstance(tf_py_env,
                          tf_py_environment.TFPyEnvironment), "passed tf_py_env is not an instance of TFPyEnvironment"
        assert isinstance(tf_py_env.pyenv,
                          py_environment.PyEnvironment), "passed TFPyEnvironment.pyenv does not contain a PyEnvironment"
        assert len(tf_py_env.pyenv.envs) == 1, "passed TFPyEnvironment.pyenv does not contain a unique environment"

        result = tf_py_env.pyenv.envs[0]._env.gym
        assert isinstance(result, EasyEnv), "passed TFPyEnvironment does not contain a EasyEnv"
        return result

    def _play_episode(self, max_steps: int = None, callback=None) -> (float, int, bool):
        """ Plays a full episode using the previously trained policy, yielding
            the sum of rewards, the totale number of steps taken over the episode.

            Args:
            max_steps   : if the episode is not done after max_steps it is aborted.
            callback    : callback(action,state,reward,step,done,info) is called after each step.
                          if the callback yields True, the episode is aborted.
            :returns rewards,steps
        """
        assert self._trained_policy is not None, "policy not yet trained. call train() first."
        assert max_steps == None or max_steps > 0, "max_steps must be > 0"

        if self._gym_eval_env is None:
            self._gym_eval_env = self._create_tfagent_env()

        easy_env = self._get_easyenv(self._gym_eval_env)
        sum_rewards = 0.0
        step_count = 0
        time_step = self._gym_eval_env.reset()
        if callback is not None:
            easy_env._set_step_callback(callback)
        while not time_step.is_last():
            action_step = self._trained_policy.action(time_step)
            time_step = self._gym_eval_env.step(action_step.action)
            sum_rewards += time_step.reward
            step_count += 1
            if step_count == max_steps:
                break
        easy_env._set_step_callback(None)
        return float(sum_rewards), step_count


class PpoAgent(TfAgent):
    """ creates a new agent based on the PPO algorithm using the tfagents implementation.
        PPO is an actor-critic algorithm using 2 neural networks. The actor network
        to predict the next action to be taken and the critic network to estimate
        the value of the game state we are currently in (the expected, discounted
        sum of future rewards when following the current actor network).

        Args:
        gym_env_name    :   name of an OpenAI gym environment to be used for training and evaluation
        fc_layers       :   defines the neural network to be used, a sequence of fully connected
                            layers of the given size. Eg (75,40) yields a neural network consisting
                            out of 2 hidden layers, the first one containing 75 and the second layer
                            containing 40 neurons.
        training                : instance of config.Training to configure the #episodes used for training.
        num_training_steps_in_replay_buffer : defines the size of the replay buffer in terms of game steps.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and
                                  the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.
        logging                 : instance of config.Logging to define the logging behaviour
        num_training_steps_in_replay_buffer : size of the replay buffer

        see also: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(self,
                 gym_env_name: str,
                 fc_layers=None,
                 training: Training = None,
                 num_training_steps_in_replay_buffer: int = 10001,
                 learning_rate: float = 0.001,
                 reward_discount_gamma: float = 1,
                 logging: Logging = None):
        super().__init__(gym_env_name=gym_env_name,
                         fc_layers=fc_layers,
                         training=training,
                         learning_rate=learning_rate,
                         reward_discount_gamma=reward_discount_gamma,
                         logging=logging)
        assert num_training_steps_in_replay_buffer >= 1, "num_training_steps_in_replay_buffer must be >= 1"
        self._num_training_steps_in_replay_buffer = num_training_steps_in_replay_buffer
        return

    def _train(self):
        """ trains a policy using the gym_env.
            Sets training_losses and training_average_returns, depending on the training scheme
            defined in Training configuration.
        """
        # Create Training Environment
        self._log_agent("Creating environment:")
        train_env = self._create_tfagent_env()
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and PpoAgent
        self._log_agent("Creating agent:")
        self._log_agent("  creating  tf.compat.v1.train.AdamOptimizer( ... )")
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._learning_rate)

        actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec,
                                                                        action_spec,
                                                                        fc_layer_params=self.fc_layers)
        value_net = value_network.ValueNetwork(observation_spec, fc_layer_params=self.fc_layers)

        self._log_agent("  creating  PpoAgent( ... )")
        tf_agent = ppo_agent.PPOAgent(timestep_spec, action_spec, optimizer,
                                      actor_net=actor_net,
                                      value_net=value_net,
                                      num_epochs=self._training.num_epochs_per_iteration)
        self._log_agent("  executing tf_agent.initialize()")
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        self._log_agent("Creating data collection:")
        collect_data_spec = tf_agent.collect_data_spec
        self._log_agent("  creating TFUniformReplayBuffer()")
        replay_buffer = TFUniformReplayBuffer(collect_data_spec,
                                              batch_size=1,
                                              max_length=self._num_training_steps_in_replay_buffer)

        collect_policy = tf_agent.collect_policy
        self._log_agent("  creating DynamicEpisodeDriver()")
        collect_driver = DynamicEpisodeDriver(train_env,
                                              collect_policy,
                                              observers=[replay_buffer.add_batch],
                                              num_episodes=self._training.num_episodes_per_iteration)

        # Train
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        tf_agent.train = common.function(tf_agent.train, autograph=False)

        self._log_agent("Starting training:")
        self._train_iteration_completed(0)
        for iteration in range(1, self._training.num_iterations + 1):
            msg = f'training {iteration:4} of {self._training.num_iterations:<4}:'
            self._log_agent(msg + " executing collect_driver.run()")
            collect_driver.run()

            self._log_agent(msg + " executing replay_buffer.gather_all()")
            trajectories = replay_buffer.gather_all()

            self._log_agent(msg + " executing tf_agent.train(...)")
            total_loss, _ = tf_agent.train(experience=trajectories)

            self._log_agent(msg + " executing replay_buffer.clear()")
            replay_buffer.clear()

            self._train_iteration_completed(iteration, total_loss)
        return


class DqnAgent(TfAgent):
    """ creates a new agent based on the DQN algorithm using the tfagents implementation.
        DQN stands for deep Q-network and was developed by deepmind combining deep neural
        networks with reinforcement learning.

        Args:
        gym_env_name                : name of an OpenAI gym environment to be used for training and evaluation
        fc_layers                   : defines the neural network to be used, a sequence of fully connected
                                      layers of the given size. Eg (75,40) yields a neural network consisting
                                      out of 2 hidden layers, the first one containing 75 and the second layer
                                      containing 40 neurons.
        training                    : instance of config.Training to configure the #episodes used for training.
        learning_rate               : value in (0,1]. Factor by which the impact on the policy update is reduced
                                      for each training step. The same learning rate is used for the value and
                                      the policy network.
        reward_discount_gamma       : value in (0,1]. Factor by which a future reward is discounted for each step.
        num_steps_in_replay_buffer  : size of the replay buffer. If None
                                      Training.max_steps_per_episode * Training.num_episodes_per_iteration
        num_episodes_to_preload_replay_buffer:  num of episodes to play to initially fill the replay buffer. If None
                                                Training.num_episodes_per_iteration
        num_steps_in_replay_batch   : num of steps sampled from replay buffer for each optimizer call. If None
                                      Training.max_steps_per_episode
        logging                     : instance of config.Logging to define the logging behaviour

        see also: https://deepmind.com/research/dqn/
    """

    def __init__(self,
                 gym_env_name: str,
                 fc_layers=None,
                 training: Training = None,
                 learning_rate: float = 0.001,
                 reward_discount_gamma: float = 1,
                 num_steps_in_replay_buffer: int = None,
                 num_episodes_to_preload_replay_buffer: int = None,
                 num_steps_in_replay_batch: int = None,
                 logging: Logging = None):
        super().__init__(gym_env_name=gym_env_name,
                         fc_layers=fc_layers,
                         training=training,
                         learning_rate=learning_rate,
                         reward_discount_gamma=reward_discount_gamma,
                         logging=logging)
        assert num_steps_in_replay_buffer is None or num_steps_in_replay_buffer >= 1, \
            "num_training_steps_in_replay_buffer must be >= 1"
        assert num_episodes_to_preload_replay_buffer is None or num_episodes_to_preload_replay_buffer >= 0, \
            "num_episodes_to_preload_replay_buffer must be >= 0"
        assert num_steps_in_replay_batch is None or num_steps_in_replay_batch >= 1, \
            "num_steps_in_replay_batch must be >= 1"

        self._num_steps_in_replay_buffer = self._training.max_steps_per_episode * \
                                           self._training.num_episodes_per_iteration
        if num_steps_in_replay_buffer is not None:
            self._num_steps_in_replay_buffer = num_steps_in_replay_buffer

        self._num_episodes_to_preload_replay_buffer = self._training.num_episodes_per_iteration
        if num_episodes_to_preload_replay_buffer is not None:
            self._num_episodes_to_preload_replay_buffer = num_episodes_to_preload_replay_buffer

        self._num_steps_in_replay_batch = self._training.max_steps_per_episode
        if num_steps_in_replay_batch is not None:
            self._num_steps_in_replay_batch = num_steps_in_replay_batch
        return

    def _train(self):
        """ trains a policy using the gym_env."""
        # Create Training Environment
        self._log_agent("Creating environment:")
        train_env = self._create_tfagent_env()

        # SetUp Optimizer, Networks and DqnAgent
        self._log_agent("Creating agent:")
        self._log_agent("  creating  tf.compat.v1.train.AdamOptimizer( ... )")
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._log_agent("  creating  QNetwork( ... )")
        q_net = q_network.QNetwork(train_env.observation_spec(),
                                   train_env.action_spec(),
                                   fc_layer_params=self.fc_layers)
        self._log_agent("  creating  DqnAgent( ... )")
        tf_agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                                      train_env.action_spec(),
                                      q_network=q_net,
                                      optimizer=optimizer,
                                      td_errors_loss_fn=common.element_wise_squared_loss)
        self._log_agent("  executing tf_agent.initialize()")
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        self._log_agent("Creating data collection:")
        self._log_agent("  creating  TFUniformReplayBuffer( ... )")
        replay_buffer = TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
                                              batch_size=1,
                                              max_length=self._num_steps_in_replay_buffer)
        self._log_agent("  creating  DynamicEpisodeDriver(RandomTFPolicy)")
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
        preload_driver = DynamicEpisodeDriver(env=train_env,
                                              policy=random_policy,
                                              observers=[replay_buffer.add_batch],
                                              num_episodes=self._num_episodes_to_preload_replay_buffer)

        self._log_agent("  executing DynamicEpisodeDriver.run (preloading replay buffer)")
        preload_driver.run()

        self._log_agent(f'  creating  DynamicEpisodeDriver(trained_policy, ' +
                        f'{self._training.num_episodes_per_iteration} episodes/iteration)')
        driver = DynamicEpisodeDriver(env=train_env,
                                      policy=self._trained_policy,
                                      observers=[replay_buffer.add_batch],
                                      num_episodes=self._training.num_episodes_per_iteration)
        # Train
        self._log_agent("Starting training:")
        tf_agent.train = common.function(tf_agent.train, autograph=False)
        dataset = replay_buffer.as_dataset(num_parallel_calls=1,
                                           sample_batch_size=self._num_steps_in_replay_batch,
                                           num_steps=2).prefetch(3)
        iter_dataset = iter(dataset)
        self._train_iteration_completed(0)
        for iteration in range(1, self._training.num_iterations + 1):
            msg = f'iteration {iteration:4} of {self._training.num_iterations:<4}:'
            self._log_agent(f'{msg} executing DynamicEpisodeDriver.run() (collecting data for ' +
                            f'{self._training.num_episodes_per_iteration} episodes)')
            driver.run()

            tf_lossInfo = None
            for t in range(1,self._training.num_epochs_per_iteration+1):
                trajectories, _ = next(iter_dataset)
                self._log_agent(f'{msg} batch {t:2} of {self._training.num_epochs_per_iteration:<2}' +
                                f'    executing  tf_agent.train({len(trajectories.action)} steps from replay_buffer)')
                tf_lossInfo = tf_agent.train(experience=trajectories)
            self._train_iteration_completed(iteration, tf_lossInfo.loss)
        return

from tf_agents.networks.network import DistributionNetwork
from tf_agents.networks.actor_distribution_network import _categorical_projection_net, _normal_projection_net
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
        
# most of the initial code copied from https://github.com/tensorflow/agents/blob/master/tf_agents/networks/actor_distribution_network.py         
class CustomActorDistributionNetwork(DistributionNetwork):
  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               hidden_layers,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               name='CustomActorDistributionNetwork'):
    if len(tf.nest.flatten(input_tensor_spec)) > 1:
      raise ValueError('Only a single observation is supported by this network')

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(CustomActorDistributionNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._mlp_layers = hidden_layers
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observations, step_type, network_state):
    del step_type  # unused.
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    observations = tf.nest.flatten(observations)
    states = tf.cast(observations[0], tf.float32)

    # Reshape to only a single batch dimension for neural network functions.
    batch_squash = utils.BatchSquash(outer_rank)
    states = batch_squash.flatten(states)

    for layer in self._mlp_layers:
      states = layer(states)

    # TODO(oars): Can we avoid unflattening to flatten again
    states = batch_squash.unflatten(states)
    output_actions = tf.nest.map_structure(
        lambda proj_net: proj_net(states, outer_rank),
        self._projection_networks)
    return output_actions, network_state

class ReinforceAgent(TfAgent):
    """ creates a new agent based on the Reinforce algorithm using the tfagents implementation.
        Reinforce is also known as Vanilla Policy Gradient as it is the most basic policy gradient approach.

        Args:
        gym_env_name    :   name of an OpenAI gym environment to be used for training and evaluation
        fc_layers       :   defines the neural network to be used, a sequence of fully connected
                            layers of the given size. Eg (75,40) yields a neural network consisting
                            out of 2 hidden layers, the first one containing 75 and the second layer
                            containing 40 neurons.
        custom_hidden_layers : alternative to the fc_layers parameter, accepts a list of tensorflow layers
                               serving as the hidden layers of the actor network. Expects to have 1d output,
                               so CNN output needs to be flattened as the final layer.                            
        training                : instance of config.Training to configure the #episodes used for training.
        num_training_steps_in_replay_buffer : defines the size of the replay buffer in terms of game steps.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and
                                  the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.
        logging                 : instance of config.Logging to define the logging behaviour
        num_training_steps_in_replay_buffer : size of the replay buffer

        see also: https://spinningup.openai.com/en/latest/algorithms/vpg.html
    """

    def __init__(self,
                 gym_env_name: str,
                 fc_layers=None,
                 custom_hidden_layers=None,
                 training: Training = None,
                 num_training_steps_in_replay_buffer: int = 10001,
                 learning_rate: float = 0.001,
                 reward_discount_gamma: float = 1,
                 logging: Logging = None):
        super().__init__(gym_env_name=gym_env_name,
                         fc_layers=fc_layers,
                         training=training,
                         learning_rate=learning_rate,
                         reward_discount_gamma=reward_discount_gamma,
                         logging=logging)
        assert num_training_steps_in_replay_buffer >= 1, "num_training_steps_in_replay_buffer must be >= 1"
        self._num_training_steps_in_replay_buffer = num_training_steps_in_replay_buffer
        self.custom_hidden_layers = custom_hidden_layers
        return

    def _train(self):
        """ trains a policy using the gym_env.
            Sets training_losses and training_average_returns, depending on the training scheme
            defined in Training configuration.
        """
        # Create Training Environment
        self._log_agent("Creating environment:")
        train_env = self._create_tfagent_env()
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        # SetUp Optimizer, Networks and PpoAgent
        self._log_agent("Creating agent:")
        self._log_agent("  creating  tf.compat.v1.train.AdamOptimizer( ... )")
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._learning_rate)

        if self.custom_hidden_layers is not None:
            actor_net = CustomActorDistributionNetwork(observation_spec,
                                                       action_spec,
                                                       self.custom_hidden_layers)
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(observation_spec,
                                                                            action_spec,
                                                                            fc_layer_params=self.fc_layers)

        self._log_agent("  creating  ReinforceAgent( ... )")
        tf_agent = reinforce_agent.ReinforceAgent(timestep_spec, action_spec,
                                                  actor_network=actor_net,
                                                  optimizer=optimizer)
        self._log_agent("  executing tf_agent.initialize()")
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # SetUp Data collection & Buffering
        self._log_agent("Creating data collection:")
        collect_data_spec = tf_agent.collect_data_spec
        self._log_agent("  creating TFUniformReplayBuffer()")
        replay_buffer = TFUniformReplayBuffer(collect_data_spec,
                                              batch_size=1,
                                              max_length=self._num_training_steps_in_replay_buffer)

        collect_policy = tf_agent.collect_policy
        self._log_agent("  creating DynamicEpisodeDriver()")
        collect_driver = DynamicEpisodeDriver(train_env,
                                              collect_policy,
                                              observers=[replay_buffer.add_batch],
                                              num_episodes=self._training.num_episodes_per_iteration)

        # Train
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        tf_agent.train = common.function(tf_agent.train, autograph=False)

        self._log_agent("Starting training:")
        self._train_iteration_completed(0)
        for iteration in range(1, self._training.num_iterations + 1):
            msg = f'training {iteration:4} of {self._training.num_iterations:<4}:'
            self._log_agent(msg + " executing collect_driver.run()")
            collect_driver.run()

            self._log_agent(msg + " executing replay_buffer.gather_all()")
            trajectories = replay_buffer.gather_all()

            self._log_agent(msg + " executing tf_agent.train(...)")
            total_loss, _ = tf_agent.train(experience=trajectories)

            self._log_agent(msg + " executing replay_buffer.clear()")
            replay_buffer.clear()

            self._train_iteration_completed(iteration, total_loss)
        return

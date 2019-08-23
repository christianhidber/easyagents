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
from easyagents.easyenv import _EasyEnv
from easyagents.tfagents import CustomActorDistributionNetwork



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
        py_env = suite_gym.load("Orso-v1")
        train_env = tf_py_environment.TFPyEnvironment(py_env)

        action_spec = train_env.action_spec()
        observation_spec=train_env.observation_spec()
        time_spec=train_env.time_step_spec()
        q_net = q_network.QNetwork(observation_spec,action_spec(), fc_layer_params=(500,500))

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._learning_rate)
        tf_agent = dqn_agent.DqnAgent(time_spec, action_spec, q_network=q_net, optimizer=optimizer)
        tf_agent.initialize()

        replay_buffer = TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,batch_size=64,max_length=100000)
        random_policy = random_tf_policy.RandomTFPolicy(time_spec, action_spec)
        preload_driver = DynamicEpisodeDriver(env=train_env, policy=random_policy, observers=[replay_buffer.add_batch],
                                              num_episodes=1000)
        preload_driver.run()

        driver = DynamicEpisodeDriver(env=train_env, policy=self._trained_policy, observers=[replay_buffer.add_batch])
        tf_agent.train = common.function(tf_agent.train, autograph=False)
        dataset = replay_buffer.as_dataset(num_parallel_calls=1,sample_batch_size=self._num_steps_in_replay_batch,
                                           num_steps=2).prefetch(3)
        iter_dataset = iter(dataset)
        self._train_iteration_completed(0)
        for iteration in range(1, self._training.num_iterations + 1):
            driver.run()

            tf_lossInfo = None
            for t in range(1,self._training.num_epochs_per_iteration+1):
                trajectories, _ = next(iter_dataset)
                tf_lossInfo = tf_agent.train(experience=trajectories)
            self._train_iteration_completed(iteration, tf_lossInfo.loss)
        return
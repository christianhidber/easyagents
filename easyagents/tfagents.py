import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common

from easyagents.agents import EasyAgent
from easyagents.easyenv import EasyEnv
from easyagents.config import TrainingDuration
from easyagents.config import Logging


class TfAgent(EasyAgent):
    """ Reinforcement learning agents based on googles tf_agent implementations
        https://github.com/tensorflow/agents
    """
    def __init__(   self,
                    gym_env_name : str,
                    training_duration : TrainingDuration = None,
                    fc_layers = None,
                    reward_discount_gamma : float = 1,
                    learning_rate : float = 0.001,
                    logging : Logging = None
                ):
        super().__init__(   gym_env_name=gym_env_name,
                            training_duration=training_duration,
                            fc_layers=fc_layers,
                            reward_discount_gamma=reward_discount_gamma,
                            learning_rate=learning_rate,
                            logging = logging )
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
        py_env = suite_gym.load( self._gym_env_name,
                                 discount=self._reward_discount_gamma,
                                 max_episode_steps=self._training_duration.max_steps_per_episode  )
        result = tf_py_environment.TFPyEnvironment( py_env)
        return result

    def _get_EasyEnv(self, tf_py_env: tf_py_environment.TFPyEnvironment ) -> EasyEnv:
        """ extracts the underlying EasyEnv from tf_py_env created by _create_tfagent_env
        """
        assert isinstance(tf_py_env, tf_py_environment.TFPyEnvironment), "passed tf_py_env is not an instance of TFPyEnvironment"
        assert isinstance(tf_py_env.pyenv, py_environment.PyEnvironment), "passed TFPyEnvironment.pyenv does not contain a PyEnvironment"
        assert len(tf_py_env.pyenv.envs) == 1, "passed TFPyEnvironment.pyenv does not contain a unique environment"

        result = tf_py_env.pyenv.envs[0]._env.gym
        assert isinstance(result, EasyEnv), "passed TFPyEnvironment does not contain a EasyEnv"
        return result

    def play_episode (self, callback = None) -> (float, int):
        """ Plays a full episode using the previously trained policy, returning tuple (rewards,steps)
            representing the sum of rewards and the totale number of steps over the full episode.

            Args:
            callback    : callback(action,state,reward,done,info) is called after each step.
                          if the callback yields True, the episode is aborted.      
        """
        assert self._trained_policy is not None, "policy not yet trained. call train() first."

        if self._gym_eval_env is None:
            self._gym_eval_env = self._create_tfagent_env()

        easy_env = self._get_EasyEnv( self._gym_eval_env )
        if callback is not None:
            easy_env._set_step_callback( callback )

        sum_rewards = 0.0
        step_count = 0
        time_step = self._gym_eval_env.reset()
        while not time_step.is_last():
            action_step = self._trained_policy.action(time_step)
            time_step = self._gym_eval_env.step(action_step.action)
            sum_rewards += time_step.reward
            step_count += 1
        easy_env._set_step_callback( None )
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
        training_duration                   : instance of config.TrainingDuration to configure the #episodes used for training.
        num_training_steps_in_replay_buffer : defines the size of the replay buffer in terms of game steps.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.
        logging                 : instance of config.Logging to define the logging behaviour

        see also: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(   self,
                    gym_env_name : str,
                    fc_layers = None,
                    training_duration : TrainingDuration = None,
                    num_training_steps_in_replay_buffer : int = 10001,
                    learning_rate : float = 0.001,
                    reward_discount_gamma : float = 1,
                    logging : Logging = None ):
        super().__init__(   gym_env_name= gym_env_name,
                            fc_layers= fc_layers,
                            training_duration=training_duration,
                            learning_rate=learning_rate,
                            reward_discount_gamma=reward_discount_gamma,
                            logging=logging)
        assert num_training_steps_in_replay_buffer >= 1, "num_training_steps_in_replay_buffer must be >= 1"
        self._num_training_steps_in_replay_buffer = num_training_steps_in_replay_buffer
        return

    def train( self):
        """ trains a policy using the gym_env.
            Sets training_losses and training_average_returns, depending on the training scheme
            defined in TrainingDuration configuration.
        """
        # Create Training Environment, Optimizer and PpoAgent
        self._log_agent("Creating environment:")
        train_env = self._create_tfagent_env()
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        self._log_agent("Creating agent:")
        self._log_agent("  creating  tf.compat.v1.train.AdamOptimizer( ... )")
        optimizer = tf.compat.v1.train.AdamOptimizer( learning_rate=self._learning_rate )

        actor_net = actor_distribution_network.ActorDistributionNetwork( observation_spec,
                                                                         action_spec,
                                                                         fc_layer_params=self.fc_layers )
        value_net = value_network.ValueNetwork( observation_spec, fc_layer_params=self.fc_layers )

        self._log_agent("  creating  PpoAgent( ... )")
        tf_agent = ppo_agent.PPOAgent(  timestep_spec, action_spec, optimizer,
                                        actor_net=actor_net,
                                        value_net=value_net,
                                        num_epochs=self._training_duration.num_epochs_per_iteration)
        self._log_agent("  executing tf_agent.initialize()")
        tf_agent.initialize()
        self._trained_policy = tf_agent.policy

        # Data collection
        self._log_agent("Creating data collection:")
        collect_data_spec = tf_agent.collect_data_spec
        self._log_agent("  creating TFUniformReplayBuffer()")
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer( collect_data_spec,
                                                                        batch_size=1,
                                                                        max_length=self._num_training_steps_in_replay_buffer )

        collect_policy = tf_agent.collect_policy
        self._log_agent("  creating DynamicEpisodeDriver()")
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(   train_env,
                                                                        collect_policy,
                                                                        observers=[replay_buffer.add_batch],
                                                                        num_episodes=self._training_duration.num_episodes_per_iteration )

        # Train
        collect_driver.run = common.function( collect_driver.run, autograph=False )
        tf_agent.train = common.function( tf_agent.train, autograph=False )

        self._clear_average_rewards_and_steps_log()
        self._record_average_rewards_and_steps()
        self.training_losses = []

        self._log_agent("Starting training:")
        for step in range( 1, self._training_duration.num_iterations + 1):
            msg = f'training {step:3} of {self._training_duration.num_iterations:3}:'
            self._log_agent(msg + " executing collect_driver.run()")
            collect_driver.run()

            self._log_agent(msg + " executing replay_buffer.gather_all()")
            trajectories = replay_buffer.gather_all()

            self._log_agent(msg + " executing tf_agent.train(...)")
            total_loss, _ = tf_agent.train( experience=trajectories )
            self.training_losses.append( float(total_loss) )
            self._log_minimal(f'{msg} completed tf_agent.train(...) = {total_loss.numpy():.3f} [loss]')

            self._log_agent(msg + " executing replay_buffer.clear()")
            replay_buffer.clear()

            if step % self._training_duration.num_iterations_between_eval == 0:
                self._record_average_rewards_and_steps()
        return


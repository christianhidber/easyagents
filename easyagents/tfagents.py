import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common

from easyagents.agents import EasyAgent
from easyagents.config import TrainingDuration


class TfAgent(EasyAgent):
    """ Reinforcement learning agents based on googles tf_agent implementations
        https://github.com/tensorflow/agents
    """
    def __init__(   self,
                    gym_env_name : str,
                    training_duration : TrainingDuration = None,
                    fc_layers = None,
                    reward_discount_gamma : float = 1,
                    learning_rate : float = 0.001
                     ):
        super().__init__(   gym_env_name=gym_env_name, 
                            training_duration=training_duration, 
                            fc_layers=fc_layers,
                            reward_discount_gamma=reward_discount_gamma,
                            learning_rate=learning_rate )
        self.__initialize()
        return

    def __initialize(self):
        """ initializes TensorFlow behaviour and random seeds.
        """
        self._log.debug("executing: tf.compat.v1.enable_v2_behavior()")
        tf.compat.v1.enable_v2_behavior()
        self._log.debug("executing: tf.enable_eager_execution()")
        tf.compat.v1.enable_eager_execution()
        self._log.debug("executing: tf.compat.v1.set_random_seed(0)")
        tf.compat.v1.set_random_seed(0)
        return

    def compute_avg_return(self, policy ):
        """ computes the expected sum of rewards for 'policy'.

            Note:
            The evaluation is performed on a instance of gym_env_name.
        """
        self._log.debug(f'executing compute_avg_return(...)')
        gym_env = self.load_tfagent_env()
        total_return = 0.0
        for _ in range(self.training_duration.num_eval_episodes):
            time_step = gym_env.reset()
            episode_return = 0.0

            time_step = gym_env.current_time_step()
            action_step = policy.action(time_step)
            time_step = gym_env.step(action_step.action)

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = gym_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        result = total_return / self.training_duration.num_eval_episodes
        self._log.debug(f'completed compute_avg_return(...) = {float(result):.3f}')
        return result.numpy()[0]

    def load_tfagent_env(self):
        """ loads the gym environment and wraps it in a tfagent TFPyEnvironment
        """
        self._log.debug("   executing tf_py_environment.TFPyEnvironment( suite_gym.load )")
        py_env = suite_gym.load( self.gym_env_name, discount=self.reward_discount_gamma,max_episode_steps=100000  )
        result = tf_py_environment.TFPyEnvironment( py_env)
        return result



class Ppo(TfAgent):
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
        training_duration                   : instance of TrainingDuration to configure the #episodes used for training.
        num_training_steps_in_replay_buffer : defines the size of the replay buffer in terms of game steps.
        learning_rate           : value in (0,1]. Factor by which the impact on the policy update is reduced
                                  for each training step. The same learning rate is used for the value and the policy network.
        reward_discount_gamma   : value in (0,1]. Factor by which a future reward is discounted for each step.

        see also: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(   self,
                    gym_env_name : str,
                    fc_layers = None,
                    training_duration : TrainingDuration = None,
                    num_training_steps_in_replay_buffer : int = 10001,
                    learning_rate : float = 0.001,
                    reward_discount_gamma : float = 1 ):
        super().__init__(   gym_env_name= gym_env_name, 
                            fc_layers= fc_layers,
                            training_duration=training_duration,
                            learning_rate=learning_rate,
                            reward_discount_gamma=reward_discount_gamma)
        assert num_training_steps_in_replay_buffer >= 1, "num_training_steps_in_replay_buffer must be >= 1"
        self._num_training_steps_in_replay_buffer = num_training_steps_in_replay_buffer
        return

    def train(  self):
        """ trains a policy using the gym_env

            Args:
                num_training_steps_in_replay_buffer     : size of the replay buffer in number of game steps

            Returns:
                a list with the evaluated returns during training. 
                The list starts with the average return before training started and contains a value for each completed eval interval.
        """
        # Create Training Environment, Optimizer and PpoAgent
        self._log.debug("Creating environment:")
        train_env = self.load_tfagent_env()
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        self._log.debug("Creating agent:")
        self._log.debug("  creating  tf.compat.v1.train.AdamOptimizer( ... )")
        optimizer = tf.compat.v1.train.AdamOptimizer( learning_rate=self.learning_rate )

        actor_net = actor_distribution_network.ActorDistributionNetwork( observation_spec, action_spec, fc_layer_params=self.fc_layers )
        value_net = value_network.ValueNetwork( observation_spec, fc_layer_params=self.fc_layers )

        self._log.debug("  creating  PPOAgent( ... )")
        tf_agent = ppo_agent.PPOAgent(  timestep_spec, action_spec, optimizer, 
                                        actor_net=actor_net, 
                                        value_net=value_net,
                                        num_epochs=self.training_duration.num_epochs_per_iteration)
        self._log.debug("  executing tf_agent.initialize()")
        tf_agent.initialize()

        # Data collection
        self._log.debug("Creating data collection:")
        collect_data_spec = tf_agent.collect_data_spec
        self._log.debug("  creating TFUniformReplayBuffer()")
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(     collect_data_spec,
                                                                            batch_size=1,
                                                                            max_length=self._num_training_steps_in_replay_buffer )

        collect_policy = tf_agent.collect_policy
        self._log.debug("  creating DynamicEpisodeDriver()")
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(   train_env,
                                                                        collect_policy,
                                                                        observers=[replay_buffer.add_batch],
                                                                        num_episodes=self.training_duration.num_episodes_per_iteration )

        # Train
        collect_driver.run = common.function( collect_driver.run, autograph=False )
        tf_agent.train = common.function( tf_agent.train, autograph=False )

        returns=[]

        self._log.debug("Starting training:")
        for step in range( self.training_duration.num_iterations ):
            msg = f'training {step} of {self.training_duration.num_iterations}:'

            if step % self.training_duration.num_iterations_between_eval == 0:
                avg_return = self.compute_avg_return( tf_agent.policy )
                returns.append( avg_return )

            self._log.debug(msg + " executing collect_driver.run()")
            collect_driver.run()
            self._log.debug(msg + " executing replay_buffer.gather_all()")
            trajectories = replay_buffer.gather_all()
            self._log.debug(msg + " executing tf_agent.train(...)")
            total_loss, _ = tf_agent.train(experience=trajectories)
            self._log.debug( f'{msg} completed tf_agent.train(...) = {total_loss.numpy():.3f} [loss]')
            self._log.debug(msg + " executing replay_buffer.clear()")
            replay_buffer.clear()
        return returns

    def step(self):
        """ performs a single step in the gym_env using the trained policy."""
        return

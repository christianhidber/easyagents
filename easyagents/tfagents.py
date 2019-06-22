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


class TfAgent(EasyAgent):
    """ Reinforcement learning agents based on googles tf_agent implementations
        https://github.com/tensorflow/agents

        Abstract base class.
    """

    def __init__(   self,
                    gym_env_name,
                    fc_layers ):
        super().__init__(gym_env_name, fc_layers)
        self.reward_discount_gamma = 1
        self.__initialize()
        return

    def __initialize(self):
        """ initializes TensorFlow behaviour and random seeds.
        """
        self._log.debug("executing: tf.compat.v1.enable_v2_behavior()")
        tf.compat.v1.enable_v2_behavior()
        self._log.debug("executing: tf.compat.v1.set_random_seed(0)")
        tf.compat.v1.set_random_seed(0)
        return

    def compute_avg_return(self, gym_env_name, policy, num_eval_episodes):
        """ computes the expected sum of rewards for 'policy' in 'gym_env'.

            Note:
            o gym_env.reset is called before computing the reward for each episode
        """
        total_return = 0.0
        gym_env = self.load_tfagent_env()
        for _ in range(num_eval_episodes):
            time_step = gym_env.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = gym_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        result = total_return / num_eval_episodes
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

        see also: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(   self,
                    gym_env_name,
                    fc_layers=None ):
        super().__init__(gym_env_name, fc_layers)
        return

    def train(  self,
                num_training_episodes=100,
                num_training_episodes_per_iteration=10,
                num_eval_episodes=10,
                learning_rate=0.99,
                reward_discount_gamma=1 ):
        """ trains a policy using the gym_env

            Args:
                num_training_episodes_total         : total number of episodes played during training
                num_training_episodes_per_iteration : number of episodes included per training iteration 
                                                      (forward pass and backpropagation)
                num_eval_episodes     : number of episodes played to estimate the average return
                learning_rate         : value in (0,1]. Factor by which the impact on the policy update is reduced
                                        for each training step. The same learning rate is used for the value and
                                        the policy network.
                reward_discount_gamma : value in (0,1]. Factor by which a future reward is discounted for each step.
                random_seed           : random seed for all sampling purposes.
        """
        assert num_training_episodes >= 1, "num_training_episodes must be >= 1"
        assert num_training_episodes >= num_training_episodes_per_iteration, "num_training_episodes must be >= num_training_episodes_per_iteration"
        assert num_training_episodes_per_iteration >= 1, "num_training_episodes_per_iteration must be >= 1"
        assert num_eval_episodes >= 1, "num_eval_episodes must be >= 1"
        assert learning_rate > 0, "learning_rate must be in (0,1]"
        assert learning_rate <= 1, "learning_rate must be in (0,1]"
        assert reward_discount_gamma > 0, "reward_discount_gamma must be in (0,1]"
        assert reward_discount_gamma <= 1, "reward_discount_gamma must be in (0,1]"

        self.reward_discount_gamma = reward_discount_gamma
        self._log.debug("   executing tf.compat.v1.train.get_or_create_global_step()")
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Create Training Environment, Optimizer and PpoAgent
        self._log.debug("Creating environment:")
        train_env = self.load_tfagent_env()
        observation_spec = train_env.observation_spec()
        action_spec = train_env.action_spec()
        timestep_spec = train_env.time_step_spec()

        self._log.debug("Creating agent:")
        self._log.debug("  creating  tf.compat.v1.train.AdamOptimizer( ... )")
        optimizer = tf.compat.v1.train.AdamOptimizer( learning_rate=learning_rate )

        actor_net = actor_distribution_network.ActorDistributionNetwork( observation_spec, action_spec, fc_layer_params=self.fc_layers )
        value_net = value_network.ValueNetwork( observation_spec, fc_layer_params=self.fc_layers )

        self._log.debug("  creating  PPOAgent( ... )")
        tf_agent = ppo_agent.PPOAgent(  timestep_spec, action_spec, optimizer, 
                                        actor_net=actor_net, 
                                        value_net=value_net,
                                        num_epochs=25,
                                        # entropy_regularization=0.0,
                                        debug_summaries=False,
                                        summarize_grads_and_vars=False,
                                        train_step_counter=global_step )
        self._log.debug("  executing tf_agent.initialize()")
        tf_agent.initialize()

        # Data collection
        self._log.debug("Creating data collection:")
        collect_data_spec = tf_agent.collect_data_spec
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(     collect_data_spec,
                                                                            batch_size=1,
                                                                            max_length=10001 )

        collect_policy = tf_agent.collect_policy
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(   train_env,
                                                                        collect_policy,
                                                                        observers=[replay_buffer.add_batch],
                                                                        num_episodes=num_training_episodes_per_iteration )

        # Train
        collect_driver.run = common.function( collect_driver.run, autograph=False )
        tf_agent.train = common.function( tf_agent.train, autograph=False )

        # Eval Environment
        self._log.debug("Creating eval environment:")
        eval_env = self.load_tfagent_env()
        avg_return = self.compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)

        returns=[avg_return]

        num_training_iterations = int(num_training_episodes / num_training_episodes_per_iteration)
        eval_interval = 10 * num_eval_episodes
        self._log.debug("Starting training:")
        for step in range( num_training_iterations ):
            msg = f'[training {step}]'
            self._log.debug(msg + " executing collect_driver.run()")
            collect_driver.run()
            self._log.debug(msg + " completed collect_driver.run()")
            trajectories = replay_buffer.gather_all()
            self._log.debug(msg + " executing tf_agent.train")
            total_loss, _ = tf_agent.train(experience=trajectories)
            self._log.debug( f'{msg} completed tf_agent.train loss={total_loss.numpy()}')
            replay_buffer.clear()

            if step % eval_interval == 0:
                self._log.debug(f'{msg} executing compute_avg_return')
                avg_return = self.compute_avg_return( eval_env, tf_agent.policy, num_eval_episodes )
                self._log.debug(f'{msg} completed compute_avg_returnAverage = {avg_return}')
                returns.append( avg_return )
        return

    def step(self):
        """ performs a single step in the gym_env using the trained policy."""
        return

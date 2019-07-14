import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common

def train( self):
    tf.compat.v1.enable_v2_behavior()
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.set_random_seed(0)

    py_env = suite_gym.load( self._gym_env_name,
                             discount=self._reward_discount_gamma,
                             max_episode_steps=self._training_duration.max_steps_per_episode  )
    tfenv = tf_py_environment.TFPyEnvironment( py_env)

    actor_net = actor_distribution_network.ActorDistributionNetwork( tfenv.observation_spec(),
                                                                     tfenv.action_spec(),
                                                                     fc_layer_params=self.fc_layers )
    value_net = value_network.ValueNetwork( tfenv.observation_spec(), fc_layer_params=self.fc_layers )
    optimizer = tf.compat.v1.train.AdamOptimizer( learning_rate=self._learning_rate )
    tf_agent = ppo_agent.PPOAgent(  tfenv.time_step_spec(), tfenv.action_spec(), optimizer,
                                    actor_net=actor_net,
                                    value_net=value_net,
                                    num_epochs=self._training_duration.num_epochs_per_iteration)
    tf_agent.initialize()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer( tf_agent.collect_data_spec,
                                                                    batch_size=1,
                                                                    max_length=self._num_training_steps_in_replay_buffer )
    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(   tfenv,
                                                                    tf_agent.collect_policy,
                                                                    observers=[replay_buffer.add_batch],
                                                                    num_episodes=self._training_duration.num_episodes_per_iteration )

    collect_driver.run = common.function( collect_driver.run, autograph=False )
    tf_agent.train = common.function( tf_agent.train, autograph=False )

    for step in range( 1, self._training_duration.num_iterations + 1):
        collect_driver.run()
        trajectories = replay_buffer.gather_all()
        total_loss, _ = tf_agent.train( experience=trajectories )
        replay_buffer.clear()

def play_episode (self, callback = None) -> (float, int):
    time_step = self._gym_eval_env.reset()
    while not time_step.is_last():
        action_step = self._trained_policy.action(time_step)
        time_step = self._gym_eval_env.step(action_step.action)


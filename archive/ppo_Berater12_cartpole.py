# Environment
#%%

env_name = 'CartPole-v0'

import numpy as np
import random

import gym
from gym.utils import seeding
from gym import spaces

import pdb

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment

tf.compat.v1.enable_v2_behavior()

#@test {"skip": true}
def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

"""### Setup and train"""

actor_fc_layers=(500,500,500) # @param
value_fc_layers=(500,500,500) # @param


# Params for collect
collect_episodes_per_iteration=100 # @param
num_parallel_environments=1
replay_buffer_capacity=10001  # @param
# Params for train
num_train_iterations=2000 # @param
num_epochs=5 # @param
learning_rate=1e-4 # @param
# Params for summaries and logging
log_interval=1 # @param
use_tf_functions=True
debug_summaries=False
summarize_grads_and_vars=False

num_eval_episodes = 10  # @param
eval_interval = 10  # @param

global_step = tf.compat.v1.train.get_or_create_global_step()
tf.compat.v1.set_random_seed(0)
eval_py_env = suite_gym.load(env_name)
tf_env = tf_py_environment.TFPyEnvironment( suite_gym.load(env_name))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=actor_fc_layers)
value_net = value_network.ValueNetwork(
    tf_env.observation_spec(), fc_layer_params=value_fc_layers)

tf_agent = ppo_agent.PPOAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=num_epochs,
    debug_summaries=debug_summaries,
    summarize_grads_and_vars=summarize_grads_and_vars,
    train_step_counter=global_step)
tf_agent.initialize()

eval_py_env = suite_gym.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=num_parallel_environments,
    max_length=replay_buffer_capacity)

collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    tf_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_episodes=collect_episodes_per_iteration)

### PPO TF-Agent: train
#%%

collect_driver.run = common.function(collect_driver.run, autograph=False)
tf_agent.train = common.function(tf_agent.train, autograph=False)

avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns=[avg_return]
loss=[]

for step in range(num_train_iterations):
  collect_driver.run()
  trajectories = replay_buffer.gather_all()
  total_loss, _ = tf_agent.train(experience=trajectories)
  replay_buffer.clear()

  if step % log_interval == 0:
    print('iteration/train_step = {}, loss = {}'.format(step, total_loss.numpy()))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('iteration/train_step = {}, Average Return = {}'.format(step, avg_return))
    returns.append(avg_return)

### PPO TF-Agent: visualize
#%%

#@test {"skip": true}
import matplotlib
import matplotlib.pyplot as plt


steps = range(0, len(returns)*eval_interval, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=1,bottom=-5)
# https://github.com/tensorflow/agents
# https://colab.research.google.com/github/tensorflow/agents/blob/master/tf_agents/colabs/1_dqn_tutorial.ipynb
#%%

### Setup
#%%
#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


tf.compat.v1.enable_v2_behavior()

### Hyperparameters
#%%

env_name = 'CartPole-v0'  # @param
#num_iterations = 20000  # @param
num_iterations = 2000  # @param

initial_collect_steps = 1000  # @param
collect_steps_per_iteration = 1  # @param
replay_buffer_capacity = 1001  # @param

fc_layer_params = (100,)

batch_size = 64  # @param
learning_rate = 1e-3  # @param
log_interval = 200  # @param

num_eval_episodes = 10  # @param
eval_interval = 1000  # @param

### Environment
#%%

env = suite_gym.load(env_name)

env.reset()
##PIL.Image.fromarray(env.render())
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)
action = 1
next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

train_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_py_env = suite_gym.load(env_name)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

### Agents
#%%

actor_net = actor_distribution_network.ActorDistributionNetwork(    train_env.observation_spec(),
                                                                    train_env.action_spec(),
                                                                    fc_layer_params=fc_layer_params )
value_net = value_network.ValueNetwork( train_env.observation_spec(), fc_layer_params=fc_layer_params )
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
tf_agent = ppo_agent.PPOAgent(  train_env.time_step_spec(),
                                train_env.action_spec(),
                                optimizer,
                                actor_net=actor_net,
                                value_net=value_net)
tf_agent.initialize()

### Policies
#%%

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

### Metrics and Evaluation (compute_avg_return)
#%%

def compute_avg_return(environment, policy, num_episodes=10):
  """
  Liefert den durchschnittlich erreichten reward fuer policy.

  Hinweis:
  o policy wird num_epsiodes mal in environment ausgefuehrt
  """

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

compute_avg_return(eval_env, random_policy, num_eval_episodes)

### Replay Buffer (replay_buffer)
#%%

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer( data_spec=tf_agent.collect_data_spec,
                                                                batch_size=1,
                                                                max_length=1001)







### Data Collection (collect_step, initialisiert replay_buffer)
#%%

def collect_step(environment, policy):
  """
  Fuehrt auf auf dem aktuellen gamestate in environment 1 action aus (bestimmt durch policy)
  und speichert das ergebnis als trajectory in 'replay_buffer'
  """
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.
for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset( num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2 ).prefetch(3)
iterator = iter(dataset)

### Training
#%%
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
#tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, tf_agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = tf_agent.train(experience)

  step = tf_agent.train_step_counter.numpy()
  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))
  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

### Visualization
#%%
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=250)

### Create Video (imageio.mp4)
#%%
def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

num_episodes = 3
video_filename = 'imageio.mp4'
with imageio.get_writer(video_filename, fps=60) as video:
  for _ in range(num_episodes):
    time_step = eval_env.reset()
    video.append_data(eval_py_env.render())
    while not time_step.is_last():
      action_step = tf_agent.policy.action(time_step)
      time_step = eval_env.step(action_step.action)
      video.append_data(eval_py_env.render())

embed_mp4(video_filename)

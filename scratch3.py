import os

import tensorflow as tf

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(v=tf.logging.ERROR)

environment = Environment.create(environment='gym', level='CartPole-v1')
network_spec = [
  dict(type='dense', size=32, activation='relu'),
  dict(type='dense', size=32, activation='relu')
]


def callback(runner: Runner) -> bool:
  if runner.episode_updated:
    pass
  return True

# Create a PPO agent
agent = Agent.create(
    agent='ppo',
    environment=environment,
    # Automatically configured network
    network=network_spec,
    # Optimization
    learning_rate=3e-4,
    optimization_steps=10,
    # Reward estimation
    discount=0.99,
    # Critic
    critic_network=None,
    critic_optimizer=None,
    # TensorFlow etc
    seed=None,
)

# Initialize the runner
runner = Runner(agent=agent, environment=environment)

# Start the runner
runner.run(num_episodes=300,evaluation_callback=callback,evaluation_frequency=None,evaluation=False,num_evaluation_iterations=0)
runner.close()

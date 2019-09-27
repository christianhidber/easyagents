from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


def callback(runner: Runner) -> bool:
  if runner.episode_updated:
    pass
  return True

# Create an OpenAIgym environment.
environment = Environment.create(environment='gym', level='CartPole-v1')
network_spec = [
  dict(type='dense', size=32, activation='relu'),
  dict(type='dense', size=32, activation='relu')
]
agent = PPOAgent(
  states=environment.states,
  actions=environment.actions,
  network=network_spec,
  max_episode_timesteps=100,
  parallel_interactions=1,
  # Optimization
  batch_size=10, learning_rate=1e-3, subsampling_fraction=0.2,
  optimization_steps=5,
   # Critic
  critic_network='auto',
  critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
  # Exploration
  exploration=0.0, variable_noise=0.0,
  # Regularization
  l2_regularization=0.0
)
# Create the runner
runner = Runner(agent=agent, environment=environment)
# Start learning
runner.run(episodes=600, max_episode_timesteps=200,callback=callback)
runner.close()
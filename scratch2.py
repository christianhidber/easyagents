from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner

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
  saver=dict(directory='./saver/', basename='PPO_model.ckpt', load=False, seconds=600),
  summarizer=dict(directory='./record/', labels=["losses", "entropy"], seconds=600),
  max_episode_timesteps=100
)
# Create the runner
runner = Runner(agent=agent, environment=environment)
# Start learning
runner.run(episodes=600, max_episode_timesteps=200)
runner.close()
#
# simple easyagents demo using cartpole

from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingDurationFast
from easyagents.config import LoggingVerbose

ppo_agent = PpoAgent( gym_env_name='CartPole-v0', training_duration=TrainingDurationFast(), logging=LoggingVerbose())
ppo_agent.train()

ppo_agent.plot_average_returns()
ppo_agent.plot_losses()




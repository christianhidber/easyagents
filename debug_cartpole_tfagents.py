#
# simple easyagents demo using cartpole

from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingDurationSingleEpisode
from easyagents.config import LoggingVerbose


ppo_agent = PpoAgent( gym_env_name='CartPole-v0', training_duration=TrainingDurationSingleEpisode(), logging=LoggingVerbose())
ppo_agent.train()

ppo_agent.plot_average_returns()
ppo_agent.plot_losses()
y = ppo_agent.render_episodes_to_mp4()
x = ppo_agent.render_episodes_to_html(fps=20)





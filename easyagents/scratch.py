import easyagents

from easyagents.agents import DqnAgent, PpoAgent, RandomAgent
from easyagents.callbacks import duration, plot, log

rndAgent = RandomAgent('CartPole-v0')
# rndAgent.train([plot.StepRewards()])
rndAgent.play([plot.StepRewards()],num_episodes=3)


input('press enter')



import easyagents

from easyagents.agents import DqnAgent, PpoAgent, RandomAgent
from easyagents.callbacks import duration, plot, log

a = PpoAgent('CartPole-v0')
a.train(duration._SingleIteration())
# rndAgent.train([plot.StepRewards()])
a.play()


input('press enter')



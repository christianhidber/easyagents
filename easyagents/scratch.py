import easyagents

from easyagents.agents import DqnAgent, PpoAgent, RandomAgent
from easyagents.callbacks import duration, plot, log

a = RandomAgent('CartPole-v0')
#a.train(duration._SingleIteration())
a.play([plot.Actions()])


input('press enter')



from easyagents.agents import PpoAgent
from easyagents.callbacks import duration, plot

ppoAgent = PpoAgent('CartPole-v0')
ppoAgent.train([duration.Fast(),plot.Actions(),plot.State()])
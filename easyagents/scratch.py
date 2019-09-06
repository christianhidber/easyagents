import easyagents
from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import PlotLoss, PlotRewards, PlotSteps, PlotState

agent = easyagents.PpoAgent("CartPole-v0")
agent.train([Fast(), PlotRewards()])

input('press enter')
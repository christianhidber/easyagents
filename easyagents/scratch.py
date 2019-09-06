import easyagents
from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import PlotLoss, PlotRewards, PlotSteps, PlotState, ToMovie

agent = easyagents.PpoAgent("CartPole-v0")
agent.train([Fast(), PlotRewards(), ToMovie()])

input('press enter')
import easyagents

from easyagents.callbacks.duration import Fast
from easyagents.callbacks.plot import Loss, Rewards, Steps, State, ToMovie

agent = easyagents.PpoAgent("CartPole-v0")
agent.train([Fast(), Rewards(), Loss(), State()])

input('press enter')
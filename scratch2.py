import easyagents
from easyagents.agents import DqnAgent
from easyagents.callbacks import plot, duration, log

print(easyagents.agents.get_backends())
# dqn_agent = DqnAgent('CartPole-v0')
dqn_agent = DqnAgent('CartPole-v0', backend='huskarl')
dqn_agent.train([plot.Actions(),plot.Rewards(),plot.State()])

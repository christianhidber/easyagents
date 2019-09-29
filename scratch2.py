import easyagents
from easyagents.agents import DqnAgent, PpoAgent
from easyagents.callbacks import plot, duration, log

#print(easyagents.agents.get_backends())
# dqn_agent = DqnAgent('CartPole-v0')
#dqn_agent = DqnAgent('CartPole-v0', backend='huskarl')
#dqn_agent.train([plot.Actions(),plot.Rewards(),plot.State(), plot.Loss()])

ppoAgent = PpoAgent('CartPole-v0')
ppoAgent.train([duration.Fast()])
pc = ppoAgent.play(num_episodes=10)



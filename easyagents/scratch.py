import easyagents

from easyagents.agents import DqnAgent, PpoAgent, RandomAgent
from easyagents.callbacks import duration, plot, log


from easyagents.agents import DqnAgent
from easyagents.callbacks import plot

#dqnAgent = DqnAgent('CartPole-v0', fc_layers=(100, ))
#dqnAgent.train([plot.Actions(),plot.StepRewards(),plot.Rewards()],  num_iterations=1000, num_iterations_between_eval=100)

#a = RandomAgent('CartPole-v0')
#a.train(duration._SingleIteration())
#a.train([plot.State()],num_iterations=1,num_episodes_per_eval=1)

rndAgent = RandomAgent('CartPole-v0')

rndAgent.play([plot.Actions(), plot.StepRewards()], num_episodes=10)

input('press enter')



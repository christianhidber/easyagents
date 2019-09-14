import easyagents

from easyagents.agents import DqnAgent, PpoAgent, RandomAgent, ReinforceAgent
from easyagents.callbacks import duration, plot, log


from easyagents.agents import DqnAgent
from easyagents.callbacks import plot

reinforce_agent = ReinforceAgent('CartPole-v0')
reinforce_agent.train(duration.Fast())
reinforce_agent.play([plot.Actions(),plot.Rewards()],num_episodes=10)

#dqnAgent = DqnAgent('CartPole-v0', fc_layers=(100, ))
#dqnAgent.train([plot.Actions(),plot.StepRewards(),plot.Rewards()],  num_iterations=1000, num_iterations_between_eval=100)

#a = RandomAgent('CartPole-v0')
#a.train(duration._SingleIteration())
#a.train([plot.State()],num_iterations=1,num_episodes_per_eval=1)

input('press enter')



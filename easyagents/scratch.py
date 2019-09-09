import easyagents

from easyagents.agents import DqnAgent, PpoAgent
from easyagents.callbacks import duration, plot, log

dqnAgent = DqnAgent('CartPole-v0', fc_layers=(100, ))
dqnAgent.train([log.Iteration(num_iterations_between_log=200)],
               num_iterations=20000,
               num_iterations_between_eval=1000,
               default_callbacks=False)

input('press enter')
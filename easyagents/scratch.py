from easyagents.agents import PpoAgent
from easyagents.callbacks import duration, log

ppoAgent = PpoAgent('CartPole-v0', fc_layers=(100,50))
ppoAgent.train([duration.Fast()])
directory = ppoAgent.save()
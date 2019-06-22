#
# simple easyagents demo using cartpole

from easyagents.tfagents import Ppo
import easyagents.logenv
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("starting")
logname = easyagents.logenv.register('CartPole-v0')
agent = Ppo( logname )
agent.train()
logging.info("completed")

input("press enter to terminate...")

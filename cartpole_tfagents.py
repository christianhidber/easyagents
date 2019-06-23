#
# simple easyagents demo using cartpole

from easyagents.tfagents import Ppo
from easyagents.config import TrainingDurationFast
import easyagents.logenv
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("starting")
logname = easyagents.logenv.register('CartPole-v0')
ppoAgent = Ppo( gym_env_name='CartPole-v0',  training_duration=TrainingDurationFast() )
returns = ppoAgent.train()
logging.info("completed")

input("press enter to terminate...")

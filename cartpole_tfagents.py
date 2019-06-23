#
# simple easyagents demo using cartpole

from easyagents.tfagents import Ppo
from easyagents.config import TrainingDurationFast
from easyagents.config import LoggingVerbose
import easyagents.logenv
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info("starting")
ppoAgent = Ppo( gym_env_name='CartPole-v0', training_duration=TrainingDurationFast(), logging=LoggingVerbose())
returns = ppoAgent.train()
logging.info("completed")



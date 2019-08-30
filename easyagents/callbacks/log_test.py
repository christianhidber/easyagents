import unittest

import easyagents
import easyagents.env
from easyagents.callbacks.duration import SingleEpisode, Fast
from easyagents.callbacks.log import *

env_name = easyagents.env._StepCountEnv.register_with_gym()

class LogCallbacksTest(unittest.TestCase):

    def test_log_callbacks(self):
        agent = easyagents.PpoAgent(env_name)
        agent.train([LogCallbacks(), SingleEpisode()])

    def test_log_iteration(self):
        agent = easyagents.PpoAgent(env_name)
        agent.train([LogIteration(), Fast()])

    def test_cartpole_log_iteration(self):
        ppo = easyagents.agents.PpoAgent(gym_env_name="CartPole-v0", backend_name='tfagents')
        ppo.train([LogIteration(),Fast()])


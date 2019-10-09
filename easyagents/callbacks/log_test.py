import unittest

from easyagents import env, core, agents
from easyagents.callbacks import duration, log

env_name = env._StepCountEnv.register_with_gym()

class LogCallbacksTest(unittest.TestCase):

    def test_log_callbacks(self):
        agent = agents.PpoAgent(env_name)
        agent.train([log._Callbacks(), duration._SingleIteration()])

    def test_log_agent(self):
        agent = agents.PpoAgent(env_name)
        agent.train([log.Agent(), duration.Fast()])

    def test_log_iteration(self):
        agent = agents.PpoAgent(env_name)
        agent.train([log.Iteration(), duration._SingleIteration()])

    def test_log_step(self):
        agent = agents.PpoAgent(env_name)
        agent.train([log.Step(), duration._SingleIteration()])

    def test_cartpole_log_iteration(self):
        ppo = agents.PpoAgent(gym_env_name="CartPole-v0", backend='tfagents')
        ppo.train([log.Iteration(), duration._SingleIteration()])


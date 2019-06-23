import unittest
import gym
import logging
import easyagents.logenv
import easyagents.tfagents
from easyagents.config import TrainingDurationFast
from easyagents.config import LoggingVerbose

logging.basicConfig(level=logging.DEBUG)

class TestLogEnv(unittest.TestCase):

    def test_register(self):
        logging.basicConfig(level=logging.DEBUG)
        name = easyagents.logenv.register('CartPole-v0')

        assert name == "LogCartPole-v0"

        env = gym.make(name)
        assert env is not None
        return

    def test_register_twiceSameName_success(self):
        easyagents.logenv.register('CartPole-v0')
        easyagents.logenv.register('CartPole-v0')
        return

    def test_register_twiceDifferentNames_fail(self):
        easyagents.logenv.register('CartPole-v0')
        self.assertRaises(Exception, easyagents.logenv.register, 'Dummy-v0')
        return

    def test_LoggingVerbose(self):
        ppo_agent = easyagents.tfagents.PpoAgent( 'CartPole-v0', training_duration=TrainingDurationFast(), logging=LoggingVerbose() )
        ppo_agent.train()
        return    

if __name__ == '__main__':
    unittest.main()
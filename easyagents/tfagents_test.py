import unittest
import tensorflow as tf
from easyagents.tfagents import PpoAgent
from easyagents.config import TrainingDurationFast

class TestTfAgents(unittest.TestCase):

    def setUp(self):
        self.gym_env_name='CartPole-v0'

    def test_ppo_create(self):
        ppo_agent = PpoAgent( self.gym_env_name )
        self.assertIsNotNone( ppo_agent, "failed to create a tfagents.Ppo instance for " + self.gym_env_name )
        return

    def test_ppo_train(self):
        ppo_agent = PpoAgent( self.gym_env_name, training_duration=TrainingDurationFast() )
        ppo_agent.train()
        return

    def test_ppo_str(self):
        ppo_agent = PpoAgent( self.gym_env_name, training_duration=TrainingDurationFast() )
        result = str(ppo_agent)
        print(result)
        return            

if __name__ == '__main__':
    unittest.main()

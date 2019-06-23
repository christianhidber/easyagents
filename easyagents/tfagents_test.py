import unittest
import tensorflow as tf
from easyagents.tfagents import Ppo
from easyagents.config import TrainingDurationFast

class TestTfAgents(unittest.TestCase):

    def setUp(self):
        self.gym_env_name='CartPole-v0'

    def test_ppo_create(self):
        ppoAgent = Ppo( self.gym_env_name )
        self.assertIsNotNone( ppoAgent, "failed to create a tfagents.Ppo instance for " + self.gym_env_name )
        return

    def test_ppo_train(self):
        ppoAgent = Ppo( self.gym_env_name, training_duration=TrainingDurationFast() )
        ppoAgent.train()
        return

    def test_ppo_str(self):
        ppoAgent = Ppo( self.gym_env_name, training_duration=TrainingDurationFast() )
        result = str(ppoAgent)
        print(result)
        return            

if __name__ == '__main__':
    unittest.main()

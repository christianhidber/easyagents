import unittest
import tensorflow as tf
from easyagents.tfagents import Ppo

class TestTfAgents(unittest.TestCase):

    def setUp(self):
        self.gym_env_name='CartPole-v0'

    def test_ppo_create(self):
        ppoAgent = Ppo( self.gym_env_name )
        self.assertIsNotNone( ppoAgent, "failed to create a tfagents.Ppo instance for " + self.gym_env_name )
        return

    def test_ppo_train(self):
        ppoAgent = Ppo( self.gym_env_name )
        ppoAgent.train( num_training_episodes=5,
                        num_training_episodes_per_iteration=2,
                        num_eval_episodes=2)
        return

    def test_ppo_str(self):
        ppoAgent = Ppo( self.gym_env_name )
        result = str(ppoAgent)
        print(result)
        return            

if __name__ == '__main__':
    unittest.main()

import unittest
import gym
from easyagents.logenv import register
from easyagents.tfagents import Ppo

class TestLogEnv(unittest.TestCase):

    def setUp(self):
        self.gym_env_name='CartPole-v0'

    def test_register(self):
        name = register('CartPole-v0')

        assert name == "LogCartPole-v0"

        env = gym.make(name)
        assert env is not None
        return

    def test_register_twice_success(self):
        register('CartPole-v0')
        register('CartPole-v0')
        return

    def test_ppo_train(self):
        logenvname = register('CartPole-v0')
        ppoAgent = Ppo( logenvname )
        ppoAgent.train( num_training_episodes=2,
                        num_training_episodes_per_iteration=1,
                        num_eval_episodes=1)
        return    

if __name__ == '__main__':
    unittest.main()
import unittest
import gym
import logging
import easyagents.logenv
import easyagents.tfagents

logging.basicConfig(level=logging.DEBUG)

class TestLogEnv(unittest.TestCase):

    def test_register(self):
        logging.basicConfig(level=logging.DEBUG)
        name = easyagents.logenv.register('CartPole-v0')

        assert name == "LogCartPole-v0"

        env = gym.make(name)
        assert env is not None
        return

    def test_register_twice_success(self):
        easyagents.logenv.register('CartPole-v0')
        easyagents.logenv.register('CartPole-v0')
        return

    def test_ppo_train(self):
        logenvname = easyagents.logenv.register('CartPole-v0')
        ppoAgent = easyagents.tfagents.Ppo( logenvname )
        ppoAgent.train( num_training_episodes=2,
                        num_training_episodes_per_iteration=1,
                        num_eval_episodes=1)
        return    

if __name__ == '__main__':
    unittest.main()